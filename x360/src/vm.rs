//! Guest virtual-memory manager behind NtAllocateVirtualMemory / NtFreeVirtualMemory and the
//! physical (GPU-visible) heap. Both heaps hand out ranges from a first-fit free list that
//! coalesces neighbours, so a long-running title that allocates and frees constantly never
//! exhausts address space.

use crate::memory::GuestMemory;
use std::collections::BTreeMap;

pub const MEM_COMMIT: u32 = 0x1000;
pub const MEM_RESERVE: u32 = 0x2000;
pub const MEM_DECOMMIT: u32 = 0x4000;
pub const MEM_RELEASE: u32 = 0x8000;

pub const STATUS_SUCCESS: u32 = 0;
pub const STATUS_INVALID_PARAMETER: u32 = 0xC000_000D;
pub const STATUS_NO_MEMORY: u32 = 0xC000_0017;
pub const STATUS_MEMORY_NOT_ALLOCATED: u32 = 0xC000_00A0;

const HEAP_START: u32 = 0x4000_0000;
const HEAP_END: u32 = 0x6FFF_0000;
const PAGE: u32 = 4096;
const ALIGN: u32 = 0x1_0000;

/// Physical-memory views (GPU-visible pool). Real hardware aliases the same RAM at
/// 0xA0000000 (64KB pages), 0xC0000000 (16MB) and 0xE0000000 (4KB); aliasing is TODO.
const PHYS_START: u32 = 0xA000_0000;
const PHYS_END: u32 = 0xC000_0000;

fn round_up(v: u32, a: u32) -> u32 {
    v.wrapping_add(a - 1) & !(a - 1)
}

/// A first-fit range allocator over [start, end) with a coalescing free list.
#[derive(Default)]
struct RangeAlloc {
    /// Free ranges (start -> length).
    free: BTreeMap<u32, u32>,
    /// Live allocations (base -> length), so frees of foreign ranges are ignored.
    live: BTreeMap<u32, u32>,
    used: u32,
}

impl RangeAlloc {
    fn new(start: u32, end: u32) -> Self {
        Self { free: BTreeMap::from([(start, end - start)]), live: BTreeMap::new(), used: 0 }
    }

    fn alloc(&mut self, size: u32, align: u32) -> Option<u32> {
        let (start, len, base) = self.free.iter().find_map(|(&start, &len)| {
            let base = round_up(start, align);
            (base >= start && base.checked_add(size)? <= start + len).then_some((start, len, base))
        })?;
        self.carve(start, len, base, size);
        Some(base)
    }

    /// Take exactly [base, base+size) if all of it is free.
    fn alloc_fixed(&mut self, base: u32, size: u32) -> bool {
        let Some((&start, &len)) = self.free.range(..=base).next_back() else { return false };
        if base.checked_add(size).is_none_or(|end| end > start + len) {
            return false;
        }
        self.carve(start, len, base, size);
        true
    }

    fn carve(&mut self, start: u32, len: u32, base: u32, size: u32) {
        self.free.remove(&start);
        if base > start {
            self.free.insert(start, base - start);
        }
        let end = base + size;
        if end < start + len {
            self.free.insert(end, start + len - end);
        }
        self.live.insert(base, size);
        self.used += size;
    }

    /// Return the allocation based at `base`; false if it isn't one of ours.
    fn release(&mut self, base: u32) -> bool {
        let Some(size) = self.live.remove(&base) else { return false };
        self.used -= size;
        let (mut start, mut len) = (base, size);
        if let Some((&ps, &pl)) = self.free.range(..start).next_back() {
            if ps + pl == start {
                self.free.remove(&ps);
                start = ps;
                len += pl;
            }
        }
        if let Some(&nl) = self.free.get(&(start + len)) {
            self.free.remove(&(start + len));
            len += nl;
        }
        self.free.insert(start, len);
        true
    }

    fn largest_free(&self) -> u32 {
        self.free.values().copied().max().unwrap_or(0)
    }
}

pub struct Vm {
    phys: RangeAlloc,
    virt: RangeAlloc,
    /// Every reserved virtual region and every physical allocation: base -> size (page rounded).
    regions: BTreeMap<u32, u32>,
}

impl Vm {
    pub fn new() -> Self {
        Self { phys: RangeAlloc::new(PHYS_START, PHYS_END), virt: RangeAlloc::new(HEAP_START, HEAP_END), regions: BTreeMap::new() }
    }

    pub fn phys_usage(&self) -> String {
        format!(
            "physical heap used {:#x} of {:#x} (largest free block {:#x}, {} allocations)",
            self.phys.used,
            PHYS_END - PHYS_START,
            self.phys.largest_free(),
            self.phys.live.len()
        )
    }

    pub fn virtual_usage(&self) -> String {
        format!(
            "virtual heap used {:#x} of {:#x} (largest free block {:#x}, {} regions)",
            self.virt.used,
            HEAP_END - HEAP_START,
            self.virt.largest_free(),
            self.virt.live.len()
        )
    }

    /// Allocate committed "physical" memory. Contents are whatever the previous owner left.
    pub fn allocate_physical(&mut self, mem: &mut GuestMemory, size: u32, alignment: u32) -> Result<u32, u32> {
        if size == 0 {
            return Err(STATUS_INVALID_PARAMETER);
        }
        let size = round_up(size, PAGE);
        let base = self.phys.alloc(size, alignment.max(PAGE).next_power_of_two()).ok_or(STATUS_NO_MEMORY)?;
        mem.map(base, size);
        self.regions.insert(base, size);
        Ok(base)
    }

    /// Return a physical allocation to the free list. The pages stay mapped (host memory is bounded
    /// by the heap size, and the GPU may still be reading a just-freed buffer).
    pub fn free_physical(&mut self, base: u32) -> bool {
        if !self.phys.release(base) {
            return false;
        }
        self.regions.remove(&base);
        true
    }

    /// Reserve and/or commit virtual memory. Returns (base, size) actually allocated, or an NTSTATUS.
    pub fn allocate(&mut self, mem: &mut GuestMemory, requested_base: u32, size: u32, kind: u32) -> Result<(u32, u32), u32> {
        if size == 0 || kind & (MEM_COMMIT | MEM_RESERVE) == 0 {
            return Err(STATUS_INVALID_PARAMETER);
        }
        let size = round_up(size, PAGE);
        let base = if requested_base != 0 {
            let base = requested_base & !(PAGE - 1);
            if let Some((rb, rs)) = self.region_of(base) {
                // Committing inside an existing reservation: no new address space is needed.
                let end = (base + size).min(rb + rs);
                if kind & MEM_COMMIT != 0 {
                    mem.map(base, end - base);
                }
                return Ok((base, end - base));
            }
            // A fixed new reservation: take it from the heap if it lies there, otherwise just record it.
            let _ = self.virt.alloc_fixed(base, round_up(size, ALIGN));
            base
        } else {
            self.virt.alloc(round_up(size, ALIGN), ALIGN).ok_or(STATUS_NO_MEMORY)?
        };
        if kind & MEM_RESERVE != 0 {
            self.regions.insert(base, size);
        }
        if kind & MEM_COMMIT != 0 {
            mem.map(base, size);
        }
        Ok((base, size))
    }

    /// NtFreeVirtualMemory: MEM_RELEASE frees a whole reservation, MEM_DECOMMIT unmaps pages but
    /// keeps the reservation. Returns (base, size) affected.
    pub fn free(&mut self, mem: &mut GuestMemory, base: u32, size: u32, kind: u32) -> Result<(u32, u32), u32> {
        match kind & (MEM_RELEASE | MEM_DECOMMIT) {
            MEM_RELEASE => {
                let Some(region_size) = self.regions.get(&base).copied() else { return Err(STATUS_MEMORY_NOT_ALLOCATED) };
                self.regions.remove(&base);
                mem.unmap(base, region_size);
                self.virt.release(base);
                Ok((base, region_size))
            }
            MEM_DECOMMIT => {
                let Some((rb, rs)) = self.region_of(base) else { return Err(STATUS_MEMORY_NOT_ALLOCATED) };
                let end = if size == 0 { rb + rs } else { (base + size).min(rb + rs) };
                mem.unmap(base, end - base);
                Ok((base, end - base))
            }
            _ => Err(STATUS_INVALID_PARAMETER),
        }
    }

    pub fn region_of(&self, addr: u32) -> Option<(u32, u32)> {
        let (&b, &s) = self.regions.range(..=addr).next_back()?;
        (addr < b + s).then_some((b, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xenon_cpu::memory::Bus;

    #[test]
    fn physical_free_list_reuses_and_coalesces() {
        let mut mem = GuestMemory::new();
        let mut vm = Vm::new();
        let a = vm.allocate_physical(&mut mem, 0x1000, 0).unwrap();
        let b = vm.allocate_physical(&mut mem, 0x2000, 0).unwrap();
        let c = vm.allocate_physical(&mut mem, 0x1000, 0).unwrap();
        assert_eq!((b, c), (a + 0x1000, a + 0x3000));
        assert!(vm.free_physical(b));
        assert!(!vm.free_physical(b)); // double free is rejected
        // a hole of exactly 0x2000 exists between a and c: a 0x2000 request reuses it
        assert_eq!(vm.allocate_physical(&mut mem, 0x2000, 0).unwrap(), b);
        // freeing everything coalesces back into one range spanning the whole heap
        for x in [a, b, c] {
            vm.free_physical(x);
        }
        assert_eq!(vm.phys.free.len(), 1);
        assert_eq!(vm.phys.free.values().next().copied(), Some(PHYS_END - PHYS_START));
        assert_eq!(vm.phys.used, 0);
        // alignment is honoured
        let d = vm.allocate_physical(&mut mem, 0x1000, 0x10000).unwrap();
        assert_eq!(d % 0x10000, 0);
    }

    #[test]
    fn reserve_then_commit() {
        let mut mem = GuestMemory::new();
        let mut vm = Vm::new();
        let (base, size) = vm.allocate(&mut mem, 0, 0x2001, MEM_RESERVE).unwrap();
        assert_eq!(base % ALIGN, 0);
        assert_eq!(size, 0x3000);
        assert!(!mem.is_mapped(base)); // reserved only
        vm.allocate(&mut mem, base, 0x1000, MEM_COMMIT).unwrap();
        mem.write32(base, 7);
        assert_eq!(mem.read32(base), 7);
        assert_eq!(vm.region_of(base + 0x2000), Some((base, 0x3000)));
        let (b2, _) = vm.allocate(&mut mem, 0, 1, MEM_RESERVE | MEM_COMMIT).unwrap();
        assert!(b2 >= base + 0x10000);
    }

    #[test]
    fn release_unmaps_and_the_address_space_is_reused() {
        let mut mem = GuestMemory::new();
        let mut vm = Vm::new();
        let (a, _) = vm.allocate(&mut mem, 0, 0x4000, MEM_RESERVE | MEM_COMMIT).unwrap();
        mem.write32(a, 0xCAFE);
        let (b, _) = vm.allocate(&mut mem, 0, 0x4000, MEM_RESERVE | MEM_COMMIT).unwrap();
        assert_ne!(a, b);
        assert_eq!(vm.free(&mut mem, a, 0, MEM_RELEASE), Ok((a, 0x4000)));
        assert!(!mem.is_mapped(a), "released pages are unmapped");
        assert!(vm.free(&mut mem, a, 0, MEM_RELEASE).is_err(), "double free is rejected");
        // the freed 64KB slot is handed out again before new address space is consumed
        let (c, _) = vm.allocate(&mut mem, 0, 0x1000, MEM_RESERVE | MEM_COMMIT).unwrap();
        assert_eq!(c, a);
        assert_eq!(mem.read32(c), 0, "reused memory starts zeroed");
    }

    #[test]
    fn decommit_keeps_the_reservation() {
        let mut mem = GuestMemory::new();
        let mut vm = Vm::new();
        let (a, _) = vm.allocate(&mut mem, 0, 0x4000, MEM_RESERVE | MEM_COMMIT).unwrap();
        assert_eq!(vm.free(&mut mem, a + 0x1000, 0x2000, MEM_DECOMMIT), Ok((a + 0x1000, 0x2000)));
        assert!(mem.is_mapped(a) && !mem.is_mapped(a + 0x1000) && !mem.is_mapped(a + 0x2000) && mem.is_mapped(a + 0x3000));
        assert_eq!(vm.region_of(a + 0x1000), Some((a, 0x4000))); // still reserved
        // recommit the hole
        vm.allocate(&mut mem, a + 0x1000, 0x2000, MEM_COMMIT).unwrap();
        assert!(mem.is_mapped(a + 0x1000));
    }
}
