//! Machine memory: sparse 4KB pages over a 32-bit guest address space.
//! Unmapped accesses return 0 and record the first fault so the run loop can
//! report "crashed reading/writing 0xXXXXXXXX".

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use xenon_cpu::memory::Bus;

const PAGE: usize = 4096;
const PAGES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    pub addr: u32,
    pub write: bool,
}

pub struct GuestMemory {
    pages: Vec<Option<Box<[u8; PAGE]>>>,
    /// First unmapped access since last taken: 0 = none, else bit 63 | write << 32 | address. Atomic so the
    /// renderer's worker threads can share `&GuestMemory`.
    fault: AtomicU64,
    kernel_next: u32,
    /// Freed kernel blocks by size, reused (and re-zeroed) by `alloc_kernel`.
    kernel_free: std::collections::HashMap<u32, Vec<u32>>,
    /// Xenos register file, memory-mapped at GPU_MMIO_BASE (register n at base + 4n).
    pub gpu_regs: Vec<u32>,
    /// XMA decoder registers: plain storage for now (no decoding is performed).
    pub xma_regs: Vec<u32>,
    /// Set whenever the title writes a GPU register; the run loop drains it.
    pub gpu_dirty: AtomicBool,
    /// Set whenever a fault, watch hit or GPU register write needs the run loop's attention, so the
    /// hot loop can skip all of those checks with one flag test.
    pub attn: AtomicBool,
    /// Debug: guest address whose 32-bit writes are reported (0 = off), and the last such write.
    pub watch_addr: u32,
    pub watch_hit: Option<(u32, u32)>,
}

/// XMA audio decoder register block (little-endian device registers, read via lwbrx/stwbrx).
pub const XMA_MMIO_BASE: u32 = 0x7FEA_0000;
const XMA_REG_COUNT: usize = 0x4000;
const XMA_MMIO_END: u32 = XMA_MMIO_BASE + (XMA_REG_COUNT as u32) * 4;

pub const GPU_MMIO_BASE: u32 = 0x7FC8_0000;
pub const GPU_REG_COUNT: usize = 0x10000;
const GPU_MMIO_END: u32 = GPU_MMIO_BASE + (GPU_REG_COUNT as u32) * 4;

/// Where kernel-owned structures (variable imports, module info, ...) live.
pub const KERNEL_DATA_BASE: u32 = 0x7E00_0000;

impl GuestMemory {
    pub fn new() -> Self {
        Self {
            pages: (0..PAGES).map(|_| None).collect(),
            fault: AtomicU64::new(0),
            kernel_next: KERNEL_DATA_BASE,
            kernel_free: std::collections::HashMap::new(),
            gpu_regs: vec![0; GPU_REG_COUNT],
            xma_regs: vec![0; XMA_REG_COUNT],
            watch_addr: 0,
            watch_hit: None,
            gpu_dirty: AtomicBool::new(false),
            attn: AtomicBool::new(false),
        }
    }

    /// Map (zero-filled) every page touching [addr, addr+size).
    pub fn map(&mut self, addr: u32, size: u32) {
        let first = addr as usize / PAGE;
        let last = (addr as usize + size as usize).div_ceil(PAGE).min(PAGES);
        for p in first..last {
            if self.pages[p].is_none() {
                self.pages[p] = Some(Box::new([0; PAGE]));
            }
        }
    }

    /// Unmap every page that lies entirely inside [addr, addr+size). Their contents are dropped,
    /// so remapping later yields zeroed memory.
    pub fn unmap(&mut self, addr: u32, size: u32) {
        let first = (addr as usize).div_ceil(PAGE);
        let last = ((addr as usize + size as usize) / PAGE).min(PAGES);
        for p in first..last {
            self.pages[p] = None;
        }
    }

    /// Bump-allocate zeroed, 16-byte-aligned kernel-owned guest memory.
    pub fn alloc_kernel(&mut self, size: u32) -> u32 {
        if let Some(addr) = self.kernel_free.get_mut(&size).and_then(Vec::pop) {
            for off in (0..size.next_multiple_of(4)).step_by(4) {
                self.write32_raw(addr + off, 0);
            }
            return addr;
        }
        let addr = self.kernel_next;
        assert!(
            self.kernel_next.saturating_add(size) < 0x8200_0000,
            "kernel heap exhausted: it would run into the executable image (leaking kernel allocations?)"
        );
        self.kernel_next = (self.kernel_next + size + 15) & !15;
        self.map(addr, size);
        addr
    }

    /// Return a block from `alloc_kernel` (same `size`) for reuse.
    pub fn free_kernel(&mut self, addr: u32, size: u32) {
        if addr != 0 {
            self.kernel_free.entry(size).or_default().push(addr);
        }
    }

    /// Store without watch/MMIO handling (kernel-owned pages only).
    fn write32_raw(&mut self, addr: u32, val: u32) {
        if let Some(p) = &mut self.pages[addr as usize / PAGE] {
            let off = addr as usize % PAGE;
            p[off..off + 4].copy_from_slice(&val.to_be_bytes());
        }
    }

    pub fn is_mapped(&self, addr: u32) -> bool {
        self.pages[addr as usize / PAGE].is_some()
    }

    /// Bulk copy into already-mapped memory (loader use; no fault tracking).
    pub fn write_bytes(&mut self, addr: u32, data: &[u8]) {
        for (i, b) in data.iter().enumerate() {
            self.write8(addr.wrapping_add(i as u32), *b);
        }
    }

    pub fn take_fault(&self) -> Option<Fault> {
        let v = self.fault.swap(0, Relaxed);
        (v != 0).then(|| Fault { addr: v as u32, write: (v >> 32) & 1 != 0 })
    }

    fn note(&self, addr: u32, write: bool) {
        self.attn.store(true, Relaxed);
        let _ = self.fault.compare_exchange(0, (1 << 63) | ((write as u64) << 32) | addr as u64, Relaxed, Relaxed);
    }
}

impl Bus for GuestMemory {
    fn page_table(&self) -> Option<*const *mut u8> {
        // Vec<Option<Box<[u8; PAGE]>>>: a thin nullable pointer per page, allocated once and never resized
        // a memory watch must see every store, so the JIT falls back to the access methods while one is set
        if self.watch_addr != 0 { None } else { Some(self.pages.as_ptr() as *const *mut u8) }
    }
    fn read8(&self, addr: u32) -> u8 {
        match &self.pages[addr as usize / PAGE] {
            Some(p) => p[addr as usize % PAGE],
            None => {
                self.note(addr, false);
                0
            }
        }
    }
    fn write8(&mut self, addr: u32, val: u8) {
        if addr & !3 == self.watch_addr {
            self.watch_hit = Some((addr, val as u32));
            self.attn.store(true, Relaxed);
        }
        match &mut self.pages[addr as usize / PAGE] {
            Some(p) => p[addr as usize % PAGE] = val,
            None => self.note(addr, true),
        }
    }

    fn read16(&self, addr: u32) -> u16 {
        let off = addr as usize % PAGE;
        if off <= PAGE - 2 {
            if let Some(p) = &self.pages[addr as usize / PAGE] {
                return u16::from_be_bytes([p[off], p[off + 1]]);
            }
        }
        u16::from_be_bytes([self.read8(addr), self.read8(addr.wrapping_add(1))])
    }

    fn read64(&self, addr: u32) -> u64 {
        let off = addr as usize % PAGE;
        if off <= PAGE - 8 {
            if let Some(p) = &self.pages[addr as usize / PAGE] {
                return u64::from_be_bytes(p[off..off + 8].try_into().unwrap());
            }
        }
        ((self.read32(addr) as u64) << 32) | self.read32(addr.wrapping_add(4)) as u64
    }

    fn write16(&mut self, addr: u32, val: u16) {
        let off = addr as usize % PAGE;
        if off <= PAGE - 2 && addr & !3 != self.watch_addr {
            if let Some(p) = &mut self.pages[addr as usize / PAGE] {
                p[off..off + 2].copy_from_slice(&val.to_be_bytes());
                return;
            }
        }
        for (i, b) in val.to_be_bytes().iter().enumerate() {
            self.write8(addr.wrapping_add(i as u32), *b);
        }
    }

    fn write64(&mut self, addr: u32, val: u64) {
        let off = addr as usize % PAGE;
        // watched addresses and MMIO take the general path so they still trigger
        if off <= PAGE - 8 && addr & !7 != self.watch_addr & !7 {
            if let Some(p) = &mut self.pages[addr as usize / PAGE] {
                p[off..off + 8].copy_from_slice(&val.to_be_bytes());
                return;
            }
        }
        self.write32(addr, (val >> 32) as u32);
        self.write32(addr.wrapping_add(4), val as u32);
    }

    fn read32(&self, addr: u32) -> u32 {
        let off = addr as usize % PAGE;
        if off <= PAGE - 4 {
            if let Some(p) = &self.pages[addr as usize / PAGE] {
                return u32::from_be_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]]);
            }
            if (GPU_MMIO_BASE..GPU_MMIO_END).contains(&addr) {
                return self.gpu_regs[((addr - GPU_MMIO_BASE) / 4) as usize];
            }
            if (XMA_MMIO_BASE..XMA_MMIO_END).contains(&addr) {
                return self.xma_regs[((addr - XMA_MMIO_BASE) / 4) as usize];
            }
        }
        u32::from_be_bytes([
            self.read8(addr),
            self.read8(addr.wrapping_add(1)),
            self.read8(addr.wrapping_add(2)),
            self.read8(addr.wrapping_add(3)),
        ])
    }

    fn write32(&mut self, addr: u32, val: u32) {
        if addr == self.watch_addr {
            self.watch_hit = Some((addr, val));
            self.attn.store(true, Relaxed);
        }
        let off = addr as usize % PAGE;
        if off <= PAGE - 4 {
            if let Some(p) = &mut self.pages[addr as usize / PAGE] {
                p[off..off + 4].copy_from_slice(&val.to_be_bytes());
                return;
            }
            if (GPU_MMIO_BASE..GPU_MMIO_END).contains(&addr) {
                self.gpu_regs[((addr - GPU_MMIO_BASE) / 4) as usize] = val;
                self.gpu_dirty.store(true, Relaxed);
                self.attn.store(true, Relaxed);
                return;
            }
            if (XMA_MMIO_BASE..XMA_MMIO_END).contains(&addr) {
                self.xma_regs[((addr - XMA_MMIO_BASE) / 4) as usize] = val;
                return;
            }
        }
        for (i, b) in val.to_be_bytes().iter().enumerate() {
            self.write8(addr.wrapping_add(i as u32), *b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_and_faults() {
        let mut m = GuestMemory::new();
        m.map(0x8200_0000, 0x2000);
        m.write32(0x8200_0ffe, 0x11223344); // straddles a page boundary
        assert_eq!(m.read32(0x8200_0ffe), 0x11223344);
        assert!(m.take_fault().is_none());
        assert_eq!(m.read32(0x1000), 0);
        assert_eq!(m.take_fault(), Some(Fault { addr: 0x1000, write: false }));
    }

    /// Interpreter speed against the real guest memory. Run with:
    /// cargo test --release -p x360 bench_guest_step_rate -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_guest_step_rate() {
        let mut m = GuestMemory::new();
        m.map(0x8200_0000, 0x1000);
        // addi r3,r3,1 ; add r4,r4,r3 ; cmpwi cr0,r3,0 ; bne loop
        for (i, w) in [0x3863_0001u32, 0x7C84_1A14, 0x2C03_0000, 0x4082_FFF4].iter().enumerate() {
            m.write32(0x8200_0000 + i as u32 * 4, *w);
        }
        let mut c = xenon_cpu::Cpu::new();
        c.pc = 0x8200_0000;
        let n = 100_000_000u64;
        let t = std::time::Instant::now();
        for _ in 0..n {
            c.step(&mut m).unwrap();
        }
        println!("{:.1} M instructions/sec", n as f64 / t.elapsed().as_secs_f64() / 1e6);
    }

    #[test]
    #[ignore]
    fn bench_guest_memory_loop() {
        let mut m = GuestMemory::new();
        m.map(0x8200_0000, 0x1000);
        m.map(0x4000_0000, 0x1000);
        // lwz r5,0(r6) ; addi r5,r5,1 ; stw r5,0(r6) ; lbz r7,1(r6) ; rlwinm r8,r5,2,0,29 ; b loop
        for (i, w) in [0x80A6_0000u32, 0x38A5_0001, 0x90A6_0000, 0x88E6_0001, 0x54A8_103A, 0x4BFF_FFEC].iter().enumerate() {
            m.write32(0x8200_0000 + i as u32 * 4, *w);
        }
        let mut c = xenon_cpu::Cpu::new();
        c.pc = 0x8200_0000;
        c.gpr[6] = 0x4000_0000;
        let n = 100_000_000u64;
        let t = std::time::Instant::now();
        for _ in 0..n {
            c.step(&mut m).unwrap();
        }
        println!("{:.1} M instructions/sec (memory mix)", n as f64 / t.elapsed().as_secs_f64() / 1e6);
    }
}
