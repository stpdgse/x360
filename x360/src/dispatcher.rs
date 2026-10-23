//! Dispatcher objects (events, semaphores, thread signals) that live in *guest memory*,
//! laid out like NT's DISPATCHER_HEADER so both handle-based (`Nt*`) and pointer-based (`Ke*`)
//! callers see the same state:
//!   +0 type u8 (0 notification event, 1 synchronization event, 2 mutant, 5 semaphore, 6 thread)
//!   +1 absolute, +2 size in dwords, +3 inserted, +4 signal state i32, +8/+12 wait list head
//!   semaphores keep their limit at +16.

use crate::memory::GuestMemory;
use xenon_cpu::memory::Bus;

pub const TYPE_NOTIFICATION_EVENT: u8 = 0;
pub const TYPE_SYNC_EVENT: u8 = 1;
pub const TYPE_MUTANT: u8 = 2;
pub const TYPE_SEMAPHORE: u8 = 5;
pub const TYPE_THREAD: u8 = 6;

pub const STATUS_SEMAPHORE_LIMIT_EXCEEDED: u32 = 0xC000_0047;

/// Bytes to reserve for any dispatcher object we create ourselves.
pub const OBJECT_SIZE: u32 = 0x20;

#[derive(Debug, PartialEq, Eq)]
pub enum Acquire {
    Satisfied,
    NotSignaled,
    /// The header isn't a dispatcher type we understand.
    Unknown,
}

fn init_header(mem: &mut GuestMemory, addr: u32, kind: u8, size_dwords: u8, state: i32) {
    mem.write8(addr, kind);
    mem.write8(addr + 1, 0);
    mem.write8(addr + 2, size_dwords);
    mem.write8(addr + 3, 0);
    mem.write32(addr + 4, state as u32);
    mem.write32(addr + 8, addr + 8);
    mem.write32(addr + 12, addr + 8);
}

pub fn init_event(mem: &mut GuestMemory, addr: u32, manual_reset: bool, signaled: bool) {
    let kind = if manual_reset { TYPE_NOTIFICATION_EVENT } else { TYPE_SYNC_EVENT };
    init_header(mem, addr, kind, 4, signaled as i32);
}

pub fn init_semaphore(mem: &mut GuestMemory, addr: u32, count: i32, limit: i32) {
    init_header(mem, addr, TYPE_SEMAPHORE, 5, count);
    mem.write32(addr + 16, limit as u32);
}

pub fn init_thread(mem: &mut GuestMemory, addr: u32) {
    init_header(mem, addr, TYPE_THREAD, 0x18, 0);
}

pub fn signal(mem: &mut GuestMemory, addr: u32) {
    mem.write32(addr + 4, 1);
}

/// Would a wait on `addr` be satisfied right now? (No state change.)
pub fn peek_signaled(mem: &GuestMemory, addr: u32) -> bool {
    match mem.read8(addr) {
        TYPE_NOTIFICATION_EVENT | TYPE_SYNC_EVENT | TYPE_SEMAPHORE | TYPE_MUTANT | TYPE_THREAD => (mem.read32(addr + 4) as i32) > 0,
        _ => false,
    }
}

/// Try to satisfy a wait on the object at `addr`, consuming its signal where NT semantics say so.
pub fn try_acquire(mem: &mut GuestMemory, addr: u32) -> Acquire {
    let state = mem.read32(addr + 4) as i32;
    match mem.read8(addr) {
        TYPE_NOTIFICATION_EVENT | TYPE_THREAD => {
            if state > 0 { Acquire::Satisfied } else { Acquire::NotSignaled }
        }
        TYPE_SYNC_EVENT => {
            if state > 0 {
                mem.write32(addr + 4, 0);
                Acquire::Satisfied
            } else {
                Acquire::NotSignaled
            }
        }
        TYPE_SEMAPHORE | TYPE_MUTANT => {
            if state > 0 {
                mem.write32(addr + 4, (state - 1) as u32);
                Acquire::Satisfied
            } else {
                Acquire::NotSignaled
            }
        }
        _ => Acquire::Unknown,
    }
}

/// Set an event; returns the previous signal state.
pub fn set_event(mem: &mut GuestMemory, addr: u32) -> i32 {
    let prev = mem.read32(addr + 4) as i32;
    mem.write32(addr + 4, 1);
    prev
}

pub fn reset_event(mem: &mut GuestMemory, addr: u32) -> i32 {
    let prev = mem.read32(addr + 4) as i32;
    mem.write32(addr + 4, 0);
    prev
}

/// Release `n` units of a semaphore; returns the previous count.
pub fn release_semaphore(mem: &mut GuestMemory, addr: u32, n: i32) -> Result<i32, u32> {
    let cur = mem.read32(addr + 4) as i32;
    let limit = mem.read32(addr + 16) as i32;
    if cur.saturating_add(n) > limit {
        return Err(STATUS_SEMAPHORE_LIMIT_EXCEEDED);
    }
    mem.write32(addr + 4, (cur + n) as u32);
    Ok(cur)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_with(addr: u32) -> GuestMemory {
        let mut m = GuestMemory::new();
        m.map(addr, 0x100);
        m
    }

    #[test]
    fn semaphore_counts_and_limit() {
        let mut m = mem_with(0x4000_0000);
        let s = 0x4000_0000;
        init_semaphore(&mut m, s, 0, 2);
        assert_eq!(try_acquire(&mut m, s), Acquire::NotSignaled);
        assert_eq!(release_semaphore(&mut m, s, 2), Ok(0));
        assert_eq!(release_semaphore(&mut m, s, 1), Err(STATUS_SEMAPHORE_LIMIT_EXCEEDED));
        assert_eq!(try_acquire(&mut m, s), Acquire::Satisfied);
        assert_eq!(try_acquire(&mut m, s), Acquire::Satisfied);
        assert_eq!(try_acquire(&mut m, s), Acquire::NotSignaled);
    }

    #[test]
    fn event_auto_vs_manual_reset() {
        let mut m = mem_with(0x4000_0000);
        let (auto, manual) = (0x4000_0000, 0x4000_0040);
        init_event(&mut m, auto, false, true);
        init_event(&mut m, manual, true, true);
        assert_eq!(try_acquire(&mut m, auto), Acquire::Satisfied);
        assert_eq!(try_acquire(&mut m, auto), Acquire::NotSignaled); // consumed
        assert_eq!(try_acquire(&mut m, manual), Acquire::Satisfied);
        assert_eq!(try_acquire(&mut m, manual), Acquire::Satisfied); // stays signaled
        assert_eq!(set_event(&mut m, auto), 0);
        assert_eq!(reset_event(&mut m, manual), 1);
        assert_eq!(try_acquire(&mut m, manual), Acquire::NotSignaled);
    }

    #[test]
    fn thread_signal_is_sticky() {
        let mut m = mem_with(0x4000_0000);
        init_thread(&mut m, 0x4000_0000);
        assert_eq!(try_acquire(&mut m, 0x4000_0000), Acquire::NotSignaled);
        signal(&mut m, 0x4000_0000);
        assert_eq!(try_acquire(&mut m, 0x4000_0000), Acquire::Satisfied);
        assert_eq!(try_acquire(&mut m, 0x4000_0000), Acquire::Satisfied);
    }
}
