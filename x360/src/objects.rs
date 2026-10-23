//! Kernel object manager: the handle table. Events and semaphores are *not* stored here —
//! their state lives in guest memory (`dispatcher.rs`) and a handle just points at it.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

pub const STATUS_SUCCESS: u32 = 0;
pub const STATUS_TIMEOUT: u32 = 0x0000_0102;
pub const STATUS_INVALID_HANDLE: u32 = 0xC000_0008;
pub const STATUS_OBJECT_NAME_NOT_FOUND: u32 = 0xC000_0034;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KObject {
    /// An event/semaphore/... whose dispatcher header is at guest address `addr`.
    Dispatcher { addr: u32 },
    /// An open file or directory on the game disc.
    File { entry: crate::iso::FileEntry, pos: u64 },
    /// A guest thread (`index` into the kernel's thread list); signaled once it has exited.
    Thread { index: usize, terminated: bool },
    /// A file or directory on the host-backed writable storage (`hostfs.rs`).
    HostFile { path: std::path::PathBuf, pos: u64, is_dir: bool },
    /// A XAM enumerator (achievements, saves, ...): `items` are pre-encoded records of `item_size` bytes.
    Enumerator { item_size: u32, items: std::collections::VecDeque<Vec<u8>> },
    /// XAM notification listener. `mask` selects which notification areas it receives.
    Listener { mask: u64, queue: VecDeque<(u32, u32)> },
}

#[derive(Debug, PartialEq, Eq)]
pub enum WaitResult {
    Satisfied,
    /// Not signaled; the caller decides between timeout and blocking.
    NotSignaled,
    InvalidHandle,
}

#[derive(Serialize, Deserialize)]
pub struct ObjectTable {
    next: u32,
    objects: HashMap<u32, KObject>,
}

impl ObjectTable {
    pub fn new() -> Self {
        Self { next: 0xF800_0100, objects: HashMap::new() }
    }

    pub fn create(&mut self, obj: KObject) -> u32 {
        let h = self.next;
        self.next += 4;
        self.objects.insert(h, obj);
        h
    }

    pub fn close(&mut self, handle: u32) -> u32 {
        if self.objects.remove(&handle).is_some() { STATUS_SUCCESS } else { STATUS_INVALID_HANDLE }
    }

    pub fn get(&self, handle: u32) -> Option<&KObject> {
        self.objects.get(&handle)
    }

    /// Guest address of the dispatcher object behind a handle, if it is one.
    pub fn dispatcher_addr(&self, handle: u32) -> Option<u32> {
        match self.objects.get(&handle) {
            Some(KObject::Dispatcher { addr }) => Some(*addr),
            _ => None,
        }
    }

    /// Wait on a non-dispatcher handle (dispatcher handles are resolved to an address by the caller).
    pub fn try_wait(&mut self, handle: u32) -> WaitResult {
        match self.objects.get(&handle) {
            Some(KObject::Thread { terminated: true, .. }) => WaitResult::Satisfied,
            Some(KObject::Listener { queue, .. }) if !queue.is_empty() => WaitResult::Satisfied,
            Some(_) => WaitResult::NotSignaled,
            None => WaitResult::InvalidHandle,
        }
    }

    /// Side-effect-free version of [`try_wait`]: is the handle's object signaled?
    pub fn peek_wait(&self, handle: u32) -> bool {
        match self.objects.get(&handle) {
            Some(KObject::Thread { terminated, .. }) => *terminated,
            Some(KObject::Listener { queue, .. }) => !queue.is_empty(),
            Some(_) => false,
            None => true,
        }
    }

    /// Queue a notification (`id`, `param`) on every listener whose area mask covers it.
    /// The area index lives in the id's top bits; mask bit `1 << area` selects it.
    pub fn broadcast(&mut self, id: u32, param: u32) {
        let area = (id >> 25) & 0x3F;
        for obj in self.objects.values_mut() {
            if let KObject::Listener { mask, queue } = obj {
                if *mask & (1u64 << area) != 0 {
                    queue.push_back((id, param));
                }
            }
        }
    }

    /// Dequeue the next notification for a listener (the first one with `match_id`, if non-zero).
    pub fn listener_pop(&mut self, handle: u32, match_id: u32) -> Option<(u32, u32)> {
        match self.objects.get_mut(&handle) {
            Some(KObject::Listener { queue, .. }) => {
                let pos = if match_id == 0 { Some(0) } else { queue.iter().position(|e| e.0 == match_id) };
                pos.and_then(|p| queue.remove(p))
            }
            _ => None,
        }
    }

    /// Mutable access to an open file: (entry, position).
    pub fn file_mut(&mut self, handle: u32) -> Option<(&crate::iso::FileEntry, &mut u64)> {
        match self.objects.get_mut(&handle) {
            Some(KObject::File { entry, pos }) => Some((&*entry, pos)),
            _ => None,
        }
    }

    pub fn set_thread_terminated(&mut self, handle: u32) {
        if let Some(KObject::Thread { terminated, .. }) = self.objects.get_mut(&handle) {
            *terminated = true;
        }
    }

    /// Mutable access to an open host-storage file: (path, position, is_dir).
    pub fn host_mut(&mut self, handle: u32) -> Option<(&std::path::PathBuf, &mut u64, bool)> {
        match self.objects.get_mut(&handle) {
            Some(KObject::HostFile { path, pos, is_dir }) => Some((&*path, pos, *is_dir)),
            _ => None,
        }
    }

    /// Take up to `max` records from an enumerator; None if the handle isn't one.
    pub fn enumerator_take(&mut self, handle: u32, max: usize) -> Option<(u32, Vec<Vec<u8>>)> {
        match self.objects.get_mut(&handle) {
            Some(KObject::Enumerator { item_size, items }) => {
                let n = max.min(items.len());
                Some((*item_size, items.drain(..n).collect()))
            }
            _ => None,
        }
    }

    /// Thread-list index for a thread handle.
    pub fn thread_index(&self, handle: u32) -> Option<usize> {
        match self.objects.get(&handle) {
            Some(KObject::Thread { index, .. }) => Some(*index),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_close_and_lookup() {
        let mut t = ObjectTable::new();
        let h = t.create(KObject::Dispatcher { addr: 0x4000_0000 });
        assert_eq!(t.dispatcher_addr(h), Some(0x4000_0000));
        assert_eq!(t.close(h), STATUS_SUCCESS);
        assert_eq!(t.close(h), STATUS_INVALID_HANDLE);
        let th = t.create(KObject::Thread { index: 2, terminated: false });
        assert_eq!(t.try_wait(th), WaitResult::NotSignaled);
        t.set_thread_terminated(th);
        assert_eq!(t.try_wait(th), WaitResult::Satisfied);
        assert_eq!(t.thread_index(th), Some(2));
    }
}
