//! Guest thread scheduler: thread contexts, creation, blocking waits, round-robin picking
//! on a virtual clock (executed instructions; 1 NT tick of 100ns ~= 320 cycles at 3.2GHz).

use serde::{Deserialize, Serialize};
use crate::dispatcher::{self, Acquire};
use crate::kernel::{Kernel, Next, TlsInfo};
use crate::memory::GuestMemory;
use crate::objects::{WaitResult, STATUS_INVALID_HANDLE, STATUS_SUCCESS, STATUS_TIMEOUT};
use crate::vm::{MEM_COMMIT, MEM_RESERVE};
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

/// Display-controller vblank status (GPU register index of MMIO address 0x7FC86544).
pub const REG_D1_VBLANK_STATUS: usize = 0x1951;
pub const CYCLES_PER_TICK: u64 = 320;
/// One video frame of the virtual 3.2GHz clock at 60Hz.
/// Virtual CPU clock (3.2GHz) divided by the refresh rate gives the vsync period.
pub const CPU_HZ: f64 = 3_200_000_000.0;
/// Library index used by internal thunks (never a real import library).
pub const INTERNAL_LIB: u32 = 0x7FFF;
pub const INTERNAL_THREAD_EXIT: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Wait {
    None,
    /// Wait on a non-dispatcher handle (e.g. a thread).
    Handle { handle: u32, wake_at: Option<u64> },
    /// Wait on a dispatcher object in guest memory.
    Dispatcher { addr: u32, wake_at: Option<u64> },
    Sleep { wake_at: u64 },
    CritSec { addr: u32 },
    /// Blocked acquiring a kernel spinlock: the lock word at `addr` holds the owner's thread id (0 = free).
    SpinLock { addr: u32 },
    /// KeWaitForMultipleObjects / NtWaitForMultipleObjectsEx: wait for any (or all) of several objects.
    Multi { targets: Vec<WaitTarget>, all: bool, wake_at: Option<u64> },
}

/// One object in a multi-wait: a dispatcher header in guest memory, or a handle to a non-dispatcher object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaitTarget {
    Object(u32),
    Handle(u32),
}

fn target_signaled(objects: &crate::objects::ObjectTable, mem: &GuestMemory, t: WaitTarget) -> bool {
    match t {
        WaitTarget::Object(addr) => dispatcher::peek_signaled(mem, addr),
        WaitTarget::Handle(h) => match objects.dispatcher_addr(h) {
            Some(addr) => dispatcher::peek_signaled(mem, addr),
            None => objects.peek_wait(h),
        },
    }
}

impl Kernel {
    /// Debug: one line per unsignaled target of a multi-wait (X360_BACKTRACES).
    pub fn describe_wait(&self, mem: &GuestMemory, idx: usize) -> String {
        match &self.threads[idx].wait {
            Wait::Multi { targets, all, .. } => {
                let mut s = format!("multi-wait all={all} n={}:", targets.len());
                for (i, &t) in targets.iter().enumerate() {
                    if !target_signaled(&self.objects, mem, t) {
                        let a = match t {
                            WaitTarget::Object(a) => a,
                            WaitTarget::Handle(h) => self.objects.dispatcher_addr(h).unwrap_or(0),
                        };
                        s += &format!(" [{i}] {t:?} unsignaled hdr={:08x}", if a != 0 { mem.read32(a) } else { 0 });
                    }
                }
                s
            }
            w => format!("{w:?}"),
        }
    }
}

/// Consume the target's signal (auto-reset events, semaphores) if it is signaled.
fn target_acquire(objects: &mut crate::objects::ObjectTable, mem: &mut GuestMemory, t: WaitTarget) -> bool {
    match t {
        WaitTarget::Object(addr) => dispatcher::try_acquire(mem, addr) == Acquire::Satisfied,
        WaitTarget::Handle(h) => match objects.dispatcher_addr(h) {
            Some(addr) => dispatcher::try_acquire(mem, addr) == Acquire::Satisfied,
            None => objects.try_wait(h) == WaitResult::Satisfied,
        },
    }
}

/// Resolve a multi-wait against current state: Some(status) if it completes (index of the satisfied
/// object for wait-any, 0 for wait-all), None if it must keep blocking.
fn try_multi(
    objects: &mut crate::objects::ObjectTable,
    mem: &mut GuestMemory,
    targets: &[WaitTarget],
    all: bool,
    timed_out: bool,
) -> Option<u32> {
    if all {
        if targets.iter().all(|&t| target_signaled(objects, mem, t)) {
            for &t in targets {
                target_acquire(objects, mem, t);
            }
            return Some(STATUS_SUCCESS);
        }
    } else {
        for (i, &t) in targets.iter().enumerate() {
            if target_acquire(objects, mem, t) {
                return Some(STATUS_SUCCESS + i as u32);
            }
        }
    }
    timed_out.then_some(STATUS_TIMEOUT)
}

#[derive(Serialize, Deserialize)]
pub struct ThreadCtx {
    pub cpu: Cpu,
    pub id: u32,
    pub handle: u32,
    pub suspend: u32,
    pub wait: Wait,
    pub terminated: bool,
    pub exit_code: u32,
    /// Guest address of the KTHREAD-like object (PCR+0x100 points here).
    pub thread_obj: u32,
    /// Spawned by the kernel to run an interrupt callback (vsync), not by the title.
    pub is_interrupt: bool,
    /// Spawned to run a title-termination notification during shutdown.
    pub is_shutdown: bool,
    /// Audio render-driver callback thread.
    pub is_audio: bool,
    /// Kernel-spawned helper nobody holds a handle to (audio/interrupt callbacks): its resources and
    /// its slot are reused when it exits, so the endless stream of them can't exhaust memory.
    pub recyclable: bool,
    /// Instructions this thread has executed (for the status line's CPU-share report).
    pub executed: u64,
    /// Resources to free on exit: stack (base, size), TLS block, and the object sizes.
    pub stack: (u32, u32),
    pub tls_block: u32,
    /// Current interrupt request level (0 passive, 1 APC, 2 dispatch, ...). A thread at DISPATCH
    /// level or above is never preempted by the scheduler.
    pub irql: u8,
    /// KeEnterCriticalRegion nesting depth (would disable normal kernel APCs).
    pub critical_depth: u32,
    /// Scheduling priority, -15..=15; higher runs first, equals round-robin.
    pub priority: i32,
    /// Allowed hardware threads (bit n = core n) and the core currently reported in the PCR.
    pub affinity: u32,
    pub core: u32,
    /// Guest address of this thread's PCR (kept here because `cpu` is checked out while it runs).
    pub pcr: u32,
}

impl Kernel {
    /// `lis r11,INTERNAL_LIB ; ori r11,r11,1 ; sc ; blr` — return target of every thread entry.
    fn exit_thunk(&mut self, mem: &mut GuestMemory) -> u32 {
        if self.exit_thunk == 0 {
            let t = mem.alloc_kernel(16);
            mem.write32(t, 0x3D60_0000 | INTERNAL_LIB);
            mem.write32(t + 4, 0x616B_0000 | INTERNAL_THREAD_EXIT);
            mem.write32(t + 8, 0x4400_0002);
            mem.write32(t + 12, 0x4E80_0020);
            self.exit_thunk = t;
        }
        self.exit_thunk
    }

    /// Create a thread ready to run `entry(a3, a4)`. Returns its index in `threads`.
    /// PCR: +0 TLS, +0x30 self, +0x70/+0x74 stack base/limit, +0x100 thread object, +0x10C cpu number.
    /// Thread object: +0x14C thread id.
    pub fn spawn_thread(
        &mut self,
        mem: &mut GuestMemory,
        entry: u32,
        a3: u64,
        a4: u64,
        stack_size: u32,
        suspended: bool,
    ) -> Result<usize, String> {
        let stack_size = stack_size.max(0x10000);
        let (sbase, ssize) = self
            .vm
            .allocate(mem, 0, stack_size, MEM_RESERVE | MEM_COMMIT)
            .map_err(|s| format!("no memory for thread stack ({s:#x})"))?;
        let top = sbase + ssize;
        let pcr = mem.alloc_kernel(0x1000);
        let thread_obj = mem.alloc_kernel(0x1000);
        let id = self.next_thread_id;
        self.next_thread_id += 1;

        let mut tls_block = 0;
        if let Some(t) = self.tls_info {
            let block = mem.alloc_kernel(t.data_size.max(16) + t.slot_count * 4);
            copy_tls_template(mem, block, &t);
            mem.write32(pcr, block);
            tls_block = block;
        }
        mem.write32(pcr + 0x30, pcr);
        mem.write32(pcr + 0x70, top);
        mem.write32(pcr + 0x74, sbase);
        mem.write32(pcr + 0x100, thread_obj);
        mem.write32(pcr + 0x10C, 0);
        dispatcher::init_thread(mem, thread_obj);
        mem.write32(thread_obj + 0x14C, id);

        let mut cpu = Cpu::new();
        cpu.cpi = std::env::var("X360_CPI").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
        cpu.pc = entry as u64;
        cpu.gpr[1] = (top - 0x100) as u64;
        cpu.gpr[3] = a3;
        cpu.gpr[4] = a4;
        cpu.gpr[13] = pcr as u64;
        cpu.lr = self.exit_thunk(mem) as u64;
        cpu.msr = 0x9030;

        let index = self.free_slots.pop().unwrap_or(self.threads.len());
        let handle = self.objects.create(crate::objects::KObject::Thread { index, terminated: false });
        let ctx = ThreadCtx {
            cpu,
            id,
            handle,
            suspend: suspended as u32,
            wait: Wait::None,
            terminated: false,
            exit_code: 0,
            thread_obj,
            is_interrupt: false,
            is_shutdown: false,
            is_audio: false,
            recyclable: false,
            executed: 0,
            stack: (sbase, ssize),
            tls_block,
            irql: 0,
            critical_depth: 0,
            priority: 0,
            affinity: 0x3F,
            core: 0,
            pcr,
        };
        if index == self.threads.len() {
            self.threads.push(ctx);
        } else {
            self.threads[index] = ctx;
        }
        Ok(index)
    }

    /// Move a thread's CPU out so the run loop can execute it.
    pub fn take_cpu(&mut self, i: usize) -> Cpu {
        self.cur = i;
        self.current_thread = self.threads[i].id;
        let mut cpu = std::mem::take(&mut self.threads[i].cpu);
        cpu.cycles = self.global_cycles;
        cpu
    }

    pub fn put_cpu(&mut self, i: usize, cpu: Cpu) {
        self.global_cycles = cpu.cycles;
        self.threads[i].cpu = cpu;
    }

    /// If thread `i` can run now (possibly because its wait just got satisfied), make it runnable.
    fn try_wake(&mut self, i: usize, mem: &mut GuestMemory) -> bool {
        let now = self.global_cycles;
        let t = &mut self.threads[i];
        if t.terminated || t.suspend > 0 {
            return false;
        }
        match t.wait.clone() {
            Wait::None => true,
            Wait::Sleep { wake_at } => {
                if now >= wake_at {
                    t.cpu.gpr[3] = STATUS_SUCCESS as u64;
                    t.wait = Wait::None;
                    true
                } else {
                    false
                }
            }
            Wait::Handle { handle, wake_at } => {
                let status = match self.objects.try_wait(handle) {
                    WaitResult::Satisfied => Some(STATUS_SUCCESS),
                    WaitResult::InvalidHandle => Some(STATUS_INVALID_HANDLE),
                    WaitResult::NotSignaled if wake_at.is_some_and(|w| now >= w) => Some(STATUS_TIMEOUT),
                    WaitResult::NotSignaled => None,
                };
                match status {
                    Some(s) => {
                        t.cpu.gpr[3] = crate::kernel::sign_extend_result(s as u64);
                        t.wait = Wait::None;
                        true
                    }
                    None => false,
                }
            }
            Wait::Dispatcher { addr, wake_at } => {
                let status = match dispatcher::try_acquire(mem, addr) {
                    Acquire::Satisfied => Some(STATUS_SUCCESS),
                    _ if wake_at.is_some_and(|w| now >= w) => Some(STATUS_TIMEOUT),
                    _ => None,
                };
                match status {
                    Some(s) => {
                        t.cpu.gpr[3] = s as u64;
                        t.wait = Wait::None;
                        true
                    }
                    None => false,
                }
            }
            Wait::Multi { targets, all, wake_at } => {
                let timed_out = wake_at.is_some_and(|w| now >= w);
                match try_multi(&mut self.objects, mem, &targets, all, timed_out) {
                    Some(status) => {
                        t.cpu.gpr[3] = status as u64;
                        t.wait = Wait::None;
                        true
                    }
                    None => false,
                }
            }
            Wait::SpinLock { addr } => {
                if mem.read32(addr) == 0 {
                    mem.write32(addr, t.id);
                    t.wait = Wait::None;
                    true
                } else {
                    false
                }
            }
            Wait::CritSec { addr } => {
                if mem.read32(addr + 0x18) == 0 {
                    mem.write32(addr + 0x18, t.id);
                    mem.write32(addr + 0x14, 1);
                    mem.write32(addr + 0x10, 0);
                    t.wait = Wait::None;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Once the title has registered a graphics interrupt callback, fire it every virtual frame
    /// (60Hz of the 3.2GHz clock) on a fresh thread: callback(source = 0 for vsync, context).
    fn vsync_period(&self) -> u64 {
        (CPU_HZ / self.display.refresh_hz as f64) as u64
    }

    fn fire_vsync_if_due(&mut self, mem: &mut GuestMemory) -> Result<(), String> {
        let (cb, ctx) = self.gfx_callback;
        if cb == 0 {
            return Ok(());
        }
        if self.next_vsync == 0 {
            self.next_vsync = self.global_cycles + self.vsync_period();
        }
        if self.global_cycles >= self.next_vsync {
            self.next_vsync += self.vsync_period();
            // Don't pile up interrupt threads if the previous callbacks haven't finished.
            let pending = self.threads.iter().filter(|t| t.is_interrupt && !t.terminated).count();
            if pending < 2 {
                // The ISR only does vsync work when the display controller's status register says a
                // vblank happened (register 0x1951 = MMIO 0x7FC86544, bit 0).
                mem.gpu_regs[REG_D1_VBLANK_STATUS] |= 1;
                self.spawn_gfx_interrupt(mem, 0, 0)?;
                self.vsyncs += 1;
            }
        }
        Ok(())
    }

    /// Run the title's graphics interrupt callback on a fresh interrupt thread:
    /// callback(source, context) with source 0 = vsync, 1 = command-processor interrupt.
    fn spawn_gfx_interrupt(&mut self, mem: &mut GuestMemory, source: u64, core: u32) -> Result<(), String> {
        let (cb, ctx) = self.gfx_callback;
        if cb == 0 {
            return Ok(());
        }
        let idx = self.spawn_thread(mem, cb, source, ctx as u64, 0x10000, false)?;
        self.threads[idx].is_interrupt = true;
        self.threads[idx].recyclable = true;
        self.threads[idx].priority = 15; // interrupt handlers pre-empt ordinary threads
        self.threads[idx].irql = 2; // and run at DISPATCH level
        self.threads[idx].affinity = 1 << core;
        self.set_thread_core(mem, idx, core);
        Ok(())
    }

    /// Tell a thread which hardware core it runs on: the PCR's CPU-number byte at +0x10C, which
    /// interrupt handlers use to pick their bit in per-CPU pending masks.
    pub fn set_thread_core(&mut self, mem: &mut GuestMemory, idx: usize, core: u32) {
        self.threads[idx].core = core;
        mem.write8(self.threads[idx].pcr + 0x10C, core as u8);
    }

    /// Thread index for a thread-object pointer (what Ke* thread calls receive).
    fn thread_by_obj(&self, obj: u32) -> Option<usize> {
        self.threads.iter().position(|t| t.thread_obj == obj)
    }

    /// Title termination: run every registered terminate notification (highest priority first)
    /// on its own thread. Returns how many were started; the run loop waits for them.
    pub fn begin_shutdown(&mut self, mem: &mut GuestMemory) -> Result<usize, String> {
        let mut cbs = std::mem::take(&mut self.terminate_callbacks);
        cbs.sort_by(|a, b| b.1.cmp(&a.1));
        for &(cb, _, _) in &cbs {
            let idx = self.spawn_thread(mem, cb, 0, 0, 0x10000, false)?;
            self.threads[idx].is_shutdown = true;
        }
        Ok(cbs.len())
    }

    pub fn shutdown_threads_alive(&self) -> bool {
        self.threads.iter().any(|t| t.is_shutdown && !t.terminated)
    }

    /// Deliver CP interrupts the GPU has queued (PM4_INTERRUPT packets), each on the core its
    /// mask selects (lowest set bit).
    pub fn deliver_gpu_interrupts(&mut self, mem: &mut GuestMemory) -> Result<(), String> {
        for mask in std::mem::take(&mut self.gpu.pending_interrupts) {
            self.cp_interrupts += 1;
            let core = if mask == 0 { 0 } else { mask.trailing_zeros() };
            self.spawn_gfx_interrupt(mem, 1, core)?;
        }
        Ok(())
    }

    /// Could thread `i` run right now? Looks at its wait without consuming any signal.
    fn peek_runnable(&self, i: usize, mem: &GuestMemory) -> bool {
        let t = &self.threads[i];
        if t.terminated || t.suspend > 0 {
            return false;
        }
        let now = self.global_cycles;
        match t.wait {
            Wait::None => true,
            Wait::Sleep { wake_at } => now >= wake_at,
            Wait::Handle { handle, wake_at } => {
                wake_at.is_some_and(|w| now >= w) || self.objects.peek_wait(handle)
            }
            Wait::Dispatcher { addr, wake_at } => wake_at.is_some_and(|w| now >= w) || dispatcher::peek_signaled(mem, addr),
            Wait::CritSec { addr } => mem.read32(addr + 0x18) == 0,
            Wait::SpinLock { addr } => mem.read32(addr) == 0,
            Wait::Multi { ref targets, all, wake_at } => {
                wake_at.is_some_and(|w| now >= w) || {
                    let sig = |&x: &WaitTarget| target_signaled(&self.objects, mem, x);
                    if all { targets.iter().all(sig) } else { targets.iter().any(sig) }
                }
            }
        }
    }

    /// Choose the next thread to run: the highest-priority runnable thread, round-robin among equals
    /// starting after the current one. Advances the virtual clock to the earliest timeout if
    /// everyone is blocked. Err = deadlock.
    pub fn pick_next(&mut self, mem: &mut GuestMemory) -> Result<usize, String> {
        let started = std::time::Instant::now();
        let r = self.pick_next_inner(mem);
        self.picks += 1;
        self.pick_time += started.elapsed();
        r
    }

    fn pick_next_inner(&mut self, mem: &mut GuestMemory) -> Result<usize, String> {
        loop {
            self.fire_vsync_if_due(mem)?;
            self.xam_tick();
            self.audio_tick(mem)?;
            let n = self.threads.len();
            let mut best: Option<(usize, i32)> = None;
            for k in 1..=n {
                let i = (self.cur + k) % n;
                if self.peek_runnable(i, mem) {
                    // The real console has six hardware threads, so a busy high-priority thread
                    // never starves the rest. We time-slice one CPU, so strict priority would
                    // (it did: a spinning main thread starved ready workers). Only interrupt
                    // handlers pre-empt; everything else is round-robin.
                    let p = if self.threads[i].is_interrupt || self.threads[i].is_audio { 1 } else { 0 };
                    if best.is_none_or(|(_, bp)| p > bp) {
                        best = Some((i, p));
                    }
                }
            }
            if let Some((i, _)) = best {
                if self.try_wake(i, mem) {
                    return Ok(i);
                }
            }
            let earliest = self
                .threads
                .iter()
                .filter(|t| !t.terminated && t.suspend == 0)
                .filter_map(|t| match t.wait {
                    Wait::Sleep { wake_at } => Some(wake_at),
                    Wait::Handle { wake_at: Some(w), .. }
                    | Wait::Dispatcher { wake_at: Some(w), .. }
                    | Wait::Multi { wake_at: Some(w), .. } => Some(w),
                    _ => None,
                })
                .min();
            // The next vsync is also an event that can wake the world up.
            let vsync = if self.gfx_callback.0 != 0 && self.next_vsync != 0 { Some(self.next_vsync) } else { None };
            let events = [earliest, vsync, self.audio.next_tick()];
            let target = events.into_iter().flatten().min();
            match target {
                Some(w) if w > self.global_cycles => self.global_cycles = w,
                Some(_) => self.global_cycles += 1,
                None => return Err(self.deadlock_report()),
            }
        }
    }

    /// One line per thread: id, state, and where it is. Used for end-of-run diagnostics.
    pub fn thread_summary(&self, running: Option<(usize, u64)>) -> String {
        let mut s = String::new();
        for (i, t) in self.threads.iter().enumerate() {
            let pc = match running {
                Some((idx, pc)) if idx == i => pc,
                _ => t.cpu.pc,
            };
            let state = if t.terminated {
                "terminated".to_string()
            } else if t.suspend > 0 {
                format!("suspended x{}", t.suspend)
            } else {
                match &t.wait {
                    Wait::None => if running.is_some_and(|(idx, _)| idx == i) { "running".into() } else { "ready".into() },
                    Wait::Handle { handle, .. } => format!("waiting on handle {handle:#x}"),
                    Wait::Dispatcher { addr, wake_at } => format!("waiting on object {addr:#x} (timeout {wake_at:?})"),
                    Wait::Sleep { wake_at } => format!("sleeping until {wake_at}"),
                    Wait::CritSec { addr } => format!("waiting for critical section {addr:#x}"),
                    Wait::SpinLock { addr } => format!("spinning on lock {addr:#x}"),
                    Wait::Multi { targets, all, wake_at } => {
                        format!("waiting on {} of {targets:x?} (timeout {wake_at:?})", if *all { "all" } else { "any" })
                    }
                }
            };
            let name = self.thread_names.get(&t.id).map(|n| format!(" \"{n}\"")).unwrap_or_default();
            let kind = if t.is_audio { " [audio cb]" } else if t.is_interrupt { " [irq]" } else { "" };
            s.push_str(&format!(
                "  thread {:>2}{name}{kind} core {} prio {} irql {}: pc={:#010x} {state}\n",
                t.id, t.core, t.priority, t.irql, pc
            ));
        }
        s
    }

    fn deadlock_report(&self) -> String {
        let mut s = String::from("deadlock: no thread can ever run again\n");
        for t in &self.threads {
            let state = if t.terminated {
                "terminated".to_string()
            } else if t.suspend > 0 {
                format!("suspended x{}", t.suspend)
            } else {
                match &t.wait {
                    Wait::None => "ready".into(),
                    Wait::Handle { handle, wake_at } => {
                        format!("waiting on handle {handle:#x} {:?} (timeout {wake_at:?})", self.objects.get(*handle))
                    }
                    Wait::Dispatcher { addr, wake_at } => {
                        format!("waiting on dispatcher object at {addr:#x} (timeout {wake_at:?})")
                    }
                    Wait::Sleep { wake_at } => format!("sleeping until {wake_at}"),
                    Wait::CritSec { addr } => format!("waiting for critical section {addr:#x}"),
                    Wait::SpinLock { addr } => format!("spinning on lock {addr:#x}"),
                    Wait::Multi { targets, all, wake_at } => {
                        format!("waiting on {} of {targets:x?} (timeout {wake_at:?})", if *all { "all" } else { "any" })
                    }
                }
            };
            s.push_str(&format!("  thread {} (pc={:#x}): {state}\n", t.id, t.cpu.pc));
        }
        s
    }

    /// Wait on a dispatcher object at `addr`: satisfy now, poll-timeout, or block the thread.
    fn wait_dispatcher(&mut self, cpu: &mut Cpu, mem: &mut GuestMemory, addr: u32, timeout_ptr: u32) -> Next {
        match dispatcher::try_acquire(mem, addr) {
            Acquire::Satisfied => {
                cpu.gpr[3] = STATUS_SUCCESS as u64;
                Next::Continue
            }
            _ if timeout_ptr != 0 && mem.read64(timeout_ptr) == 0 => {
                cpu.gpr[3] = STATUS_TIMEOUT as u64; // poll
                Next::Continue
            }
            _ => {
                let wake_at = self.timeout_cycles(mem, timeout_ptr, cpu.cycles);
                self.threads[self.cur].wait = Wait::Dispatcher { addr, wake_at };
                Next::Reschedule
            }
        }
    }

    fn timeout_cycles(&self, mem: &GuestMemory, ptr: u32, now: u64) -> Option<u64> {
        if ptr == 0 {
            return None;
        }
        let v = mem.read64(ptr) as i64;
        // X360_WAIT_SCALE=N stretches every relative wait timeout (debug: is a driver watchdog firing because
        // our virtual GPU is slow?).
        static SCALE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let scale = *SCALE.get_or_init(|| std::env::var("X360_WAIT_SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1));
        Some(if v < 0 { now + (-v) as u64 * CYCLES_PER_TICK * scale } else { now + v as u64 * CYCLES_PER_TICK })
    }

    fn thread_exit(&mut self, mem: &mut GuestMemory, code: u32) -> Next {
        let cur = self.cur;
        self.threads[cur].terminated = true;
        self.threads[cur].exit_code = code;
        let h = self.threads[cur].handle;
        self.objects.set_thread_terminated(h);
        dispatcher::signal(mem, self.threads[cur].thread_obj); // pointer-based waiters see it too
        if self.threads[cur].recyclable {
            // Nobody can be waiting on a helper thread, so give everything back for the next one.
            let t = &self.threads[cur];
            let (stack, pcr, obj, tls) = (t.stack, t.pcr, t.thread_obj, t.tls_block);
            let tls_size = self.tls_info.map(|t| t.data_size.max(16) + t.slot_count * 4);
            let _ = self.vm.free(mem, stack.0, 0, crate::vm::MEM_RELEASE);
            mem.free_kernel(pcr, 0x1000);
            mem.free_kernel(obj, 0x1000);
            if let Some(sz) = tls_size {
                mem.free_kernel(tls, sz);
            }
            self.objects.close(h);
            self.free_slots.push(cur);
        }
        if self.threads[cur].id == 1 {
            Next::Exit(code)
        } else {
            Next::Reschedule
        }
    }

    /// Scheduler-aware calls. Returns Ok(None) if `name` isn't one of ours.
    pub fn sched_call(&mut self, name: &str, cpu: &mut Cpu, mem: &mut GuestMemory) -> Result<Option<Next>, String> {
        let cur = self.cur;
        let now = cpu.cycles;
        Ok(Some(match name {
            "<internal-thread-exit>" => {
                let code = cpu.gpr[3] as u32;
                self.thread_exit(mem, code)
            }
            "ExTerminateThread" | "NtTerminateThread" => {
                let code = cpu.gpr[4] as u32;
                self.thread_exit(mem, code)
            }
            "ExCreateThread" => {
                // (PHANDLE, stack size, PDWORD thread id, xapi startup, start address, context, flags)
                let (startup, start, ctx) = (cpu.gpr[6] as u32, cpu.gpr[7] as u32, cpu.gpr[8]);
                let (entry, a3, a4) = if startup != 0 { (startup, start as u64, ctx) } else { (start, ctx, 0) };
                let idx = self.spawn_thread(mem, entry, a3, a4, cpu.gpr[4] as u32, cpu.gpr[9] & 1 != 0)?;
                // Creation flags: bit 0 = start suspended; bits 24..29 = mask of hardware threads the
                // new thread may run on (0 = no preference). Single-core titles never notice, but code
                // that synchronises "per hardware thread" (the audio mixer's barrier) depends on it.
                let hw_mask = (cpu.gpr[9] as u32 >> 24) & 0x3F;
                if hw_mask != 0 {
                    self.threads[idx].affinity = hw_mask;
                    self.set_thread_core(mem, idx, hw_mask.trailing_zeros());
                }
                if self.trace_calls {
                    println!("[krnl] ExCreateThread -> thread {} entry {entry:#x} flags {:#x} stack {:#x}", self.threads[idx].id, cpu.gpr[9], cpu.gpr[4]);
                }
                if cpu.gpr[3] != 0 {
                    mem.write32(cpu.gpr[3] as u32, self.threads[idx].handle);
                }
                if cpu.gpr[5] != 0 {
                    mem.write32(cpu.gpr[5] as u32, self.threads[idx].id);
                }
                cpu.gpr[3] = STATUS_SUCCESS as u64;
                Next::Continue
            }
            "NtResumeThread" | "NtSuspendThread" => {
                // (handle, PULONG previous suspend count or null)
                let handle = cpu.gpr[3] as u32;
                match self.objects.thread_index(handle) {
                    Some(i) => {
                        let prev = self.threads[i].suspend;
                        if name == "NtResumeThread" {
                            self.threads[i].suspend = prev.saturating_sub(1);
                        } else {
                            self.threads[i].suspend = prev + 1;
                        }
                        if cpu.gpr[4] != 0 {
                            mem.write32(cpu.gpr[4] as u32, prev);
                        }
                        cpu.gpr[3] = STATUS_SUCCESS as u64;
                    }
                    None => cpu.gpr[3] = STATUS_INVALID_HANDLE as u64,
                }
                Next::Continue
            }
            // ---- interrupt request level and critical regions (per thread) ----
            "KeRaiseIrqlToDpcLevel" | "KfRaiseIrql" => {
                // returns the previous level; raising never lowers it
                let new = if name == "KfRaiseIrql" { cpu.gpr[3] as u8 } else { 2 };
                let t = &mut self.threads[cur];
                cpu.gpr[3] = t.irql as u64;
                t.irql = t.irql.max(new);
                Next::Continue
            }
            "KfLowerIrql" => {
                self.threads[cur].irql = cpu.gpr[3] as u8;
                Next::Continue
            }
            "KeGetCurrentIrql" => {
                cpu.gpr[3] = self.threads[cur].irql as u64;
                Next::Continue
            }
            "KeEnterCriticalRegion" => {
                self.threads[cur].critical_depth += 1;
                Next::Continue
            }
            "KeLeaveCriticalRegion" => {
                let t = &mut self.threads[cur];
                t.critical_depth = t.critical_depth.saturating_sub(1);
                Next::Continue
            }
            // ---- priority and affinity ----
            "KeSetBasePriorityThread" | "KeSetPriorityThread" | "KeQueryBasePriorityThread" => {
                // (thread object*, increment | priority) -> previous / current priority
                match self.thread_by_obj(cpu.gpr[3] as u32) {
                    Some(i) => {
                        let old = self.threads[i].priority;
                        let arg = cpu.gpr[4] as i32;
                        match name {
                            "KeSetBasePriorityThread" => self.threads[i].priority = arg.clamp(-15, 15),
                            "KeSetPriorityThread" => self.threads[i].priority = arg.clamp(-15, 15),
                            _ => {}
                        }
                        cpu.gpr[3] = old as i64 as u64;
                    }
                    None => cpu.gpr[3] = 0,
                }
                Next::Continue
            }
            "KeSetAffinityThread" => {
                // (thread object*, affinity mask) -> previous mask. The thread's reported core becomes
                // the lowest allowed one, which interrupt-handler bookkeeping relies on.
                match self.thread_by_obj(cpu.gpr[3] as u32) {
                    Some(i) => {
                        let old = self.threads[i].affinity;
                        let mask = cpu.gpr[4] as u32 & 0x3F;
                        if mask != 0 {
                            self.threads[i].affinity = mask;
                            self.set_thread_core(mem, i, mask.trailing_zeros());
                        }
                        cpu.gpr[3] = old as u64;
                    }
                    None => cpu.gpr[3] = 1,
                }
                Next::Continue
            }
            // ---- title termination ----
            "ExRegisterTitleTerminateNotification" => {
                // ({callback, priority}*, register?): remember callbacks to run when the title exits
                let st = cpu.gpr[3] as u32;
                if cpu.gpr[4] != 0 {
                    self.terminate_callbacks.push((mem.read32(st), mem.read32(st + 4) as i32, st));
                } else {
                    self.terminate_callbacks.retain(|c| c.2 != st);
                }
                Next::Continue
            }
            "HalReturnToFirmware" | "XamLoaderTerminateTitle" | "XamTerminateTitle" => Next::Exit(0),
            "KeAcquireSpinLockAtRaisedIrql" | "KfAcquireSpinLock" | "KeTryToAcquireSpinLockAtRaisedIrql" => {
                // (KSPIN_LOCK*): the lock word holds the owner thread id, 0 when free.
                let lock = cpu.gpr[3] as u32;
                let me = self.threads[cur].id;
                let owner = mem.read32(lock);
                if owner == 0 {
                    mem.write32(lock, me);
                    cpu.gpr[3] = if name == "KeTryToAcquireSpinLockAtRaisedIrql" { 1 } else { 0 }; // 0 = old IRQL
                    Next::Continue
                } else if name == "KeTryToAcquireSpinLockAtRaisedIrql" {
                    cpu.gpr[3] = 0;
                    Next::Continue
                } else {
                    cpu.gpr[3] = 0; // old IRQL result, delivered when the lock is finally taken
                    self.threads[cur].wait = Wait::SpinLock { addr: lock };
                    Next::Reschedule
                }
            }
            "KeReleaseSpinLockFromRaisedIrql" | "KfReleaseSpinLock" => {
                mem.write32(cpu.gpr[3] as u32, 0);
                Next::Continue
            }
            "KeResumeThread" | "KeSuspendThread" => {
                // (thread object*) -> previous suspend count
                let obj = cpu.gpr[3] as u32;
                match self.threads.iter().position(|t| t.thread_obj == obj) {
                    Some(i) => {
                        let prev = self.threads[i].suspend;
                        self.threads[i].suspend = if name == "KeResumeThread" { prev.saturating_sub(1) } else { prev + 1 };
                        cpu.gpr[3] = prev as u64;
                    }
                    None => cpu.gpr[3] = 0,
                }
                Next::Continue
            }
            "NtYieldExecution" => {
                cpu.gpr[3] = STATUS_SUCCESS as u64;
                Next::Reschedule
            }
            "KeDelayExecutionThread" => {
                // (mode, alertable, PLARGE_INTEGER interval)
                match self.timeout_cycles(mem, cpu.gpr[5] as u32, now) {
                    Some(wake_at) if wake_at > now => {
                        self.threads[cur].wait = Wait::Sleep { wake_at };
                    }
                    _ => cpu.gpr[3] = STATUS_SUCCESS as u64,
                }
                Next::Reschedule
            }
            "NtWaitForSingleObjectEx" => {
                // (handle, wait mode, alertable, PLARGE_INTEGER timeout or null)
                let h = cpu.gpr[3] as u32;
                let tp = cpu.gpr[6] as u32;
                match self.objects.dispatcher_addr(h) {
                    Some(addr) => self.wait_dispatcher(cpu, mem, addr, tp),
                    None => match self.objects.try_wait(h) {
                        WaitResult::Satisfied => {
                            cpu.gpr[3] = STATUS_SUCCESS as u64;
                            Next::Continue
                        }
                        WaitResult::InvalidHandle => {
                            cpu.gpr[3] = STATUS_INVALID_HANDLE as u64;
                            Next::Continue
                        }
                        WaitResult::NotSignaled => {
                            if tp != 0 && mem.read64(tp) == 0 {
                                cpu.gpr[3] = STATUS_TIMEOUT as u64; // poll
                                Next::Continue
                            } else {
                                let wake_at = self.timeout_cycles(mem, tp, now);
                                self.threads[cur].wait = Wait::Handle { handle: h, wake_at };
                                Next::Reschedule
                            }
                        }
                    },
                }
            }
            "KeWaitForMultipleObjects" | "NtWaitForMultipleObjectsEx" => {
                // Ke: (count, object*[], wait type, reason, mode, alertable, timeout*, wait block*)
                // Nt: (count, handle[], wait type, mode, alertable, timeout*)
                // wait type 0 = all, 1 = any
                let (count, list, all) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32 == 0);
                let (tp, kernel_form) = if name == "KeWaitForMultipleObjects" { (cpu.gpr[9] as u32, true) } else { (cpu.gpr[8] as u32, false) };
                let targets: Vec<WaitTarget> = (0..count.min(64))
                    .map(|i| {
                        let v = mem.read32(list + i * 4);
                        if kernel_form { WaitTarget::Object(v) } else { WaitTarget::Handle(v) }
                    })
                    .collect();
                match try_multi(&mut self.objects, mem, &targets, all, false) {
                    Some(status) => {
                        cpu.gpr[3] = status as u64;
                        Next::Continue
                    }
                    None if tp != 0 && mem.read64(tp) == 0 => {
                        cpu.gpr[3] = STATUS_TIMEOUT as u64; // poll
                        Next::Continue
                    }
                    None => {
                        let wake_at = self.timeout_cycles(mem, tp, now);
                        self.threads[cur].wait = Wait::Multi { targets, all, wake_at };
                        Next::Reschedule
                    }
                }
            }
            "KeWaitForSingleObject" => {
                // (object*, wait reason, wait mode, alertable, PLARGE_INTEGER timeout or null)
                let (addr, tp) = (cpu.gpr[3] as u32, cpu.gpr[7] as u32);
                self.wait_dispatcher(cpu, mem, addr, tp)
            }
            "RtlEnterCriticalSection" | "RtlTryEnterCriticalSection" => {
                // RTL_CRITICAL_SECTION: +0x10 lock_count (-1 free), +0x14 recursion, +0x18 owner thread id
                let cs = cpu.gpr[3] as u32;
                let me = self.threads[cur].id;
                let owner = mem.read32(cs + 0x18);
                if owner == 0 || owner == me {
                    let rec = if owner == me { mem.read32(cs + 0x14) } else { 0 } + 1;
                    mem.write32(cs + 0x18, me);
                    mem.write32(cs + 0x14, rec);
                    mem.write32(cs + 0x10, rec - 1);
                    cpu.gpr[3] = 1;
                    Next::Continue
                } else if name == "RtlTryEnterCriticalSection" {
                    cpu.gpr[3] = 0;
                    Next::Continue
                } else {
                    self.threads[cur].wait = Wait::CritSec { addr: cs };
                    Next::Reschedule
                }
            }
            "RtlLeaveCriticalSection" => {
                let cs = cpu.gpr[3] as u32;
                let rec = mem.read32(cs + 0x14).saturating_sub(1);
                mem.write32(cs + 0x14, rec);
                mem.write32(cs + 0x10, rec.wrapping_sub(1));
                if rec == 0 {
                    mem.write32(cs + 0x18, 0);
                }
                Next::Continue
            }
            _ => return Ok(None),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel_with_threads(priorities: &[i32]) -> (Kernel, GuestMemory) {
        let mut mem = GuestMemory::new();
        let mut k = Kernel::new();
        for &p in priorities {
            let i = k.spawn_thread(&mut mem, 0x8200_0000, 0, 0, 0x10000, false).unwrap();
            k.threads[i].priority = p;
        }
        (k, mem)
    }

    #[test]
    fn interrupt_threads_preempt_and_ordinary_priorities_round_robin() {
        let (mut k, mut mem) = kernel_with_threads(&[0, 0, 5, 0, 5]);
        k.cur = 0;
        // ordinary priorities don't starve anyone (one CPU stands in for six): plain round-robin
        assert_eq!(k.pick_next(&mut mem).unwrap(), 1);
        k.cur = 1;
        assert_eq!(k.pick_next(&mut mem).unwrap(), 2);
        // an interrupt handler jumps the queue
        k.threads[3].is_interrupt = true;
        k.cur = 0;
        assert_eq!(k.pick_next(&mut mem).unwrap(), 3);
    }

    #[test]
    fn multi_wait_any_and_all() {
        let mut mem = GuestMemory::new();
        mem.map(0x4010_0000, 0x1000);
        let mut objs = crate::objects::ObjectTable::new();
        let (a, b) = (0x4010_0000, 0x4010_0040);
        dispatcher::init_event(&mut mem, a, false, false); // auto-reset
        dispatcher::init_event(&mut mem, b, true, false); // manual-reset
        let t = [WaitTarget::Object(a), WaitTarget::Object(b)];
        assert_eq!(try_multi(&mut objs, &mut mem, &t, false, false), None);
        assert_eq!(try_multi(&mut objs, &mut mem, &t, false, true), Some(STATUS_TIMEOUT));
        dispatcher::set_event(&mut mem, b);
        assert_eq!(try_multi(&mut objs, &mut mem, &t, false, false), Some(1), "any: index of the signaled one");
        assert_eq!(try_multi(&mut objs, &mut mem, &t, true, false), None, "all: a isn't signaled yet");
        dispatcher::set_event(&mut mem, a);
        assert_eq!(try_multi(&mut objs, &mut mem, &t, true, false), Some(0));
        assert!(!dispatcher::peek_signaled(&mem, a), "all-wait consumed the auto-reset event");
        assert!(dispatcher::peek_signaled(&mem, b));
    }

    #[test]
    fn blocked_high_priority_thread_does_not_starve_others() {
        let (mut k, mut mem) = kernel_with_threads(&[0, 9]);
        k.threads[1].wait = Wait::SpinLock { addr: 0x4010_0000 };
        mem.map(0x4010_0000, 0x1000);
        mem.write32(0x4010_0000, 7); // lock held by someone else
        k.cur = 1;
        assert_eq!(k.pick_next(&mut mem).unwrap(), 0, "thread 1 is blocked, thread 0 runs");
        mem.write32(0x4010_0000, 0); // released: thread 1 is next in the rotation
        k.cur = 0;
        assert_eq!(k.pick_next(&mut mem).unwrap(), 1);
        assert_eq!(mem.read32(0x4010_0000), k.threads[1].id, "it took the lock when it woke");
    }

    #[test]
    fn terminate_notifications_run_in_priority_order() {
        let (mut k, mut mem) = kernel_with_threads(&[0]);
        k.terminate_callbacks = vec![(0x8200_1000, 1, 0), (0x8200_2000, 9, 0), (0x8200_3000, 5, 0)];
        assert_eq!(k.begin_shutdown(&mut mem), Ok(3));
        let entries: Vec<u64> = k.threads.iter().filter(|t| t.is_shutdown).map(|t| t.cpu.pc).collect();
        assert_eq!(entries, vec![0x8200_2000, 0x8200_3000, 0x8200_1000]);
        assert!(k.shutdown_threads_alive());
    }
}

fn copy_tls_template(mem: &mut GuestMemory, block: u32, t: &TlsInfo) {
    for i in 0..t.raw_data_size.min(t.data_size) {
        let b = mem.read8(t.raw_data_addr + i);
        mem.write8(block + i, b);
    }
}
