//! Kernel HLE. Every import thunk lands in `dispatch`; anything not implemented
//! returns an error naming the library and export, which is the crash-loop driver.

use serde::{Deserialize, Serialize};
use crate::export_table;
use crate::dispatcher;
use crate::fs::{Resolved, Vfs};
use crate::gpu::Gpu;
use crate::memory::GuestMemory;
use crate::objects::{KObject, ObjectTable, STATUS_INVALID_HANDLE, STATUS_OBJECT_NAME_NOT_FOUND, STATUS_SUCCESS};
use crate::sched::ThreadCtx;
use crate::vm::Vm;
use std::collections::HashMap;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

/// What the run loop should do after a kernel call.
#[derive(Debug, PartialEq, Eq)]
pub enum Next {
    /// Keep running the same thread.
    Continue,
    /// The current thread blocked, yielded, or exited: pick another.
    Reschedule,
    /// The main thread returned: the title is done.
    Exit(u32),
}

/// The executable's TLS template (XEX optional header 0x20104).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TlsInfo {
    pub slot_count: u32,
    pub raw_data_addr: u32,
    pub data_size: u32,
    pub raw_data_size: u32,
}

#[derive(Serialize, Deserialize)]
pub struct Kernel {
    /// Library names by import-library index, filled by the loader.
    pub libraries: Vec<String>,
    pub vm: Vm,
    pub objects: ObjectTable,
    /// Guest-visible object bodies handed out by ObReferenceObjectByHandle, by handle.
    object_bodies: HashMap<u32, u32>,
    /// Graphics interrupt callback + context registered by the title (0 = none).
    pub gfx_callback: (u32, u32),
    /// Virtual-clock time of the next vsync (0 until the first callback is registered), and the count fired.
    pub next_vsync: u64,
    pub vsyncs: u64,
    pub cp_interrupts: u64,
    /// Scheduler self-profile: pick_next calls and total host time spent in them.
    pub picks: u64,
    /// Bytes read from the disc and host storage so far (for the status line).
    pub io_bytes: u64,
    /// Debug: how many past-end-of-file read notes have been printed (not saved).
    #[serde(skip)]
    pub eof_notes: u32,
    /// Missing imports already reported under X360_STUB_MISSING.
    pub stubbed: std::collections::HashSet<String>,
    /// Thread-list slots whose (recyclable) thread has exited, available for the next spawn.
    pub free_slots: Vec<usize>,
    pub syscall_cycles: u64,
    pub pick_time: std::time::Duration,
    /// Title-termination notifications: (callback, priority, registration struct address).
    pub terminate_callbacks: Vec<(u32, i32, u32)>,
    pub gpu_identifier_addr: u32,
    sys_cmd_buffer: Option<u32>,
    /// Thread names set through the MS_VC_EXCEPTION convention, by thread id.
    pub thread_names: HashMap<u32, String>,
    pub gpu: Gpu,
    /// The mounted game disc, if we were started from an image.
    #[serde(skip)]
    pub vfs: Option<Vfs>,
    pub threads: Vec<ThreadCtx>,
    /// Index into `threads` of the thread currently executing.
    pub cur: usize,
    /// Virtual clock in executed instructions (synced from whichever CPU is running).
    pub global_cycles: u64,
    pub tls_info: Option<TlsInfo>,
    pub exit_thunk: u32,
    pub next_thread_id: u32,
    /// Print every kernel call (set X360_TRACE_KERNEL=1).
    pub trace_calls: bool,
    /// Module-info struct of the running executable.
    pub exec_module: u32,
    /// Lazily created fake module handles for system modules, by lowercase name.
    system_modules: HashMap<String, u32>,
    /// On-demand HLE thunks for dynamically resolved exports, by (library index, ordinal).
    dynamic_thunks: HashMap<(u32, u16), u32>,
    /// XEX system-flags optional header (key 0x30000): the executable's granted privileges.
    pub system_flags: u32,
    /// Id of the guest thread currently executing (single thread for now).
    pub current_thread: u32,
    tls_next: u32,
    tls: HashMap<(u32, u32), u32>,
    /// Profile, controller and notification state.
    pub xam: crate::xam::XamState,
    /// Video mode, scan-out switch, scaler and EDRAM calibration state.
    pub display: crate::display::DisplayState,
    /// Render-driver client, callback pacing and captured output.
    pub audio: crate::audio::AudioState,
}

impl Kernel {
    pub fn new() -> Self {
        Self {
            libraries: Vec::new(),
            vm: Vm::new(),
            objects: ObjectTable::new(),
            object_bodies: HashMap::new(),
            gfx_callback: (0, 0),
            next_vsync: 0,
            vsyncs: 0,
            cp_interrupts: 0,
            picks: 0,
            io_bytes: 0,
            eof_notes: 0,
            stubbed: Default::default(),
            free_slots: Vec::new(),
            syscall_cycles: std::env::var("X360_SYSCALL_CYCLES").ok().and_then(|v| v.parse().ok()).unwrap_or(1500),
            pick_time: std::time::Duration::ZERO,
            terminate_callbacks: Vec::new(),
            gpu_identifier_addr: 0,
            sys_cmd_buffer: None,
            thread_names: HashMap::new(),
            gpu: Gpu::new(),
            vfs: None,
            threads: Vec::new(),
            cur: 0,
            global_cycles: 0,
            tls_info: None,
            exit_thunk: 0,
            next_thread_id: 1,
            trace_calls: std::env::var_os("X360_TRACE_KERNEL").is_some(),
            exec_module: 0,
            system_modules: HashMap::new(),
            dynamic_thunks: HashMap::new(),
            system_flags: 0,
            current_thread: 0,
            tls_next: 0,
            tls: HashMap::new(),
            xam: crate::xam::XamState::new(),
            display: crate::display::DisplayState::new(),
            audio: Default::default(),
        }
    }

    /// Entry point for every `sc` from an import thunk. Scheduler-affecting calls are handled
    /// in `sched_call`; everything else is a plain function in `dispatch_simple`.
    pub fn dispatch(&mut self, cpu: &mut Cpu, mem: &mut GuestMemory) -> Result<Next, String> {
        let lib = (cpu.gpr[11] >> 16) as u32;
        let ordinal = (cpu.gpr[11] & 0xFFFF) as u16;
        let name = if lib == crate::sched::INTERNAL_LIB {
            "<internal-thread-exit>"
        } else {
            let lib_name = self.libraries.get(lib as usize).map(String::as_str).unwrap_or("?");
            export_table::lookup(lib_name, ordinal).map(|e| e.0).unwrap_or("<unknown>")
        };
        if self.trace_calls {
            println!(
                "[krnl t{}] {name} r3={:#x} r4={:#x} r5={:#x} r6={:#x} (lr={:#x})",
                self.current_thread, cpu.gpr[3], cpu.gpr[4], cpu.gpr[5], cpu.gpr[6], cpu.lr
            );
        }
        let next = match self.sched_call(name, cpu, mem)? {
            Some(next) => next,
            None => {
                self.dispatch_simple(cpu, mem, name, lib, ordinal)?;
                Next::Continue
            }
        };
        cpu.gpr[3] = sign_extend_result(cpu.gpr[3]);
        // A real kernel call costs on the order of a microsecond of CPU time. Charging it keeps polling
        // loops (waits with a zero timeout, TLS/critical-section churn) from needing absurd instruction
        // counts to let virtual time pass. Override with X360_SYSCALL_CYCLES.
        cpu.cycles += self.syscall_cycles;
        Ok(next)
    }

    fn dispatch_simple(&mut self, cpu: &mut Cpu, mem: &mut GuestMemory, name: &str, lib: u32, ordinal: u16) -> Result<(), String> {
        let lib_name = self.libraries.get(lib as usize).cloned().unwrap_or_else(|| "?".to_string());
        if self.exception_call(name, cpu, mem)? {
            return Ok(());
        }
        if self.xam_call(name, cpu, mem) || self.display_call(name, cpu, mem) || self.audio_call(name, cpu, mem) || self.crt_call(name, cpu, mem) {
            return Ok(());
        }
        match name {
            "ObReferenceObjectByHandle" => {
                // (handle, object type, PVOID *out) -> NTSTATUS. -2 is the current-thread pseudo-handle.
                let h = cpu.gpr[3] as u32;
                let body = if h == 0xFFFF_FFFE {
                    Some(self.threads[self.cur].thread_obj)
                } else if let Some(i) = self.objects.thread_index(h) {
                    Some(self.threads[i].thread_obj)
                } else if let Some(addr) = self.objects.dispatcher_addr(h) {
                    Some(addr) // the object body *is* the dispatcher header
                } else if self.objects.get(h).is_some() {
                    Some(*self.object_bodies.entry(h).or_insert_with(|| mem.alloc_kernel(0x1000)))
                } else {
                    None
                };
                cpu.gpr[3] = match body {
                    Some(b) => {
                        mem.write32(cpu.gpr[5] as u32, b);
                        STATUS_SUCCESS
                    }
                    None => STATUS_INVALID_HANDLE,
                } as u64;
                Ok(())
            }
            "ObDereferenceObject" => Ok(()),
            "VdInitializeRingBuffer" => {
                // (ring base physical address, size log2 in 8-byte units)
                self.gpu.init_ring(cpu.gpr[3] as u32, cpu.gpr[4] as u32);
                Ok(())
            }
            "VdEnableRingBufferRPtrWriteBack" => {
                // (physical address the CP mirrors its read pointer to, block size log2)
                self.gpu.rptr_writeback = cpu.gpr[3] as u32;
                self.gpu.rptr_writeback_block = cpu.gpr[4] as u32;
                Ok(())
            }
            "MmGetPhysicalAddress" => {
                // 0xA0000000 / 0xC0000000 / 0xE0000000 views all alias physical RAM at (addr & 0x1FFFFFFF).
                cpu.gpr[3] = (cpu.gpr[3] as u32 & 0x1FFF_FFFF) as u64;
                Ok(())
            }
            "VdSetSystemCommandBufferGpuIdentifierAddress" => {
                // Address the "GPU" writes a completion identifier to; recorded for when the CP exists.
                self.gpu_identifier_addr = cpu.gpr[3] as u32;
                Ok(())
            }
            // Ordinal 0x1DF (labelled KiApcNormalRoutineNop in public name lists). Observed behaviour:
            // called as f(buffer, 19) just before the buffer is submitted to the ring, and 19 dwords is
            // the size of a PM4 ME_INIT packet, so the kernel evidently builds that packet for the title.
            "KiApcNormalRoutineNop" if cpu.gpr[4] == 19 => {
                let buf = cpu.gpr[3] as u32;
                mem.write32(buf, 0xC011_4800); // type-3 packet, 18 payload dwords, opcode ME_INIT (0x48)
                mem.write32(buf + 4, 0x0000_03FF); // ME_INIT_ALL
                for i in 2..19 {
                    mem.write32(buf + i * 4, 0);
                }
                Ok(())
            }
            "ExGetXConfigSetting" => {
                // (category, setting, buffer, buffer size, PUSHORT returned size) -> NTSTATUS. All values are u32.
                let (cat, setting, buf, size, ret) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32, cpu.gpr[6] as u32, cpu.gpr[7] as u32);
                let value = match (cat, setting) {
                    (3, 0x09) => 1,          // user language: English
                    (3, 0x0A) => 0x0004_0000, // video flags: widescreen
                    (3, 0x0B) => 0x0001_0001, // audio flags: stereo + digital output
                    (3, 0x0C) => 0,          // retail flags
                    (3, 0x0E) => 103,        // country: United States
                    _ => {
                        println!("note: ExGetXConfigSetting(category {cat}, setting {setting:#x}) -> 0");
                        0
                    }
                };
                if buf != 0 && size >= 4 {
                    mem.write32(buf, value);
                }
                if ret != 0 {
                    mem.write16(ret, 4);
                }
                cpu.gpr[3] = STATUS_SUCCESS as u64;
                Ok(())
            }
            "RtlRaiseException" => {
                // EXCEPTION_RECORD*: code +0, flags +4, nested +8, address +0xC, nparams +0x10, params +0x14.
                let rec = cpu.gpr[3] as u32;
                let code = mem.read32(rec);
                if code == 0x406D_1388 {
                    // MS_VC_EXCEPTION "set thread name": params = {type 0x1000, name*, thread id, flags}.
                    // A debugger would swallow this; we just record the name.
                    let name_ptr = mem.read32(rec + 0x18);
                    let tname = read_cstring(mem, name_ptr);
                    if self.trace_calls {
                        println!("note: thread {} named {tname:?}", self.current_thread);
                    }
                    self.thread_names.insert(self.current_thread, tname);
                    Ok(())
                } else {
                    Err(format!(
                        "game raised exception {code:#010x} at {:#010x} ({} params): unwinding is not implemented",
                        mem.read32(rec + 0xC),
                        mem.read32(rec + 0x10)
                    ))
                }
            }
            "MmFreePhysicalMemory" => {
                // (type, base address): return the block to the physical heap's free list.
                let base = cpu.gpr[4] as u32;
                if base != 0 && !self.vm.free_physical(base) {
                    println!("note: MmFreePhysicalMemory({base:#x}) is not a live allocation");
                }
                Ok(())
            }
            "NtFreeVirtualMemory" => {
                // (PVOID *base, SIZE_T *size, free type) -> NTSTATUS; writes back the range affected.
                let (base_ptr, size_ptr) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32);
                let (base, size) = (mem.read32(base_ptr), mem.read32(size_ptr));
                cpu.gpr[3] = match self.vm.free(mem, base, size, cpu.gpr[5] as u32) {
                    Ok((b, s)) => {
                        mem.write32(base_ptr, b);
                        mem.write32(size_ptr, s);
                        STATUS_SUCCESS
                    }
                    Err(s) => s,
                } as u64;
                Ok(())
            }
            "VdSwap" => {
                // (command buffer to write into, front buffer texture fetch constant*, ...)
                // Emits a 64-dword XE_SWAP packet: header, 'SWAP', then the 6 fetch-constant dwords.
                // The CP finds the front buffer address / size / format from those.
                let (buf, fetch) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32);
                mem.write32(buf, 0xC03E_6400); // type 3, 63 body dwords, opcode XE_SWAP (0x64)
                mem.write32(buf + 4, 0x5041_5753); // 'SWAP'
                for i in 0..6 {
                    mem.write32(buf + 8 + i * 4, mem.read32(fetch + i * 4));
                }
                for i in 7..64 {
                    mem.write32(buf + i * 4, 0);
                }
                // The 4th argument (r6) points at a 148-byte struct the kernel fills in with pending
                // display changes (flags at +8 and +0x1C, a 56-byte rectangle at +0x20). The title
                // zeroes it itself when it skips the call, and acts on any non-zero flag, so "nothing
                // pending" must be written: leaving stale stack there made it chase garbage pointers.
                let pending = cpu.gpr[6] as u32;
                if pending != 0 && mem.is_mapped(pending) && mem.is_mapped(pending + 147) {
                    for i in 0..37 {
                        mem.write32(pending + i * 4, 0);
                    }
                }
                // Output parameters, all derived from the front buffer's texture fetch constant:
                // (r8) front buffer address, (r9) texture format, (r10) color space, then the two
                // stack arguments: width and height. The title reads them back, so leaving them
                // unwritten feeds it stale stack garbage (a zero-size rect -> divide by zero).
                let (w1, w2) = (mem.read32(fetch + 4), mem.read32(fetch + 8));
                let sp = cpu.gpr[1] as u32;
                let outs = [
                    (cpu.gpr[8] as u32, w1 & 0xFFFF_F000),
                    (cpu.gpr[9] as u32, w1 & 0x3F),
                    (cpu.gpr[10] as u32, 0),
                    (mem.read32(sp + 0x54), (w2 & 0x1FFF) + 1),
                    (mem.read32(sp + 0x5C), ((w2 >> 13) & 0x1FFF) + 1),
                ];
                for (ptr, val) in outs {
                    if ptr != 0 && mem.is_mapped(ptr) {
                        mem.write32(ptr, val);
                    }
                }
                Ok(())
            }
            "VdGetSystemCommandBuffer" => {
                // (u32 *buffer out, u32 *identifier out): scratch space for kernel-issued GPU commands.
                let buf = *self.sys_cmd_buffer.get_or_insert_with(|| mem.alloc_kernel(0x1000));
                mem.write32(cpu.gpr[3] as u32, buf);
                mem.write32(cpu.gpr[4] as u32, 1);
                Ok(())
            }
            "VdSetGraphicsInterruptCallback" => {
                // (callback, context): called by the "GPU" on vsync / command-processor interrupts.
                self.gfx_callback = (cpu.gpr[3] as u32, cpu.gpr[4] as u32);
                Ok(())
            }
            "VdInitializeEngines" => {
                // (callbacks + context for the graphics engines). Nothing to initialise in HLE yet.
                cpu.gpr[3] = 1;
                Ok(())
            }
            // Would invoke notification callbacks registered via VdRegisterGraphicsNotification;
            // nothing has registered one yet, so there is nobody to call.
            "VdCallGraphicsNotificationRoutines" => Ok(()),
            // EDRAM timing retraining is a hardware calibration step; nothing to do in emulation.
            "XNotifyPositionUI" => Ok(()), // where popup notifications appear on screen; no UI yet
            "MmQueryAddressProtect" => {
                // (address) -> PAGE_* flags; everything we map is read/write (PAGE_READWRITE = 4)
                cpu.gpr[3] = if mem.is_mapped(cpu.gpr[3] as u32) { 4 } else { 0 };
                Ok(())
            }
            "MmQueryAllocationSize" => {
                // (base address) -> size of the allocation containing it, 0 if none
                cpu.gpr[3] = self.vm.region_of(cpu.gpr[3] as u32).map(|(_, size)| size).unwrap_or(0) as u64;
                Ok(())
            }
            "RtlFillMemoryUlong" => {
                // (dest, length in bytes, 32-bit pattern)
                let (dest, len, pat) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32);
                for i in 0..len / 4 {
                    mem.write32(dest + i * 4, pat);
                }
                Ok(())
            }
            "RtlFillMemory" | "memset" | "RtlZeroMemory" => {
                // RtlFillMemory(dest, len, fill) / memset(dest, fill, len) / RtlZeroMemory(dest, len)
                let dest = cpu.gpr[3] as u32;
                let (len, fill) = match name {
                    "memset" => (cpu.gpr[5] as u32, cpu.gpr[4] as u8),
                    "RtlZeroMemory" => (cpu.gpr[4] as u32, 0),
                    _ => (cpu.gpr[4] as u32, cpu.gpr[5] as u8),
                };
                for i in 0..len {
                    mem.write8(dest + i, fill);
                }
                Ok(())
            }
            "RtlMoveMemory" | "RtlCopyMemory" | "memcpy" | "memmove" => {
                // (dest, src, length). Copy via a temporary so overlapping ranges behave like memmove.
                let (dest, src, len) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32);
                let tmp: Vec<u8> = (0..len).map(|i| mem.read8(src + i)).collect();
                mem.write_bytes(dest, &tmp);
                if name == "memcpy" || name == "memmove" {
                    // C functions return the destination pointer; the Rtl ones are void.
                    cpu.gpr[3] = dest as u64;
                }
                Ok(())
            }
            "RtlCompareMemory" => {
                // (a, b, length) -> number of leading bytes that match
                let (a, b, len) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32);
                let n = (0..len).take_while(|&i| mem.read8(a + i) == mem.read8(b + i)).count();
                cpu.gpr[3] = n as u64;
                Ok(())
            }
            "memcmp" => {
                let (a, b, len) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32);
                let mut r = 0i64;
                for i in 0..len {
                    let (x, y) = (mem.read8(a + i), mem.read8(b + i));
                    if x != y {
                        r = x as i64 - y as i64;
                        break;
                    }
                }
                cpu.gpr[3] = r as u64;
                Ok(())
            }
            "VdIsHSIOTrainingSucceeded" => {
                cpu.gpr[3] = 1; // the CPU<->GPU link trained fine
                Ok(())
            }
            "KeQueryPerformanceFrequency" => {
                cpu.gpr[3] = 49_875_000; // same 49.875 MHz as the time base
                Ok(())
            }
            "VdGetCurrentDisplayGamma" => {
                // (u32 *type, f32 *power): gamma type 2 (PWL), power ~2.22
                mem.write32(cpu.gpr[3] as u32, 2);
                mem.write32(cpu.gpr[4] as u32, 2.2222223f32.to_bits());
                Ok(())
            }
            "RtlNtStatusToDosError" => {
                cpu.gpr[3] = nt_status_to_dos_error(cpu.gpr[3] as u32) as u64;
                Ok(())
            }
            "NtQueryFullAttributesFile" => {
                // (OBJECT_ATTRIBUTES *, FILE_NETWORK_OPEN_INFORMATION *out) -> NTSTATUS
                let path = read_object_name(mem, cpu.gpr[3] as u32);
                let found = match self.vfs.as_ref().map(|v| (v.resolve(&path), v)) {
                    Some((Resolved::Disc(rel), v)) => v.stat(&rel),
                    _ => None,
                };
                if self.trace_calls || found.is_none() {
                    println!("note: NtQueryFullAttributesFile({path:?}) -> {}", if found.is_some() { "found" } else { "not found" });
                }
                cpu.gpr[3] = match found {
                    Some(e) => {
                        let out = cpu.gpr[4] as u32;
                        for off in (0..56).step_by(4) {
                            mem.write32(out + off, 0);
                        }
                        let alloc = (e.size as u64 + 2047) & !2047;
                        mem.write64(out + 32, if e.is_dir { 0 } else { alloc });
                        mem.write64(out + 40, if e.is_dir { 0 } else { e.size as u64 });
                        mem.write32(out + 48, if e.is_dir { 0x10 } else { 0x81 }); // DIRECTORY / READONLY|NORMAL
                        STATUS_SUCCESS
                    }
                    None => STATUS_OBJECT_NAME_NOT_FOUND,
                } as u64;
                Ok(())
            }
            // ANSI_STRING / UNICODE_STRING: u16 Length (bytes, no NUL), u16 MaximumLength, u32 Buffer.
            "RtlInitAnsiString" | "RtlInitUnicodeString" => {
                let (dest, src) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32);
                let wide = name == "RtlInitUnicodeString";
                let mut bytes = 0u32;
                if src != 0 {
                    while bytes < 0xFFFE {
                        let unit = if wide { mem.read16(src + bytes) } else { mem.read8(src + bytes) as u16 };
                        if unit == 0 {
                            break;
                        }
                        bytes += if wide { 2 } else { 1 };
                    }
                }
                mem.write16(dest, bytes as u16);
                mem.write16(dest + 2, if src != 0 { bytes as u16 + if wide { 2 } else { 1 } } else { 0 });
                mem.write32(dest + 4, src);
                Ok(())
            }
            // Priority boosting is a scheduler heuristic we don't model; the call itself is a no-op.
            "KeSetDisableBoostThread" => {
                cpu.gpr[3] = 0;
                Ok(())
            }
            "NetDll_WSAStartupEx" | "NetDll_WSAStartup" => {
                // (caller, requested version, WSADATA *out[, flags]) -> 0. wVersion/wHighVersion only.
                let out = cpu.gpr[5] as u32;
                if out != 0 {
                    mem.write16(out, 0x0202);
                    mem.write16(out + 2, 0x0202);
                }
                cpu.gpr[3] = 0;
                Ok(())
            }
            "NetDll_XNetGetTitleXnAddr" => {
                // (caller, XNADDR *out) -> XNET_GET_XNADDR_* flags. 0x01 = NONE: no address assigned.
                let out = cpu.gpr[4] as u32;
                for off in (0..36).step_by(4) {
                    mem.write32(out + off, 0);
                }
                cpu.gpr[3] = 1;
                Ok(())
            }
            "NetDll_XNetGetEthernetLinkStatus" => {
                cpu.gpr[3] = 0; // no link: offline mode
                Ok(())
            }
            "NetDll_WSACleanup" | "NetDll_XNetCleanup" => {
                cpu.gpr[3] = 0;
                Ok(())
            }
            "NetDll_XNetStartupEx" | "NetDll_XNetStartup" => {
                cpu.gpr[3] = 0; // networking is not emulated; report success
                Ok(())
            }
            "RtlImageXexHeaderField" => {
                cpu.gpr[3] = rtl_image_xex_header_field(mem, cpu.gpr[3] as u32, cpu.gpr[4] as u32) as u64;
                Ok(())
            }
            // RTL_CRITICAL_SECTION: 16-byte dispatcher header, then lock_count (+0x10, -1 = free),
            // recursion_count (+0x14), owning_thread (+0x18). Single-threaded for now, so
            // enter/leave only keep the bookkeeping fields consistent.
            "RtlInitializeCriticalSection" | "RtlInitializeCriticalSectionAndSpinCount" => {
                let cs = cpu.gpr[3] as u32;
                for off in (0..28).step_by(4) {
                    mem.write32(cs + off, 0);
                }
                mem.write32(cs, 1); // header type: synchronization event
                mem.write32(cs + 0x10, (-1i32) as u32);
                cpu.gpr[3] = STATUS_SUCCESS as u64;
                Ok(())
            }
            "KeTlsAlloc" => {
                cpu.gpr[3] = self.tls_next as u64;
                self.tls_next += 1;
                Ok(())
            }
            "KeTlsFree" => {
                let idx = cpu.gpr[3] as u32;
                self.tls.retain(|&(_, i), _| i != idx);
                cpu.gpr[3] = 1;
                Ok(())
            }
            "KeTlsGetValue" => {
                cpu.gpr[3] = self.tls.get(&(self.current_thread, cpu.gpr[3] as u32)).copied().unwrap_or(0) as u64;
                Ok(())
            }
            "KeTlsSetValue" => {
                self.tls.insert((self.current_thread, cpu.gpr[3] as u32), cpu.gpr[4] as u32);
                cpu.gpr[3] = 1;
                Ok(())
            }
            "XexCheckExecutablePrivilege" => {
                let bit = cpu.gpr[3] as u32;
                cpu.gpr[3] = if bit < 32 { ((self.system_flags >> bit) & 1) as u64 } else { 0 };
                Ok(())
            }
            "XexGetProcedureAddress" => {
                // (module handle, ordinal or name pointer, PDWORD out) -> NTSTATUS
                let handle = cpu.gpr[3] as u32;
                let key = cpu.gpr[4] as u32;
                let lib_name = self.system_modules.iter().find(|(_, h)| **h == handle).map(|(n, _)| n.clone());
                cpu.gpr[3] = match (lib_name, key < 0x10000) {
                    (Some(lib), true) => {
                        let lib_idx = match self.libraries.iter().position(|l| l.eq_ignore_ascii_case(&lib)) {
                            Some(i) => i as u32,
                            None => {
                                self.libraries.push(lib.clone());
                                self.libraries.len() as u32 - 1
                            }
                        };
                        let is_var = export_table::lookup(&lib, key as u16).map(|e| e.1).unwrap_or(false);
                        let addr = if is_var {
                            mem.alloc_kernel(0x1000) // storage for the exported variable
                        } else {
                            *self.dynamic_thunks.entry((lib_idx, key as u16)).or_insert_with(|| {
                                let t = mem.alloc_kernel(16);
                                mem.write32(t, 0x3D60_0000 | lib_idx);
                                mem.write32(t + 4, 0x616B_0000 | key);
                                mem.write32(t + 8, 0x4400_0002);
                                mem.write32(t + 12, 0x4E80_0020);
                                t
                            })
                        };
                        mem.write32(cpu.gpr[5] as u32, addr);
                        STATUS_SUCCESS
                    }
                    _ => {
                        println!("note: XexGetProcedureAddress(handle={handle:#x}, key={key:#x}) unresolved");
                        0xC000_0225
                    }
                } as u64;
                Ok(())
            }
            "XexGetModuleHandle" => {
                // (name or null, PHANDLE out) -> NTSTATUS. Null name = the running executable.
                let name_ptr = cpu.gpr[3] as u32;
                let want = if name_ptr == 0 { None } else { Some(read_cstring(mem, name_ptr).to_lowercase()) };
                let handle = match want.as_deref() {
                    None => Some(self.exec_module),
                    Some("xboxkrnl.exe") | Some("xam.xex") | Some("xbdm.xex") => {
                        let key = want.clone().unwrap();
                        Some(match self.system_modules.get(&key) {
                            Some(&h) => h,
                            None => {
                                let h = mem.alloc_kernel(0x1000);
                                self.system_modules.insert(key, h);
                                h
                            }
                        })
                    }
                    Some(_) => None,
                };
                cpu.gpr[3] = match handle {
                    Some(h) => {
                        mem.write32(cpu.gpr[4] as u32, h);
                        STATUS_SUCCESS
                    }
                    None => {
                        println!("note: XexGetModuleHandle({:?}) not found", want);
                        0xC000_0225 // STATUS_NOT_FOUND
                    }
                } as u64;
                Ok(())
            }
            "XamGetSystemVersion" => {
                // major<<28 | minor<<24 | build<<8 | qfe : pretend 2.0.17559.0
                cpu.gpr[3] = ((2u32 << 28) | (17559 << 8)) as u64;
                Ok(())
            }
            "XamNotifyCreateListener" => {
                // (mask, max version) -> handle. No notifications are ever queued yet.
                cpu.gpr[3] = self.objects.create(KObject::Listener { mask: cpu.gpr[3], queue: Default::default() }) as u64;
                Ok(())
            }
            "NtCreateSemaphore" => {
                // (PHANDLE out, attrs, initial count, limit)
                let addr = mem.alloc_kernel(dispatcher::OBJECT_SIZE);
                dispatcher::init_semaphore(mem, addr, cpu.gpr[5] as i32, cpu.gpr[6] as i32);
                let h = self.objects.create(KObject::Dispatcher { addr });
                if self.trace_calls {
                    println!("[obj] semaphore {h:#x} at {addr:#x} count {} limit {} (lr={:#x})", cpu.gpr[5], cpu.gpr[6], cpu.lr);
                }
                mem.write32(cpu.gpr[3] as u32, h);
                cpu.gpr[3] = STATUS_SUCCESS as u64;
                Ok(())
            }
            "NtCreateEvent" => {
                // (PHANDLE out, attrs, event type [0 = notification/manual, 1 = synchronization/auto], initial state)
                let addr = mem.alloc_kernel(dispatcher::OBJECT_SIZE);
                dispatcher::init_event(mem, addr, cpu.gpr[5] == 0, cpu.gpr[6] != 0);
                let h = self.objects.create(KObject::Dispatcher { addr });
                if self.trace_calls {
                    println!("[obj] event {h:#x} at {addr:#x} type {} init {} (lr={:#x})", cpu.gpr[5], cpu.gpr[6], cpu.lr);
                }
                mem.write32(cpu.gpr[3] as u32, h);
                cpu.gpr[3] = STATUS_SUCCESS as u64;
                Ok(())
            }
            "NtSetEvent" | "NtResetEvent" => {
                // (handle, PLONG previous state or null)
                let set = name == "NtSetEvent";
                cpu.gpr[3] = match self.objects.dispatcher_addr(cpu.gpr[3] as u32) {
                    Some(addr) => {
                        let prev = if set { dispatcher::set_event(mem, addr) } else { dispatcher::reset_event(mem, addr) };
                        if cpu.gpr[4] != 0 {
                            mem.write32(cpu.gpr[4] as u32, prev as u32);
                        }
                        STATUS_SUCCESS
                    }
                    None => STATUS_INVALID_HANDLE,
                } as u64;
                Ok(())
            }
            "NtReleaseSemaphore" => {
                // (handle, release count, PLONG previous count or null)
                cpu.gpr[3] = match self.objects.dispatcher_addr(cpu.gpr[3] as u32) {
                    Some(addr) => match dispatcher::release_semaphore(mem, addr, cpu.gpr[4] as i32) {
                        Ok(prev) => {
                            if cpu.gpr[5] != 0 {
                                mem.write32(cpu.gpr[5] as u32, prev as u32);
                            }
                            STATUS_SUCCESS
                        }
                        Err(s) => s,
                    },
                    None => STATUS_INVALID_HANDLE,
                } as u64;
                Ok(())
            }
            // Pointer-based (Ke*) variants operate directly on a dispatcher object in guest memory.
            "KeInitializeEvent" => {
                // (event*, type [0 notification, 1 synchronization], initial state)
                dispatcher::init_event(mem, cpu.gpr[3] as u32, cpu.gpr[4] == 0, cpu.gpr[5] != 0);
                Ok(())
            }
            "KeSetEvent" | "KeResetEvent" => {
                // (event*, increment, wait) -> previous state
                let addr = cpu.gpr[3] as u32;
                let prev = if name == "KeSetEvent" { dispatcher::set_event(mem, addr) } else { dispatcher::reset_event(mem, addr) };
                cpu.gpr[3] = prev as u32 as u64;
                Ok(())
            }
            "KeInitializeSemaphore" => {
                // (semaphore*, count, limit)
                dispatcher::init_semaphore(mem, cpu.gpr[3] as u32, cpu.gpr[4] as i32, cpu.gpr[5] as i32);
                Ok(())
            }
            "KeReleaseSemaphore" => {
                // (semaphore*, increment, adjustment, wait) -> previous count
                cpu.gpr[3] = dispatcher::release_semaphore(mem, cpu.gpr[3] as u32, cpu.gpr[5] as i32).unwrap_or(0) as u32 as u64;
                Ok(())
            }
            "NtClose" => {
                cpu.gpr[3] = self.objects.close(cpu.gpr[3] as u32) as u64;
                Ok(())
            }
            "MmAllocatePhysicalMemoryEx" => {
                // (type, size, protect, min_addr, max_addr, alignment) -> base address, 0 on failure
                let (size, align) = (cpu.gpr[4] as u32, cpu.gpr[8] as u32);
                cpu.gpr[3] = match self.vm.allocate_physical(mem, size, align) {
                    Ok(a) => a as u64,
                    Err(s) => {
                        println!("note: MmAllocatePhysicalMemoryEx({size:#x}, align {align:#x}) FAILED ({s:#x}); {}", self.vm.phys_usage());
                        0
                    }
                };
                Ok(())
            }
            "KeGetCurrentProcessType" => {
                cpu.gpr[3] = 1; // X_PROCTYPE_USER
                Ok(())
            }
            "NtAllocateVirtualMemory" => {
                // (PVOID *base, SIZE_T *size, ULONG type, ULONG protect)
                let (base_ptr, size_ptr) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32);
                let status = match self.vm.allocate(mem, mem.read32(base_ptr), mem.read32(size_ptr), cpu.gpr[5] as u32) {
                    Ok((base, size)) => {
                        mem.write32(base_ptr, base);
                        mem.write32(size_ptr, size);
                        STATUS_SUCCESS
                    }
                    Err(s) => s,
                };
                cpu.gpr[3] = status as u64;
                Ok(())
            }
            _ => {
                if self.host_io_call(name, cpu, mem)? || self.io_call(name, cpu, mem)? {
                    return Ok(());
                }
                // X360_STUB_MISSING=1: log each missing import once and answer "success" (0) so a single run
                // can reveal many of them; the run may go wrong later because of it - implement them for real.
                if std::env::var_os("X360_STUB_MISSING").is_some() {
                    if self.stubbed.insert(name.to_string()) {
                        println!(
                            "[stub] {lib_name}!{name} (ordinal {ordinal:#06x}) args r3={:#x} r4={:#x} r5={:#x} r6={:#x} r7={:#x} r8={:#x} lr={:#x}",
                            cpu.gpr[3], cpu.gpr[4], cpu.gpr[5], cpu.gpr[6], cpu.gpr[7], cpu.gpr[8], cpu.lr
                        );
                    }
                    cpu.gpr[3] = 0;
                    return Ok(());
                }
                Err(format!(
                    "unimplemented kernel import: {lib_name}!{name} (ordinal {ordinal:#06x}), args r3={:#x} r4={:#x} r5={:#x} r6={:#x}",
                    cpu.gpr[3], cpu.gpr[4], cpu.gpr[5], cpu.gpr[6]
                ))
            }
        }
    }
}

/// A 32-bit return value (NTSTATUS, pointer, ...) lives sign-extended in the 64-bit r3, exactly as
/// compiled code (`lis`/`li`) would leave it. Game code that records CR0 from the full register
/// (`or. r31,r3`) relies on negative statuses looking negative.
pub fn sign_extend_result(r3: u64) -> u64 {
    if r3 >> 32 == 0 && r3 & 0x8000_0000 != 0 {
        r3 | 0xFFFF_FFFF_0000_0000
    } else {
        r3
    }
}

/// NTSTATUS -> Win32 error code for the statuses we generate; unknown errors map to 317
/// (ERROR_MR_MID_NOT_FOUND, what Windows returns for unmapped statuses).
fn nt_status_to_dos_error(status: u32) -> u32 {
    match status {
        0 => 0,
        0x0000_0102 => 258,                      // STATUS_TIMEOUT -> WAIT_TIMEOUT
        0x8000_0006 => 18,                       // NO_MORE_FILES
        0xC000_0002 => 1,                        // NOT_IMPLEMENTED -> INVALID_FUNCTION
        0xC000_0003 | 0xC000_000D => 87,         // INVALID_INFO_CLASS / INVALID_PARAMETER
        0xC000_0008 => 6,                        // INVALID_HANDLE
        0xC000_000E => 2,                        // NO_SUCH_DEVICE -> FILE_NOT_FOUND
        0xC000_000F | 0xC000_0034 => 2,          // NO_SUCH_FILE / OBJECT_NAME_NOT_FOUND
        0xC000_0011 => 38,                       // END_OF_FILE -> HANDLE_EOF
        0xC000_0017 => 8,                        // NO_MEMORY -> NOT_ENOUGH_MEMORY
        0xC000_0022 => 5,                        // ACCESS_DENIED
        0xC000_0225 => 1168,                     // NOT_FOUND -> ERROR_NOT_FOUND
        _ => 317,
    }
}

/// Path from an OBJECT_ATTRIBUTES: {root dir u32, PANSI_STRING name u32, attributes u32}.
fn read_object_name(mem: &GuestMemory, attrs: u32) -> String {
    let name = mem.read32(attrs + 4);
    if name == 0 {
        return String::new();
    }
    let (len, buf) = (mem.read16(name) as u32, mem.read32(name + 4));
    (0..len).map(|i| mem.read8(buf + i) as char).collect()
}

fn read_cstring(mem: &GuestMemory, addr: u32) -> String {
    let mut s = String::new();
    for i in 0..256 {
        let b = mem.read8(addr + i);
        if b == 0 {
            break;
        }
        s.push(b as char);
    }
    s
}

/// Look `key` up in the XEX optional-header table at `header` (big-endian: count @+20,
/// {id,value} pairs from +24). Keys whose low byte is 0 or 1 carry their value inline;
/// everything else returns a guest pointer to the data. Missing keys return 0.
fn rtl_image_xex_header_field(mem: &GuestMemory, header: u32, key: u32) -> u32 {
    let count = mem.read32(header + 20);
    for i in 0..count {
        let e = header + 24 + i * 8;
        if mem.read32(e) == key {
            let v = mem.read32(e + 4);
            return if key & 0xFF <= 1 { v } else { header.wrapping_add(v) };
        }
    }
    0
}
