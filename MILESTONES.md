# Xbox 360 Emulator — Milestones

Crates: `xenon-cpu` (PowerPC interpreter), `x360` (disc/XEX loader, memory, kernel HLE, scheduler, VFS).
Approach: **crash-driven**. Run GTA IV, the emulator halts with a report on the first thing it
can't do (unknown opcode / unmapped access / unimplemented kernel import), we implement it + a
unit test, rerun. Log each fix below.

Run: `cargo run --release -p x360 -- <game.iso> [max_instructions]`
Debug env vars: `X360_TRACE_KERNEL=1` (log every kernel call), `X360_DUMP="0xADDR,N"` (dump N guest words).

## Infrastructure (done)
- [x] Big-endian memory bus, paged guest memory with unmapped-access faults
- [x] GDF/XISO disc reader (all 4 partition offsets), runtime VFS (`game:`, `d:`, `\Device\Cdrom0`)
- [x] XEX2: parser, AES decrypt, basic + LZX decompression, import-thunk patching
- [x] Interpreter with 64-entry trace + crash reports; deterministic time base (cycles/64)
- [x] Scheduler: per-thread CPU/PCR/TLS/stack, 20k-instruction slices, blocking waits, virtual clock, deadlock report

## CPU coverage
- [x] Integer ALU (add/sub/carry, mul/div, logic, shifts, rotates incl. 64-bit), compares, CR ops
- [x] Loads/stores (D/DS/X forms, update, byte-reversed, lmw/stmw), lwarx/stwcx, dcbz/dcbz128
- [x] Branches, traps (tw/td), sc
- [x] FPU: loads/stores, single+double arith, FMA, converts, fcmp, FPSCR moves (rounding/exceptions not modelled)
- [x] VMX: logic, splats, add/sub, shifts, rotates, compares, min/max, merge/pack/unpack, vperm/vsel/vsldoi, float basics
- [x] VMX128: lvx128/stvx128 family (incl. lvlx/stvrx), float arithmetic + logic (opcode 5)
- [ ] VMX128 opcode 6 (vperm128, vpermwi128, vrlimi128, vcmp*128, ...), remaining VMX float/pack ops, saturating pack/add

## Crash log (crash -> fix)
| # | Instr. executed | Crash | Fix |
|---|---|---|---|
| 1 | 0 | entry code read garbage | loader copied PE by section; XEX payload is already the in-memory image |
| 2 | 3 | `std` (opcode 62) | full D/DS/X load-store family |
| 3 | 32 | read from variable-import placeholder | back variable imports with kernel memory; `XexExecutableModuleHandle` |
| 4 | 41 | `RtlImageXexHeaderField` | first kernel call; copy XEX header into guest memory |
| 5 | 141 | `cmpw` | whole integer ALU |
| 6 | 271 | `NtAllocateVirtualMemory` | VM manager (reserve/commit, 64KB aligned) |
| 7 | 380 | `KeGetCurrentProcessType`, `RtlInitializeCriticalSection`, `XexCheckExecutablePrivilege`, `KeTlsAlloc` | small kernel calls |
| 8 | 2479 | `twi` | trap family |
| 9 | 2868 | read at `0x100` (r13) | PCR + KTHREAD for the main thread |
| 10 | 9756 | `vspltisb` | VMX foundation; then lvsl/lvsr, shifts, `stvlx` (+ my own xo-number bug for `stvrx`), VMX128 `stvx128`, `dcbz128` |
| 11 | 9935 | `MmAllocatePhysicalMemoryEx` | physical heap; game then `memset`s ~285MB (~50M instructions) |
| 12 | 54.5M | write via TLS pointer (PCR+0) | static TLS block from XEX header 0x20104 |
| 13 | 54.6M | `NtCreateSemaphore`, `mftb`, `lfs` | object table + semaphores/events; time base; FPU |
| 14 | 56.9M | `vmulfp128` (opcode 5) | VMX128 arithmetic with scattered register fields |
| 15 | 57.0M | XAM: `XamNotifyCreateListener`, `XamGetSystemVersion`, `XexGetModuleHandle`, `XexGetProcedureAddress` -> `NetDll_*` | listeners, module registry, on-demand HLE thunks, network stubs |
| 16 | 57.0M | `ExCreateThread` | **scheduler** (see above) |
| 17 | 57.0M | `RtlInitAnsiString`, `NtQueryFullAttributesFile`, `NtCreateFile` | VFS + file I/O (create/open/read/query/set-info) |
| 18 | 136M | `NtQueryDirectoryFile` (wrong signature at first), `RtlNtStatusToDosError`, `XGetVideoMode` | directory enumeration, status table, 720p video mode |
| 19 | **139.9M** | `VdInitializeEngines` | **<- here: GPU bring-up begins** |

## Where we are (checkpoint)
- ~700M guest instructions to the first presented frame; 15 `XE_SWAP`s (triple-buffered 1280x720, tiled 8:8:8:8)
- GPU command processor decodes real traffic: ME_INIT, INDIRECT_BUFFER, WAIT_REG_MEM, COND_WRITE, IM_LOAD(_IMMEDIATE),
  SET_BIN_MASK_LO, EVENT_WRITE(_SHD), XE_SWAP and ~12.7k `DRAW_INDX` (draws are currently ignored)
- Frames dumped with `X360_DUMP_FRAMES=<dir>` are fully black - nothing rasterizes yet
- Hard-won fix worth remembering: 32-bit kernel return values must be **sign-extended** in r3 (`or. r31,r3` records CR0 from all 64 bits)
- Debug tools: `X360_TRACE_KERNEL/IO/GPU`, `X360_DUMP` (disassembled), `X360_WATCH=pc`, `X360_CRASH_DUMP=addr,len`,
  `X360_HEXDUMP=file,off,len`, `X360_LIST_DISC`, `X360_SLICE`, crash backtraces, per-thread state at exit

## FIRST IMAGE (milestone reached)
GTA IV's copyright/legal screen renders legibly from the retail disc (frame dump `frames/frame_0015.png`).
Pipeline that produced it, all in-tree and written from scratch:
- `xenos_vm.rs`   Xenos shader microcode interpreter (control flow, co-issued vector+scalar ALU, vertex/texture fetch)
- `xenos_draw.rs` vertex fetch -> VS -> primitive assembly -> viewport -> rasterize -> PS -> blend -> surface; EDRAM resolve
- `xenos_tex.rs`  fetch-constant decode, tiled addressing, 8/16/32-bit formats, DXT1/3/5, point/bilinear
- Real vsync: a virtual 60Hz interrupt thread calls the title's graphics callback (that unstuck the game)
- Real physical-heap allocator with free list (leaking it broke texture creation)
- Reference data used: public Xenos register/ucode layout tables, export names (see THIRD_PARTY note TODO)

## Known gaps / debt
- Physical-memory aliasing (0xA0000000 / 0xC0000000 / 0xE0000000 views of the same RAM) not modelled
- Only the main XEX's dynamic TLS slots live in a host hashmap, not in guest memory
- `NtQueryInformationFile` class 26 (XCTD compression) answered "invalid class" — fine for retail discs
- Float exceptions/rounding modes, VMX saturation flags, reservation shared across threads
- Kernel struct offsets (PCR/KTHREAD) are from memory; verify as games poke at them

## Platform services (done: the "everything but network" pass)
Each is its own module, with unit tests, and shows up in the end-of-run summary.
- `sched.rs`     One CPU time-slices six hardware threads, so only interrupt/audio callbacks pre-empt; ordinary
                 priorities are bookkeeping (strict priority starved ready workers: a spinning main thread froze
                 the game). Per-thread IRQL gates pre-emption. Affinity/core -> PCR+0x10C. ExCreateThread flags
                 bits 24..29 pin the hardware thread (the audio mixer's per-CPU barrier needs that).
                 KeWaitForMultipleObjects / NtWaitForMultipleObjectsEx (any/all, timeouts, auto-reset consume).
                 Title-terminate notifications run on shutdown.
- `vm.rs`        Real virtual/physical allocators with free, coalescing, fixed-address allocation.
- `gpu.rs`       Resumable command processor: WAIT_REG_MEM / COND_WRITE stall and resume, INTERRUPT (0x54)
                 delivered on the core its mask selects, scratch-register mirroring, bin mask/select, EVENT_WRITE.
- `xam.rs`       Signed-in profile (offline gamertag + XUID), XInput state/capabilities/vibration/keystrokes,
                 notification queues per listener (listener handle is signaled while non-empty).
                 No window/pad yet: input is scripted with `X360_PAD="300:tap=START,600:+A,..."`.
- `display.rs`   Video mode, VdTurnDisplayOn/Off (frames swapped while off aren't shown), refresh override
                 drives the vsync period, scaler/EDRAM-retrain bookkeeping. Scaler command buffer is empty on purpose.
- `audio.rs`     The render-driver callback runs every 256 samples @48kHz of virtual time on its own thread;
                 submitted 6-channel frames are downmixed to a stereo WAV (`X360_DUMP_AUDIO=file.wav`).
                 GTA IV's mixer runs end to end (hundreds of callbacks/frames) - all silent so far.
- `cstdio.rs/crt.rs` printf engine (flags/width/precision/ll/%ls/floats, register + va_list arguments),
                 sprintf, _vsnprintf, DbgPrint, KeQuerySystemTime (deterministic virtual wall clock), Rtl time/string helpers.
- `exceptions.rs` KeBugCheck(Ex) and RtlCaptureContext (layout unverified). RtlUnwind / __C_specific_handler stop
                 the run with an exact report: real SEH needs a .pdata walker.

## Past the legal screen (milestone reached)
GTA IV now runs the legal screen to its end and shows the **Rockstar Games logo** (fade in/out), ~25 s of
virtual time, ~550 frames. Fixes that got it there, all found by crash-looping on the retail disc:
- `vadduws` & the rest of VMX integer saturating/multiply/sum ops (`vmx_int.rs`), `XMACreateContext/Release`
- `VdSwap` must zero its 148-byte "pending display changes" struct (stale stack made the renderer divide by zero)
- `VdInitializeScalerCommandBuffer` returns its dword count (fills no-op packets)
- Virtual time: 3 cycles/instruction (`X360_CPI`) + 1500 cycles per kernel call (`X360_SYSCALL_CYCLES`)
- Writable storage: `hostfs.rs`/`hostio.rs` map `cache:`, `\Device\Harddisk0\...` etc. onto `./hdd` (`X360_HDD`)
- Speed: LTO + fast 16/64-bit guest memory + taken-branch trace ring: ~33 -> ~45 M instr/s (`X360_NO_RENDER=1` skips pixels)
Debug additions: `X360_BACKTRACES=1` (every live thread's call chain at exit), `X360_WATCH` keeps the *last* hits and
dumps what r4/r6 point at, byte-store aware `X360_WATCH_MEM`, scheduler/kernel/GPU self-profile lines.

## Honest gaps (not done)
- XMA: the decoder isn't written (contexts/registers are plain storage). Will matter when the game plays XMA audio.
- SEH/C++ exception unwinding (no .pdata walker). Thread-naming exception is handled.
- Network (excluded on purpose).
- Storage: `\Device\Harddisk0\partition0`, `cache:`/`cache1:` and XamContent* (saves) are the current wall -
  the game keeps polling for its cache partition (thread 11 spins at 0x829cfe90).
- Dozens of kernel imports GTA IV uses are still unimplemented (see `X360_LIST_IMPORTS=1` diffed against the arms).

## Debug knobs
`X360_TRACE_KERNEL/IO/GPU/GPU_WAIT`, `X360_DUMP=addr,n`, `X360_WATCH=pc,..`, `X360_WATCH_MEM=addr`,
`X360_CRASH_DUMP=addr,n`, `X360_LIST_IMPORTS=1`, `X360_DUMP_FRAMES=dir`, `X360_DUMP_AUDIO=file.wav`, `X360_PAD=script`.

## Later
- [ ] HDD/cache devices + XamContent (saves), XMA decode, .pdata unwinder, save states, debugger, JIT
