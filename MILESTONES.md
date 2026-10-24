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

## JIT and the 32-bit-mode fix
- `xenon-cpu/src/jit.rs`: block JIT (Cranelift). Natively translates the hot integer / load-store / branch
  instructions (~86% of what GTA IV executes) with inline guest-memory access; everything else runs through the
  interpreter (`Cpu::exec`) from inside the block. Differential tests compare it against the interpreter on random
  instructions in both CPU modes, including page edges and unmapped pages. ~33 -> ~120-170 M instr/s.
  `X360_NOJIT=1` = interpreter only (exact fault pcs). The memory/PC watches work with the JIT.
- **Xbox 360 user code runs in 32-bit mode (MSR[SF]=0)**: CR0, carries/overflow and the CTR test look at the low
  word. The C runtime's `--cnt < 0` on a zero-extended int depends on it; with 64-bit semantics `sscanf` ran
  off the end of its buffer and smashed the main thread's stack (crash after ~36 G instructions on the loading
  screen). Fixed in interpreter and JIT, with a regression test.
- Other fixes this stretch: audio/interrupt helper threads recycle their resources (the kernel heap used to run
  into the game image after ~6500 threads), `VdSwap` pending-state struct, scaler command buffer size.
- Tooling: `X360_TRACE_CALLS=N,M` (instruction trace of the Nth visit to the `X360_WATCH` pc, main thread only),
  `cargo build --profile fast` for quick rebuilds.

## Renderer rework (after the predication fix)
- **Loading screen renders correctly past frame 9604** (GTA IV artwork). The frames after ~9620 are black; the game draws
  ~835 draws/frame (z-prepass, 16x16 visibility-query points, a 992-triangle lighting pass) and nothing reaches the front buffer yet.
- RB_MODECONTROL was misread: **4 = colour+depth draw, 5 = depth-only draw, 6 = EDRAM copy (resolve)**. Resolves now run on mode 6,
  restricted to the window-scissor rectangle (per-tile copies and clears only touch their own rows); mode-5 draws no longer resolve.
- Predicated tiling uses one full-frame surface: tile scissors are absolute frame rows, so surfaces now grow to cover the scissor.
- **Depth/stencil** (f32 depth + 8-bit stencil per EDRAM depth base, early test, depth clear on resolve), a depth-only fast path,
  decoded-texture cache validated by content hash (`xenos_texcache.rs`), `X360_RENDER_SCALE` (e.g. 0.5) for fast development frames,
  quicker resolves. Profile of the loading-screen phase: 496k draws, 16.6 G pixels shaded, 1110 s raster, 122 s resolves.
- **Batched pixel shaders** (`xenos_batch.rs`, `xenos_batch_exec.rs`): shaders are compiled once per draw (constants and
  bool-constant branches resolved, operands pre-decoded) and run 16 pixels in lockstep over structure-of-arrays registers,
  with a runtime-dispatched AVX2 build. Loops/calls/jumps/relative addressing fall back to the reference interpreter.
  `X360_VERIFY_BATCH=1` cross-checks every batch against the reference (0 mismatches on the loading-screen and first 3D
  frames); `X360_NO_BATCH=1` disables it. Big draws now run on a persistent rayon pool in small row bands (work stealing)
  instead of six fixed bands. The 992-triangle, 7.6 M pixel lighting draw: ~25 s -> ~1.7 s (half resolution).
- (Before batching) remaining cost was the interpreted pixel shader (~3 us/pixel for the 100+ op lighting shaders, 7-30 M pixels per draw).
  Next: pre-decoded, batched (SoA, lockstep) pixel shader execution.
- Debug: `X360_LOG_SLOW_DRAWS=1|2`, `X360_STATUS_RENDER=1`, `X360_SAMPLE_DRAWS=1` (now with frame number and state), `X360_LOG_RESOLVES`,
  `X360_LOG_TILE_REGS`, `X360_PRED_HIST`.

## BREAKTHROUGH: predicated tiling (frame 9604 stall solved)
- Root cause: the CP ignored **PM4 predication**. Type-3 packets with the predicate bit (header bit 0) must only run when
  `BIN_SELECT & BIN_MASK != 0`. The driver's per-tile epilogue IB (scratch callback, waits, INTERRUPT) is masked to bit 31,
  which only the first tile's select (`0x80000003`) has, so real hardware raises ONE interrupt per frame. We ran the
  epilogue for all 3 tiles; the extra interrupts re-enqueued the tile item for the driver thread forever.
- Fix: `Gpu::exec_packet` skips predicated type-3 packets when the tile does not intersect the mask
  (`X360_NO_PREDICATION=1` to disable). Also fixed my own `exec_bin` regression (SELECT_LO/MASK_HI arms).
- Result: frames advance past 9604 (9835+ in 3 G instructions) and disc I/O jumps from 122 MB to 2 GB: the game moved on
  from the loading screen into streaming real content.

## Save states and the frame-9604 stall (investigation log)
- **Save states**: `X360_SAVE_STATE=file,step` / `X360_LOAD_STATE=file` snapshot the whole machine (CPU, kernel, GPU,
  memory; ~430 MB). Resuming takes 0.3 s, so experiments that used to replay 50 G instructions (8 min) now take seconds.
  State files are `*.bin` (git-ignored).
- Where the game stops: after swap #9604 (~50.15 G instrs) the render thread waits for the GPU fence forever while the
  command processor loops over the same 18 one-primitive draw+resolve jobs (rotating destinations 0x1af27000 ...).
- Findings: (1) the driver watchdog thread (thread 4, 30 ms timeout, check at 0x829d5338) fires against our slow virtual
  GPU; `X360_WAIT_SCALE=100` stretches timeouts and buys ~50 more frames, but the loop returns, so it is not the root
  cause. (2) CP interrupts only ran at the end of a time slice; `X360_IRQ_PREEMPT=1` runs them at once (10x CP speed) but
  starves normal threads, so it is opt-in. (3) EVENT_WRITE_EXT extents and RB sample-count results are irrelevant here
  (RB_SAMPLE_COUNT_ADDR is never set in the loop). EVENT_WRITE ZPD/VIZQUERY_END now writes a sample count when it is.
- Narrowing (later): the post-9604 ring traffic is written by **thread 4** (the driver's deferred-command/tiling worker),
  not the render thread. Each CP INTERRUPT runs the ISR, which calls the driver callback `0x829ddd30` to enqueue work item
  `0xd7664d80` for thread 4; thread 4 walks the item's records (extent/visibility rectangles, then IB calls
  `0x175db800`/`0x175d6160`, 29 and 5538 dwords) and submits them again, mostly sitting in the ring-space wait. The same
  item is re-enqueued forever. Ruled out: JIT miscompile (identical under `X360_NOJIT=1`), ISR/CP ordering race
  (`X360_NO_IRQ_HOLD` vs default hold), query/extent values, watchdog timeouts, ring wrap/replay (only 134k ring dwords
  vs 14M IB-packet executions: the ring is fine, the IBs are what repeat). Open question: why the tile/query state
  machine in thread 4 (struct at 0x400ae5c0, work queue at 0x400ae7d4) never reaches its end record.
- Tooling: `X360_TRACE_TID=<id>` picks the thread for `X360_TRACE_CALLS`, `X360_LOG_IB=1` logs IB calls and the ring base,
  GPU summary counts ring vs IB packets, `X360_NO_IRQ_HOLD=1` disables the CP-waits-for-ISR ordering.
- Tile-replay anatomy (frame 9605 onwards): the render path switched to **predicated tiling** (3 tiles of 256 rows). The ring
  holds only `SET_BIN_SELECT`, scene IB `0x175d6160` (5538 dw) and a 29-dword epilogue IB `0x175db800` per tile. The epilogue
  is: mask `0x80000000`; scratch callback `0x829ddd30` + arg (work item); scratch0=4 (mirrored to memory = the flag at
  `0x17b0b000`); wait flag==4; `INTERRUPT`; wait flag==0 (the ISR clears it); clear callback; mask all. The ISR runs the
  callback, which enqueues the item for thread 4. Thread 4 passes alternate "resume-set" / "finish (cancel one pending
  enqueue)" (counts 1654/1654 per 3308 passes vs 4962 enqueues, i.e. 3 enqueues per 2 passes + 1 cancel), and each
  pass issues the next tiles, so the loop is self-sustaining: nothing ever makes a pass issue no tile commands.
  Unresolved: what the real final pass does differently. Tried without effect: CP start delay (`X360_CP_DELAY`), CP hold
  during ISR (`X360_NO_IRQ_HOLD` to disable), ISR preemption (`X360_IRQ_PREEMPT`), wait scaling, JIT off.
- More tooling: `X360_LOG_RING=1` (ring packets), `X360_LOG_IB=1` also dumps small IBs and bin selects,
  `X360_TRACE_LEN`, `X360_WATCH_KEEP=n`, watch lines include r29, watchmem lines include sp.
- Tooling: `X360_TRACE_GPU_FROM/LEN` bound the packet trace, backtraces list unsignaled multi-wait objects.

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
