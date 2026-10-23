# x360 — an Xbox 360 emulator in Rust

An Xbox 360 emulator written from scratch, built **crash-driven**: run a real game, see exactly what
breaks, implement that piece, repeat. The test title is *Grand Theft Auto IV* (retail disc image you
supply yourself — no game data is in this repository).

**Where it is:** boots the retail XEX, runs the legal screen, the Rockstar logo and the GTA IV title
screen, and reaches the loading screen with correct artwork, using its own CPU, kernel, GPU and
audio emulation. It does not reach gameplay yet.

## What's in here

| Crate / module | What it does |
|---|---|
| `xenon-cpu` | PowerPC (Xenon) interpreter incl. VMX/VMX128, plus a **block JIT** (Cranelift) that translates the hot integer / load-store / branch instructions to native code and falls back to the interpreter for the rest. The interpreter is the reference: random-instruction differential tests compare every register and memory byte. |
| `x360/src/xex.rs`, `loader.rs`, `iso.rs`, `fs.rs` | XEX2 decryption/decompression, XISO/GDF disc reader, module loading and kernel import patching |
| `x360/src/kernel.rs`, `sched.rs`, `dispatcher.rs`, `objects.rs`, `vm.rs` | HLE kernel: threads and scheduler (IRQL, affinity, waits, interrupts), dispatcher objects, memory manager |
| `x360/src/xam.rs`, `display.rs`, `audio.rs`, `hostio.rs` | profile/controller/notifications, display hardware state, audio render driver, writable storage |
| `x360/src/gpu.rs`, `xenos_*.rs` | Xenos command processor (PM4), shader microcode interpreter, software rasteriser with texture sampling, EDRAM resolve |
| `x360/src/window.rs` | optional live window with keyboard-as-controller |

## Build and run

```sh
cargo build --release
cargo run --release -p x360 -- path/to/game.iso [max_instructions]
```

You must provide your own legally obtained disc image. Useful switches (environment variables):

| Variable | Effect |
|---|---|
| `X360_WINDOW=1` | live preview window (arrows = D-pad, Z/X/C/V = A/B/X/Y, Enter = Start, WASD = left stick, Esc quits) |
| `X360_NOJIT=1` | interpreter only (exact fault addresses; use when debugging) |
| `X360_NO_RENDER=1` | skip pixel work — much faster when only CPU/kernel progress matters |
| `X360_DUMP_FRAMES=dir` (+ `X360_DUMP_EVERY=N`) | write presented frames as PNG; `dir/latest.png` is always the newest |
| `X360_DUMP_AUDIO=file.wav` | capture the audio output |
| `X360_PAD="300:tap=START,600:+A"` | scripted controller input by frame |
| `X360_BACKTRACES=1` | at exit, a call chain for every live thread |
| `X360_WATCH=pc,...` / `X360_WATCH_MEM=addr` | log register state at a pc / every write to an address (last hits kept) |
| `X360_TRACE_KERNEL=1`, `X360_TRACE_IO=1`, `X360_TRACE_GPU=1` | trace kernel calls / file I/O / GPU packets |
| `X360_LIST_IMPORTS=1`, `X360_LIST_DISC=1`, `X360_OPSTATS=1` | list imports / disc contents / instruction mix |

See `MILESTONES.md` for the running log of what was fixed and what's next.

## Tests

```sh
cargo test --release
```

Includes the JIT-vs-interpreter differential tests, GPU command-processor tests, scheduler tests and
format/ABI tests.

## Status and honest limits

* Speed: ~35 M instructions/s interpreted, ~120–170 M with the JIT. A real console needs on the order of
  a billion per second, so the game runs far below real time for now. Planned: inline memory paths (done),
  block chaining, native vector/float translation, multi-threaded guest cores, a GPU back end.
* Not implemented: XMA audio decoding, SEH/C++ exception unwinding, networking, real save games.
* Software rasteriser only; no hardware GPU back end yet.

## Provenance

All emulation logic is written from scratch in this repository. Reference data (kernel export names,
Xenos register/microcode layouts) was taken from public documentation and the Xenia project — see
`THIRD_PARTY.md`.

## Legal

This project contains no Microsoft or game code or data. You need your own console-dumped game image.
Emulating hardware for interoperability and preservation is the goal; do not use this to run pirated
software.
