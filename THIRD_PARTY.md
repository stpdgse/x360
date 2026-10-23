# Third-party material

All emulation logic (PowerPC/VMX interpreter, XEX/XISO loaders, kernel HLE, scheduler, Xenos shader
interpreter, rasterizer, audio/input/display services) is written from scratch in this repository.

## Reference data taken from the Xenia project (BSD-3-Clause)
Used as *hardware/ABI reference tables*, not as emulator code:
- `x360/src/export_table.rs` — the (library, ordinal) -> export-name table for xboxkrnl.exe / xam.xex.
  Generated from Xenia's export tables. These are names and numbers that Microsoft's kernel defines; the
  table only tells us which function an import thunk refers to.
- Xenos register indices and shader-microcode / fetch-constant bit layouts (`gpu.rs`, `xenos_vm.rs`,
  `xenos_tex.rs`) were cross-checked against Xenia's `register_table.inc` and `ucode.h`.

Xenia: https://github.com/xenia-project/xenia — Copyright (c) Xenia project contributors, BSD 3-Clause License.

## Crates (see Cargo.toml)
- `aes`, `cbc` (RustCrypto, MIT/Apache-2.0) — decrypting the XEX2 file key and image.
- `lzxd` — LZX decompression of compressed XEX images.

## Game content
No game data is in this repository. The emulator reads a retail disc image the user owns, at runtime only.
