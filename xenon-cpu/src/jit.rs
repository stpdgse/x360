//! Block JIT: translates straight-line runs of PowerPC code (ending at a branch) into native x86-64
//! with Cranelift. Hot integer / load-store / branch instructions are translated natively; anything
//! else is executed by calling back into the interpreter (`Cpu::exec`), so every instruction works
//! from day one and only the translation of the common ones needs to be exactly right.
//!
//! The interpreter stays the reference: `tests` run random instructions through both and compare
//! every register and memory byte.
//!
//! Limits (by design, for now): self-modifying code isn't detected, faults inside a block surface at
//! the block's end (set `X360_NOJIT=1` to debug with exact pcs), and blocks only form inside
//! `[code_start, code_end)` (the loaded executable image).

use crate::interp::Trap;
use crate::memory::Bus;
use crate::Cpu;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, AbiParam, FuncRef, InstBuilder, MemFlagsData, Signature, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::marker::PhantomData;
use std::mem::offset_of;

const MAX_BLOCK: u32 = 64;
const HEAT_THRESHOLD: u8 = 24;
const TRAP_FLAG: u64 = 1 << 63;

/// (cpu, bus, page table of the bus or null)
type BlockFn = unsafe extern "C" fn(*mut Cpu, *mut u8, *const *mut u8) -> u64;

/// Per-entry-point state packed in a u64: 0..=255 = times seen (cold), NEVER = compilation refused,
/// anything else = address of the compiled block.
const NEVER: u64 = 1 << 63;

pub enum Outcome {
    /// Not compiled (yet): the caller should interpret one instruction.
    Interpret,
    /// Ran a block of `n` instructions.
    Ran(u64),
    /// Ran `n` instructions and then trapped (a kernel call, trap instruction, unimplemented op...).
    Trapped(u64, Trap),
}

pub struct Jit<B: Bus> {
    module: JITModule,
    ctx: Context,
    fb: FunctionBuilderContext,
    slots: Vec<u64>,
    code_start: u32,
    code_end: u32,
    counter: u32,
    /// Addresses a block must never run through (debug watches): blocks end just before them so the
    /// run loop sees each visit.
    break_pcs: std::collections::HashSet<u32>,
    pub blocks_compiled: u64,
    pub instrs_compiled: u64,
    pub native_instrs: u64,
    pub compile_time: std::time::Duration,
    _bus: PhantomData<B>,
}

// ---- helpers called from generated code (C ABI) ------------------------------------------------

unsafe extern "C" fn h_exec<B: Bus>(cpu: *mut Cpu, bus: *mut B, pc: u64, inst: u32) -> u32 {
    let (cpu, bus) = unsafe { (&mut *cpu, &mut *bus) };
    match cpu.exec(bus, pc, inst) {
        Ok(()) => 0,
        Err(t) => {
            cpu.pending_trap = Some(t);
            1
        }
    }
}
unsafe extern "C" fn h_r8<B: Bus>(bus: *mut B, a: u32) -> u32 {
    unsafe { (*bus).read8(a) as u32 }
}
unsafe extern "C" fn h_r16<B: Bus>(bus: *mut B, a: u32) -> u32 {
    unsafe { (*bus).read16(a) as u32 }
}
unsafe extern "C" fn h_r32<B: Bus>(bus: *mut B, a: u32) -> u32 {
    unsafe { (*bus).read32(a) }
}
unsafe extern "C" fn h_r64<B: Bus>(bus: *mut B, a: u32) -> u64 {
    unsafe { (*bus).read64(a) }
}
unsafe extern "C" fn h_w8<B: Bus>(bus: *mut B, a: u32, v: u32) {
    unsafe { (*bus).write8(a, v as u8) }
}
unsafe extern "C" fn h_w16<B: Bus>(bus: *mut B, a: u32, v: u32) {
    unsafe { (*bus).write16(a, v as u16) }
}
unsafe extern "C" fn h_w32<B: Bus>(bus: *mut B, a: u32, v: u32) {
    unsafe { (*bus).write32(a, v) }
}
unsafe extern "C" fn h_w64<B: Bus>(bus: *mut B, a: u32, v: u64) {
    unsafe { (*bus).write64(a, v) }
}

// ---- field offsets in Cpu ----------------------------------------------------------------------

const O_GPR: i32 = offset_of!(Cpu, gpr) as i32;
const O_CR: i32 = offset_of!(Cpu, cr) as i32;
const O_LR: i32 = offset_of!(Cpu, lr) as i32;
const O_CTR: i32 = offset_of!(Cpu, ctr) as i32;
const O_XER: i32 = offset_of!(Cpu, xer) as i32;
const O_PC: i32 = offset_of!(Cpu, pc) as i32;
const O_CYCLES: i32 = offset_of!(Cpu, cycles) as i32;
const O_CPI: i32 = offset_of!(Cpu, cpi) as i32;

fn mask64(mb: u32, me: u32) -> u64 {
    let a = !0u64 >> mb;
    let b = !0u64 << (63 - me);
    if mb <= me { a & b } else { a | b }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reg {
    Lr,
    Ctr,
    Xer,
}

struct HelperRefs {
    exec: FuncRef,
    r8: FuncRef,
    r16: FuncRef,
    r32: FuncRef,
    r64: FuncRef,
    w8: FuncRef,
    w16: FuncRef,
    w32: FuncRef,
    w64: FuncRef,
}

/// Per-block code generator state: cached register values (written back lazily).
struct Em<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    cpu: Value,
    bus: Value,
    h: HelperRefs,
    cpi: Value,
    /// The bus's page table, passed in per call (None: every access calls a helper).
    table: Option<Value>,
    gpr: [Option<Value>; 32],
    gpr_dirty: [bool; 32],
    cr: Option<Value>, // i64 (zero-extended u32)
    cr_dirty: bool,
    spr: [Option<Value>; 3],
    spr_dirty: [bool; 3],
    /// instructions executed since cycles were last written back
    pending: u32,
    /// total instructions emitted so far (including the one being emitted)
    count: u32,
    native: u32,
}

impl<'a, 'b> Em<'a, 'b> {
    fn iconst(&mut self, v: i64) -> Value {
        self.b.ins().iconst(types::I64, v)
    }
    fn load(&mut self, off: i32, ty: cranelift_codegen::ir::Type) -> Value {
        self.b.ins().load(ty, MemFlagsData::trusted(), self.cpu, off)
    }
    fn store(&mut self, v: Value, off: i32) {
        self.b.ins().store(MemFlagsData::trusted(), v, self.cpu, off);
    }

    fn get(&mut self, r: usize) -> Value {
        if let Some(v) = self.gpr[r] {
            return v;
        }
        let v = self.load(O_GPR + 8 * r as i32, types::I64);
        self.gpr[r] = Some(v);
        v
    }
    fn set(&mut self, r: usize, v: Value) {
        self.gpr[r] = Some(v);
        self.gpr_dirty[r] = true;
    }
    /// ra-or-zero base for effective addresses
    fn base(&mut self, ra: usize) -> Value {
        if ra == 0 { self.iconst(0) } else { self.get(ra) }
    }

    fn get_cr(&mut self) -> Value {
        if let Some(v) = self.cr {
            return v;
        }
        let v32 = self.load(O_CR, types::I32);
        let v = self.b.ins().uextend(types::I64, v32);
        self.cr = Some(v);
        v
    }
    fn set_cr(&mut self, v: Value) {
        self.cr = Some(v);
        self.cr_dirty = true;
    }

    fn get_spr(&mut self, r: Reg) -> Value {
        let i = r as usize;
        if let Some(v) = self.spr[i] {
            return v;
        }
        let off = match r {
            Reg::Lr => O_LR,
            Reg::Ctr => O_CTR,
            Reg::Xer => O_XER,
        };
        let v = self.load(off, types::I64);
        self.spr[i] = Some(v);
        v
    }
    fn set_spr(&mut self, r: Reg, v: Value) {
        self.spr[r as usize] = Some(v);
        self.spr_dirty[r as usize] = true;
    }

    /// Write every dirty cached register back to the Cpu.
    fn flush(&mut self) {
        for r in 0..32 {
            if self.gpr_dirty[r] {
                let v = self.gpr[r].unwrap();
                self.store(v, O_GPR + 8 * r as i32);
                self.gpr_dirty[r] = false;
            }
        }
        if self.cr_dirty {
            let v = self.cr.unwrap();
            let v32 = self.b.ins().ireduce(types::I32, v);
            self.store(v32, O_CR);
            self.cr_dirty = false;
        }
        for (i, off) in [(0, O_LR), (1, O_CTR), (2, O_XER)] {
            if self.spr_dirty[i] {
                let v = self.spr[i].unwrap();
                self.store(v, off);
                self.spr_dirty[i] = false;
            }
        }
    }
    fn invalidate(&mut self) {
        self.gpr = [None; 32];
        self.cr = None;
        self.spr = [None; 3];
    }

    /// cycles += pending * cpi
    fn sync_cycles(&mut self) {
        if self.pending == 0 {
            return;
        }
        let cyc = self.load(O_CYCLES, types::I64);
        let n = self.iconst(self.pending as i64);
        let add = self.b.ins().imul(n, self.cpi);
        let new = self.b.ins().iadd(cyc, add);
        self.store(new, O_CYCLES);
        self.pending = 0;
    }

    fn ret(&mut self, v: u64) {
        let r = self.iconst(v as i64);
        self.b.ins().return_(&[r]);
    }

    /// Block exit: write state back, set pc, return the instruction count.
    fn finish(&mut self, next_pc: Value) {
        self.flush();
        self.sync_cycles();
        self.store(next_pc, O_PC);
        self.ret(self.count as u64);
    }

    // ---- flag helpers ----

    /// CR field `n` (0..7) = lt/gt/eq (i8 booleans) | SO from XER
    fn set_cr_field(&mut self, n: u32, lt: Value, gt: Value, eq: Value) {
        let xer = self.get_spr(Reg::Xer);
        let so = self.b.ins().ushr_imm(xer, 31);
        let so = self.b.ins().band_imm(so, 1);
        let widen = |me: &mut Self, v: Value, sh: i64| {
            let w = me.b.ins().uextend(types::I64, v);
            me.b.ins().ishl_imm(w, sh)
        };
        let l = widen(self, lt, 3);
        let g = widen(self, gt, 2);
        let e = widen(self, eq, 1);
        let t = self.b.ins().bor(l, g);
        let t = self.b.ins().bor(t, e);
        let nib = self.b.ins().bor(t, so);
        let shift = 28 - 4 * n as i64;
        let cr = self.get_cr();
        let cleared = self.b.ins().band_imm(cr, !(0xFi64 << shift) & 0xFFFF_FFFF);
        let placed = self.b.ins().ishl_imm(nib, shift);
        let new = self.b.ins().bor(cleared, placed);
        self.set_cr(new);
    }
    fn set_cr0(&mut self, v: Value) {
        let zero = self.iconst(0);
        let lt = self.b.ins().icmp(IntCC::SignedLessThan, v, zero);
        let gt = self.b.ins().icmp(IntCC::SignedGreaterThan, v, zero);
        let eq = self.b.ins().icmp(IntCC::Equal, v, zero);
        self.set_cr_field(0, lt, gt, eq);
    }

    // ---- memory ----

    fn addr32(&mut self, ea: Value) -> Value {
        self.b.ins().ireduce(types::I32, ea)
    }
    /// Fast-path check for an access of `size` bytes at guest address `a` (i32): if the page is mapped and the
    /// access stays inside it, returns the host pointer (i64) in a new "fast" block; otherwise jumps to `slow`.
    fn page_ptr(&mut self, tbl: Value, a: Value, size: u32, slow: cranelift_codegen::ir::Block) -> Value {
        let a64 = self.b.ins().uextend(types::I64, a);
        let idx = self.b.ins().ushr_imm(a64, 12);
        let idx8 = self.b.ins().ishl_imm(idx, 3);
        let slot = self.b.ins().iadd(tbl, idx8);
        let base = self.b.ins().load(types::I64, MemFlagsData::trusted(), slot, 0);
        let off = self.b.ins().band_imm(a64, 0xFFF);
        let mapped = self.b.ins().icmp_imm(IntCC::NotEqual, base, 0);
        let fits = self.b.ins().icmp_imm(IntCC::UnsignedLessThanOrEqual, off, (4096 - size) as i64);
        let ok = self.b.ins().band(mapped, fits);
        let fast = self.b.create_block();
        self.b.ins().brif(ok, fast, &[], slow, &[]);
        self.b.switch_to_block(fast);
        self.b.seal_block(fast);
        self.b.ins().iadd(base, off)
    }

    fn read(&mut self, size: u32, ea: Value) -> Value {
        let a = self.addr32(ea);
        let slow_fn = match size {
            1 => self.h.r8,
            2 => self.h.r16,
            4 => self.h.r32,
            _ => self.h.r64,
        };
        let Some(table) = self.table else {
            let call = self.b.ins().call(slow_fn, &[self.bus, a]);
            let r = self.b.inst_results(call)[0];
            return if size == 8 { r } else { self.b.ins().uextend(types::I64, r) };
        };
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        let p = self.page_ptr(table, a, size, slow);
        let ty = match size {
            1 => types::I8,
            2 => types::I16,
            4 => types::I32,
            _ => types::I64,
        };
        let raw = self.b.ins().load(ty, MemFlagsData::new().with_notrap(), p, 0);
        let v = if size == 1 { raw } else { self.b.ins().bswap(raw) };
        let v64 = if size == 8 { v } else { self.b.ins().uextend(types::I64, v) };
        self.b.ins().jump(done, &[cranelift_codegen::ir::BlockArg::Value(v64)]);
        self.b.switch_to_block(slow);
        self.b.seal_block(slow);
        let call = self.b.ins().call(slow_fn, &[self.bus, a]);
        let r = self.b.inst_results(call)[0];
        let r64 = if size == 8 { r } else { self.b.ins().uextend(types::I64, r) };
        self.b.ins().jump(done, &[cranelift_codegen::ir::BlockArg::Value(r64)]);
        self.b.switch_to_block(done);
        self.b.seal_block(done);
        self.b.block_params(done)[0]
    }

    fn write(&mut self, size: u32, ea: Value, v: Value) {
        let a = self.addr32(ea);
        let (slow_fn, val32) = match size {
            1 => (self.h.w8, true),
            2 => (self.h.w16, true),
            4 => (self.h.w32, true),
            _ => (self.h.w64, false),
        };
        let Some(table) = self.table else {
            let val = if val32 { self.b.ins().ireduce(types::I32, v) } else { v };
            self.b.ins().call(slow_fn, &[self.bus, a, val]);
            return;
        };
        let slow = self.b.create_block();
        let done = self.b.create_block();
        let p = self.page_ptr(table, a, size, slow);
        let narrow = match size {
            1 => self.b.ins().ireduce(types::I8, v),
            2 => self.b.ins().ireduce(types::I16, v),
            4 => self.b.ins().ireduce(types::I32, v),
            _ => v,
        };
        let swapped = if size == 1 { narrow } else { self.b.ins().bswap(narrow) };
        self.b.ins().store(MemFlagsData::new().with_notrap(), swapped, p, 0);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(slow);
        self.b.seal_block(slow);
        let val = if val32 { self.b.ins().ireduce(types::I32, v) } else { v };
        self.b.ins().call(slow_fn, &[self.bus, a, val]);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
        self.b.seal_block(done);
    }

    /// Run `inst` through the interpreter. State is flushed first and re-read afterwards.
    fn helper(&mut self, pc: u32, inst: u32) {
        self.flush();
        self.pending += 1; // this instruction's cycles (the interpreter charges them before executing)
        self.sync_cycles();
        let pcv = self.iconst(pc as i64);
        self.store(pcv, O_PC);
        let instv = self.b.ins().iconst(types::I32, inst as i64);
        let call = self.b.ins().call(self.h.exec, &[self.cpu, self.bus, pcv, instv]);
        let trapped = self.b.inst_results(call)[0];
        let exit = self.b.create_block();
        let cont = self.b.create_block();
        self.b.ins().brif(trapped, exit, &[], cont, &[]);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
        let flag = TRAP_FLAG | self.count as u64;
        self.ret(flag);
        self.b.switch_to_block(cont);
        self.b.seal_block(cont);
        self.invalidate();
    }

    /// Translate one non-terminal instruction natively. Returns false if it must go to the interpreter.
    fn native(&mut self, pc: u32, inst: u32) -> bool {
        let opcd = inst >> 26;
        let rd = ((inst >> 21) & 31) as usize;
        let ra = ((inst >> 16) & 31) as usize;
        let rb = ((inst >> 11) & 31) as usize;
        let simm = (inst & 0xFFFF) as i16 as i64;
        let uimm = (inst & 0xFFFF) as i64;
        let rc = inst & 1 != 0;
        let _ = pc;
        match opcd {
            14 | 15 => {
                let b = self.base(ra);
                let imm = if opcd == 14 { simm } else { simm << 16 };
                let r = self.b.ins().iadd_imm(b, imm);
                self.set(rd, r);
            }
            24 | 25 => {
                let s = self.get(rd);
                let r = self.b.ins().bor_imm(s, if opcd == 24 { uimm } else { uimm << 16 });
                self.set(ra, r);
            }
            26 | 27 => {
                let s = self.get(rd);
                let r = self.b.ins().bxor_imm(s, if opcd == 26 { uimm } else { uimm << 16 });
                self.set(ra, r);
            }
            28 | 29 => {
                let s = self.get(rd);
                let r = self.b.ins().band_imm(s, if opcd == 28 { uimm } else { uimm << 16 });
                self.set(ra, r);
                self.set_cr0(r);
            }
            // cmpli / cmpi
            10 | 11 => {
                let bf = (inst >> 23) & 7;
                let l64 = (inst >> 21) & 1 != 0;
                let a = self.get(ra);
                let signed = opcd == 11;
                let (x, y) = if signed {
                    let x = if l64 { a } else { let t = self.b.ins().ireduce(types::I32, a); self.b.ins().sextend(types::I64, t) };
                    (x, self.iconst(simm))
                } else {
                    let x = if l64 { a } else { let t = self.b.ins().ireduce(types::I32, a); self.b.ins().uextend(types::I64, t) };
                    (x, self.iconst(uimm))
                };
                let (clt, cgt) = if signed { (IntCC::SignedLessThan, IntCC::SignedGreaterThan) } else { (IntCC::UnsignedLessThan, IntCC::UnsignedGreaterThan) };
                let lt = self.b.ins().icmp(clt, x, y);
                let gt = self.b.ins().icmp(cgt, x, y);
                let eq = self.b.ins().icmp(IntCC::Equal, x, y);
                self.set_cr_field(bf, lt, gt, eq);
            }
            // rlwinm
            21 => {
                let sh = (inst >> 11) & 31;
                let mb = (inst >> 6) & 31;
                let me = (inst >> 1) & 31;
                let s = self.get(rd);
                let r = self.rot32(s, sh);
                let r = self.b.ins().band_imm(r, mask64(mb + 32, me + 32) as i64);
                self.set(ra, r);
                if rc {
                    self.set_cr0(r);
                }
            }
            // rlwimi
            20 => {
                let sh = (inst >> 11) & 31;
                let m = mask64(((inst >> 6) & 31) + 32, ((inst >> 1) & 31) + 32);
                let s = self.get(rd);
                let a = self.get(ra);
                let r = self.rot32(s, sh);
                let ins = self.b.ins().band_imm(r, m as i64);
                let keep = self.b.ins().band_imm(a, !m as i64);
                let out = self.b.ins().bor(ins, keep);
                self.set(ra, out);
                if rc {
                    self.set_cr0(out);
                }
            }
            // rlwnm
            23 => {
                let s = self.get(rd);
                let n = self.get(rb);
                let n = self.b.ins().band_imm(n, 31);
                let lo = self.b.ins().band_imm(s, 0xFFFF_FFFF);
                let hi = self.b.ins().ishl_imm(lo, 32);
                let dup = self.b.ins().bor(lo, hi);
                let r = self.b.ins().rotl(dup, n);
                let m = mask64(((inst >> 6) & 31) + 32, ((inst >> 1) & 31) + 32);
                let r = self.b.ins().band_imm(r, m as i64);
                self.set(ra, r);
                if rc {
                    self.set_cr0(r);
                }
            }
            // rldicl / rldicr / rldic
            30 if matches!((inst >> 2) & 7, 0..=2) => {
                let raw = (inst >> 5) & 0x3F;
                let mbme = ((raw & 1) << 5) | (raw >> 1);
                let sh = ((inst >> 11) & 31) | (((inst >> 1) & 1) << 5);
                let s = self.get(rd);
                let rot = self.b.ins().rotl_imm(s, sh as i64);
                let m = match (inst >> 2) & 7 {
                    0 => mask64(mbme, 63),
                    1 => mask64(0, mbme),
                    _ => mask64(mbme, 63 - sh),
                };
                let r = self.b.ins().band_imm(rot, m as i64);
                self.set(ra, r);
                if rc {
                    self.set_cr0(r);
                }
            }
            // D-form loads and stores
            32..=45 => {
                let (size, signed, update, store) = match opcd {
                    32 => (4, false, false, false),
                    33 => (4, false, true, false),
                    34 => (1, false, false, false),
                    35 => (1, false, true, false),
                    36 => (4, false, false, true),
                    37 => (4, false, true, true),
                    38 => (1, false, false, true),
                    39 => (1, false, true, true),
                    40 => (2, false, false, false),
                    41 => (2, false, true, false),
                    42 => (2, true, false, false),
                    43 => (2, true, true, false),
                    44 => (2, false, false, true),
                    _ => (2, false, true, true),
                };
                let b = self.base(ra);
                let ea = self.b.ins().iadd_imm(b, simm);
                self.mem_op(size, signed, update, store, ea, rd, ra);
            }
            // ld / ldu / lwa
            58 => {
                let (size, signed, update) = match inst & 3 {
                    0 => (8, false, false),
                    1 => (8, false, true),
                    2 => (4, true, false),
                    _ => return false,
                };
                let b = self.base(ra);
                let ea = self.b.ins().iadd_imm(b, (simm & !3) as i64);
                self.mem_op(size, signed, update, false, ea, rd, ra);
            }
            // std / stdu
            62 => {
                let update = match inst & 3 {
                    0 => false,
                    1 => true,
                    _ => return false,
                };
                let b = self.base(ra);
                let ea = self.b.ins().iadd_imm(b, (simm & !3) as i64);
                self.mem_op(8, false, update, true, ea, rd, ra);
            }
            31 => return self.native_31(inst, rd, ra, rb, rc),
            _ => return false,
        }
        true
    }

    fn rot32(&mut self, s: Value, sh: u32) -> Value {
        let lo = self.b.ins().band_imm(s, 0xFFFF_FFFF);
        let hi = self.b.ins().ishl_imm(lo, 32);
        let dup = self.b.ins().bor(lo, hi);
        self.b.ins().rotl_imm(dup, sh as i64)
    }

    #[allow(clippy::too_many_arguments)]
    fn mem_op(&mut self, size: u32, signed: bool, update: bool, store: bool, ea: Value, rt: usize, ra: usize) {
        if store {
            let v = self.get(rt);
            self.write(size, ea, v);
        } else {
            let mut v = self.read(size, ea);
            if signed {
                v = match size {
                    2 => {
                        let t = self.b.ins().ireduce(types::I16, v);
                        self.b.ins().sextend(types::I64, t)
                    }
                    4 => {
                        let t = self.b.ins().ireduce(types::I32, v);
                        self.b.ins().sextend(types::I64, t)
                    }
                    _ => v,
                };
            }
            self.set(rt, v);
        }
        if update {
            self.set(ra, ea);
        }
    }

    fn native_31(&mut self, inst: u32, rd: usize, ra: usize, rb: usize, rc: bool) -> bool {
        let xo = (inst >> 1) & 0x3FF;
        // logical ops: destination ra, sources rs (rd field) and rb
        let logical2 = |me: &mut Self, f: &dyn Fn(&mut FunctionBuilder, Value, Value) -> Value| {
            let s = me.get(rd);
            let b = me.get(rb);
            let r = f(me.b, s, b);
            me.set(ra, r);
            if rc {
                me.set_cr0(r);
            }
        };
        match xo {
            444 => logical2(self, &|b, s, x| b.ins().bor(s, x)),
            28 => logical2(self, &|b, s, x| b.ins().band(s, x)),
            316 => logical2(self, &|b, s, x| b.ins().bxor(s, x)),
            60 => logical2(self, &|b, s, x| b.ins().band_not(s, x)),
            412 => logical2(self, &|b, s, x| b.ins().bor_not(s, x)),
            124 => logical2(self, &|b, s, x| {
                let o = b.ins().bor(s, x);
                b.ins().bnot(o)
            }),
            476 => logical2(self, &|b, s, x| {
                let o = b.ins().band(s, x);
                b.ins().bnot(o)
            }),
            284 => logical2(self, &|b, s, x| {
                let o = b.ins().bxor(s, x);
                b.ins().bnot(o)
            }),
            // sign extension
            954 | 922 | 986 => {
                let s = self.get(rd);
                let (ty, _) = match xo {
                    954 => (types::I8, 0),
                    922 => (types::I16, 0),
                    _ => (types::I32, 0),
                };
                let t = self.b.ins().ireduce(ty, s);
                let r = self.b.ins().sextend(types::I64, t);
                self.set(ra, r);
                if rc {
                    self.set_cr0(r);
                }
            }
            // add / subf / neg (OE = 0 only)
            266 | 40 | 104 => {
                let a = self.get(ra);
                let r = match xo {
                    266 => {
                        let b = self.get(rb);
                        self.b.ins().iadd(a, b)
                    }
                    40 => {
                        let b = self.get(rb);
                        self.b.ins().isub(b, a)
                    }
                    _ => self.b.ins().ineg(a),
                };
                self.set(rd, r);
                if rc {
                    self.set_cr0(r);
                }
            }
            // mullw (OE = 0): 32x32 -> 64 signed
            235 => {
                let a = self.get(ra);
                let b = self.get(rb);
                let a32 = self.b.ins().ireduce(types::I32, a);
                let b32 = self.b.ins().ireduce(types::I32, b);
                let x = self.b.ins().sextend(types::I64, a32);
                let y = self.b.ins().sextend(types::I64, b32);
                let r = self.b.ins().imul(x, y);
                self.set(rd, r);
                if rc {
                    self.set_cr0(r);
                }
            }
            // cmp / cmpl
            0 | 32 => {
                let bf = (inst >> 23) & 7;
                let l64 = (inst >> 21) & 1 != 0;
                let a = self.get(ra);
                let b = self.get(rb);
                let signed = xo == 0;
                let conv = |me: &mut Self, v: Value| {
                    if l64 {
                        v
                    } else {
                        let t = me.b.ins().ireduce(types::I32, v);
                        if signed { me.b.ins().sextend(types::I64, t) } else { me.b.ins().uextend(types::I64, t) }
                    }
                };
                let (x, y) = (conv(self, a), conv(self, b));
                let (clt, cgt) = if signed { (IntCC::SignedLessThan, IntCC::SignedGreaterThan) } else { (IntCC::UnsignedLessThan, IntCC::UnsignedGreaterThan) };
                let lt = self.b.ins().icmp(clt, x, y);
                let gt = self.b.ins().icmp(cgt, x, y);
                let eq = self.b.ins().icmp(IntCC::Equal, x, y);
                self.set_cr_field(bf, lt, gt, eq);
            }
            // mfspr / mtspr for XER, LR, CTR
            339 | 467 => {
                let spr = ((inst >> 16) & 31) | (((inst >> 11) & 31) << 5);
                let reg = match spr {
                    1 => Reg::Xer,
                    8 => Reg::Lr,
                    9 => Reg::Ctr,
                    _ => return false,
                };
                if xo == 339 {
                    let v = self.get_spr(reg);
                    self.set(rd, v);
                } else {
                    let v = self.get(rd);
                    self.set_spr(reg, v);
                }
            }
            // indexed loads / stores (no byte reversal)
            23 | 55 | 87 | 119 | 279 | 311 | 343 | 375 | 341 | 373 | 21 | 53 | 151 | 183 | 215 | 247 | 407 | 439 | 149 | 181 => {
                let (size, signed, update, store) = match xo {
                    23 => (4, false, false, false),
                    55 => (4, false, true, false),
                    87 => (1, false, false, false),
                    119 => (1, false, true, false),
                    279 => (2, false, false, false),
                    311 => (2, false, true, false),
                    343 => (2, true, false, false),
                    375 => (2, true, true, false),
                    341 => (4, true, false, false),
                    373 => (4, true, true, false),
                    21 => (8, false, false, false),
                    53 => (8, false, true, false),
                    151 => (4, false, false, true),
                    183 => (4, false, true, true),
                    215 => (1, false, false, true),
                    247 => (1, false, true, true),
                    407 => (2, false, false, true),
                    439 => (2, false, true, true),
                    149 => (8, false, false, true),
                    _ => (8, false, true, true),
                };
                let b = self.base(ra);
                let x = self.get(rb);
                let ea = self.b.ins().iadd(b, x);
                self.mem_op(size, signed, update, store, ea, rd, ra);
            }
            _ => return false,
        }
        true
    }

    /// Translate a branch (always the last instruction of a block). Returns false if not a branch we handle.
    fn branch(&mut self, pc: u32, inst: u32) -> bool {
        let opcd = inst >> 26;
        let next_seq = pc.wrapping_add(4) as i64;
        match opcd {
            18 => {
                let li = (((inst & 0x03FF_FFFC) << 6) as i32 >> 6) as i64;
                let target = if inst & 2 != 0 { li as u64 & 0xFFFF_FFFF } else { (pc as i64).wrapping_add(li) as u64 & 0xFFFF_FFFF };
                if inst & 1 != 0 {
                    let l = self.iconst(next_seq);
                    self.set_spr(Reg::Lr, l);
                }
                let t = self.iconst(target as i64);
                self.finish(t);
                true
            }
            16 => {
                let bo = (inst >> 21) & 31;
                let bi = (inst >> 16) & 31;
                let bd = (inst & 0xFFFC) as i16 as i64;
                let target = if inst & 2 != 0 { bd as u64 & 0xFFFF_FFFF } else { (pc as i64).wrapping_add(bd) as u64 & 0xFFFF_FFFF };
                let take = self.cond(bo, bi, true);
                if inst & 1 != 0 {
                    let l = self.iconst(next_seq);
                    self.set_spr(Reg::Lr, l);
                }
                let t = self.iconst(target as i64);
                let f = self.iconst(next_seq);
                let next = match take {
                    Some(c) => self.b.ins().select(c, t, f),
                    None => t,
                };
                self.finish(next);
                true
            }
            19 => {
                let xo = (inst >> 1) & 0x3FF;
                if xo != 16 && xo != 528 {
                    return false;
                }
                let bo = (inst >> 21) & 31;
                let bi = (inst >> 16) & 31;
                // target register is read before LR is possibly updated by LK
                let tgt = self.get_spr(if xo == 16 { Reg::Lr } else { Reg::Ctr });
                let tgt = self.b.ins().band_imm(tgt, !3i64 & 0xFFFF_FFFF);
                let take = self.cond(bo, bi, xo == 16);
                if inst & 1 != 0 {
                    let l = self.iconst(next_seq);
                    self.set_spr(Reg::Lr, l);
                }
                let f = self.iconst(next_seq);
                let next = match take {
                    Some(c) => self.b.ins().select(c, tgt, f),
                    None => tgt,
                };
                self.finish(next);
                true
            }
            _ => false,
        }
    }

    /// Branch condition for BO/BI: None = always taken. `dec_ctr` = this form decrements CTR (bc, bclr).
    fn cond(&mut self, bo: u32, bi: u32, dec_ctr: bool) -> Option<Value> {
        let mut result: Option<Value> = None;
        if dec_ctr && bo & 0b00100 == 0 {
            let c = self.get_spr(Reg::Ctr);
            let c = self.b.ins().iadd_imm(c, -1);
            self.set_spr(Reg::Ctr, c);
        }
        if dec_ctr && bo & 0b00100 == 0 {
            let c = self.get_spr(Reg::Ctr);
            let zero = self.iconst(0);
            let cc = if bo & 0b00010 != 0 { IntCC::Equal } else { IntCC::NotEqual };
            result = Some(self.b.ins().icmp(cc, c, zero));
        }
        if bo & 0b10000 == 0 {
            let cr = self.get_cr();
            let bit = self.b.ins().ushr_imm(cr, 31 - bi as i64);
            let bit = self.b.ins().band_imm(bit, 1);
            let want = (bo & 0b01000 != 0) as i64;
            let c = self.b.ins().icmp_imm(IntCC::Equal, bit, want);
            result = Some(match result {
                Some(r) => self.b.ins().band(r, c),
                None => c,
            });
        }
        result
    }
}

impl<B: Bus + 'static> Jit<B> {
    /// `code_start..code_end` is the address range blocks may be formed in (the executable image).
    pub fn new(code_start: u32, code_end: u32) -> Self {
        let mut flags = settings::builder();
        // Blocks are short and compiled often: cheap compile beats extra optimisation, and the verifier
        // only helps while developing (X360_JIT_VERIFY=1 turns it on).
        flags.set("opt_level", &std::env::var("X360_JIT_OPT").unwrap_or_else(|_| "none".into())).unwrap();
        flags.set("enable_verifier", if std::env::var_os("X360_JIT_VERIFY").is_some() { "true" } else { "false" }).unwrap();
        flags.set("is_pic", "false").unwrap();
        let isa = cranelift_native::builder().expect("host isa").finish(settings::Flags::new(flags)).expect("isa");
        let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
        jb.symbol("h_exec", h_exec::<B> as *const u8);
        jb.symbol("h_r8", h_r8::<B> as *const u8);
        jb.symbol("h_r16", h_r16::<B> as *const u8);
        jb.symbol("h_r32", h_r32::<B> as *const u8);
        jb.symbol("h_r64", h_r64::<B> as *const u8);
        jb.symbol("h_w8", h_w8::<B> as *const u8);
        jb.symbol("h_w16", h_w16::<B> as *const u8);
        jb.symbol("h_w32", h_w32::<B> as *const u8);
        jb.symbol("h_w64", h_w64::<B> as *const u8);
        let module = JITModule::new(jb);
        let n = ((code_end - code_start) >> 2) as usize;
        Self {
            module,
            ctx: Context::new(),
            fb: FunctionBuilderContext::new(),
            slots: vec![0; n],
            code_start,
            code_end,
            counter: 0,
            break_pcs: Default::default(),
            blocks_compiled: 0,
            instrs_compiled: 0,
            native_instrs: 0,
            compile_time: std::time::Duration::ZERO,
            _bus: PhantomData,
        }
    }

    pub fn add_break(&mut self, pc: u32) {
        self.break_pcs.insert(pc);
    }

    pub fn summary(&self) -> String {
        format!(
            "jit: {} blocks, {} instructions ({:.0}% native), {:.2}s compiling",
            self.blocks_compiled,
            self.instrs_compiled,
            if self.instrs_compiled == 0 { 0.0 } else { self.native_instrs as f64 * 100.0 / self.instrs_compiled as f64 },
            self.compile_time.as_secs_f64()
        )
    }

    /// Run the block at `cpu.pc` if it is (or has just become) hot. The caller interprets otherwise.
    /// Call only at block entries (after a taken branch), where the counting is meaningful.
    pub fn run(&mut self, cpu: &mut Cpu, bus: &mut B) -> Outcome {
        let pc = cpu.pc as u32;
        if pc < self.code_start || pc >= self.code_end || pc & 3 != 0 {
            return Outcome::Interpret;
        }
        let idx = ((pc - self.code_start) >> 2) as usize;
        let v = self.slots[idx];
        let f: BlockFn = if v == NEVER {
            return Outcome::Interpret;
        } else if v > 255 {
            unsafe { std::mem::transmute::<u64, BlockFn>(v) }
        } else {
            if v + 1 < HEAT_THRESHOLD as u64 {
                self.slots[idx] = v + 1;
                return Outcome::Interpret;
            }
            match self.compile(pc, bus) {
                Some(f) => {
                    self.slots[idx] = f as usize as u64;
                    f
                }
                None => {
                    self.slots[idx] = NEVER;
                    return Outcome::Interpret;
                }
            }
        };
        let table = bus.page_table().unwrap_or(std::ptr::null());
        let r = unsafe { f(cpu as *mut Cpu, bus as *mut B as *mut u8, table) };
        if r & TRAP_FLAG != 0 {
            let trap = cpu.pending_trap.take().expect("jit trap flag without a trap");
            Outcome::Trapped(r & !TRAP_FLAG, trap)
        } else {
            Outcome::Ran(r)
        }
    }

    fn compile(&mut self, start: u32, bus: &mut B) -> Option<BlockFn> {
        let t0 = std::time::Instant::now();
        let mut sig = Signature::new(self.module.isa().default_call_conv());
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        self.counter += 1;
        let name = format!("blk_{start:08x}_{}", self.counter);
        let id = self.module.declare_function(&name, Linkage::Export, &sig).ok()?;
        self.ctx.func.signature = sig;

        // collect the block's instructions first (end at a branch, the range end, or MAX_BLOCK)
        let mut insts: Vec<(u32, u32)> = Vec::new();
        let mut pc = start;
        loop {
            if pc >= self.code_end || insts.len() as u32 >= MAX_BLOCK || (pc != start && self.break_pcs.contains(&pc)) {
                break;
            }
            let inst = bus.read32(pc);
            insts.push((pc, inst));
            pc += 4;
            if is_branch(inst) {
                break;
            }
        }
        if insts.is_empty() {
            self.module.clear_context(&mut self.ctx);
            return None;
        }

        let mut native = 0u32;
        let fcfg = self.module.isa().frontend_config();
        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fb);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);
            let cpu = b.block_params(entry)[0];
            let busv = b.block_params(entry)[1];
            let tablev = b.block_params(entry)[2];

            let mut mk = |name: &str, params: &[types::Type], ret: Option<types::Type>, module: &mut JITModule, b: &mut FunctionBuilder| {
                let mut s = Signature::new(module.isa().default_call_conv());
                for p in params {
                    s.params.push(AbiParam::new(*p));
                }
                if let Some(r) = ret {
                    s.returns.push(AbiParam::new(r));
                }
                let fid = module.declare_function(name, Linkage::Import, &s).unwrap();
                module.declare_func_in_func(fid, b.func)
            };
            let (i64t, i32t) = (types::I64, types::I32);
            let h = HelperRefs {
                exec: mk("h_exec", &[i64t, i64t, i64t, i32t], Some(i32t), &mut self.module, &mut b),
                r8: mk("h_r8", &[i64t, i32t], Some(i32t), &mut self.module, &mut b),
                r16: mk("h_r16", &[i64t, i32t], Some(i32t), &mut self.module, &mut b),
                r32: mk("h_r32", &[i64t, i32t], Some(i32t), &mut self.module, &mut b),
                r64: mk("h_r64", &[i64t, i32t], Some(i64t), &mut self.module, &mut b),
                w8: mk("h_w8", &[i64t, i32t, i32t], None, &mut self.module, &mut b),
                w16: mk("h_w16", &[i64t, i32t, i32t], None, &mut self.module, &mut b),
                w32: mk("h_w32", &[i64t, i32t, i32t], None, &mut self.module, &mut b),
                w64: mk("h_w64", &[i64t, i32t, i64t], None, &mut self.module, &mut b),
            };
            let cpi = b.ins().load(types::I64, MemFlagsData::trusted(), cpu, O_CPI);
            let mut em = Em {
                b: &mut b,
                cpu,
                bus: busv,
                h,
                cpi,
                table: if bus.page_table().is_some() { Some(tablev) } else { None },
                gpr: [None; 32],
                gpr_dirty: [false; 32],
                cr: None,
                cr_dirty: false,
                spr: [None; 3],
                spr_dirty: [false; 3],
                pending: 0,
                count: 0,
                native: 0,
            };
            let n = insts.len();
            let mut terminated = false;
            for (k, &(pc, inst)) in insts.iter().enumerate() {
                em.count += 1;
                let last = k + 1 == n;
                if last && is_branch(inst) {
                    em.pending += 1;
                    if em.branch(pc, inst) {
                        em.native += 1;
                        terminated = true;
                        break;
                    }
                    em.pending -= 1;
                    // a branch form we don't translate (e.g. a CR logical op shares opcode 19):
                    // interpret it, then continue to the sequential/target pc the interpreter chose
                    em.helper(pc, inst);
                    let next = em.load(O_PC, types::I64);
                    em.finish_with_loaded_pc(next);
                    terminated = true;
                    break;
                }
                em.pending += 1;
                if em.native(pc, inst) {
                    em.native += 1;
                } else {
                    em.pending -= 1;
                    em.helper(pc, inst);
                }
            }
            if !terminated {
                let next = em.iconst(insts.last().unwrap().0.wrapping_add(4) as i64);
                em.finish(next);
            }
            native = em.native;
            b.finalize(fcfg);
        }

        if self.module.define_function(id, &mut self.ctx).is_err() {
            self.module.clear_context(&mut self.ctx);
            return None;
        }
        self.module.clear_context(&mut self.ctx);
        if self.module.finalize_definitions().is_err() {
            return None;
        }
        let ptr = self.module.get_finalized_function(id);
        self.blocks_compiled += 1;
        self.instrs_compiled += insts.len() as u64;
        self.native_instrs += native as u64;
        self.compile_time += t0.elapsed();
        Some(unsafe { std::mem::transmute::<*const u8, BlockFn>(ptr) })
    }
}

impl<'a, 'b> Em<'a, 'b> {
    /// Exit after a helper-executed branch: state is already flushed, pc already stored.
    fn finish_with_loaded_pc(&mut self, _pc: Value) {
        self.sync_cycles();
        self.ret(self.count as u64);
    }
}

/// Is `inst` a control transfer that ends a block (b, bc, bclr/bcctr, sc)?
fn is_branch(inst: u32) -> bool {
    match inst >> 26 {
        16 | 18 => true,
        17 => true,
        19 => matches!((inst >> 1) & 0x3FF, 16 | 528),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::FlatMemory;

    const BASE: u32 = 0x1000;
    const DATA: u32 = 0x4000;

    /// xorshift PRNG so the tests are deterministic
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn r32(&mut self) -> u32 {
            self.next() as u32
        }
    }

    fn random_cpu(rng: &mut Rng) -> Cpu {
        let mut c = Cpu::new();
        c.cpi = 3;
        for i in 0..32 {
            c.gpr[i] = match rng.next() % 4 {
                0 => rng.next(),
                1 => rng.next() as u32 as u64,
                2 => (rng.next() % 64) as u64,
                _ => (rng.next() as i32) as i64 as u64,
            };
        }
        // keep a pointer-ish base register in the data region for loads/stores
        for i in [3usize, 4, 5, 6] {
            c.gpr[i] = (DATA + 0x100 + (rng.r32() % 0x200)) as u64;
        }
        c.cr = rng.r32();
        c.lr = (rng.r32() & !3) as u64;
        c.ctr = rng.next() % 8;
        c.xer = (rng.r32() & 0xE000_0000) as u64;
        c.pc = BASE as u64;
        c
    }

    fn fresh_mem(rng: &mut Rng, inst: u32) -> FlatMemory {
        let mut m = FlatMemory::new(0x8000);
        for a in (DATA..DATA + 0x800).step_by(4) {
            m.write32(a, rng.r32());
        }
        m.write32(BASE, inst);
        m.write32(BASE + 4, 0x4800_0008); // b +8: ends the block with a taken branch
        m
    }

    fn same(a: &Cpu, b: &Cpu, what: &str) {
        assert_eq!(a.gpr, b.gpr, "gpr {what}");
        assert_eq!(a.cr, b.cr, "cr {what}");
        assert_eq!(a.lr, b.lr, "lr {what}");
        assert_eq!(a.ctr, b.ctr, "ctr {what}");
        assert_eq!(a.xer, b.xer, "xer {what}");
        assert_eq!(a.pc, b.pc, "pc {what}");
        assert_eq!(a.cycles, b.cycles, "cycles {what}");
    }

    /// Run `inst` (+ a terminating branch) through the interpreter and the JIT; everything must match.
    fn check(inst: u32, rng: &mut Rng) {
        let seed = rng.r32() as u64 | 1;
        let mut r1 = Rng(seed);
        let mut r2 = Rng(seed);
        let (mut c1, mut m1) = (random_cpu(&mut r1), fresh_mem(&mut r1, inst));
        let (mut c2, mut m2) = (random_cpu(&mut r2), fresh_mem(&mut r2, inst));
        let t1 = (c1.step(&mut m1).and_then(|_| c1.step(&mut m1))).err();
        let mut jit = Jit::<FlatMemory>::new(BASE, BASE + 0x100);
        warm(&mut jit, &m2);
        let mut t2 = None;
        match jit.run(&mut c2, &mut m2) {
            Outcome::Ran(n) => assert_eq!(n, 2, "block length for {inst:08x}"),
            Outcome::Trapped(_, t) => t2 = Some(t),
            Outcome::Interpret => panic!("jit refused {inst:08x}"),
        }
        assert_eq!(t1.is_some(), t2.is_some(), "trap mismatch for {inst:08x}");
        if t1.is_none() {
            same(&c1, &c2, &format!("{inst:08x}"));
        }
        assert_eq!(m1.data[DATA as usize..DATA as usize + 0x800], m2.data[DATA as usize..DATA as usize + 0x800], "memory {inst:08x}");
    }

    /// Make the block at BASE hot (so it gets compiled) by entering it on scratch state.
    fn warm(jit: &mut Jit<FlatMemory>, mem: &FlatMemory) {
        for _ in 0..HEAT_THRESHOLD {
            let mut sc = Cpu::new();
            sc.pc = BASE as u64;
            sc.cpi = 3;
            let mut sm = FlatMemory { data: mem.data.clone() };
            let _ = jit.run(&mut sc, &mut sm);
        }
    }

    fn d_form(op: u32, rt: u32, ra: u32, imm: u32) -> u32 {
        op << 26 | rt << 21 | ra << 16 | (imm & 0xFFFF)
    }
    fn x_form(op: u32, rt: u32, ra: u32, rb: u32, xo: u32, rc: u32) -> u32 {
        op << 26 | rt << 21 | ra << 16 | rb << 11 | xo << 1 | rc
    }

    #[test]
    fn differential_random_instructions() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut count = 0;
        // D-form ALU / compares / loads / stores
        for op in [14u32, 15, 24, 25, 26, 27, 28, 29, 10, 11, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45] {
            for _ in 0..60 {
                let rt = rng.r32() % 32;
                let ra = rng.r32() % 32;
                let mut imm = rng.r32() & 0xFFFF;
                // loads/stores: small offsets from the pointer-ish registers 3..6
                let ra = if (32..=45).contains(&op) { 3 + rng.r32() % 4 } else { ra };
                if (32..=45).contains(&op) {
                    imm = (rng.r32() % 0x80) & !0x0;
                }
                // update forms must not use ra == rt (invalid); avoid ra == 0 for update forms
                let rt = if matches!(op, 33 | 35 | 37 | 39 | 41 | 43 | 45) && rt == ra { (rt + 1) % 32 } else { rt };
                let inst = if op == 10 || op == 11 { d_form(op, (rng.r32() % 8) << 2 | (rng.r32() % 2), ra, imm) } else { d_form(op, rt, ra, imm) };
                check(inst, &mut rng);
                count += 1;
            }
        }
        // ld/ldu/lwa/std/stdu (DS-form)
        for (op, low) in [(58u32, 0u32), (58, 1), (58, 2), (62, 0), (62, 1)] {
            for _ in 0..40 {
                let ra = 3 + rng.r32() % 4;
                let rt = (rng.r32() % 32).max(7);
                let ds = (rng.r32() % 0x40) << 2;
                check(op << 26 | rt << 21 | ra << 16 | ds | low, &mut rng);
                count += 1;
            }
        }
        // X-form ALU
        for xo in [444u32, 28, 316, 60, 412, 124, 476, 284, 954, 922, 986, 266, 40, 104, 235] {
            for _ in 0..60 {
                let (rt, ra, rb) = (rng.r32() % 32, rng.r32() % 32, rng.r32() % 32);
                let rc = rng.r32() % 2;
                check(x_form(31, rt, ra, rb, xo, rc), &mut rng);
                count += 1;
            }
        }
        // cmp / cmpl
        for xo in [0u32, 32] {
            for _ in 0..60 {
                let (ra, rb) = (rng.r32() % 32, rng.r32() % 32);
                let bf = (rng.r32() % 8) << 2 | (rng.r32() % 2);
                check(x_form(31, bf, ra, rb, xo, 0), &mut rng);
                count += 1;
            }
        }
        // indexed loads/stores
        for xo in [23u32, 55, 87, 119, 279, 311, 343, 375, 341, 373, 21, 53, 151, 183, 215, 247, 407, 439, 149, 181] {
            for _ in 0..30 {
                let ra = 3 + rng.r32() % 2;
                let rb = 5 + rng.r32() % 2;
                let mut rt = (rng.r32() % 32).max(7);
                if rt == ra || rt == rb {
                    rt = 12;
                }
                let inst = x_form(31, rt, ra, rb, xo, 0);
                let mut rr = Rng(rng.next());
                // make rb a small offset so ea stays in the data window
                let _ = &mut rr;
                check(inst, &mut rng);
                count += 1;
            }
        }
        // rotates
        for _ in 0..400 {
            let (rs, ra, sh, mb, me, rc) = (rng.r32() % 32, rng.r32() % 32, rng.r32() % 32, rng.r32() % 32, rng.r32() % 32, rng.r32() % 2);
            check(21 << 26 | rs << 21 | ra << 16 | sh << 11 | mb << 6 | me << 1 | rc, &mut rng);
            check(20 << 26 | rs << 21 | ra << 16 | sh << 11 | mb << 6 | me << 1 | rc, &mut rng);
            let rb = rng.r32() % 32;
            check(23 << 26 | rs << 21 | ra << 16 | rb << 11 | mb << 6 | me << 1 | rc, &mut rng);
            let sh = rng.r32() % 64;
            let mbme = rng.r32() % 64;
            for sub in 0..3u32 {
                let raw = ((mbme & 31) << 1) | (mbme >> 5);
                check(30 << 26 | rs << 21 | ra << 16 | (sh & 31) << 11 | raw << 5 | sub << 2 | ((sh >> 5) << 1) | rc, &mut rng);
            }
            count += 6;
        }
        // mfspr / mtspr
        for spr in [1u32, 8, 9] {
            for _ in 0..20 {
                let rt = rng.r32() % 32;
                let f = (spr & 31) << 16 | (spr >> 5) << 11;
                check(31 << 26 | rt << 21 | f | 339 << 1, &mut rng);
                check(31 << 26 | rt << 21 | f | 467 << 1, &mut rng);
                count += 2;
            }
        }
        assert!(count > 2000);
    }

    #[test]
    fn unsupported_instruction_falls_back_to_the_interpreter() {
        // sraw (xo 792) isn't translated natively; the helper path must give identical results
        let mut rng = Rng(77);
        for _ in 0..200 {
            let (rs, ra, rb) = (rng.r32() % 32, rng.r32() % 32, rng.r32() % 32);
            check(x_form(31, rs, ra, rb, 792, rng.r32() % 2), &mut rng);
            check(x_form(31, rs, ra, rb, 26, 0), &mut rng); // cntlzw
        }
    }

    #[test]
    fn branches_match() {
        let mut rng = Rng(99);
        for _ in 0..600 {
            let bo = rng.r32() % 32;
            let bi = rng.r32() % 32;
            let bd = ((rng.r32() % 0x40) << 2) as i32 - 0x40;
            let lk = rng.r32() % 2;
            // bc forward/back within the code range; AA = 0
            let inst = 16 << 26 | bo << 21 | bi << 16 | ((bd as u32) & 0xFFFC) | lk;
            check_branch(inst, &mut rng);
            // bclr / bcctr
            for xo in [16u32, 528] {
                let inst = 19 << 26 | bo << 21 | bi << 16 | xo << 1 | lk;
                check_branch(inst, &mut rng);
            }
            // b / bl
            let li = ((rng.r32() % 0x100) << 2) as i32 - 0x80;
            check_branch(18 << 26 | ((li as u32) & 0x03FF_FFFC) | lk, &mut rng);
        }
    }

    /// Like `check`, but the instruction under test is itself the terminator (single-instruction block).
    fn check_branch(inst: u32, rng: &mut Rng) {
        let seed = rng.r32() as u64 | 1;
        let mut r1 = Rng(seed);
        let mut r2 = Rng(seed);
        let (mut c1, mut m1) = (random_cpu(&mut r1), fresh_mem(&mut r1, inst));
        let (mut c2, mut m2) = (random_cpu(&mut r2), fresh_mem(&mut r2, inst));
        // when the interpreter's pc ends up outside the test range the comparison is still valid
        c1.step(&mut m1).unwrap();
        let mut jit = Jit::<FlatMemory>::new(BASE, BASE + 0x100);
        warm(&mut jit, &m2);
        match jit.run(&mut c2, &mut m2) {
            Outcome::Ran(n) => assert_eq!(n, 1),
            _ => panic!("expected a run of {inst:08x}"),
        }
        same(&c1, &c2, &format!("branch {inst:08x}"));
    }

    /// 4 KB-paged memory with a page table, so the JIT takes its inline fast path.
    struct PagedBus {
        pages: Vec<Option<Box<[u8; 4096]>>>,
    }
    impl PagedBus {
        fn new(mapped: &[usize]) -> Self {
            let mut pages: Vec<Option<Box<[u8; 4096]>>> = (0..1 << 20).map(|_| None).collect();
            for &p in mapped {
                pages[p] = Some(Box::new([0; 4096]));
            }
            Self { pages }
        }
        fn snapshot(&self) -> Vec<u8> {
            (0..8).flat_map(|p| self.pages[p].as_ref().map(|b| b.to_vec()).unwrap_or_default()).collect()
        }
    }
    impl Bus for PagedBus {
        fn read8(&self, a: u32) -> u8 {
            self.pages[(a >> 12) as usize].as_ref().map(|p| p[(a & 0xFFF) as usize]).unwrap_or(0)
        }
        fn write8(&mut self, a: u32, v: u8) {
            if let Some(p) = &mut self.pages[(a >> 12) as usize] {
                p[(a & 0xFFF) as usize] = v;
            }
        }
        fn page_table(&self) -> Option<*const *mut u8> {
            Some(self.pages.as_ptr() as *const *mut u8)
        }
    }

    #[test]
    fn inline_memory_path_matches_including_page_edges_and_unmapped_pages() {
        // pages 0..=5 mapped (code at 0x1000), page 6 and 7 unmapped
        let mut rng = Rng(4242);
        let ops: Vec<(u32, bool)> = vec![
            (32, false), (33, false), (34, false), (35, false), (36, false), (37, false), (38, false), (39, false),
            (40, false), (41, false), (42, false), (43, false), (44, false), (45, false),
        ];
        let mut cases = 0;
        for round in 0..6 {
            for &(op, _) in &ops {
                let bases = [0x1FF9u32, 0x1FFC, 0x1FFE, 0x1FFF, 0x2000, 0x2400, 0x5FFD, 0x5FFF, 0x6000, 0x6002, 0x7000];
                let base = bases[(rng.r32() as usize) % bases.len()];
                let off = (rng.r32() % 4) as i16 - 1;
                let ra = 3 + rng.r32() % 4;
                let mut rt = (rng.r32() % 32).max(8);
                if rt == ra {
                    rt += 1;
                }
                let inst = d_form(op, rt, ra, off as u32);
                let seed = rng.r32() as u64 | 1 | ((round as u64) << 40);
                paged_check(inst, ra as usize, base, seed);
                cases += 1;
            }
            // doubleword and indexed forms
            for (op, low) in [(58u32, 0u32), (58, 2), (62, 0)] {
                let base = [0x1FF8u32, 0x1FFC, 0x2000, 0x5FFC, 0x6000][(rng.r32() as usize) % 5];
                let inst = op << 26 | 9 << 21 | 3 << 16 | low;
                paged_check(inst, 3, base, rng.r32() as u64 | 1);
                cases += 1;
            }
            for xo in [23u32, 87, 279, 343, 151, 215, 407, 21, 149] {
                let base = [0x1FFDu32, 0x1FFF, 0x2000, 0x5FFE, 0x6001][(rng.r32() as usize) % 5];
                let inst = x_form(31, 9, 3, 12, xo, 0);
                paged_check(inst, 3, base, rng.r32() as u64 | 1);
                cases += 1;
            }
        }
        assert!(cases > 1000);
    }

    fn paged_check(inst: u32, base_reg: usize, base: u32, seed: u64) {
        let build = |seed: u64| {
            let mut r = Rng(seed);
            let mut bus = PagedBus::new(&[0, 1, 2, 3, 4, 5]);
            for a in (0x2000u32..0x2800).step_by(4) {
                bus.write32(a, r.r32());
            }
            for a in (0x5800u32..0x6000).step_by(4) {
                bus.write32(a, r.r32());
            }
            bus.write32(BASE, inst);
            bus.write32(BASE + 4, 0x4800_0008);
            let mut c = random_cpu(&mut r);
            c.gpr[base_reg] = base as u64;
            c.gpr[12] = 0; // indexed forms: rb = 0
            (c, bus)
        };
        let (mut c1, mut m1) = build(seed);
        let (mut c2, mut m2) = build(seed);
        c1.step(&mut m1).unwrap();
        c1.step(&mut m1).unwrap();
        let mut jit = Jit::<PagedBus>::new(BASE, BASE + 0x100);
        for _ in 0..HEAT_THRESHOLD {
            let mut sc = Cpu::new();
            sc.pc = BASE as u64;
            sc.cpi = 3;
            let mut sm = PagedBus::new(&[0, 1, 2, 3, 4, 5]);
            sm.write32(BASE, inst);
            sm.write32(BASE + 4, 0x4800_0008);
            let _ = jit.run(&mut sc, &mut sm);
        }
        match jit.run(&mut c2, &mut m2) {
            Outcome::Ran(2) => {}
            _ => panic!("jit did not run {inst:08x}"),
        }
        same(&c1, &c2, &format!("paged {inst:08x} base {base:#x}"));
        assert_eq!(m1.snapshot(), m2.snapshot(), "memory {inst:08x} base {base:#x}");
    }
}
