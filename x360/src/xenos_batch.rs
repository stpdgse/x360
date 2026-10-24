//! Batched pixel-shader execution for the software Xenos.
//!
//! The reference interpreter in `xenos_vm` decodes every instruction again for every pixel. This module compiles a
//! pixel shader once per draw (constants and bool-constant branches resolved, operands pre-decoded) and runs it for
//! up to [`LANES`] pixels in lockstep over structure-of-arrays registers, so the host compiler can vectorise the
//! arithmetic. Shaders using features whose control flow differs per pixel (loops, calls, jumps, relative register
//! addressing) are not compiled; the caller falls back to the reference interpreter for those.
//!
//! The semantics are a lane-wise port of `Vm::alu` / `Vm::fetch`; `X360_VERIFY_BATCH=1` cross-checks every batch
//! against the reference interpreter.

use crate::xenos_vm::{TextureFetch, MAX_TEMPS};

/// Pixels executed together.
pub const LANES: usize = 16;
pub type V = [f32; LANES];
pub const ALL: u32 = (1u32 << LANES) - 1;

#[derive(Clone, Copy, Debug)]
pub enum Src {
    Temp(u8),
    Const([f32; 4]),
}

/// A decoded ALU source operand: register or constant, component selectors, |abs| and negate.
#[derive(Clone, Copy, Debug)]
pub struct Operand {
    pub src: Src,
    pub sel: [u8; 4],
    pub abs: bool,
    pub neg: bool,
    /// for constants: the four selected components (abs/neg applied) broadcast over all lanes
    pub cv: [V; 4],
}

/// How the scalar half of an ALU instruction gets its two inputs.
#[derive(Clone, Copy, Debug)]
pub enum ScalarIn {
    None,
    /// `a` = src3.w, `b` = src3.x (both from the same operand)
    Src3(Operand),
    /// constant `a` (src3 constant, .w) and temporary `b` (swizzle, abs, neg already applied by the selector)
    ConstTemp { a: [f32; 4], reg: u8, comp: u8, abs: bool, neg: bool },
}

#[derive(Clone, Copy, Debug)]
pub struct Alu {
    /// 0 = always, 1 = only where p0 is false, 2 = only where p0 is true
    pub pred: u8,
    pub export: bool,
    pub vop: u32,
    pub sop: u32,
    pub nops: usize,
    pub a: Operand,
    pub b: Operand,
    pub c: Operand,
    pub s_in: ScalarIn,
    pub vmask: u8,
    pub smask: u8,
    pub vdest: u8,
    pub sdest: u8,
    /// for exports: "scalar dest rel" means components written by neither unit become 0.0
    pub sdest_rel: bool,
    pub vclamp: bool,
    pub sclamp: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Fetch {
    pub pred: u8,
    pub f: TextureFetch,
    pub src: u8,
    pub dst: u8,
    pub src_sel: [u8; 3],
    /// per destination component: 0..3 = value component, 4 = 0.0, 5 = 1.0, 7 = keep
    pub dst_swiz: [u8; 4],
}

#[derive(Clone, Copy, Debug)]
pub enum Inst {
    Alu(Alu),
    Fetch(Fetch),
}

/// One `exec` clause: runs for the pixels where p0 == `pred` (if any); `end` finishes those pixels afterwards.
#[derive(Clone, Debug)]
pub struct Clause {
    pub pred: Option<bool>,
    pub end: bool,
    pub insts: Vec<Inst>,
}

#[derive(Clone, Debug)]
pub struct CompiledPs {
    pub clauses: Vec<Clause>,
    /// registers 0..nregs are cleared per batch
    pub nregs: usize,
    /// highest interpolant register the shader reads before writing it (+1); only these need filling
    pub ninterp: usize,
}

/// State of up to [`LANES`] pixels running one shader.
pub struct Batch {
    pub r: Vec<[V; 4]>,
    pub colors: [[V; 4]; 4],
    pub depth: V,
    pub ps: V,
    /// bit n = pixel n's predicate register
    pub p0: u32,
    /// bit n = pixel n was killed
    pub killed: u32,
}

impl Batch {
    pub fn new() -> Self {
        Self { r: vec![[[0.0; LANES]; 4]; MAX_TEMPS], colors: [[[0.0; LANES]; 4]; 4], depth: [0.0; LANES], ps: [0.0; LANES], p0: 0, killed: 0 }
    }

    pub fn reset(&mut self, nregs: usize) {
        for r in self.r[..nregs.min(MAX_TEMPS)].iter_mut() {
            *r = [[0.0; LANES]; 4];
        }
        self.colors = [[[0.0; LANES]; 4]; 4];
        self.depth = [0.0; LANES];
        self.ps = [0.0; LANES];
        self.p0 = 0;
        self.killed = 0;
    }
}

fn float_const(regs: &[u32], float_base: usize, index: usize) -> [f32; 4] {
    let base = 0x4000 + (float_base + index) * 4;
    if base + 3 >= regs.len() {
        return [0.0; 4];
    }
    [f32::from_bits(regs[base]), f32::from_bits(regs[base + 1]), f32::from_bits(regs[base + 2]), f32::from_bits(regs[base + 3])]
}

fn bool_const(regs: &[u32], index: usize) -> bool {
    regs.get(0x4900 + index / 32).map(|w| (w >> (index % 32)) & 1 != 0).unwrap_or(false)
}

/// Decode ALU source operand `n` (1..=3). None when it needs per-pixel (relative) addressing.
fn operand(regs: &[u32], float_base: usize, w: [u32; 3], n: usize, force_const: bool) -> Option<Operand> {
    let (mut is_temp, reg, swz, neg) = match n {
        1 => ((w[2] >> 31) & 1 != 0, (w[2] >> 16) & 0xFF, (w[1] >> 16) & 0xFF, (w[1] >> 26) & 1 != 0),
        2 => ((w[2] >> 30) & 1 != 0, (w[2] >> 8) & 0xFF, (w[1] >> 8) & 0xFF, (w[1] >> 25) & 1 != 0),
        _ => ((w[2] >> 29) & 1 != 0, w[2] & 0xFF, w[1] & 0xFF, (w[1] >> 24) & 1 != 0),
    };
    if force_const {
        is_temp = false;
    }
    let mut sel = [0u8; 4];
    for c in 0..4 {
        sel[c] = ((((swz >> (2 * c)) & 3) as usize + c) & 3) as u8;
    }
    if is_temp {
        if reg & 0x40 != 0 {
            return None; // relative to the loop counter
        }
        Some(Operand { src: Src::Temp((reg & 0x3F) as u8), sel, abs: reg & 0x80 != 0, neg, cv: [[0.0; LANES]; 4] })
    } else {
        let src1_temp = (w[2] >> 31) & 1 != 0;
        let src2_temp = (w[2] >> 30) & 1 != 0;
        let rel0 = (w[1] >> 31) & 1 != 0;
        let rel1 = (w[1] >> 30) & 1 != 0;
        let addressed = match n {
            1 => rel0,
            2 => if src1_temp { rel0 } else { rel1 },
            _ => if src1_temp && src2_temp { rel0 } else { rel1 },
        };
        if addressed {
            return None;
        }
        let v = float_const(regs, float_base, (reg as usize).min(255));
        let abs = (w[0] >> 7) & 1 != 0;
        let mut cv = [[0.0; LANES]; 4];
        for c in 0..4 {
            let mut x = v[sel[c] as usize];
            if abs {
                x = x.abs();
            }
            if neg {
                x = -x;
            }
            cv[c] = [x; LANES];
        }
        Some(Operand { src: Src::Const(v), sel, abs, neg, cv })
    }
}

fn vector_operands(op: u32) -> usize {
    match op {
        8 | 9 | 10 | 19 => 1,
        11 | 12 | 13 | 14 | 17 => 3,
        _ => 2,
    }
}

fn scalar_operand_kind(op: u32) -> u32 {
    match op {
        33 | 50 => 0,
        0 | 2 | 4 | 5 | 6 | 23 | 24 | 25 => 2,
        42..=47 => 3,
        _ => 1,
    }
}

fn dummy() -> Operand {
    Operand { src: Src::Const([0.0; 4]), sel: [0, 1, 2, 3], abs: false, neg: false, cv: [[0.0; LANES]; 4] }
}

fn compile_alu(regs: &[u32], fb: usize, w: [u32; 3]) -> Option<Inst> {
    let is_predicated = (w[1] >> 28) & 1 != 0;
    let pred_cond = (w[1] >> 27) & 1 != 0;
    let export = (w[0] >> 15) & 1 != 0;
    let vdest_rel = (w[0] >> 6) & 1 != 0;
    let sdest_rel = (w[0] >> 14) & 1 != 0;
    if !export && (vdest_rel || sdest_rel) {
        return None;
    }
    let vop = (w[2] >> 24) & 0x1F;
    let sop = w[0] >> 26;
    if vop > 29 || sop > 50 || sop == 41 {
        return None; // unimplemented opcode: let the reference interpreter report it
    }
    let nops = vector_operands(vop);
    let a = operand(regs, fb, w, 1, false)?;
    let b = if nops >= 2 { operand(regs, fb, w, 2, false)? } else { dummy() };
    let c = if nops >= 3 { operand(regs, fb, w, 3, false)? } else { dummy() };
    let s_in = match scalar_operand_kind(sop) {
        0 => ScalarIn::None,
        3 => {
            let cdata = operand(regs, fb, w, 3, true)?;
            let Src::Const(a) = cdata.src else { return None };
            // the constant's own swizzle/abs/neg are not applied to `a`: the interpreter takes cdata[3] after them
            let ca = {
                let mut v = [0.0f32; 4];
                for k in 0..4 {
                    let mut x = a[cdata.sel[k] as usize];
                    if cdata.abs {
                        x = x.abs();
                    }
                    if cdata.neg {
                        x = -x;
                    }
                    v[k] = x;
                }
                v
            };
            let temp_idx = ((sop & 1) | (((w[2] >> 29) & 1) << 1) | (w[1] & 0x3C)) as usize;
            ScalarIn::ConstTemp {
                a: ca,
                reg: temp_idx.min(127) as u8,
                comp: (w[1] & 3) as u8,
                abs: (w[0] >> 7) & 1 != 0,
                neg: (w[1] >> 24) & 1 != 0,
            }
        }
        _ => ScalarIn::Src3(operand(regs, fb, w, 3, false)?),
    };
    Some(Inst::Alu(Alu {
        pred: if is_predicated { if pred_cond { 2 } else { 1 } } else { 0 },
        export,
        vop,
        sop,
        nops,
        a,
        b,
        c,
        s_in,
        vmask: ((w[0] >> 16) & 0xF) as u8,
        smask: ((w[0] >> 20) & 0xF) as u8,
        vdest: (w[0] & 0x3F) as u8,
        sdest: ((w[0] >> 8) & 0x3F) as u8,
        sdest_rel,
        vclamp: (w[0] >> 24) & 1 != 0,
        sclamp: (w[0] >> 25) & 1 != 0,
    }))
}

fn compile_fetch(w: [u32; 3]) -> Option<Option<Inst>> {
    let opcode = w[0] & 0x1F;
    if (w[0] >> 11) & 1 != 0 || (w[0] >> 18) & 1 != 0 {
        return None; // relative source/destination
    }
    let predicated = (w[1] >> 31) & 1 != 0;
    let pred_cond = (w[2] >> 31) & 1 != 0;
    match opcode {
        0 => None, // vertex fetch in a pixel shader
        1 => {
            let f = TextureFetch {
                constant: (w[0] >> 20) & 0x1F,
                dimension: (w[2] >> 14) & 3,
                unnormalized: (w[0] >> 25) & 1 != 0,
                mag_filter: (w[1] >> 12) & 3,
                min_filter: (w[1] >> 14) & 3,
                mip_filter: (w[1] >> 16) & 3,
                lod_bias: ((((w[2] >> 2) & 0x7F) as i32) << 25 >> 25) as f32 / 16.0,
            };
            let swz = (w[0] >> 26) & 0x3F;
            let dswz = w[1] & 0xFFF;
            let mut dst_swiz = [0u8; 4];
            for c in 0..4 {
                dst_swiz[c] = ((dswz >> (3 * c)) & 7) as u8;
            }
            Some(Some(Inst::Fetch(Fetch {
                pred: if predicated { if pred_cond { 2 } else { 1 } } else { 0 },
                f,
                src: ((w[0] >> 5) & 0x3F) as u8,
                dst: ((w[0] >> 12) & 0x3F) as u8,
                src_sel: [(swz & 3) as u8, ((swz >> 2) & 3) as u8, ((swz >> 4) & 3) as u8],
                dst_swiz,
            })))
        }
        _ => Some(None), // set LOD / gradients: no register result modelled
    }
}

impl CompiledPs {
    /// Compile a pixel shader against the current constants. None if it needs the reference interpreter.
    pub fn compile(ucode: &[u32], regs: &[u32]) -> Option<Self> {
        let fb = 256;
        let cf = |n: usize| -> Option<(u32, u32)> {
            let base = (n / 2) * 3;
            let (d0, d1, d2) = (*ucode.get(base)?, *ucode.get(base + 1)?, *ucode.get(base + 2)?);
            Some(if n % 2 == 0 { (d0, d1 & 0xFFFF) } else { ((d1 >> 16) | (d2 << 16), d2 >> 16) })
        };
        let mut clauses: Vec<Clause> = Vec::new();
        let mut pc = 0usize;
        let mut budget = 4096;
        while let Some((d0, d1)) = cf(pc) {
            budget -= 1;
            if budget == 0 {
                return None;
            }
            let opcode = d1 >> 12;
            let address = (d0 & 0xFFF) as usize;
            let count = ((d0 >> 12) & 7) as usize;
            let sequence = (d0 >> 16) & 0xFFF;
            let decode_clause = |pred: Option<bool>, end: bool| -> Option<Clause> {
                let mut insts = Vec::new();
                for i in 0..count {
                    let unit = (address + i) * 3;
                    let w = ucode.get(unit..unit + 3)?;
                    let w = [w[0], w[1], w[2]];
                    if (sequence >> (2 * i)) & 1 != 0 {
                        if let Some(f) = compile_fetch(w)? {
                            insts.push(f);
                        }
                    } else {
                        insts.push(compile_alu(regs, fb, w)?);
                    }
                }
                Some(Clause { pred, end, insts })
            };
            match opcode {
                0 | 12 | 15 => {}
                1 | 2 => {
                    clauses.push(decode_clause(None, opcode == 2)?);
                    if opcode == 2 {
                        break;
                    }
                }
                3 | 4 | 13 | 14 => {
                    let b = bool_const(regs, ((d1 >> 2) & 0xFF) as usize);
                    let want = (d1 >> 10) & 1 != 0;
                    if b == want {
                        clauses.push(decode_clause(None, opcode == 4 || opcode == 14)?);
                        if opcode == 4 || opcode == 14 {
                            break;
                        }
                    }
                }
                5 | 6 => {
                    let want = (d1 >> 10) & 1 != 0;
                    clauses.push(decode_clause(Some(want), opcode == 6)?);
                }
                _ => return None, // loops, calls, jumps
            }
            pc += 1;
        }
        // registers touched, and which interpolants are read before being written
        let mut max_reg = 0usize;
        let mut written = [false; 64];
        let mut ninterp = 0usize;
        let mut note_read = |op: &Operand, written: &[bool; 64], max_reg: &mut usize, ninterp: &mut usize| {
            if let Src::Temp(r) = op.src {
                *max_reg = (*max_reg).max(r as usize + 1);
                if (r as usize) < 16 && !written[r as usize] {
                    *ninterp = (*ninterp).max(r as usize + 1);
                }
            }
        };
        for cl in &clauses {
            for inst in &cl.insts {
                match inst {
                    Inst::Alu(a) => {
                        note_read(&a.a, &written, &mut max_reg, &mut ninterp);
                        if a.nops >= 2 {
                            note_read(&a.b, &written, &mut max_reg, &mut ninterp);
                        }
                        if a.nops >= 3 {
                            note_read(&a.c, &written, &mut max_reg, &mut ninterp);
                        }
                        match a.s_in {
                            ScalarIn::Src3(o) => note_read(&o, &written, &mut max_reg, &mut ninterp),
                            ScalarIn::ConstTemp { reg, .. } => {
                                note_read(&Operand { src: Src::Temp(reg), sel: [0; 4], abs: false, neg: false, cv: [[0.0; LANES]; 4] }, &written, &mut max_reg, &mut ninterp)
                            }
                            ScalarIn::None => {}
                        }
                        if !a.export {
                            max_reg = max_reg.max(a.vdest as usize + 1).max(a.sdest as usize + 1);
                            // a partially masked write doesn't define the whole register: only count full writes
                            if a.pred == 0 && cl.pred.is_none() {
                                if a.vmask == 0xF {
                                    written[(a.vdest as usize) & 63] = true;
                                }
                                if a.smask == 0xF {
                                    written[(a.sdest as usize) & 63] = true;
                                }
                            }
                        }
                    }
                    Inst::Fetch(f) => {
                        note_read(&Operand { src: Src::Temp(f.src), sel: [0; 4], abs: false, neg: false, cv: [[0.0; LANES]; 4] }, &written, &mut max_reg, &mut ninterp);
                        max_reg = max_reg.max(f.dst as usize + 1);
                    }
                }
            }
        }
        Some(Self { clauses, nregs: max_reg.max(ninterp).clamp(1, MAX_TEMPS), ninterp })
    }
}
