//! Xenos shader microcode interpreter (vertex and pixel shaders).
//!
//! Program layout: a stream of 3-dword units. Control-flow instructions are 48 bits wide and are
//! packed two per unit at the start of the program; `exec` instructions then point at units of
//! ALU / fetch instructions elsewhere in the same stream. One ALU unit co-issues a vector and a
//! scalar operation sharing a destination or an export.
//!
//! Everything that touches the outside world (vertex buffers, textures) goes through [`ShaderIo`].

use std::f32::consts::LN_2;

pub const MAX_TEMPS: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShaderKind {
    Vertex,
    Pixel,
}

/// A decoded vertex-fetch instruction (the fields the fetcher needs).
#[derive(Clone, Copy, Debug, Default)]
pub struct VertexFetch {
    /// Fetch-constant index 0..95 (two dwords each, three per 6-dword slot).
    pub constant: u32,
    pub format: u32,
    pub is_signed: bool,
    pub is_normalized: bool,
    pub exp_adjust: i32,
    pub stride_dwords: u32,
    pub offset_dwords: i32,
    pub index_rounded: bool,
    pub is_mini: bool,
}

/// A decoded texture-fetch instruction.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextureFetch {
    pub constant: u32,
    /// 0 = 1D, 1 = 2D, 2 = 3D, 3 = cube
    pub dimension: u32,
    pub unnormalized: bool,
    pub mag_filter: u32,
    pub min_filter: u32,
    pub mip_filter: u32,
    pub lod_bias: f32,
}

/// Services the interpreter needs from the GPU: vertex and texture memory.
pub trait ShaderIo {
    /// Returns the raw fetched vector (x, y, z, w) for vertex `index` (already rounded/floored).
    fn vertex_fetch(&mut self, f: &VertexFetch, index: i64) -> [f32; 4];
    /// Samples a texture at `coords` (xyz, normalized unless `f.unnormalized`).
    fn texture_fetch(&mut self, f: &TextureFetch, coords: [f32; 3]) -> [f32; 4];
}

/// Everything a shader invocation can export.
#[derive(Clone, Debug)]
pub struct Exports {
    pub position: [f32; 4],
    pub point_size_edge_kill: [f32; 4],
    pub interpolators: [[f32; 4]; 16],
    pub colors: [[f32; 4]; 4],
    pub depth: f32,
}

impl Default for Exports {
    fn default() -> Self {
        Self {
            position: [0.0; 4],
            point_size_edge_kill: [0.0; 4],
            interpolators: [[0.0; 4]; 16],
            colors: [[0.0; 4]; 4],
            depth: 0.0,
        }
    }
}

/// Mutable machine state of one shader invocation.
pub struct Invocation {
    pub r: [[f32; 4]; MAX_TEMPS],
    pub p0: bool,
    pub a0: i32,
    pub al: i32,
    /// Result of the previous scalar operation (`ps`).
    pub ps: f32,
    pub killed: bool,
    pub exports: Exports,
}

impl Invocation {
    pub fn new() -> Self {
        Self { r: [[0.0; 4]; MAX_TEMPS], p0: false, a0: 0, al: 0, ps: 0.0, killed: false, exports: Exports::default() }
    }
}

impl Default for Invocation {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU constant state visible to a shader: the whole register file plus where this shader's
/// float constants start (pixel shaders use the upper half of the 512 float constants).
pub struct Consts<'a> {
    pub regs: &'a [u32],
    pub float_base: usize,
}

const REG_FLOAT_CONSTS: usize = 0x4000;
const REG_BOOL_CONSTS: usize = 0x4900;
const REG_LOOP_CONSTS: usize = 0x4908;

impl Consts<'_> {
    fn float(&self, index: usize) -> [f32; 4] {
        let base = REG_FLOAT_CONSTS + (self.float_base + index) * 4;
        if base + 3 >= self.regs.len() {
            return [0.0; 4];
        }
        [
            f32::from_bits(self.regs[base]),
            f32::from_bits(self.regs[base + 1]),
            f32::from_bits(self.regs[base + 2]),
            f32::from_bits(self.regs[base + 3]),
        ]
    }

    fn boolean(&self, index: usize) -> bool {
        self.regs.get(REG_BOOL_CONSTS + index / 32).map(|w| (w >> (index % 32)) & 1 != 0).unwrap_or(false)
    }

    /// (count, start, step) of a loop constant.
    fn loop_const(&self, id: usize) -> (i32, i32, i32) {
        let v = self.regs.get(REG_LOOP_CONSTS + id).copied().unwrap_or(0);
        (((v & 0xFF) as i32), ((v >> 8) & 0xFF) as i32, ((v >> 16) & 0xFF) as u8 as i8 as i32)
    }
}

#[derive(Debug)]
pub enum VmError {
    /// An opcode this interpreter doesn't implement yet: (description, raw words).
    Unimplemented(String),
    /// The program ran past its end or looped too long.
    RanAway,
}

#[derive(Clone, Copy, Default)]
struct LoopFrame {
    /// CF index of the loop_start instruction's successor (where the body begins).
    body: usize,
    count: i32,
    iter: i32,
    step: i32,
    saved_al: i32,
}

fn saturate(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

fn dot(a: &[f32; 4], b: &[f32; 4], n: usize) -> f32 {
    (0..n).map(|i| a[i] * b[i]).sum()
}

fn b2f(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// Decoded 48-bit control-flow word: (dword_0, 16-bit dword_1).
fn cf_word(ucode: &[u32], n: usize) -> Option<(u32, u32)> {
    let base = (n / 2) * 3;
    let (d0, d1, d2) = (*ucode.get(base)?, *ucode.get(base + 1)?, *ucode.get(base + 2)?);
    Some(if n % 2 == 0 { (d0, d1 & 0xFFFF) } else { ((d1 >> 16) | (d2 << 16), d2 >> 16) })
}

pub const VECTOR_NAMES: [&str; 30] = [
    "add", "mul", "max", "min", "seq", "sgt", "sge", "sne", "frc", "trunc", "floor", "mad", "cndeq", "cndge", "cndgt",
    "dp4", "dp3", "dp2add", "cube", "max4", "setp_eq_push", "setp_ne_push", "setp_gt_push", "setp_ge_push", "kill_eq",
    "kill_gt", "kill_ge", "kill_ne", "dst", "maxa",
];

pub const SCALAR_NAMES: [&str; 51] = [
    "adds", "adds_prev", "muls", "muls_prev", "muls_prev2", "maxs", "mins", "seqs", "sgts", "sges", "snes", "frcs",
    "truncs", "floors", "exp", "logc", "log", "rcpc", "rcpf", "rcp", "rsqc", "rsqf", "rsq", "maxas", "maxasf", "subs",
    "subs_prev", "setp_eq", "setp_ne", "setp_gt", "setp_ge", "setp_inv", "setp_pop", "setp_clr", "setp_rstr",
    "kills_eq", "kills_gt", "kills_ge", "kills_ne", "kills_one", "sqrt", "?", "mulsc0", "mulsc1", "addsc0", "addsc1",
    "subsc0", "subsc1", "sin", "cos", "retain_prev",
];

/// Number of vector source operands for each vector opcode.
fn vector_operands(op: u32) -> usize {
    match op {
        8 | 9 | 10 | 19 => 1,        // frc trunc floor max4
        11 | 12 | 13 | 14 | 17 => 3, // mad cnd* dp2add
        _ => 2,
    }
}

/// 0 = no operand, 1 = single component `a`, 2 = `a` and `b` from one register, 3 = const + temp.
fn scalar_operand_kind(op: u32) -> u32 {
    match op {
        33 | 50 => 0,
        0 | 2 | 4 | 5 | 6 | 23 | 24 | 25 => 2,
        42..=47 => 3,
        _ => 1,
    }
}

pub struct Vm<'a> {
    pub ucode: &'a [u32],
    pub kind: ShaderKind,
    pub consts: Consts<'a>,
}

impl<'a> Vm<'a> {
    pub fn new(ucode: &'a [u32], kind: ShaderKind, consts: Consts<'a>) -> Self {
        Self { ucode, kind, consts }
    }

    /// Run the whole program for one vertex / pixel.
    pub fn run(&self, inv: &mut Invocation, io: &mut dyn ShaderIo) -> Result<(), VmError> {
        let mut pc = 0usize;
        let mut loops: Vec<LoopFrame> = Vec::new();
        let mut calls: Vec<usize> = Vec::new();
        let mut budget = 100_000u32;
        while let Some((d0, d1)) = cf_word(self.ucode, pc) {
            budget -= 1;
            if budget == 0 {
                return Err(VmError::RanAway);
            }
            let opcode = d1 >> 12;
            let address = d0 & 0xFFF;
            let count = (d0 >> 12) & 7;
            let sequence = (d0 >> 16) & 0xFFF;
            match opcode {
                0 | 12 | 15 => {} // nop, alloc, mark_vs_fetch_done
                1 | 2 => {
                    self.exec(inv, io, address as usize, count, sequence)?;
                    if opcode == 2 {
                        return Ok(());
                    }
                }
                3 | 4 | 13 | 14 => {
                    // cond_exec on a bool constant: bool index in w1[2..9], condition in w1[10]
                    let b = self.consts.boolean(((d1 >> 2) & 0xFF) as usize);
                    let want = (d1 >> 10) & 1 != 0;
                    if b == want {
                        self.exec(inv, io, address as usize, count, sequence)?;
                        if opcode == 4 || opcode == 14 {
                            return Ok(());
                        }
                    }
                }
                5 | 6 => {
                    // cond_exec_pred: predicate p0 compared with the condition bit (w1[10])
                    let want = (d1 >> 10) & 1 != 0;
                    if inv.p0 == want {
                        self.exec(inv, io, address as usize, count, sequence)?;
                        if opcode == 6 {
                            return Ok(());
                        }
                    }
                }
                7 => {
                    // loop_start: address (13 bits) = where to go when skipping; loop_id in dw0[16..20]
                    let skip_to = (d0 & 0x1FFF) as usize;
                    let repeat = (d0 >> 13) & 1 != 0;
                    let id = ((d0 >> 16) & 0x1F) as usize;
                    let (cnt, start, step) = self.consts.loop_const(id);
                    if cnt == 0 {
                        pc = skip_to;
                        continue;
                    }
                    let saved_al = inv.al;
                    if !repeat {
                        inv.al = start;
                    }
                    loops.push(LoopFrame { body: pc + 1, count: cnt, iter: 0, step, saved_al });
                }
                8 => {
                    // loop_end: address = body start; predicated break in dw0[21], condition in dw1[10]
                    let predicated_break = (d0 >> 21) & 1 != 0;
                    let want = (d1 >> 10) & 1 != 0;
                    if let Some(f) = loops.last_mut() {
                        f.iter += 1;
                        inv.al = inv.al.wrapping_add(f.step);
                        let brk = predicated_break && inv.p0 == want;
                        if f.iter < f.count && !brk {
                            pc = f.body;
                            continue;
                        }
                        inv.al = f.saved_al;
                        loops.pop();
                    }
                }
                9 => {
                    // cond_call: unconditional bit dw0[13], predicated bit dw0[14], bool dw1[2..9], cond dw1[10]
                    let unconditional = (d0 >> 13) & 1 != 0;
                    let predicated = (d0 >> 14) & 1 != 0;
                    let want = (d1 >> 10) & 1 != 0;
                    let go = unconditional
                        || if predicated { inv.p0 == want } else { self.consts.boolean(((d1 >> 2) & 0xFF) as usize) == want };
                    if go && calls.len() < 4 {
                        calls.push(pc + 1);
                        pc = (d0 & 0x1FFF) as usize;
                        continue;
                    }
                }
                10 => {
                    if let Some(ret) = calls.pop() {
                        pc = ret;
                        continue;
                    }
                }
                11 => {
                    // cond_jmp
                    let unconditional = (d0 >> 13) & 1 != 0;
                    let predicated = (d0 >> 14) & 1 != 0;
                    let want = (d1 >> 10) & 1 != 0;
                    let go = unconditional
                        || if predicated { inv.p0 == want } else { self.consts.boolean(((d1 >> 2) & 0xFF) as usize) == want };
                    if go {
                        pc = (d0 & 0x1FFF) as usize;
                        continue;
                    }
                }
                o => return Err(VmError::Unimplemented(format!("control flow opcode {o} (cf {pc}: {d0:08x} {d1:04x})"))),
            }
            pc += 1;
        }
        Ok(())
    }

    fn exec(&self, inv: &mut Invocation, io: &mut dyn ShaderIo, address: usize, count: u32, sequence: u32) -> Result<(), VmError> {
        for i in 0..count as usize {
            let unit = (address + i) * 3;
            let w = match self.ucode.get(unit..unit + 3) {
                Some(w) => [w[0], w[1], w[2]],
                None => return Err(VmError::RanAway),
            };
            if (sequence >> (2 * i)) & 1 != 0 {
                self.fetch(inv, io, w)?;
            } else {
                self.alu(inv, w)?;
            }
        }
        Ok(())
    }

    fn fetch(&self, inv: &mut Invocation, io: &mut dyn ShaderIo, w: [u32; 3]) -> Result<(), VmError> {
        let opcode = w[0] & 0x1F;
        let src_reg = ((w[0] >> 5) & 0x3F) as i32 + if (w[0] >> 11) & 1 != 0 { inv.al } else { 0 };
        let dst_reg = ((w[0] >> 12) & 0x3F) as i32 + if (w[0] >> 18) & 1 != 0 { inv.al } else { 0 };
        let dst_swiz = w[1] & 0xFFF;
        let predicated = (w[1] >> 31) & 1 != 0;
        let pred_cond = (w[2] >> 31) & 1 != 0;
        if predicated && inv.p0 != pred_cond {
            return Ok(());
        }
        let src = inv.r[src_reg.clamp(0, 127) as usize];
        let value = match opcode {
            0 => {
                // vertex fetch
                let f = VertexFetch {
                    constant: ((w[0] >> 20) & 0x1F) * 3 + ((w[0] >> 25) & 3),
                    format: (w[1] >> 16) & 0x3F,
                    is_signed: (w[1] >> 12) & 1 != 0,
                    is_normalized: (w[1] >> 13) & 1 == 0,
                    exp_adjust: (((w[1] >> 24) & 0x3F) as i32) << 26 >> 26,
                    stride_dwords: w[2] & 0xFF,
                    offset_dwords: (((w[2] >> 8) & 0x7F_FFFF) as i32) << 9 >> 9,
                    index_rounded: (w[1] >> 15) & 1 != 0,
                    is_mini: (w[1] >> 30) & 1 != 0,
                };
                let comp = src[((w[0] >> 30) & 3) as usize];
                let index = if f.index_rounded { (comp + 0.5).floor() } else { comp.floor() } as i64;
                io.vertex_fetch(&f, index)
            }
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
                let pick = |c: u32| src[((swz >> (2 * c)) & 3) as usize];
                io.texture_fetch(&f, [pick(0), pick(1), pick(2)])
            }
            // set LOD / gradients / weights etc. don't produce a register result we model yet
            _ => return Ok(()),
        };
        let d = &mut inv.r[dst_reg.clamp(0, 127) as usize];
        for c in 0..4 {
            match (dst_swiz >> (3 * c)) & 7 {
                s @ 0..=3 => d[c] = value[s as usize],
                4 => d[c] = 0.0,
                5 => d[c] = 1.0,
                _ => {} // 7 = keep
            }
        }
        Ok(())
    }

    /// Read ALU source operand `n` (1..=3) as a swizzled, abs/negated 4-vector.
    fn src(&self, inv: &Invocation, w: [u32; 3], n: usize) -> [f32; 4] {
        let (is_temp, reg, swz, neg) = match n {
            1 => ((w[2] >> 31) & 1 != 0, (w[2] >> 16) & 0xFF, (w[1] >> 16) & 0xFF, (w[1] >> 26) & 1 != 0),
            2 => ((w[2] >> 30) & 1 != 0, (w[2] >> 8) & 0xFF, (w[1] >> 8) & 0xFF, (w[1] >> 25) & 1 != 0),
            _ => ((w[2] >> 29) & 1 != 0, w[2] & 0xFF, w[1] & 0xFF, (w[1] >> 24) & 1 != 0),
        };
        let (value, abs) = if is_temp {
            let mut idx = (reg & 0x3F) as i32;
            if reg & 0x40 != 0 {
                idx += inv.al;
            }
            (inv.r[idx.clamp(0, 127) as usize], reg & 0x80 != 0)
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
            let mut idx = reg as i32;
            if addressed {
                idx += if (w[1] >> 29) & 1 != 0 { inv.a0 } else { inv.al };
            }
            (self.consts.float(idx.clamp(0, 255) as usize), (w[0] >> 7) & 1 != 0)
        };
        let mut out = [0.0; 4];
        for (c, o) in out.iter_mut().enumerate() {
            let mut v = value[((swz >> (2 * c)) as usize + c) & 3];
            if abs {
                v = v.abs();
            }
            if neg {
                v = -v;
            }
            *o = v;
        }
        out
    }

    fn alu(&self, inv: &mut Invocation, w: [u32; 3]) -> Result<(), VmError> {
        let is_predicated = (w[1] >> 28) & 1 != 0;
        let pred_cond = (w[1] >> 27) & 1 != 0;
        if is_predicated && inv.p0 != pred_cond {
            return Ok(());
        }
        let export = (w[0] >> 15) & 1 != 0;
        let vmask = (w[0] >> 16) & 0xF;
        let smask = (w[0] >> 20) & 0xF;
        let vdest = w[0] & 0x3F;
        let vdest_rel = (w[0] >> 6) & 1 != 0;
        let sdest = (w[0] >> 8) & 0x3F;
        let sdest_rel = (w[0] >> 14) & 1 != 0;
        let vclamp = (w[0] >> 24) & 1 != 0;
        let sclamp = (w[0] >> 25) & 1 != 0;
        let vop = (w[2] >> 24) & 0x1F;
        let sop = w[0] >> 26;

        // ---- vector operation ----
        let nops = vector_operands(vop);
        let a = self.src(inv, w, 1);
        let b = if nops >= 2 { self.src(inv, w, 2) } else { [0.0; 4] };
        let c = if nops >= 3 { self.src(inv, w, 3) } else { [0.0; 4] };
        let mut new_p0 = None;
        let mut new_a0 = None;
        let mut kill = false;
        let per = |f: &dyn Fn(usize) -> f32| -> [f32; 4] { [f(0), f(1), f(2), f(3)] };
        let mut v: [f32; 4] = match vop {
            0 => per(&|i| a[i] + b[i]),
            1 => per(&|i| a[i] * b[i]),
            2 => per(&|i| if a[i] >= b[i] { a[i] } else { b[i] }),
            3 => per(&|i| if a[i] < b[i] { a[i] } else { b[i] }),
            4 => per(&|i| b2f(a[i] == b[i])),
            5 => per(&|i| b2f(a[i] > b[i])),
            6 => per(&|i| b2f(a[i] >= b[i])),
            7 => per(&|i| b2f(a[i] != b[i])),
            8 => per(&|i| a[i] - a[i].floor()),
            9 => per(&|i| a[i].trunc()),
            10 => per(&|i| a[i].floor()),
            11 => per(&|i| a[i].mul_add(b[i], c[i])),
            12 => per(&|i| if a[i] == 0.0 { b[i] } else { c[i] }),
            13 => per(&|i| if a[i] >= 0.0 { b[i] } else { c[i] }),
            14 => per(&|i| if a[i] > 0.0 { b[i] } else { c[i] }),
            15 => [dot(&a, &b, 4); 4],
            16 => [dot(&a, &b, 3); 4],
            17 => [a[0] * b[0] + a[1] * b[1] + c[0]; 4],
            18 => {
                // cube: expects src0.zzxy / src1.yxzz style swizzles
                let (x, y, z) = (b[2], b[1], b[0]);
                let (tc, sc, ma, id);
                if z.abs() >= x.abs() && z.abs() >= y.abs() {
                    tc = -y;
                    sc = if z < 0.0 { -x } else { x };
                    ma = 2.0 * z;
                    id = if z < 0.0 { 5.0 } else { 4.0 };
                } else if y.abs() >= x.abs() {
                    tc = if y < 0.0 { -z } else { z };
                    sc = x;
                    ma = 2.0 * y;
                    id = if y < 0.0 { 3.0 } else { 2.0 };
                } else {
                    tc = -y;
                    sc = if x < 0.0 { z } else { -z };
                    ma = 2.0 * x;
                    id = if x < 0.0 { 1.0 } else { 0.0 };
                }
                [tc, sc, ma, id]
            }
            19 => {
                let m = if a[0] > a[1] && a[0] > a[2] && a[0] > a[3] {
                    a[0]
                } else if a[1] > a[2] && a[1] > a[3] {
                    a[1]
                } else if a[2] > a[3] {
                    a[2]
                } else {
                    a[3]
                };
                [m; 4]
            }
            20..=23 => {
                // setp_*_push
                let cmp = |x: f32| match vop {
                    20 => x == 0.0,
                    21 => x != 0.0,
                    22 => x > 0.0,
                    _ => x >= 0.0,
                };
                // For eq the second term compares src1 == 0; for the others it is the named test.
                let second_w = if vop == 20 { b[3] == 0.0 } else { cmp(b[3]) };
                let second_x = if vop == 20 { b[0] == 0.0 } else { cmp(b[0]) };
                new_p0 = Some(a[3] == 0.0 && second_w);
                if a[0] == 0.0 && second_x { [0.0; 4] } else { [a[0] + 1.0; 4] }
            }
            24..=27 => {
                let hit = (0..4).any(|i| match vop {
                    24 => a[i] == b[i],
                    25 => a[i] > b[i],
                    26 => a[i] >= b[i],
                    _ => a[i] != b[i],
                });
                kill = hit;
                [b2f(hit); 4]
            }
            28 => [1.0, a[1] * b[1], a[2], b[3]],
            29 => {
                new_a0 = Some((a[3] + 0.5).floor().clamp(-256.0, 255.0) as i32);
                per(&|i| if a[0] >= b[i] { a[i] } else { b[i] })
            }
            o => return Err(VmError::Unimplemented(format!("vector opcode {o}"))),
        };
        let _ = (nops, c);

        // ---- scalar operation ----
        let (sa, sb) = match scalar_operand_kind(sop) {
            0 => (0.0, 0.0),
            3 => {
                // constant `a` (W of src3) and temporary `b` (X of src3's swizzle)
                let swz = w[1] & 0xFF;
                let neg = (w[1] >> 24) & 1 != 0;
                let abs = (w[0] >> 7) & 1 != 0;
                let temp_idx = ((sop & 1) | (((w[2] >> 29) & 1) << 1) | (w[1] & 0x3C)) as usize;
                let cdata = {
                    let mut tmp = w;
                    tmp[2] &= !(1 << 29); // force src3 to be read as a constant
                    self.src(inv, tmp, 3)
                };
                let comp_a = cdata[3];
                let tv = inv.r[temp_idx.min(127)][((swz as usize) + 0) & 3];
                let mut tb = if abs { tv.abs() } else { tv };
                if neg {
                    tb = -tb;
                }
                (comp_a, tb)
            }
            _ => {
                let s = self.src(inv, w, 3);
                (s[3], s[0])
            }
        };
        let ps = inv.ps;
        let sres: f32 = match sop {
            0 => sa + sb,
            1 => sa + ps,
            2 => sa * sb,
            3 => sa * ps,
            4 => {
                if ps == f32::MIN || !ps.is_finite() || !sb.is_finite() || sb <= 0.0 { -f32::MAX } else { sa * ps }
            }
            5 => if sa >= sb { sa } else { sb },
            6 => if sa < sb { sa } else { sb },
            7 => b2f(sa == 0.0),
            8 => b2f(sa > 0.0),
            9 => b2f(sa >= 0.0),
            10 => b2f(sa != 0.0),
            11 => sa - sa.floor(),
            12 => sa.trunc(),
            13 => sa.floor(),
            14 => if sa == 0.0 { 1.0 } else { sa.exp2() },
            15 => {
                let t = if sa == 1.0 { 0.0 } else { sa.ln() / LN_2 };
                if t == f32::NEG_INFINITY { -f32::MAX } else { t }
            }
            16 => if sa == 1.0 { 0.0 } else { sa.ln() / LN_2 },
            17 | 18 | 19 => {
                let t = if sa == 1.0 { 1.0 } else { 1.0 / sa };
                match sop {
                    17 if t == f32::NEG_INFINITY => -f32::MAX,
                    17 if t == f32::INFINITY => f32::MAX,
                    18 if t == f32::NEG_INFINITY => -0.0,
                    18 if t == f32::INFINITY => 0.0,
                    _ => t,
                }
            }
            20 | 21 | 22 => {
                let t = if sa == 1.0 { 1.0 } else { 1.0 / sa.sqrt() };
                match sop {
                    20 if t == f32::NEG_INFINITY => -f32::MAX,
                    20 if t == f32::INFINITY => f32::MAX,
                    21 if t == f32::NEG_INFINITY => -0.0,
                    21 if t == f32::INFINITY => 0.0,
                    _ => t,
                }
            }
            23 => {
                new_a0 = Some((sa + 0.5).floor().clamp(-256.0, 255.0) as i32);
                if sa >= sb { sa } else { sb }
            }
            24 => {
                new_a0 = Some(sa.floor().clamp(-256.0, 255.0) as i32);
                if sa >= sb { sa } else { sb }
            }
            25 => sa - sb,
            26 => sa - ps,
            27..=30 => {
                let hit = match sop {
                    27 => sa == 0.0,
                    28 => sa != 0.0,
                    29 => sa > 0.0,
                    _ => sa >= 0.0,
                };
                new_p0 = Some(hit);
                if hit { 0.0 } else { 1.0 }
            }
            31 => {
                if sa == 1.0 {
                    new_p0 = Some(true);
                    0.0
                } else {
                    new_p0 = Some(false);
                    if sa == 0.0 { 1.0 } else { sa }
                }
            }
            32 => {
                if sa - 1.0 <= 0.0 {
                    new_p0 = Some(true);
                    0.0
                } else {
                    new_p0 = Some(false);
                    sa - 1.0
                }
            }
            33 => {
                new_p0 = Some(false);
                f32::MAX
            }
            34 => {
                if sa == 0.0 {
                    new_p0 = Some(true);
                    0.0
                } else {
                    new_p0 = Some(false);
                    sa
                }
            }
            35..=39 => {
                let hit = match sop {
                    35 => sa == 0.0,
                    36 => sa > 0.0,
                    37 => sa >= 0.0,
                    38 => sa != 0.0,
                    _ => sa == 1.0,
                };
                kill |= hit;
                b2f(hit)
            }
            40 => sa.sqrt(),
            42 | 43 => sa * sb,
            44 | 45 => sa + sb,
            46 | 47 => sa - sb,
            48 => sa.sin(),
            49 => sa.cos(),
            50 => ps,
            o => return Err(VmError::Unimplemented(format!("scalar opcode {o}"))),
        };

        // ---- side effects ----
        if let Some(p) = new_p0 {
            inv.p0 = p;
        }
        if let Some(a) = new_a0 {
            inv.a0 = a;
        }
        if kill && self.kind == ShaderKind::Pixel {
            inv.killed = true;
        }
        inv.ps = sres;

        // ---- writes ----
        if vclamp {
            v = v.map(saturate);
        }
        let s = if sclamp { saturate(sres) } else { sres };
        if export {
            // Per component: vector-only -> vector, scalar-only -> scalar, both -> constant 1,
            // neither -> constant 0 when scalar_dest_rel is set, otherwise unchanged.
            let mut out: [Option<f32>; 4] = [None; 4];
            for i in 0..4 {
                let (vm, sm) = ((vmask >> i) & 1 != 0, (smask >> i) & 1 != 0);
                out[i] = match (vm, sm) {
                    (true, false) => Some(v[i]),
                    (false, true) => Some(s),
                    (true, true) => Some(1.0),
                    (false, false) if sdest_rel => Some(0.0),
                    _ => None,
                };
            }
            self.write_export(inv, vdest, out);
        } else {
            let vi = (vdest as i32 + if vdest_rel { inv.al } else { 0 }).clamp(0, 127) as usize;
            for i in 0..4 {
                if (vmask >> i) & 1 != 0 {
                    inv.r[vi][i] = v[i];
                }
            }
            let si = (sdest as i32 + if sdest_rel { inv.al } else { 0 }).clamp(0, 127) as usize;
            for i in 0..4 {
                if (smask >> i) & 1 != 0 {
                    inv.r[si][i] = s;
                }
            }
        }
        Ok(())
    }

    fn write_export(&self, inv: &mut Invocation, reg: u32, vals: [Option<f32>; 4]) {
        let target: &mut [f32; 4] = match (self.kind, reg) {
            (ShaderKind::Vertex, 62) => &mut inv.exports.position,
            (ShaderKind::Vertex, 63) => &mut inv.exports.point_size_edge_kill,
            (ShaderKind::Vertex, 0..=15) => &mut inv.exports.interpolators[reg as usize],
            (ShaderKind::Pixel, 0..=3) => &mut inv.exports.colors[reg as usize],
            (ShaderKind::Pixel, 61) => {
                if let Some(d) = vals[0] {
                    inv.exports.depth = d;
                }
                return;
            }
            _ => return, // memory export etc.: not modelled
        };
        for i in 0..4 {
            if let Some(x) = vals[i] {
                target[i] = x;
            }
        }
    }
}

fn swizzle_text(swz: u32) -> String {
    (0..4).map(|c| ['x', 'y', 'z', 'w'][(((swz >> (2 * c)) as usize) + c) & 3]).collect()
}

fn dest_text(export: bool, reg: u32, mask: u32, kind_is_vertex: bool) -> String {
    let comps: String = (0..4).filter(|i| (mask >> i) & 1 != 0).map(|i| ['x', 'y', 'z', 'w'][i]).collect();
    let name = if export {
        match (kind_is_vertex, reg) {
            (true, 62) => "oPos".to_string(),
            (true, 63) => "oPts".to_string(),
            (true, r) => format!("o{r}"),
            (false, 61) => "oDepth".to_string(),
            (false, r) => format!("oC{r}"),
        }
    } else {
        format!("r{reg}")
    };
    format!("{name}.{}", if comps.is_empty() { "_".to_string() } else { comps })
}

fn src_text(w: [u32; 3], n: usize) -> String {
    let (is_temp, reg, swz, neg) = match n {
        1 => ((w[2] >> 31) & 1 != 0, (w[2] >> 16) & 0xFF, (w[1] >> 16) & 0xFF, (w[1] >> 26) & 1 != 0),
        2 => ((w[2] >> 30) & 1 != 0, (w[2] >> 8) & 0xFF, (w[1] >> 8) & 0xFF, (w[1] >> 25) & 1 != 0),
        _ => ((w[2] >> 29) & 1 != 0, w[2] & 0xFF, w[1] & 0xFF, (w[1] >> 24) & 1 != 0),
    };
    let base = if is_temp { format!("r{}", reg & 0x3F) } else { format!("c{reg}") };
    format!("{}{base}.{}", if neg { "-" } else { "" }, swizzle_text(swz))
}

/// One instruction unit (ALU, or fetch if `is_fetch`) as text. Vertex vs pixel is not known here,
/// so exports are shown with the vertex-shader names for registers that overlap.
pub fn disassemble_instruction(w: [u32; 3], is_fetch: bool) -> String {
    if is_fetch {
        let opcode = w[0] & 0x1F;
        let dst = (w[0] >> 12) & 0x3F;
        let src = (w[0] >> 5) & 0x3F;
        let dswz: String = (0..4)
            .map(|c| match (w[1] >> (3 * c)) & 7 {
                s @ 0..=3 => ['x', 'y', 'z', 'w'][s as usize],
                4 => '0',
                5 => '1',
                _ => '_',
            })
            .collect();
        return if opcode == 0 {
            format!(
                "vfetch r{dst}.{dswz}, r{src}.{} const={} fmt={} stride={} offset={}{}",
                ['x', 'y', 'z', 'w'][((w[0] >> 30) & 3) as usize],
                ((w[0] >> 20) & 0x1F) * 3 + ((w[0] >> 25) & 3),
                (w[1] >> 16) & 0x3F,
                w[2] & 0xFF,
                (((w[2] >> 8) & 0x7F_FFFF) as i32) << 9 >> 9,
                if (w[1] >> 30) & 1 != 0 { " (mini)" } else { "" },
            )
        } else if opcode == 1 {
            format!("tfetch r{dst}.{dswz}, r{src}, tf{} dim={}", (w[0] >> 20) & 0x1F, (w[2] >> 14) & 3)
        } else {
            format!("fetch opcode {opcode}")
        };
    }
    let export = (w[0] >> 15) & 1 != 0;
    let vop = ((w[2] >> 24) & 0x1F) as usize;
    let sop = (w[0] >> 26) as usize;
    let vname = VECTOR_NAMES.get(vop).copied().unwrap_or("?");
    let sname = SCALAR_NAMES.get(sop).copied().unwrap_or("?");
    let mut s = format!(
        "{vname} {}, {}",
        dest_text(export, w[0] & 0x3F, (w[0] >> 16) & 0xF, true),
        src_text(w, 1)
    );
    if vector_operands(vop as u32) >= 2 {
        s.push_str(&format!(", {}", src_text(w, 2)));
    }
    if vector_operands(vop as u32) >= 3 {
        s.push_str(&format!(", {}", src_text(w, 3)));
    }
    if sop != 50 {
        s.push_str(&format!("  +  {sname} {}", dest_text(export, if export { w[0] & 0x3F } else { (w[0] >> 8) & 0x3F }, (w[0] >> 20) & 0xF, true)));
    }
    s
}

/// Human-readable dump of a shader program, for debugging.
pub fn disassemble(ucode: &[u32]) -> String {
    let mut out = String::new();
    let mut n = 0;
    while let Some((d0, d1)) = cf_word(ucode, n) {
        let op = d1 >> 12;
        let name = [
            "nop", "exec", "exec_end", "cexec", "cexec_end", "cexec_pred", "cexec_pred_end", "loop", "endloop", "call",
            "return", "jmp", "alloc", "cexec_pc", "cexec_pc_end", "mark_vs_fetch_done",
        ][op as usize];
        out.push_str(&format!("cf{n:>2}: {name}"));
        if (1..=6).contains(&op) || op == 13 || op == 14 {
            let (addr, cnt, seq) = (d0 & 0xFFF, (d0 >> 12) & 7, (d0 >> 16) & 0xFFF);
            out.push_str(&format!(" addr={addr} count={cnt} seq={seq:#x}"));
            for i in 0..cnt as usize {
                let unit = (addr as usize + i) * 3;
                if let Some(w) = ucode.get(unit..unit + 3) {
                    out.push_str("\n      ");
                    out.push_str(&disassemble_instruction([w[0], w[1], w[2]], (seq >> (2 * i)) & 1 != 0));
                }
            }
        } else if op == 12 {
            out.push_str(&format!(" type={} size={}", (d1 >> 9) & 3, d0 & 7));
        }
        out.push('\n');
        if op == 2 || op == 4 || op == 6 || op == 14 {
            break;
        }
        n += 1;
        if n > 64 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoIo;
    impl ShaderIo for NoIo {
        fn vertex_fetch(&mut self, _: &VertexFetch, _: i64) -> [f32; 4] {
            [0.0; 4]
        }
        fn texture_fetch(&mut self, _: &TextureFetch, _: [f32; 3]) -> [f32; 4] {
            [0.5; 4]
        }
    }

    /// The 9-dword pixel shader GTA IV uploads for its 2D quads: `alloc colors; exec_end { max oC0, r0, r0 }`.
    const GTA_PS: [u32; 9] = [0x0000_0000, 0x1001_c400, 0x2200_0000, 0xc80f_8000, 0x0000_0000, 0xc200_0000, 0, 0, 0];

    #[test]
    fn real_pixel_shader_passes_interpolant_to_color0() {
        let regs = vec![0u32; 0x5000];
        let vm = Vm::new(&GTA_PS, ShaderKind::Pixel, Consts { regs: &regs, float_base: 256 });
        let mut inv = Invocation::new();
        inv.r[0] = [0.25, 0.5, 0.75, 1.0];
        vm.run(&mut inv, &mut NoIo).unwrap();
        assert_eq!(inv.exports.colors[0], [0.25, 0.5, 0.75, 1.0]);
        assert!(!inv.killed);
    }

    #[test]
    fn disassembler_reads_control_flow() {
        let text = disassemble(&GTA_PS);
        assert!(text.contains("alloc type=2"), "{text}");
        assert!(text.contains("exec_end addr=1 count=1"), "{text}");
    }

    /// Hand-assembled `mul r1, r0, c0` + `mov oPos, r1`-style program: exercises a constant operand,
    /// swizzles, the scalar co-issue slot and a vertex-shader position export.
    #[test]
    fn vertex_shader_multiplies_by_constant_and_exports_position() {
        // CF: exec_end addr=1 count=2 (both ALU: sequence 0). Unit 0 holds cf0 (exec_end) + cf1 (nop).
        let cf0_d0 = 1 | (2 << 12); // address 1, count 2
        let cf0_d1 = 2u32 << 12; // opcode 2 = exec_end
        let d0 = cf0_d0;
        let d1 = cf0_d1; // low 16 bits: cf0 word1; high 16 bits: cf1 word0 low half (0)
        let d2 = 0;
        // ALU #1: mul r1.xyzw = r0 * c0  (vector_opc=1; src1 = temp r0, src2 = const c0)
        let a1_d0 = (1 & 0x3F) | (0xF << 16) | (50 << 26); // dest r1, vector mask xyzw, scalar retain_prev (no write)
        let a1_d1 = 0; // identity swizzles
        let a1_d2 = 0 | (0 << 8) | (0 << 16) | (1 << 24) | (1 << 31); // src1=r0 temp; src2=c0 (sel=0 => const); vector_opc=mul
        // ALU #2: max oPos = r1, r1 (export to register 62)
        let a2_d0 = 62 | (1 << 15) | (0xF << 16) | (50 << 26);
        let a2_d1 = 0;
        let a2_d2 = (1 << 16) | (1 << 8) | (2 << 24) | (1 << 30) | (1 << 31); // src1=r1, src2=r1, both temp, max
        let ucode = [d0, d1, d2, a1_d0, a1_d1, a1_d2, a2_d0, a2_d1, a2_d2];
        let mut regs = vec![0u32; 0x5000];
        for (i, v) in [2.0f32, 3.0, 4.0, 0.5].iter().enumerate() {
            regs[0x4000 + i] = v.to_bits();
        }
        let vm = Vm::new(&ucode, ShaderKind::Vertex, Consts { regs: &regs, float_base: 0 });
        let mut inv = Invocation::new();
        inv.r[0] = [1.0, 2.0, 3.0, 4.0];
        vm.run(&mut inv, &mut NoIo).unwrap();
        assert_eq!(inv.exports.position, [2.0, 6.0, 12.0, 2.0]);
    }

    #[test]
    fn scalar_and_predicate_ops() {
        // adds with two components: dest r2.x = src3.w + src3.x ; single ALU, retain nothing vector-side
        let cf_d0 = 1 | (1 << 12);
        let cf_d1 = 2u32 << 12;
        // vector op: max r0,r0 with empty write mask (nop); scalar op: adds (opc 0) writing r2.x
        let a_d0 = (2) | (0 << 16) | (1 << 20) | (0 << 26); // vector dest r2? mask 0; scalar_dest=0? use scalar dest field below
        let a_d0 = a_d0 | (2 << 8); // scalar_dest = r2
        let a_d1 = 0;
        // src3 = temp r1 (sel bit 29), swizzle identity => a = r1.w, b = r1.x
        let a_d2 = 1 | (2 << 24) | (1 << 29) | (1 << 30) | (1 << 31);
        let ucode = [cf_d0, cf_d1, 0, a_d0, a_d1, a_d2];
        let regs = vec![0u32; 0x5000];
        let vm = Vm::new(&ucode, ShaderKind::Vertex, Consts { regs: &regs, float_base: 0 });
        let mut inv = Invocation::new();
        inv.r[1] = [10.0, 0.0, 0.0, 5.0];
        vm.run(&mut inv, &mut NoIo).unwrap();
        assert_eq!(inv.r[2][0], 15.0);
        assert_eq!(inv.ps, 15.0);
    }
}
