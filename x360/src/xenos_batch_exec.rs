//! Executor for [`CompiledPs`]: lane-wise ports of the reference interpreter's vector and scalar operations.
//! Common operations have straight loops over the lanes (which the host compiler vectorises); the rest go through
//! per-lane helpers that copy the reference semantics exactly.

use crate::xenos_batch::*;
use crate::xenos_vm::ShaderIo;
use std::f32::consts::LN_2;

#[inline(always)]
fn saturate(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

#[inline(always)]
fn b2f(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

#[inline(always)]
fn pmask(pred: u8, p0: u32) -> u32 {
    match pred {
        0 => ALL,
        1 => !p0 & ALL,
        _ => p0 & ALL,
    }
}

/// The four selected components of an operand as references: straight into the register file when no
/// abs/negate is needed, into the precomputed splat for constants, else into `tmp`.
#[inline(always)]
fn operand_refs<'a>(b: &'a Batch, o: &'a Operand, tmp: &'a mut [V; 4]) -> [&'a V; 4] {
    match o.src {
        Src::Const(_) => [&o.cv[0], &o.cv[1], &o.cv[2], &o.cv[3]],
        Src::Temp(r) => {
            let regs = &b.r[r as usize];
            if !o.abs && !o.neg {
                [&regs[o.sel[0] as usize], &regs[o.sel[1] as usize], &regs[o.sel[2] as usize], &regs[o.sel[3] as usize]]
            } else {
                for c in 0..4 {
                    let src = &regs[o.sel[c] as usize];
                    match (o.abs, o.neg) {
                        (true, false) => (0..LANES).for_each(|l| tmp[c][l] = src[l].abs()),
                        (false, true) => (0..LANES).for_each(|l| tmp[c][l] = -src[l]),
                        _ => (0..LANES).for_each(|l| tmp[c][l] = -src[l].abs()),
                    }
                }
                let t: &'a [V; 4] = tmp;
                [&t[0], &t[1], &t[2], &t[3]]
            }
        }
    }
}

/// Result of the vector unit for all lanes.
struct VecOut {
    v: [V; 4],
    /// lanes whose p0 this operation sets, and the values
    p0_mask: u32,
    p0_bits: u32,
    kill: u32,
}

/// Reference semantics of one vector operation for one pixel (used for the less common opcodes).
fn vec_lane(vop: u32, a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> ([f32; 4], Option<bool>, bool) {
    let per = |f: &dyn Fn(usize) -> f32| -> [f32; 4] { [f(0), f(1), f(2), f(3)] };
    let mut p0 = None;
    let mut kill = false;
    let v = match vop {
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
        11 => per(&|i| a[i] * b[i] + c[i]),
        12 => per(&|i| if a[i] == 0.0 { b[i] } else { c[i] }),
        13 => per(&|i| if a[i] >= 0.0 { b[i] } else { c[i] }),
        14 => per(&|i| if a[i] > 0.0 { b[i] } else { c[i] }),
        15 => [((a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]) + a[3] * b[3]; 4],
        16 => [(a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]; 4],
        17 => [a[0] * b[0] + a[1] * b[1] + c[0]; 4],
        18 => {
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
            let cmp = |x: f32| match vop {
                20 => x == 0.0,
                21 => x != 0.0,
                22 => x > 0.0,
                _ => x >= 0.0,
            };
            let second_w = if vop == 20 { b[3] == 0.0 } else { cmp(b[3]) };
            let second_x = if vop == 20 { b[0] == 0.0 } else { cmp(b[0]) };
            p0 = Some(a[3] == 0.0 && second_w);
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
        _ => per(&|i| if a[0] >= b[i] { a[i] } else { b[i] }), // 29: maxa (address register write is not modelled)
    };
    (v, p0, kill)
}

#[inline(always)]
fn vec_exec(vop: u32, a: &[&V; 4], b: &[&V; 4], c: &[&V; 4]) -> VecOut {
    let mut out = VecOut { v: [[0.0; LANES]; 4], p0_mask: 0, p0_bits: 0, kill: 0 };
    macro_rules! per_comp {
        ($f:expr) => {{
            for ch in 0..4 {
                for l in 0..LANES {
                    out.v[ch][l] = $f(a[ch][l], b[ch][l], c[ch][l]);
                }
            }
        }};
    }
    match vop {
        0 => per_comp!(|x: f32, y: f32, _z: f32| x + y),
        1 => per_comp!(|x: f32, y: f32, _z: f32| x * y),
        2 => per_comp!(|x: f32, y: f32, _z: f32| if x >= y { x } else { y }),
        3 => per_comp!(|x: f32, y: f32, _z: f32| if x < y { x } else { y }),
        4 => per_comp!(|x: f32, y: f32, _z: f32| b2f(x == y)),
        5 => per_comp!(|x: f32, y: f32, _z: f32| b2f(x > y)),
        6 => per_comp!(|x: f32, y: f32, _z: f32| b2f(x >= y)),
        7 => per_comp!(|x: f32, y: f32, _z: f32| b2f(x != y)),
        8 => per_comp!(|x: f32, _y: f32, _z: f32| x - x.floor()),
        10 => per_comp!(|x: f32, _y: f32, _z: f32| x.floor()),
        11 => per_comp!(|x: f32, y: f32, z: f32| x * y + z),
        12 => per_comp!(|x: f32, y: f32, z: f32| if x == 0.0 { y } else { z }),
        13 => per_comp!(|x: f32, y: f32, z: f32| if x >= 0.0 { y } else { z }),
        14 => per_comp!(|x: f32, y: f32, z: f32| if x > 0.0 { y } else { z }),
        15 => {
            for l in 0..LANES {
                let d = ((a[0][l] * b[0][l] + a[1][l] * b[1][l]) + a[2][l] * b[2][l]) + a[3][l] * b[3][l];
                for ch in 0..4 {
                    out.v[ch][l] = d;
                }
            }
        }
        16 => {
            for l in 0..LANES {
                let d = (a[0][l] * b[0][l] + a[1][l] * b[1][l]) + a[2][l] * b[2][l];
                for ch in 0..4 {
                    out.v[ch][l] = d;
                }
            }
        }
        17 => {
            for l in 0..LANES {
                let d = a[0][l] * b[0][l] + a[1][l] * b[1][l] + c[0][l];
                for ch in 0..4 {
                    out.v[ch][l] = d;
                }
            }
        }
        _ => {
            for l in 0..LANES {
                let la = [a[0][l], a[1][l], a[2][l], a[3][l]];
                let lb = [b[0][l], b[1][l], b[2][l], b[3][l]];
                let lc = [c[0][l], c[1][l], c[2][l], c[3][l]];
                let (v, p0, kill) = vec_lane(vop, la, lb, lc);
                for ch in 0..4 {
                    out.v[ch][l] = v[ch];
                }
                if let Some(p) = p0 {
                    out.p0_mask |= 1 << l;
                    if p {
                        out.p0_bits |= 1 << l;
                    }
                }
                if kill {
                    out.kill |= 1 << l;
                }
            }
        }
    }
    out
}

/// Reference semantics of one scalar operation for one pixel: (result, new p0, kill).
fn sca_lane(sop: u32, sa: f32, sb: f32, ps: f32) -> (f32, Option<bool>, bool) {
    let mut p0 = None;
    let mut kill = false;
    let r = match sop {
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
        23 => if sa >= sb { sa } else { sb },
        24 => if sa >= sb { sa } else { sb },
        25 => sa - sb,
        26 => sa - ps,
        27..=30 => {
            let hit = match sop {
                27 => sa == 0.0,
                28 => sa != 0.0,
                29 => sa > 0.0,
                _ => sa >= 0.0,
            };
            p0 = Some(hit);
            if hit { 0.0 } else { 1.0 }
        }
        31 => {
            if sa == 1.0 {
                p0 = Some(true);
                0.0
            } else {
                p0 = Some(false);
                if sa == 0.0 { 1.0 } else { sa }
            }
        }
        32 => {
            if sa - 1.0 <= 0.0 {
                p0 = Some(true);
                0.0
            } else {
                p0 = Some(false);
                sa - 1.0
            }
        }
        33 => {
            p0 = Some(false);
            f32::MAX
        }
        34 => {
            if sa == 0.0 {
                p0 = Some(true);
                0.0
            } else {
                p0 = Some(false);
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
            kill = hit;
            b2f(hit)
        }
        40 => sa.sqrt(),
        42 | 43 => sa * sb,
        44 | 45 => sa + sb,
        46 | 47 => sa - sb,
        48 => sa.sin(),
        49 => sa.cos(),
        _ => ps, // 50: retain previous
    };
    (r, p0, kill)
}

struct ScaOut {
    s: V,
    p0_mask: u32,
    p0_bits: u32,
    kill: u32,
}

#[inline(always)]
fn sca_exec(sop: u32, sa: &V, sb: &V, ps: &V) -> ScaOut {
    let mut out = ScaOut { s: [0.0; LANES], p0_mask: 0, p0_bits: 0, kill: 0 };
    macro_rules! fast {
        ($f:expr) => {{
            for l in 0..LANES {
                out.s[l] = $f(sa[l], sb[l], ps[l]);
            }
        }};
    }
    match sop {
        0 => fast!(|a: f32, b: f32, _p: f32| a + b),
        1 => fast!(|a: f32, _b: f32, p: f32| a + p),
        2 => fast!(|a: f32, b: f32, _p: f32| a * b),
        3 => fast!(|a: f32, _b: f32, p: f32| a * p),
        5 => fast!(|a: f32, b: f32, _p: f32| if a >= b { a } else { b }),
        6 => fast!(|a: f32, b: f32, _p: f32| if a < b { a } else { b }),
        11 => fast!(|a: f32, _b: f32, _p: f32| a - a.floor()),
        13 => fast!(|a: f32, _b: f32, _p: f32| a.floor()),
        25 => fast!(|a: f32, b: f32, _p: f32| a - b),
        26 => fast!(|a: f32, _b: f32, p: f32| a - p),
        40 => fast!(|a: f32, _b: f32, _p: f32| a.sqrt()),
        42 | 43 => fast!(|a: f32, b: f32, _p: f32| a * b),
        44 | 45 => fast!(|a: f32, b: f32, _p: f32| a + b),
        46 | 47 => fast!(|a: f32, b: f32, _p: f32| a - b),
        50 => out.s = *ps,
        _ => {
            for l in 0..LANES {
                let (r, p0, kill) = sca_lane(sop, sa[l], sb[l], ps[l]);
                out.s[l] = r;
                if let Some(p) = p0 {
                    out.p0_mask |= 1 << l;
                    if p {
                        out.p0_bits |= 1 << l;
                    }
                }
                if kill {
                    out.kill |= 1 << l;
                }
            }
        }
    }
    out
}

#[inline(always)]
fn masked_store(dst: &mut V, src: &V, m: u32) {
    if m & ALL == ALL {
        *dst = *src;
        return;
    }
    for l in 0..LANES {
        if (m >> l) & 1 != 0 {
            dst[l] = src[l];
        }
    }
}

#[inline(always)]
fn exec_alu(a: &Alu, b: &mut Batch, clause_mask: u32) {
    let m = clause_mask & pmask(a.pred, b.p0);
    if m == 0 {
        return;
    }
    let (mut ta, mut tb, mut tc, mut ts) = ([[0.0f32; LANES]; 4], [[0.0f32; LANES]; 4], [[0.0f32; LANES]; 4], [[0.0f32; LANES]; 4]);
    let zero = [[0.0f32; LANES]; 4];
    let zr = [&zero[0], &zero[1], &zero[2], &zero[3]];
    let oa = operand_refs(b, &a.a, &mut ta);
    let ob = if a.nops >= 2 { operand_refs(b, &a.b, &mut tb) } else { zr };
    let oc = if a.nops >= 3 { operand_refs(b, &a.c, &mut tc) } else { zr };
    let vo = vec_exec(a.vop, &oa, &ob, &oc);

    // scalar unit inputs
    let (sa, sb): (V, V) = match &a.s_in {
        ScalarIn::None => ([0.0; LANES], [0.0; LANES]),
        ScalarIn::Src3(o) => {
            let s = operand_refs(b, o, &mut ts);
            (*s[3], *s[0])
        }
        &ScalarIn::ConstTemp { a: ca, reg, comp, abs, neg } => {
            let mut tb = b.r[reg as usize][comp as usize];
            for l in 0..LANES {
                if abs {
                    tb[l] = tb[l].abs();
                }
                if neg {
                    tb[l] = -tb[l];
                }
            }
            ([ca[3]; LANES], tb)
        }
    };
    let so = sca_exec(a.sop, &sa, &sb, &b.ps);

    // side effects: p0 (scalar overrides vector), kill, previous-scalar register
    let mut wm = vo.p0_mask & m;
    let mut bits = vo.p0_bits;
    let swm = so.p0_mask & m;
    bits = (bits & !swm) | (so.p0_bits & swm);
    wm |= swm;
    b.p0 = (b.p0 & !wm) | (bits & wm);
    b.killed |= (vo.kill | so.kill) & m;
    masked_store(&mut b.ps, &so.s, m);

    // writes
    let mut v = vo.v;
    if a.vclamp {
        for ch in 0..4 {
            for l in 0..LANES {
                v[ch][l] = saturate(v[ch][l]);
            }
        }
    }
    let mut s = so.s;
    if a.sclamp {
        for l in 0..LANES {
            s[l] = saturate(s[l]);
        }
    }
    if a.export {
        // per component: vector only -> vector, scalar only -> scalar, both -> 1.0, neither -> 0.0 if scalar-dest-rel
        let comp = |i: usize| -> Option<(V, bool)> {
            let (vm, sm) = ((a.vmask >> i) & 1 != 0, (a.smask >> i) & 1 != 0);
            match (vm, sm) {
                (true, false) => Some((v[i], true)),
                (false, true) => Some((s, true)),
                (true, true) => Some(([1.0; LANES], true)),
                (false, false) if a.sdest_rel => Some(([0.0; LANES], true)),
                _ => None,
            }
        };
        match a.vdest {
            0..=3 => {
                for i in 0..4 {
                    if let Some((val, _)) = comp(i) {
                        masked_store(&mut b.colors[a.vdest as usize][i], &val, m);
                    }
                }
            }
            61 => {
                if let Some((val, _)) = comp(0) {
                    masked_store(&mut b.depth, &val, m);
                }
            }
            _ => {}
        }
    } else {
        let vi = (a.vdest as usize).min(127);
        for i in 0..4 {
            if (a.vmask >> i) & 1 != 0 {
                masked_store(&mut b.r[vi][i], &v[i], m);
            }
        }
        let si = (a.sdest as usize).min(127);
        for i in 0..4 {
            if (a.smask >> i) & 1 != 0 {
                masked_store(&mut b.r[si][i], &s, m);
            }
        }
    }
}

#[inline(always)]
fn exec_fetch(f: &Fetch, b: &mut Batch, clause_mask: u32, io: &mut dyn ShaderIo) {
    let m = clause_mask & pmask(f.pred, b.p0);
    if m == 0 {
        return;
    }
    for l in 0..LANES {
        if (m >> l) & 1 == 0 {
            continue;
        }
        let src = [b.r[f.src as usize][0][l], b.r[f.src as usize][1][l], b.r[f.src as usize][2][l], b.r[f.src as usize][3][l]];
        let coords = [src[f.src_sel[0] as usize], src[f.src_sel[1] as usize], src[f.src_sel[2] as usize]];
        let value = io.texture_fetch(&f.f, coords);
        let d = (f.dst as usize).min(127);
        for c in 0..4 {
            match f.dst_swiz[c] {
                s @ 0..=3 => b.r[d][c][l] = value[s as usize],
                4 => b.r[d][c][l] = 0.0,
                5 => b.r[d][c][l] = 1.0,
                _ => {}
            }
        }
    }
}

impl CompiledPs {
    /// Run the shader for the first `lanes` pixels of the batch. Interpolants must already be in `b.r[0..ninterp]`.
    pub fn run(&self, b: &mut Batch, lanes: usize, io: &mut dyn ShaderIo) {
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                // SAFETY: AVX2 and FMA support were just verified at runtime.
                unsafe { self.run_avx2(b, lanes, io) };
                return;
            }
        }
        self.run_impl(b, lanes, io);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn run_avx2(&self, b: &mut Batch, lanes: usize, io: &mut dyn ShaderIo) {
        self.run_impl(b, lanes, io);
    }

    #[inline(always)]
    fn run_impl(&self, b: &mut Batch, lanes: usize, io: &mut dyn ShaderIo) {
        let valid = if lanes >= LANES { ALL } else { (1u32 << lanes) - 1 };
        let mut act = valid;
        for cl in &self.clauses {
            let cm = act
                & match cl.pred {
                    None => ALL,
                    Some(true) => b.p0,
                    Some(false) => !b.p0 & ALL,
                };
            if cm != 0 {
                for inst in &cl.insts {
                    match inst {
                        Inst::Alu(a) => exec_alu(a, b, cm),
                        Inst::Fetch(f) => exec_fetch(f, b, cm, io),
                    }
                }
            }
            if cl.end {
                act &= !cm;
                if act == 0 {
                    break;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS.get_or_init(|| std::env::var_os("X360_NO_AVX2").is_none() && is_x86_feature_detected!("avx2"))
}
