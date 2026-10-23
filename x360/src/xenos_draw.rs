//! Software draw pipeline: vertex fetch -> vertex shader -> primitive assembly -> viewport
//! transform -> rasterization -> pixel shader -> blend -> render-target surface; plus the
//! EDRAM "resolve" (copy) into guest memory.
//!
//! Render targets are kept as plain f32 RGBA surfaces keyed by their EDRAM base. This ignores
//! MSAA and the real EDRAM tile layout (a later refinement); it is enough to see images.

use crate::gpu::{tiled_offset_2d, PHYS_VIEW};
use crate::memory::GuestMemory;
use crate::xenos_tex::TextureSampler;
use crate::xenos_vm::{Consts, Invocation, ShaderIo, ShaderKind, TextureFetch, VertexFetch, Vm};
use std::collections::HashMap;
use xenon_cpu::memory::Bus;

// Register indices used by the draw pipeline.
const REG_SURFACE_INFO: usize = 0x2000;
const REG_COLOR_INFO: usize = 0x2001;
const REG_WINDOW_SCISSOR_TL: usize = 0x2081;
const REG_WINDOW_SCISSOR_BR: usize = 0x2082;
const REG_COLOR_MASK: usize = 0x2104;
const REG_VPORT_XSCALE: usize = 0x210F;
const REG_VPORT_XOFFSET: usize = 0x2110;
const REG_VPORT_YSCALE: usize = 0x2111;
const REG_VPORT_YOFFSET: usize = 0x2112;
const REG_VTE_CNTL: usize = 0x2206;
const REG_BLENDCONTROL0: usize = 0x2201;
const REG_INDX_OFFSET: usize = 0x2102;
const REG_COPY_DEST_BASE: usize = 0x2319;
const REG_COPY_DEST_PITCH: usize = 0x231A;
const REG_COPY_CONTROL: usize = 0x2318;
const REG_COLOR_CLEAR: usize = 0x231E;
const REG_VERTEX_FETCH_CONSTS: usize = 0x4800;

#[derive(Clone)]
pub struct Surface {
    pub width: u32,
    pub height: u32,
    pub px: Vec<[f32; 4]>,
}

impl Surface {
    fn new(width: u32, height: u32) -> Self {
        Self { width, height, px: vec![[0.0; 4]; (width * height) as usize] }
    }
}

#[derive(Default)]
pub struct Renderer {
    /// Colour render targets by EDRAM base (RB_COLOR_INFO[11:0]).
    pub surfaces: HashMap<u32, Surface>,
    pub vs: Option<Vec<u32>>,
    pub ps: Option<Vec<u32>>,
    pub draws_rasterized: u64,
    pub triangles: u64,
    pub pixels_shaded: u64,
    pub resolves: u64,
    /// Things we skipped or didn't understand, counted by reason, for the end-of-run summary.
    pub skipped: HashMap<String, u64>,
}

/// Vertex/texture memory access for the shader VM.
struct GpuIo<'a> {
    mem: &'a GuestMemory,
    regs: &'a [u32],
}

pub fn apply_endian(raw_be: u32, mode: u32) -> u32 {
    // The GPU first assembles the dword little-endian from memory, then applies the fetch
    // constant's endian mode to it.
    let d = raw_be.swap_bytes();
    match mode & 3 {
        0 => d,
        1 => ((d & 0x00FF_00FF) << 8) | ((d >> 8) & 0x00FF_00FF), // 8in16
        2 => d.swap_bytes(),                                       // 8in32
        _ => d.rotate_left(16),                                    // 16in32
    }
}

/// Inverse of [`apply_endian`]: the big-endian memory word whose GPU view is `value`.
pub fn encode_endian(value: u32, mode: u32) -> u32 {
    match mode & 3 {
        0 => value.swap_bytes(),
        1 => (((value & 0x00FF_00FF) << 8) | ((value >> 8) & 0x00FF_00FF)).swap_bytes(),
        2 => value,
        _ => value.rotate_left(16).swap_bytes(),
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal: renormalize
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | 0x7F80_0000 | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

impl ShaderIo for GpuIo<'_> {
    fn vertex_fetch(&mut self, f: &VertexFetch, index: i64) -> [f32; 4] {
        let c = REG_VERTEX_FETCH_CONSTS + f.constant as usize * 2;
        let (d0, d1) = (self.regs[c], self.regs[c + 1]);
        let base = PHYS_VIEW | (d0 & 0x1FFF_FFFC);
        let endian = d1 & 3;
        let dword = (index * f.stride_dwords as i64 + f.offset_dwords as i64) as i32 as u32;
        let addr = base.wrapping_add(dword.wrapping_mul(4));
        let word = |i: u32| apply_endian(self.mem.read32(addr + i * 4), endian);
        let fl = |i: u32| f32::from_bits(word(i));
        let norm = f.is_normalized;
        let sgn = f.is_signed;
        match f.format {
            38 => [fl(0), fl(1), fl(2), fl(3)],         // 32_32_32_32_FLOAT
            57 => [fl(0), fl(1), fl(2), 1.0],           // 32_32_32_FLOAT
            37 => [fl(0), fl(1), 0.0, 1.0],             // 32_32_FLOAT
            36 => [fl(0), 0.0, 0.0, 1.0],               // 32_FLOAT
            6 => {
                // 8_8_8_8: bytes of the endian-corrected dword, x = lowest byte
                let w = word(0);
                let comp = |i: u32| {
                    let b = (w >> (8 * i)) as u8;
                    match (sgn, norm) {
                        (false, true) => b as f32 / 255.0,
                        (false, false) => b as f32,
                        (true, true) => (b as i8 as f32 / 127.0).max(-1.0),
                        (true, false) => b as i8 as f32,
                    }
                };
                [comp(0), comp(1), comp(2), comp(3)]
            }
            25 | 26 => {
                // 16_16 / 16_16_16_16 (integer, optionally normalized)
                let comp = |i: u32| {
                    let w = word(i / 2);
                    let h = (w >> (16 * (i % 2))) as u16;
                    match (sgn, norm) {
                        (false, true) => h as f32 / 65535.0,
                        (false, false) => h as f32,
                        (true, true) => (h as i16 as f32 / 32767.0).max(-1.0),
                        (true, false) => h as i16 as f32,
                    }
                };
                if f.format == 25 { [comp(0), comp(1), 0.0, 1.0] } else { [comp(0), comp(1), comp(2), comp(3)] }
            }
            31 | 32 => {
                // 16_16_FLOAT / 16_16_16_16_FLOAT
                let comp = |i: u32| {
                    let w = word(i / 2);
                    f16_to_f32((w >> (16 * (i % 2))) as u16)
                };
                if f.format == 31 { [comp(0), comp(1), 0.0, 1.0] } else { [comp(0), comp(1), comp(2), comp(3)] }
            }
            33 | 34 | 35 => {
                // 32 / 32_32 / 32_32_32_32 integers
                let n = f.format - 32;
                let comp = |i: u32| if i < n { word(i) as i32 as f32 } else { 0.0 };
                [comp(0), comp(1), comp(2), if n == 4 { word(3) as i32 as f32 } else { 1.0 }]
            }
            _ => [0.0, 0.0, 0.0, 1.0],
        }
    }

    fn texture_fetch(&mut self, f: &TextureFetch, coords: [f32; 3]) -> [f32; 4] {
        let sampler = TextureSampler { mem: self.mem, regs: self.regs };
        let desc = crate::xenos_tex::TexDesc::from_regs(self.regs, f.constant);
        let (mut u, mut v) = (coords[0], coords[1]);
        if f.unnormalized {
            u /= desc.width as f32;
            v /= desc.height as f32;
        }
        // instruction filter: 0 point, 1 linear, 2 base-map, 3 = take it from the fetch constant
        let filter = if f.mag_filter == 3 { desc.mag_filter } else { f.mag_filter };
        sampler.sample(f.constant, u, v, filter == 1)
    }
}

#[derive(Clone, Copy)]
struct Vertex {
    /// Screen-space x, y, z and the clip-space 1/w used for perspective correct interpolation.
    x: f32,
    y: f32,
    z: f32,
    inv_w: f32,
    interp: [[f32; 4]; 16],
}

fn blend_factor(f: u32, src: [f32; 4], dst: [f32; 4], c: usize, is_alpha: bool) -> f32 {
    let comp = |v: [f32; 4]| if is_alpha { v[3] } else { v[c] };
    match f {
        0 => 0.0,
        1 => 1.0,
        4 => comp(src),
        5 => 1.0 - comp(src),
        6 => src[3],
        7 => 1.0 - src[3],
        8 => comp(dst),
        9 => 1.0 - comp(dst),
        10 => dst[3],
        11 => 1.0 - dst[3],
        16 => if is_alpha { 1.0 } else { src[3].min(1.0 - dst[3]) },
        _ => 1.0,
    }
}

fn blend(ctl: u32, src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let (csrc, cfn, cdst) = (ctl & 0x1F, (ctl >> 5) & 7, (ctl >> 8) & 0x1F);
    let (asrc, afn, adst) = ((ctl >> 16) & 0x1F, (ctl >> 21) & 7, (ctl >> 24) & 0x1F);
    let combine = |fn_: u32, s: f32, d: f32| match fn_ {
        0 => s + d,
        1 => s - d,
        2 => s.min(d),
        3 => s.max(d),
        4 => d - s,
        _ => s + d,
    };
    let mut out = [0.0; 4];
    for c in 0..3 {
        let (sf, df) = (blend_factor(csrc, src, dst, c, false), blend_factor(cdst, src, dst, c, false));
        out[c] = if matches!(cfn, 2 | 3) { combine(cfn, src[c], dst[c]) } else { combine(cfn, src[c] * sf, dst[c] * df) };
    }
    let (sf, df) = (blend_factor(asrc, src, dst, 3, true), blend_factor(adst, src, dst, 3, true));
    out[3] = if matches!(afn, 2 | 3) { combine(afn, src[3], dst[3]) } else { combine(afn, src[3] * sf, dst[3] * df) };
    out
}

impl Renderer {
    pub fn load_shader(&mut self, kind: ShaderKind, ucode: Vec<u32>) {
        match kind {
            ShaderKind::Vertex => self.vs = Some(ucode),
            ShaderKind::Pixel => self.ps = Some(ucode),
        }
    }

    fn skip(&mut self, why: &str) {
        *self.skipped.entry(why.to_string()).or_insert(0) += 1;
    }

    fn surface(&mut self, regs: &[u32]) -> &mut Surface {
        let base = regs[REG_COLOR_INFO] & 0xFFF;
        let br = regs[REG_WINDOW_SCISSOR_BR];
        let (w, h) = ((br & 0x7FFF).clamp(1, 2560), ((br >> 16) & 0x7FFF).clamp(1, 1440));
        self.surfaces.entry(base).or_insert_with(|| Surface::new(w, h))
    }

    /// Execute one draw call into the bound colour render target.
    pub fn draw(&mut self, mem: &GuestMemory, regs: &[u32], prim: u32, indices: &[u32]) {
        let (Some(vs), Some(ps)) = (self.vs.clone(), self.ps.clone()) else {
            self.skip("draw without both shaders");
            return;
        };
        if !matches!(prim, 4 | 5 | 6 | 8 | 0xD) {
            self.skip(&format!("primitive type {prim}"));
            return;
        }
        let indx_offset = regs[REG_INDX_OFFSET] as i32;

        // ---- vertex stage (with a per-draw cache by index) ----
        let mut cache: HashMap<u32, Vertex> = HashMap::new();
        let vte = regs[REG_VTE_CNTL];
        let f = |i: usize| f32::from_bits(regs[i]);
        let (xs, xo, ys, yo) = (f(REG_VPORT_XSCALE), f(REG_VPORT_XOFFSET), f(REG_VPORT_YSCALE), f(REG_VPORT_YOFFSET));
        let mut io = GpuIo { mem, regs };
        let mut fetch_vertex = |idx: u32, io: &mut GpuIo<'_>| -> Option<Vertex> {
            if let Some(v) = cache.get(&idx) {
                return Some(*v);
            }
            let vm = Vm::new(&vs, ShaderKind::Vertex, Consts { regs, float_base: 0 });
            let mut inv = Invocation::new();
            inv.r[0][0] = (idx as i32).wrapping_add(indx_offset) as f32;
            vm.run(&mut inv, io).ok()?;
            let p = inv.exports.position;
            // perspective divide unless the vertex format says xy/z are already divided
            let (mut x, mut y, mut z) = (p[0], p[1], p[2]);
            let w0_is_inverse = (vte >> 10) & 1 != 0;
            let inv_w = if w0_is_inverse { p[3] } else if p[3] != 0.0 { 1.0 / p[3] } else { 1.0 };
            if (vte >> 8) & 1 == 0 {
                x *= inv_w;
                y *= inv_w;
            }
            if (vte >> 9) & 1 == 0 {
                z *= inv_w;
            }
            if vte & 1 != 0 {
                x *= xs;
            }
            if vte & 2 != 0 {
                x += xo;
            }
            if vte & 4 != 0 {
                y *= ys;
            }
            if vte & 8 != 0 {
                y += yo;
            }
            let v = Vertex { x, y, z, inv_w, interp: inv.exports.interpolators };
            cache.insert(idx, v);
            Some(v)
        };

        // ---- primitive assembly: list of triangles ----
        let n = indices.len();
        let mut tri_idx: Vec<[usize; 3]> = Vec::new();
        let mut extra: Vec<Vertex> = Vec::new(); // synthesized vertices (rect list 4th corner)
        let mut verts: Vec<Vertex> = Vec::with_capacity(n);
        for &i in indices {
            match fetch_vertex(i, &mut io) {
                Some(v) => verts.push(v),
                None => {
                    self.skip("vertex shader error");
                    return;
                }
            }
        }
        match prim {
            4 => (0..n / 3).for_each(|t| tri_idx.push([3 * t, 3 * t + 1, 3 * t + 2])),
            5 => (1..n.saturating_sub(1)).for_each(|t| tri_idx.push([0, t, t + 1])),
            6 => (0..n.saturating_sub(2)).for_each(|t| {
                // alternate winding so every triangle has the same facing
                if t % 2 == 0 { tri_idx.push([t, t + 1, t + 2]) } else { tri_idx.push([t + 1, t, t + 2]) }
            }),
            0xD => (0..n / 4).for_each(|q| {
                tri_idx.push([4 * q, 4 * q + 1, 4 * q + 2]);
                tri_idx.push([4 * q, 4 * q + 2, 4 * q + 3]);
            }),
            8 => {
                for r in 0..n / 3 {
                    let (a, b, c) = (verts[3 * r], verts[3 * r + 1], verts[3 * r + 2]);
                    // 4th corner of the rectangle: b + c - a (positions and every interpolant)
                    let mut d = a;
                    d.x = b.x + c.x - a.x;
                    d.y = b.y + c.y - a.y;
                    d.z = b.z + c.z - a.z;
                    d.inv_w = b.inv_w + c.inv_w - a.inv_w;
                    for k in 0..16 {
                        for ch in 0..4 {
                            d.interp[k][ch] = b.interp[k][ch] + c.interp[k][ch] - a.interp[k][ch];
                        }
                    }
                    let base = verts.len() + extra.len();
                    extra.push(d);
                    tri_idx.push([3 * r, 3 * r + 1, 3 * r + 2]);
                    tri_idx.push([3 * r + 1, 3 * r + 2, base]);
                }
            }
            _ => {}
        }
        verts.extend(extra);

        // ---- raster state ----
        let scissor_tl = regs[REG_WINDOW_SCISSOR_TL];
        let scissor_br = regs[REG_WINDOW_SCISSOR_BR];
        let (sx0, sy0) = ((scissor_tl & 0x7FFF) as i32, ((scissor_tl >> 16) & 0x7FFF) as i32);
        let (sx1, sy1) = ((scissor_br & 0x7FFF) as i32, ((scissor_br >> 16) & 0x7FFF) as i32);
        let blend_ctl = regs[REG_BLENDCONTROL0];
        let write_mask = regs[REG_COLOR_MASK] & 0xF;
        let consts = Consts { regs, float_base: 256 };
        let ps_vm = Vm::new(&ps, ShaderKind::Pixel, consts);

        // Split the borrow: take the surface out while rasterizing.
        let key = regs[REG_COLOR_INFO] & 0xFFF;
        let _ = self.surface(regs);
        let mut surf = self.surfaces.remove(&key).unwrap();
        let (w, h) = (surf.width as i32, surf.height as i32);
        self.draws_rasterized += 1;

        self.triangles += tri_idx.len() as u64;

        // Rasterize rows [y0, y1) of the surface into `rows` (the pixels of exactly those rows) for every
        // triangle, in order. Rows are independent, so large draws are split into bands run on separate
        // threads with results identical to the serial order. Returns pixels shaded.
        let shade_rows = |y0: i32, y1: i32, rows: &mut [[f32; 4]]| -> u64 {
            let mut io = GpuIo { mem, regs };
            let mut shaded = 0u64;
            for t in &tri_idx {
                let (a, b, c) = (verts[t[0]], verts[t[1]], verts[t[2]]);
                let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
                if area == 0.0 || !area.is_finite() {
                    continue;
                }
                let min_x = a.x.min(b.x).min(c.x).floor().max(sx0.max(0) as f32) as i32;
                let max_x = a.x.max(b.x).max(c.x).ceil().min(sx1.min(w) as f32) as i32;
                let min_y = (a.y.min(b.y).min(c.y).floor().max(sy0.max(0) as f32) as i32).max(y0);
                let max_y = (a.y.max(b.y).max(c.y).ceil().min(sy1.min(h) as f32) as i32).min(y1);
                let inv_area = 1.0 / area;
                for py in min_y..max_y {
                    for px in min_x..max_x {
                        let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
                        // barycentric weights (signed so both windings work)
                        let w0 = ((b.x - cx) * (c.y - cy) - (b.y - cy) * (c.x - cx)) * inv_area;
                        let w1 = ((c.x - cx) * (a.y - cy) - (c.y - cy) * (a.x - cx)) * inv_area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        // perspective-correct interpolation
                        let (p0, p1, p2) = (w0 * a.inv_w, w1 * b.inv_w, w2 * c.inv_w);
                        let denom = p0 + p1 + p2;
                        let (q0, q1, q2) = if denom != 0.0 { (p0 / denom, p1 / denom, p2 / denom) } else { (w0, w1, w2) };
                        let mut inv = Invocation::new();
                        for k in 0..16 {
                            for ch in 0..4 {
                                inv.r[k][ch] = a.interp[k][ch] * q0 + b.interp[k][ch] * q1 + c.interp[k][ch] * q2;
                            }
                        }
                        if ps_vm.run(&mut inv, &mut io).is_err() || inv.killed {
                            continue;
                        }
                        shaded += 1;
                        let src = inv.exports.colors[0];
                        let dst = &mut rows[((py - y0) * w + px) as usize];
                        let blended = blend(blend_ctl, src, *dst);
                        for ch in 0..4 {
                            if (write_mask >> ch) & 1 != 0 {
                                dst[ch] = blended[ch];
                            }
                        }
                    }
                }
            }
            shaded
        };

        // Rough cost: total clipped bounding-box area. Small draws stay on this thread (spawning isn't free).
        let work: i64 = tri_idx
            .iter()
            .map(|t| {
                let (a, b, c) = (verts[t[0]], verts[t[1]], verts[t[2]]);
                let wd = (a.x.max(b.x).max(c.x).ceil().min(sx1.min(w) as f32) - a.x.min(b.x).min(c.x).floor().max(0.0)).max(0.0) as i64;
                let ht = (a.y.max(b.y).max(c.y).ceil().min(sy1.min(h) as f32) - a.y.min(b.y).min(c.y).floor().max(0.0)).max(0.0) as i64;
                wd * ht
            })
            .sum();
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8);
        if threads > 1 && work > 40_000 && h >= threads as i32 {
            let band = ((h as usize + threads - 1) / threads) as i32;
            let shaded: u64 = std::thread::scope(|sc| {
                let shade_rows = &shade_rows;
                let handles: Vec<_> = surf
                    .px
                    .chunks_mut((band * w) as usize)
                    .enumerate()
                    .map(|(k, rows)| {
                        let y0 = k as i32 * band;
                        let y1 = (y0 + band).min(h);
                        sc.spawn(move || shade_rows(y0, y1, rows))
                    })
                    .collect();
                handles.into_iter().map(|hd| hd.join().unwrap()).sum()
            });
            self.pixels_shaded += shaded;
        } else {
            self.pixels_shaded += shade_rows(0, h, &mut surf.px);
        }
        self.surfaces.insert(key, surf);
    }

    /// EDRAM resolve: copy the bound render target to guest memory as tiled 8:8:8:8.
    pub fn resolve(&mut self, mem: &mut GuestMemory, regs: &[u32]) {
        let key = regs[REG_COLOR_INFO] & 0xFFF;
        let dest = regs[REG_COPY_DEST_BASE] & 0x1FFF_F000;
        let pitch_h = regs[REG_COPY_DEST_PITCH];
        let (pitch, height) = (pitch_h & 0x3FFF, (pitch_h >> 16) & 0x3FFF);
        if dest == 0 || pitch == 0 || height == 0 {
            self.skip("resolve without destination");
            return;
        }
        let Some(surf) = self.surfaces.get(&key) else {
            self.skip("resolve of a render target that was never drawn");
            return;
        };
        let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        let base = PHYS_VIEW | dest;
        for y in 0..height.min(surf.height) {
            for x in 0..pitch.min(surf.width) {
                let p = surf.px[(y * surf.width + x) as usize];
                // bytes in memory order R, G, B, A (little-endian dword A<<24|B<<16|G<<8|R, endian "none")
                let dword = to8(p[0]) | (to8(p[1]) << 8) | (to8(p[2]) << 16) | (to8(p[3]) << 24);
                let off = tiled_offset_2d(x, y, pitch, 2);
                mem.write_bytes(base + off, &dword.to_le_bytes());
            }
        }
        // optional colour clear after the copy
        if regs[REG_COPY_CONTROL] & (1 << 8) != 0 {
            let c = regs[REG_COLOR_CLEAR];
            let clear = [(c & 0xFF) as f32 / 255.0, ((c >> 8) & 0xFF) as f32 / 255.0, ((c >> 16) & 0xFF) as f32 / 255.0, (c >> 24) as f32 / 255.0];
            if let Some(s) = self.surfaces.get_mut(&key) {
                s.px.iter_mut().for_each(|p| *p = clear);
            }
        }
        self.resolves += 1;
    }

    pub fn summary(&self) -> String {
        let mut skipped: Vec<_> = self.skipped.iter().map(|(k, v)| format!("{k} x{v}")).collect();
        skipped.sort();
        format!(
            "render: {} draws, {} triangles, {} pixels shaded, {} resolves; skipped: {}",
            self.draws_rasterized,
            self.triangles,
            self.pixels_shaded,
            self.resolves,
            if skipped.is_empty() { "nothing".to_string() } else { skipped.join(", ") }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endian_modes_reconstruct_floats() {
        // A big-endian CPU stores 1.0f as bytes 3F 80 00 00; read32() gives 0x3F800000.
        let raw_be = 0x3F80_0000u32;
        assert_eq!(f32::from_bits(apply_endian(raw_be, 2)), 1.0); // 8in32: reverse -> numeric
        assert_ne!(f32::from_bits(apply_endian(raw_be, 0)), 1.0); // "none" leaves it byte-reversed
    }

    #[test]
    fn encode_endian_inverts_apply_endian() {
        for mode in 0..4 {
            for v in [0u32, 1, 0xDEAD_BEEF, 0x0102_0304] {
                assert_eq!(apply_endian(encode_endian(v, mode), mode), v, "mode {mode} value {v:#x}");
            }
        }
    }

    #[test]
    fn half_floats() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn blend_replace_and_alpha() {
        let src = [1.0, 0.0, 0.0, 0.5];
        let dst = [0.0, 0.0, 1.0, 1.0];
        // src*ONE + dst*ZERO (replace): ctl = csrc 1 | cfn 0 | cdst 0 | same for alpha
        let replace = 1 | (1 << 16);
        assert_eq!(blend(replace, src, dst), src);
        // classic alpha blend: src*SRC_ALPHA + dst*(1-SRC_ALPHA)
        let alpha = 6 | (7 << 8) | (1 << 16);
        let out = blend(alpha, src, dst);
        assert!((out[0] - 0.5).abs() < 1e-6 && (out[2] - 0.5).abs() < 1e-6);
    }
}
