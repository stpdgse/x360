//! Software draw pipeline: vertex fetch -> vertex shader -> primitive assembly -> viewport
//! transform -> rasterization -> pixel shader -> blend -> render-target surface; plus the
//! EDRAM "resolve" (copy) into guest memory.
//!
//! Render targets are kept as plain f32 RGBA surfaces keyed by their EDRAM base. This ignores
//! MSAA and the real EDRAM tile layout (a later refinement); it is enough to see images.

use serde::{Deserialize, Serialize};
use crate::gpu::{tiled_offset_2d, PHYS_VIEW};
use crate::memory::GuestMemory;
use crate::xenos_tex::{TexDesc, TextureSampler};
use crate::xenos_batch::{Batch, CompiledPs, LANES};
use crate::xenos_texcache::{DecodedTex, TexCache};
use std::sync::{Arc, Mutex, OnceLock};
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
const REG_DEPTH_INFO: usize = 0x2002;
const REG_DEPTHCONTROL: usize = 0x2200;
const REG_STENCILREFMASK: usize = 0x210D;
const REG_VPORT_ZSCALE: usize = 0x2113;
const REG_VPORT_ZOFFSET: usize = 0x2114;
const REG_DEPTH_CLEAR: usize = 0x231D;

thread_local! {
    static BATCH_POOL: std::cell::RefCell<Vec<Batch>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn take_batch() -> Batch {
    BATCH_POOL.with(|p| p.borrow_mut().pop()).unwrap_or_else(Batch::new)
}

fn give_batch(b: Batch) {
    BATCH_POOL.with(|p| p.borrow_mut().push(b));
}

/// A covered pixel waiting for the batched shader: index into the band's rows and its interpolated depth.
#[derive(Clone, Copy, Default)]
struct PendPx {
    di: usize,
    pz: f32,
}

/// X360_VERIFY_BATCH=1: run every batch through the reference interpreter too and report differences.
fn verify_batch(ps_vm: &Vm<'_>, io: &mut GpuIo<'_>, batch: &Batch, interp: &[[[f32; 4]; 16]], n: usize, ps: &[u32]) {
    static REPORTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for l in 0..n {
        let mut inv = Invocation::new();
        for k in 0..16 {
            inv.r[k] = interp[l][k];
        }
        let err = ps_vm.run(&mut inv, io).is_err();
        let killed_ref = err || inv.killed;
        let killed_batch = (batch.killed >> l) & 1 != 0;
        CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut bad = killed_ref != killed_batch;
        if !bad && !killed_ref {
            for ch in 0..4 {
                let (x, y) = (inv.exports.colors[0][ch], batch.colors[0][ch][l]);
                if !(x == y || (x - y).abs() <= 1e-3 * x.abs().max(y.abs()).max(1.0) || (x.is_nan() && y.is_nan())) {
                    bad = true;
                }
            }
        }
        if bad && REPORTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            let h = ps.iter().fold(0xcbf29ce484222325u64, |a, w| (a ^ *w as u64).wrapping_mul(0x100000001b3));
            println!(
                "[batch MISMATCH] shader {h:016x} lane {l}: reference killed={killed_ref} colour={:?}; batch killed={killed_batch} colour={:?} (after {} checks)",
                inv.exports.colors[0],
                [batch.colors[0][0][l], batch.colors[0][1][l], batch.colors[0][2][l], batch.colors[0][3][l]],
                CHECKED.load(std::sync::atomic::Ordering::Relaxed)
            );
        }
    }
}

/// X360_RENDER_SCALE=0.5 renders at that fraction of the console resolution (resolves scale back up), a large
/// speed-up for development. Default 1.0 = exact.
pub fn render_scale() -> f32 {
    static SCALE: OnceLock<f32> = OnceLock::new();
    *SCALE.get_or_init(|| std::env::var("X360_RENDER_SCALE").ok().and_then(|v| v.parse::<f32>().ok()).filter(|v| *v > 0.0 && *v <= 1.0).unwrap_or(1.0))
}

#[derive(Clone, Serialize, Deserialize)]
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

/// Depth (f32, 0..1) and stencil planes for one EDRAM depth base, the same size as the colour target.
pub struct DepthSurface {
    pub width: u32,
    pub height: u32,
    pub z: Vec<f32>,
    pub s: Vec<u8>,
}

impl DepthSurface {
    fn new(width: u32, height: u32, fill_z: f32) -> Self {
        Self { width, height, z: vec![fill_z; (width * height) as usize], s: vec![0; (width * height) as usize] }
    }

    fn grow(&mut self, w: u32, h: u32, fill_z: f32) {
        if self.width >= w && self.height >= h {
            return;
        }
        let (nw, nh) = (self.width.max(w), self.height.max(h));
        let mut g = DepthSurface::new(nw, nh, fill_z);
        for y in 0..self.height {
            let (a, b) = ((y * self.width) as usize, (y * nw) as usize);
            g.z[b..b + self.width as usize].copy_from_slice(&self.z[a..a + self.width as usize]);
            g.s[b..b + self.width as usize].copy_from_slice(&self.s[a..a + self.width as usize]);
        }
        *self = g;
    }
}

/// Compare function codes shared by depth and stencil tests: never, <, ==, <=, >, !=, >=, always.
#[inline]
fn cmp_f32(func: u32, a: f32, b: f32) -> bool {
    match func & 7 {
        0 => false,
        1 => a < b,
        2 => a == b,
        3 => a <= b,
        4 => a > b,
        5 => a != b,
        6 => a >= b,
        _ => true,
    }
}

#[inline]
fn cmp_u32(func: u32, a: u32, b: u32) -> bool {
    match func & 7 {
        0 => false,
        1 => a < b,
        2 => a == b,
        3 => a <= b,
        4 => a > b,
        5 => a != b,
        6 => a >= b,
        _ => true,
    }
}

fn stencil_op(op: u32, old: u8, reference: u8) -> u8 {
    match op & 7 {
        0 => old,
        1 => 0,
        2 => reference,
        3 => old.saturating_add(1),
        4 => old.saturating_sub(1),
        5 => !old,
        6 => old.wrapping_add(1),
        _ => old.wrapping_sub(1),
    }
}

/// Depth/stencil state of one draw, decoded from registers.
#[derive(Clone, Copy)]
struct DepthState {
    z_enable: bool,
    z_write: bool,
    z_func: u32,
    stencil_enable: bool,
    s_func: u32,
    s_fail: u32,
    s_zpass: u32,
    s_zfail: u32,
    s_ref: u8,
    s_mask: u8,
    s_wmask: u8,
}

impl DepthState {
    fn from_regs(regs: &[u32]) -> Self {
        let d = regs[REG_DEPTHCONTROL];
        let sr = regs[REG_STENCILREFMASK];
        Self {
            z_enable: d & 2 != 0,
            z_write: d & 4 != 0,
            z_func: (d >> 4) & 7,
            stencil_enable: d & 1 != 0,
            s_func: (d >> 8) & 7,
            s_fail: (d >> 11) & 7,
            s_zpass: (d >> 14) & 7,
            s_zfail: (d >> 17) & 7,
            s_ref: sr as u8,
            s_mask: (sr >> 8) as u8,
            s_wmask: (sr >> 16) as u8,
        }
    }

    fn active(&self) -> bool {
        self.z_enable || self.stencil_enable
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct Renderer {
    /// Colour render targets by EDRAM base (RB_COLOR_INFO[11:0]).
    pub surfaces: HashMap<u32, Surface>,
    pub vs: Option<Vec<u32>>,
    pub ps: Option<Vec<u32>>,
    pub draws_rasterized: u64,
    pub triangles: u64,
    pub pixels_shaded: u64,
    pub resolves: u64,
    /// Profile (host time): whole draw calls, their vertex stage, their raster+pixel stage, and resolves.
    #[serde(skip)]
    pub t_draw: std::time::Duration,
    #[serde(skip)]
    pub t_vertex: std::time::Duration,
    #[serde(skip)]
    pub t_raster: std::time::Duration,
    #[serde(skip)]
    pub t_resolve: std::time::Duration,
    #[serde(skip)]
    pub verts_shaded: u64,
    #[serde(skip)]
    pub last_covered: u64,
    #[serde(skip)]
    pub last_rej_s: u64,
    #[serde(skip)]
    pub last_rej_z: u64,
    /// Depth/stencil planes by EDRAM depth base (not saved in save states; rebuilt as the game draws).
    #[serde(skip)]
    pub depths: HashMap<u32, DepthSurface>,
    /// Decoded textures shared by every draw (and every raster thread).
    #[serde(skip)]
    pub tex_cache: Mutex<TexCache>,
    /// Things we skipped or didn't understand, counted by reason, for the end-of-run summary.
    pub skipped: HashMap<String, u64>,
}

/// Vertex/texture memory access for the shader VM.
type TexSlot = (TexDesc, Arc<Option<DecodedTex>>);

struct GpuIo<'a> {
    mem: &'a GuestMemory,
    regs: &'a [u32],
    /// Shared decoded-texture cache, or None for tiny draws where decoding a whole texture isn't worth it.
    cache: Option<&'a Mutex<TexCache>>,
    /// Per-draw lazily resolved texture slots (fetch constants 0..31).
    slots: &'a [OnceLock<TexSlot>; 32],
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
        static NO_TEX: OnceLock<bool> = OnceLock::new();
        if *NO_TEX.get_or_init(|| std::env::var_os("X360_DEBUG_NO_TEX").is_some()) {
            return [0.5, 0.5, 0.5, 1.0]; // profiling aid: measure the shader without texture sampling
        }
        if let Some(cache) = self.cache {
            let slot = self.slots[(f.constant & 31) as usize].get_or_init(|| {
                let desc = TexDesc::from_regs(self.regs, f.constant);
                let tex = cache.lock().unwrap().get(self.mem, &desc);
                (desc, tex)
            });
            if let Some(tex) = slot.1.as_ref() {
                let desc = &slot.0;
                let (mut u, mut v) = (coords[0], coords[1]);
                if f.unnormalized {
                    u /= desc.width as f32;
                    v /= desc.height as f32;
                }
                let filter = if f.mag_filter == 3 { desc.mag_filter } else { f.mag_filter };
                return tex.sample(desc, u, v, filter == 1);
            }
        }
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
        let sc = render_scale();
        let (fw, fh) = ((br & 0x7FFF).clamp(1, 2560), ((br >> 16) & 0x7FFF).clamp(1, 1440));
        let (w, h) = (((fw as f32 * sc).ceil() as u32).max(1), ((fh as f32 * sc).ceil() as u32).max(1));
        let s = self.surfaces.entry(base).or_insert_with(|| Surface::new(w, h));
        // Predicated tiling draws each tile with a scissor covering only that tile's rows of one full-frame
        // surface, so a surface created by the first tile must grow when a later tile reaches further.
        if s.width < w || s.height < h {
            let (nw, nh) = (s.width.max(w), s.height.max(h));
            let mut grown = Surface::new(nw, nh);
            for y in 0..s.height {
                let (a, b) = ((y * s.width) as usize, (y * nw) as usize);
                grown.px[b..b + s.width as usize].copy_from_slice(&s.px[a..a + s.width as usize]);
            }
            *s = grown;
        }
        s
    }

    /// Execute one draw call into the bound colour render target.
    pub fn draw(&mut self, mem: &GuestMemory, regs: &[u32], prim: u32, indices: &[u32], depth_only: bool) {
        let t0 = std::time::Instant::now();
        let shaded0 = self.pixels_shaded;
        let tris0 = self.triangles;
        self.draw_inner(mem, regs, prim, indices, depth_only);
        let el = t0.elapsed();
        self.t_draw += el;
        if el.as_millis() > 300 && std::env::var_os("X360_LOG_SLOW_DRAWS").is_some() {
            let shaded = self.pixels_shaded - shaded0;
            println!(
                "[slow draw] {:.0?}: prim {prim} {} indices, {} triangles, {} px covered, {} stencil-rejected, {} depth-rejected, {} pixels shaded ({:.0} ns/px) depthctl {:#x} stencilref {:#x} depth_info {:#x}, ps {:016x}",
                el,
                indices.len(),
                self.triangles - tris0,
                self.last_covered,
                self.last_rej_s,
                self.last_rej_z,
                shaded,
                el.as_nanos() as f64 / shaded.max(1) as f64,
                regs[REG_DEPTHCONTROL],
                regs[REG_STENCILREFMASK],
                regs[REG_DEPTH_INFO],
                self.ps.as_ref().map(|p| p.iter().fold(0u64, |h, w| h.wrapping_mul(0x100000001b3).wrapping_add(*w as u64))).unwrap_or(0)
            );
            if std::env::var_os("X360_LOG_SLOW_DRAWS").map(|v| v == "2").unwrap_or(false) {
                if let Some(ps) = &self.ps {
                    println!("{}", crate::xenos_vm::disassemble(ps));
                }
            }
        }
    }

    fn draw_inner(&mut self, mem: &GuestMemory, regs: &[u32], prim: u32, indices: &[u32], depth_only: bool) {
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
        let (zs, zo) = (f(REG_VPORT_ZSCALE), f(REG_VPORT_ZOFFSET));
        let slots: [OnceLock<TexSlot>; 32] = std::array::from_fn(|_| OnceLock::new());
        let mut io = GpuIo { mem, regs, cache: None, slots: &slots };
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
            if vte & 0x10 != 0 {
                z *= zs;
            }
            if vte & 0x20 != 0 {
                z += zo;
            }
            let sc = render_scale();
            let v = Vertex { x: x * sc, y: y * sc, z, inv_w, interp: inv.exports.interpolators };
            cache.insert(idx, v);
            Some(v)
        };

        // ---- primitive assembly: list of triangles ----
        let t_v = std::time::Instant::now();
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
        self.verts_shaded += cache.len() as u64;
        self.t_vertex += t_v.elapsed();
        let t_r = std::time::Instant::now();

        // ---- raster state ----
        let scissor_tl = regs[REG_WINDOW_SCISSOR_TL];
        let scissor_br = regs[REG_WINDOW_SCISSOR_BR];
        let sc = render_scale();
        let scl = |v: u32, up: bool| if up { (v as f32 * sc).ceil() as i32 } else { (v as f32 * sc).floor() as i32 };
        let (sx0, sy0) = (scl(scissor_tl & 0x7FFF, false), scl((scissor_tl >> 16) & 0x7FFF, false));
        let (sx1, sy1) = (scl(scissor_br & 0x7FFF, true), scl((scissor_br >> 16) & 0x7FFF, true));
        let blend_ctl = regs[REG_BLENDCONTROL0];
        let write_mask = if depth_only { 0 } else { regs[REG_COLOR_MASK] & 0xF };
        let consts = Consts { regs, float_base: 256 };
        let ps_vm = Vm::new(&ps, ShaderKind::Pixel, consts);

        // Split the borrow: take the surface out while rasterizing.
        let key = regs[REG_COLOR_INFO] & 0xFFF;
        let _ = self.surface(regs);
        let mut surf = self.surfaces.remove(&key).unwrap();
        let (w, h) = (surf.width as i32, surf.height as i32);
        self.draws_rasterized += 1;

        // depth/stencil target (only when the draw tests or writes it)
        let dstate = DepthState::from_regs(regs);
        let depth_key = regs[REG_DEPTH_INFO] & 0xFFF;
        let far_fill = if matches!(dstate.z_func, 4 | 6) { 0.0 } else { 1.0 };
        let mut depth = if dstate.active() {
            let mut d = self.depths.remove(&depth_key).unwrap_or_else(|| DepthSurface::new(surf.width, surf.height, far_fill));
            d.grow(surf.width, surf.height, far_fill);
            Some(d)
        } else {
            None
        };
        let dwidth = depth.as_ref().map(|d| d.width as i32).unwrap_or(w);
        debug_assert!(depth.is_none() || dwidth == w || true);

        self.triangles += tri_idx.len() as u64;

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
        let tex_cache = &self.tex_cache;
        let use_cache = work >= 16;
        let rej_s = std::sync::atomic::AtomicU64::new(0);
        let rej_z = std::sync::atomic::AtomicU64::new(0);
        let covered = std::sync::atomic::AtomicU64::new(0);
        // Depth-only passes (RB_MODECONTROL = 5) with a tiny pixel shader only need the depth/stencil update.
        let skip_ps = depth_only && ps.len() <= 12;

        // Rasterize rows [y0, y1) of the surface into `rows` (the pixels of exactly those rows) for every
        // triangle, in order. Rows are independent, so large draws are split into bands run on separate
        // threads with results identical to the serial order. Returns pixels shaded.
        // Compile the pixel shader once per draw (constants resolved, operands decoded) when it is worth it and uses
        // only per-pixel-uniform control flow; otherwise pixels go through the reference interpreter one by one.
        let compiled: Option<CompiledPs> = if !skip_ps && work >= 24 && std::env::var_os("X360_NO_BATCH").is_none() {
            CompiledPs::compile(&ps, regs)
        } else {
            None
        };
        let verify = compiled.is_some() && std::env::var_os("X360_VERIFY_BATCH").is_some();

        let shade_rows = |y0: i32, y1: i32, rows: &mut [[f32; 4]], mut drows: Option<(&mut [f32], &mut [u8])>| -> u64 {
            let mut io = GpuIo { mem, regs, cache: if use_cache { Some(tex_cache) } else { None }, slots: &slots };
            let mut shaded = 0u64;
            let (mut covered_local, mut rej_s_local, mut rej_z_local) = (0u64, 0u64, 0u64);
            let mut batch = take_batch();
            let mut pend: [PendPx; LANES] = [PendPx::default(); LANES];
            let mut npend = 0usize;
            let mut interp_copy: Vec<[[f32; 4]; 16]> = if verify { vec![[[0.0; 4]; 16]; LANES] } else { Vec::new() };
            let ninterp = compiled.as_ref().map(|c| c.ninterp).unwrap_or(16);

            // Run the compiled shader on the pending pixels and finish them in order (stencil/depth writes, blend).
            macro_rules! flush {
                () => {{
                    if npend > 0 {
                        let cp = compiled.as_ref().unwrap();
                        cp.run(&mut batch, npend, &mut io);
                        if verify {
                            verify_batch(&ps_vm, &mut io, &batch, &interp_copy, npend, &ps);
                        }
                        for l in 0..npend {
                            if (batch.killed >> l) & 1 != 0 {
                                continue;
                            }
                            let pp = pend[l];
                            shaded += 1;
                            if let Some((zrow, srow)) = drows.as_mut() {
                                if dstate.stencil_enable {
                                    let n = stencil_op(dstate.s_zpass, srow[pp.di], dstate.s_ref);
                                    srow[pp.di] = (srow[pp.di] & !dstate.s_wmask) | (n & dstate.s_wmask);
                                }
                                if dstate.z_enable && dstate.z_write {
                                    zrow[pp.di] = pp.pz;
                                }
                            }
                            let src = [batch.colors[0][0][l], batch.colors[0][1][l], batch.colors[0][2][l], batch.colors[0][3][l]];
                            let dst = &mut rows[pp.di];
                            let blended = blend(blend_ctl, src, *dst);
                            for ch in 0..4 {
                                if (write_mask >> ch) & 1 != 0 {
                                    dst[ch] = blended[ch];
                                }
                            }
                        }
                        npend = 0;
                    }
                }};
            }

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
                if min_x >= max_x || min_y >= max_y {
                    continue;
                }
                let inv_area = 1.0 / area;
                // barycentric weights are affine in x along a row: w_k = al_k * cx + be_k(cy)
                let (al0, al1) = ((b.y - c.y) * inv_area, (c.y - a.y) * inv_area);
                let al2 = -(al0 + al1);
                for py in min_y..max_y {
                    let cy = py as f32 + 0.5;
                    let be0 = (b.x * (c.y - cy) - (b.y - cy) * c.x) * inv_area;
                    let be1 = (c.x * (a.y - cy) - (c.y - cy) * a.x) * inv_area;
                    let be2 = 1.0 - be0 - be1;
                    // pixel range where all three weights can be >= 0 (padded by one pixel for rounding; the exact
                    // per-pixel test below decides coverage)
                    let (mut lo, mut hi) = (f32::NEG_INFINITY, f32::INFINITY);
                    let mut empty = false;
                    for (al, be) in [(al0, be0), (al1, be1), (al2, be2)] {
                        if al > 0.0 {
                            lo = lo.max(-be / al);
                        } else if al < 0.0 {
                            hi = hi.min(-be / al);
                        } else if be < 0.0 {
                            empty = true;
                        }
                    }
                    if empty || lo > hi + 1.0 {
                        continue;
                    }
                    let row_min = if lo.is_finite() { ((lo - 0.5).floor() as i64 - 1).clamp(min_x as i64, max_x as i64) as i32 } else { min_x };
                    let row_max = if hi.is_finite() { ((hi - 0.5).ceil() as i64 + 2).clamp(min_x as i64, max_x as i64) as i32 } else { max_x };
                    for px in row_min..row_max {
                        let cx = px as f32 + 0.5;
                        // barycentric weights (signed so both windings work)
                        let w0 = ((b.x - cx) * (c.y - cy) - (b.y - cy) * (c.x - cx)) * inv_area;
                        let w1 = ((c.x - cx) * (a.y - cy) - (c.y - cy) * (a.x - cx)) * inv_area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        // early depth/stencil test (screen-space z is affine in the barycentrics)
                        let di = ((py - y0) * w + px) as usize;
                        let pz = w0 * a.z + w1 * b.z + w2 * c.z;
                        covered_local += 1;
                        if let Some((zrow, srow)) = drows.as_mut() {
                            let (old_z, old_s) = (zrow[di], srow[di]);
                            if dstate.stencil_enable {
                                let pass = cmp_u32(dstate.s_func, (dstate.s_ref & dstate.s_mask) as u32, (old_s & dstate.s_mask) as u32);
                                if !pass {
                                    rej_s_local += 1;
                                    let n = stencil_op(dstate.s_fail, old_s, dstate.s_ref);
                                    srow[di] = (old_s & !dstate.s_wmask) | (n & dstate.s_wmask);
                                    continue;
                                }
                            }
                            if dstate.z_enable && !cmp_f32(dstate.z_func, pz, old_z) {
                                rej_z_local += 1;
                                if dstate.stencil_enable {
                                    let n = stencil_op(dstate.s_zfail, old_s, dstate.s_ref);
                                    srow[di] = (old_s & !dstate.s_wmask) | (n & dstate.s_wmask);
                                }
                                continue;
                            }
                        }
                        if skip_ps {
                            // depth-only pass: the shader has nothing to contribute
                            shaded += 1;
                            if let Some((zrow, srow)) = drows.as_mut() {
                                if dstate.stencil_enable {
                                    let n = stencil_op(dstate.s_zpass, srow[di], dstate.s_ref);
                                    srow[di] = (srow[di] & !dstate.s_wmask) | (n & dstate.s_wmask);
                                }
                                if dstate.z_enable && dstate.z_write {
                                    zrow[di] = pz;
                                }
                            }
                            continue;
                        }
                        // perspective-correct interpolation
                        let (p0, p1, p2) = (w0 * a.inv_w, w1 * b.inv_w, w2 * c.inv_w);
                        let denom = p0 + p1 + p2;
                        let (q0, q1, q2) = if denom != 0.0 { (p0 / denom, p1 / denom, p2 / denom) } else { (w0, w1, w2) };
                        if compiled.is_some() {
                            if npend == 0 {
                                batch.reset(compiled.as_ref().unwrap().nregs);
                            }
                            let l = npend;
                            for k in 0..ninterp {
                                for ch in 0..4 {
                                    batch.r[k][ch][l] = a.interp[k][ch] * q0 + b.interp[k][ch] * q1 + c.interp[k][ch] * q2;
                                }
                            }
                            if verify {
                                for k in 0..16 {
                                    for ch in 0..4 {
                                        interp_copy[l][k][ch] = a.interp[k][ch] * q0 + b.interp[k][ch] * q1 + c.interp[k][ch] * q2;
                                    }
                                }
                            }
                            pend[l] = PendPx { di, pz };
                            npend += 1;
                            if npend == LANES {
                                flush!();
                            }
                            continue;
                        }
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
                        if let Some((zrow, srow)) = drows.as_mut() {
                            if dstate.stencil_enable {
                                let n = stencil_op(dstate.s_zpass, srow[di], dstate.s_ref);
                                srow[di] = (srow[di] & !dstate.s_wmask) | (n & dstate.s_wmask);
                            }
                            if dstate.z_enable && dstate.z_write {
                                zrow[di] = pz;
                            }
                        }
                        let src = inv.exports.colors[0];
                        let dst = &mut rows[di];
                        let blended = blend(blend_ctl, src, *dst);
                        for ch in 0..4 {
                            if (write_mask >> ch) & 1 != 0 {
                                dst[ch] = blended[ch];
                            }
                        }
                    }
                }
                // finish this triangle's pixels before the next one so overlapping triangles keep their order
                flush!();
            }
            give_batch(batch);
            covered.fetch_add(covered_local, std::sync::atomic::Ordering::Relaxed);
            rej_s.fetch_add(rej_s_local, std::sync::atomic::Ordering::Relaxed);
            rej_z.fetch_add(rej_z_local, std::sync::atomic::Ordering::Relaxed);
            shaded
        };

        // Bands of a few rows are independent (every band runs all triangles in order over its own rows), so large
        // draws are spread over the rayon pool with work stealing; small draws stay on this thread.
        let threads = rayon::current_num_threads();
        if threads > 1 && work > 20_000 && h >= 16 {
            use rayon::prelude::*;
            let band = ((h as usize / (threads * 8)).max(4)) as i32;
            let chunk = (band * w) as usize;
            let mut bands: Vec<(i32, &mut [[f32; 4]], Option<(&mut [f32], &mut [u8])>)> = Vec::new();
            match depth.as_mut() {
                Some(d) => {
                    for (k, ((rows, z), st)) in surf.px.chunks_mut(chunk).zip(d.z.chunks_mut(chunk)).zip(d.s.chunks_mut(chunk)).enumerate() {
                        bands.push((k as i32 * band, rows, Some((z, st))));
                    }
                }
                None => {
                    for (k, rows) in surf.px.chunks_mut(chunk).enumerate() {
                        bands.push((k as i32 * band, rows, None));
                    }
                }
            }
            let shaded: u64 = bands
                .into_par_iter()
                .map(|(y0, rows, dr)| shade_rows(y0, (y0 + band).min(h), rows, dr))
                .sum();
            self.pixels_shaded += shaded;
        } else {
            let dr = depth.as_mut().map(|d| (&mut d.z[..], &mut d.s[..]));
            self.pixels_shaded += shade_rows(0, h, &mut surf.px, dr);
        }
        self.last_covered = covered.load(std::sync::atomic::Ordering::Relaxed);
        self.last_rej_s = rej_s.load(std::sync::atomic::Ordering::Relaxed);
        self.last_rej_z = rej_z.load(std::sync::atomic::Ordering::Relaxed);
        if let Some(d) = depth {
            self.depths.insert(depth_key, d);
        }
        self.surfaces.insert(key, surf);
        self.t_raster += t_r.elapsed();
    }

    /// EDRAM resolve: copy the bound render target to guest memory as tiled 8:8:8:8.
    pub fn resolve(&mut self, mem: &mut GuestMemory, regs: &[u32]) {
        let t0 = std::time::Instant::now();
        self.resolve_inner(mem, regs);
        let el = t0.elapsed();
        self.t_resolve += el;
        if el.as_millis() >= 5 && std::env::var_os("X360_LOG_SLOW_DRAWS").is_some() {
            let pitch_h = regs[REG_COPY_DEST_PITCH];
            println!("[slow resolve] {el:.1?}: dest {}x{} src surface {:#x}", pitch_h & 0x3FFF, (pitch_h >> 16) & 0x3FFF, regs[REG_COLOR_INFO] & 0xFFF);
        }
    }

    fn resolve_inner(&mut self, mem: &mut GuestMemory, regs: &[u32]) {
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
        let sc = render_scale();
        // Copy region = the window scissor (per-tile copies use their own rows), clipped to the destination size.
        let (tl, br) = (regs[REG_WINDOW_SCISSOR_TL], regs[REG_WINDOW_SCISSOR_BR]);
        let (rx0, ry0) = ((tl & 0x7FFF).min(pitch), ((tl >> 16) & 0x7FFF).min(height));
        let (rx1, ry1) = ((br & 0x7FFF).min(pitch), ((br >> 16) & 0x7FFF).min(height));
        let (rx0, ry0, rx1, ry1) = if rx1 <= rx0 || ry1 <= ry0 { (0, 0, pitch, height) } else { (rx0, ry0, rx1, ry1) };
        let full_w = ((surf.width as f32 / sc).round() as u32).min(rx1);
        let full_h = ((surf.height as f32 / sc).round() as u32).min(ry1);
        for y in ry0..full_h {
            let sy = (((y as f32 + 0.5) * sc) as u32).min(surf.height - 1);
            let row = &surf.px[(sy * surf.width) as usize..((sy + 1) * surf.width) as usize];
            for x in rx0..full_w {
                let sx = (((x as f32 + 0.5) * sc) as u32).min(surf.width - 1);
                let p = row[sx as usize];
                // bytes in memory order R, G, B, A (endian "none")
                let bytes = [to8(p[0]) as u8, to8(p[1]) as u8, to8(p[2]) as u8, to8(p[3]) as u8];
                mem.write32(base + tiled_offset_2d(x, y, pitch, 2), u32::from_be_bytes(bytes));
            }
        }
        // optional colour clear after the copy
        if regs[REG_COPY_CONTROL] & (1 << 8) != 0 {
            let c = regs[REG_COLOR_CLEAR];
            let clear = [(c & 0xFF) as f32 / 255.0, ((c >> 8) & 0xFF) as f32 / 255.0, ((c >> 16) & 0xFF) as f32 / 255.0, (c >> 24) as f32 / 255.0];
            if let Some(s) = self.surfaces.get_mut(&key) {
                let (cx0, cx1) = (((rx0 as f32 * sc) as u32).min(s.width), (((rx1 as f32 * sc).ceil()) as u32).min(s.width));
                let (cy0, cy1) = (((ry0 as f32 * sc) as u32).min(s.height), (((ry1 as f32 * sc).ceil()) as u32).min(s.height));
                for y in cy0..cy1 {
                    s.px[(y * s.width + cx0) as usize..(y * s.width + cx1) as usize].iter_mut().for_each(|p| *p = clear);
                }
            }
        }
        // optional depth/stencil clear after the copy
        if regs[REG_COPY_CONTROL] & (1 << 9) != 0 {
            let v = regs[REG_DEPTH_CLEAR];
            let (z, st) = ((v >> 8) as f32 / 16_777_215.0, v as u8);
            if let Some(d) = self.depths.get_mut(&(regs[REG_DEPTH_INFO] & 0xFFF)) {
                d.z.iter_mut().for_each(|p| *p = z);
                d.s.iter_mut().for_each(|p| *p = st);
            }
        }
        self.resolves += 1;
    }

    pub fn summary(&self) -> String {
        let mut skipped: Vec<_> = self.skipped.iter().map(|(k, v)| format!("{k} x{v}")).collect();
        skipped.sort();
        format!(
            "render: {} draws, {} triangles, {} verts, {} pixels shaded, {} resolves | host time: draws {:.1?} (vertex {:.1?}, raster {:.1?}), resolves {:.1?}; skipped: {}",
            self.draws_rasterized,
            self.triangles,
            self.verts_shaded,
            self.pixels_shaded,
            self.resolves,
            self.t_draw,
            self.t_vertex,
            self.t_raster,
            self.t_resolve,
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
