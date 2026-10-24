//! Decoded-texture cache for the software renderer. A whole mip-0 texture is decoded once into packed RGBA8, so the
//! pixel loop never touches guest memory, tiling, endian swaps or DXT blocks again. Entries live across draws and
//! are reused only while the texture's guest bytes still hash the same.

use crate::gpu::{tiled_offset_2d, PHYS_VIEW};
use crate::memory::GuestMemory;
use crate::xenos_draw::apply_endian;
use crate::xenos_tex::{dxt_texel, expand565, TexDesc, TextureSampler};
use std::collections::HashMap;
use std::sync::Arc;

/// Packed RGBA8 texels (r | g << 8 | b << 16 | a << 24), row-major, `width` x `height`.
pub struct DecodedTex {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<u32>,
}

#[inline]
fn to8(v: f32) -> u32 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32
}

#[inline]
fn pack(c: [f32; 4]) -> u32 {
    to8(c[0]) | (to8(c[1]) << 8) | (to8(c[2]) << 16) | (to8(c[3]) << 24)
}

impl DecodedTex {
    /// Decode the texture described by `t` from `words` (the raw dwords of its whole surface as stored in guest
    /// memory). None for formats the sampler doesn't support.
    pub fn decode(t: &TexDesc, words: &[u32]) -> Option<Self> {
        let (bw, bytes, log) = t.layout()?;
        let pitch_elems = t.pitch.max(t.width).div_ceil(bw);
        // endian-corrected little-endian view of the surface, exactly what TextureSampler::texel reads
        let dword = |off: u32| apply_endian(words.get((off / 4) as usize).copied().unwrap_or(0), t.endian);
        let byte = |off: u32| (dword(off & !3) >> (8 * (off & 3))) as u8;
        let offset_of = |ex: u32, ey: u32| {
            if t.tiled { tiled_offset_2d(ex, ey, pitch_elems, log) } else { (ey * pitch_elems + ex) * bytes }
        };
        let mut texels = vec![0u32; (t.width * t.height) as usize];
        match t.format {
            2 | 8 | 9 | 10 | 4 | 3 | 15 | 6 | 14 => {
                for y in 0..t.height {
                    for x in 0..t.width {
                        let off = offset_of(x, y);
                        let c = match t.format {
                            2 | 8 | 9 => {
                                let v = byte(off) as f32 / 255.0;
                                if t.format == 8 { [0.0, 0.0, 0.0, v] } else { [v, 0.0, 0.0, 1.0] }
                            }
                            10 => [byte(off) as f32 / 255.0, byte(off + 1) as f32 / 255.0, 0.0, 1.0],
                            4 => {
                                let h = ((byte(off + 1) as u16) << 8) | byte(off) as u16;
                                let c = expand565(h);
                                [c[0], c[1], c[2], 1.0]
                            }
                            3 => {
                                let h = ((byte(off + 1) as u16) << 8) | byte(off) as u16;
                                [((h >> 10) & 31) as f32 / 31.0, ((h >> 5) & 31) as f32 / 31.0, (h & 31) as f32 / 31.0, (h >> 15) as f32]
                            }
                            15 => {
                                let h = ((byte(off + 1) as u16) << 8) | byte(off) as u16;
                                [((h >> 8) & 15) as f32 / 15.0, ((h >> 4) & 15) as f32 / 15.0, (h & 15) as f32 / 15.0, ((h >> 12) & 15) as f32 / 15.0]
                            }
                            _ => {
                                let d = dword(off & !3);
                                [(d & 0xFF) as f32 / 255.0, ((d >> 8) & 0xFF) as f32 / 255.0, ((d >> 16) & 0xFF) as f32 / 255.0, (d >> 24) as f32 / 255.0]
                            }
                        };
                        texels[(y * t.width + x) as usize] = pack(c);
                    }
                }
            }
            18 | 19 | 20 => {
                for by in 0..t.height.div_ceil(4) {
                    for bx in 0..t.width.div_ceil(4) {
                        let off = offset_of(bx, by);
                        let mut block = [0u8; 16];
                        for i in 0..(bytes / 4) {
                            block[(i * 4) as usize..(i * 4 + 4) as usize].copy_from_slice(&dword(off + i * 4).to_le_bytes());
                        }
                        for ty in 0..4 {
                            for tx in 0..4 {
                                let (x, y) = (bx * 4 + tx, by * 4 + ty);
                                if x < t.width && y < t.height {
                                    texels[(y * t.width + x) as usize] = pack(dxt_texel(t.format, &block, tx, ty));
                                }
                            }
                        }
                    }
                }
            }
            _ => return None,
        }
        Some(Self { width: t.width, height: t.height, texels })
    }

    #[inline]
    fn texel(&self, x: u32, y: u32) -> [f32; 4] {
        let p = self.texels[(y * self.width + x) as usize];
        const K: f32 = 1.0 / 255.0;
        [(p & 0xFF) as f32 * K, ((p >> 8) & 0xFF) as f32 * K, ((p >> 16) & 0xFF) as f32 * K, (p >> 24) as f32 * K]
    }

    /// Same filtering, wrapping and swizzle as `TextureSampler::sample`, from the decoded copy.
    pub fn sample(&self, t: &TexDesc, u: f32, v: f32, linear: bool) -> [f32; 4] {
        let (fx, fy) = (u * t.width as f32, v * t.height as f32);
        let c = if linear {
            let (x0, y0) = ((fx - 0.5).floor(), (fy - 0.5).floor());
            let (tx, ty) = (fx - 0.5 - x0, fy - 0.5 - y0);
            let (xa, xb) = (TextureSampler::wrap(x0 as i64, t.width, t.clamp_x), TextureSampler::wrap(x0 as i64 + 1, t.width, t.clamp_x));
            let (ya, yb) = (TextureSampler::wrap(y0 as i64, t.height, t.clamp_y), TextureSampler::wrap(y0 as i64 + 1, t.height, t.clamp_y));
            let (p00, p10, p01, p11) = (self.texel(xa, ya), self.texel(xb, ya), self.texel(xa, yb), self.texel(xb, yb));
            let mut out = [0.0; 4];
            for i in 0..4 {
                let top = p00[i] * (1.0 - tx) + p10[i] * tx;
                let bot = p01[i] * (1.0 - tx) + p11[i] * tx;
                out[i] = top * (1.0 - ty) + bot * ty;
            }
            out
        } else {
            self.texel(
                TextureSampler::wrap(fx.floor() as i64, t.width, t.clamp_x),
                TextureSampler::wrap(fy.floor() as i64, t.height, t.clamp_y),
            )
        };
        let pick = |s: u32| match s {
            0..=3 => c[s as usize],
            4 => 0.0,
            _ => 1.0,
        };
        [pick(t.swizzle[0]), pick(t.swizzle[1]), pick(t.swizzle[2]), pick(t.swizzle[3])]
    }
}

/// Upper bound of the bytes a texture occupies, so it can be read out of guest memory in one go.
fn texture_bytes(t: &TexDesc) -> Option<usize> {
    let (bw, bytes, _) = t.layout()?;
    let pitch_elems = t.pitch.max(t.width).div_ceil(bw);
    let h_elems = t.height.div_ceil(bw);
    let (w32, h32) = ((pitch_elems + 31) & !31, (h_elems + 31) & !31);
    let n = (w32 as usize) * (h32 as usize) * bytes as usize;
    (n > 0 && n <= 64 << 20).then_some(n)
}

fn hash_words(words: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ words.len() as u64;
    for ch in words.chunks(2) {
        let v = ch[0] as u64 | (ch.get(1).copied().unwrap_or(0) as u64) << 32;
        h = (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    h
}

/// Decoded textures kept across draws.
#[derive(Default)]
pub struct TexCache {
    map: HashMap<TexDesc, (u64, Arc<Option<DecodedTex>>)>,
    pub hits: u64,
    pub decodes: u64,
}

impl TexCache {
    pub fn get(&mut self, mem: &GuestMemory, t: &TexDesc) -> Arc<Option<DecodedTex>> {
        let Some(n) = texture_bytes(t) else { return Arc::new(None) };
        let mut raw = vec![0u8; n];
        mem.read_bytes(PHYS_VIEW | t.base_phys, &mut raw);
        // raw guest bytes are big-endian dwords; apply_endian expects those as read32() returns them
        let words: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])).collect();
        let h = hash_words(&words);
        if let Some((old, tex)) = self.map.get(t) {
            if *old == h {
                self.hits += 1;
                return tex.clone();
            }
        }
        self.decodes += 1;
        let tex = Arc::new(DecodedTex::decode(t, &words));
        if tex.is_none() && std::env::var_os("X360_LOG_TEX_FORMATS").is_some() {
            use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
            static SEEN: AtomicU32 = AtomicU32::new(0);
            if SEEN.fetch_add(1, Relaxed) < 40 {
                println!("[tex] unsupported texture format {} ({}x{} pitch {} tiled {} endian {}) at {:#x}", t.format, t.width, t.height, t.pitch, t.tiled, t.endian, t.base_phys);
            }
        }
        if self.map.len() > 512 {
            self.map.clear();
        }
        self.map.insert(*t, (h, tex.clone()));
        tex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xenos_tex::REG_FETCH_CONSTS;
    use xenon_cpu::memory::Bus;

    /// The cached sampler must agree with the reference one (up to 8-bit quantisation) on a tiled DXT1 texture.
    #[test]
    fn cache_matches_reference_sampler() {
        let mut mem = GuestMemory::new();
        let base = 0x0010_0000u32;
        mem.map(PHYS_VIEW | base, 0x20000);
        // fill with pseudo-random DXT blocks
        let mut x = 0x1234_5678u32;
        for i in 0..0x4000u32 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            mem.write32(PHYS_VIEW | base + i * 4, x);
        }
        let mut regs = vec![0u32; 0x5000];
        let c = REG_FETCH_CONSTS;
        // 64x64 DXT1 (format 18), tiled, pitch 64, endian none, point sampling
        regs[c] = (1 << 31) | ((64 / 32) << 22);
        regs[c + 1] = base | 18;
        regs[c + 2] = 63 | (63 << 13);
        regs[c + 3] = (0 | (1 << 3) | (2 << 6) | (3 << 9)) << 1;
        let t = TexDesc::from_regs(&regs, 0);
        let mut cache = TexCache::default();
        let tex = cache.get(&mem, &t);
        let tex = tex.as_ref().as_ref().expect("DXT1 decodes");
        let reference = TextureSampler { mem: &mem, regs: &regs };
        for k in 0..200 {
            let (u, v) = ((k as f32 * 0.0137) % 1.0, (k as f32 * 0.0291) % 1.0);
            for linear in [false, true] {
                let a = reference.sample(0, u, v, linear);
                let b = tex.sample(&t, u, v, linear);
                for i in 0..4 {
                    assert!((a[i] - b[i]).abs() < 0.01, "k={k} linear={linear} ch={i}: {} vs {}", a[i], b[i]);
                }
            }
        }
        // second lookup is a hit; changing the guest bytes invalidates it
        let _ = cache.get(&mem, &t);
        assert_eq!(cache.hits, 1);
        mem.write32(PHYS_VIEW | base, 0xDEAD_BEEF);
        let _ = cache.get(&mem, &t);
        assert_eq!(cache.decodes, 2);
    }
}
