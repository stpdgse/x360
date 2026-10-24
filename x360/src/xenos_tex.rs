//! Texture sampling for the software Xenos: fetch-constant decoding, tiled addressing, texel
//! formats (including DXT block compression) and point/bilinear filtering of mip level 0.

use crate::gpu::{tiled_offset_2d, PHYS_VIEW};
use crate::memory::GuestMemory;
use crate::xenos_draw::apply_endian;
use xenon_cpu::memory::Bus;

pub const REG_FETCH_CONSTS: usize = 0x4800;

/// A decoded texture fetch constant (6 dwords per slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TexDesc {
    pub base_phys: u32,
    pub width: u32,
    pub height: u32,
    /// Pitch in texels.
    pub pitch: u32,
    pub format: u32,
    pub endian: u32,
    pub tiled: bool,
    /// Per-channel source selectors: 0..3 = R,G,B,A, 4 = 0, 5 = 1.
    pub swizzle: [u32; 4],
    pub clamp_x: u32,
    pub clamp_y: u32,
    pub mag_filter: u32,
}

impl TexDesc {
    pub fn from_regs(regs: &[u32], slot: u32) -> Self {
        let c = REG_FETCH_CONSTS + slot as usize * 6;
        let (d0, d1, d2, d3) = (regs[c], regs[c + 1], regs[c + 2], regs[c + 3]);
        let sw = (d3 >> 1) & 0xFFF;
        Self {
            base_phys: d1 & 0xFFFF_F000,
            width: (d2 & 0x1FFF) + 1,
            height: ((d2 >> 13) & 0x1FFF) + 1,
            pitch: ((d0 >> 22) & 0x1FF) * 32,
            format: d1 & 0x3F,
            endian: (d1 >> 6) & 3,
            tiled: d0 >> 31 != 0,
            swizzle: [sw & 7, (sw >> 3) & 7, (sw >> 6) & 7, (sw >> 9) & 7],
            clamp_x: (d0 >> 10) & 7,
            clamp_y: (d0 >> 13) & 7,
            mag_filter: (d3 >> 19) & 3,
        }
    }

    /// (block width/height in texels, bytes per element, log2 bytes per element)
    pub(crate) fn layout(&self) -> Option<(u32, u32, u32)> {
        Some(match self.format {
            2 | 8 | 9 => (1, 1, 0),                // 8-bit single channel
            3 | 4 | 5 | 10 | 15 | 24 => (1, 2, 1), // 16-bit packed / two 8-bit
            6 | 7 | 14 | 16 | 17 => (1, 4, 2),     // 32-bit
            18 | 50 | 51 | 52 => (4, 8, 3),        // DXT1 / DXT3A / DXT5A / CTX1
            19 | 20 => (4, 16, 4),                 // DXT3 / DXT5
            _ => return None,
        })
    }
}

pub(crate) fn expand565(c: u16) -> [f32; 3] {
    [((c >> 11) & 31) as f32 / 31.0, ((c >> 5) & 63) as f32 / 63.0, (c & 31) as f32 / 31.0]
}

/// Decode a DXT block (given as canonical little-endian bytes) and return texel (x, y) in [0,4).
pub(crate) fn dxt_texel(kind: u32, block: &[u8; 16], x: u32, y: u32) -> [f32; 4] {
    // colour block: DXT1 occupies the first 8 bytes, DXT3/5 the last 8
    let cb = if kind == 18 { &block[0..8] } else { &block[8..16] };
    let c0 = u16::from_le_bytes([cb[0], cb[1]]);
    let c1 = u16::from_le_bytes([cb[2], cb[3]]);
    let (a, b) = (expand565(c0), expand565(c1));
    let idx = (u32::from_le_bytes([cb[4], cb[5], cb[6], cb[7]]) >> (2 * (y * 4 + x))) & 3;
    let mut alpha = 1.0;
    let rgb = match idx {
        0 => a,
        1 => b,
        2 => {
            if c0 > c1 || kind != 18 {
                [(2.0 * a[0] + b[0]) / 3.0, (2.0 * a[1] + b[1]) / 3.0, (2.0 * a[2] + b[2]) / 3.0]
            } else {
                [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0]
            }
        }
        _ => {
            if c0 > c1 || kind != 18 {
                [(a[0] + 2.0 * b[0]) / 3.0, (a[1] + 2.0 * b[1]) / 3.0, (a[2] + 2.0 * b[2]) / 3.0]
            } else {
                alpha = 0.0; // DXT1 punch-through
                [0.0; 3]
            }
        }
    };
    match kind {
        19 => {
            // explicit 4-bit alpha, 16 bits per row
            let row = u16::from_le_bytes([block[(y * 2) as usize], block[(y * 2 + 1) as usize]]);
            alpha = ((row >> (4 * x)) & 0xF) as f32 / 15.0;
        }
        20 => {
            let (a0, a1) = (block[0] as f32, block[1] as f32);
            let bits = u64::from_le_bytes([block[2], block[3], block[4], block[5], block[6], block[7], 0, 0]);
            let code = (bits >> (3 * (y * 4 + x))) & 7;
            alpha = match code {
                0 => a0,
                1 => a1,
                c if a0 > a1 => ((8 - c) as f32 * a0 + (c - 1) as f32 * a1) / 7.0,
                6 => 0.0,
                7 => 255.0,
                c => ((6 - c) as f32 * a0 + (c - 1) as f32 * a1) / 5.0,
            } / 255.0;
        }
        _ => {}
    }
    [rgb[0], rgb[1], rgb[2], alpha]
}

pub struct TextureSampler<'a> {
    pub mem: &'a GuestMemory,
    pub regs: &'a [u32],
}

impl TextureSampler<'_> {
    /// Fetch one texel as RGBA in 0..1 (before the fetch constant's swizzle).
    fn texel(&self, t: &TexDesc, x: u32, y: u32) -> [f32; 4] {
        let Some((bw, bytes, log)) = t.layout() else {
            return [1.0, 0.0, 1.0, 1.0]; // unsupported format: loud magenta
        };
        let (ex, ey) = (x / bw, y / bw);
        let pitch_elems = t.pitch.max(t.width).div_ceil(bw);
        let off = if t.tiled { tiled_offset_2d(ex, ey, pitch_elems, log) } else { (ey * pitch_elems + ex) * bytes };
        let addr = PHYS_VIEW | t.base_phys;
        let byte_at = |o: u32| -> u8 {
            let a = addr + o;
            (apply_endian(self.mem.read32(a & !3), t.endian) >> (8 * (a & 3))) as u8
        };
        let dword_at = |o: u32| apply_endian(self.mem.read32(addr + (o & !3)), t.endian);
        match t.format {
            2 | 8 | 9 => {
                let v = byte_at(off) as f32 / 255.0;
                if t.format == 8 { [0.0, 0.0, 0.0, v] } else { [v, 0.0, 0.0, 1.0] }
            }
            10 => {
                let (a, b) = (byte_at(off) as f32 / 255.0, byte_at(off + 1) as f32 / 255.0);
                [a, b, 0.0, 1.0]
            }
            4 => {
                let h = ((byte_at(off + 1) as u16) << 8) | byte_at(off) as u16;
                let c = expand565(h);
                [c[0], c[1], c[2], 1.0]
            }
            3 => {
                let h = ((byte_at(off + 1) as u16) << 8) | byte_at(off) as u16;
                [((h >> 10) & 31) as f32 / 31.0, ((h >> 5) & 31) as f32 / 31.0, (h & 31) as f32 / 31.0, (h >> 15) as f32]
            }
            15 => {
                let h = ((byte_at(off + 1) as u16) << 8) | byte_at(off) as u16;
                [((h >> 8) & 15) as f32 / 15.0, ((h >> 4) & 15) as f32 / 15.0, (h & 15) as f32 / 15.0, ((h >> 12) & 15) as f32 / 15.0]
            }
            6 | 14 => {
                let d = dword_at(off);
                [(d & 0xFF) as f32 / 255.0, ((d >> 8) & 0xFF) as f32 / 255.0, ((d >> 16) & 0xFF) as f32 / 255.0, (d >> 24) as f32 / 255.0]
            }
            18 | 19 | 20 => {
                let mut block = [0u8; 16];
                for i in 0..(bytes / 4) {
                    block[(i * 4) as usize..(i * 4 + 4) as usize].copy_from_slice(&dword_at(off + i * 4).to_le_bytes());
                }
                dxt_texel(t.format, &block, x % 4, y % 4)
            }
            _ => [1.0, 0.0, 1.0, 1.0],
        }
    }

    pub(crate) fn wrap(coord: i64, size: u32, mode: u32) -> u32 {
        let s = size as i64;
        match mode {
            0 => coord.rem_euclid(s) as u32, // repeat
            1 => {
                // mirrored repeat
                let m = coord.rem_euclid(2 * s);
                (if m < s { m } else { 2 * s - 1 - m }) as u32
            }
            _ => coord.clamp(0, s - 1) as u32, // clamp to edge (and everything else)
        }
    }

    /// Sample slot `slot` at normalized (u, v). `linear` selects bilinear filtering.
    pub fn sample(&self, slot: u32, u: f32, v: f32, linear: bool) -> [f32; 4] {
        let t = TexDesc::from_regs(self.regs, slot);
        if t.width == 0 || t.height == 0 {
            return [0.0; 4];
        }
        let (fx, fy) = (u * t.width as f32, v * t.height as f32);
        let c = if linear {
            let (x0, y0) = ((fx - 0.5).floor(), (fy - 0.5).floor());
            let (tx, ty) = (fx - 0.5 - x0, fy - 0.5 - y0);
            let at = |dx: i64, dy: i64| {
                self.texel(&t, Self::wrap(x0 as i64 + dx, t.width, t.clamp_x), Self::wrap(y0 as i64 + dy, t.height, t.clamp_y))
            };
            let (p00, p10, p01, p11) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
            let mut out = [0.0; 4];
            for i in 0..4 {
                let top = p00[i] * (1.0 - tx) + p10[i] * tx;
                let bot = p01[i] * (1.0 - tx) + p11[i] * tx;
                out[i] = top * (1.0 - ty) + bot * ty;
            }
            out
        } else {
            self.texel(&t, Self::wrap(fx.floor() as i64, t.width, t.clamp_x), Self::wrap(fy.floor() as i64, t.height, t.clamp_y))
        };
        // apply the fetch constant's channel swizzle
        let pick = |s: u32| match s {
            0..=3 => c[s as usize],
            4 => 0.0,
            _ => 1.0,
        };
        [pick(t.swizzle[0]), pick(t.swizzle[1]), pick(t.swizzle[2]), pick(t.swizzle[3])]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxt1_decodes_endpoints_and_blends() {
        // c0 = pure red (565: F800), c1 = pure blue (001F); indices: texel0 -> 0, texel1 -> 1, texel2 -> 2
        let mut block = [0u8; 16];
        block[0..2].copy_from_slice(&0xF800u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x001Fu16.to_le_bytes());
        let idx: u32 = 0b10_01_00;
        block[4..8].copy_from_slice(&idx.to_le_bytes());
        assert_eq!(dxt_texel(18, &block, 0, 0), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(dxt_texel(18, &block, 1, 0), [0.0, 0.0, 1.0, 1.0]);
        let mid = dxt_texel(18, &block, 2, 0);
        assert!((mid[0] - 2.0 / 3.0).abs() < 1e-5 && (mid[2] - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn dxt5_alpha_interpolation() {
        let mut block = [0u8; 16];
        block[0] = 255; // a0
        block[1] = 0; // a1  -> 8-alpha mode (a0 > a1)
        // texel 0 uses code 0 (a0), texel 1 uses code 1 (a1), texel 2 uses code 2: (6*255 + 1*0)/7
        let bits: u64 = 0 | (1 << 3) | (2 << 6);
        block[2..8].copy_from_slice(&bits.to_le_bytes()[0..6]);
        assert_eq!(dxt_texel(20, &block, 0, 0)[3], 1.0);
        assert_eq!(dxt_texel(20, &block, 1, 0)[3], 0.0);
        assert!((dxt_texel(20, &block, 2, 0)[3] - 6.0 / 7.0).abs() < 1e-5);
    }

    #[test]
    fn wrap_modes() {
        assert_eq!(TextureSampler::wrap(-1, 8, 0), 7);
        assert_eq!(TextureSampler::wrap(9, 8, 0), 1);
        assert_eq!(TextureSampler::wrap(-1, 8, 2), 0);
        assert_eq!(TextureSampler::wrap(20, 8, 2), 7);
        assert_eq!(TextureSampler::wrap(8, 8, 1), 7); // mirror
    }

    #[test]
    fn fetch_constant_decoding() {
        let mut regs = vec![0u32; 0x5000];
        let c = REG_FETCH_CONSTS + 6; // slot 1
        regs[c] = (40 << 22) | (1 << 31) | 2; // pitch 1280, tiled, type 2
        regs[c + 1] = 0x1234_5000 | (2 << 6) | 18; // base, endian 2, DXT1
        regs[c + 2] = (255 << 13) | 511; // 512 x 256
        regs[c + 3] = (0 << 1) | (1 << 4) | (2 << 7) | (3 << 10); // swizzle x,y,z,w = 0,1,2,3
        let t = TexDesc::from_regs(&regs, 1);
        assert_eq!((t.width, t.height, t.pitch, t.format, t.endian, t.tiled), (512, 256, 1280, 18, 2, true));
        assert_eq!(t.base_phys, 0x1234_5000);
        assert_eq!(t.swizzle, [0, 1, 2, 3]);
    }
}
