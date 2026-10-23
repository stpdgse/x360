//! VMX128 opcode 6 (and vperm128 from opcode 5). Register indices are 7 bits wide and scattered:
//!   VD = inst[25:21] | inst[3:2]<<5      VB = inst[15:11] | inst[1:0]<<5
//!   VA = inst[20:16] | inst[5] | inst[10]<<6   (3-operand forms only; 2-operand forms use
//!        inst[20:16] as a 5-bit immediate instead).
//! The operation is identified by `inst & 0x3D0` for 3-operand forms and `inst & 0x7F0` for
//! 2-operand forms. Encodings are checked against real instructions as the game executes them.

use crate::vmx::{f4, from_f4, map_f2, vx_float, vx_lanes, zip32};
use crate::Cpu;

fn splat(w: u32) -> u128 {
    zip32(0, 0, |_, _| w)
}

fn words(v: u128) -> [u32; 4] {
    let b = v.to_be_bytes();
    std::array::from_fn(|i| u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

fn from_words(w: [u32; 4]) -> u128 {
    let mut o = [0u8; 16];
    for i in 0..4 {
        o[4 * i..4 * i + 4].copy_from_slice(&w[i].to_be_bytes());
    }
    u128::from_be_bytes(o)
}

fn vperm(a: u128, b: u128, c: u128) -> u128 {
    let (ab, bb, cb) = (a.to_be_bytes(), b.to_be_bytes(), c.to_be_bytes());
    let mut o = [0u8; 16];
    for i in 0..16 {
        let idx = (cb[i] & 31) as usize;
        o[i] = if idx < 16 { ab[idx] } else { bb[idx - 16] };
    }
    u128::from_be_bytes(o)
}

/// Direct3D data unpack (vupkd3d128). Only the D3DCOLOR format is modelled: the four
/// bytes of the source word become floats of the form 3.0 + byte * 2^-22, in R,G,B,A order.
fn unpack_d3dcolor(v: u32) -> u128 {
    let ch = |b: u32| 0x4040_0000 | (b & 0xFF);
    from_words([ch(v >> 16), ch(v >> 8), ch(v), ch(v >> 24)])
}

/// NORMSHORT2: two signed 16-bit values become 3.0 + s * 2^-22 (x = high half, y = low half),
/// with z = 0.0 and w = 1.0. (Best-effort; the exact fill values for z/w are from memory.)
fn unpack_normshort2(v: u32) -> u128 {
    let ch = |s: i16| 0x4040_0000u32.wrapping_add(s as i32 as u32);
    from_words([ch((v >> 16) as i16), ch(v as i16), 0, 0x3F80_0000])
}

impl Cpu {
    #[inline(never)]
    pub(crate) fn exec_vmx128_vperm(&mut self, inst: u32) -> bool {
        // vperm128: opcode 5 with inst[9] and inst[4] clear; the permute vector is v0..v7 from inst[8:6].
        if inst & 0x210 != 0 {
            return false;
        }
        let vd = (((inst >> 21) & 31) | ((inst & 0xC) << 3)) as usize;
        let va = (((inst >> 16) & 31) | (inst & 0x20) | ((inst & 0x400) >> 4)) as usize;
        let vb = (((inst >> 11) & 31) | ((inst & 3) << 5)) as usize;
        let vc = ((inst >> 6) & 7) as usize;
        self.vr[vd] = vperm(self.vr[va], self.vr[vb], self.vr[vc]);
        true
    }

    #[inline(never)]
    pub(crate) fn exec_vmx128_op6(&mut self, inst: u32) -> bool {
        let vd = (((inst >> 21) & 31) | ((inst & 0xC) << 3)) as usize;
        let va = (((inst >> 16) & 31) | (inst & 0x20) | ((inst & 0x400) >> 4)) as usize;
        let vb = (((inst >> 11) & 31) | ((inst & 3) << 5)) as usize;
        let imm5 = (inst >> 16) & 31;
        let (a, b, d) = (self.vr[va], self.vr[vb], self.vr[vd]);

        // vpermwi128: word permute by an 8-bit immediate split across two fields
        if inst & 0x630 == 0x210 {
            let perm = ((inst >> 16) & 0x1F) | (((inst >> 6) & 7) << 5);
            let w = words(b);
            self.vr[vd] = from_words(std::array::from_fn(|i| w[((perm >> (6 - 2 * i)) & 3) as usize]));
            return true;
        }

        // vrlimi128 keeps its rotate count in inst[7:6], so those bits are excluded from the key:
        // rotate vB left by `rot` words, then insert the lanes selected by the 4-bit mask into vD.
        if inst & 0x730 == 0x710 {
            let rot = ((inst >> 6) & 3) as usize;
            let mask = imm5 & 0xF;
            let w = words(b);
            let mut out = words(d);
            for i in 0..4 {
                if mask & (8 >> i) != 0 {
                    out[i] = w[(i + rot) & 3];
                }
            }
            self.vr[vd] = from_words(out);
            return true;
        }

        // 2-operand forms (immediate in the VA field)
        let two = match inst & 0x7F0 {
            0x730 => Some(splat(words(b)[(imm5 & 3) as usize])), // vspltw128
            0x770 => Some(splat((((imm5 as i32) << 27) >> 27) as u32)), // vspltisw128
            0x330 => vx_float(714, 0, b, 0),                     // vrfim128
            0x370 => vx_float(522, 0, b, 0),                     // vrfin128
            0x3B0 => vx_float(650, 0, b, 0),                     // vrfip128
            0x3F0 => vx_float(586, 0, b, 0),                     // vrfiz128
            0x630 => vx_float(266, 0, b, 0),                     // vrefp128
            0x670 => vx_float(330, 0, b, 0),                     // vrsqrtefp128
            0x6B0 => Some(from_f4(f4(b).map(f32::exp2))),        // vexptefp128
            0x6F0 => Some(from_f4(f4(b).map(f32::log2))),        // vlogefp128
            0x230 => vx_float(970, 0, b, imm5 as usize),         // vcfpsxws128 (float -> signed int)
            0x2B0 => vx_float(842, 0, b, imm5 as usize),         // vcsxwfp128 (signed int -> float)
            0x2F0 => {
                // vcuxwfp128: unsigned int -> float, scaled by 2^-imm
                let scale = 1.0 / (1u64 << (imm5 & 31)) as f32;
                Some(from_f4(words(b).map(|w| w as f32 * scale)))
            }
            0x7F0 => {
                // vupkd3d128: format selected by imm[4:2]. Modelled: D3DCOLOR (0) and NORMSHORT2 (1).
                match imm5 >> 2 {
                    0 => Some(unpack_d3dcolor(words(b)[3])),
                    1 => Some(unpack_normshort2(words(b)[3])),
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some(r) = two {
            self.vr[vd] = r;
            return true;
        }

        // 3-operand forms
        let rc = inst & 0x40 != 0;
        let key = inst & 0x3D0;
        let cmp = |me: &mut Self, r: u128| {
            me.vr[vd] = r;
            if rc {
                let cr6 = ((r == !0) as u32) << 3 | ((r == 0) as u32) << 1;
                me.cr = (me.cr & !(0xF << 4)) | (cr6 << 4);
            }
        };
        let m = |t: bool| if t { !0u32 } else { 0 };
        match inst & 0x3B0 {
            0x000 => cmp(self, zip32(a, b, |x, y| m(f32::from_bits(x) == f32::from_bits(y)))), // vcmpeqfp128
            0x080 => cmp(self, zip32(a, b, |x, y| m(f32::from_bits(x) >= f32::from_bits(y)))), // vcmpgefp128
            0x100 => cmp(self, zip32(a, b, |x, y| m(f32::from_bits(x) > f32::from_bits(y)))),  // vcmpgtfp128
            0x200 => cmp(self, zip32(a, b, |x, y| m(x == y))),                                 // vcmpequw128
            _ => {
                self.vr[vd] = match key {
                    0x280 => map_f2(a, b, f32::max),                       // vmaxfp128
                    0x2C0 => map_f2(a, b, f32::min),                       // vminfp128
                    0x300 => from_words([words(a)[0], words(b)[0], words(a)[1], words(b)[1]]), // vmrghw128
                    0x340 => from_words([words(a)[2], words(b)[2], words(a)[3], words(b)[3]]), // vmrglw128
                    0x050 => vx_lanes(132, a, b, 0).unwrap(),              // vrlw128
                    0x0D0 => vx_lanes(388, a, b, 0).unwrap(),              // vslw128
                    0x150 => vx_lanes(900, a, b, 0).unwrap(),              // vsraw128
                    0x1D0 => vx_lanes(644, a, b, 0).unwrap(),              // vsrw128
                    0x380 => vx_lanes(526, 0, b, 0).unwrap(),              // vupkhsb128
                    0x3C0 => vx_lanes(654, 0, b, 0).unwrap(),              // vupklsb128
                    _ => return false,
                };
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Bus, FlatMemory};

    #[test]
    fn vupkd3d128_unpacks_color_bytes() {
        let mut m = FlatMemory::new(16);
        let mut c = Cpu::new();
        // The real crash instruction: 0x180007F0 = vupkd3d128 v0, v0, 0
        m.write32(0, 0x1800_07F0);
        c.vr[0] = 0x0000_0000_0000_0000_0000_0000_80FF_4010; // word 3 = A=0x80 R=0xFF G=0x40 B=0x10
        c.step(&mut m).unwrap();
        let w = words(c.vr[0]);
        assert_eq!(w[0], 0x4040_00FF); // R
        assert_eq!(w[1], 0x4040_0040); // G
        assert_eq!(w[2], 0x4040_0010); // B
        assert_eq!(w[3], 0x4040_0080); // A
        assert_eq!(f32::from_bits(w[0]), 3.0 + 255.0 / 4194304.0);
    }

    #[test]
    fn vupkd3d128_normshort2() {
        let mut m = FlatMemory::new(16);
        let mut c = Cpu::new();
        // The real crash instruction: 0x18E407F0 = vupkd3d128 with imm 4 (format 1)
        m.write32(0, 0x18E4_07F0);
        // vD = ((0x18E407F0 >> 21) & 31) = 7; vB = 0
        c.vr[0] = 0x0000_0000_0000_0000_0000_0000_0003_FFFE; // hi = +3, lo = -2
        c.step(&mut m).unwrap();
        let w = words(c.vr[7]);
        assert_eq!(w[0], 0x4040_0003);
        assert_eq!(w[1], 0x403F_FFFE); // 3.0 - 2 * 2^-22
        assert_eq!(w[3], 0x3F80_0000);
    }

    #[test]
    fn vrlimi_vspltw_vmrghw_vpermwi() {
        let mut m = FlatMemory::new(32);
        let mut c = Cpu::new();
        c.vr[2] = from_words([1, 2, 3, 4]);
        c.vr[3] = from_words([10, 20, 30, 40]);
        // vmrghw128 v4,v2,v3 : key 0x300, vd=4 va=2 vb=3
        m.write32(0, (6 << 26) | (4 << 21) | (2 << 16) | (3 << 11) | 0x300);
        // vspltw128 v5,v3,imm 2 : key 0x730, imm in the VA field
        m.write32(4, (6 << 26) | (5 << 21) | (2 << 16) | (3 << 11) | 0x730);
        // vrlimi128 v2,v3,mask 0b0101,rot 1 : key 0x710
        m.write32(8, (6 << 26) | (2 << 21) | (5 << 16) | (3 << 11) | (1 << 6) | 0x710);
        // vpermwi128 v6,v3 with perm 0b00_01_10_11 (reverse): low 5 bits in VA field, high 3 in inst[8:6]
        let perm = 0b11_10_01_00u32; // lane i takes word (perm >> (6 - 2i)) & 3 -> reverse order
        m.write32(12, (6 << 26) | (6 << 21) | ((perm & 0x1F) << 16) | (3 << 11) | ((perm >> 5) << 6) | 0x210);
        for _ in 0..4 {
            c.step(&mut m).unwrap();
        }
        assert_eq!(words(c.vr[4]), [1, 10, 2, 20]);
        assert_eq!(words(c.vr[5]), [30, 30, 30, 30]);
        // rotated vB = [20,30,40,10]; mask 0101 -> lanes 1 and 3 replaced in vD ([1,2,3,4])
        assert_eq!(words(c.vr[2]), [1, 30, 3, 10]);
        assert_eq!(words(c.vr[6]), [40, 30, 20, 10]);
    }
}
