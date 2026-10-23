//! VMX (AltiVec) vector unit, standard encodings (v0..v31). VMX128 forms come later.
//! A vector register is a `u128` in big-endian lane order: lane 0 is the most significant.

use crate::memory::Bus;
use crate::Cpu;

fn zip8(a: u128, b: u128, f: impl Fn(u8, u8) -> u8) -> u128 {
    let (x, y) = (a.to_be_bytes(), b.to_be_bytes());
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = f(x[i], y[i]);
    }
    u128::from_be_bytes(o)
}

fn zip16(a: u128, b: u128, f: impl Fn(u16, u16) -> u16) -> u128 {
    let (x, y) = (a.to_be_bytes(), b.to_be_bytes());
    let mut o = [0u8; 16];
    for i in 0..8 {
        let r = f(u16::from_be_bytes([x[2 * i], x[2 * i + 1]]), u16::from_be_bytes([y[2 * i], y[2 * i + 1]]));
        o[2 * i..2 * i + 2].copy_from_slice(&r.to_be_bytes());
    }
    u128::from_be_bytes(o)
}

pub(crate) fn zip32(a: u128, b: u128, f: impl Fn(u32, u32) -> u32) -> u128 {
    let (x, y) = (a.to_be_bytes(), b.to_be_bytes());
    let mut o = [0u8; 16];
    for i in 0..4 {
        let r = f(
            u32::from_be_bytes(x[4 * i..4 * i + 4].try_into().unwrap()),
            u32::from_be_bytes(y[4 * i..4 * i + 4].try_into().unwrap()),
        );
        o[4 * i..4 * i + 4].copy_from_slice(&r.to_be_bytes());
    }
    u128::from_be_bytes(o)
}

/// Lane-wise VX ops: shifts/rotates, min/max, merge, pack, unpack, whole-vector shifts.
pub(crate) fn vx_lanes(xo: u32, a: u128, b: u128, va: usize) -> Option<u128> {
    let (ab, bb) = (a.to_be_bytes(), b.to_be_bytes());
    Some(match xo {
        // rotate left
        4 => zip8(a, b, |x, y| x.rotate_left((y & 7) as u32)),
        68 => zip16(a, b, |x, y| x.rotate_left((y & 15) as u32)),
        132 => zip32(a, b, |x, y| x.rotate_left(y & 31)),
        // shift left / right logical / right arithmetic
        260 => zip8(a, b, |x, y| x << (y & 7)),
        324 => zip16(a, b, |x, y| x << (y & 15)),
        388 => zip32(a, b, |x, y| x << (y & 31)),
        516 => zip8(a, b, |x, y| x >> (y & 7)),
        580 => zip16(a, b, |x, y| x >> (y & 15)),
        644 => zip32(a, b, |x, y| x >> (y & 31)),
        772 => zip8(a, b, |x, y| ((x as i8) >> (y & 7)) as u8),
        836 => zip16(a, b, |x, y| ((x as i16) >> (y & 15)) as u16),
        900 => zip32(a, b, |x, y| ((x as i32) >> (y & 31)) as u32),
        // whole-vector shifts (bit: vsl/vsr, octet: vslo/vsro)
        452 => a << (bb[15] & 7),
        708 => a >> (bb[15] & 7),
        1036 => a << (((bb[15] >> 3) & 15) as u32 * 8),
        1100 => a >> (((bb[15] >> 3) & 15) as u32 * 8),
        // max / min
        2 => zip8(a, b, |x, y| x.max(y)),
        66 => zip16(a, b, |x, y| x.max(y)),
        130 => zip32(a, b, |x, y| x.max(y)),
        258 => zip8(a, b, |x, y| (x as i8).max(y as i8) as u8),
        322 => zip16(a, b, |x, y| (x as i16).max(y as i16) as u16),
        386 => zip32(a, b, |x, y| (x as i32).max(y as i32) as u32),
        514 => zip8(a, b, |x, y| x.min(y)),
        578 => zip16(a, b, |x, y| x.min(y)),
        642 => zip32(a, b, |x, y| x.min(y)),
        770 => zip8(a, b, |x, y| (x as i8).min(y as i8) as u8),
        834 => zip16(a, b, |x, y| (x as i16).min(y as i16) as u16),
        898 => zip32(a, b, |x, y| (x as i32).min(y as i32) as u32),
        // merge high/low (sizes 1, 2, 4)
        12 | 76 | 140 | 268 | 332 | 396 => {
            let (size, low) = match xo {
                12 => (1, false),
                76 => (2, false),
                140 => (4, false),
                268 => (1, true),
                332 => (2, true),
                _ => (4, true),
            };
            let base = if low { 8 } else { 0 };
            let mut o = [0u8; 16];
            for i in 0..8 / size {
                for k in 0..size {
                    o[i * 2 * size + k] = ab[base + i * size + k];
                    o[i * 2 * size + size + k] = bb[base + i * size + k];
                }
            }
            u128::from_be_bytes(o)
        }
        // pack unsigned modulo (keep low half of each element): vpkuhum, vpkuwum
        14 => {
            let mut o = [0u8; 16];
            for i in 0..8 {
                o[i] = ab[2 * i + 1];
                o[8 + i] = bb[2 * i + 1];
            }
            u128::from_be_bytes(o)
        }
        78 => {
            let mut o = [0u8; 16];
            for i in 0..4 {
                o[2 * i..2 * i + 2].copy_from_slice(&ab[4 * i + 2..4 * i + 4]);
                o[8 + 2 * i..8 + 2 * i + 2].copy_from_slice(&bb[4 * i + 2..4 * i + 4]);
            }
            u128::from_be_bytes(o)
        }
        // unpack signed (source is vB): vupkhsb/vupkhsh/vupklsb/vupklsh
        526 | 654 => {
            let base = if xo == 526 { 0 } else { 8 };
            let mut o = [0u8; 16];
            for i in 0..8 {
                o[2 * i..2 * i + 2].copy_from_slice(&(bb[base + i] as i8 as i16).to_be_bytes());
            }
            u128::from_be_bytes(o)
        }
        590 | 718 => {
            let base = if xo == 590 { 0 } else { 4 };
            let mut o = [0u8; 16];
            for i in 0..4 {
                let h = i16::from_be_bytes([bb[2 * (base + i)], bb[2 * (base + i) + 1]]);
                o[4 * i..4 * i + 4].copy_from_slice(&(h as i32).to_be_bytes());
            }
            u128::from_be_bytes(o)
        }
        _ => {
            let _ = va;
            return None;
        }
    })
}

pub(crate) fn f4(v: u128) -> [f32; 4] {
    let b = v.to_be_bytes();
    std::array::from_fn(|i| f32::from_bits(u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap())))
}

pub(crate) fn from_f4(l: [f32; 4]) -> u128 {
    let mut o = [0u8; 16];
    for i in 0..4 {
        o[4 * i..4 * i + 4].copy_from_slice(&l[i].to_bits().to_be_bytes());
    }
    u128::from_be_bytes(o)
}

pub(crate) fn map_f2(a: u128, b: u128, f: impl Fn(f32, f32) -> f32) -> u128 {
    let (x, y) = (f4(a), f4(b));
    from_f4(std::array::from_fn(|i| f(x[i], y[i])))
}

pub(crate) fn map_f1(a: u128, f: impl Fn(f32) -> f32) -> u128 {
    from_f4(f4(a).map(f))
}

fn fma(a: u128, b: u128, c: u128, negate: bool) -> u128 {
    // a*b + c, or -(a*b - c) when `negate` (vnmsubfp)
    let (x, y, z) = (f4(a), f4(b), f4(c));
    from_f4(std::array::from_fn(|i| if negate { -x[i].mul_add(y[i], -z[i]) } else { x[i].mul_add(y[i], z[i]) }))
}

/// Standard-VMX float ops (VX-form xo). Returns None if `xo` isn't one of them.
pub(crate) fn vx_float(xo: u32, a: u128, b: u128, va: usize) -> Option<u128> {
    Some(match xo {
        10 => map_f2(a, b, |x, y| x + y),
        74 => map_f2(a, b, |x, y| x - y),
        1034 => map_f2(a, b, f32::max),
        1098 => map_f2(a, b, f32::min),
        394 => map_f1(b, f32::exp2),                     // vexptefp
        458 => map_f1(b, f32::log2),                     // vlogefp
        266 => map_f1(b, |x| 1.0 / x),                   // vrefp
        330 => map_f1(b, |x| 1.0 / x.sqrt()),            // vrsqrtefp
        522 => map_f1(b, f32::round_ties_even),          // vrfin
        586 => map_f1(b, f32::trunc),                    // vrfiz
        650 => map_f1(b, f32::ceil),                     // vrfip
        714 => map_f1(b, f32::floor),                    // vrfim
        842 => {
            // vcfsx: signed int -> float, scaled by 2^-UIMM
            let scale = 1.0 / (1u64 << (va & 31)) as f32;
            let w = b.to_be_bytes();
            from_f4(std::array::from_fn(|i| i32::from_be_bytes(w[4 * i..4 * i + 4].try_into().unwrap()) as f32 * scale))
        }
        970 => {
            // vctsxs: float -> signed int saturate, scaled by 2^UIMM
            let scale = (1u64 << (va & 31)) as f32;
            let v = f4(b);
            let mut o = [0u8; 16];
            for i in 0..4 {
                let r = if v[i].is_nan() { 0 } else { (v[i] * scale).trunc().clamp(i32::MIN as f32, i32::MAX as f32) as i32 };
                o[4 * i..4 * i + 4].copy_from_slice(&r.to_be_bytes());
            }
            u128::from_be_bytes(o)
        }
        _ => return None,
    })
}

fn sext5(v: u32) -> i32 {
    ((v as i32) << 27) >> 27
}

impl Cpu {
    /// Opcode 4: VA-form (6-bit xo) and VX-form (11-bit xo).
    #[inline(never)]
    pub(crate) fn exec_vmx(&mut self, inst: u32) -> bool {
        let vd = ((inst >> 21) & 31) as usize;
        let va = ((inst >> 16) & 31) as usize;
        let vb = ((inst >> 11) & 31) as usize;
        let vc = ((inst >> 6) & 31) as usize;
        let (a, b, c) = (self.vr[va], self.vr[vb], self.vr[vc]);

        // VA-form
        match inst & 0x3F {
            42 => {
                // vsel: take b where c is 1
                self.vr[vd] = (a & !c) | (b & c);
                return true;
            }
            43 => {
                // vperm
                let (ab, bb, cb) = (a.to_be_bytes(), b.to_be_bytes(), c.to_be_bytes());
                let mut o = [0u8; 16];
                for i in 0..16 {
                    let idx = (cb[i] & 31) as usize;
                    o[i] = if idx < 16 { ab[idx] } else { bb[idx - 16] };
                }
                self.vr[vd] = u128::from_be_bytes(o);
                return true;
            }
            46 => {
                self.vr[vd] = fma(a, c, b, false); // vmaddfp: vA*vC + vB
                return true;
            }
            47 => {
                self.vr[vd] = fma(a, c, b, true); // vnmsubfp: -(vA*vC - vB)
                return true;
            }
            44 => {
                // vsldoi
                let sh = ((inst >> 6) & 15) * 8;
                self.vr[vd] = if sh == 0 { a } else { (a << sh) | (b >> (128 - sh)) };
                return true;
            }
            op @ 32..=41 => {
                if let Some(r) = crate::vmx_int::va_int(op, a, b, c) {
                    self.vr[vd] = r;
                    return true;
                }
            }
            _ => {}
        }

        // VX-form
        if matches!(inst & 0x3FF, 6 | 70 | 134 | 198 | 454 | 518 | 582 | 646 | 710 | 774 | 838 | 902 | 966) {
            return self.exec_vcmp(inst);
        }
        let splat =|imm: u32, lane: u32| -> u128 {
            match lane {
                8 => u128::from_be_bytes([imm as u8; 16]),
                16 => zip16(0, 0, |_, _| imm as u16),
                _ => zip32(0, 0, |_, _| imm),
            }
        };
        let r = match inst & 0x7FF {
            // splat immediate
            780 => splat(sext5(va as u32) as u32, 8),
            844 => splat(sext5(va as u32) as u32, 16),
            908 => splat(sext5(va as u32) as u32, 32),
            // splat element
            524 => u128::from_be_bytes([b.to_be_bytes()[va & 15]; 16]),
            588 => {
                let by = b.to_be_bytes();
                let i = (va & 7) * 2;
                splat(u16::from_be_bytes([by[i], by[i + 1]]) as u32, 16)
            }
            652 => {
                let by = b.to_be_bytes();
                let i = (va & 3) * 4;
                splat(u32::from_be_bytes(by[i..i + 4].try_into().unwrap()), 32)
            }
            // logic
            1028 => a & b,
            1092 => a & !b,
            1156 => a | b,
            1220 => a ^ b,
            1284 => !(a | b),
            // modulo add / subtract
            0 => zip8(a, b, |x, y| x.wrapping_add(y)),
            64 => zip16(a, b, |x, y| x.wrapping_add(y)),
            128 => zip32(a, b, |x, y| x.wrapping_add(y)),
            1024 => zip8(a, b, |x, y| x.wrapping_sub(y)),
            1088 => zip16(a, b, |x, y| x.wrapping_sub(y)),
            1152 => zip32(a, b, |x, y| x.wrapping_sub(y)),
            x => match vx_lanes(x, a, b, va).or_else(|| vx_float(x, a, b, va)).or_else(|| crate::vmx_int::vx_int(x, a, b)) {
                Some(r) => r,
                None => return false,
            },
        };
        self.vr[vd] = r;
        true
    }

    /// VMX128 arithmetic/logic (primary opcode 5). The 7-bit register indices are scattered:
    /// VD = inst[25:21] | inst[3:2]<<5, VA = inst[20:16] | inst[5] | inst[10]<<6,
    /// VB = inst[15:11] | inst[1:0]<<5. The operation key is inst & 0x3D0.
    #[inline(never)]
    pub(crate) fn exec_vmx128_arith(&mut self, inst: u32) -> bool {
        let vd = (((inst >> 21) & 31) | ((inst & 0xC) << 3)) as usize;
        let va = (((inst >> 16) & 31) | (inst & 0x20) | ((inst & 0x400) >> 4)) as usize;
        let vb = (((inst >> 11) & 31) | ((inst & 3) << 5)) as usize;
        let (a, b, d) = (self.vr[va], self.vr[vb], self.vr[vd]);
        let dot = |n: usize| {
            let (x, y) = (f4(a), f4(b));
            let s: f32 = (0..n).map(|i| x[i] * y[i]).sum();
            from_f4([s; 4])
        };
        self.vr[vd] = match inst & 0x3D0 {
            0x010 => map_f2(a, b, |x, y| x + y),  // vaddfp128
            0x050 => map_f2(a, b, |x, y| x - y),  // vsubfp128
            0x090 => map_f2(a, b, |x, y| x * y),  // vmulfp128
            0x0D0 => fma(a, b, d, false),         // vmaddfp128: va*vb + vd
            0x110 => fma(a, d, b, false),         // vmaddcfp128: va*vd + vb
            0x150 => fma(a, b, d, true),          // vnmsubfp128: -(va*vb - vd)
            0x190 => dot(3),                      // vmsum3fp128
            0x1D0 => dot(4),                      // vmsum4fp128
            0x210 => a & b,                       // vand128
            0x250 => a & !b,                      // vandc128
            0x290 => !(a | b),                    // vnor128
            0x2D0 => a | b,                       // vor128
            0x310 => a ^ b,                       // vxor128
            0x350 => (a & !d) | (b & d),          // vsel128
            0x390 => vx_lanes(1036, a, b, 0).unwrap(), // vslo128
            0x3D0 => vx_lanes(1100, a, b, 0).unwrap(), // vsro128
            0x300 => vx_lanes(14, a, b, 0).unwrap(),   // vpkuhum128
            0x380 => vx_lanes(78, a, b, 0).unwrap(),   // vpkuwum128
            _ => return false,
        };
        true
    }

    /// vcmp* family. `base` has the Rc bit (0x400) stripped. Rc updates CR6.
    #[inline(never)]
    fn exec_vcmp(&mut self, inst: u32) -> bool {
        let vd = ((inst >> 21) & 31) as usize;
        let (a, b) = (self.vr[((inst >> 16) & 31) as usize], self.vr[((inst >> 11) & 31) as usize]);
        let m = |t: bool| if t { !0u8 } else { 0 };
        let r = match inst & 0x3FF {
            6 => zip8(a, b, |x, y| m(x == y)),
            70 => zip16(a, b, |x, y| m(x == y) as u16 * 0x101),
            134 => zip32(a, b, |x, y| m(x == y) as u32 * 0x0101_0101),
            518 => zip8(a, b, |x, y| m(x > y)),
            582 => zip16(a, b, |x, y| m(x > y) as u16 * 0x101),
            646 => zip32(a, b, |x, y| m(x > y) as u32 * 0x0101_0101),
            198 => zip32(a, b, |x, y| m(f32::from_bits(x) == f32::from_bits(y)) as u32 * 0x0101_0101), // vcmpeqfp
            454 => zip32(a, b, |x, y| m(f32::from_bits(x) >= f32::from_bits(y)) as u32 * 0x0101_0101), // vcmpgefp
            710 => zip32(a, b, |x, y| m(f32::from_bits(x) > f32::from_bits(y)) as u32 * 0x0101_0101),  // vcmpgtfp
            966 => zip32(a, b, |x, y| {
                // vcmpbfp: bit31 = a > b, bit30 = a < -b (out of bounds); zero means within [-b, b]
                let (x, y) = (f32::from_bits(x), f32::from_bits(y));
                (((x > y) as u32) << 31) | (((x < -y) as u32) << 30)
            }),
            774 => zip8(a, b, |x, y| m(x as i8 > y as i8)),
            838 => zip16(a, b, |x, y| m(x as i16 > y as i16) as u16 * 0x101),
            902 => zip32(a, b, |x, y| m(x as i32 > y as i32) as u32 * 0x0101_0101),
            _ => return false,
        };
        self.vr[vd] = r;
        if inst & 0x400 != 0 {
            let cr6 = ((r == !0) as u32) << 3 | ((r == 0) as u32) << 1;
            self.cr = (self.cr & !(0xF << 4)) | (cr6 << 4);
        }
        true
    }

    /// Vector loads/stores under primary opcode 31. EA low 4 bits are ignored.
    #[inline(never)]
    pub(crate) fn exec_vmx_x<B: Bus>(&mut self, bus: &mut B, inst: u32) -> bool {
        let xo = (inst >> 1) & 0x3FF;
        let vd = ((inst >> 21) & 31) as usize;
        let ra = ((inst >> 16) & 31) as usize;
        let rb = ((inst >> 11) & 31) as usize;
        let raw_ea = (if ra == 0 { 0 } else { self.gpr[ra] }).wrapping_add(self.gpr[rb]) as u32;
        self.vec_mem(bus, xo, vd, raw_ea)
    }

    /// VMX128 vector loads/stores (opcode 4, mask 0x7F3). The register index has two extra
    /// high bits at inst[3:2]: vd = ((inst >> 21) & 31) | (inst & 0xC) << 3.
    #[inline(never)]
    pub(crate) fn exec_vmx128_mem<B: Bus>(&mut self, bus: &mut B, inst: u32) -> bool {
        // 128-form key -> equivalent standard X-form xo
        let std_xo = match inst & 0x7F3 {
            0x003 => 6,   // lvsl128
            0x043 => 38,  // lvsr128
            0x083 => 71,  // lvewx128
            0x0C3 => 103, // lvx128
            0x183 => 199, // stvewx128
            0x1C3 => 231, // stvx128
            0x2C3 => 359, // lvxl128
            0x3C3 => 487, // stvxl128
            0x403 => 519, // lvlx128
            0x443 => 551, // lvrx128
            0x503 => 647, // stvlx128
            0x543 => 679, // stvrx128
            0x603 => 775, // lvlxl128
            0x643 => 807, // lvrxl128
            0x703 => 903, // stvlxl128
            0x743 => 935, // stvrxl128
            _ => return false,
        };
        let vd = (((inst >> 21) & 31) | ((inst & 0xC) << 3)) as usize;
        let ra = ((inst >> 16) & 31) as usize;
        let rb = ((inst >> 11) & 31) as usize;
        let raw_ea = (if ra == 0 { 0 } else { self.gpr[ra] }).wrapping_add(self.gpr[rb]) as u32;
        self.vec_mem(bus, std_xo, vd, raw_ea)
    }

    fn vec_mem<B: Bus>(&mut self, bus: &mut B, xo: u32, vd: usize, raw_ea: u32) -> bool {
        let ea = raw_ea & !15;
        match xo {
            6 | 38 => {
                // lvsl / lvsr: permute controls for unaligned access
                let sh = (raw_ea & 15) as u8;
                let mut o = [0u8; 16];
                for i in 0..16u8 {
                    o[i as usize] = if xo == 6 { sh + i } else { 16 - sh + i };
                }
                self.vr[vd] = u128::from_be_bytes(o);
            }
            519 | 775 | 551 | 807 => {
                // lvlx(l) / lvrx(l): Xenon partial loads around a 16-byte boundary.
                // lvlx: bytes EA..end-of-block, left-justified. lvrx: block start..EA, right-justified.
                let k = (raw_ea & 15) as usize;
                let mut o = [0u8; 16];
                if xo == 519 || xo == 775 {
                    for i in 0..16 - k {
                        o[i] = bus.read8(raw_ea + i as u32);
                    }
                } else {
                    for i in 0..k {
                        o[16 - k + i] = bus.read8(ea + i as u32);
                    }
                }
                self.vr[vd] = u128::from_be_bytes(o);
            }
            647 | 903 | 679 | 935 => {
                // stvlx(l) / stvrx(l): store the left (16-k) bytes at EA.., or the right k bytes at block start.
                let k = (raw_ea & 15) as usize;
                let bytes = self.vr[vd].to_be_bytes();
                if xo == 647 || xo == 903 {
                    for i in 0..16 - k {
                        bus.write8(raw_ea + i as u32, bytes[i]);
                    }
                } else {
                    for i in 0..k {
                        bus.write8(ea + i as u32, bytes[16 - k + i]);
                    }
                }
            }
            7 | 39 | 71 => {
                // lvebx / lvehx / lvewx: element lands in the lane matching EA's position
                let size = 1u32 << ((xo - 7) / 32).min(2);
                let mut bytes = self.vr[vd].to_be_bytes();
                let lane = (raw_ea & 15 & !(size - 1)) as usize;
                for i in 0..size as usize {
                    bytes[lane + i] = bus.read8((raw_ea & !(size - 1)) + i as u32);
                }
                self.vr[vd] = u128::from_be_bytes(bytes);
            }
            135 | 167 | 199 => {
                // stvebx / stvehx / stvewx
                let size = 1u32 << ((xo - 135) / 32).min(2);
                let bytes = self.vr[vd].to_be_bytes();
                let aligned = raw_ea & !(size - 1);
                let lane = (aligned & 15) as usize;
                for i in 0..size as usize {
                    bus.write8(aligned + i as u32, bytes[lane + i]);
                }
            }
            103 | 359 => {
                // lvx / lvxl
                self.vr[vd] = ((bus.read64(ea) as u128) << 64) | bus.read64(ea + 8) as u128;
            }
            231 | 487 => {
                // stvx / stvxl
                let v = self.vr[vd];
                bus.write64(ea, (v >> 64) as u64);
                bus.write64(ea + 8, v as u64);
            }
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::{Bus, FlatMemory};
    use crate::Cpu;

    fn vx(vd: u32, va: u32, vb: u32, xo: u32) -> u32 {
        (4 << 26) | (vd << 21) | (va << 16) | (vb << 11) | xo
    }

    #[test]
    fn splat_logic_and_add() {
        let mut m = FlatMemory::new(16);
        let mut c = Cpu::new();
        m.write32(0, vx(0, 4, 0, 780)); // vspltisb v0,4
        m.write32(4, vx(1, 0x1F, 0, 908)); // vspltisw v1,-1
        m.write32(8, vx(2, 0, 1, 1220)); // vxor v2,v0,v1
        m.write32(12, vx(3, 0, 0, 128)); // vadduwm v3,v0,v0
        for _ in 0..4 {
            c.step(&mut m).unwrap();
        }
        assert_eq!(c.vr[0], 0x0404_0404_0404_0404_0404_0404_0404_0404);
        assert_eq!(c.vr[1], u128::MAX);
        assert_eq!(c.vr[2], !c.vr[0]);
        assert_eq!(c.vr[3], 0x0808_0808_0808_0808_0808_0808_0808_0808);
    }

    fn setf(c: &mut Cpu, r: usize, v: [f32; 4]) {
        c.vr[r] = super::from_f4(v);
    }

    #[test]
    fn vmx128_float_arith_and_scattered_register_bits() {
        let mut m = FlatMemory::new(32);
        let mut c = Cpu::new();
        setf(&mut c, 13, [1.0, 2.0, 3.0, 4.0]);
        setf(&mut c, 6, [10.0, 10.0, 10.0, 10.0]);
        // The real crash: 0x14cd3090 = vmulfp128 v6 = v13 * v6
        m.write32(0, 0x14CD3090);
        c.step(&mut m).unwrap();
        assert_eq!(super::f4(c.vr[6]), [10.0, 20.0, 30.0, 40.0]);
        // vmsum4fp128 broadcast: use va=13 (low bits only), vb=13, vd=100 -> vd high bits 0b11 -> inst&0xC = 0xC
        setf(&mut c, 100, [0.0; 4]);
        let inst = (5 << 26) | ((100 & 31) << 21) | (13 << 16) | (13 << 11) | 0x1D0 | ((100 >> 5) << 2);
        m.write32(4, inst);
        c.step(&mut m).unwrap();
        assert_eq!(super::f4(c.vr[100]), [30.0; 4]); // 1+4+9+16
    }

    #[test]
    fn standard_vmx_float() {
        let mut m = FlatMemory::new(32);
        let mut c = Cpu::new();
        setf(&mut c, 1, [1.0, 2.0, 3.0, 4.0]);
        setf(&mut c, 2, [0.5, 0.5, 0.5, 0.5]);
        setf(&mut c, 3, [10.0, 10.0, 10.0, 10.0]);
        m.write32(0, vx(4, 1, 2, 10)); // vaddfp v4 = v1 + v2
        m.write32(4, (4 << 26) | (5 << 21) | (1 << 16) | (3 << 11) | (2 << 6) | 46); // vmaddfp v5 = v1*v2 + v3
        c.step(&mut m).unwrap();
        c.step(&mut m).unwrap();
        assert_eq!(super::f4(c.vr[4]), [1.5, 2.5, 3.5, 4.5]);
        assert_eq!(super::f4(c.vr[5]), [10.5, 11.0, 11.5, 12.0]);
    }

    #[test]
    fn vmx128_load_store_uses_extended_register_index() {
        let mut m = FlatMemory::new(0x100);
        let mut c = Cpu::new();
        // stvx128 v70,r0,r11 : vd = 70 = 0b1000110 -> low5 = 6, high2 = 2 -> inst[3:2] = 0b10 (0x8)
        c.vr[70] = 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00;
        c.gpr[11] = 0x40;
        let inst = (4 << 26) | (6 << 21) | (11 << 11) | 0x1C3 | 0x8;
        m.write32(0, inst);
        c.step(&mut m).unwrap();
        assert_eq!(m.read32(0x40), 0x1122_3344);
        // lvx128 v71 <- same memory (71 = low5 7, high2 2)
        m.write32(4, (4 << 26) | (7 << 21) | (11 << 11) | 0x0C3 | 0x8);
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[71], c.vr[70]);
    }

    #[test]
    fn float_compares_set_masks_and_cr6() {
        let mut m = FlatMemory::new(16);
        let mut c = Cpu::new();
        setf(&mut c, 1, [1.0, 5.0, 2.0, 9.0]);
        setf(&mut c, 2, [0.0, 5.0, 3.0, 1.0]);
        m.write32(0, vx(3, 1, 2, 710)); // vcmpgtfp v3 = v1 > v2 per lane
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[3], 0xFFFF_FFFF_0000_0000_0000_0000_FFFF_FFFF);
        // The real crash instruction 0x10A00AC6 is vcmpgtfp v5,v0,v1 (no Rc); with Rc (|0x400) CR6 is set too.
        c.vr[0] = super::from_f4([2.0; 4]);
        c.vr[1] = super::from_f4([1.0; 4]);
        m.write32(4, 0x10A0_0AC6);
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[5], u128::MAX);
        assert_eq!((c.cr >> 4) & 0xF, 0); // untouched without Rc
        m.write32(8, 0x10A0_0EC6);
        c.step(&mut m).unwrap();
        assert_eq!((c.cr >> 4) & 0xF, 0b1000); // CR6: all lanes true
    }

    #[test]
    fn partial_vector_loads_stores() {
        let mut m = FlatMemory::new(0x100);
        let mut c = Cpu::new();
        let x = |vd: u32, xo: u32| (31 << 26) | (vd << 21) | (5 << 11) | (xo << 1);
        c.vr[1] = 0x0001_0203_0405_0607_0809_0A0B_0C0D_0E0F;
        // stvlx at 0x2C (k=12): stores vector bytes 0..4 to 0x2C..0x30
        c.gpr[5] = 0x2C;
        m.write32(0, x(1, 647));
        c.step(&mut m).unwrap();
        assert_eq!(m.read32(0x2C), 0x0001_0203);
        assert_eq!(m.read32(0x30), 0); // next block untouched
        // stvrx at 0x34 (k=4): stores vector bytes 12..16 to 0x30..0x34
        c.gpr[5] = 0x34;
        m.write32(4, x(1, 679));
        c.step(&mut m).unwrap();
        assert_eq!(m.read32(0x30), 0x0C0D_0E0F);
        // lvlx at 0x2C : bytes 0x2C..0x30 left-justified
        c.gpr[5] = 0x2C;
        m.write32(8, x(2, 519));
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[2], 0x0001_0203_0000_0000_0000_0000_0000_0000);
        // lvrx at 0x34 : bytes 0x30..0x34 right-justified
        c.gpr[5] = 0x34;
        m.write32(12, x(3, 551));
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[3], 0x0C0D_0E0F);
    }

    #[test]
    fn shifts_compares_merge() {
        let mut m = FlatMemory::new(32);
        let mut c = Cpu::new();
        c.vr[1] = 0x0101_0101_0101_0101_0101_0101_0101_0101;
        c.vr[2] = 0x0303_0303_0303_0303_0303_0303_0303_0303;
        m.write32(0, vx(3, 1, 2, 260)); // vslb v3,v1,v2 : 1<<3 = 8
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[3], 0x0808_0808_0808_0808_0808_0808_0808_0808);
        // vcmpequb. v4,v1,v1 -> all ones, CR6 "all true" (LT bit of field 6 = 0x80 of bits 4..7)
        m.write32(4, vx(4, 1, 1, 6 + 1024));
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[4], u128::MAX);
        assert_eq!((c.cr >> 4) & 0xF, 0b1000);
        // vmrghb v5,v1,v2 interleaves 01 03 01 03 ...
        m.write32(8, vx(5, 1, 2, 12));
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[5], 0x0103_0103_0103_0103_0103_0103_0103_0103);
        // vupkhsb v6,v?,v7 sign-extends high 8 bytes
        c.vr[7] = 0xFF01_0000_0000_0000_0000_0000_0000_0000;
        m.write32(12, vx(6, 0, 7, 526));
        c.step(&mut m).unwrap();
        assert_eq!((c.vr[6] >> 112) as u16, 0xFFFF);
        assert_eq!(((c.vr[6] >> 96) & 0xFFFF) as u16, 0x0001);
    }

    #[test]
    fn lvsl_lvsr_and_element_access() {
        let mut m = FlatMemory::new(0x100);
        let mut c = Cpu::new();
        c.gpr[5] = 0x23; // misalignment 3
        let x = |vd: u32, xo: u32| (31 << 26) | (vd << 21) | (5 << 11) | (xo << 1);
        m.write32(0, x(1, 6)); // lvsl v1,0,r5
        m.write32(4, x(2, 38)); // lvsr v2,0,r5
        c.step(&mut m).unwrap();
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[1].to_be_bytes()[0], 3);
        assert_eq!(c.vr[1].to_be_bytes()[15], 18);
        assert_eq!(c.vr[2].to_be_bytes()[0], 13);
        // stvewx v3 at 0x24 stores lane 1 (bytes 4..8 of the vector)
        c.vr[3] = 0x0000_0000_AABB_CCDD_0000_0000_0000_0000;
        c.gpr[5] = 0x24;
        m.write32(8, x(3, 199));
        c.step(&mut m).unwrap();
        assert_eq!(m.read32(0x24), 0xAABB_CCDD);
        // lvewx v4 at 0x24 puts the word back in lane 1
        m.write32(12, x(4, 71));
        c.step(&mut m).unwrap();
        assert_eq!((c.vr[4] >> 64) as u32, 0xAABB_CCDD);
    }

    #[test]
    fn perm_and_memory() {
        let mut m = FlatMemory::new(0x100);
        let mut c = Cpu::new();
        c.vr[1] = 0x0001_0203_0405_0607_0809_0A0B_0C0D_0E0F;
        c.vr[2] = 0x1011_1213_1415_1617_1819_1A1B_1C1D_1E1F;
        // reverse-bytes control: 31,30,...,16 picks b reversed
        c.vr[3] = 0x1F1E_1D1C_1B1A_1918_1716_1514_1312_1110;
        m.write32(0, (4 << 26) | (4 << 21) | (1 << 16) | (2 << 11) | (3 << 6) | 43); // vperm v4,v1,v2,v3
        c.step(&mut m).unwrap();
        assert_eq!(c.vr[4], 0x1F1E_1D1C_1B1A_1918_1716_1514_1312_1110);
        // stvx v4,r0,r5 ; lvx v6,r0,r5 (EA 0x83 -> aligned to 0x80)
        c.gpr[5] = 0x83;
        m.write32(4, (31 << 26) | (4 << 21) | (5 << 11) | (231 << 1));
        m.write32(8, (31 << 26) | (6 << 21) | (5 << 11) | (103 << 1));
        c.step(&mut m).unwrap();
        c.step(&mut m).unwrap();
        assert_eq!(m.read8(0x80), 0x1F);
        assert_eq!(c.vr[6], c.vr[4]);
    }
}
