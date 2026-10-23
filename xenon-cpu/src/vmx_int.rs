//! VMX integer arithmetic beyond the modulo forms: saturating add/sub, averages, widening
//! multiplies, sum-across, saturating packs, and the VA-form multiply-add/sum family.
//! Saturation never sets VSCR[SAT] (nothing in practice reads it).

fn lanes8(v: u128) -> [u8; 16] {
    v.to_be_bytes()
}
fn lanes16(v: u128) -> [u16; 8] {
    let b = v.to_be_bytes();
    std::array::from_fn(|i| u16::from_be_bytes([b[2 * i], b[2 * i + 1]]))
}
fn lanes32(v: u128) -> [u32; 4] {
    let b = v.to_be_bytes();
    std::array::from_fn(|i| u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}
fn pack16(l: [u16; 8]) -> u128 {
    let mut o = [0u8; 16];
    for i in 0..8 {
        o[2 * i..2 * i + 2].copy_from_slice(&l[i].to_be_bytes());
    }
    u128::from_be_bytes(o)
}
fn pack32(l: [u32; 4]) -> u128 {
    let mut o = [0u8; 16];
    for i in 0..4 {
        o[4 * i..4 * i + 4].copy_from_slice(&l[i].to_be_bytes());
    }
    u128::from_be_bytes(o)
}

fn sat_u(v: i128, max: i128) -> i128 {
    v.clamp(0, max)
}
fn sat_s(v: i128, bits: u32) -> i128 {
    v.clamp(-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
}

/// Element `i` of width `bits` (big-endian lane order), sign- or zero-extended.
fn elem(v: u128, bits: u32, i: u32, signed: bool) -> i128 {
    let sh = 128 - bits * (i + 1);
    let mask = (1u128 << bits) - 1;
    let x = (v >> sh) & mask;
    if signed { ((x << (128 - bits)) as i128) >> (128 - bits) } else { x as i128 }
}

fn put(out: &mut u128, bits: u32, i: u32, val: i128) {
    let sh = 128 - bits * (i + 1);
    let mask = (1u128 << bits) - 1;
    *out |= ((val as u128) & mask) << sh;
}

/// VX-form integer ops by 11-bit xo. None if not one of ours.
pub(crate) fn vx_int(xo: u32, a: u128, b: u128) -> Option<u128> {
    let sat_arith = |bits: u32, signed: bool, sub: bool| -> u128 {
        let mut out = 0u128;
        for i in 0..128 / bits {
            let (x, y) = (elem(a, bits, i, signed), elem(b, bits, i, signed));
            let r = if sub { x - y } else { x + y };
            let r = if signed { sat_s(r, bits) } else { sat_u(r, (1i128 << bits) - 1) };
            put(&mut out, bits, i, r);
        }
        out
    };
    let avg = |bits: u32, signed: bool| -> u128 {
        let mut out = 0u128;
        for i in 0..128 / bits {
            put(&mut out, bits, i, (elem(a, bits, i, signed) + elem(b, bits, i, signed) + 1) >> 1);
        }
        out
    };
    // widening multiply of the even (or odd) source elements into double-width lanes
    let mul = |bits: u32, signed: bool, odd: bool| -> u128 {
        let mut out = 0u128;
        for k in 0..(128 / bits) / 2 {
            let i = 2 * k + odd as u32;
            put(&mut out, 2 * bits, k, elem(a, bits, i, signed) * elem(b, bits, i, signed));
        }
        out
    };
    Some(match xo {
        512 => sat_arith(8, false, false),
        576 => sat_arith(16, false, false),
        640 => sat_arith(32, false, false),
        768 => sat_arith(8, true, false),
        832 => sat_arith(16, true, false),
        896 => sat_arith(32, true, false),
        1536 => sat_arith(8, false, true),
        1600 => sat_arith(16, false, true),
        1664 => sat_arith(32, false, true),
        1792 => sat_arith(8, true, true),
        1856 => sat_arith(16, true, true),
        1920 => sat_arith(32, true, true),
        384 => pack32(std::array::from_fn(|i| (lanes32(a)[i] as u64 + lanes32(b)[i] as u64 > u32::MAX as u64) as u32)),
        1408 => pack32(std::array::from_fn(|i| (lanes32(a)[i] >= lanes32(b)[i]) as u32)),
        1026 => avg(8, false),
        1090 => avg(16, false),
        1154 => avg(32, false),
        1282 => avg(8, true),
        1346 => avg(16, true),
        1410 => avg(32, true),
        520 => mul(8, false, false),
        8 => mul(8, false, true),
        776 => mul(8, true, false),
        264 => mul(8, true, true),
        584 => mul(16, false, false),
        72 => mul(16, false, true),
        840 => mul(16, true, false),
        328 => mul(16, true, true),
        // vsum4ubs / vsum4sbs: each word = sum of 4 bytes of vA + the word of vB
        1544 | 1800 => {
            let (ab, bw) = (lanes8(a), lanes32(b));
            pack32(std::array::from_fn(|i| {
                let s: i128 = (0..4).map(|k| if xo == 1544 { ab[4 * i + k] as i128 } else { ab[4 * i + k] as i8 as i128 }).sum();
                if xo == 1544 { sat_u(s + bw[i] as i128, u32::MAX as i128) as u32 } else { sat_s(s + bw[i] as i32 as i128, 32) as u32 }
            }))
        }
        // vsum4shs
        1608 => {
            let (ah, bw) = (lanes16(a), lanes32(b));
            pack32(std::array::from_fn(|i| {
                sat_s(ah[2 * i] as i16 as i128 + ah[2 * i + 1] as i16 as i128 + bw[i] as i32 as i128, 32) as u32
            }))
        }
        // vsum2sws: words 1 and 3 get (a0+a1)+b1 and (a2+a3)+b3
        1672 => {
            let (aw, bw) = (lanes32(a), lanes32(b));
            let s = |x: usize, y: usize, z: usize| sat_s(aw[x] as i32 as i128 + aw[y] as i32 as i128 + bw[z] as i32 as i128, 32) as u32;
            pack32([0, s(0, 1, 1), 0, s(2, 3, 3)])
        }
        // vsumsws: word 3 = sum of all four words of vA + word 3 of vB
        1928 => {
            let (aw, bw) = (lanes32(a), lanes32(b));
            let s: i128 = aw.iter().map(|&x| x as i32 as i128).sum::<i128>() + bw[3] as i32 as i128;
            pack32([0, 0, 0, sat_s(s, 32) as u32])
        }
        // vpkuhus / vpkshus / vpkshss: halfwords -> bytes
        142 | 270 | 398 => {
            let (x, y) = (lanes16(a), lanes16(b));
            let cvt = |v: u16| -> u8 {
                match xo {
                    142 => sat_u(v as i128, 255) as u8,
                    270 => sat_u(v as i16 as i128, 255) as u8,
                    _ => sat_s(v as i16 as i128, 8) as i8 as u8,
                }
            };
            let mut o = [0u8; 16];
            for i in 0..8 {
                o[i] = cvt(x[i]);
                o[8 + i] = cvt(y[i]);
            }
            u128::from_be_bytes(o)
        }
        // vpkuwus / vpkswus / vpkswss: words -> halfwords
        206 | 334 | 462 => {
            let (x, y) = (lanes32(a), lanes32(b));
            let cvt = |v: u32| -> u16 {
                match xo {
                    206 => sat_u(v as i128, 65535) as u16,
                    334 => sat_u(v as i32 as i128, 65535) as u16,
                    _ => sat_s(v as i32 as i128, 16) as i16 as u16,
                }
            };
            pack16(std::array::from_fn(|i| if i < 4 { cvt(x[i]) } else { cvt(y[i - 4]) }))
        }
        // vpkpx: 8:8:8:8 words -> 1:5:5:5 halfwords
        782 => {
            let (x, y) = (lanes32(a), lanes32(b));
            let cvt = |w: u32| -> u16 { (((w >> 9) & 0x8000) | ((w >> 6) & 0x7C00) | ((w >> 3) & 0x03E0) | ((w >> 3) & 0x1F)) as u16 };
            pack16(std::array::from_fn(|i| if i < 4 { cvt(x[i]) } else { cvt(y[i - 4]) }))
        }
        // vupkhpx / vupklpx: 1:5:5:5 halfwords of vB -> 8:8:8:8 words (the caller passes vB as `b`)
        846 | 974 => {
            let h = lanes16(b);
            let base = if xo == 846 { 0 } else { 4 };
            pack32(std::array::from_fn(|i| {
                let v = h[base + i] as u32;
                (((v >> 15) & 1) * 0xFF) << 24 | ((v >> 10) & 0x1F) << 16 | ((v >> 5) & 0x1F) << 8 | (v & 0x1F)
            }))
        }
        _ => return None,
    })
}

/// VA-form integer ops (6-bit xo): multiply-add and multiply-sum family.
pub(crate) fn va_int(op: u32, a: u128, b: u128, c: u128) -> Option<u128> {
    let (ah, bh, ch) = (lanes16(a), lanes16(b), lanes16(c));
    let cw = lanes32(c);
    let (ab, bb) = (lanes8(a), lanes8(b));
    Some(match op {
        // vmhaddshs / vmhraddshs: ((a*b [+ 0x4000]) >> 15) + c, saturating
        32 | 33 => pack16(std::array::from_fn(|i| {
            let p = ah[i] as i16 as i32 * bh[i] as i16 as i32 + if op == 33 { 0x4000 } else { 0 };
            sat_s((p >> 15) as i128 + ch[i] as i16 as i128, 16) as i16 as u16
        })),
        // vmladduhm: low 16 bits of a*b + c
        34 => pack16(std::array::from_fn(|i| ah[i].wrapping_mul(bh[i]).wrapping_add(ch[i]))),
        // vmsumubm / vmsummbm: sum of four byte products per word + c (vA signed for vmsummbm)
        36 | 37 => pack32(std::array::from_fn(|i| {
            let mut s = cw[i] as i64;
            for k in 0..4 {
                let x = if op == 37 { ab[4 * i + k] as i8 as i64 } else { ab[4 * i + k] as i64 };
                s += x * bb[4 * i + k] as i64;
            }
            s as u32
        })),
        // vmsumuhm / vmsumuhs
        38 | 39 => pack32(std::array::from_fn(|i| {
            let s = ah[2 * i] as u64 * bh[2 * i] as u64 + ah[2 * i + 1] as u64 * bh[2 * i + 1] as u64 + cw[i] as u64;
            if op == 39 { s.min(u32::MAX as u64) as u32 } else { s as u32 }
        })),
        // vmsumshm / vmsumshs
        40 | 41 => pack32(std::array::from_fn(|i| {
            let s = ah[2 * i] as i16 as i64 * bh[2 * i] as i16 as i64
                + ah[2 * i + 1] as i16 as i64 * bh[2 * i + 1] as i16 as i64
                + cw[i] as i32 as i64;
            if op == 41 { s.clamp(i32::MIN as i64, i32::MAX as i64) as u32 } else { s as u32 }
        })),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(a: u32, b: u32, c: u32, d: u32) -> u128 {
        pack32([a, b, c, d])
    }

    #[test]
    fn unsigned_word_saturating_add() {
        let r = vx_int(640, w(0xFFFF_FFF0, 1, 5, 0xFFFF_FFFF), w(0x20, 2, 6, 1)).unwrap();
        assert_eq!(lanes32(r), [u32::MAX, 3, 11, u32::MAX]);
    }

    #[test]
    fn signed_byte_saturating() {
        let a = u128::from_be_bytes([100; 16]);
        assert_eq!(lanes8(vx_int(768, a, a).unwrap()), [127; 16]);
        assert_eq!(lanes8(vx_int(1792, u128::from_be_bytes([(-100i8) as u8; 16]), a).unwrap()), [0x80; 16]);
        assert_eq!(lanes8(vx_int(1536, u128::from_be_bytes([5; 16]), u128::from_be_bytes([9; 16])).unwrap()), [0; 16]);
    }

    #[test]
    fn avg_rounds_up_and_even_odd_multiply() {
        assert_eq!(lanes8(vx_int(1026, u128::from_be_bytes([1; 16]), u128::from_be_bytes([2; 16])).unwrap()), [2; 16]);
        let a = pack16([2, 3, 4, 5, 6, 7, 8, 9]);
        let b = pack16([10; 8]);
        assert_eq!(lanes32(vx_int(584, a, b).unwrap()), [20, 40, 60, 80]); // vmuleuh
        assert_eq!(lanes32(vx_int(72, a, b).unwrap()), [30, 50, 70, 90]); // vmulouh
    }

    #[test]
    fn saturating_packs() {
        let a = w(70000, 5, 0xFFFF_FFFF, 65535);
        assert_eq!(lanes16(vx_int(206, a, a).unwrap())[..4], [65535, 5, 65535, 65535]); // vpkuwus
        assert_eq!(lanes16(vx_int(334, a, a).unwrap())[..4], [65535, 5, 0, 65535]); // vpkswus
        assert_eq!(lanes16(vx_int(462, a, a).unwrap())[..4], [0x7FFF, 5, 0xFFFF, 0x7FFF]); // vpkswss
    }

    #[test]
    fn multiply_sum_and_add() {
        let a = pack16([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = pack16([10; 8]);
        assert_eq!(lanes32(va_int(38, a, b, w(1, 1, 1, 1)).unwrap()), [31, 71, 111, 151]); // vmsumuhm
        assert_eq!(lanes16(va_int(34, a, b, pack16([1; 8])).unwrap()), [11, 21, 31, 41, 51, 61, 71, 81]); // vmladduhm
    }

    #[test]
    fn sums_across() {
        let a = u128::from_be_bytes([1; 16]);
        let b = w(10, 20, 30, 40);
        assert_eq!(lanes32(vx_int(1544, a, b).unwrap()), [14, 24, 34, 44]); // vsum4ubs
        assert_eq!(lanes32(vx_int(1928, w(1, 2, 3, 4), b).unwrap()), [0, 0, 0, 50]); // vsumsws
    }
}
