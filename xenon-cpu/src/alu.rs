//! Integer ALU: arithmetic, logic, shifts, rotates, compares, CR/MSR moves, reservations.
//! XER bits (64-bit register, LSB numbering): SO = 31, OV = 30, CA = 29.

use crate::memory::Bus;
use crate::Cpu;

const XER_SO: u64 = 1 << 31;
const XER_OV: u64 = 1 << 30;
const XER_CA: u64 = 1 << 29;

fn mask64(mb: u32, me: u32) -> u64 {
    let a = !0u64 >> mb;
    let b = !0u64 << (63 - me);
    if mb <= me { a & b } else { a | b }
}

/// a + b + cin with the carry/overflow of the current mode: 64-bit, or - in 32-bit mode (MSR[SF] = 0, which is
/// how all Xbox 360 user code runs) - of the low word. The 64-bit sum itself is the same either way.
fn add3(a: u64, b: u64, cin: u64, mode32: bool) -> (u64, bool, bool) {
    let (s1, c1) = a.overflowing_add(b);
    let (s2, c2) = s1.overflowing_add(cin);
    if mode32 {
        let wide = (a as u32 as u64) + (b as u32 as u64) + cin;
        let ov = ((a ^ s2) & (b ^ s2)) >> 31 & 1 != 0;
        (s2, wide >> 32 != 0, ov)
    } else {
        let ov = ((a ^ s2) & (b ^ s2)) >> 63 != 0;
        (s2, c1 || c2, ov)
    }
}

impl Cpu {
    fn ca(&self) -> u64 {
        (self.xer & XER_CA) >> 29
    }
    fn set_ca(&mut self, c: bool) {
        if c { self.xer |= XER_CA } else { self.xer &= !XER_CA }
    }
    fn set_ov(&mut self, o: bool) {
        if o { self.xer |= XER_OV | XER_SO } else { self.xer &= !XER_OV }
    }

    /// D-form arithmetic/logic: mulli, subfic, addic(.), xori(s), andi(s).
    pub(crate) fn exec_alu_d(&mut self, opcd: u32, inst: u32) -> bool {
        let rd = ((inst >> 21) & 31) as usize;
        let ra = ((inst >> 16) & 31) as usize;
        let simm = (inst & 0xFFFF) as i16 as i64 as u64;
        let uimm = (inst & 0xFFFF) as u64;
        match opcd {
            7 => self.gpr[rd] = (self.gpr[ra] as i64).wrapping_mul(simm as i64) as u64,
            8 => {
                let (r, c, _) = add3(!self.gpr[ra], simm, 1, self.mode32());
                self.gpr[rd] = r;
                self.set_ca(c);
            }
            12 | 13 => {
                let (r, c, _) = add3(self.gpr[ra], simm, 0, self.mode32());
                self.gpr[rd] = r;
                self.set_ca(c);
                if opcd == 13 {
                    self.set_cr0(r);
                }
            }
            26 => self.gpr[ra] = self.gpr[rd] ^ uimm,
            27 => self.gpr[ra] = self.gpr[rd] ^ (uimm << 16),
            28 | 29 => {
                let v = self.gpr[rd] & if opcd == 28 { uimm } else { uimm << 16 };
                self.gpr[ra] = v;
                self.set_cr0(v);
            }
            _ => return false,
        }
        true
    }

    /// Rotate-and-mask family: rlwimi, rlwnm, and 64-bit MD/MDS forms (opcode 30).
    pub(crate) fn exec_rot(&mut self, opcd: u32, inst: u32) -> bool {
        let rs = ((inst >> 21) & 31) as usize;
        let ra = ((inst >> 16) & 31) as usize;
        let rb = ((inst >> 11) & 31) as usize;
        let rc = inst & 1 != 0;
        let rot32 = |v: u64, sh: u32| {
            let v = v as u32 as u64;
            (v | (v << 32)).rotate_left(sh)
        };
        let result = match opcd {
            20 => {
                // rlwimi
                let sh = (inst >> 11) & 31;
                let m = mask64(((inst >> 6) & 31) + 32, ((inst >> 1) & 31) + 32);
                (rot32(self.gpr[rs], sh) & m) | (self.gpr[ra] & !m)
            }
            23 => {
                // rlwnm
                let sh = (self.gpr[rb] & 31) as u32;
                rot32(self.gpr[rs], sh) & mask64(((inst >> 6) & 31) + 32, ((inst >> 1) & 31) + 32)
            }
            30 => {
                let raw = (inst >> 5) & 0x3F;
                let mbme = ((raw & 1) << 5) | (raw >> 1);
                let sh = ((inst >> 11) & 31) | (((inst >> 1) & 1) << 5);
                let v = self.gpr[rs];
                match (inst >> 2) & 7 {
                    0 => v.rotate_left(sh) & mask64(mbme, 63),                 // rldicl
                    1 => v.rotate_left(sh) & mask64(0, mbme),                  // rldicr
                    2 => v.rotate_left(sh) & mask64(mbme, 63 - sh),            // rldic
                    3 => {
                        // rldimi
                        let m = mask64(mbme, 63 - sh);
                        (v.rotate_left(sh) & m) | (self.gpr[ra] & !m)
                    }
                    4 => {
                        // rldcl (bit 1 = 0) / rldcr (bit 1 = 1)
                        let n = (self.gpr[rb] & 63) as u32;
                        if inst & 2 == 0 {
                            v.rotate_left(n) & mask64(mbme, 63)
                        } else {
                            v.rotate_left(n) & mask64(0, mbme)
                        }
                    }
                    _ => return false,
                }
            }
            _ => return false,
        };
        self.gpr[ra] = result;
        if rc {
            self.set_cr0(result);
        }
        true
    }

    /// X/XO-form operations under primary opcode 31. Returns false if `xo` isn't ours.
    pub(crate) fn exec_alu_x<B: Bus>(&mut self, bus: &mut B, inst: u32) -> bool {
        let xo = (inst >> 1) & 0x3FF;
        let rd = ((inst >> 21) & 31) as usize; // rt / rs
        let ra = ((inst >> 16) & 31) as usize;
        let rb = ((inst >> 11) & 31) as usize;
        let rc = inst & 1 != 0;
        let a = self.gpr[ra];
        let b = self.gpr[rb];
        let s = self.gpr[rd];

        // XO-form arithmetic: OE is bit 9 of the 10-bit xo.
        let oe = xo & 0x200 != 0;
        let mut arith = |me: &mut Self, r: u64, carry: Option<bool>, ov: bool| {
            me.gpr[rd] = r;
            if let Some(c) = carry {
                me.set_ca(c);
            }
            if oe {
                me.set_ov(ov);
            }
            if rc {
                me.set_cr0(r);
            }
        };
        match xo & 0x1FF {
            266 => { let (r, _, o) = add3(a, b, 0, self.mode32()); arith(self, r, None, o); return true; }
            10 => { let (r, c, o) = add3(a, b, 0, self.mode32()); arith(self, r, Some(c), o); return true; }
            138 => { let (r, c, o) = add3(a, b, self.ca(), self.mode32()); arith(self, r, Some(c), o); return true; }
            202 => { let (r, c, o) = add3(a, 0, self.ca(), self.mode32()); arith(self, r, Some(c), o); return true; }
            234 => { let (r, c, o) = add3(a, !0, self.ca(), self.mode32()); arith(self, r, Some(c), o); return true; }
            40 => { let (r, _, o) = add3(!a, b, 1, self.mode32()); arith(self, r, None, o); return true; }
            8 => { let (r, c, o) = add3(!a, b, 1, self.mode32()); arith(self, r, Some(c), o); return true; }
            136 => { let (r, c, o) = add3(!a, b, self.ca(), self.mode32()); arith(self, r, Some(c), o); return true; }
            200 => { let (r, c, o) = add3(!a, 0, self.ca(), self.mode32()); arith(self, r, Some(c), o); return true; }
            232 => { let (r, c, o) = add3(!a, !0, self.ca(), self.mode32()); arith(self, r, Some(c), o); return true; }
            104 => { let (r, _, o) = add3(!a, 0, 1, self.mode32()); arith(self, r, None, o); return true; }
            235 => {
                let p = (a as i32 as i64).wrapping_mul(b as i32 as i64);
                arith(self, p as u64, None, p != p as i32 as i64);
                return true;
            }
            233 => {
                let (p, o) = (a as i64).overflowing_mul(b as i64);
                arith(self, p as u64, None, o);
                return true;
            }
            75 => { arith(self, (((a as i32 as i64) * (b as i32 as i64)) >> 32) as u64, None, false); return true; }
            11 => { arith(self, ((a as u32 as u64 * b as u32 as u64) >> 32) as u64, None, false); return true; }
            73 => { arith(self, (((a as i64 as i128) * (b as i64 as i128)) >> 64) as u64, None, false); return true; }
            9 => { arith(self, (((a as u128) * (b as u128)) >> 64) as u64, None, false); return true; }
            491 => {
                let (x, y) = (a as i32, b as i32);
                let bad = y == 0 || (x == i32::MIN && y == -1);
                arith(self, if bad { 0 } else { (x / y) as i64 as u64 & 0xFFFF_FFFF }, None, bad);
                return true;
            }
            459 => {
                let (x, y) = (a as u32, b as u32);
                arith(self, if y == 0 { 0 } else { (x / y) as u64 }, None, y == 0);
                return true;
            }
            489 => {
                let (x, y) = (a as i64, b as i64);
                let bad = y == 0 || (x == i64::MIN && y == -1);
                arith(self, if bad { 0 } else { (x / y) as u64 }, None, bad);
                return true;
            }
            457 => {
                arith(self, if b == 0 { 0 } else { a / b }, None, b == 0);
                return true;
            }
            _ => {}
        }

        // Logical / shift / count / extend: destination is ra, source is rs (rd field).
        let logical = |me: &mut Self, r: u64| {
            me.gpr[ra] = r;
            if rc {
                me.set_cr0(r);
            }
        };
        match xo {
            28 => logical(self, s & b),
            60 => logical(self, s & !b),
            444 => logical(self, s | b),
            412 => logical(self, s | !b),
            124 => logical(self, !(s | b)),
            316 => logical(self, s ^ b),
            476 => logical(self, !(s & b)),
            284 => logical(self, !(s ^ b)),
            24 => { let n = b & 0x3F; logical(self, if n > 31 { 0 } else { ((s as u32) << n) as u64 }) }
            536 => { let n = b & 0x3F; logical(self, if n > 31 { 0 } else { ((s as u32) >> n) as u64 }) }
            27 => { let n = b & 0x7F; logical(self, if n > 63 { 0 } else { s << n }) }
            539 => { let n = b & 0x7F; logical(self, if n > 63 { 0 } else { s >> n }) }
            792 | 824 => {
                let n = if xo == 792 { (b & 0x3F) as u32 } else { rb as u32 };
                let v = s as i32;
                let (r, c) = if n > 31 {
                    (v >> 31, v < 0)
                } else {
                    (v >> n, v < 0 && n > 0 && (v & ((1i32 << n) - 1)) != 0)
                };
                self.set_ca(c);
                logical(self, r as i64 as u64);
            }
            794 | 826 | 827 => {
                let n = if xo == 794 { (b & 0x7F) as u32 } else { rb as u32 | ((inst & 2) << 4) };
                let v = s as i64;
                let (r, c) = if n > 63 {
                    (v >> 63, v < 0)
                } else {
                    (v >> n, v < 0 && n > 0 && (v & ((1i64 << n) - 1)) != 0)
                };
                self.set_ca(c);
                logical(self, r as u64);
            }
            26 => logical(self, (s as u32).leading_zeros() as u64),
            58 => logical(self, s.leading_zeros() as u64),
            954 => logical(self, s as i8 as i64 as u64),
            922 => logical(self, s as i16 as i64 as u64),
            986 => logical(self, s as i32 as i64 as u64),
            0 | 32 => {
                let bf = (inst >> 23) & 7;
                let l64 = (inst >> 21) & 1 != 0;
                let (lt, gt, eq) = if xo == 0 {
                    let (x, y) = if l64 { (a as i64, b as i64) } else { (a as i32 as i64, b as i32 as i64) };
                    (x < y, x > y, x == y)
                } else {
                    let (x, y) = if l64 { (a, b) } else { (a as u32 as u64, b as u32 as u64) };
                    (x < y, x > y, x == y)
                };
                self.set_cr_field(bf, lt, gt, eq);
            }
            371 => {
                // mftb: TBR 268 = time base (64-bit), 269 = upper half
                let tbr = ((inst >> 16) & 31) | (((inst >> 11) & 31) << 5);
                let tb = self.cycles / 64;
                match tbr {
                    268 => self.gpr[rd] = tb,
                    269 => self.gpr[rd] = tb >> 32,
                    _ => return false,
                }
            }
            19 => self.gpr[rd] = self.cr as u64,
            144 => {
                let fxm = (inst >> 12) & 0xFF;
                for i in 0..8 {
                    if fxm & (0x80 >> i) != 0 {
                        let shift = 28 - 4 * i;
                        self.cr = (self.cr & !(0xF << shift)) | (((s as u32) >> shift) & 0xF) << shift;
                    }
                }
            }
            83 => self.gpr[rd] = self.msr,
            146 | 178 => self.msr = s,
            // dcbz zeroes a 32-byte block; the rt=1 variant (dcbz128) zeroes a 128-byte cache line.
            1014 => {
                let ea = (if ra == 0 { 0 } else { a }).wrapping_add(b) as u32;
                let len = if rd == 1 { 128 } else { 32 };
                let start = ea & !(len - 1);
                for i in 0..len {
                    bus.write8(start + i, 0);
                }
            }
            // Barriers and cache hints: no-ops in a single-threaded interpreter.
            598 | 854 | 86 | 278 | 246 | 54 | 982 => {}
            // Load-reserve / store-conditional (single reservation, single core for now)
            20 | 84 => {
                let ea = (if ra == 0 { 0 } else { a }).wrapping_add(b) as u32;
                self.reservation = Some(ea);
                self.gpr[rd] = if xo == 20 { bus.read32(ea) as u64 } else { bus.read64(ea) };
            }
            150 | 214 => {
                let ea = (if ra == 0 { 0 } else { a }).wrapping_add(b) as u32;
                let ok = self.reservation == Some(ea);
                if ok {
                    if xo == 150 { bus.write32(ea, s as u32) } else { bus.write64(ea, s) }
                }
                self.reservation = None;
                self.set_cr_field(0, false, false, ok);
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

    fn run1(c: &mut Cpu, m: &mut FlatMemory, inst: u32) {
        m.write32(c.pc as u32, inst);
        c.step(m).unwrap();
    }

    fn x(op: u32, rt: u32, ra: u32, rb: u32, xo: u32) -> u32 {
        (op << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1)
    }

    #[test]
    fn carry_chain_addc_adde() {
        let mut m = FlatMemory::new(64);
        let mut c = Cpu::new();
        c.gpr[3] = u64::MAX;
        c.gpr[4] = 1;
        run1(&mut c, &mut m, x(31, 5, 3, 4, 10)); // addc r5,r3,r4
        assert_eq!(c.gpr[5], 0);
        assert_eq!(c.xer & (1 << 29), 1 << 29);
        c.gpr[6] = 7;
        c.gpr[7] = 0;
        run1(&mut c, &mut m, x(31, 8, 6, 7, 138)); // adde r8,r6,r7
        assert_eq!(c.gpr[8], 8);
    }

    #[test]
    fn subf_and_cmpw() {
        let mut m = FlatMemory::new(64);
        let mut c = Cpu::new();
        c.gpr[3] = 3;
        c.gpr[4] = 10;
        run1(&mut c, &mut m, x(31, 5, 3, 4, 40)); // subf r5,r3,r4 = r4-r3
        assert_eq!(c.gpr[5], 7);
        run1(&mut c, &mut m, x(31, 0, 3, 4, 0)); // cmpw r3,r4
        assert_eq!(c.cr >> 28, 0b1000); // LT
    }

    #[test]
    fn sraw_sets_carry() {
        let mut m = FlatMemory::new(64);
        let mut c = Cpu::new();
        c.gpr[3] = 0xFFFF_FFFF_FFFF_FFFD; // -3
        run1(&mut c, &mut m, x(31, 3, 4, 1, 824)); // srawi r4,r3,1
        assert_eq!(c.gpr[4] as i64, -2);
        assert_eq!(c.xer & (1 << 29), 1 << 29);
    }

    #[test]
    fn rldicl_and_rlwimi() {
        let mut m = FlatMemory::new(64);
        let mut c = Cpu::new();
        c.gpr[3] = 0x1234_5678_9ABC_DEF0;
        // clrldi r4,r3,32 == rldicl r4,r3,0,32: mb=32 -> raw field (mb&31)<<1 | mb>>5 = 0|1 = 1 at bit 5
        run1(&mut c, &mut m, (30 << 26) | (3 << 21) | (4 << 16) | (1 << 5));
        assert_eq!(c.gpr[4], 0x9ABC_DEF0);
        // rlwimi r5,r3,0,28,31 : insert low nibble of r3 into r5
        c.gpr[5] = 0xFFFF_FFFF_FFFF_FF00;
        run1(&mut c, &mut m, (20 << 26) | (3 << 21) | (5 << 16) | (28 << 6) | (31 << 1));
        assert_eq!(c.gpr[5] & 0xFF, 0x00);
        assert_eq!(c.gpr[5] >> 8, 0x00FF_FFFF_FFFF_FFFF);
    }

    #[test]
    fn dcbz_and_dcbz128_zero_aligned_lines() {
        let mut m = FlatMemory::new(0x400);
        for i in 0..0x400u32 {
            m.write8(i, 0xFF);
        }
        let mut c = Cpu::new();
        c.gpr[4] = 0x1A5;
        run1(&mut c, &mut m, x(31, 0, 0, 4, 1014)); // dcbz 0,r4 -> 0x1A0..0x1C0
        assert_eq!(m.read8(0x19F), 0xFF);
        assert_eq!(m.read8(0x1A0), 0);
        assert_eq!(m.read8(0x1BF), 0);
        assert_eq!(m.read8(0x1C0), 0xFF);
        run1(&mut c, &mut m, x(31, 1, 0, 4, 1014)); // dcbz128 -> 0x180..0x200
        assert_eq!(m.read8(0x17F), 0xFF);
        assert_eq!(m.read8(0x180), 0);
        assert_eq!(m.read8(0x1FF), 0);
        assert_eq!(m.read8(0x200), 0xFF);
    }

    #[test]
    fn mftb_advances_with_cycles() {
        let mut m = FlatMemory::new(64);
        let mut c = Cpu::new();
        c.cycles = 64 * 1000;
        // mftb r5 : spr 268 -> field encoding (268 & 31) << 16 | (268 >> 5) << 11
        let inst = (31 << 26) | (5 << 21) | ((268 & 31) << 16) | ((268 >> 5) << 11) | (371 << 1);
        run1(&mut c, &mut m, inst);
        assert_eq!(c.gpr[5], 1000);
    }

    #[test]
    fn lwarx_stwcx() {
        let mut m = FlatMemory::new(0x400);
        let mut c = Cpu::new();
        c.gpr[4] = 0x100;
        m.write32(0x100, 41);
        run1(&mut c, &mut m, x(31, 3, 0, 4, 20)); // lwarx r3,0,r4
        assert_eq!(c.gpr[3], 41);
        c.gpr[3] = 42;
        run1(&mut c, &mut m, x(31, 3, 0, 4, 150) | 1); // stwcx. r3,0,r4
        assert_eq!(m.read32(0x100), 42);
        assert_eq!((c.cr >> 28) & 2, 2); // EQ set
    }
}
