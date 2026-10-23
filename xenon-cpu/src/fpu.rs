//! Floating-point unit. FPRs hold raw 64-bit patterns (`u64`) so loads/stores and moves
//! preserve NaN payloads exactly; arithmetic goes through `f64`.
//! Not modelled yet: FPSCR rounding mode, exception flags, FPRF.

use crate::memory::Bus;
use crate::Cpu;

fn to_i32_sat(x: f64, round: bool) -> i32 {
    if x.is_nan() {
        return i32::MIN;
    }
    let r = if round { x.round_ties_even() } else { x.trunc() };
    if r >= i32::MAX as f64 {
        i32::MAX
    } else if r <= i32::MIN as f64 {
        i32::MIN
    } else {
        r as i32
    }
}

fn to_i64_sat(x: f64, round: bool) -> i64 {
    if x.is_nan() {
        return i64::MIN;
    }
    let r = if round { x.round_ties_even() } else { x.trunc() };
    if r >= i64::MAX as f64 {
        i64::MAX
    } else if r <= i64::MIN as f64 {
        i64::MIN
    } else {
        r as i64
    }
}

impl Cpu {
    fn fr(&self, r: usize) -> f64 {
        f64::from_bits(self.fpr[r])
    }
    fn set_fr(&mut self, r: usize, v: f64) {
        self.fpr[r] = v.to_bits();
    }

    /// Rc=1 on a float op copies FPSCR[0:3] into CR1.
    fn fp_record(&mut self, rc: bool) {
        if rc {
            self.cr = (self.cr & !(0xF << 24)) | (((self.fpscr >> 28) & 0xF) << 24);
        }
    }

    /// D-form float loads/stores, opcodes 48..=55.
    #[inline(never)]
    pub(crate) fn exec_fpu_d<B: Bus>(&mut self, bus: &mut B, inst: u32) -> bool {
        let opcd = inst >> 26;
        let frt = ((inst >> 21) & 31) as usize;
        let ra = ((inst >> 16) & 31) as usize;
        let d = (inst & 0xFFFF) as i16 as i64 as u64;
        let ea = (if ra == 0 { 0 } else { self.gpr[ra] }).wrapping_add(d);
        self.fp_mem(bus, opcd, frt, ra, ea)
    }

    /// Shared by the D-form opcodes and the indexed X-forms (mapped onto the D-form opcode).
    fn fp_mem<B: Bus>(&mut self, bus: &mut B, opcd: u32, frt: usize, ra: usize, ea: u64) -> bool {
        let a = ea as u32;
        match opcd {
            48 | 49 => self.set_fr(frt, f32::from_bits(bus.read32(a)) as f64),
            50 | 51 => self.fpr[frt] = bus.read64(a),
            52 | 53 => bus.write32(a, (self.fr(frt) as f32).to_bits()),
            54 | 55 => bus.write64(a, self.fpr[frt]),
            _ => return false,
        }
        if opcd & 1 == 1 {
            self.gpr[ra] = ea; // update forms
        }
        true
    }

    /// Opcode 31 float memory ops: lfsx/lfsux/lfdx/lfdux/stfsx/stfsux/stfdx/stfdux/stfiwx.
    #[inline(never)]
    pub(crate) fn exec_fpu_x<B: Bus>(&mut self, bus: &mut B, inst: u32) -> bool {
        let xo = (inst >> 1) & 0x3FF;
        let frt = ((inst >> 21) & 31) as usize;
        let ra = ((inst >> 16) & 31) as usize;
        let rb = ((inst >> 11) & 31) as usize;
        let ea = (if ra == 0 { 0 } else { self.gpr[ra] }).wrapping_add(self.gpr[rb]);
        let opcd = match xo {
            535 => 48,
            567 => 49,
            599 => 50,
            631 => 51,
            663 => 52,
            695 => 53,
            727 => 54,
            759 => 55,
            983 => {
                // stfiwx: store the low 32 bits of the FPR verbatim
                bus.write32(ea as u32, self.fpr[frt] as u32);
                return true;
            }
            _ => return false,
        };
        self.fp_mem(bus, opcd, frt, ra, ea)
    }

    /// Opcodes 59 (single) and 63 (double): A-form arithmetic and X-form moves/compares/conversions.
    #[inline(never)]
    pub(crate) fn exec_fpu_arith(&mut self, inst: u32) -> bool {
        let single = inst >> 26 == 59;
        let frt = ((inst >> 21) & 31) as usize;
        let fra = ((inst >> 16) & 31) as usize;
        let frb = ((inst >> 11) & 31) as usize;
        let frc = ((inst >> 6) & 31) as usize;
        let rc = inst & 1 != 0;
        let (a, b, c) = (self.fr(fra), self.fr(frb), self.fr(frc));
        let round = |v: f64| if single { v as f32 as f64 } else { v };

        // A-form (5-bit xo)
        let r = match (inst >> 1) & 0x1F {
            18 => Some(round(a / b)),
            20 => Some(round(a - b)),
            21 => Some(round(a + b)),
            22 => Some(round(b.sqrt())),
            23 if !single => Some(if a >= 0.0 { c } else { b }), // fsel
            24 if single => Some(round(1.0 / b)),                // fres
            25 => Some(round(a * c)),
            26 if !single => Some(1.0 / b.sqrt()), // frsqrte
            28 => Some(round(a.mul_add(c, -b))),   // fmsub
            29 => Some(round(a.mul_add(c, b))),    // fmadd
            30 => Some(round(-a.mul_add(c, -b))),  // fnmsub
            31 => Some(round(-a.mul_add(c, b))),   // fnmadd
            _ => None,
        };
        if let Some(v) = r {
            self.set_fr(frt, v);
            self.fp_record(rc);
            return true;
        }
        if single {
            return false;
        }

        // X-form (10-bit xo), opcode 63 only
        match (inst >> 1) & 0x3FF {
            0 | 32 => {
                // fcmpu / fcmpo
                let bf = (inst >> 23) & 7;
                let (lt, gt, eq, un) = (a < b, a > b, a == b, a.is_nan() || b.is_nan());
                let v = ((lt as u32) << 3) | ((gt as u32) << 2) | ((eq as u32) << 1) | un as u32;
                let sh = 28 - 4 * bf;
                self.cr = (self.cr & !(0xF << sh)) | (v << sh);
            }
            12 => self.set_fr(frt, b as f32 as f64), // frsp
            14 | 15 => {
                // fctiw / fctiwz: result in the low word
                let v = to_i32_sat(b, (inst >> 1) & 0x3FF == 14) as u32 as u64;
                self.fpr[frt] = 0xFFF8_0000_0000_0000 | v;
            }
            814 | 815 => self.fpr[frt] = to_i64_sat(b, (inst >> 1) & 0x3FF == 814) as u64,
            846 => self.set_fr(frt, self.fpr[frb] as i64 as f64), // fcfid
            72 => self.fpr[frt] = self.fpr[frb],                  // fmr
            40 => self.fpr[frt] = self.fpr[frb] ^ (1 << 63),      // fneg
            264 => self.fpr[frt] = self.fpr[frb] & !(1 << 63),    // fabs
            136 => self.fpr[frt] = self.fpr[frb] | (1 << 63),     // fnabs
            583 => self.fpr[frt] = self.fpscr as u64,             // mffs
            711 => {
                // mtfsf FM, frb
                let fm = (inst >> 17) & 0xFF;
                let v = self.fpr[frb] as u32;
                for i in 0..8 {
                    if fm & (0x80 >> i) != 0 {
                        let sh = 28 - 4 * i;
                        self.fpscr = (self.fpscr & !(0xF << sh)) | (((v >> sh) & 0xF) << sh);
                    }
                }
            }
            134 => {
                // mtfsfi BF, U
                let bf = (inst >> 23) & 7;
                let u = (inst >> 12) & 0xF;
                let sh = 28 - 4 * bf;
                self.fpscr = (self.fpscr & !(0xF << sh)) | (u << sh);
            }
            70 | 38 => {
                // mtfsb0 / mtfsb1
                let bit = 31 - ((inst >> 21) & 31);
                if (inst >> 1) & 0x3FF == 38 {
                    self.fpscr |= 1 << bit;
                } else {
                    self.fpscr &= !(1 << bit);
                }
            }
            _ => return false,
        }
        self.fp_record(rc);
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::{Bus, FlatMemory};
    use crate::Cpu;

    fn a(op: u32, frt: u32, fra: u32, frb: u32, frc: u32, xo: u32) -> u32 {
        (op << 26) | (frt << 21) | (fra << 16) | (frb << 11) | (frc << 6) | (xo << 1)
    }
    fn x63(frt: u32, fra: u32, frb: u32, xo: u32) -> u32 {
        (63 << 26) | (frt << 21) | (fra << 16) | (frb << 11) | (xo << 1)
    }

    #[test]
    fn single_load_store_and_arith() {
        let mut m = FlatMemory::new(0x100);
        let mut c = Cpu::new();
        m.write32(0x40, 1.5f32.to_bits());
        c.gpr[3] = 0x40;
        m.write32(0, (48 << 26) | (1 << 21) | (3 << 16)); // lfs f1,0(r3)
        m.write32(4, a(63, 2, 1, 1, 0, 21)); // fadd f2,f1,f1
        m.write32(8, (52 << 26) | (2 << 21) | (3 << 16) | 4); // stfs f2,4(r3)
        for _ in 0..3 {
            c.step(&mut m).unwrap();
        }
        assert_eq!(f32::from_bits(m.read32(0x44)), 3.0);
    }

    #[test]
    fn fmadd_fsel_fcmpu() {
        let mut m = FlatMemory::new(0x100);
        let mut c = Cpu::new();
        c.fpr[1] = 2.0f64.to_bits();
        c.fpr[2] = 3.0f64.to_bits();
        c.fpr[3] = 4.0f64.to_bits();
        m.write32(0, a(63, 4, 1, 3, 2, 29)); // fmadd f4 = f1*f2 + f3 = 10
        m.write32(4, x63(0, 1, 2, 0) | (0 << 23)); // fcmpu cr0,f1,f2 : 2<3 -> LT
        c.step(&mut m).unwrap();
        c.step(&mut m).unwrap();
        assert_eq!(f64::from_bits(c.fpr[4]), 10.0);
        assert_eq!(c.cr >> 28, 0b1000);
        c.fpr[5] = (-1.0f64).to_bits();
        m.write32(8, a(63, 6, 5, 3, 2, 23)); // fsel f6 = (f5 >= 0) ? f2 : f3 -> f3 (4.0)
        c.step(&mut m).unwrap();
        assert_eq!(f64::from_bits(c.fpr[6]), 4.0);
    }

    #[test]
    fn convert_and_stfiwx() {
        let mut m = FlatMemory::new(0x100);
        let mut c = Cpu::new();
        c.fpr[1] = (-7.9f64).to_bits();
        c.gpr[3] = 0x20;
        m.write32(0, x63(2, 0, 1, 15)); // fctiwz f2,f1 -> -7
        m.write32(4, (31 << 26) | (2 << 21) | (3 << 11) | (983 << 1)); // stfiwx f2,0,r3
        c.step(&mut m).unwrap();
        c.step(&mut m).unwrap();
        assert_eq!(m.read32(0x20) as i32, -7);
    }
}
