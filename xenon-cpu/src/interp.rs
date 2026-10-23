//! Interpreter. `step` executes one instruction or returns a `Trap` describing
//! exactly what we don't support yet â€” that is the driver of the crash/fix loop.

use crate::memory::Bus;
use crate::Cpu;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Trap {
    /// Opcode (or form) not implemented yet.
    Unimplemented { pc: u64, inst: u32 },
    /// `sc` executed. `pc` has already advanced past it; the host services the call
    /// (convention for HLE import thunks: r11 = (library << 16) | ordinal).
    Syscall,
    /// A tw/td/twi/tdi condition fired (assert/breakpoint in the guest). `pc` is the trap instruction.
    ProgramTrap { pc: u64, inst: u32 },
}

/// TO field: 16 = signed <, 8 = signed >, 4 = equal, 2 = unsigned <, 1 = unsigned >.
fn trap_fires(to: u32, a: u64, b: u64, word: bool) -> bool {
    let (sa, sb, ua, ub) = if word {
        (a as i32 as i64, b as i32 as i64, a as u32 as u64, b as u32 as u64)
    } else {
        (a as i64, b as i64, a, b)
    };
    (to & 16 != 0 && sa < sb)
        || (to & 8 != 0 && sa > sb)
        || (to & 4 != 0 && ua == ub)
        || (to & 2 != 0 && ua < ub)
        || (to & 1 != 0 && ua > ub)
}

fn mask64(mb: u32, me: u32) -> u64 {
    let a = !0u64 >> mb;
    let b = !0u64 << (63 - me);
    if mb <= me { a & b } else { a | b }
}

#[derive(Clone, Copy)]
enum MemOp {
    Load { size: u32, signed: bool, update: bool },
    Store { size: u32, update: bool },
}

fn decode_d_mem(opcd: u32) -> Option<MemOp> {
    use MemOp::*;
    Some(match opcd {
        32 => Load { size: 4, signed: false, update: false },
        33 => Load { size: 4, signed: false, update: true },
        34 => Load { size: 1, signed: false, update: false },
        35 => Load { size: 1, signed: false, update: true },
        36 => Store { size: 4, update: false },
        37 => Store { size: 4, update: true },
        38 => Store { size: 1, update: false },
        39 => Store { size: 1, update: true },
        40 => Load { size: 2, signed: false, update: false },
        41 => Load { size: 2, signed: false, update: true },
        42 => Load { size: 2, signed: true, update: false },
        43 => Load { size: 2, signed: true, update: true },
        44 => Store { size: 2, update: false },
        45 => Store { size: 2, update: true },
        _ => return None,
    })
}

/// X-form (opcode 31) indexed loads/stores. Returns (op, byte_reversed).
fn decode_x_mem(xo: u32) -> Option<(MemOp, bool)> {
    use MemOp::*;
    let ld = |size, signed, update| Load { size, signed, update };
    Some(match xo {
        23 => (ld(4, false, false), false),   // lwzx
        55 => (ld(4, false, true), false),    // lwzux
        87 => (ld(1, false, false), false),   // lbzx
        119 => (ld(1, false, true), false),   // lbzux
        279 => (ld(2, false, false), false),  // lhzx
        311 => (ld(2, false, true), false),   // lhzux
        343 => (ld(2, true, false), false),   // lhax
        375 => (ld(2, true, true), false),    // lhaux
        341 => (ld(4, true, false), false),   // lwax
        373 => (ld(4, true, true), false),    // lwaux
        21 => (ld(8, false, false), false),   // ldx
        53 => (ld(8, false, true), false),    // ldux
        534 => (ld(4, false, false), true),   // lwbrx
        790 => (ld(2, false, false), true),   // lhbrx
        151 => (Store { size: 4, update: false }, false), // stwx
        183 => (Store { size: 4, update: true }, false),  // stwux
        215 => (Store { size: 1, update: false }, false), // stbx
        247 => (Store { size: 1, update: true }, false),  // stbux
        407 => (Store { size: 2, update: false }, false), // sthx
        439 => (Store { size: 2, update: true }, false),  // sthux
        149 => (Store { size: 8, update: false }, false), // stdx
        181 => (Store { size: 8, update: true }, false),  // stdux
        662 => (Store { size: 4, update: false }, true),  // stwbrx
        918 => (Store { size: 2, update: false }, true),  // sthbrx
        _ => return None,
    })
}

impl Cpu {
    fn exec_mem<B: Bus>(&mut self, bus: &mut B, op: MemOp, ea: u64, rt: usize, ra: usize, byterev: bool) {
        let a = ea as u32;
        match op {
            MemOp::Load { size, signed, update } => {
                let mut v = match size {
                    1 => bus.read8(a) as u64,
                    2 => bus.read16(a) as u64,
                    4 => bus.read32(a) as u64,
                    _ => bus.read64(a),
                };
                if byterev {
                    v = match size {
                        2 => (v as u16).swap_bytes() as u64,
                        4 => (v as u32).swap_bytes() as u64,
                        _ => v.swap_bytes(),
                    };
                }
                if signed {
                    v = match size {
                        1 => v as i8 as i64 as u64,
                        2 => v as i16 as i64 as u64,
                        _ => v as i32 as i64 as u64,
                    };
                }
                self.gpr[rt] = v;
                if update {
                    self.gpr[ra] = ea;
                }
            }
            MemOp::Store { size, update } => {
                let v = self.gpr[rt];
                match size {
                    1 => bus.write8(a, v as u8),
                    2 => bus.write16(a, if byterev { (v as u16).swap_bytes() } else { v as u16 }),
                    4 => bus.write32(a, if byterev { (v as u32).swap_bytes() } else { v as u32 }),
                    _ => bus.write64(a, v),
                }
                if update {
                    self.gpr[ra] = ea;
                }
            }
        }
    }

    pub(crate) fn set_cr_field(&mut self, n: u32, lt: bool, gt: bool, eq: bool) {
        let so = (self.xer >> 31) & 1 != 0;
        let v = ((lt as u32) << 3) | ((gt as u32) << 2) | ((eq as u32) << 1) | so as u32;
        let shift = 28 - 4 * n;
        self.cr = (self.cr & !(0xF << shift)) | (v << shift);
    }

    pub(crate) fn set_cr0(&mut self, val: u64) {
        let v = val as i64;
        self.set_cr_field(0, v < 0, v > 0, v == 0);
    }

    fn cr_bit(&self, bi: u32) -> bool {
        (self.cr >> (31 - bi)) & 1 != 0
    }

    fn ea_base(&self, ra: usize) -> u64 {
        if ra == 0 { 0 } else { self.gpr[ra] }
    }

    fn do_branch_cond(&mut self, bo: u32, bi: u32) -> bool {
        if bo & 0b00100 == 0 {
            self.ctr = self.ctr.wrapping_sub(1);
        }
        let ctr_ok = bo & 0b00100 != 0 || ((self.ctr != 0) ^ (bo & 0b00010 != 0));
        let cond_ok = bo & 0b10000 != 0 || (self.cr_bit(bi) == (bo & 0b01000 != 0));
        ctr_ok && cond_ok
    }

    pub fn step<B: Bus>(&mut self, bus: &mut B) -> Result<(), Trap> {
        let pc = self.pc;
        self.cycles += self.cpi;
        let inst = bus.read32(pc as u32);
        self.exec(bus, pc, inst)
    }

    /// Execute one already-fetched instruction located at `pc` (also used by the JIT for anything it
    /// doesn't translate). Updates `self.pc` to the successor and the taken-branch trace.
    pub fn exec<B: Bus>(&mut self, bus: &mut B, pc: u64, inst: u32) -> Result<(), Trap> {

        let opcd = inst >> 26;
        let rd = ((inst >> 21) & 31) as usize; // also rs / rt
        let ra = ((inst >> 16) & 31) as usize;
        let rb = ((inst >> 11) & 31) as usize;
        let simm = (inst & 0xFFFF) as i16 as i64;
        let uimm = (inst & 0xFFFF) as u64;
        let rc = inst & 1 != 0;
        let mut next = pc.wrapping_add(4);

        match opcd {
            // cmpli / cmpi
            10 | 11 => {
                let bf = (inst >> 23) & 7;
                let l64 = (inst >> 21) & 1 != 0;
                if opcd == 11 {
                    let a = if l64 { self.gpr[ra] as i64 } else { self.gpr[ra] as i32 as i64 };
                    self.set_cr_field(bf, a < simm, a > simm, a == simm);
                } else {
                    let a = if l64 { self.gpr[ra] } else { self.gpr[ra] as u32 as u64 };
                    self.set_cr_field(bf, a < uimm, a > uimm, a == uimm);
                }
            }
            // addi / addis
            14 => self.gpr[rd] = self.ea_base(ra).wrapping_add(simm as u64),
            15 => self.gpr[rd] = self.ea_base(ra).wrapping_add((simm << 16) as u64),
            // bc
            16 => {
                let bo = (inst >> 21) & 31;
                let bi = (inst >> 16) & 31;
                let aa = inst & 2 != 0;
                let lk = inst & 1 != 0;
                let bd = (inst & 0xFFFC) as i16 as i64 as u64;
                let take = self.do_branch_cond(bo, bi);
                if lk {
                    self.lr = pc.wrapping_add(4);
                }
                if take {
                    next = if aa { bd } else { pc.wrapping_add(bd) };
                }
            }
            // b / bl / ba / bla
            18 => {
                let li = (((inst & 0x03FF_FFFC) << 6) as i32 >> 6) as i64 as u64;
                if inst & 1 != 0 {
                    self.lr = pc.wrapping_add(4);
                }
                next = if inst & 2 != 0 { li } else { pc.wrapping_add(li) };
            }
            // XL-form: bclr / bcctr
            19 => {
                let xo = (inst >> 1) & 0x3FF;
                let bo = (inst >> 21) & 31;
                let bi = (inst >> 16) & 31;
                match xo {
                    16 | 528 => {
                        let target = if xo == 16 { self.lr } else { self.ctr } & !3;
                        let take = if xo == 16 {
                            self.do_branch_cond(bo, bi)
                        } else {
                            // bcctr never decrements CTR
                            bo & 0b10000 != 0 || (self.cr_bit(bi) == (bo & 0b01000 != 0))
                        };
                        if inst & 1 != 0 {
                            self.lr = pc.wrapping_add(4);
                        }
                        if take {
                            next = target;
                        }
                    }
                    _ => return Err(Trap::Unimplemented { pc, inst }),
                }
            }
            6 => {
                if !self.exec_vmx128_op6(inst) {
                    return Err(Trap::Unimplemented { pc, inst });
                }
            }
            5 => {
                if !self.exec_vmx128_vperm(inst) && !self.exec_vmx128_arith(inst) {
                    return Err(Trap::Unimplemented { pc, inst });
                }
            }
            4 => {
                if !self.exec_vmx128_mem(bus, inst) && !self.exec_vmx(inst) {
                    return Err(Trap::Unimplemented { pc, inst });
                }
            }
            48..=55 => {
                if !self.exec_fpu_d(bus, inst) {
                    return Err(Trap::Unimplemented { pc, inst });
                }
            }
            59 | 63 => {
                if !self.exec_fpu_arith(inst) {
                    return Err(Trap::Unimplemented { pc, inst });
                }
            }
            // tdi / twi
            2 | 3 => {
                if trap_fires(rd as u32, self.gpr[ra], simm as u64, opcd == 3) {
                    return Err(Trap::ProgramTrap { pc, inst });
                }
            }
            7 | 8 | 12 | 13 | 26 | 27 | 28 | 29 => {
                if !self.exec_alu_d(opcd, inst) {
                    return Err(Trap::Unimplemented { pc, inst });
                }
            }
            20 | 23 | 30 => {
                if !self.exec_rot(opcd, inst) {
                    return Err(Trap::Unimplemented { pc, inst });
                }
            }
            // sc
            17 if inst & 2 != 0 => {
                self.pc = next;
                return Err(Trap::Syscall);
            }
            // rlwinm
            21 => {
                let sh = (inst >> 11) & 31;
                let mb = (inst >> 6) & 31;
                let me = (inst >> 1) & 31;
                let v = self.gpr[rd] as u32 as u64;
                let r = (v | (v << 32)).rotate_left(sh);
                self.gpr[ra] = r & mask64(mb + 32, me + 32);
                if rc {
                    self.set_cr0(self.gpr[ra]);
                }
            }
            // ori / oris
            24 => self.gpr[ra] = self.gpr[rd] | uimm,
            25 => self.gpr[ra] = self.gpr[rd] | (uimm << 16),
            // X-form integer / SPR moves
            31 => {
                let xo = (inst >> 1) & 0x3FF;
                match xo {
                    339 | 467 => {
                        let spr = ((inst >> 16) & 31) | (((inst >> 11) & 31) << 5);
                        let slot = match spr {
                            1 => &mut self.xer,
                            8 => &mut self.lr,
                            9 => &mut self.ctr,
                            _ => return Err(Trap::Unimplemented { pc, inst }),
                        };
                        if xo == 339 {
                            self.gpr[rd] = *slot;
                        } else {
                            *slot = self.gpr[rd];
                        }
                    }
                    _ if decode_x_mem(xo).is_some() => {
                        let (op, byterev) = decode_x_mem(xo).unwrap();
                        let ea = self.ea_base(ra).wrapping_add(self.gpr[rb]);
                        self.exec_mem(bus, op, ea, rd, ra, byterev);
                    }
                    // tw / td
                    4 | 68 => {
                        if trap_fires(rd as u32, self.gpr[ra], self.gpr[rb], xo == 4) {
                            return Err(Trap::ProgramTrap { pc, inst });
                        }
                    }
                    _ => {
                        if !self.exec_alu_x(bus, inst)
                            && !self.exec_vmx_x(bus, inst)
                            && !self.exec_fpu_x(bus, inst)
                        {
                            return Err(Trap::Unimplemented { pc, inst });
                        }
                    }
                }
            }
            // lmw / stmw
            46 | 47 => {
                let mut ea = self.ea_base(ra).wrapping_add(simm as u64) as u32;
                for r in rd..32 {
                    if opcd == 46 {
                        self.gpr[r] = bus.read32(ea) as u64;
                    } else {
                        bus.write32(ea, self.gpr[r] as u32);
                    }
                    ea = ea.wrapping_add(4);
                }
            }
            // D-form / DS-form loads and stores (ld/std are DS-form: 58/62)
            32..=45 | 58 | 62 => {
                let (op, d) = match opcd {
                    58 => match inst & 3 {
                        0 => (MemOp::Load { size: 8, signed: false, update: false }, simm & !3),
                        1 => (MemOp::Load { size: 8, signed: false, update: true }, simm & !3),
                        2 => (MemOp::Load { size: 4, signed: true, update: false }, simm & !3),
                        _ => return Err(Trap::Unimplemented { pc, inst }),
                    },
                    62 => match inst & 3 {
                        0 => (MemOp::Store { size: 8, update: false }, simm & !3),
                        1 => (MemOp::Store { size: 8, update: true }, simm & !3),
                        _ => return Err(Trap::Unimplemented { pc, inst }),
                    },
                    _ => match decode_d_mem(opcd) {
                        Some(op) => (op, simm),
                        None => return Err(Trap::Unimplemented { pc, inst }),
                    },
                };
                let ea = self.ea_base(ra).wrapping_add(d as u64);
                self.exec_mem(bus, op, ea, rd, ra, false);
            }
            _ => return Err(Trap::Unimplemented { pc, inst }),
        }

        // Instruction addresses are 32-bit on this machine; registers may hold sign-extended values.
        let next = next & 0xFFFF_FFFF;
        if next != (pc.wrapping_add(4) & 0xFFFF_FFFF) {
            self.trace[self.trace_pos & 63] = (pc as u32, next as u32);
            self.trace_pos = self.trace_pos.wrapping_add(1);
        }
        self.pc = next;
        Ok(())
    }

    /// Addresses of the most recent `last` instructions ending at the current pc, oldest first,
    /// rebuilt from the taken-branch ring (may be shorter if the ring runs out).
    pub fn recent_pcs(&self, last: usize) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        let mut seg_end = self.pc as u32;
        let have = self.trace_pos.min(64);
        for k in 0..=have {
            let (start, next_end) = if k < have {
                let (from, to) = self.trace[(self.trace_pos - 1 - k) & 63];
                (to, from)
            } else {
                (seg_end.saturating_sub(4 * (last as u32)), 0)
            };
            // segment start..=seg_end, newest-first accumulation
            let mut a = seg_end;
            loop {
                out.push(a);
                if out.len() >= last || a <= start {
                    break;
                }
                a -= 4;
            }
            if out.len() >= last || k >= have {
                break;
            }
            seg_end = next_end;
        }
        out.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::FlatMemory;

    fn run(words: &[u32], steps: usize) -> (Cpu, FlatMemory) {
        let mut m = FlatMemory::new(0x1000);
        for (i, w) in words.iter().enumerate() {
            m.write32(i as u32 * 4, *w);
        }
        let mut c = Cpu::new();
        for _ in 0..steps {
            c.step(&mut m).unwrap();
        }
        (c, m)
    }

    #[test]
    fn addi_addis_ori() {
        // li r3,-1 ; lis r4,0x1234 ; ori r4,r4,0x5678
        let (c, _) = run(&[0x3860FFFF, 0x3C801234, 0x60845678], 3);
        assert_eq!(c.gpr[3], u64::MAX);
        assert_eq!(c.gpr[4], 0x12345678);
    }

    #[test]
    fn bl_blr_and_store() {
        // 0: bl +8 ; 4: b +0x10(end) ; 8: li r5,7 ; c: blr
        let (c, _) = run(&[0x48000009, 0x48000010, 0x38A00007, 0x4E800020], 3);
        assert_eq!(c.gpr[5], 7);
        assert_eq!(c.lr, 4);
        assert_eq!(c.pc, 4);
    }

    #[test]
    fn cmpwi_beq() {
        // li r3,5 ; cmpwi r3,5 ; beq +8
        let (c, _) = run(&[0x38600005, 0x2C030005, 0x41820008], 3);
        assert_eq!(c.pc, 8 + 8);
    }

    #[test]
    fn twi_fires_only_when_condition_true() {
        let mut m = FlatMemory::new(16);
        m.write32(0, 0x0CC30000); // twi 6,r3,0 : trap if r3 == 0
        let mut c = Cpu::new();
        c.gpr[3] = 1;
        c.step(&mut m).unwrap();
        c.pc = 0;
        c.gpr[3] = 0;
        assert_eq!(c.step(&mut m), Err(Trap::ProgramTrap { pc: 0, inst: 0x0CC30000 }));
    }

    #[test]
    fn unknown_traps() {
        let mut m = FlatMemory::new(16);
        m.write32(0, 0x00000000);
        let mut c = Cpu::new();
        assert_eq!(c.step(&mut m), Err(Trap::Unimplemented { pc: 0, inst: 0 }));
    }

    #[test]
    fn std_ld_roundtrip_and_stwu() {
        let mut m = FlatMemory::new(0x1000);
        // std r3,8(r1) ; ld r4,8(r1) ; stwu r1,-16(r1)
        for (i, w) in [0xF8610008u32, 0xE8810008, 0x9421FFF0].iter().enumerate() {
            m.write32(i as u32 * 4, *w);
        }
        let mut c = Cpu::new();
        c.gpr[1] = 0x800;
        c.gpr[3] = 0x1122334455667788;
        for _ in 0..3 {
            c.step(&mut m).unwrap();
        }
        assert_eq!(c.gpr[4], 0x1122334455667788);
        assert_eq!(m.read64(0x808), 0x1122334455667788);
        assert_eq!(c.gpr[1], 0x7F0);
        assert_eq!(m.read32(0x7F0), 0x800);
    }

    #[test]
    fn byte_reversed_and_algebraic() {
        let mut m = FlatMemory::new(0x1000);
        // stwbrx r3,r0,r4 ; lwbrx r5,r0,r4
        m.write32(0, 0x7C60252C);
        m.write32(4, 0x7CA0242C);
        let mut c = Cpu::new();
        c.gpr[3] = 0x11223344;
        c.gpr[4] = 0x200;
        c.step(&mut m).unwrap();
        assert_eq!(m.read32(0x200), 0x44332211);
        c.step(&mut m).unwrap();
        assert_eq!(c.gpr[5], 0x11223344);
        // lha sign-extends: lha r6,0(r4) with 0x8000 stored
        m.write16(0x200, 0x8000);
        m.write32(8, 0xA8C40000);
        c.step(&mut m).unwrap();
        assert_eq!(c.gpr[6], 0xFFFF_FFFF_FFFF_8000);
    }

    #[test]
    fn rlwinm_slwi() {
        // li r3,1 ; slwi r4,r3,4 == rlwinm r4,r3,4,0,27
        let (c, _) = run(&[0x38600001, 0x54642036], 2);
        assert_eq!(c.gpr[4], 16);
    }

    /// Raw interpreter speed on a tight integer loop. Run with:
    /// cargo test --release -p xenon-cpu bench_step_rate -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_step_rate() {
        let mut m = FlatMemory::new(0x1000);
        // loop: addi r3,r3,1 ; add r4,r4,r3 ; cmpwi cr0,r3,0 ; bne loop  (never exits within the run)
        m.write32(0, 0x3863_0001);
        m.write32(4, 0x7C84_1A14);
        m.write32(8, 0x2C03_0000);
        m.write32(12, 0x4082_FFF4);
        let mut c = Cpu::new();
        let n = 100_000_000u64;
        let t = std::time::Instant::now();
        for _ in 0..n {
            c.step(&mut m).unwrap();
        }
        let secs = t.elapsed().as_secs_f64();
        println!("{:.1} M instructions/sec", n as f64 / secs / 1e6);
    }

    #[test]
    fn recent_pcs_rebuilds_history_across_branches() {
        let mut m = FlatMemory::new(0x1000);
        // 0: addi r3,r3,1 ; 4: b 0x20 ; 0x20: addi ; 0x24: addi ; 0x28: b 0x10 ; 0x10: addi ; 0x14: addi
        for (a, w) in [(0u32, 0x3863_0001u32), (4, 0x4800_001C), (0x20, 0x3863_0001), (0x24, 0x3863_0001), (0x28, 0x4BFF_FFE8), (0x10, 0x3863_0001), (0x14, 0x3863_0001)] {
            m.write32(a, w);
        }
        let mut c = Cpu::new();
        for _ in 0..7 {
            c.step(&mut m).unwrap();
        }
        assert_eq!(c.pc, 0x18);
        assert_eq!(c.recent_pcs(5), vec![0x24, 0x28, 0x10, 0x14, 0x18]);
        assert_eq!(c.recent_pcs(3), vec![0x10, 0x14, 0x18]);
        assert_eq!(c.recent_pcs(8), vec![0, 4, 0x20, 0x24, 0x28, 0x10, 0x14, 0x18]);
    }
}
