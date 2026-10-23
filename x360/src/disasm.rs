//! Minimal PowerPC disassembler for crash analysis. Covers the integer/branch/load-store core
//! well, and prints float/vector opcodes generically. Unknown encodings print as `.long`.

fn sext(v: u32, bits: u32) -> i64 {
    (((v as i64) << (64 - bits)) >> (64 - bits)) as i64
}

fn spr_name(spr: u32) -> String {
    match spr {
        1 => "xer".into(),
        8 => "lr".into(),
        9 => "ctr".into(),
        n => format!("spr{n}"),
    }
}

pub fn disasm(pc: u32, i: u32) -> String {
    let op = i >> 26;
    let rd = (i >> 21) & 31;
    let ra = (i >> 16) & 31;
    let rb = (i >> 11) & 31;
    let simm = sext(i & 0xFFFF, 16);
    let uimm = i & 0xFFFF;
    let rc = if i & 1 != 0 { "." } else { "" };
    let ea = |r: u32| if r == 0 { "0".to_string() } else { format!("r{r}") };
    match op {
        7 => format!("mulli r{rd},r{ra},{simm}"),
        8 => format!("subfic r{rd},r{ra},{simm}"),
        10 => format!("cmplwi cr{},r{ra},{uimm:#x}", (i >> 23) & 7),
        11 => format!("cmpwi cr{},r{ra},{simm}", (i >> 23) & 7),
        12 => format!("addic r{rd},r{ra},{simm}"),
        13 => format!("addic. r{rd},r{ra},{simm}"),
        14 => if ra == 0 { format!("li r{rd},{simm}") } else { format!("addi r{rd},r{ra},{simm}") },
        15 => if ra == 0 { format!("lis r{rd},{:#x}", uimm) } else { format!("addis r{rd},r{ra},{:#x}", uimm) },
        16 => {
            let (bo, bi) = (rd, ra);
            let bd = sext(i & 0xFFFC, 16);
            let target = if i & 2 != 0 { bd as u32 } else { pc.wrapping_add(bd as u32) };
            let lk = if i & 1 != 0 { "l" } else { "" };
            let cond = match (bo & 0x1F, bi & 3) {
                (0b01100, 0) | (0b01101, 0) => Some("lt"),
                (0b01100, 1) | (0b01101, 1) => Some("gt"),
                (0b01100, 2) | (0b01101, 2) => Some("eq"),
                (0b00100, 0) | (0b00101, 0) => Some("ge"),
                (0b00100, 1) | (0b00101, 1) => Some("le"),
                (0b00100, 2) | (0b00101, 2) => Some("ne"),
                _ => None,
            };
            match cond {
                Some(c) => format!("b{c}{lk} cr{},{target:#010x}", bi / 4),
                None => format!("bc{lk} {bo},{bi},{target:#010x}"),
            }
        }
        17 => "sc".into(),
        18 => {
            let li = sext(i & 0x03FF_FFFC, 26);
            let target = if i & 2 != 0 { li as u32 } else { pc.wrapping_add(li as u32) };
            format!("b{}{} {target:#010x}", if i & 1 != 0 { "l" } else { "" }, if i & 2 != 0 { "a" } else { "" })
        }
        19 => match (i >> 1) & 0x3FF {
            16 => if rd == 20 { format!("blr{}", if i & 1 != 0 { "l" } else { "" }) } else { format!("bclr {rd},{ra}") },
            528 => if rd == 20 { format!("bctr{}", if i & 1 != 0 { "l" } else { "" }) } else { format!("bcctr {rd},{ra}") },
            150 => "isync".into(),
            _ => format!(".long {i:#010x}"),
        },
        20 => format!("rlwimi{rc} r{ra},r{rd},{},{},{}", rb, (i >> 6) & 31, (i >> 1) & 31),
        21 => format!("rlwinm{rc} r{ra},r{rd},{},{},{}", rb, (i >> 6) & 31, (i >> 1) & 31),
        23 => format!("rlwnm{rc} r{ra},r{rd},r{rb},{},{}", (i >> 6) & 31, (i >> 1) & 31),
        24 => if i == 0x6000_0000 { "nop".into() } else { format!("ori r{ra},r{rd},{uimm:#x}") },
        25 => format!("oris r{ra},r{rd},{uimm:#x}"),
        26 => format!("xori r{ra},r{rd},{uimm:#x}"),
        27 => format!("xoris r{ra},r{rd},{uimm:#x}"),
        28 => format!("andi. r{ra},r{rd},{uimm:#x}"),
        29 => format!("andis. r{ra},r{rd},{uimm:#x}"),
        30 => format!("rld* r{ra},r{rd} (md-form {:#x})", i),
        2 => format!("tdi {rd},r{ra},{simm}"),
        3 => format!("twi {rd},r{ra},{simm}"),
        31 => {
            let xo = (i >> 1) & 0x3FF;
            match xo {
                0 => format!("cmpw cr{},r{ra},r{rb}", (i >> 23) & 7),
                32 => format!("cmplw cr{},r{ra},r{rb}", (i >> 23) & 7),
                339 => { let spr = ra | (rb << 5); format!("mf{} r{rd}", spr_name(spr)) }
                467 => { let spr = ra | (rb << 5); format!("mt{} r{rd}", spr_name(spr)) }
                371 => format!("mftb r{rd}"),
                444 => if rd == rb { format!("mr{rc} r{ra},r{rd}") } else { format!("or{rc} r{ra},r{rd},r{rb}") },
                28 => format!("and{rc} r{ra},r{rd},r{rb}"),
                316 => format!("xor{rc} r{ra},r{rd},r{rb}"),
                24 => format!("slw{rc} r{ra},r{rd},r{rb}"),
                536 => format!("srw{rc} r{ra},r{rd},r{rb}"),
                824 => format!("srawi{rc} r{ra},r{rd},{rb}"),
                26 => format!("cntlzw{rc} r{ra},r{rd}"),
                922 => format!("extsh{rc} r{ra},r{rd}"),
                954 => format!("extsb{rc} r{ra},r{rd}"),
                986 => format!("extsw{rc} r{ra},r{rd}"),
                x if x & 0x1FF == 266 => format!("add{rc} r{rd},r{ra},r{rb}"),
                x if x & 0x1FF == 40 => format!("subf{rc} r{rd},r{ra},r{rb}"),
                x if x & 0x1FF == 235 => format!("mullw{rc} r{rd},r{ra},r{rb}"),
                x if x & 0x1FF == 491 => format!("divw{rc} r{rd},r{ra},r{rb}"),
                x if x & 0x1FF == 459 => format!("divwu{rc} r{rd},r{ra},r{rb}"),
                23 => format!("lwzx r{rd},{},r{rb}", ea(ra)),
                87 => format!("lbzx r{rd},{},r{rb}", ea(ra)),
                279 => format!("lhzx r{rd},{},r{rb}", ea(ra)),
                151 => format!("stwx r{rd},{},r{rb}", ea(ra)),
                215 => format!("stbx r{rd},{},r{rb}", ea(ra)),
                407 => format!("sthx r{rd},{},r{rb}", ea(ra)),
                20 => format!("lwarx r{rd},{},r{rb}", ea(ra)),
                150 => format!("stwcx. r{rd},{},r{rb}", ea(ra)),
                598 => "sync".into(),
                854 => "eieio".into(),
                1014 => if rd == 1 { format!("dcbz128 {},r{rb}", ea(ra)) } else { format!("dcbz {},r{rb}", ea(ra)) },
                _ => format!(".long {i:#010x}  # op31 xo={xo}"),
            }
        }
        32 => format!("lwz r{rd},{simm}({})", ea(ra)),
        33 => format!("lwzu r{rd},{simm}(r{ra})"),
        34 => format!("lbz r{rd},{simm}({})", ea(ra)),
        35 => format!("lbzu r{rd},{simm}(r{ra})"),
        36 => format!("stw r{rd},{simm}({})", ea(ra)),
        37 => format!("stwu r{rd},{simm}(r{ra})"),
        38 => format!("stb r{rd},{simm}({})", ea(ra)),
        39 => format!("stbu r{rd},{simm}(r{ra})"),
        40 => format!("lhz r{rd},{simm}({})", ea(ra)),
        41 => format!("lhzu r{rd},{simm}(r{ra})"),
        42 => format!("lha r{rd},{simm}({})", ea(ra)),
        44 => format!("sth r{rd},{simm}({})", ea(ra)),
        45 => format!("sthu r{rd},{simm}(r{ra})"),
        46 => format!("lmw r{rd},{simm}({})", ea(ra)),
        47 => format!("stmw r{rd},{simm}({})", ea(ra)),
        48 => format!("lfs f{rd},{simm}({})", ea(ra)),
        50 => format!("lfd f{rd},{simm}({})", ea(ra)),
        52 => format!("stfs f{rd},{simm}({})", ea(ra)),
        54 => format!("stfd f{rd},{simm}({})", ea(ra)),
        58 => {
            let d = sext(i & 0xFFFC, 16);
            match i & 3 {
                0 => format!("ld r{rd},{d}({})", ea(ra)),
                1 => format!("ldu r{rd},{d}(r{ra})"),
                _ => format!("lwa r{rd},{d}({})", ea(ra)),
            }
        }
        62 => {
            let d = sext(i & 0xFFFC, 16);
            if i & 3 == 1 { format!("stdu r{rd},{d}(r{ra})") } else { format!("std r{rd},{d}({})", ea(ra)) }
        }
        59 | 63 => format!("fp{} {i:#010x}", if op == 59 { "s" } else { "d" }),
        4 | 5 | 6 => format!("vmx{op} {i:#010x}"),
        _ => format!(".long {i:#010x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_common_instructions() {
        assert_eq!(disasm(0, 0x7D8802A6), "mflr r12");
        assert_eq!(disasm(0, 0x4E80_0020), "blr");
        assert_eq!(disasm(0x1000, 0x4800_0010), "b 0x00001010");
        assert_eq!(disasm(0x1000, 0x4BFF_FFF1), "bl 0x00000ff0");
        assert_eq!(disasm(0, 0x9421_FF60), "stwu r1,-160(r1)");
        assert_eq!(disasm(0, 0x3860_0001), "li r3,1");
        assert_eq!(disasm(0, 0x816B_0000), "lwz r11,0(r11)");
        assert_eq!(disasm(0, 0x7C63_1B78), "mr r3,r3");
        assert_eq!(disasm(0x1000, 0x4182_0008), "beq cr0,0x00001008");
        assert_eq!(disasm(0, 0x6000_0000), "nop");
    }
}
