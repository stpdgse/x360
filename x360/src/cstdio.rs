//! printf-style formatting for guest CRT calls (sprintf, _vsnprintf, DbgPrint).
//!
//! Variadic arguments follow the console's PowerPC ABI as far as we've verified it:
//! argument n (counting the fixed ones) lives in r3+n for n < 8 and in the 8-byte home slot at
//! r1+0x50+8(n-8) above that (r3..r10 hold the first eight); 32-bit values sit in the slot's low word (+4), 64-bit values fill it.
//! Floating-point arguments arrive in f1.. in order. A `va_list` is a pointer to the slots.

use crate::memory::GuestMemory;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

/// Where the next variadic argument comes from.
pub trait ArgSource {
    fn next_u32(&mut self, mem: &GuestMemory) -> u32;
    fn next_u64(&mut self, mem: &GuestMemory) -> u64;
    fn next_f64(&mut self, mem: &GuestMemory) -> f64;
}

/// Arguments of a direct variadic call, starting at argument index `first` (the first variadic one).
pub struct RegArgs<'a> {
    cpu: &'a Cpu,
    n: usize,
    floats: usize,
}

impl<'a> RegArgs<'a> {
    pub fn new(cpu: &'a Cpu, first: usize) -> Self {
        Self { cpu, n: first, floats: 0 }
    }
    fn slot(&mut self, mem: &GuestMemory) -> u64 {
        let n = self.n;
        self.n += 1;
        if n < 8 {
            self.cpu.gpr[3 + n]
        } else {
            mem.read64(self.cpu.gpr[1] as u32 + 0x50 + 8 * (n as u32 - 8))
        }
    }
}

impl ArgSource for RegArgs<'_> {
    fn next_u32(&mut self, mem: &GuestMemory) -> u32 {
        self.slot(mem) as u32
    }
    fn next_u64(&mut self, mem: &GuestMemory) -> u64 {
        self.slot(mem)
    }
    fn next_f64(&mut self, mem: &GuestMemory) -> f64 {
        self.n += 1; // the float also consumes its integer slot
        self.floats += 1;
        f64::from_bits(self.cpu.fpr[self.floats.min(13)])
    }
}

/// A `va_list`: consecutive 8-byte slots in guest memory.
pub struct VaList(pub u32);

impl ArgSource for VaList {
    fn next_u32(&mut self, mem: &GuestMemory) -> u32 {
        let v = mem.read32(self.0 + 4);
        self.0 += 8;
        v
    }
    fn next_u64(&mut self, mem: &GuestMemory) -> u64 {
        let v = mem.read64(self.0);
        self.0 += 8;
        v
    }
    fn next_f64(&mut self, mem: &GuestMemory) -> f64 {
        f64::from_bits(self.next_u64(mem))
    }
}

pub fn read_cstr(mem: &GuestMemory, mut p: u32, limit: usize) -> Vec<u8> {
    let mut v = Vec::new();
    while v.len() < limit {
        let b = mem.read8(p);
        if b == 0 {
            break;
        }
        v.push(b);
        p += 1;
    }
    v
}

fn read_wstr(mem: &GuestMemory, mut p: u32, limit: usize) -> Vec<u8> {
    let mut v = Vec::new();
    while v.len() < limit {
        let c = mem.read16(p);
        if c == 0 {
            break;
        }
        v.push(if c < 0x80 { c as u8 } else { b'?' });
        p += 2;
    }
    v
}

fn pad(mut s: Vec<u8>, width: usize, left: bool, zero: bool) -> Vec<u8> {
    if s.len() < width {
        let fill = width - s.len();
        if left {
            s.extend(std::iter::repeat(b' ').take(fill));
        } else if zero {
            // zeros go after any sign / 0x prefix
            let at = if s.first().is_some_and(|c| *c == b'-' || *c == b'+' || *c == b' ') { 1 } else { 0 };
            let at = if s[at..].starts_with(b"0x") || s[at..].starts_with(b"0X") { at + 2 } else { at };
            for _ in 0..fill {
                s.insert(at, b'0');
            }
        } else {
            let mut out = vec![b' '; fill];
            out.extend(s);
            s = out;
        }
    }
    s
}

/// Format `fmt` with arguments from `args`.
pub fn format(mem: &GuestMemory, fmt: &[u8], args: &mut dyn ArgSource) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        i += 1;
        // flags
        let (mut left, mut plus, mut space, mut alt, mut zero) = (false, false, false, false, false);
        while i < fmt.len() {
            match fmt[i] {
                b'-' => left = true,
                b'+' => plus = true,
                b' ' => space = true,
                b'#' => alt = true,
                b'0' => zero = true,
                _ => break,
            }
            i += 1;
        }
        // width
        let mut width = 0usize;
        if i < fmt.len() && fmt[i] == b'*' {
            let w = args.next_u32(mem) as i32;
            if w < 0 {
                left = true;
            }
            width = w.unsigned_abs() as usize;
            i += 1;
        } else {
            while i < fmt.len() && fmt[i].is_ascii_digit() {
                width = width * 10 + (fmt[i] - b'0') as usize;
                i += 1;
            }
        }
        // precision
        let mut prec: Option<usize> = None;
        if i < fmt.len() && fmt[i] == b'.' {
            i += 1;
            let mut p = 0usize;
            if i < fmt.len() && fmt[i] == b'*' {
                p = args.next_u32(mem).max(0) as usize;
                i += 1;
            } else {
                while i < fmt.len() && fmt[i].is_ascii_digit() {
                    p = p * 10 + (fmt[i] - b'0') as usize;
                    i += 1;
                }
            }
            prec = Some(p);
        }
        // length
        let (mut long, mut wide) = (false, false);
        loop {
            match fmt.get(i) {
                Some(b'l') => {
                    if long {
                        // "ll": already 64-bit
                    }
                    long = true;
                    wide = true; // `%ls` / `%lc` are wide
                    i += 1;
                }
                Some(b'h') => {
                    i += 1;
                    wide = false;
                }
                Some(b'I') if fmt.get(i + 1) == Some(&b'6') && fmt.get(i + 2) == Some(&b'4') => {
                    long = true;
                    i += 3;
                }
                Some(b'w') => {
                    wide = true;
                    i += 1;
                }
                _ => break,
            }
        }
        let Some(&conv) = fmt.get(i) else { break };
        i += 1;
        let sign_prefix = |neg: bool| -> &'static [u8] {
            if neg { b"-" } else if plus { b"+" } else if space { b" " } else { b"" }
        };
        let piece: Vec<u8> = match conv {
            b'%' => b"%".to_vec(),
            b'c' => {
                let c = args.next_u32(mem);
                vec![if c < 256 { c as u8 } else { b'?' }]
            }
            b's' | b'S' => {
                let p = args.next_u32(mem);
                let is_wide = (conv == b's' && wide) || (conv == b'S' && !wide);
                let limit = prec.unwrap_or(usize::MAX);
                if p == 0 {
                    b"(null)".to_vec()
                } else if is_wide {
                    read_wstr(mem, p, limit.min(4096))
                } else {
                    read_cstr(mem, p, limit.min(4096))
                }
            }
            b'd' | b'i' => {
                let v = if long { args.next_u64(mem) as i64 } else { args.next_u32(mem) as i32 as i64 };
                let mut digits = v.unsigned_abs().to_string().into_bytes();
                if let Some(p) = prec {
                    while digits.len() < p {
                        digits.insert(0, b'0');
                    }
                }
                let mut s = sign_prefix(v < 0).to_vec();
                s.extend(digits);
                s
            }
            b'u' | b'x' | b'X' | b'o' => {
                let v = if long { args.next_u64(mem) } else { args.next_u32(mem) as u64 };
                let mut digits = match conv {
                    b'u' => v.to_string(),
                    b'x' => format!("{v:x}"),
                    b'X' => format!("{v:X}"),
                    _ => format!("{v:o}"),
                }
                .into_bytes();
                if let Some(p) = prec {
                    while digits.len() < p {
                        digits.insert(0, b'0');
                    }
                }
                let mut s = Vec::new();
                if alt && v != 0 {
                    match conv {
                        b'x' => s.extend(b"0x"),
                        b'X' => s.extend(b"0X"),
                        b'o' => s.push(b'0'),
                        _ => {}
                    }
                }
                s.extend(digits);
                s
            }
            b'p' => format!("{:08X}", args.next_u32(mem)).into_bytes(),
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
                let v = args.next_f64(mem);
                let p = prec.unwrap_or(6);
                let body = if v.is_nan() {
                    "nan".to_string()
                } else if v.is_infinite() {
                    "inf".to_string()
                } else {
                    match conv {
                        b'f' | b'F' => format!("{:.*}", p, v.abs()),
                        b'e' | b'E' => fmt_exp(v.abs(), p, conv == b'E'),
                        _ => fmt_general(v.abs(), p.max(1), conv == b'G'),
                    }
                };
                let mut s = sign_prefix(v.is_sign_negative() && !v.is_nan()).to_vec();
                s.extend(body.into_bytes());
                s
            }
            b'n' => {
                args.next_u32(mem);
                Vec::new()
            }
            other => vec![b'%', other],
        };
        let numeric = matches!(conv, b'd' | b'i' | b'u' | b'x' | b'X' | b'o' | b'f' | b'F' | b'e' | b'E' | b'g' | b'G');
        let zero_ok = zero && !left && numeric && !(prec.is_some() && matches!(conv, b'd' | b'i' | b'u' | b'x' | b'X' | b'o'));
        out.extend(pad(piece, width, left, zero_ok));
    }
    out
}

fn fmt_exp(v: f64, prec: usize, upper: bool) -> String {
    let s = format!("{:.*e}", prec, v); // Rust: 1.5e2, C: 1.5e+02
    let (m, e) = s.split_once('e').unwrap();
    let exp: i32 = e.parse().unwrap();
    let r = format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    if upper { r.to_uppercase() } else { r }
}

fn fmt_general(v: f64, prec: usize, upper: bool) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let exp = v.log10().floor() as i32;
    let s = if exp < -4 || exp >= prec as i32 {
        let t = fmt_exp(v, prec - 1, upper);
        // strip trailing zeros of the mantissa
        let (m, e) = t.split_at(t.find(['e', 'E']).unwrap());
        let m = if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.') } else { m };
        format!("{m}{e}")
    } else {
        let decimals = (prec as i32 - 1 - exp).max(0) as usize;
        let t = format!("{:.*}", decimals, v);
        if t.contains('.') { t.trim_end_matches('0').trim_end_matches('.').to_string() } else { t }
    };
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed {
        ints: Vec<u64>,
        floats: Vec<f64>,
    }
    impl ArgSource for Fixed {
        fn next_u32(&mut self, _: &GuestMemory) -> u32 {
            self.ints.remove(0) as u32
        }
        fn next_u64(&mut self, _: &GuestMemory) -> u64 {
            self.ints.remove(0)
        }
        fn next_f64(&mut self, _: &GuestMemory) -> f64 {
            self.floats.remove(0)
        }
    }

    fn fmt(f: &str, ints: &[u64], floats: &[f64]) -> String {
        let mem = GuestMemory::new();
        let mut a = Fixed { ints: ints.to_vec(), floats: floats.to_vec() };
        String::from_utf8(format(&mem, f.as_bytes(), &mut a)).unwrap()
    }

    #[test]
    fn integers_and_flags() {
        assert_eq!(fmt("%d|%5d|%-5d|%05d|%+d", &[42, 42, 42, 42, 42], &[]), "42|   42|42   |00042|+42");
        assert_eq!(fmt("%d", &[(-7i32) as u32 as u64], &[]), "-7");
        assert_eq!(fmt("%u", &[0xFFFF_FFFF], &[]), "4294967295");
        assert_eq!(fmt("%x %X %#x %08x", &[255, 255, 255, 0xBEEF], &[]), "ff FF 0xff 0000beef");
        assert_eq!(fmt("%lld", &[(-5i64) as u64], &[]), "-5");
        assert_eq!(fmt("%.3d", &[7], &[]), "007");
        assert_eq!(fmt("%*d", &[4, 9], &[]), "   9");
        assert_eq!(fmt("100%%", &[], &[]), "100%");
    }

    #[test]
    fn floats() {
        assert_eq!(fmt("%f", &[], &[3.14159]), "3.141590");
        assert_eq!(fmt("%.2f|%8.3f|%-8.1f|", &[], &[2.0, 1.5, 9.99]), "2.00|   1.500|10.0    |");
        assert_eq!(fmt("%e", &[], &[1234.5]), "1.234500e+03");
        assert_eq!(fmt("%g %g %g", &[], &[0.5, 100000.0, 1e10]), "0.5 100000 1e+10");
        assert_eq!(fmt("%f", &[], &[-0.25]), "-0.250000");
    }

    #[test]
    fn strings_chars_from_guest_memory() {
        let mut mem = GuestMemory::new();
        mem.map(0x4000_0000, 0x1000);
        for (i, b) in b"hello\0".iter().enumerate() {
            mem.write8(0x4000_0000 + i as u32, *b);
        }
        for (i, c) in "wide".encode_utf16().enumerate() {
            mem.write16(0x4000_0100 + i as u32 * 2, c);
        }
        let mut a = Fixed { ints: vec![0x4000_0000, 0x4000_0000, 0x4000_0100, 0, 'Z' as u64], floats: vec![] };
        let out = format(&mem, b"%s|%.3s|%ls|%s|%c", &mut a);
        assert_eq!(String::from_utf8(out).unwrap(), "hello|hel|wide|(null)|Z");
    }

    #[test]
    fn va_list_slots() {
        let mut mem = GuestMemory::new();
        mem.map(0x4000_0000, 0x1000);
        mem.write32(0x4000_0004, 5); // slot 0, low word
        mem.write64(0x4000_0008, 1.5f64.to_bits()); // slot 1
        let mut v = VaList(0x4000_0000);
        assert_eq!(String::from_utf8(format(&mem, b"%d %.1f", &mut v)).unwrap(), "5 1.5");
    }
}
