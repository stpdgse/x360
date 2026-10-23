//! Kernel-exported C runtime and Rtl string/time helpers, plus the system-information queries
//! (language, region, AV pack) that titles call during start-up.

use crate::cstdio::{self, RegArgs, VaList};
use crate::kernel::Kernel;
use crate::memory::GuestMemory;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

/// 100ns FILETIME ticks between 1601-01-01 and 2008-05-01 (GTA IV's release month), the epoch of the
/// virtual wall clock: deterministic runs, advancing with executed instructions.
const EPOCH_FILETIME: u64 = 128_540_736_000_000_000;
const TICKS_PER_SEC: u64 = 10_000_000;
const STATUS_BUFFER_OVERFLOW: u64 = 0x8000_0005;

/// Seconds since 1601 -> (year, month, day, weekday), civil calendar.
fn civil_from_days(days_since_1601: i64) -> (i32, u32, u32, u32) {
    // 1601-01-01 is 134_774 days before 1970-01-01; Hinnant's algorithm counts from 0000-03-01
    let z = days_since_1601 - 134_774 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = (y + (m <= 2) as i64) as i32;
    let weekday = ((days_since_1601 + 1).rem_euclid(7)) as u32; // 1601-01-01 was a Monday
    (year, m, d, weekday)
}

fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = y as i64 - (m <= 2) as i64;
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    // days since 1970-01-01, then shift to 1601-01-01
    era * 146_097 + doe - 719_468 + 134_774
}

impl Kernel {
    /// The virtual wall clock as a FILETIME.
    pub fn system_time(&self) -> u64 {
        // cycles at 3.2GHz -> 100ns ticks
        EPOCH_FILETIME + self.global_cycles / 320
    }

    /// CRT / Rtl helpers: returns false if `name` isn't one of ours.
    pub fn crt_call(&mut self, name: &str, cpu: &mut Cpu, mem: &mut GuestMemory) -> bool {
        let (a, b, c) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32);
        match name {
            "sprintf" => {
                // (char *buf, const char *fmt, ...)
                let fmt = cstdio::read_cstr(mem, b, 4096);
                let out = cstdio::format(mem, &fmt, &mut RegArgs::new(cpu, 2));
                for (i, ch) in out.iter().enumerate() {
                    mem.write8(a + i as u32, *ch);
                }
                mem.write8(a + out.len() as u32, 0);
                cpu.gpr[3] = out.len() as u64;
            }
            "_vsnprintf" => {
                // (char *buf, size_t count, const char *fmt, va_list args) -> chars written, or -1 if truncated
                let fmt = cstdio::read_cstr(mem, c, 4096);
                let out = cstdio::format(mem, &fmt, &mut VaList(cpu.gpr[6] as u32));
                let room = b as usize;
                let n = out.len().min(room);
                for (i, ch) in out[..n].iter().enumerate() {
                    mem.write8(a + i as u32, *ch);
                }
                if out.len() < room {
                    mem.write8(a + out.len() as u32, 0);
                    cpu.gpr[3] = out.len() as u64;
                } else {
                    cpu.gpr[3] = u64::MAX; // -1: didn't fit (and no terminator, as in the MS CRT)
                }
            }
            "DbgPrint" => {
                // (const char *fmt, ...)
                let fmt = cstdio::read_cstr(mem, a, 4096);
                let out = cstdio::format(mem, &fmt, &mut RegArgs::new(cpu, 1));
                print!("[guest DbgPrint] {}", String::from_utf8_lossy(&out));
                if !out.ends_with(b"\n") {
                    println!();
                }
                cpu.gpr[3] = 0;
            }
            "KeQuerySystemTime" => {
                let t = self.system_time();
                mem.write32(a, (t >> 32) as u32);
                mem.write32(a + 4, t as u32);
            }
            "RtlTimeToTimeFields" => {
                // (const LARGE_INTEGER *time, TIME_FIELDS *out): year, month, day, hour, minute, second, ms, weekday (u16 each)
                let t = ((mem.read32(a) as u64) << 32) | mem.read32(a + 4) as u64;
                let secs = t / TICKS_PER_SEC;
                let (y, mo, d, wd) = civil_from_days((secs / 86_400) as i64);
                let tod = secs % 86_400;
                let fields = [y as u16, mo as u16, d as u16, (tod / 3600) as u16, (tod / 60 % 60) as u16, (tod % 60) as u16,
                    ((t % TICKS_PER_SEC) / 10_000) as u16, wd as u16];
                for (i, f) in fields.iter().enumerate() {
                    mem.write16(b + i as u32 * 2, *f);
                }
            }
            "RtlTimeFieldsToTime" => {
                // (const TIME_FIELDS *in, LARGE_INTEGER *out) -> BOOLEAN
                let f: Vec<u16> = (0..8).map(|i| mem.read16(a + i * 2)).collect();
                let valid = (1..=12).contains(&f[1]) && (1..=31).contains(&f[2]) && f[3] < 24 && f[4] < 60 && f[5] < 60 && f[6] < 1000;
                if valid {
                    let days = days_from_civil(f[0] as i32, f[1] as u32, f[2] as u32);
                    let secs = days as u64 * 86_400 + f[3] as u64 * 3600 + f[4] as u64 * 60 + f[5] as u64;
                    let t = secs * TICKS_PER_SEC + f[6] as u64 * 10_000;
                    mem.write32(b, (t >> 32) as u32);
                    mem.write32(b + 4, t as u32);
                }
                cpu.gpr[3] = valid as u64;
            }
            "RtlUpcaseUnicodeChar" => {
                let ch = char::from_u32(a & 0xFFFF).unwrap_or('\0');
                let up = ch.to_uppercase().next().unwrap_or(ch) as u32;
                cpu.gpr[3] = if up <= 0xFFFF { up as u64 } else { ch as u64 };
            }
            "RtlCompareMemoryUlong" => {
                // (source, length in bytes, 32-bit pattern) -> number of leading bytes that match
                let n = b & !3;
                let mut matched = 0;
                while matched < n && mem.read32(a + matched) == c {
                    matched += 4;
                }
                cpu.gpr[3] = matched as u64;
            }
            "RtlCompareStringN" => {
                // (ANSI_STRING *a, ANSI_STRING *b, BOOLEAN ignore case): STRING = {u16 length, u16 max, char *buffer}
                let get = |mem: &GuestMemory, s: u32| cstdio::read_cstr(mem, mem.read32(s + 4), mem.read16(s) as usize);
                let (x, y) = (get(mem, a), get(mem, b));
                let (x, y) = if c & 0xFF != 0 { (x.to_ascii_uppercase(), y.to_ascii_uppercase()) } else { (x, y) };
                cpu.gpr[3] = match x.cmp(&y) {
                    std::cmp::Ordering::Less => u64::MAX,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                };
            }
            "RtlMultiByteToUnicodeN" => {
                // (WCHAR *out, ULONG out bytes, ULONG *bytes written, const char *src, ULONG src bytes)
                let (src, src_len) = (cpu.gpr[6] as u32, cpu.gpr[7] as u32);
                let n = src_len.min(b / 2);
                for i in 0..n {
                    mem.write16(a + i * 2, mem.read8(src + i) as u16);
                }
                if c != 0 {
                    mem.write32(c, n * 2);
                }
                cpu.gpr[3] = 0;
            }
            "RtlUnicodeToMultiByteN" => {
                // (char *out, ULONG out bytes, ULONG *bytes written, const WCHAR *src, ULONG src bytes)
                let (src, src_len) = (cpu.gpr[6] as u32, cpu.gpr[7] as u32);
                let n = (src_len / 2).min(b);
                for i in 0..n {
                    let ch = mem.read16(src + i * 2);
                    mem.write8(a + i, if ch < 0x100 { ch as u8 } else { b'?' });
                }
                if c != 0 {
                    mem.write32(c, n);
                }
                cpu.gpr[3] = 0;
            }
            "RtlUnicodeStringToAnsiString" => {
                // (ANSI_STRING *dest, UNICODE_STRING *src, BOOLEAN allocate): dest buffer must be provided
                // by the caller (allocation is `ExAllocatePoolWithTag` territory we don't model yet).
                let (dest, src) = (a, b);
                let (len, buf) = (mem.read16(src) as u32 / 2, mem.read32(src + 4));
                let out = mem.read32(dest + 4);
                if c & 0xFF != 0 || out == 0 {
                    cpu.gpr[3] = 0xC000_0017; // STATUS_NO_MEMORY: allocation mode not supported
                } else if len > mem.read16(dest + 2) as u32 {
                    cpu.gpr[3] = STATUS_BUFFER_OVERFLOW;
                } else {
                    for i in 0..len {
                        let ch = mem.read16(buf + i * 2);
                        mem.write8(out + i, if ch < 0x100 { ch as u8 } else { b'?' });
                    }
                    mem.write16(dest, len as u16);
                    cpu.gpr[3] = 0;
                }
            }
            "RtlFreeAnsiString" => {}
            // ---- system information ----
            "XGetLanguage" => cpu.gpr[3] = 1, // XC_LANGUAGE_ENGLISH
            "XGetGameRegion" => cpu.gpr[3] = 0x00FF, // region-free
            "XGetAVPack" => cpu.gpr[3] = 6, // HDMI
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_round_trip() {
        // 2008-04-29 was a Tuesday (GTA IV's North American release)
        let d = days_from_civil(2008, 4, 29);
        assert_eq!(civil_from_days(d), (2008, 4, 29, 2));
        assert_eq!(civil_from_days(0), (1601, 1, 1, 1)); // a Monday
        assert_eq!(civil_from_days(days_from_civil(2000, 2, 29)), (2000, 2, 29, 2));
        assert_eq!(days_from_civil(1970, 1, 1), 134_774);
        let epoch = civil_from_days((EPOCH_FILETIME / TICKS_PER_SEC / 86_400) as i64);
        assert_eq!((epoch.0, epoch.1, epoch.2), (2008, 5, 1));
    }

    #[test]
    fn sprintf_writes_terminated_string() {
        let mut k = Kernel::new();
        let mut mem = GuestMemory::new();
        mem.map(0x4000_0000, 0x1000);
        for (i, b) in b"%s-%d-%.1f\0".iter().enumerate() {
            mem.write8(0x4000_0100 + i as u32, *b);
        }
        for (i, b) in b"abc\0".iter().enumerate() {
            mem.write8(0x4000_0200 + i as u32, *b);
        }
        let mut cpu = Cpu::new();
        cpu.gpr[3] = 0x4000_0000;
        cpu.gpr[4] = 0x4000_0100;
        cpu.gpr[5] = 0x4000_0200;
        cpu.gpr[6] = 7;
        cpu.fpr[1] = 2.5f64.to_bits();
        assert!(k.crt_call("sprintf", &mut cpu, &mut mem));
        assert_eq!(cpu.gpr[3], 9);
        assert_eq!(cstdio::read_cstr(&mem, 0x4000_0000, 64), b"abc-7-2.5");
    }

    #[test]
    fn vsnprintf_truncation() {
        let mut k = Kernel::new();
        let mut mem = GuestMemory::new();
        mem.map(0x4000_0000, 0x1000);
        for (i, b) in b"%d\0".iter().enumerate() {
            mem.write8(0x4000_0100 + i as u32, *b);
        }
        mem.write32(0x4000_0204, 123456);
        let mut cpu = Cpu::new();
        cpu.gpr[3] = 0x4000_0000;
        cpu.gpr[4] = 4;
        cpu.gpr[5] = 0x4000_0100;
        cpu.gpr[6] = 0x4000_0200;
        assert!(k.crt_call("_vsnprintf", &mut cpu, &mut mem));
        assert_eq!(cpu.gpr[3] as i64, -1, "6 chars don't fit in 4");
        assert_eq!(cstdio::read_cstr(&mem, 0x4000_0000, 4), b"1234");
    }
}
