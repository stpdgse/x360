//! Exceptions and bug checks. The console dispatches structured exceptions by walking stack frames
//! with the executable's .pdata unwind tables; we don't have that walker, so anything that needs
//! real unwinding stops the run with a precise report instead of continuing wrongly. What *is*
//! handled: the debugger's thread-naming exception (in `kernel.rs`), bug checks, and context capture.

use crate::kernel::Kernel;
use crate::memory::GuestMemory;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

/// Layout used by RtlCaptureContext. Self-consistent but not verified against the SDK's CONTEXT:
/// +0x00 flags, +0x08 iar, +0x10 lr, +0x18 ctr, +0x20 gpr[32] (u64), +0x120 cr, +0x124 xer,
/// +0x128 fpscr (u64), +0x130 fpr[32] (f64).
pub const CTX_IAR: u32 = 0x08;
pub const CTX_GPR: u32 = 0x20;
pub const CTX_SIZE: u32 = 0x230;

pub fn capture_context(cpu: &Cpu, mem: &mut GuestMemory, addr: u32) {
    for off in (0..CTX_SIZE).step_by(4) {
        mem.write32(addr + off, 0);
    }
    mem.write32(addr, 0x0001_0001); // CONTEXT_CONTROL | CONTEXT_INTEGER-ish marker
    mem.write64(addr + CTX_IAR, cpu.lr); // a captured context "resumes" at the caller
    mem.write64(addr + 0x10, cpu.lr);
    mem.write64(addr + 0x18, cpu.ctr);
    for i in 0..32 {
        mem.write64(addr + CTX_GPR + i * 8, cpu.gpr[i as usize]);
        mem.write64(addr + 0x130 + i * 8, cpu.fpr[i as usize]);
    }
    mem.write32(addr + 0x120, cpu.cr);
    mem.write32(addr + 0x124, cpu.xer as u32);
}

impl Kernel {
    /// Exception-related calls: Ok(false) if `name` isn't one of ours, Err to stop the run.
    pub fn exception_call(&mut self, name: &str, cpu: &mut Cpu, mem: &mut GuestMemory) -> Result<bool, String> {
        match name {
            "KeBugCheck" => Err(format!("KeBugCheck: the title crashed the kernel on purpose, code {:#x}", cpu.gpr[3] as u32)),
            "KeBugCheckEx" => Err(format!(
                "KeBugCheckEx: code {:#x}, params {:#x} {:#x} {:#x} {:#x}",
                cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32, cpu.gpr[6] as u32, cpu.gpr[7] as u32
            )),
            "RtlCaptureContext" => {
                capture_context(cpu, mem, cpu.gpr[3] as u32);
                Ok(true)
            }
            "RtlUnwind" => Err(format!(
                "RtlUnwind(target frame {:#x}, target ip {:#x}, record {:#x}) at lr={:#x}: unwinding needs the .pdata walker, which isn't written yet",
                cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32, cpu.lr
            )),
            "__C_specific_handler" => {
                let rec = cpu.gpr[3] as u32;
                Err(format!(
                    "__C_specific_handler: a C++/SEH exception {:#010x} at {:#010x} reached a handler search; \
                     scope-table dispatch isn't implemented",
                    mem.read32(rec),
                    mem.read32(rec + 0xC)
                ))
            }
            _ => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_records_registers() {
        let mut mem = GuestMemory::new();
        mem.map(0x4000_0000, 0x1000);
        let mut cpu = Cpu::new();
        cpu.gpr[1] = 0x4000_F000;
        cpu.gpr[31] = 0x1234_5678_9ABC_DEF0;
        cpu.lr = 0x8200_1234;
        capture_context(&cpu, &mut mem, 0x4000_0100);
        assert_eq!(mem.read64(0x4000_0100 + CTX_GPR + 8), 0x4000_F000);
        assert_eq!(mem.read64(0x4000_0100 + CTX_GPR + 31 * 8), 0x1234_5678_9ABC_DEF0);
        assert_eq!(mem.read64(0x4000_0100 + CTX_IAR), 0x8200_1234);
    }

    #[test]
    fn bug_check_stops_the_run() {
        let mut k = Kernel::new();
        let mut mem = GuestMemory::new();
        let mut cpu = Cpu::new();
        cpu.gpr[3] = 0xDEAD;
        assert!(k.exception_call("KeBugCheck", &mut cpu, &mut mem).unwrap_err().contains("0xdead"));
        assert_eq!(k.exception_call("NotOurs", &mut cpu, &mut mem), Ok(false));
    }
}
