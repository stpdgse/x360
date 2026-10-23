pub mod alu;
pub mod fpu;
pub mod interp;
pub mod jit;
pub mod memory;
pub mod vmx;
mod vmx_int;
pub mod vmx128;


/// PowerPC architectural state for one hardware thread.
pub struct Cpu {
    /// Ring buffer of the last executed (pc, instruction) pairs, for crash reports.
    /// Ring of the last 64 taken control transfers as (from, to) addresses. Straight-line code between
    /// them is implied, so the recent instruction history can be rebuilt exactly (`recent_pcs`) while the
    /// hot loop only pays for the (rare) taken branches.
    pub trace: [(u32, u32); 64],
    pub trace_pos: usize,
    /// Address held by the last lwarx/ldarx, cleared by a store-conditional.
    pub reservation: Option<u32>,
    /// Executed-instruction count (1 "cycle" each). The time base is derived from it:
    /// 3.2GHz core / 49.875MHz time base ~= 64 cycles per tick, so runs stay deterministic.
    pub cycles: u64,
    /// Virtual cycles charged per executed instruction (an in-order core averages well above 1).
    pub cpi: u64,
    pub gpr: [u64; 32],
    pub fpr: [u64; 32], // raw bit patterns
    pub fpscr: u32,
    pub vr: [u128; 128], // VMX128: 128 vector registers, lane 0 = most significant
    pub cr: u32,
    pub lr: u64,
    pub ctr: u64,
    pub xer: u64,
    pub msr: u64,
    pub pc: u64,
    /// Trap raised inside JIT-compiled code, handed back to the caller when the block exits.
    pub pending_trap: Option<interp::Trap>,
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            trace: [(0, 0); 64],
            trace_pos: 0,
            reservation: None,
            cycles: 0,
            cpi: 1,
            gpr: [0; 32],
            fpr: [0; 32],
            fpscr: 0,
            vr: [0; 128],
            cr: 0,
            lr: 0,
            ctr: 0,
            xer: 0,
            msr: 1 << 63, // 64-bit mode by default; the kernel starts guest threads with 0x9030 (32-bit)
            pc: 0,
            pending_trap: None,
        }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

/// A plain-data copy of a [`Cpu`] for save states (the register arrays are too big for serde's array impls).
#[derive(serde::Serialize, serde::Deserialize)]
struct CpuSnap {
    trace: Vec<(u32, u32)>,
    trace_pos: usize,
    reservation: Option<u32>,
    cycles: u64,
    cpi: u64,
    gpr: Vec<u64>,
    fpr: Vec<u64>,
    fpscr: u32,
    vr: Vec<u128>,
    cr: u32,
    lr: u64,
    ctr: u64,
    xer: u64,
    msr: u64,
    pc: u64,
}

impl serde::Serialize for Cpu {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        CpuSnap {
            trace: self.trace.to_vec(),
            trace_pos: self.trace_pos,
            reservation: self.reservation,
            cycles: self.cycles,
            cpi: self.cpi,
            gpr: self.gpr.to_vec(),
            fpr: self.fpr.to_vec(),
            fpscr: self.fpscr,
            vr: self.vr.to_vec(),
            cr: self.cr,
            lr: self.lr,
            ctr: self.ctr,
            xer: self.xer,
            msr: self.msr,
            pc: self.pc,
        }
        .serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for Cpu {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let n = CpuSnap::deserialize(d)?;
        let mut c = Cpu::new();
        c.trace.copy_from_slice(&n.trace);
        c.trace_pos = n.trace_pos;
        c.reservation = n.reservation;
        c.cycles = n.cycles;
        c.cpi = n.cpi;
        c.gpr.copy_from_slice(&n.gpr);
        c.fpr.copy_from_slice(&n.fpr);
        c.fpscr = n.fpscr;
        c.vr.copy_from_slice(&n.vr);
        c.cr = n.cr;
        c.lr = n.lr;
        c.ctr = n.ctr;
        c.xer = n.xer;
        c.msr = n.msr;
        c.pc = n.pc;
        Ok(c)
    }
}
