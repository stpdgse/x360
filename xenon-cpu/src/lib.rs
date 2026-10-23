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
            msr: 0,
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
