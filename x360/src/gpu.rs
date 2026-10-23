//! Xenos GPU front end: ring-buffer state and the PM4 command processor.
//! Physical addresses coming from the title are turned into guest virtual addresses through
//! the 0xA0000000 physical-memory view, which is where every physical allocation lives.
//!
//! The CP runs synchronously: when the title writes CP_RB_WPTR, the run loop calls `process`,
//! which consumes every pending packet and updates the read pointer.

use serde::{Deserialize, Serialize};
use crate::memory::GuestMemory;
use crate::xenos_draw::Renderer;
use crate::xenos_vm::ShaderKind;
use std::collections::{HashMap, VecDeque};
use xenon_cpu::memory::Bus;

pub const PHYS_VIEW: u32 = 0xA000_0000;

// Register indices (dword index into the register file; MMIO address = 0x7FC80000 + 4 * index).
pub const REG_CP_RB_BASE: usize = 0x01C0;
pub const REG_CP_RB_CNTL: usize = 0x01C1;
pub const REG_CP_RB_RPTR_ADDR: usize = 0x01C3;
pub const REG_CP_RB_RPTR: usize = 0x01C4;
pub const REG_CP_RB_WPTR: usize = 0x01C5;
pub const REG_SCRATCH_UMSK: usize = 0x01DC;
pub const REG_SCRATCH_ADDR: usize = 0x01DD;
// Render-backend / sequencer state the draw logger looks at (indices from public Xenos register lists).
pub const REG_RB_SURFACE_INFO: usize = 0x2000;
pub const REG_RB_COLOR_INFO: usize = 0x2001;
pub const REG_RB_MODECONTROL: usize = 0x2208;
pub const REG_RB_COPY_CONTROL: usize = 0x2318;
pub const REG_RB_COPY_DEST_BASE: usize = 0x2319;
pub const REG_SQ_VS_PROGRAM: usize = 0x21A3;
pub const REG_SQ_PS_PROGRAM: usize = 0x21A4;

fn prim_name(p: u32) -> &'static str {
    match p {
        1 => "points",
        2 => "lines",
        3 => "line_strip",
        4 => "tri_list",
        5 => "tri_fan",
        6 => "tri_strip",
        8 => "rect_list",
        0xD => "quad_list",
        _ => "prim?",
    }
}

fn opcode_name(op: u32) -> &'static str {
    match op {
        0x10 => "NOP",
        0x21 => "REG_RMW",
        0x22 => "DRAW_INDX",
        0x25 => "VIZ_QUERY",
        0x27 => "IM_LOAD",
        0x2B => "IM_LOAD_IMMEDIATE",
        0x2D => "SET_CONSTANT",
        0x2F => "LOAD_ALU_CONSTANT",
        0x36 => "DRAW_INDX_2",
        0x37 => "INDIRECT_BUFFER_PFD",
        0x3B => "INVALIDATE_STATE",
        0x3C => "WAIT_REG_MEM",
        0x3E => "MEM_WRITE",
        0x3F => "INDIRECT_BUFFER",
        0x45 => "COND_WRITE",
        0x46 => "EVENT_WRITE",
        0x48 => "ME_INIT",
        0x54 => "INTERRUPT",
        0x61 => "SET_BIN_MASK_HI",
        0x62 => "SET_BIN_SELECT_LO",
        0x63 => "SET_BIN_SELECT_HI",
        0x4B => "SET_SHADER_CONSTANTS",
        0x55 => "SET_CONSTANT2",
        0x58 => "EVENT_WRITE_SHD",
        0x5A => "EVENT_WRITE_EXT",
        0x60 => "SET_BIN_MASK_LO",
        0x64 => "XE_SWAP",
        _ => "?",
    }
}

/// Byte offset of pixel (x, y) in a Xenos 2D-tiled surface (32x32-pixel macro tiles, swizzled
/// inside). `width` is the surface width in pixels (rounded up to 32 here), `log_bpp` is
/// log2(bytes per pixel). Returns an offset in *pixels* scaled back to bytes by the caller.
/// Algorithm after the well-known XGAddress2DTiledOffset routine.
pub fn tiled_offset_2d(x: u32, y: u32, width: u32, log_bpp: u32) -> u32 {
    let width = (width + 31) & !31;
    // outer part depends on y (and the macro-tile row)
    let macro_y = ((y >> 5) * (width >> 5)) << (log_bpp + 7);
    let micro_y = ((y & 6) << 2) << log_bpp;
    let base = macro_y + ((micro_y & !0xF) << 1) + (micro_y & 0xF) + ((y & 8) << (3 + log_bpp)) + ((y & 1) << 4);
    // inner part adds x
    let macro_x = (x >> 5) << (log_bpp + 7);
    let micro_x = (x & 7) << log_bpp;
    let offset = base + macro_x + ((micro_x & !0xF) << 1) + (micro_x & 0xF);
    ((offset & !0x1FF) << 3)
        + ((offset & 0x1C0) << 2)
        + (offset & 0x3F)
        + ((y & 16) << 7)
        + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6)
}

impl FrontBuffer {
    /// Read the front buffer out of guest memory as a top-down RGBA8 image (assumes a 32-bit
    /// 8:8:8:8 format; with endian mode 0 the bytes are already in A,R,G,B memory order).
    pub fn read_rgba(&self, mem: &GuestMemory) -> Vec<u8> {
        let base = PHYS_VIEW | self.base_phys;
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for y in 0..self.height {
            for x in 0..self.width {
                let off = if self.tiled {
                    tiled_offset_2d(x, y, self.width, 2)
                } else {
                    y * self.pitch.max(self.width * 4) + x * 4
                };
                // Apply the surface's endian mode, then the k_8_8_8_8 layout: R in the low byte.
                let p = crate::xenos_draw::apply_endian(mem.read32(base + (off & !3)), self.endian);
                out.extend_from_slice(&[p as u8, (p >> 8) as u8, (p >> 16) as u8, 0xFF]);
            }
        }
        out
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// Write an RGBA8 image as a PNG using stored (uncompressed) deflate blocks.
pub fn write_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    // raw scanlines, each prefixed with filter type 0
    let mut raw = Vec::with_capacity(((width * 4 + 1) * height) as usize);
    for y in 0..height {
        raw.push(0);
        raw.extend_from_slice(&rgba[(y * width * 4) as usize..((y + 1) * width * 4) as usize]);
    }
    // zlib stream: header, stored blocks of <= 65535 bytes, adler32
    let mut z = vec![0x78, 0x01];
    let mut chunks = raw.chunks(65535).peekable();
    while let Some(c) = chunks.next() {
        z.push(if chunks.peek().is_none() { 1 } else { 0 });
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A])?;
    let mut chunk = |kind: &[u8; 4], data: &[u8]| -> std::io::Result<()> {
        f.write_all(&(data.len() as u32).to_be_bytes())?;
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        f.write_all(&body)?;
        f.write_all(&crc32(&body).to_be_bytes())
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA
    chunk(b"IHDR", &ihdr)?;
    chunk(b"IDAT", &z)?;
    chunk(b"IEND", &[])
}

/// Write an RGBA8 image as an uncompressed 32-bit BMP (opens in any viewer).
pub fn write_bmp(path: &str, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let pixel_bytes = width * height * 4;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"BM")?;
    f.write_all(&(54 + pixel_bytes).to_le_bytes())?;
    f.write_all(&[0; 4])?;
    f.write_all(&54u32.to_le_bytes())?;
    f.write_all(&40u32.to_le_bytes())?;
    f.write_all(&width.to_le_bytes())?;
    f.write_all(&height.to_le_bytes())?; // positive height = bottom-up rows
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&32u16.to_le_bytes())?;
    f.write_all(&[0; 4])?; // BI_RGB
    f.write_all(&pixel_bytes.to_le_bytes())?;
    f.write_all(&[0; 16])?;
    for row in (0..height).rev() {
        for x in 0..width {
            let i = ((row * width + x) * 4) as usize;
            f.write_all(&[rgba[i + 2], rgba[i + 1], rgba[i], 0xFF])?; // BGRA
        }
    }
    Ok(())
}

/// Front buffer description decoded from a texture fetch constant at swap time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontBuffer {
    pub base_phys: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub format: u32,
    pub endian: u32,
    pub tiled: bool,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Gpu {
    pub renderer: Renderer,
    /// Number of XE_SWAP packets executed (frames presented).
    pub frames: u64,
    /// Is the display output switched on? Frames swapped while it is off are not shown.
    pub display_on: bool,
    pub no_render: bool,
    sample_every: u64,
    /// X360_SNAPSHOT="frame,draws,dir": from that frame on, render `draws` draws, then write every render
    /// target to dir as PNG and stop rendering (shows what the title draws off-screen).
    snapshot: Option<(u64, u64, String)>,
    snapshot_draws: u64,
    /// Live preview window, if X360_WINDOW is set: presented frames are handed to it.
    #[serde(skip)]
    pub window: Option<std::sync::Arc<crate::window::Shared>>,
    pub front_buffer: Option<FrontBuffer>,
    /// Physical address of the primary ring buffer.
    pub ring_base: u32,
    /// log2 of the ring size in 8-byte units (CP_RB_CNTL.RB_BUFSZ convention).
    pub ring_size_log2: u32,
    /// Read pointer, in dwords; advanced as the command processor consumes packets.
    pub rptr: u32,
    /// Physical address the read pointer is mirrored to after each update (0 = disabled).
    pub rptr_writeback: u32,
    pub rptr_writeback_block: u32,
    /// Print every packet (X360_TRACE_GPU=1).
    pub trace: bool,
    /// X360_TRACE_GPU_FROM=N / X360_TRACE_GPU_LEN=M: trace only packets N..N+M (counted from the start of the run).
    #[serde(skip)]
    pub trace_from: u64,
    #[serde(skip)]
    pub trace_len: u64,
    #[serde(skip)]
    zpd_logged: u32,
    /// Debug counters: ring dwords pulled from guest memory, packets run from the ring, packets run from IBs.
    #[serde(skip)]
    ib_logged: u32,
    #[serde(skip)]
    pred_logged: u32,
    #[serde(skip)]
    no_predication: bool,
    #[serde(skip)]
    pub predicated_skipped: u64,
    #[serde(skip)]
    ib_dump_logged: u32,
    #[serde(skip)]
    ring_logged: u32,
    #[serde(skip)]
    bin_logged: u32,
    /// Set by an INTERRUPT packet: stop pulling packets until the interrupt handler thread has finished.
    #[serde(skip)]
    pub hold: bool,
    #[serde(skip)]
    yield_after: bool,
    #[serde(skip)]
    no_irq_hold: bool,
    #[serde(skip)]
    pub dbg_ring_dwords: u64,
    #[serde(skip)]
    pub dbg_ring_packets: u64,
    #[serde(skip)]
    pub dbg_ib_packets: u64,
    /// Packets executed, by type-3 opcode, for the end-of-run summary.
    pub opcode_counts: HashMap<u32, u64>,
    /// Histogram of (opcode, first body dword, body length) for EVENT_WRITE / EVENT_WRITE_EXT / EVENT_WRITE_ZPD.
    pub event_kinds: HashMap<(u32, u32, usize), u64>,
    /// EVENT_WRITE_EXT result addresses: first few, and the histogram of strides between consecutive ones.
    pub ext_addrs: Vec<u32>,
    pub ext_strides: HashMap<i64, u64>,
    ext_last: u32,
    pub packets: u64,
    /// Draw calls seen (decoded and logged, not rasterized yet) and how many more to log.
    pub draws: u64,
    pub draw_log_budget: u32,
    pub copy_log_budget: u32,
    pub shader_log_budget: u32,
    // ---- resumable command-processor state ----
    /// Ring dwords pulled from guest memory but not yet consumed (starting at `rptr`).
    queue: VecDeque<u32>,
    /// Ring index up to which dwords have been pulled into `queue`.
    pulled: u32,
    /// Indirect-buffer frames currently being executed: (dwords, position).
    ib_stack: Vec<(Vec<u32>, usize)>,
    pending_ib: Option<Vec<u32>>,
    stall_request: bool,
    stall_counted: bool,
    /// True while the CP is blocked on an unsatisfied WAIT_REG_MEM (the run loop keeps polling).
    pub stalled: bool,
    /// CP interrupts (PM4_INTERRUPT cpu masks) waiting to be delivered to the title's callback.
    pub pending_interrupts: Vec<u32>,
    /// Tile-pass predication state set by SET_BIN_MASK_* / SET_BIN_SELECT_*.
    pub bin_mask: u64,
    pub bin_select: u64,
}

enum Step {
    Progress,
    Idle,
    Stalled,
}

impl Gpu {
    pub fn new() -> Self {
        let budget = std::env::var("X360_LOG_DRAWS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let copies = std::env::var("X360_LOG_COPIES").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        Self {
            trace: std::env::var_os("X360_TRACE_GPU").is_some(),
            trace_from: std::env::var("X360_TRACE_GPU_FROM").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
            trace_len: std::env::var("X360_TRACE_GPU_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(u64::MAX),
            zpd_logged: 0,
            ib_logged: 0,
            pred_logged: 0,
            no_predication: std::env::var_os("X360_NO_PREDICATION").is_some(),
            predicated_skipped: 0,
            ib_dump_logged: 0,
            ring_logged: 0,
            bin_logged: 0,
            hold: false,
            yield_after: false,
            no_irq_hold: std::env::var_os("X360_NO_IRQ_HOLD").is_some(),
            dbg_ring_dwords: 0,
            dbg_ring_packets: 0,
            dbg_ib_packets: 0,
            draw_log_budget: budget,
            copy_log_budget: copies,
            shader_log_budget: std::env::var("X360_LOG_SHADERS").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
            display_on: true,
            no_render: std::env::var_os("X360_NO_RENDER").is_some(),
            sample_every: std::env::var("X360_SAMPLE_DRAWS").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            snapshot: std::env::var("X360_SNAPSHOT").ok().and_then(|v| {
                let mut it = v.splitn(3, ',');
                Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?, it.next()?.to_string()))
            }),
            snapshot_draws: 0,
            window: None,
            ..Default::default()
        }
    }

    fn tracing(&self) -> bool {
        self.trace && self.packets >= self.trace_from && self.packets - self.trace_from < self.trace_len
    }

    /// After loading a save state: take the debug switches from this run's environment, not the saved run's.
    pub fn reload_env(&mut self) {
        let fresh = Gpu::new();
        self.trace = fresh.trace;
        self.trace_from = fresh.trace_from;
        self.trace_len = fresh.trace_len;
        self.no_irq_hold = fresh.no_irq_hold;
        self.no_predication = fresh.no_predication;
        self.no_render = fresh.no_render;
        self.sample_every = fresh.sample_every;
        self.snapshot = fresh.snapshot;
        self.snapshot_draws = 0;
        self.draw_log_budget = fresh.draw_log_budget;
        self.copy_log_budget = fresh.copy_log_budget;
        self.shader_log_budget = fresh.shader_log_budget;
    }

    pub fn ring_virtual(&self) -> u32 {
        PHYS_VIEW | self.ring_base
    }

    pub fn ring_size_bytes(&self) -> u32 {
        8u32 << self.ring_size_log2
    }

    pub fn init_ring(&mut self, base_phys: u32, size_log2: u32) {
        self.ring_base = base_phys;
        self.ring_size_log2 = size_log2;
        self.rptr = 0;
        self.pulled = 0;
        self.queue.clear();
        self.ib_stack.clear();
    }

    /// Total length in dwords (header included) of the packet starting with `hdr`.
    fn packet_len(hdr: u32) -> usize {
        match hdr >> 30 {
            1 => 3,
            2 => 1,
            _ => 2 + ((hdr >> 16) & 0x3FFF) as usize, // type 0 and 3: count+1 body dwords
        }
    }

    /// Pull newly written ring dwords, then run packets until the stream is empty or a wait stalls.
    /// Safe to call repeatedly: a stalled packet is re-evaluated on the next call.
    pub fn process(&mut self, mem: &mut GuestMemory) {
        if self.ring_base == 0 {
            return;
        }
        let mask = self.ring_size_bytes() / 4 - 1;
        let wptr = mem.gpu_regs[REG_CP_RB_WPTR] & mask;
        let base = self.ring_virtual();
        if self.pulled != wptr && self.dbg_ring_dwords == 0 && std::env::var_os("X360_LOG_IB").is_some() {
            println!("[gpu] ring at {base:#x}, {} dwords, wptr {wptr}, rptr writeback {:#x}", mask + 1, self.rptr_writeback);
        }
        while self.pulled != wptr {
            self.dbg_ring_dwords += 1;
            self.queue.push_back(mem.read32(base + self.pulled * 4));
            self.pulled = (self.pulled + 1) & mask;
        }
        let mut stalled = false;
        loop {
            match self.step(mem, mask) {
                Step::Progress => {
                    if std::mem::take(&mut self.yield_after) {
                        stalled = true;
                        break;
                    }
                }
                Step::Idle => break,
                Step::Stalled => {
                    stalled = true;
                    break;
                }
            }
        }
        self.stalled = stalled;
        mem.gpu_regs[REG_CP_RB_RPTR] = self.rptr;
        if self.rptr_writeback != 0 {
            mem.write32(PHYS_VIEW | self.rptr_writeback, self.rptr);
        }
    }

    /// Execute (at most) one packet from the innermost stream.
    fn step(&mut self, mem: &mut GuestMemory, mask: u32) -> Step {
        let (hdr, body, total, from_ring) = if let Some((words, pos)) = self.ib_stack.last() {
            if *pos >= words.len() {
                self.ib_stack.pop();
                return Step::Progress;
            }
            let hdr = words[*pos];
            let end = (*pos + Self::packet_len(hdr)).min(words.len());
            (hdr, words[*pos + 1..end].to_vec(), end - *pos, false)
        } else {
            if self.queue.is_empty() {
                return Step::Idle;
            }
            let s = self.queue.make_contiguous();
            let need = Self::packet_len(s[0]);
            if s.len() < need {
                return Step::Idle; // a partially written packet: wait for the rest
            }
            (s[0], s[1..need].to_vec(), need, true)
        };

        if from_ring && self.ring_logged < 150 && std::env::var_os("X360_LOG_RING").is_some() {
            self.ring_logged += 1;
            let vals: Vec<String> = body.iter().take(5).map(|v| format!("{v:#x}")).collect();
            if hdr >> 30 == 3 {
                println!("[ring] {} {}", opcode_name((hdr >> 8) & 0x7F), vals.join(","));
            } else if hdr >> 30 == 0 {
                println!("[ring] type0 reg={:#x} n={} {}", hdr & 0x7FFF, body.len(), vals.join(","));
            }
        }
        self.stall_request = false;
        self.pending_ib = None;
        self.exec_packet(mem, hdr, &body);
        if self.stall_request {
            return Step::Stalled;
        }
        if from_ring {
            self.dbg_ring_packets += 1;
        } else {
            self.dbg_ib_packets += 1;
        }
        if from_ring {
            self.queue.drain(..total);
            self.rptr = (self.rptr + total as u32) & mask;
        } else if let Some((_, pos)) = self.ib_stack.last_mut() {
            *pos += total;
        }
        if let Some(sub) = self.pending_ib.take() {
            if self.ib_stack.len() < 8 {
                self.ib_stack.push((sub, 0));
            }
        }
        self.packets += 1;
        Step::Progress
    }

    /// Write a GPU register on behalf of a command-stream packet. Scratch registers
    /// (SCRATCH_REG0..7 = 0x578..0x57F) are additionally mirrored to memory at
    /// SCRATCH_ADDR + 4*index when the matching SCRATCH_UMSK bit is set.
    fn write_reg(mem: &mut GuestMemory, reg: usize, value: u32) {
        if reg >= mem.gpu_regs.len() {
            return;
        }
        mem.gpu_regs[reg] = value;
        if (0x578..=0x57F).contains(&reg) {
            let idx = (reg - 0x578) as u32;
            if mem.gpu_regs[REG_SCRATCH_UMSK] & (1 << idx) != 0 {
                let base = mem.gpu_regs[REG_SCRATCH_ADDR] & 0x1FFF_FFFC;
                mem.write32(PHYS_VIEW | (base + idx * 4), value);
            }
        }
    }

    fn exec_packet(&mut self, mem: &mut GuestMemory, hdr: u32, body: &[u32]) {
        match hdr >> 30 {
            0 => {
                // Type 0: write the body dwords to consecutive (or one repeated) registers
                let base = (hdr & 0x7FFF) as usize;
                let one_reg = hdr & (1 << 15) != 0;
                for (k, &v) in body.iter().enumerate() {
                    Self::write_reg(mem, if one_reg { base } else { base + k }, v);
                }
                if self.tracing() {
                    let vals: Vec<String> = body.iter().take(4).map(|v| format!("{v:#x}")).collect();
                    println!("[gpu] type0 base={base:#x} count={} values={}", body.len(), vals.join(","));
                }
            }
            1 => {
                // Type 1: two register writes
                let (a, b) = ((hdr & 0x7FF) as usize, ((hdr >> 11) & 0x7FF) as usize);
                Self::write_reg(mem, a, body.first().copied().unwrap_or(0));
                Self::write_reg(mem, b, body.get(1).copied().unwrap_or(0));
            }
            2 => {} // type 2: filler
            _ => {
                let op = (hdr >> 8) & 0x7F;
                // Predicated tiling: a type-3 packet with the predicate bit (header bit 0) only runs when the current
                // tile (BIN_SELECT) intersects the packet's BIN_MASK. The driver's per-tile epilogue (scratch callback,
                // waits and the INTERRUPT) is masked to the first tile only, so one interrupt fires per frame.
                if hdr & 1 != 0 && self.pred_logged < 3000 && op == 0x54 && std::env::var_os("X360_LOG_IB").is_some() {
                    self.pred_logged += 1;
                    println!("[gpu] predicated {} hdr={hdr:#x} mask={:#x} select={:#x}", opcode_name(op), self.bin_mask, self.bin_select);
                }
                if hdr & 1 != 0 && !self.no_predication {
                    let (m, sel) = if self.bin_mask == 0 && self.bin_select == 0 { (u64::MAX, u64::MAX) } else { (self.bin_mask, self.bin_select) };
                    if m & sel == 0 {
                        self.predicated_skipped += 1;
                        return;
                    }
                }
                if !self.stall_counted {
                    *self.opcode_counts.entry(op).or_insert(0) += 1;
                    if self.tracing() || self.opcode_counts[&op] == 1 {
                        println!("[gpu] type3 {:#04x} {} ({} dwords)", op, opcode_name(op), body.len());
                    }
                }
                if matches!(op, 0x46 | 0x47 | 0x5A) && !self.stall_counted {
                    *self.event_kinds.entry((op, body.first().copied().unwrap_or(0), body.len())).or_insert(0) += 1;
                }
                self.exec_type3(mem, op, body, 0);
            }
        }
    }

    fn exec_bin(&mut self, op: u32, body: &[u32]) {
        let v = body.first().copied().unwrap_or(0) as u64;
        match op {
            0x60 => self.bin_mask = (self.bin_mask & !0xFFFF_FFFF) | v,
            0x61 => self.bin_mask = (self.bin_mask & 0xFFFF_FFFF) | v << 32,
            0x62 => self.bin_select = (self.bin_select & !0xFFFF_FFFF) | v,
            _ => self.bin_select = (self.bin_select & 0xFFFF_FFFF) | v << 32,
        }
    }

    /// The comparison used by WAIT_REG_MEM and COND_WRITE (function code 0..7).
    fn compare(func: u32, value: u32, reference: u32) -> bool {
        match func & 7 {
            0 => false,
            1 => value < reference,
            2 => value <= reference,
            3 => value == reference,
            4 => value != reference,
            5 => value >= reference,
            6 => value > reference,
            _ => true,
        }
    }

    /// Read the poll target of WAIT_REG_MEM / COND_WRITE: guest memory (physical, with endian bits
    /// in the low two address bits) or a GPU register.
    fn poll_value(mem: &GuestMemory, poll_memory: bool, addr: u32) -> u32 {
        if poll_memory {
            crate::xenos_draw::apply_endian(mem.read32(PHYS_VIEW | (addr & 0x1FFF_FFFC)), addr & 3)
        } else {
            mem.gpu_regs.get((addr & 0x7FFF) as usize).copied().unwrap_or(0)
        }
    }

    /// Register-file base for the SET_CONSTANT "type" field.
    fn constant_base(kind: u32) -> Option<usize> {
        Some(match kind {
            0 => 0x4000, // ALU (float) constants
            1 => 0x4800, // texture / vertex fetch constants
            2 => 0x4900, // bool constants
            3 => 0x4908, // loop constants
            4 => 0x2000, // plain registers
            _ => return None,
        })
    }

    fn exec_type3(&mut self, mem: &mut GuestMemory, op: u32, body: &[u32], depth: u32) {
        match op {
            0x10 | 0x48 => {} // NOP, ME_INIT: nothing to do for an HLE command processor
            0x2B | 0x27 => {
                // IM_LOAD_IMMEDIATE: [type, size in dwords, ucode dwords...]
                // IM_LOAD:           [address | type, size] (ucode read from memory)
                // Either way the low bits of the first dword select vertex (0) or pixel (1) shader.
                let kind = if op == 0x2B { "IM_LOAD_IMMEDIATE" } else { "IM_LOAD" };
                if body.len() >= 2 {
                    let shader_kind = if body[0] & 3 == 0 { ShaderKind::Vertex } else { ShaderKind::Pixel };
                    let size = (body[1] & 0xFFF) as usize;
                    let ucode: Vec<u32> = if op == 0x2B {
                        body.iter().skip(2).take(size).copied().collect()
                    } else {
                        let addr = PHYS_VIEW | (body[0] & 0x1FFF_FFFC & !3);
                        (0..size as u32).map(|i| mem.read32(addr + i * 4)).collect()
                    };
                    self.renderer.load_shader(shader_kind, ucode);
                }
                if self.shader_log_budget > 0 && body.len() >= 2 {
                    self.shader_log_budget -= 1;
                    let words: Vec<String> = body.iter().skip(2).take(12).map(|w| format!("{w:08x}")).collect();
                    println!(
                        "[gpu] {kind}: hdr={:#010x} size={:#x} dwords ({} in packet) ucode: {}",
                        body[0], body[1], body.len() - 2, words.join(" ")
                    );
                    if op == 0x2B {
                        print!("{}", crate::xenos_vm::disassemble(&body[2..]));
                    }
                }
            }
            0x2D => {
                // SET_CONSTANT: [type << 16 | index, values...] -> regs[base(type) + index + i]
                if let Some(&hdr) = body.first() {
                    if let Some(base) = Self::constant_base((hdr >> 16) & 0xF) {
                        let start = base + (hdr & 0x7FF) as usize;
                        for (i, &v) in body[1..].iter().enumerate() {
                            if start + i < mem.gpu_regs.len() {
                                mem.gpu_regs[start + i] = v;
                            }
                        }
                    }
                }
            }
            0x55 => {
                // SET_CONSTANT2: [absolute register index, values...]
                if let Some(&start) = body.first() {
                    for (i, &v) in body[1..].iter().enumerate() {
                        let r = (start & 0xFFFF) as usize + i;
                        if r < mem.gpu_regs.len() {
                            mem.gpu_regs[r] = v;
                        }
                    }
                }
            }
            0x2F => {
                // LOAD_ALU_CONSTANT: [source address | endian << 30, type << 16 | index, size in dwords]
                if body.len() >= 3 {
                    let addr = PHYS_VIEW | (body[0] & 0x1FFF_FFFC);
                    let swap = body[0] >> 30;
                    if let Some(base) = Self::constant_base((body[1] >> 16) & 0xFF) {
                        let start = base + (body[1] & 0x7FF) as usize;
                        for i in 0..(body[2] & 0xFFF) as usize {
                            let raw = mem.read32(addr + i as u32 * 4);
                            let v = if swap == 2 { raw } else { raw.swap_bytes() };
                            if start + i < mem.gpu_regs.len() {
                                mem.gpu_regs[start + i] = v;
                            }
                        }
                    }
                }
            }
            0x22 | 0x36 => self.draw(mem, op, body),
            0x3F | 0x37 => {
                // INDIRECT_BUFFER(_PFD): [physical address, size in dwords]. The sub-stream is pushed
                // as a new frame so a stall inside it resumes exactly where it stopped.
                if body.len() >= 2 {
                    let (addr, size) = (PHYS_VIEW | body[0], body[1] & 0xF_FFFF);
                    if self.packets >= self.trace_from && self.ib_logged < 400 && std::env::var_os("X360_LOG_IB").is_some() {
                        self.ib_logged += 1;
                        println!("[gpu] IB addr={:#x} size={} depth={} select={:#x} mask={:#x}", body[0], size, self.ib_stack.len(), self.bin_select, self.bin_mask);
                    }
                    let words: Vec<u32> = (0..size).map(|k| mem.read32(addr + k * 4)).collect();
                    if size <= 40 && self.ib_dump_logged < 60 && std::env::var_os("X360_LOG_IB").is_some() {
                        self.ib_dump_logged += 1;
                        let hex: Vec<String> = words.iter().map(|w| format!("{w:08x}")).collect();
                        println!("[gpu] IB {:#x} contents: {}", body[0], if body[0] == 0x175d_b800 { format!("mask={} cb={} arg={} flagwait={} {}", hex[1], hex[5], hex[6], hex[9], hex[24]) } else { hex.join(" ") });
                    }
                    self.pending_ib = Some(words);
                }
            }
            0x3C => {
                // WAIT_REG_MEM: [wait info (func [2:0], 1 = poll memory [4]), poll address / register,
                //                reference, mask, wait interval]. Stalls the CP until satisfied.
                if body.len() >= 4 {
                    let value = Self::poll_value(mem, body[0] & 0x10 != 0, body[1]) & body[3];
                    if !Self::compare(body[0], value, body[2]) {
                        if !self.stall_counted && std::env::var_os("X360_TRACE_GPU_WAIT").is_some() {
                            println!(
                                "[gpu] WAIT_REG_MEM stalls: info={:#x} poll={:#x} ({}) current(masked)={:#x} ref={:#x} mask={:#x}",
                                body[0], body[1], if body[0] & 0x10 != 0 { "memory" } else { "register" }, value, body[2], body[3]
                            );
                        }
                        self.stall_request = true;
                        self.stall_counted = true;
                        return;
                    }
                    self.stall_counted = false;
                }
            }
            0x45 => {
                // COND_WRITE: [wait info (func, poll-memory [4], write-memory [8]), poll address, reference,
                //              mask, write address, write data]
                if body.len() >= 6 {
                    let value = Self::poll_value(mem, body[0] & 0x10 != 0, body[1]) & body[3];
                    if Self::compare(body[0], value, body[2]) {
                        if body[0] & 0x100 != 0 {
                            let data = crate::xenos_draw::encode_endian(body[5], body[4] & 3);
                            mem.write32(PHYS_VIEW | (body[4] & 0x1FFF_FFFC), data);
                        } else if let Some(r) = mem.gpu_regs.get_mut((body[4] & 0x7FFF) as usize) {
                            *r = body[5];
                        }
                    }
                }
            }
            0x54 => {
                // INTERRUPT: [cpu mask]. Raises a CP interrupt, delivered to the title's graphics
                // callback on the CPU core(s) selected by the mask.
                self.pending_interrupts.push(body.first().copied().unwrap_or(1));
                // The ISR reads state the stream is about to overwrite (scratch registers) and releases the
                // waits that follow, so the CP must not run on until it has run.
                if !self.no_irq_hold {
                    self.yield_after = true;
                    self.hold = true;
                }
            }
            0x62 | 0x63 if self.bin_logged < 400 && std::env::var_os("X360_LOG_IB").is_some() && (self.ib_stack.is_empty() || (self.bin_select != u64::MAX && self.bin_select != 0xFFFF_FFFF)) => {
                self.bin_logged += 1;
                println!("[gpu] {} {:#x} (ib depth {})", opcode_name(op), body.first().copied().unwrap_or(0), self.ib_stack.len());
                self.exec_bin(op, body);
            }
            0x60..=0x63 => self.exec_bin(op, body),
            0x64 => {
                // XE_SWAP: ['SWAP', texture fetch constant (6 dwords) of the front buffer, ...]
                if body.len() >= 7 && body[0] == 0x5041_5753 {
                    let (w0, w1, w2) = (body[1], body[2], body[3]);
                    let frame = FrontBuffer {
                        base_phys: w1 & 0xFFFF_F000,
                        width: (w2 & 0x1FFF) + 1,
                        height: ((w2 >> 13) & 0x1FFF) + 1,
                        pitch: ((w0 >> 22) & 0x1FF) * 32,
                        format: w1 & 0x3F,
                        endian: (w1 >> 6) & 3,
                        tiled: w0 >> 31 != 0,
                    };
                    self.frames += 1;
                    if let Some((f, _, _)) = &self.snapshot {
                        if *f == self.frames {
                            self.no_render = false;
                        }
                    }
                    // X360_TRACE_GPU_AT_FRAME=N: start the packet trace once frame N has been presented
                    if std::env::var("X360_TRACE_GPU_AT_FRAME").ok().and_then(|v| v.parse::<u64>().ok()) == Some(self.frames) {
                        self.trace = true;
                    }
                    println!(
                        "[gpu] SWAP #{}: front buffer phys {:#010x} {}x{} pitch {} fmt {} endian {} tiled {}",
                        self.frames, frame.base_phys, frame.width, frame.height, frame.pitch, frame.format, frame.endian, frame.tiled
                    );
                    if !self.display_on {
                        println!("[gpu] display is off: frame not shown");
                        return;
                    }
                    self.front_buffer = Some(frame);
                    if let Some(win) = &self.window {
                        let rgba = frame.read_rgba(mem);
                        let pixels = rgba.chunks_exact(4).map(|p| (p[0] as u32) << 16 | (p[1] as u32) << 8 | p[2] as u32).collect();
                        win.publish(crate::window::Frame { width: frame.width as usize, height: frame.height as usize, pixels });
                    }
                    // X360_DUMP_FRAMES="dir" writes presented frames as dir/frame_NNNN.png (plus dir/latest.png).
                    // X360_DUMP_EVERY=N keeps only every Nth frame (default 1); X360_DUMP_BMP=1 also writes .bmp.
                    if let Ok(dir) = std::env::var("X360_DUMP_FRAMES") {
                        let every: u64 = std::env::var("X360_DUMP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
                        if self.frames % every == 0 {
                            let _ = std::fs::create_dir_all(&dir);
                            let rgba = frame.read_rgba(mem);
                            let non_black = rgba.chunks(4).filter(|p| p[0] | p[1] | p[2] != 0).count();
                            let path = format!("{dir}/frame_{:04}.png", self.frames);
                            let _ = write_png(&path, frame.width, frame.height, &rgba);
                            let _ = std::fs::copy(&path, format!("{dir}/latest.png"));
                            if std::env::var_os("X360_DUMP_BMP").is_some() {
                                let _ = write_bmp(&path.replace(".png", ".bmp"), frame.width, frame.height, &rgba);
                            }
                            println!("[gpu] wrote {path} ({non_black} non-black pixels of {})", frame.width * frame.height);
                        }
                    }
                }
            }
            0x46 | 0x47 => {
                // EVENT_WRITE / EVENT_WRITE_ZPD. A visibility-query end (VIZQUERY_END = 6) or z-pass done (0x16)
                // makes the render backend store its sample counters at RB_SAMPLE_COUNT_ADDR (reg 0x2322): a
                // begin/end pair of 64-bit counts. We count nothing, so report "plenty of samples passed".
                let init = body.first().copied().unwrap_or(0) & 0x3F;
                if op == 0x47 || init == 6 || init == 0x16 {
                    let raw = mem.gpu_regs[0x2322];
                    if self.zpd_logged < 4 || std::env::var_os("X360_LOG_ZPD").is_some() && self.zpd_logged < 40 {
                        self.zpd_logged += 1;
                        println!("[gpu] sample-count write op={op:#x} init={init:#x} RB_SAMPLE_COUNT_ADDR={raw:#x}");
                    }
                    let a = raw & 0x1FFF_FFF8;
                    if a != 0 {
                        let base = PHYS_VIEW | a;
                        // begin (u64) = 0, end (u64) = 0x100, big-endian as the title reads it
                        for (i, v) in [0u32, 0, 0, 0x100].iter().enumerate() {
                            mem.write32(base + i as u32 * 4, *v);
                        }
                    }
                }
            }
            0x5A => {
                // EVENT_WRITE_EXT: [event initiator, address (low 2 bits = endian)]. The extended events report
                // pipeline statistics - for the visibility/z-pass query (0x1A and friends) the GPU stores a begin/end
                // pair of sample counters. We have no real counters, so report "plenty of samples passed": a title
                // polling for the result then sees it complete, and visibility culling keeps everything visible.
                if body.len() >= 2 {
                    let a = body[1] & !3;
                    if self.ext_addrs.len() < 12 {
                        self.ext_addrs.push(a);
                    }
                    if self.ext_last != 0 {
                        *self.ext_strides.entry(a as i64 - self.ext_last as i64).or_insert(0) += 1;
                    }
                    self.ext_last = a;
                    let endian = body[1] & 3;
                    let base = PHYS_VIEW | (body[1] & 0x1FFF_FFFC);
                    // The titles' result records are 16 bytes apart with the result in the last 8 (GTA IV
                    // reserves exactly two dwords), so write exactly two: begin = 0, end = 0x100 samples.
                    let vals: [u32; 2] = std::env::var("X360_EXT")
                        .ok()
                        .and_then(|v| {
                            let (a, b) = v.split_once(',')?;
                            Some([u32::from_str_radix(a, 16).ok()?, u32::from_str_radix(b, 16).ok()?])
                        })
                        .unwrap_or([0, 0x100]);
                    for (i, c) in vals.iter().enumerate() {
                        mem.write32(base + i as u32 * 4, crate::xenos_draw::encode_endian(*c, endian));
                    }
                }
            }
            0x58 => {
                // EVENT_WRITE_SHD: [initiator, physical address (low bits = endian), value]
                // The title polls this location as a fence; the CP "completes" instantly, so write now.
                if body.len() >= 3 {
                    mem.write32(PHYS_VIEW | (body[1] & 0x1FFF_FFFC), crate::xenos_draw::encode_endian(body[2], body[1] & 3));
                }
            }
            0x21 => {
                // REG_RMW: [reg | flags, and mask, or mask]
                if body.len() >= 3 {
                    let reg = (body[0] & 0x1FFF) as usize;
                    if reg < mem.gpu_regs.len() {
                        mem.gpu_regs[reg] = (mem.gpu_regs[reg] & body[1]) | body[2];
                    }
                }
            }
            _ => {} // logged on first sight; behaviour comes as the game needs it
        }
    }

    /// DRAW_INDX (0x22: [viz query, initiator, ...]) and DRAW_INDX_2 (0x36: [initiator, ...]).
    /// For now draws are decoded and logged, not rasterized.
    fn draw(&mut self, mem: &mut GuestMemory, op: u32, body: &[u32]) {
        let initiator = if op == 0x22 { body.get(1) } else { body.first() }.copied().unwrap_or(0);
        let prim = initiator & 0x3F;
        let src_sel = (initiator >> 6) & 3; // 0 = DMA indices, 1 = immediate, 2 = auto-index
        let count = initiator >> 16;
        let index32 = (initiator >> 11) & 1 != 0;
        self.draws += 1;
        let mode = mem.gpu_regs[REG_RB_MODECONTROL] & 7;
        // X360_SAMPLE_DRAWS=N: print one line for every Nth draw (shader hashes, count, target) to tell a repeating
        // loop from steady progress
        if self.sample_every != 0 && self.draws % self.sample_every == 0 {
            let h = |u: &Option<Vec<u32>>| u.as_ref().map(|v| v.iter().fold(0xcbf29ce484222325u64, |a, w| (a ^ *w as u64).wrapping_mul(0x100000001b3))).unwrap_or(0);
            println!(
                "[draw #{}] prim {} src {} count {} mode {} vs {:016x} ps {:016x} colorinfo {:#x} edram_base {:#x} scissor {:#x}/{:#x}",
                self.draws, prim, src_sel, count, mode, h(&self.renderer.vs), h(&self.renderer.ps),
                mem.gpu_regs[REG_RB_COLOR_INFO], mem.gpu_regs[REG_RB_COLOR_INFO] & 0xFFF,
                mem.gpu_regs[0x2081], mem.gpu_regs[0x2082]
            );
        }

        // ---- gather the index list ----
        let indices: Vec<u32> = match src_sel {
            2 => (0..count).collect(),
            1 => {
                // immediate indices follow the initiator in the packet (two 16-bit per dword, high half first)
                let data = if op == 0x22 { body.get(2..) } else { body.get(1..) }.unwrap_or(&[]);
                if index32 {
                    data.iter().take(count as usize).copied().collect()
                } else {
                    data.iter().flat_map(|w| [w >> 16, w & 0xFFFF]).take(count as usize).collect()
                }
            }
            _ => {
                // DMA from memory: [initiator, base address (physical), size in dwords | endian << 30]
                let (base_idx, size_idx) = if op == 0x22 { (2, 3) } else { (1, 2) };
                let (base, size_endian) = (body.get(base_idx).copied().unwrap_or(0), body.get(size_idx).copied().unwrap_or(0));
                let addr = PHYS_VIEW | (base & 0x1FFF_FFFC);
                let endian = size_endian >> 30;
                (0..count)
                    .map(|i| {
                        if index32 {
                            crate::xenos_draw::apply_endian(mem.read32(addr + i * 4), endian)
                        } else {
                            let w = crate::xenos_draw::apply_endian(mem.read32(addr + (i / 2) * 4), endian);
                            if i % 2 == 0 { w & 0xFFFF } else { w >> 16 }
                        }
                    })
                    .collect()
            }
        };

        // ---- execute ----
        let regs = std::mem::take(&mut mem.gpu_regs);
        match mode {
            // X360_NO_RENDER=1 skips all pixel work (command stream still decoded): much faster
            // when only CPU / kernel progress matters. Frames come out black.
            4 if !self.no_render => {
                self.renderer.draw(mem, &regs, prim, &indices);
                if let Some((_, n, dir)) = self.snapshot.clone() {
                    self.snapshot_draws += 1;
                    if self.snapshot_draws == n {
                        let _ = std::fs::create_dir_all(&dir);
                        for (key, surf) in &self.renderer.surfaces {
                            let rgba: Vec<u8> = surf.px.iter().flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8)).collect();
                            let _ = write_png(&format!("{dir}/surface_{key:03x}.png"), surf.width, surf.height, &rgba);
                        }
                        println!("[gpu] snapshot: wrote {} render target(s) to {dir} after {n} draws", self.renderer.surfaces.len());
                        self.no_render = true;
                        self.snapshot = None;
                    }
                }
            }
            5 if !self.no_render => self.renderer.resolve(mem, &regs),
            _ => {}
        }
        mem.gpu_regs = regs;
        let r = |i: usize| mem.gpu_regs[i];
        // Copies (edram_mode 5) are the resolves we most need to understand: always log those.
        if mode == 5 && self.copy_log_budget > 0 {
            self.copy_log_budget -= 1;
            println!(
                "[gpu] COPY at draw #{}: {} x{} | color_info={:#x} surface={:#x} | ctl={:#x} dest_base={:#x} dest_pitch={:#x} dest_info={:#x} slice={:#x} | clear color={:#x} depth={:#x} | vport scale={:.1}x{:.1} off={:.1},{:.1}",
                self.draws, prim_name(prim), count,
                r(REG_RB_COLOR_INFO), r(REG_RB_SURFACE_INFO),
                r(REG_RB_COPY_CONTROL), r(REG_RB_COPY_DEST_BASE), r(0x231A), r(0x231B), r(0x2323),
                r(0x231E), r(0x231D),
                f32::from_bits(r(0x210F)), f32::from_bits(r(0x2111)), f32::from_bits(r(0x2110)), f32::from_bits(r(0x2112)),
            );
        } else if self.draw_log_budget > 0 {
            self.draw_log_budget -= 1;
            println!(
                "[gpu] draw #{}: {} x{} src_sel={} mode={} bin={:#x}/{:#x} color_info={:#x} surface={:#x} blend={:#x} mask={:#x} | vport scale={:.1}x{:.1} off={:.1},{:.1} | scissor={:#x}-{:#x} | cl_vte={:#x}",
                self.draws, prim_name(prim), count, src_sel, mode, self.bin_mask, self.bin_select,
                r(REG_RB_COLOR_INFO), r(REG_RB_SURFACE_INFO), r(0x2201), r(0x2104),
                f32::from_bits(r(0x210F)), f32::from_bits(r(0x2111)), f32::from_bits(r(0x2110)), f32::from_bits(r(0x2112)),
                r(0x2081), r(0x2082), r(0x2206),
            );
        }
    }

    pub fn summary(&self) -> String {
        let mut ops: Vec<_> = self.opcode_counts.iter().collect();
        ops.sort();
        let list: Vec<String> = ops.iter().map(|(op, n)| format!("{}({:#x})x{}", opcode_name(**op), op, n)).collect();
        let mut ev: Vec<_> = self.event_kinds.iter().collect();
        ev.sort_by(|a, b| b.1.cmp(a.1));
        let ev_list: Vec<String> = ev
            .iter()
            .take(8)
            .map(|((op, init, len), n)| format!("{}(first dword {:#x}, {} dwords)x{}", opcode_name(*op), init, len, n))
            .collect();
        let mut st: Vec<_> = self.ext_strides.iter().collect();
        st.sort_by(|a, b| b.1.cmp(a.1));
        let st_list: Vec<String> = st.iter().take(8).map(|(d, n)| format!("{d:+#x}x{n}")).collect();
        let addr_list: Vec<String> = self.ext_addrs.iter().map(|a| format!("{a:#x}")).collect();
        format!(
            "gpu ext-event addresses: {} ; strides: {}
gpu events: {}
gpu: {} packets ({} predicated-off, {} from ring, {} from IBs, {} ring dwords pulled); ring rptr={} pulled={} queued={} ib_depth={} stalled={}; type-3 opcodes: {}\n{}",
            addr_list.join(" "),
            st_list.join(" "),
            ev_list.join("; "),
            self.packets,
            self.predicated_skipped,
            self.dbg_ring_packets,
            self.dbg_ib_packets,
            self.dbg_ring_dwords,
            self.rptr,
            self.pulled,
            self.queue.len(),
            self.ib_stack.len(),
            self.stalled,
            list.join(", "),
            self.renderer.summary()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_with_ring() -> (Gpu, GuestMemory) {
        let mut mem = GuestMemory::new();
        mem.map(PHYS_VIEW | 0x0100_0000, 0x10000);
        let mut g = Gpu::new();
        g.init_ring(0x0100_0000, 8); // 8 << 8 = 2 KB = 512 dwords
        (g, mem)
    }

    #[test]
    fn tiled_offsets_are_a_bijection_within_a_tile_grid() {
        // Every pixel of a 64x64 surface must map to a unique 4-byte slot within 64*64*4 bytes.
        let mut seen = std::collections::HashSet::new();
        for y in 0..64 {
            for x in 0..64 {
                let off = tiled_offset_2d(x, y, 64, 2);
                assert_eq!(off % 4, 0);
                assert!(off < 64 * 64 * 4, "({x},{y}) -> {off}");
                assert!(seen.insert(off), "({x},{y}) collides at {off}");
            }
        }
        assert_eq!(tiled_offset_2d(0, 0, 64, 2), 0);
    }

    #[test]
    fn swap_while_display_off_is_not_shown() {
        let (mut g, mut mem) = gpu_with_ring();
        g.display_on = false;
        let base = g.ring_virtual();
        mem.write32(base, 0xC03E_6400);
        mem.write32(base + 4, 0x5041_5753);
        mem.write32(base + 8, (40 << 22) | (1 << 31));
        mem.write32(base + 12, 0x1A00_0000 | 6);
        mem.write32(base + 16, (719 << 13) | 1279);
        mem.gpu_regs[REG_CP_RB_WPTR] = 64;
        g.process(&mut mem);
        assert_eq!(g.frames, 1, "the swap still executes");
        assert!(g.front_buffer.is_none(), "but nothing reaches the screen");
    }

    #[test]
    fn xe_swap_decodes_front_buffer() {
        let (mut g, mut mem) = gpu_with_ring();
        let base = g.ring_virtual();
        // type 3, opcode 0x64, 63 body dwords; fetch constant: pitch 1280/32 = 40 in w0[30:22]
        mem.write32(base, 0xC03E_6400);
        mem.write32(base + 4, 0x5041_5753);
        mem.write32(base + 8, (40 << 22) | (1 << 31)); // w0: pitch, tiled
        mem.write32(base + 12, 0x1A00_0000 | 6); // w1: base 0x1A000000, format 6
        mem.write32(base + 16, (719 << 13) | 1279); // w2: 1280x720
        mem.gpu_regs[REG_CP_RB_WPTR] = 64;
        g.process(&mut mem);
        let f = g.front_buffer.expect("a frame was presented");
        assert_eq!((f.width, f.height, f.pitch), (1280, 720, 1280));
        assert_eq!(f.base_phys, 0x1A00_0000);
        assert_eq!(f.format, 6);
        assert!(f.tiled);
        assert_eq!(g.frames, 1);
    }

    #[test]
    fn wait_reg_mem_stalls_until_memory_changes_then_resumes() {
        let (mut g, mut mem) = gpu_with_ring();
        let base = g.ring_virtual();
        let flag = 0x0100_9000u32; // physical address of the flag word
        mem.map(PHYS_VIEW | flag, 0x1000);
        // WAIT_REG_MEM: poll memory, func == (3), ref = 7, mask = 0xFF, endian bits = 2 (8in32)
        mem.write32(base, 0xC004_3C00); // type 3, 5 body dwords, opcode 0x3C
        mem.write32(base + 4, 0x13); // func 3 (equal) | poll memory
        mem.write32(base + 8, flag | 2);
        mem.write32(base + 12, 7);
        mem.write32(base + 16, 0xFF);
        mem.write32(base + 20, 10);
        // a type-0 register write follows the wait
        mem.write32(base + 24, 0x3000);
        mem.write32(base + 28, 0xABCD);
        mem.gpu_regs[REG_CP_RB_WPTR] = 8;
        g.process(&mut mem);
        assert!(g.stalled);
        assert_eq!(g.rptr, 0); // nothing consumed
        assert_eq!(mem.gpu_regs[0x3000], 0); // the register write after the wait hasn't run
        mem.write32(PHYS_VIEW | flag, 7); // the "CPU" releases the wait
        g.process(&mut mem);
        assert!(!g.stalled);
        assert_eq!(mem.gpu_regs[0x3000], 0xABCD);
        assert_eq!(g.rptr, 8);
    }

    #[test]
    fn scratch_register_write_mirrors_to_memory() {
        let (mut g, mut mem) = gpu_with_ring();
        let base = g.ring_virtual();
        let scratch = 0x0100_9000u32;
        mem.map(PHYS_VIEW | scratch, 0x1000);
        mem.gpu_regs[REG_SCRATCH_ADDR] = scratch;
        mem.gpu_regs[REG_SCRATCH_UMSK] = 0b10; // only SCRATCH_REG1 mirrors
        // type 0: write 1 to reg 0x579 (SCRATCH_REG1) and 9 to 0x578 (SCRATCH_REG0): two separate packets
        mem.write32(base, 0x0000_0579);
        mem.write32(base + 4, 1);
        mem.write32(base + 8, 0x0000_0578);
        mem.write32(base + 12, 9);
        mem.gpu_regs[REG_CP_RB_WPTR] = 4;
        g.process(&mut mem);
        assert_eq!(mem.read32(PHYS_VIEW | (scratch + 4)), 1, "enabled scratch register mirrors");
        assert_eq!(mem.read32(PHYS_VIEW | scratch), 0, "masked-off register does not");
    }

    #[test]
    fn cond_write_and_interrupt_and_bin_state() {
        let (mut g, mut mem) = gpu_with_ring();
        let base = g.ring_virtual();
        let out = 0x0100_9000u32;
        mem.map(PHYS_VIEW | out, 0x1000);
        // COND_WRITE: poll register 0x3100 (== 5), write 0x77 to memory `out`
        mem.gpu_regs[0x3100] = 5;
        mem.write32(base, 0xC005_4500); // 6 body dwords, opcode 0x45
        mem.write32(base + 4, 0x103); // func 3 (equal), write to memory
        mem.write32(base + 8, 0x3100);
        mem.write32(base + 12, 5);
        mem.write32(base + 16, 0xFF);
        mem.write32(base + 20, out | 2);
        mem.write32(base + 24, 0x77);
        // INTERRUPT (opcode 0x54), 1 body dword
        mem.write32(base + 28, 0xC000_5400);
        mem.write32(base + 32, 1);
        // SET_BIN_MASK_LO / SELECT_HI
        mem.write32(base + 36, 0xC000_6000);
        mem.write32(base + 40, 0xF);
        mem.write32(base + 44, 0xC000_6300);
        mem.write32(base + 48, 0x3);
        mem.gpu_regs[REG_CP_RB_WPTR] = 13;
        g.process(&mut mem);
        assert_eq!(mem.read32(PHYS_VIEW | out), 0x77);
        assert_eq!(g.pending_interrupts, vec![1]);
        assert_eq!(g.bin_mask, 0xF);
        assert_eq!(g.bin_select, 0x3_0000_0000);
    }

    #[test]
    fn ring_geometry() {
        let mut g = Gpu::default();
        g.init_ring(0x1701_0000, 0xC);
        assert_eq!(g.ring_virtual(), 0xB701_0000);
        assert_eq!(g.ring_size_bytes(), 0x8000); // 32 KB
    }

    #[test]
    fn type0_writes_registers_and_rptr_advances() {
        let (mut g, mut mem) = gpu_with_ring();
        let base = g.ring_virtual();
        // type-0: base reg 0x2000, count 2 -> hdr = (1<<16) | 0x2000
        mem.write32(base, (1 << 16) | 0x2000);
        mem.write32(base + 4, 0xAAAA);
        mem.write32(base + 8, 0xBBBB);
        mem.gpu_regs[REG_CP_RB_WPTR] = 3;
        g.process(&mut mem);
        assert_eq!(mem.gpu_regs[0x2000], 0xAAAA);
        assert_eq!(mem.gpu_regs[0x2001], 0xBBBB);
        assert_eq!(g.rptr, 3);
        assert_eq!(mem.gpu_regs[REG_CP_RB_RPTR], 3);
    }

    #[test]
    fn indirect_buffer_and_wraparound() {
        let (mut g, mut mem) = gpu_with_ring();
        let size_dw = g.ring_size_bytes() / 4;
        g.rptr = size_dw - 2; // start two dwords before the end so the packet wraps
        g.pulled = g.rptr;
        let base = g.ring_virtual();
        // type-3 INDIRECT_BUFFER at the end: header + [addr, size]
        let ib_phys = 0x0100_8000;
        mem.write32(base + (size_dw - 2) * 4, 0xC001_3F00); // type 3, count 2, opcode 0x3F
        mem.write32(base + (size_dw - 1) * 4, ib_phys);
        mem.write32(base, 2); // size, wrapped to index 0
        // the indirect buffer holds a type-0 packet writing register 0x3000
        mem.write32(PHYS_VIEW | ib_phys, 0x3000);
        mem.write32((PHYS_VIEW | ib_phys) + 4, 0x1234);
        mem.gpu_regs[REG_CP_RB_WPTR] = 1;
        g.process(&mut mem);
        assert_eq!(mem.gpu_regs[0x3000], 0x1234);
        assert_eq!(g.rptr, 1);
    }
}
