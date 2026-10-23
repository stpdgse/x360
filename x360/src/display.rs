//! Display-output hardware state behind the Vd* kernel calls: video mode, the scan-out switch,
//! scaler setup, EDRAM calibration. Nothing here drives a real panel, but the state is tracked
//! so the title sees consistent answers and so a frame is only "presented" while the output is on.
//!
//! Argument layouts for the configuration calls are not documented anywhere official; the raw
//! register values are recorded (see `last_*`) rather than guessed at, and only the fields we
//! are confident of (on/off, refresh, persisted buffer) drive behaviour.

use crate::kernel::Kernel;
use crate::memory::GuestMemory;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

pub struct DisplayState {
    pub width: u32,
    pub height: u32,
    pub interlaced: bool,
    pub widescreen: bool,
    pub hd: bool,
    pub refresh_hz: f32,
    /// Is the scan-out enabled? (VdTurnDisplayOn/Off.) Frames are only shown while it is.
    pub on: bool,
    /// Raw arguments of the last VdSetDisplayMode / VdSetDisplayModeOverride / VdInitializeScaler.
    pub last_mode_args: [u32; 4],
    pub last_override_args: [u32; 4],
    pub last_scaler_args: [u32; 4],
    pub mode_sets: u32,
    pub scaler_inits: u32,
    pub scaler_buffer_builds: u32,
    pub edram_retrains: u32,
    /// Front-buffer address the title asked to keep on screen across a reload (VdPersistDisplay).
    pub persisted_front_buffer: u32,
    pub cgms: u32,
    pub wss: u32,
    pub closed_captions: bool,
}

impl DisplayState {
    pub fn new() -> Self {
        Self {
            width: 1280,
            height: 720,
            interlaced: false,
            widescreen: true,
            hd: true,
            refresh_hz: 60.0,
            on: true,
            last_mode_args: [0; 4],
            last_override_args: [0; 4],
            last_scaler_args: [0; 4],
            mode_sets: 0,
            scaler_inits: 0,
            scaler_buffer_builds: 0,
            edram_retrains: 0,
            persisted_front_buffer: 0,
            cgms: 0,
            wss: 0,
            closed_captions: false,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "display: {}x{}@{}Hz {}, output {}; mode sets {}, scaler inits {}, scaler buffers {}, EDRAM retrains {}",
            self.width,
            self.height,
            self.refresh_hz,
            if self.interlaced { "interlaced" } else { "progressive" },
            if self.on { "on" } else { "off" },
            self.mode_sets,
            self.scaler_inits,
            self.scaler_buffer_builds,
            self.edram_retrains
        )
    }
}

fn args(cpu: &Cpu) -> [u32; 4] {
    [cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32, cpu.gpr[6] as u32]
}

impl Kernel {
    /// Display-hardware calls: returns false if `name` isn't one of ours.
    pub fn display_call(&mut self, name: &str, cpu: &mut Cpu, mem: &mut GuestMemory) -> bool {
        match name {
            "VdSetDisplayMode" => {
                self.display.last_mode_args = args(cpu);
                self.display.mode_sets += 1;
                cpu.gpr[3] = 0;
            }
            "VdSetDisplayModeOverride" => {
                self.display.last_override_args = args(cpu);
                // The refresh rate is passed as a float argument (f1) when the title overrides it.
                let hz = f64::from_bits(cpu.fpr[1]) as f32;
                if hz.is_finite() && (24.0..=120.0).contains(&hz) {
                    self.display.refresh_hz = hz;
                }
                self.display.mode_sets += 1;
                cpu.gpr[3] = 0;
            }
            "VdInitializeScaler" => {
                self.display.last_scaler_args = args(cpu);
                self.display.scaler_inits += 1;
                cpu.gpr[3] = 0;
            }
            "VdInitializeScalerCommandBuffer" => {
                // (six scaler parameters, command buffer*, size in dwords) -> dwords written.
                // The real kernel programs the display scaler here. Our scan-out reads the front buffer
                // directly, so the buffer is filled with no-op packets - but the caller needs the size
                // back (a zero made it divide by zero downstream).
                let (dest, count) = (cpu.gpr[9] as u32, cpu.gpr[10] as u32);
                self.display.scaler_buffer_builds += 1;
                if dest != 0 && mem.is_mapped(dest) {
                    for i in 0..count.min(0x4000) {
                        mem.write32(dest + i * 4, 0x8000_0000);
                    }
                }
                cpu.gpr[3] = count as u64;
            }
            "VdTurnDisplayOn" | "VdTurnDisplayOff" => {
                self.display.on = name == "VdTurnDisplayOn";
                self.gpu.display_on = self.display.on;
                cpu.gpr[3] = 0;
            }
            "VdPersistDisplay" => {
                // (front buffer address, u32 *out): keep this buffer on screen across a title reload.
                // Nothing survives a reload in an emulator, so report "not persisted" but remember the request.
                self.display.persisted_front_buffer = cpu.gpr[3] as u32;
                if cpu.gpr[4] != 0 {
                    mem.write32(cpu.gpr[4] as u32, 0);
                }
                cpu.gpr[3] = 1;
            }
            "VdRetrainEDRAM" | "VdRetrainEDRAMWorker" => {
                // Calibrates the GPU<->EDRAM link; the emulated EDRAM is perfect, so it always succeeds.
                self.display.edram_retrains += 1;
                cpu.gpr[3] = 0;
            }
            "VdSetCGMSOption" => {
                self.display.cgms = cpu.gpr[3] as u32;
                cpu.gpr[3] = 0;
            }
            "VdSetWSSOption" => {
                self.display.wss = cpu.gpr[3] as u32;
                cpu.gpr[3] = 0;
            }
            "VdEnableClosedCaption" => {
                self.display.closed_captions = cpu.gpr[3] != 0;
                cpu.gpr[3] = 0;
            }
            "VdGetCurrentDisplayInformation" => {
                // X_DISPLAY_INFO: front buffer size at +0/+2 (u16), display size at +0x14/+0x18; the rest
                // of the layout is unverified, so it stays zero until the game proves it matters.
                let out = cpu.gpr[3] as u32;
                for off in (0..0x58).step_by(4) {
                    mem.write32(out + off, 0);
                }
                mem.write16(out, self.display.width as u16);
                mem.write16(out + 2, self.display.height as u16);
                mem.write32(out + 0x14, self.display.width);
                mem.write32(out + 0x18, self.display.height);
            }
            "XGetVideoMode" | "VdQueryVideoMode" => {
                // XVIDEO_MODE: width, height, interlaced, widescreen, hi-def, refresh (f32), standard.
                let d = &self.display;
                let out = cpu.gpr[3] as u32;
                for off in (0..0x30).step_by(4) {
                    mem.write32(out + off, 0);
                }
                mem.write32(out, d.width);
                mem.write32(out + 4, d.height);
                mem.write32(out + 8, d.interlaced as u32);
                mem.write32(out + 12, d.widescreen as u32);
                mem.write32(out + 16, d.hd as u32);
                mem.write32(out + 20, d.refresh_hz.to_bits());
                mem.write32(out + 24, 1); // NTSC-M
                // +0x1C / +0x20 are two small constants the real kernel fills in; the game divides by
                // the first, so it must be non-zero (value recalled from other implementations).
                mem.write32(out + 0x1C, 0x4A);
                mem.write32(out + 0x20, 1);
            }
            "VdQueryVideoFlags" => {
                // Bitfield describing the attached display: bit 1 = HD, bit 2 = widescreen.
                cpu.gpr[3] = (self.display.hd as u64) << 1 | (self.display.widescreen as u64) << 2;
            }
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Kernel, Cpu, GuestMemory) {
        let mut mem = GuestMemory::new();
        mem.map(0x4000_0000, 0x1000);
        (Kernel::new(), Cpu::new(), mem)
    }

    #[test]
    fn display_switch_reaches_the_gpu() {
        let (mut k, mut cpu, mut mem) = setup();
        assert!(k.display.on && k.gpu.display_on);
        assert!(k.display_call("VdTurnDisplayOff", &mut cpu, &mut mem));
        assert!(!k.display.on && !k.gpu.display_on);
        assert!(k.display_call("VdTurnDisplayOn", &mut cpu, &mut mem));
        assert!(k.gpu.display_on);
    }

    #[test]
    fn video_mode_follows_state() {
        let (mut k, mut cpu, mut mem) = setup();
        k.display.interlaced = true;
        cpu.gpr[3] = 0x4000_0000;
        assert!(k.display_call("VdQueryVideoMode", &mut cpu, &mut mem));
        assert_eq!(mem.read32(0x4000_0000), 1280);
        assert_eq!(mem.read32(0x4000_0008), 1);
        assert_eq!(f32::from_bits(mem.read32(0x4000_0014)), 60.0);
    }

    #[test]
    fn refresh_override_and_counters() {
        let (mut k, mut cpu, mut mem) = setup();
        cpu.fpr[1] = 50.0f64.to_bits();
        k.display_call("VdSetDisplayModeOverride", &mut cpu, &mut mem);
        assert_eq!(k.display.refresh_hz, 50.0);
        k.display_call("VdRetrainEDRAM", &mut cpu, &mut mem);
        k.display_call("VdRetrainEDRAMWorker", &mut cpu, &mut mem);
        assert_eq!(k.display.edram_retrains, 2);
    }
}
