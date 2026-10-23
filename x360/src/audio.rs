//! Audio render driver: the title registers a callback that the audio hardware invokes every
//! time it wants another frame of samples; the title answers by calling
//! XAudioSubmitRenderDriverFrame with 256 samples x 6 channels of big-endian float.
//!
//! We play the hardware's part: every 256 samples of virtual time (48 kHz) the callback runs on
//! its own thread, and submitted frames are downmixed to stereo 16-bit. With `X360_DUMP_AUDIO=file.wav`
//! the result is written out at exit. Channel layout (interleaved FL FR FC LFE BL BR) is assumed
//! from how the format is usually described; unverified until a title produces real sound.

use serde::{Deserialize, Serialize};
use crate::kernel::Kernel;
use crate::memory::GuestMemory;
use crate::sched::CPU_HZ;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

pub const SAMPLE_RATE: u32 = 48_000;
pub const FRAME_SAMPLES: u32 = 256;
pub const CHANNELS: u32 = 6;
/// Don't keep more than this much output in memory (about two minutes).
const MAX_STEREO_SAMPLES: usize = SAMPLE_RATE as usize * 120;
pub const DRIVER_HANDLE: u32 = 0x4155_0001;

#[derive(Default, Serialize, Deserialize)]
pub struct AudioState {
    pub client_fn: u32,
    pub client_ctx: u32,
    next_at: u64,
    pub callbacks: u64,
    /// Callback ticks skipped because the previous callback hadn't returned yet.
    pub missed: u64,
    pub frames: u64,
    pub peak: f32,
    pub nonsilent_frames: u64,
    pcm: Vec<i16>, // stereo interleaved
    /// Live XMA context blocks (guest addresses). Bookkeeping only: nothing decodes them yet.
    pub xma_contexts: Vec<u32>,
}

pub fn interval_cycles() -> u64 {
    (CPU_HZ * FRAME_SAMPLES as f64 / SAMPLE_RATE as f64) as u64
}

impl AudioState {
    pub fn next_tick(&self) -> Option<u64> {
        (self.client_fn != 0 && self.next_at != 0).then_some(self.next_at)
    }

    /// Downmix one submitted frame (interleaved 6-channel floats) into the stereo output.
    pub fn push_frame(&mut self, samples: &[f32]) {
        self.frames += 1;
        let mut loud = false;
        for s in samples.chunks_exact(CHANNELS as usize) {
            let (fl, fr, fc, bl, br) = (s[0], s[1], s[2], s[4], s[5]);
            let l = fl + 0.707 * fc + 0.707 * bl;
            let r = fr + 0.707 * fc + 0.707 * br;
            self.peak = self.peak.max(l.abs()).max(r.abs());
            loud |= l != 0.0 || r != 0.0;
            if self.pcm.len() < MAX_STEREO_SAMPLES * 2 {
                self.pcm.push((l.clamp(-1.0, 1.0) * 32767.0) as i16);
                self.pcm.push((r.clamp(-1.0, 1.0) * 32767.0) as i16);
            }
        }
        self.nonsilent_frames += loud as u64;
    }

    pub fn wav_bytes(&self) -> Vec<u8> {
        let data_len = (self.pcm.len() * 2) as u32;
        let mut w = Vec::with_capacity(44 + data_len as usize);
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&(36 + data_len).to_le_bytes());
        w.extend_from_slice(b"WAVEfmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes()); // PCM
        w.extend_from_slice(&2u16.to_le_bytes()); // stereo
        w.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        w.extend_from_slice(&(SAMPLE_RATE * 4).to_le_bytes());
        w.extend_from_slice(&4u16.to_le_bytes());
        w.extend_from_slice(&16u16.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data_len.to_le_bytes());
        for s in &self.pcm {
            w.extend_from_slice(&s.to_le_bytes());
        }
        w
    }

    pub fn summary(&self) -> String {
        format!(
            "audio: {} render callbacks ({} missed), {} frames submitted ({} non-silent, peak {:.3}), {:.2}s of output",
            self.callbacks,
            self.missed,
            self.frames,
            self.nonsilent_frames,
            self.peak,
            self.pcm.len() as f64 / 2.0 / SAMPLE_RATE as f64
        )
    }

    /// Write the captured output if `X360_DUMP_AUDIO` names a file.
    pub fn finish(&self) {
        if let Ok(path) = std::env::var("X360_DUMP_AUDIO") {
            if !self.pcm.is_empty() {
                match std::fs::write(&path, self.wav_bytes()) {
                    Ok(()) => println!("audio: wrote {path}"),
                    Err(e) => println!("audio: could not write {path}: {e}"),
                }
            }
        }
    }
}

impl Kernel {
    /// Run the render-driver callback when its 256-sample period has elapsed.
    pub fn audio_tick(&mut self, mem: &mut GuestMemory) -> Result<(), String> {
        if self.audio.client_fn == 0 {
            return Ok(());
        }
        if self.audio.next_at == 0 {
            self.audio.next_at = self.global_cycles + interval_cycles();
        }
        if self.global_cycles >= self.audio.next_at {
            // Catch up without firing a burst of callbacks after a long idle skip.
            while self.audio.next_at <= self.global_cycles {
                self.audio.next_at += interval_cycles();
            }
            if self.threads.iter().any(|t| t.is_audio && !t.terminated) {
                self.audio.missed += 1;
            } else {
                let (cb, ctx) = (self.audio.client_fn, self.audio.client_ctx);
                let idx = self.spawn_thread(mem, cb, ctx as u64, 0, 0x10000, false)?;
                self.threads[idx].is_audio = true;
                self.threads[idx].recyclable = true;
                self.audio.callbacks += 1;
            }
        }
        Ok(())
    }

    /// Audio-driver calls: returns false if `name` isn't one of ours.
    pub fn audio_call(&mut self, name: &str, cpu: &mut Cpu, mem: &mut GuestMemory) -> bool {
        match name {
            "XAudioRegisterRenderDriverClient" => {
                // ({callback fn, context}*, u32 *driver handle out) -> HRESULT
                let client = cpu.gpr[3] as u32;
                self.audio.client_fn = mem.read32(client);
                self.audio.client_ctx = mem.read32(client + 4);
                self.audio.next_at = 0;
                mem.write32(cpu.gpr[4] as u32, DRIVER_HANDLE);
                cpu.gpr[3] = 0;
            }
            "XAudioUnregisterRenderDriverClient" => {
                self.audio.client_fn = 0;
                cpu.gpr[3] = 0;
            }
            "XAudioSubmitRenderDriverFrame" => {
                // (driver handle, samples*): 256 x 6 big-endian floats
                let p = cpu.gpr[4] as u32;
                if cpu.gpr[3] as u32 != DRIVER_HANDLE || p == 0 || !mem.is_mapped(p) {
                    cpu.gpr[3] = 0x8007_0057; // E_INVALIDARG
                } else {
                    let samples: Vec<f32> =
                        (0..FRAME_SAMPLES * CHANNELS).map(|i| f32::from_bits(mem.read32(p + i * 4))).collect();
                    self.audio.push_frame(&samples);
                    cpu.gpr[3] = 0;
                }
            }
            "XAudioGetVoiceCategoryVolumeChangeMask" => {
                // (driver handle, u32 *mask out): bit n set = category n's volume changed since last asked.
                // The system never changes volumes behind the title's back.
                mem.write32(cpu.gpr[4] as u32, 0);
                cpu.gpr[3] = 0;
            }
            "XAudioGetVoiceCategoryVolume" => {
                // (category, f32 *volume out): full volume
                mem.write32(cpu.gpr[4] as u32, 1.0f32.to_bits());
                cpu.gpr[3] = 0;
            }
            "XAudioEnableDucker" | "XAudioSetVoiceCategoryVolume" => {
                cpu.gpr[3] = 0;
            }
            "XMACreateContext" => {
                // (u32 *context out): a 64-byte XMA context block the title fills in and the (emulated)
                // decoder would read. Decoding isn't implemented: contexts exist but produce no samples.
                let ctx = mem.alloc_kernel(64);
                for off in (0..64).step_by(4) {
                    mem.write32(ctx + off, 0);
                }
                self.audio.xma_contexts.push(ctx);
                mem.write32(cpu.gpr[3] as u32, ctx);
                cpu.gpr[3] = 0;
            }
            "XMAReleaseContext" => {
                let ctx = cpu.gpr[3] as u32;
                self.audio.xma_contexts.retain(|&c| c != ctx);
                cpu.gpr[3] = 0;
            }
            "XAudioGetSpeakerConfig" => {
                mem.write32(cpu.gpr[3] as u32, 0x0001_0001); // stereo
                cpu.gpr[3] = 0;
            }
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_is_256_samples_at_48k() {
        // 3.2 GHz * 256 / 48000
        assert_eq!(interval_cycles(), 17_066_666);
    }

    #[test]
    fn frame_downmix_and_wav() {
        let mut a = AudioState::default();
        let mut frame = vec![0.0f32; (FRAME_SAMPLES * CHANNELS) as usize];
        frame[0] = 0.5; // FL of sample 0
        frame[2] = 1.0; // FC of sample 0
        a.push_frame(&frame);
        assert_eq!(a.nonsilent_frames, 1);
        assert_eq!(a.pcm.len(), FRAME_SAMPLES as usize * 2);
        assert_eq!(a.pcm[0], ((0.5 + 0.707) * 32767.0) as i16);
        assert_eq!(a.pcm[1], (0.707 * 32767.0) as i16);
        let w = a.wav_bytes();
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(w[40..44].try_into().unwrap()) as usize, a.pcm.len() * 2);
        a.push_frame(&vec![0.0; (FRAME_SAMPLES * CHANNELS) as usize]);
        assert_eq!(a.nonsilent_frames, 1, "silence isn't counted");
    }

    #[test]
    fn callback_fires_on_period_and_not_twice() {
        let mut mem = GuestMemory::new();
        let mut k = Kernel::new();
        k.audio.client_fn = 0x8200_0000;
        k.audio.client_ctx = 0x1234;
        k.audio_tick(&mut mem).unwrap(); // arms the timer
        assert_eq!(k.audio.callbacks, 0);
        k.global_cycles = interval_cycles() + 1;
        k.audio_tick(&mut mem).unwrap();
        assert_eq!(k.audio.callbacks, 1);
        k.global_cycles += interval_cycles();
        k.audio_tick(&mut mem).unwrap(); // previous callback still running
        assert_eq!((k.audio.callbacks, k.audio.missed), (1, 1));
    }
}
