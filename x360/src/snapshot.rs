//! Save states: the whole machine (guest memory, every thread's CPU, kernel, GPU, audio, input) in one file, so a
//! long boot only has to run once. `X360_SAVE_STATE=file,steps` writes one when the run reaches that many
//! instructions; `X360_LOAD_STATE=file` resumes from it (the disc is reopened from the path given as usual).
//!
//! Not saved: the JIT's code cache (rebuilt on demand), the live window, and the open disc file.

use crate::kernel::Kernel;
use crate::memory::GuestMemory;
use std::io::{BufReader, BufWriter};

const MAGIC: &[u8; 8] = b"X360SAV1";

pub fn save(path: &str, kernel: &Kernel, mem: &GuestMemory, steps: u64) -> std::io::Result<()> {
    let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
    use std::io::Write;
    w.write_all(MAGIC)?;
    w.write_all(&steps.to_le_bytes())?;
    bincode::serialize_into(&mut w, kernel).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    mem.save_to(&mut w)?;
    w.flush()
}

/// Returns the saved kernel (without its disc handle) and the instruction count it was saved at; `mem` is overwritten.
pub fn load(path: &str, mem: &mut GuestMemory) -> std::io::Result<(Kernel, u64)> {
    use std::io::Read;
    let mut r = BufReader::with_capacity(1 << 20, std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a save state (or from another version)"));
    }
    let mut steps = [0u8; 8];
    r.read_exact(&mut steps)?;
    let kernel: Kernel = bincode::deserialize_from(&mut r).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    mem.load_from(&mut r)?;
    Ok((kernel, u64::from_le_bytes(steps)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xenon_cpu::memory::Bus;

    #[test]
    fn round_trip_preserves_memory_and_kernel() {
        let mut mem = GuestMemory::new();
        mem.map(0x4000_0000, 0x3000);
        mem.write32(0x4000_0010, 0xDEAD_BEEF);
        mem.write32(0x4000_2FFC, 0x1234_5678);
        let _ = mem.alloc_kernel(0x100);
        let mut k = Kernel::new();
        k.vsyncs = 77;
        k.global_cycles = 123_456;
        let id = k.spawn_thread(&mut mem, 0x8200_0000, 1, 2, 0x10000, false).unwrap();
        k.threads[id].cpu.gpr[5] = 0xAABB_CCDD_EEFF_0011;
        k.threads[id].cpu.vr[100] = 0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10;
        k.xam.packet = 42;
        let path = std::env::temp_dir().join(format!("x360_snap_{}.bin", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        save(&path, &k, &mem, 999).unwrap();
        let mut mem2 = GuestMemory::new();
        let (k2, steps) = load(&path, &mut mem2).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(steps, 999);
        assert_eq!(mem2.read32(0x4000_0010), 0xDEAD_BEEF);
        assert_eq!(mem2.read32(0x4000_2FFC), 0x1234_5678);
        assert!(mem2.is_mapped(0x4000_1000));
        for a in [0x4000_3000u32, 0x4001_0000, 0x7000_0000, 0x8200_0000] {
            assert_eq!(mem2.is_mapped(a), mem.is_mapped(a), "mapping differs at {a:#x}");
        }
        assert_eq!((k2.vsyncs, k2.global_cycles, k2.xam.packet), (77, 123_456, 42));
        assert_eq!(k2.threads.len(), k.threads.len());
        assert_eq!(k2.threads[id].cpu.gpr[5], 0xAABB_CCDD_EEFF_0011);
        assert_eq!(k2.threads[id].cpu.vr[100], 0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10);
        assert_eq!(k2.threads[id].pcr, k.threads[id].pcr);
    }
}
