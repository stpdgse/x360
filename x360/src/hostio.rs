//! NtCreateFile/Read/Write/... for the writable host-backed storage (see `hostfs.rs`).
//! Called before the read-only disc code; it only claims calls that target host storage.

use crate::kernel::Kernel;
use crate::memory::GuestMemory;
use crate::objects::{KObject, STATUS_INVALID_HANDLE, STATUS_OBJECT_NAME_NOT_FOUND, STATUS_SUCCESS};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

const STATUS_END_OF_FILE: u32 = 0xC000_0011;
const STATUS_NAME_COLLISION: u32 = 0xC000_0035;
const STATUS_IS_A_DIRECTORY: u32 = 0xC000_00BA;
const STATUS_NOT_A_DIRECTORY: u32 = 0xC000_0103;
const STATUS_ACCESS_DENIED: u32 = 0xC000_0022;
const STATUS_NO_MORE_FILES: u32 = 0x8000_0006;
const STATUS_INVALID_INFO_CLASS: u32 = 0xC000_0003;

const FILE_SUPERSEDED: u32 = 0;
const FILE_OPENED: u32 = 1;
const FILE_CREATED: u32 = 2;
const FILE_OVERWRITTEN: u32 = 3;

fn io_status(mem: &mut GuestMemory, iosb: u32, status: u32, info: u32) {
    if iosb != 0 {
        mem.write32(iosb, status);
        mem.write32(iosb + 4, info);
    }
}

fn guest_string(mem: &GuestMemory, s: u32) -> String {
    if s == 0 {
        return String::new();
    }
    let (len, buf) = (mem.read16(s) as u32, mem.read32(s + 4));
    (0..len).map(|i| mem.read8(buf + i) as char).collect()
}

/// Outcome of opening/creating a path given an NT disposition. Pure over the file system state.
pub fn open_path(path: &PathBuf, disposition: u32, options: u32) -> Result<(bool, u32), u32> {
    let want_dir = options & 1 != 0;
    let want_file = options & 0x40 != 0;
    let exists = path.exists();
    let is_dir = path.is_dir();
    if exists && is_dir && want_file {
        return Err(STATUS_IS_A_DIRECTORY);
    }
    if exists && !is_dir && want_dir {
        return Err(STATUS_NOT_A_DIRECTORY);
    }
    // disposition: 0 supersede, 1 open, 2 create, 3 open-if, 4 overwrite, 5 overwrite-if
    let (create, truncate, info_new, info_old) = match disposition {
        1 => (false, false, FILE_OPENED, FILE_OPENED),
        2 => {
            if exists {
                return Err(STATUS_NAME_COLLISION);
            }
            (true, false, FILE_CREATED, FILE_CREATED)
        }
        3 => (true, false, FILE_CREATED, FILE_OPENED),
        4 => (false, true, FILE_OVERWRITTEN, FILE_OVERWRITTEN),
        0 | 5 => (true, true, FILE_CREATED, if disposition == 0 { FILE_SUPERSEDED } else { FILE_OVERWRITTEN }),
        _ => (false, false, FILE_OPENED, FILE_OPENED),
    };
    if !exists {
        if !create {
            return Err(STATUS_OBJECT_NAME_NOT_FOUND);
        }
        if want_dir {
            std::fs::create_dir_all(path).map_err(|_| STATUS_ACCESS_DENIED)?;
            return Ok((true, info_new));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|_| STATUS_ACCESS_DENIED)?;
        }
        std::fs::File::create(path).map_err(|_| STATUS_ACCESS_DENIED)?;
        return Ok((false, info_new));
    }
    if is_dir {
        return Ok((true, FILE_OPENED));
    }
    if truncate {
        std::fs::File::create(path).map_err(|_| STATUS_ACCESS_DENIED)?;
    }
    Ok((false, info_old))
}

impl Kernel {
    /// Host-storage path an OBJECT_ATTRIBUTES refers to, if it isn't a disc path.
    fn host_target(&self, mem: &GuestMemory, attrs: u32) -> Option<PathBuf> {
        let root = mem.read32(attrs);
        let name = guest_string(mem, mem.read32(attrs + 4));
        if root != 0 && root != 0xFFFF_FFFD {
            return match self.objects.get(root) {
                Some(KObject::HostFile { path, .. }) => {
                    let mut p = path.clone();
                    for c in name.to_lowercase().split(['\\', '/']).filter(|c| !c.is_empty() && *c != "." && *c != "..") {
                        p.push(c);
                    }
                    Some(p)
                }
                _ => None,
            };
        }
        if let Some(v) = &self.vfs {
            if matches!(v.resolve(&name), crate::fs::Resolved::Disc(_)) {
                return None;
            }
        }
        crate::hostfs::resolve(&crate::hostfs::root(), &name)
    }

    /// Returns Ok(true) if the call was handled here.
    pub fn host_io_call(&mut self, name: &str, cpu: &mut Cpu, mem: &mut GuestMemory) -> Result<bool, String> {
        let status: u32 = match name {
            "NtCreateFile" | "NtOpenFile" => {
                let (out, attrs, iosb) = (cpu.gpr[3] as u32, cpu.gpr[5] as u32, cpu.gpr[6] as u32);
                let Some(path) = self.host_target(mem, attrs) else { return Ok(false) };
                let (disposition, options) = if name == "NtCreateFile" {
                    (cpu.gpr[10] as u32, mem.read32(cpu.gpr[1] as u32 + 0x54))
                } else {
                    (1, cpu.gpr[8] as u32)
                };
                match open_path(&path, disposition, options) {
                    Ok((is_dir, info)) => {
                        if self.trace_calls {
                            println!("note: {name}({}) -> host {path:?} (dir={is_dir}, info={info})", guest_string(mem, mem.read32(attrs + 4)));
                        }
                        let h = self.objects.create(KObject::HostFile { path, pos: 0, is_dir });
                        mem.write32(out, h);
                        io_status(mem, iosb, STATUS_SUCCESS, info);
                        STATUS_SUCCESS
                    }
                    Err(s) => {
                        mem.write32(out, 0);
                        io_status(mem, iosb, s, 0);
                        s
                    }
                }
            }
            "NtReadFile" => {
                let (h, event, iosb, buf, len, off_ptr) =
                    (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[7] as u32, cpu.gpr[8] as u32, cpu.gpr[9] as usize, cpu.gpr[10] as u32);
                let explicit = (off_ptr != 0).then(|| mem.read64(off_ptr)).filter(|o| *o != u64::MAX);
                let Some((path, pos, _)) = self.objects.host_mut(h) else { return Ok(false) };
                let at = explicit.unwrap_or(*pos);
                let path = path.clone();
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let status = if at >= size && len > 0 {
                    io_status(mem, iosb, STATUS_END_OF_FILE, 0);
                    STATUS_END_OF_FILE
                } else {
                    let mut data = vec![0u8; len.min((size - at) as usize)];
                    let ok = std::fs::File::open(&path)
                        .and_then(|mut f| f.seek(SeekFrom::Start(at)).and_then(|_| f.read_exact(&mut data)))
                        .is_ok();
                    if ok {
                        mem.write_bytes(buf, &data);
                        *self.objects.host_mut(h).unwrap().1 = at + data.len() as u64;
                        io_status(mem, iosb, STATUS_SUCCESS, data.len() as u32);
                        STATUS_SUCCESS
                    } else {
                        io_status(mem, iosb, STATUS_ACCESS_DENIED, 0);
                        STATUS_ACCESS_DENIED
                    }
                };
                self.signal_io_event(mem, event, status);
                status
            }
            "NtWriteFile" => {
                let (h, event, iosb, buf, len, off_ptr) =
                    (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[7] as u32, cpu.gpr[8] as u32, cpu.gpr[9] as usize, cpu.gpr[10] as u32);
                let explicit = (off_ptr != 0).then(|| mem.read64(off_ptr)).filter(|o| *o != u64::MAX);
                let Some((path, pos, _)) = self.objects.host_mut(h) else { return Ok(false) };
                let at = explicit.unwrap_or(*pos);
                let path = path.clone();
                let data: Vec<u8> = (0..len as u32).map(|i| mem.read8(buf + i)).collect();
                let ok = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .and_then(|mut f| f.seek(SeekFrom::Start(at)).and_then(|_| f.write_all(&data)))
                    .is_ok();
                let status = if ok {
                    *self.objects.host_mut(h).unwrap().1 = at + len as u64;
                    io_status(mem, iosb, STATUS_SUCCESS, len as u32);
                    STATUS_SUCCESS
                } else {
                    io_status(mem, iosb, STATUS_ACCESS_DENIED, 0);
                    STATUS_ACCESS_DENIED
                };
                self.signal_io_event(mem, event, status);
                status
            }
            "NtFlushBuffersFile" => {
                let (h, iosb) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32);
                if self.objects.host_mut(h).is_none() {
                    return Ok(false);
                }
                io_status(mem, iosb, STATUS_SUCCESS, 0);
                STATUS_SUCCESS
            }
            "NtQueryInformationFile" | "NtSetInformationFile" => {
                let (h, iosb, info, len, class) =
                    (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32, cpu.gpr[6] as u32, cpu.gpr[7] as u32);
                let Some((path, pos, is_dir)) = self.objects.host_mut(h) else { return Ok(false) };
                let path = path.clone();
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let alloc = (size + 4095) & !4095;
                let attrs = if is_dir { 0x10 } else { 0x80 };
                let _ = len;
                let status = match (class, name == "NtSetInformationFile") {
                    (14, true) => {
                        *pos = mem.read64(info);
                        STATUS_SUCCESS
                    }
                    (14, false) => {
                        mem.write64(info, *pos);
                        STATUS_SUCCESS
                    }
                    (5, false) => {
                        mem.write64(info, alloc);
                        mem.write64(info + 8, size);
                        mem.write32(info + 16, 1);
                        mem.write8(info + 20, 0);
                        mem.write8(info + 21, is_dir as u8);
                        STATUS_SUCCESS
                    }
                    (4, false) => {
                        for off in (0..32).step_by(8) {
                            mem.write64(info + off, 0);
                        }
                        mem.write32(info + 32, attrs);
                        STATUS_SUCCESS
                    }
                    (34, false) => {
                        for off in (0..56).step_by(8) {
                            mem.write64(info + off, 0);
                        }
                        mem.write64(info + 32, alloc);
                        mem.write64(info + 40, size);
                        mem.write32(info + 48, attrs);
                        STATUS_SUCCESS
                    }
                    (20, true) => {
                        // FileEndOfFileInformation: truncate / extend
                        let n = mem.read64(info);
                        match std::fs::OpenOptions::new().write(true).open(&path).and_then(|f| f.set_len(n)) {
                            Ok(()) => STATUS_SUCCESS,
                            Err(_) => STATUS_ACCESS_DENIED,
                        }
                    }
                    (13, true) => {
                        // FileDispositionInformation: delete (done immediately; nothing else holds it open)
                        if mem.read8(info) != 0 {
                            let r = if is_dir { std::fs::remove_dir(&path) } else { std::fs::remove_file(&path) };
                            if r.is_err() { STATUS_ACCESS_DENIED } else { STATUS_SUCCESS }
                        } else {
                            STATUS_SUCCESS
                        }
                    }
                    (_, true) => STATUS_SUCCESS, // allocation size, basic (timestamps/attributes), ... ignored
                    _ => {
                        println!("note: {name} class {class} on host file not implemented");
                        STATUS_INVALID_INFO_CLASS
                    }
                };
                io_status(mem, iosb, status, 0);
                status
            }
            "NtQueryDirectoryFile" => {
                let h = cpu.gpr[3] as u32;
                let (iosb, buf, len, pattern_ptr) = (cpu.gpr[7] as u32, cpu.gpr[8] as u32, cpu.gpr[9] as u32, cpu.gpr[10] as u32);
                let restart = mem.read32(cpu.gpr[1] as u32 + 0x54) != 0;
                let Some((path, pos, is_dir)) = self.objects.host_mut(h) else { return Ok(false) };
                if !is_dir {
                    return Ok(false);
                }
                let pattern = guest_string(mem, pattern_ptr).to_lowercase();
                let wild = pattern.is_empty() || pattern == "*" || pattern == "*.*";
                let mut kids: Vec<(String, u64, bool)> = std::fs::read_dir(path)
                    .map(|rd| {
                        rd.flatten()
                            .filter_map(|e| {
                                let m = e.metadata().ok()?;
                                Some((e.file_name().to_string_lossy().to_string(), m.len(), m.is_dir()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                kids.sort_by_key(|k| k.0.to_lowercase());
                kids.retain(|k| wild || k.0.to_lowercase() == pattern);
                if restart {
                    *pos = 0;
                }
                let mut idx = *pos as usize;
                let (mut off, mut prev, mut written) = (0u32, None::<u32>, 0);
                while idx < kids.len() {
                    let (name, size, dir) = &kids[idx];
                    let need = (64 + name.len() as u32 + 7) & !7;
                    if off + need > len {
                        break;
                    }
                    let at = buf + off;
                    for o in (0..64).step_by(4) {
                        mem.write32(at + o, 0);
                    }
                    mem.write32(at + 4, idx as u32);
                    mem.write64(at + 40, if *dir { 0 } else { *size });
                    mem.write64(at + 48, if *dir { 0 } else { (*size + 4095) & !4095 });
                    mem.write32(at + 56, if *dir { 0x10 } else { 0x80 });
                    mem.write32(at + 60, name.len() as u32);
                    for (i, b) in name.bytes().enumerate() {
                        mem.write8(at + 64 + i as u32, b);
                    }
                    if let Some(p) = prev {
                        mem.write32(buf + p, off - p);
                    }
                    prev = Some(off);
                    off += need;
                    written += 1;
                    idx += 1;
                }
                *self.objects.host_mut(h).unwrap().1 = idx as u64;
                if written == 0 {
                    let s = if idx == 0 && !wild { STATUS_OBJECT_NAME_NOT_FOUND } else { STATUS_NO_MORE_FILES };
                    io_status(mem, iosb, s, 0);
                    s
                } else {
                    io_status(mem, iosb, STATUS_SUCCESS, off);
                    STATUS_SUCCESS
                }
            }
            "NtQueryVolumeInformationFile" => {
                // (handle, IO_STATUS_BLOCK*, buffer, length, class) - for any open file, disc or host
                let (h, iosb, info, class) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32, cpu.gpr[7] as u32);
                let host = self.objects.host_mut(h).is_some();
                if !host && self.objects.file_mut(h).is_none() {
                    cpu.gpr[3] = STATUS_INVALID_HANDLE as u64;
                    return Ok(true);
                }
                let status = match class {
                    3 => {
                        // FileFsSizeInformation: 20 GB drive, 16 GB free, 4 KB allocation units
                        mem.write64(info, 5_000_000);
                        mem.write64(info + 8, if host { 4_000_000 } else { 0 });
                        mem.write32(info + 16, 8);
                        mem.write32(info + 20, 512);
                        io_status(mem, iosb, STATUS_SUCCESS, 24);
                        STATUS_SUCCESS
                    }
                    4 => {
                        // FileFsDeviceInformation: type (7 disk / 2 cdrom), characteristics
                        mem.write32(info, if host { 7 } else { 2 });
                        mem.write32(info + 4, if host { 0x20 } else { 0x1 });
                        io_status(mem, iosb, STATUS_SUCCESS, 8);
                        STATUS_SUCCESS
                    }
                    5 => {
                        // FileFsAttributeInformation: attributes, max component length, fs name "FATX"/"UDFS"
                        let fs = if host { "FATX" } else { "UDFS" };
                        mem.write32(info, 0x3);
                        mem.write32(info + 4, 42);
                        mem.write32(info + 8, fs.len() as u32);
                        for (i, b) in fs.bytes().enumerate() {
                            mem.write8(info + 12 + i as u32, b);
                        }
                        io_status(mem, iosb, STATUS_SUCCESS, 12 + fs.len() as u32);
                        STATUS_SUCCESS
                    }
                    _ => {
                        println!("note: NtQueryVolumeInformationFile class {class} not implemented");
                        io_status(mem, iosb, STATUS_INVALID_INFO_CLASS, 0);
                        STATUS_INVALID_INFO_CLASS
                    }
                };
                status
            }
            _ => return Ok(false),
        };
        cpu.gpr[3] = status as u64;
        Ok(true)
    }

    fn signal_io_event(&mut self, mem: &mut GuestMemory, event: u32, status: u32) {
        if status == STATUS_SUCCESS && event != 0 {
            if let Some(addr) = self.objects.dispatcher_addr(event) {
                crate::dispatcher::set_event(mem, addr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("x360_hostio_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn dispositions() {
        let d = scratch("disp");
        let f = d.join("a.txt");
        assert_eq!(open_path(&f, 1, 0), Err(STATUS_OBJECT_NAME_NOT_FOUND));
        assert_eq!(open_path(&f, 2, 0), Ok((false, FILE_CREATED)));
        assert_eq!(open_path(&f, 2, 0), Err(STATUS_NAME_COLLISION));
        assert_eq!(open_path(&f, 3, 0), Ok((false, FILE_OPENED)));
        std::fs::write(&f, b"hello").unwrap();
        assert_eq!(open_path(&f, 5, 0), Ok((false, FILE_OVERWRITTEN)));
        assert_eq!(std::fs::metadata(&f).unwrap().len(), 0);
        let sub = d.join("x").join("y");
        assert_eq!(open_path(&sub, 3, 1), Ok((true, FILE_CREATED)));
        assert_eq!(open_path(&sub, 1, 0x40), Err(STATUS_IS_A_DIRECTORY));
        assert_eq!(open_path(&f, 1, 1), Err(STATUS_NOT_A_DIRECTORY));
        let deep = d.join("p").join("q.bin");
        assert_eq!(open_path(&deep, 2, 0), Ok((false, FILE_CREATED)), "parents are created");
        let _ = std::fs::remove_dir_all(&d);
    }
}
