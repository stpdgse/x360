//! File I/O kernel calls (NtCreateFile / NtReadFile / ...), backed by the disc VFS.

use crate::fs::Resolved;
use crate::kernel::Kernel;
use crate::memory::GuestMemory;
use crate::objects::{KObject, STATUS_INVALID_HANDLE, STATUS_OBJECT_NAME_NOT_FOUND, STATUS_SUCCESS};
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

const STATUS_END_OF_FILE: u32 = 0xC000_0011;
const STATUS_INVALID_INFO_CLASS: u32 = 0xC000_0003;
const STATUS_NOT_IMPLEMENTED: u32 = 0xC000_0002;
const STATUS_NO_SUCH_DEVICE: u32 = 0xC000_000E;
const FILE_OPENED: u32 = 1;

fn io_status(mem: &mut GuestMemory, iosb: u32, status: u32, info: u32) {
    if iosb != 0 {
        mem.write32(iosb, status);
        mem.write32(iosb + 4, info);
    }
}

/// OBJECT_ATTRIBUTES {root dir handle u32, PANSI_STRING name u32, attributes u32} -> (root, name).
fn object_attributes(mem: &GuestMemory, attrs: u32) -> (u32, String) {
    let root = mem.read32(attrs);
    let name = mem.read32(attrs + 4);
    if name == 0 {
        return (root, String::new());
    }
    let (len, buf) = (mem.read16(name) as u32, mem.read32(name + 4));
    (root, (0..len).map(|i| mem.read8(buf + i) as char).collect())
}

impl Kernel {
    /// Resolve an OBJECT_ATTRIBUTES to a disc-relative path, honouring a root-directory handle.
    fn attrs_to_disc_path(&mut self, mem: &GuestMemory, attrs: u32) -> Result<String, u32> {
        let (root, name) = object_attributes(mem, attrs);
        // 0xFFFFFFFD (-3) is the "no root directory" sentinel, same as 0.
        if root != 0 && root != 0xFFFF_FFFD {
            let parent = match self.objects.get(root) {
                Some(KObject::File { entry, .. }) => entry.path.clone(),
                _ => {
                    println!("note: object attributes root handle {root:#x} is not an open file (name {name:?})");
                    return Err(STATUS_INVALID_HANDLE);
                }
            };
            let child = name.replace('\\', "/");
            return Ok(if parent.is_empty() { child } else { format!("{parent}/{child}") });
        }
        match self.vfs.as_ref().map(|v| v.resolve(&name)) {
            Some(Resolved::Disc(rel)) => Ok(rel),
            Some(Resolved::Unsupported(p)) => {
                println!("note: file open on unsupported device: {p:?}");
                Err(STATUS_OBJECT_NAME_NOT_FOUND)
            }
            None => Err(STATUS_NO_SUCH_DEVICE),
        }
    }

    /// Returns Ok(true) if `name` was an I/O call we handled.
    pub fn io_call(&mut self, name: &str, cpu: &mut Cpu, mem: &mut GuestMemory) -> Result<bool, String> {
        match name {
            "NtCreateFile" | "NtOpenFile" => {
                // NtCreateFile(PHANDLE, access, OBJECT_ATTRIBUTES*, IO_STATUS_BLOCK*, alloc size*, attrs, share, disposition, options)
                // NtOpenFile  (PHANDLE, access, OBJECT_ATTRIBUTES*, IO_STATUS_BLOCK*, share, options)
                let (out, attrs, iosb) = (cpu.gpr[3] as u32, cpu.gpr[5] as u32, cpu.gpr[6] as u32);
                let (_, orig) = object_attributes(mem, attrs);
                let status = match self.attrs_to_disc_path(mem, attrs) {
                    Ok(rel) => match self.vfs.as_ref().and_then(|v| v.stat(&rel)) {
                        Some(entry) => {
                            let h = self.objects.create(KObject::File { entry, pos: 0 });
                            mem.write32(out, h);
                            io_status(mem, iosb, STATUS_SUCCESS, FILE_OPENED);
                            STATUS_SUCCESS
                        }
                        None => STATUS_OBJECT_NAME_NOT_FOUND,
                    },
                    Err(s) => s,
                };
                if status != STATUS_SUCCESS {
                    // On failure the handle out-parameter is zeroed so callers that go on to use it
                    // see a null handle rather than stale stack contents.
                    if out != 0 {
                        mem.write32(out, 0);
                    }
                    io_status(mem, iosb, status, 0);
                    let mut chain = format!("lr={:#x}", cpu.lr);
                    let mut sp = cpu.gpr[1] as u32;
                    for _ in 0..12 {
                        let caller = mem.read32(sp);
                        if caller <= sp || !mem.is_mapped(caller - 8) {
                            break;
                        }
                        chain.push_str(&format!(" <- {:#x}", mem.read32(caller - 8)));
                        sp = caller;
                    }
                    // Loose-file probes that fall back to an archive fail all the time; only show under trace.
                    if std::env::var_os("X360_TRACE_IO").is_some() {
                        println!("note: {name}({orig:?}) failed: {status:#x}  [{chain}]");
                    }
                } else if self.trace_calls {
                    println!("note: {name}({orig:?}) ok");
                }
                cpu.gpr[3] = status as u64;
                Ok(true)
            }
            "NtReadFile" => {
                // (handle, event, apc routine, apc context, IO_STATUS_BLOCK*, buffer, length, PLARGE_INTEGER offset)
                let (h, event, iosb, buf, len, off_ptr) =
                    (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[7] as u32, cpu.gpr[8] as u32, cpu.gpr[9] as usize, cpu.gpr[10] as u32);
                let explicit = if off_ptr != 0 { Some(mem.read64(off_ptr)) } else { None };
                if self.objects.file_mut(h).is_none() {
                    let mut chain = format!("lr={:#x}", cpu.lr);
                    let mut sp = cpu.gpr[1] as u32;
                    for _ in 0..10 {
                        let caller = mem.read32(sp);
                        if caller <= sp || !mem.is_mapped(caller - 8) {
                            break;
                        }
                        chain.push_str(&format!(" <- {:#x}", mem.read32(caller - 8)));
                        sp = caller;
                    }
                    println!("note: bad-read backtrace: {chain}");
                    println!(
                        "note: NtReadFile on invalid handle {h:#x}: event={event:#x} iosb={iosb:#x} buf={buf:#x} len={len:#x} off_ptr={off_ptr:#x}"
                    );
                }
                let status = match self.objects.file_mut(h) {
                    None => STATUS_INVALID_HANDLE,
                    Some((entry, pos)) => {
                        let at = match explicit {
                            Some(o) if o != u64::MAX => o,
                            _ => *pos,
                        };
                        if at >= entry.size as u64 && len > 0 {
                            io_status(mem, iosb, STATUS_END_OF_FILE, 0);
                            STATUS_END_OF_FILE
                        } else {
                            let entry = entry.clone();
                            match self.vfs.as_mut().map(|v| v.read(&entry, at, len)) {
                                Some(Ok(data)) => {
                                    if std::env::var_os("X360_TRACE_IO").is_some() {
                                        println!("[io] read {} off={at:#x} len={len:#x} -> {:#x} bytes into {buf:#010x}", entry.path, data.len());
                                    }
                                    mem.write_bytes(buf, &data);
                                    if let Some((_, pos)) = self.objects.file_mut(h) {
                                        *pos = at + data.len() as u64;
                                    }
                                    io_status(mem, iosb, STATUS_SUCCESS, data.len() as u32);
                                    STATUS_SUCCESS
                                }
                                _ => {
                                    io_status(mem, iosb, STATUS_NO_SUCH_DEVICE, 0);
                                    STATUS_NO_SUCH_DEVICE
                                }
                            }
                        }
                    }
                };
                if status == STATUS_SUCCESS && event != 0 {
                    if let Some(addr) = self.objects.dispatcher_addr(event) {
                        crate::dispatcher::set_event(mem, addr);
                    }
                }
                cpu.gpr[3] = status as u64;
                Ok(true)
            }
            "NtQueryInformationFile" | "NtSetInformationFile" => {
                // (handle, IO_STATUS_BLOCK*, info buffer, length, class)
                let (h, iosb, info, class) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32, cpu.gpr[7] as u32);
                let setting = name == "NtSetInformationFile";
                let status = match self.objects.file_mut(h) {
                    None => STATUS_INVALID_HANDLE,
                    Some((entry, pos)) => {
                        let size = entry.size as u64;
                        let dir = entry.is_dir;
                        let alloc = (size + 2047) & !2047;
                        match (class, setting) {
                            (14, true) => {
                                *pos = mem.read64(info); // FilePositionInformation
                                STATUS_SUCCESS
                            }
                            (14, false) => {
                                mem.write64(info, *pos);
                                STATUS_SUCCESS
                            }
                            (5, false) => {
                                // FileStandardInformation: alloc u64, eof u64, links u32, delete pending u8, directory u8
                                mem.write64(info, alloc);
                                mem.write64(info + 8, size);
                                mem.write32(info + 16, 1);
                                mem.write8(info + 20, 0);
                                mem.write8(info + 21, dir as u8);
                                STATUS_SUCCESS
                            }
                            (4, false) => {
                                // FileBasicInformation: 4 times (zero) + attributes
                                for off in (0..32).step_by(8) {
                                    mem.write64(info + off, 0);
                                }
                                mem.write32(info + 32, if dir { 0x10 } else { 0x81 });
                                STATUS_SUCCESS
                            }
                            (34, false) => {
                                // FileNetworkOpenInformation
                                for off in (0..56).step_by(8) {
                                    mem.write64(info + off, 0);
                                }
                                mem.write64(info + 32, alloc);
                                mem.write64(info + 40, size);
                                mem.write32(info + 48, if dir { 0x10 } else { 0x81 });
                                STATUS_SUCCESS
                            }
                            (_, true) => STATUS_SUCCESS, // ignore other set-info classes on a read-only disc
                            _ => {
                                println!("note: {name} class {class} not implemented");
                                STATUS_INVALID_INFO_CLASS
                            }
                        }
                    }
                };
                io_status(mem, iosb, status, 0);
                cpu.gpr[3] = status as u64;
                Ok(true)
            }
            "NtQueryDirectoryFile" => {
                // (handle, event, apc, apc ctx, IO_STATUS_BLOCK*, buffer, length, name pattern*, restart scan [stack])
                // The 360 variant has no info-class argument: entries are always FileDirectoryInformation.
                let h = cpu.gpr[3] as u32;
                let (iosb, buf, len, pattern_ptr) = (cpu.gpr[7] as u32, cpu.gpr[8] as u32, cpu.gpr[9] as u32, cpu.gpr[10] as u32);
                let (class, single) = (1u32, false);
                let restart = mem.read32(cpu.gpr[1] as u32 + 0x54) != 0;
                let pattern = if pattern_ptr == 0 {
                    String::new()
                } else {
                    let (l, b) = (mem.read16(pattern_ptr) as u32, mem.read32(pattern_ptr + 4));
                    (0..l).map(|i| mem.read8(b + i) as char).collect()
                };
                if self.trace_calls {
                    println!("note: NtQueryDirectoryFile class={class} single={single} restart={restart} pattern={pattern:?} len={len}");
                }
                let dir = match self.objects.file_mut(h) {
                    Some((e, _)) if e.is_dir => Some(e.path.clone()),
                    _ => None,
                };
                let status = match dir {
                    None => STATUS_INVALID_HANDLE,
                    Some(_) if class != 1 => {
                        println!("note: NtQueryDirectoryFile class {class} not implemented");
                        STATUS_INVALID_INFO_CLASS
                    }
                    Some(path) => {
                        let mut kids = self.vfs.as_ref().map(|v| v.children(&path)).unwrap_or_default();
                        kids.sort_by_key(|k| k.path.to_lowercase());
                        let leaf = |p: &str| p.rsplit('/').next().unwrap_or(p).to_string();
                        let wild = pattern.is_empty() || pattern == "*" || pattern == "*.*";
                        kids.retain(|k| wild || leaf(&k.path).eq_ignore_ascii_case(&pattern));
                        let (_, pos) = self.objects.file_mut(h).unwrap();
                        if restart {
                            *pos = 0;
                        }
                        let mut idx = *pos as usize;
                        let (mut off, mut prev, mut written) = (0u32, None::<u32>, 0);
                        while idx < kids.len() {
                            let k = &kids[idx];
                            let name = leaf(&k.path);
                            let need = (64 + name.len() as u32 + 7) & !7;
                            if off + need > len {
                                break;
                            }
                            let at = buf + off;
                            for o in (0..64).step_by(4) {
                                mem.write32(at + o, 0);
                            }
                            mem.write32(at + 4, idx as u32);
                            let alloc = (k.size as u64 + 2047) & !2047;
                            mem.write64(at + 40, if k.is_dir { 0 } else { k.size as u64 });
                            mem.write64(at + 48, if k.is_dir { 0 } else { alloc });
                            mem.write32(at + 56, if k.is_dir { 0x10 } else { 0x81 });
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
                            if single {
                                break;
                            }
                        }
                        *self.objects.file_mut(h).map(|(_, p)| p).unwrap() = idx as u64;
                        if written == 0 {
                            io_status(mem, iosb, 0x8000_0006, 0);
                            cpu.gpr[3] = if idx == 0 && !wild { 0xC000_000Fu32 } else { 0x8000_0006u32 } as u64;
                            return Ok(true);
                        }
                        io_status(mem, iosb, STATUS_SUCCESS, off);
                        STATUS_SUCCESS
                    }
                };
                cpu.gpr[3] = status as u64;
                Ok(true)
            }
            "NtWriteFile" | "NtDeleteFile" | "NtSetEvent_unused" => {
                cpu.gpr[3] = STATUS_NOT_IMPLEMENTED as u64; // disc is read-only
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
