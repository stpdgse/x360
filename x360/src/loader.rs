//! Maps a decoded XEX into guest memory and patches kernel imports.

use crate::memory::GuestMemory;
use crate::xex::{Xex, OPT_ENTRY_POINT, OPT_IMPORT_LIBS, OPT_STACK_SIZE};
use xenon_cpu::memory::Bus;

fn le16(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(d[o..o + 2].try_into().unwrap())
}
fn le32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}
fn be32(d: &[u8], o: usize) -> u32 {
    u32::from_be_bytes(d[o..o + 4].try_into().unwrap())
}
fn be16(d: &[u8], o: usize) -> u16 {
    u16::from_be_bytes(d[o..o + 2].try_into().unwrap())
}

#[derive(Debug, Clone)]
pub struct ImportedFunction {
    pub library: String,
    pub lib_index: u16,
    pub ordinal: u16,
    /// Address of the patched thunk in guest memory.
    pub thunk: u32,
}

#[derive(Debug)]
pub struct LoadedModule {
    pub base: u32,
    pub size: u32,
    pub entry: u32,
    pub stack_size: u32,
    /// Guest address of this executable's module-info struct (what XexExecutableModuleHandle points at).
    pub module_info: u32,
    pub imports: Vec<ImportedFunction>,
    /// Import library names, indexed by `ImportedFunction::lib_index`.
    pub libraries: Vec<String>,
    /// (library, ordinal, record address) for data imports we have not backed yet.
    pub variable_imports: Vec<(String, u16, u32)>,
}

pub fn load(xex: &Xex, mem: &mut GuestMemory) -> Result<LoadedModule, String> {
    let img = xex.extract_image()?;
    let base = xex.load_address;

    // --- PE mapping (headers are little-endian regardless of the big-endian code) ---
    let pe = le32(&img, 0x3C) as usize;
    if img.get(pe..pe + 4) != Some(b"PE\0\0") {
        return Err("missing PE signature".into());
    }
    let nsec = le16(&img, pe + 6) as usize;
    let opt_size = le16(&img, pe + 20) as usize;
    let opt = pe + 24;
    let size_of_headers = le32(&img, opt + 60) as usize;
    let map_size = xex.image_size.max(le32(&img, opt + 56));
    mem.map(base, map_size);
    // The decoded XEX payload is already laid out as the in-memory image
    // (sections at their virtual addresses), so it is copied verbatim.
    let _ = (size_of_headers, opt_size);
    mem.write_bytes(base, &img);

    let entry = *xex.opt.get(&OPT_ENTRY_POINT).ok_or("no entry point header")?;
    println!(
        "pe: image_base={:#x} pe_entry_rva={:#x} (-> {:#x}) sections={} xex_entry={:#x}",
        le32(&img, opt + 28),
        le32(&img, opt + 16),
        base.wrapping_add(le32(&img, opt + 16)),
        nsec,
        entry
    );
    let stack_size = xex.opt.get(&OPT_STACK_SIZE).copied().unwrap_or(0x10_0000);

    let mut module = LoadedModule { base, size: map_size, entry, stack_size, module_info: 0, imports: vec![], libraries: vec![], variable_imports: vec![] };
    patch_imports(xex, mem, &mut module)?;
    Ok(module)
}

fn patch_imports(xex: &Xex, mem: &mut GuestMemory, module: &mut LoadedModule) -> Result<(), String> {
    let Some(blob) = xex.opt_blob(OPT_IMPORT_LIBS) else { return Ok(()) };
    let strtab_size = be32(blob, 4) as usize;
    let lib_count = be32(blob, 8) as usize;
    let strtab = &blob[12..12 + strtab_size];
    let names: Vec<String> = strtab
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();
    let mut p = 12 + strtab_size;
    for lib_idx in 0..lib_count {
        let size = be32(blob, p) as usize;
        let name_index = (be16(blob, p + 36) & 0xFF) as usize;
        let count = be16(blob, p + 38) as usize;
        let lib = names.get(name_index).cloned().unwrap_or_else(|| format!("lib{name_index}"));
        module.libraries.push(lib.clone());
        for r in 0..count {
            let rec_addr = be32(blob, p + 40 + r * 4);
            let value = mem.read32(rec_addr);
            let kind = value >> 24;
            let ordinal = (value & 0xFFFF) as u16;
            match kind {
                0 => {
                    // Variable import: the record receives the address of kernel-owned storage.
                    let var = mem.alloc_kernel(0x1000);
                    if lib == "xboxkrnl.exe" && ordinal == 0x193 {
                        // XexExecutableModuleHandle: holds a pointer to the module's info struct.
                        let module_info = mem.alloc_kernel(0x1000);
                        mem.write32(var, module_info);
                        module.module_info = module_info;
                        // +0x58: pointer to the XEX header image in guest memory.
                        let hdr = mem.alloc_kernel(xex.data_offset as u32);
                        mem.write_bytes(hdr, &xex.file[..xex.data_offset]);
                        mem.write32(module_info + 0x58, hdr);
                    }
                    mem.write32(rec_addr, var);
                    module.variable_imports.push((lib.clone(), ordinal, rec_addr));
                }
                _ => {
                    // Function thunk -> `lis r11,lib ; ori r11,r11,ord ; sc ; blr`
                    mem.write32(rec_addr, 0x3D60_0000 | lib_idx as u32);
                    mem.write32(rec_addr + 4, 0x616B_0000 | ordinal as u32);
                    mem.write32(rec_addr + 8, 0x4400_0002);
                    mem.write32(rec_addr + 12, 0x4E80_0020);
                    module.imports.push(ImportedFunction {
                        library: lib.clone(),
                        lib_index: lib_idx as u16,
                        ordinal,
                        thunk: rec_addr,
                    });
                }
            }
        }
        p += size;
    }
    Ok(())
}
