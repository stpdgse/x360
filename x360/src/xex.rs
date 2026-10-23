//! XEX2 container: header parsing, AES decryption, basic/normal(LZX) decompression.
//! Produces the raw PE image; mapping + imports live in `loader.rs`.

use aes::cipher::{BlockModeDecrypt, KeyInit, KeyIvInit, block_padding::NoPadding, BlockCipherDecrypt};
use std::collections::HashMap;

pub const OPT_FILE_FORMAT: u32 = 0x0000_03FF;
pub const OPT_ENTRY_POINT: u32 = 0x0001_0100;
pub const OPT_IMAGE_BASE: u32 = 0x0001_0201;
pub const OPT_IMPORT_LIBS: u32 = 0x0001_03FF;
pub const OPT_STACK_SIZE: u32 = 0x0002_0200;

const RETAIL_KEY: [u8; 16] = [
    0x20, 0xB1, 0x85, 0xA5, 0x9D, 0x28, 0xFD, 0xC3, 0x40, 0x58, 0x3F, 0xBB, 0x08, 0x96, 0xBF, 0x91,
];
const DEVKIT_KEY: [u8; 16] = [0; 16];

fn be32(d: &[u8], o: usize) -> u32 {
    u32::from_be_bytes(d[o..o + 4].try_into().unwrap())
}
fn be16(d: &[u8], o: usize) -> u16 {
    u16::from_be_bytes(d[o..o + 2].try_into().unwrap())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    Basic,
    Normal { window_size: u32, first_block_size: u32 },
}

pub struct Xex {
    pub file: Vec<u8>,
    pub module_flags: u32,
    /// File offset of the (encrypted/compressed) PE payload.
    pub data_offset: usize,
    pub opt: HashMap<u32, u32>,
    pub image_size: u32,
    pub load_address: u32,
    pub file_key: [u8; 16],
    pub encrypted: bool,
    pub compression: Compression,
    /// Basic compression: (data_size, zero_size) pairs.
    pub basic_blocks: Vec<(u32, u32)>,
}

impl Xex {
    pub fn parse(file: Vec<u8>) -> Result<Self, String> {
        if file.len() < 24 || &file[0..4] != b"XEX2" {
            return Err("not an XEX2 file".into());
        }
        let module_flags = be32(&file, 4);
        let data_offset = be32(&file, 8) as usize;
        let sec_off = be32(&file, 16) as usize;
        let count = be32(&file, 20) as usize;
        let mut opt = HashMap::new();
        for i in 0..count {
            let o = 24 + i * 8;
            if o + 8 > file.len() {
                return Err("optional header table overruns file".into());
            }
            opt.insert(be32(&file, o), be32(&file, o + 4));
        }
        if sec_off + 0x184 > file.len() {
            return Err("security info overruns file".into());
        }
        let image_size = be32(&file, sec_off + 4);
        let load_address = be32(&file, sec_off + 0x110);
        let file_key: [u8; 16] = file[sec_off + 0x150..sec_off + 0x160].try_into().unwrap();

        let ff = *opt.get(&OPT_FILE_FORMAT).ok_or("missing file-format header")? as usize;
        let encrypted = be16(&file, ff + 4) != 0;
        let comp_type = be16(&file, ff + 6);
        let mut basic_blocks = Vec::new();
        let compression = match comp_type {
            0 => Compression::None,
            1 => {
                let size = be32(&file, ff) as usize;
                let n = (size - 8) / 8;
                for i in 0..n {
                    basic_blocks.push((be32(&file, ff + 8 + i * 8), be32(&file, ff + 12 + i * 8)));
                }
                Compression::Basic
            }
            2 => Compression::Normal {
                window_size: be32(&file, ff + 8),
                first_block_size: be32(&file, ff + 12),
            },
            t => return Err(format!("unknown compression type {t}")),
        };
        Ok(Xex { file, module_flags, data_offset, opt, image_size, load_address, file_key, encrypted, compression, basic_blocks })
    }

    /// Raw bytes of an optional header that stores its data out-of-line (size prefix at offset).
    pub fn opt_blob(&self, id: u32) -> Option<&[u8]> {
        let off = *self.opt.get(&id)? as usize;
        let size = be32(&self.file, off) as usize;
        self.file.get(off..off + size)
    }

    fn decrypt_payload(&self, key: &[u8; 16]) -> Vec<u8> {
        let mut data = self.file[self.data_offset..].to_vec();
        if !self.encrypted {
            return data;
        }
        // File key is itself AES-ECB-encrypted with the master key.
        let mut session = aes::Block::try_from(&self.file_key[..]).unwrap();
        aes::Aes128::new(key.into()).decrypt_block(&mut session);
        let n = data.len() & !15;
        let dec = cbc::Decryptor::<aes::Aes128>::new(&session, &[0u8; 16].into());
        dec.decrypt_padded::<NoPadding>(&mut data[..n]).expect("block aligned");
        data
    }

    /// Decrypt + decompress into the raw PE file. Tries retail then devkit key.
    pub fn extract_image(&self) -> Result<Vec<u8>, String> {
        let mut last = String::from("no key tried");
        for key in [&RETAIL_KEY, &DEVKIT_KEY] {
            let payload = self.decrypt_payload(key);
            match self.decompress(payload) {
                Ok(img) if img.starts_with(b"MZ") => return Ok(img),
                Ok(_) => last = "decrypted image does not start with MZ (wrong key?)".into(),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn decompress(&self, payload: Vec<u8>) -> Result<Vec<u8>, String> {
        match self.compression {
            Compression::None => Ok(payload),
            Compression::Basic => {
                let mut out = Vec::with_capacity(self.image_size as usize);
                let mut p = 0usize;
                for &(data, zero) in &self.basic_blocks {
                    let end = p + data as usize;
                    out.extend_from_slice(payload.get(p..end).ok_or("basic block overruns payload")?);
                    out.resize(out.len() + zero as usize, 0);
                    p = end;
                }
                Ok(out)
            }
            Compression::Normal { window_size, first_block_size } => {
                self.decompress_lzx(&payload, window_size, first_block_size)
            }
        }
    }

    fn decompress_lzx(&self, payload: &[u8], window: u32, first_block: u32) -> Result<Vec<u8>, String> {
        use lzxd::{Lzxd, WindowSize};
        let ws = match window {
            0x8000 => WindowSize::KB32,
            0x10000 => WindowSize::KB64,
            0x20000 => WindowSize::KB128,
            0x40000 => WindowSize::KB256,
            0x80000 => WindowSize::KB512,
            0x100000 => WindowSize::MB1,
            0x200000 => WindowSize::MB2,
            0x400000 => WindowSize::MB4,
            0x800000 => WindowSize::MB8,
            0x1000000 => WindowSize::MB16,
            0x2000000 => WindowSize::MB32,
            w => return Err(format!("unsupported LZX window size {w:#x}")),
        };
        let mut lzx = Lzxd::new(ws);
        let total = self.image_size as usize;
        let mut out: Vec<u8> = Vec::with_capacity(total);
        let mut p = 0usize;
        let mut block_size = first_block as usize;
        // Each block: next_size u32, hash[20], then u16-length-prefixed LZX chunks ending in 0.
        while block_size != 0 {
            let next_p = p + block_size;
            let next_size = be32(payload, p) as usize;
            p += 4 + 20;
            loop {
                if p + 2 > payload.len() {
                    return Err("LZX chunk table overruns payload".into());
                }
                let chunk = be16(payload, p) as usize;
                p += 2;
                if chunk == 0 {
                    break;
                }
                let data = payload.get(p..p + chunk).ok_or("LZX chunk overruns payload")?;
                p += chunk;
                let want = (total - out.len()).min(32768);
                let dec = lzx
                    .decompress_next(data, want)
                    .map_err(|e| format!("LZX error at output {:#x}: {e:?}", out.len()))?;
                out.extend_from_slice(dec);
            }
            p = next_p;
            block_size = next_size;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic unencrypted, basic-compressed XEX2 with one data block + one zero block.
    fn synth() -> Vec<u8> {
        let mut f = vec![0u8; 0x1000];
        f[0..4].copy_from_slice(b"XEX2");
        f[8..12].copy_from_slice(&0x800u32.to_be_bytes()); // data offset
        f[16..20].copy_from_slice(&0x200u32.to_be_bytes()); // security info
        f[20..24].copy_from_slice(&1u32.to_be_bytes()); // one opt header
        f[24..28].copy_from_slice(&OPT_FILE_FORMAT.to_be_bytes());
        f[28..32].copy_from_slice(&0x100u32.to_be_bytes());
        f[0x200 + 4..0x200 + 8].copy_from_slice(&6u32.to_be_bytes()); // image size
        f[0x200 + 0x110..0x200 + 0x114].copy_from_slice(&0x8200_0000u32.to_be_bytes());
        // file format info @0x100: size=16, enc=0, comp=1, one block (2 data, 4 zero)
        f[0x100..0x104].copy_from_slice(&16u32.to_be_bytes());
        f[0x106..0x108].copy_from_slice(&1u16.to_be_bytes());
        f[0x108..0x10C].copy_from_slice(&2u32.to_be_bytes());
        f[0x10C..0x110].copy_from_slice(&4u32.to_be_bytes());
        f[0x800..0x802].copy_from_slice(b"MZ");
        f
    }

    #[test]
    fn parses_and_expands_basic() {
        let x = Xex::parse(synth()).unwrap();
        assert_eq!(x.load_address, 0x8200_0000);
        assert!(!x.encrypted);
        assert_eq!(x.compression, Compression::Basic);
        let img = x.extract_image().unwrap();
        assert_eq!(img, b"MZ\0\0\0\0");
    }

    #[test]
    fn rejects_garbage() {
        assert!(Xex::parse(vec![0; 64]).is_err());
    }
}
