//! Xbox 360 game disc (GDF / XISO) reader. Generic over any `Read + Seek`.

use std::io::{self, Read, Seek, SeekFrom};

const SECTOR: u64 = 2048;
const MAGIC: &[u8; 20] = b"MICROSOFT*XBOX*MEDIA";
/// Known game-partition start offsets: raw, XGD3, XGD2, XGD1.
const PARTITION_OFFSETS: [u64; 4] = [0, 0x0208_0000, 0x0FD9_0000, 0x1830_0000];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub path: String,
    pub sector: u32,
    pub size: u32,
    pub is_dir: bool,
}

pub struct Gdf<R: Read + Seek> {
    r: R,
    base: u64,
    pub files: Vec<FileEntry>,
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

impl<R: Read + Seek> Gdf<R> {
    pub fn open(mut r: R) -> io::Result<Self> {
        let mut found = None;
        for off in PARTITION_OFFSETS {
            let mut hdr = [0u8; 28];
            if r.seek(SeekFrom::Start(off + 32 * SECTOR)).is_err() || r.read_exact(&mut hdr).is_err() {
                continue;
            }
            if &hdr[..20] == MAGIC {
                let root_sector = u32::from_le_bytes(hdr[20..24].try_into().unwrap());
                let root_size = u32::from_le_bytes(hdr[24..28].try_into().unwrap());
                found = Some((off, root_sector, root_size));
                break;
            }
        }
        let (base, root_sector, root_size) = found.ok_or_else(|| bad("no GDF volume descriptor found"))?;
        let mut gdf = Gdf { r, base, files: Vec::new() };
        gdf.walk_dir(root_sector, root_size, "")?;
        Ok(gdf)
    }

    fn walk_dir(&mut self, sector: u32, size: u32, prefix: &str) -> io::Result<()> {
        if size == 0 {
            return Ok(());
        }
        let data = self.read_raw(sector, size)?;
        // Entries are stored back to back (each padded to 4 bytes) and never straddle a sector;
        // 0xFF padding fills the rest of a sector. A linear scan visits every entry, unlike
        // following the sort tree, which would silently drop anything the pointers miss.
        let mut off = 0usize;
        while off + 14 <= data.len() {
            let left = u16::from_le_bytes([data[off], data[off + 1]]);
            let right = u16::from_le_bytes([data[off + 2], data[off + 3]]);
            if left == 0xFFFF && right == 0xFFFF && data[off + 4..off + 12] == [0xFF; 8] {
                // padding: continue at the next sector boundary
                off = (off / SECTOR as usize + 1) * SECTOR as usize;
                continue;
            }
            let esector = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
            let esize = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
            let attr = data[off + 12];
            let nlen = data[off + 13] as usize;
            if off + 14 + nlen > data.len() {
                return Err(bad("directory entry overruns directory"));
            }
            let name = String::from_utf8_lossy(&data[off + 14..off + 14 + nlen]).into_owned();
            let path = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
            let is_dir = attr & 0x10 != 0;
            self.files.push(FileEntry { path: path.clone(), sector: esector, size: esize, is_dir });
            if is_dir {
                self.walk_dir(esector, esize, &path)?;
            }
            off += (14 + nlen + 3) & !3;
        }
        Ok(())
    }

    fn read_raw(&mut self, sector: u32, size: u32) -> io::Result<Vec<u8>> {
        self.r.seek(SeekFrom::Start(self.base + sector as u64 * SECTOR))?;
        let mut buf = vec![0u8; size as usize];
        self.r.read_exact(&mut buf)?;
        Ok(buf)
    }

    pub fn find(&self, path: &str) -> Option<&FileEntry> {
        self.files.iter().find(|f| !f.is_dir && f.path.eq_ignore_ascii_case(path))
    }

    /// Look up a file *or directory* by path (case-insensitive, `/` separators; "" is the root).
    pub fn find_any(&self, path: &str) -> Option<FileEntry> {
        if path.is_empty() {
            return Some(FileEntry { path: String::new(), sector: 0, size: 0, is_dir: true });
        }
        self.files.iter().find(|f| f.path.eq_ignore_ascii_case(path)).cloned()
    }

    /// Read up to `len` bytes of a file starting at `offset` (clamped to the file size).
    pub fn read_range(&mut self, e: &FileEntry, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        if offset >= e.size as u64 {
            return Ok(Vec::new());
        }
        let n = len.min((e.size as u64 - offset) as usize);
        self.r.seek(SeekFrom::Start(self.base + e.sector as u64 * SECTOR + offset))?;
        let mut buf = vec![0u8; n];
        self.r.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Immediate children of a directory path ("" = root), as (name, entry).
    pub fn children(&self, dir: &str) -> Vec<FileEntry> {
        let prefix = if dir.is_empty() { String::new() } else { format!("{}/", dir.to_lowercase()) };
        self.files
            .iter()
            .filter(|f| {
                let p = f.path.to_lowercase();
                p.starts_with(&prefix) && !p[prefix.len()..].contains('/')
            })
            .cloned()
            .collect()
    }

    pub fn read_file(&mut self, path: &str) -> io::Result<Vec<u8>> {
        let e = self.find(path).ok_or_else(|| bad("file not found on disc"))?.clone();
        self.read_raw(e.sector, e.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a tiny image: descriptor at sector 32, root dir at sector 33, file at sector 34.
    fn tiny_image() -> Vec<u8> {
        let mut img = vec![0u8; 40 * SECTOR as usize];
        let d = 32 * SECTOR as usize;
        img[d..d + 20].copy_from_slice(MAGIC);
        img[d + 20..d + 24].copy_from_slice(&33u32.to_le_bytes());
        img[d + 24..d + 28].copy_from_slice(&2048u32.to_le_bytes());
        let e = 33 * SECTOR as usize;
        // one entry: no children, sector 34, size 5, attr 0, name "default.xex"
        img[e..e + 2].copy_from_slice(&0u16.to_le_bytes());
        img[e + 2..e + 4].copy_from_slice(&0u16.to_le_bytes());
        img[e + 4..e + 8].copy_from_slice(&34u32.to_le_bytes());
        img[e + 8..e + 12].copy_from_slice(&5u32.to_le_bytes());
        img[e + 12] = 0x20;
        img[e + 13] = 11;
        img[e + 14..e + 25].copy_from_slice(b"default.xex");
        // the rest of a directory sector is 0xFF padding on real discs
        for b in &mut img[e + 28..e + SECTOR as usize] {
            *b = 0xFF;
        }
        let f = 34 * SECTOR as usize;
        img[f..f + 5].copy_from_slice(b"hello");
        img
    }

    #[test]
    fn finds_and_reads_file() {
        let mut g = Gdf::open(Cursor::new(tiny_image())).unwrap();
        assert_eq!(g.files.len(), 1);
        assert_eq!(g.read_file("DEFAULT.XEX").unwrap(), b"hello");
    }
}
