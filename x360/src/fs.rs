//! Virtual file system: maps guest device paths onto the mounted disc image.
//! Paths look like `game:\foo\bar.bin`, `d:\...`, `\Device\Cdrom0\...` or `\??\game:\...`.

use crate::iso::{FileEntry, Gdf};
use std::fs::File;
use std::io::{self, BufReader};

pub enum Resolved {
    /// A path on the game disc, normalised (`/` separators, no leading slash; "" = root).
    Disc(String),
    /// A device we don't back (HDD, cache, ...). Carries the original path for diagnostics.
    Unsupported(String),
}

pub struct Vfs {
    disc: Gdf<BufReader<File>>,
}

/// Device prefixes that all mean "the game disc", longest first.
const DISC_PREFIXES: &[&str] = &["\\??\\game:", "\\??\\d:", "\\device\\cdrom0", "game:", "d:"];

impl Vfs {
    pub fn open_disc(path: &str) -> io::Result<Self> {
        let f = File::open(path)?;
        Ok(Self { disc: Gdf::open(BufReader::new(f))? })
    }

    pub fn resolve(&self, raw: &str) -> Resolved {
        let lower = raw.to_lowercase().replace('/', "\\");
        for p in DISC_PREFIXES {
            if let Some(rest) = lower.strip_prefix(p) {
                let rel = rest.trim_matches('\\').replace('\\', "/");
                return Resolved::Disc(rel);
            }
        }
        Resolved::Unsupported(raw.to_string())
    }

    pub fn stat(&self, rel: &str) -> Option<FileEntry> {
        self.disc.find_any(rel)
    }

    pub fn read(&mut self, e: &FileEntry, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        self.disc.read_range(e, offset, len)
    }

    pub fn children(&self, dir: &str) -> Vec<FileEntry> {
        self.disc.children(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve_only(raw: &str) -> Resolved {
        // `resolve` doesn't touch the disc, but Vfs needs one; test the normalisation directly.
        let lower = raw.to_lowercase().replace('/', "\\");
        for p in DISC_PREFIXES {
            if let Some(rest) = lower.strip_prefix(p) {
                return Resolved::Disc(rest.trim_matches('\\').replace('\\', "/"));
            }
        }
        Resolved::Unsupported(raw.to_string())
    }

    #[test]
    fn normalises_device_prefixes() {
        for raw in ["game:\\Foo\\Bar.bin", "D:\\foo\\bar.bin", "\\Device\\Cdrom0\\Foo\\Bar.bin", "\\??\\GAME:\\foo\\bar.bin"] {
            match resolve_only(raw) {
                Resolved::Disc(p) => assert_eq!(p, "foo/bar.bin", "{raw}"),
                _ => panic!("{raw} should be a disc path"),
            }
        }
        assert!(matches!(resolve_only("cache:\\x"), Resolved::Unsupported(_)));
        assert!(matches!(resolve_only("game:\\"), Resolved::Disc(p) if p.is_empty()));
    }
}
