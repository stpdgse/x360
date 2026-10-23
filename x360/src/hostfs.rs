//! The console's writable storage (hard drive partitions, cache, content/saves), backed by a
//! directory on the host (`./hdd` or `X360_HDD`). Each guest device name becomes a subdirectory.
//!
//!   `cache:\valid.txt`                         -> hdd/cache/valid.txt
//!   `\Device\Harddisk0\Partition1\Content\x`   -> hdd/partition1/content/x
//!   `\Device\Harddisk0\Cache0\x`               -> hdd/cache0/x
//!   `hdd:\x`, `e:\x`, `content:\x`             -> hdd/hdd/x ...
//!
//! Names are lower-cased (the guest file system is case-insensitive) and `..` can never escape the root.

use std::path::{Path, PathBuf};

pub fn root() -> PathBuf {
    std::env::var_os("X360_HDD").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("hdd"))
}

/// Guest path -> host path under `root`, or None if it isn't a writable-storage device.
pub fn resolve(root: &Path, raw: &str) -> Option<PathBuf> {
    let mut s = raw.to_lowercase().replace('/', "\\");
    if let Some(r) = s.strip_prefix("\\??\\") {
        s = r.to_string();
    }
    let (device, rest) = if let Some(r) = s.strip_prefix("\\device\\harddisk0\\") {
        let (dev, rest) = r.split_once('\\').unwrap_or((r, ""));
        (dev.to_string(), rest.to_string())
    } else if let Some(r) = s.strip_prefix("\\device\\") {
        let (dev, rest) = r.split_once('\\').unwrap_or((r, ""));
        (dev.to_string(), rest.to_string())
    } else if let Some((dev, rest)) = s.split_once(':') {
        if dev.is_empty() || dev.contains('\\') {
            return None;
        }
        (dev.to_string(), rest.to_string())
    } else {
        return None;
    };
    // the disc is handled elsewhere
    if matches!(device.as_str(), "game" | "d" | "cdrom0") {
        return None;
    }
    let mut p = root.join(device);
    for comp in rest.split('\\').filter(|c| !c.is_empty() && *c != "." && *c != "..") {
        p.push(comp);
    }
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_devices_to_subdirectories() {
        let r = Path::new("hdd");
        assert_eq!(resolve(r, "cache:\\valid.txt"), Some(r.join("cache").join("valid.txt")));
        assert_eq!(resolve(r, "\\Device\\Harddisk0\\Partition1\\Content\\X"), Some(r.join("partition1").join("content").join("x")));
        assert_eq!(resolve(r, "\\??\\CACHE1:\\a\\b"), Some(r.join("cache1").join("a").join("b")));
        assert_eq!(resolve(r, "hdd:"), Some(r.join("hdd")));
        assert_eq!(resolve(r, "game:\\x"), None);
        assert_eq!(resolve(r, "\\Device\\Cdrom0\\x"), None);
        assert_eq!(resolve(r, "relative\\path"), None);
    }

    #[test]
    fn cannot_escape_the_root() {
        let r = Path::new("hdd");
        assert_eq!(resolve(r, "cache:\\..\\..\\secret"), Some(r.join("cache").join("secret")));
    }
}
