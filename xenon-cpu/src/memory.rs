//! Guest memory. The Xenon is big-endian; all accessors take/return host values
//! and swap internally.

pub trait Bus {
    /// For the JIT: base of a table of one `*mut u8` per 4 KB page (null = unmapped, handled by the
    /// slow path). The table must stay at a fixed address. None = always use the access methods.
    fn page_table(&self) -> Option<*const *mut u8> {
        None
    }
    fn read8(&self, addr: u32) -> u8;
    fn write8(&mut self, addr: u32, val: u8);

    fn read16(&self, addr: u32) -> u16 {
        u16::from_be_bytes([self.read8(addr), self.read8(addr.wrapping_add(1))])
    }
    fn read32(&self, addr: u32) -> u32 {
        u32::from_be_bytes([
            self.read8(addr),
            self.read8(addr.wrapping_add(1)),
            self.read8(addr.wrapping_add(2)),
            self.read8(addr.wrapping_add(3)),
        ])
    }
    fn read64(&self, addr: u32) -> u64 {
        ((self.read32(addr) as u64) << 32) | self.read32(addr.wrapping_add(4)) as u64
    }
    fn write16(&mut self, addr: u32, val: u16) {
        for (i, b) in val.to_be_bytes().iter().enumerate() {
            self.write8(addr.wrapping_add(i as u32), *b);
        }
    }
    fn write32(&mut self, addr: u32, val: u32) {
        for (i, b) in val.to_be_bytes().iter().enumerate() {
            self.write8(addr.wrapping_add(i as u32), *b);
        }
    }
    fn write64(&mut self, addr: u32, val: u64) {
        self.write32(addr, (val >> 32) as u32);
        self.write32(addr.wrapping_add(4), val as u32);
    }
}

/// Simple flat RAM for early bring-up and tests. Out-of-range reads return 0.
pub struct FlatMemory {
    pub data: Vec<u8>,
}

impl FlatMemory {
    pub fn new(size: usize) -> Self {
        Self { data: vec![0; size] }
    }
}

impl Bus for FlatMemory {
    fn read8(&self, addr: u32) -> u8 {
        self.data.get(addr as usize).copied().unwrap_or(0)
    }
    fn write8(&mut self, addr: u32, val: u8) {
        if let Some(b) = self.data.get_mut(addr as usize) {
            *b = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_endian_roundtrip() {
        let mut m = FlatMemory::new(64);
        m.write32(0, 0xDEADBEEF);
        assert_eq!(m.read8(0), 0xDE);
        assert_eq!(m.read8(3), 0xEF);
        assert_eq!(m.read32(0), 0xDEADBEEF);
        m.write64(8, 0x0102030405060708);
        assert_eq!(m.read64(8), 0x0102030405060708);
        assert_eq!(m.read16(8), 0x0102);
    }
}
