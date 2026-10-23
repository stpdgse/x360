//! XAM user-facing services: the signed-in profile, the controller, and the notification queue.
//!
//! There is no window or physical pad yet, so input is *scripted*: `X360_PAD` holds a
//! comma-separated list of `frame:action` events keyed by the virtual vsync count, e.g.
//!   `X360_PAD="300:tap=START,600:+A,640:-A,900:signout,950:signin,1000:lx=20000"`
//! Actions: `+BTN`/`-BTN` press/release, `tap=BTN` press then release 3 frames later,
//! `lx|ly|rx|ry=N` stick axes, `lt|rt=N` triggers, `signin`/`signout` (user 0),
//! `connect`/`disconnect` (the pad), `guide` (toggle the Guide overlay notification).
//! Buttons: A B X Y UP DOWN LEFT RIGHT START BACK LB RB LS RS.

use crate::kernel::Kernel;
use crate::memory::GuestMemory;
use std::collections::VecDeque;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

pub const ERROR_DEVICE_NOT_CONNECTED: u64 = 0x48F;
pub const ERROR_EMPTY: u64 = 0x10D2;
pub const E_NO_SUCH_USER: u64 = 0x8007_0525;
pub const ERROR_INSUFFICIENT_BUFFER: u64 = 0x7A;
pub const ERROR_IO_PENDING: u64 = 0x3E5;
pub const ERROR_NO_MORE_FILES: u64 = 0x12;
pub const ERROR_INVALID_HANDLE: u64 = 0x6;

// notification ids (area index in the top bits; see ObjectTable::broadcast)
pub const XN_SYS_UI: u32 = 0x0000_0009;
pub const XN_SYS_SIGNINCHANGED: u32 = 0x0000_000A;
pub const XN_SYS_INPUTDEVICESCHANGED: u32 = 0x0000_0012;

pub const GAMERTAG: &str = "Player1";
/// Offline profile ids live in the 0xE000... range.
pub const XUID: u64 = 0xE000_0000_0000_0001;

const KEYDOWN: u16 = 1;
const KEYUP: u16 = 2;

fn button_bit(name: &str) -> Option<u16> {
    Some(match name.to_ascii_uppercase().as_str() {
        "UP" => 0x0001,
        "DOWN" => 0x0002,
        "LEFT" => 0x0004,
        "RIGHT" => 0x0008,
        "START" => 0x0010,
        "BACK" => 0x0020,
        "LS" => 0x0040,
        "RS" => 0x0080,
        "LB" => 0x0100,
        "RB" => 0x0200,
        "A" => 0x1000,
        "B" => 0x2000,
        "X" => 0x4000,
        "Y" => 0x8000,
        _ => return None,
    })
}

/// Virtual-key code for a button (VK_PAD_*), what XamInputGetKeystroke reports.
fn button_vk(bit: u16) -> u16 {
    match bit {
        0x1000 => 0x5800,
        0x2000 => 0x5801,
        0x4000 => 0x5802,
        0x8000 => 0x5803,
        0x0200 => 0x5804,
        0x0100 => 0x5805,
        0x0001 => 0x5810,
        0x0002 => 0x5811,
        0x0004 => 0x5812,
        0x0008 => 0x5813,
        0x0010 => 0x5814,
        0x0020 => 0x5815,
        0x0040 => 0x5816,
        _ => 0x5817,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadAction {
    Press(u16),
    Release(u16),
    Axis(Axis, i32),
    SignIn,
    SignOut,
    Connect,
    Disconnect,
    Guide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Lx,
    Ly,
    Rx,
    Ry,
    Lt,
    Rt,
}

/// Parse an `X360_PAD` script into (frame, action) pairs sorted by frame.
pub fn parse_script(text: &str) -> Result<Vec<(u64, PadAction)>, String> {
    let mut out = Vec::new();
    for item in text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (frame, act) = item.split_once(':').ok_or_else(|| format!("bad pad event '{item}' (want frame:action)"))?;
        let frame: u64 = frame.trim().parse().map_err(|_| format!("bad frame in '{item}'"))?;
        let act = act.trim();
        let btn = |n: &str| button_bit(n).ok_or_else(|| format!("unknown button '{n}'"));
        if let Some(n) = act.strip_prefix('+') {
            out.push((frame, PadAction::Press(btn(n)?)));
        } else if let Some(n) = act.strip_prefix('-') {
            out.push((frame, PadAction::Release(btn(n)?)));
        } else if let Some(n) = act.strip_prefix("tap=") {
            let b = btn(n)?;
            out.push((frame, PadAction::Press(b)));
            out.push((frame + 3, PadAction::Release(b)));
        } else if let Some((axis, v)) = act.split_once('=') {
            let axis = match axis {
                "lx" => Axis::Lx,
                "ly" => Axis::Ly,
                "rx" => Axis::Rx,
                "ry" => Axis::Ry,
                "lt" => Axis::Lt,
                "rt" => Axis::Rt,
                _ => return Err(format!("unknown action '{act}'")),
            };
            out.push((frame, PadAction::Axis(axis, v.trim().parse().map_err(|_| format!("bad value in '{item}'"))?)));
        } else {
            out.push((
                frame,
                match act {
                    "signin" => PadAction::SignIn,
                    "signout" => PadAction::SignOut,
                    "connect" => PadAction::Connect,
                    "disconnect" => PadAction::Disconnect,
                    "guide" => PadAction::Guide,
                    _ => return Err(format!("unknown action '{act}'")),
                },
            ));
        }
    }
    out.sort_by_key(|e| e.0); // stable: same-frame events keep script order
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keystroke {
    pub vk: u16,
    pub flags: u16,
}

pub struct XamState {
    pub signed_in: bool,
    pub pad_connected: bool,
    pub buttons: u16,
    pub lt: u8,
    pub rt: u8,
    pub thumbs: [i16; 4], // lx ly rx ry
    pub packet: u32,
    pub vibration: (u16, u16),
    pub guide_open: bool,
    keystrokes: VecDeque<Keystroke>,
    /// Keyboard state from the live window (OR-ed with the scripted state).
    pub live: Option<std::sync::Arc<crate::window::Shared>>,
    live_seen: (u16, [i16; 4], [u8; 2]),
    script: VecDeque<(u64, PadAction)>,
}

impl XamState {
    pub fn new() -> Self {
        let script = match std::env::var("X360_PAD") {
            Ok(text) => parse_script(&text).unwrap_or_else(|e| {
                eprintln!("X360_PAD ignored: {e}");
                Vec::new()
            }),
            Err(_) => Vec::new(),
        };
        Self::with_script(script)
    }

    pub fn with_script(script: Vec<(u64, PadAction)>) -> Self {
        Self {
            signed_in: true,
            pad_connected: true,
            buttons: 0,
            lt: 0,
            rt: 0,
            thumbs: [0; 4],
            packet: 1,
            vibration: (0, 0),
            guide_open: false,
            keystrokes: VecDeque::new(),
            live: None,
            live_seen: (0, [0; 4], [0; 2]),
            script: script.into(),
        }
    }

    pub fn signin_mask(&self) -> u32 {
        self.signed_in as u32
    }

    fn set_buttons(&mut self, new: u16) {
        let changed = self.buttons ^ new;
        if changed == 0 {
            return;
        }
        for bit in 0..16 {
            let b = 1u16 << bit;
            if changed & b != 0 {
                let flags = if new & b != 0 { KEYDOWN } else { KEYUP };
                self.keystrokes.push_back(Keystroke { vk: button_vk(b), flags });
            }
        }
        self.buttons = new;
        self.packet = self.packet.wrapping_add(1);
    }

    /// Apply every scripted event due at `frame`; returns notifications (id, param) to broadcast.
    pub fn advance(&mut self, frame: u64) -> Vec<(u32, u32)> {
        let mut notes = Vec::new();
        while self.script.front().is_some_and(|e| e.0 <= frame) {
            let (_, act) = self.script.pop_front().unwrap();
            match act {
                PadAction::Press(b) => self.set_buttons(self.buttons | b),
                PadAction::Release(b) => self.set_buttons(self.buttons & !b),
                PadAction::Axis(a, v) => {
                    match a {
                        Axis::Lx => self.thumbs[0] = v as i16,
                        Axis::Ly => self.thumbs[1] = v as i16,
                        Axis::Rx => self.thumbs[2] = v as i16,
                        Axis::Ry => self.thumbs[3] = v as i16,
                        Axis::Lt => self.lt = v as u8,
                        Axis::Rt => self.rt = v as u8,
                    }
                    self.packet = self.packet.wrapping_add(1);
                }
                PadAction::SignIn | PadAction::SignOut => {
                    let want = act == PadAction::SignIn;
                    if self.signed_in != want {
                        self.signed_in = want;
                        notes.push((XN_SYS_SIGNINCHANGED, self.signin_mask()));
                    }
                }
                PadAction::Connect | PadAction::Disconnect => {
                    let want = act == PadAction::Connect;
                    if self.pad_connected != want {
                        self.pad_connected = want;
                        notes.push((XN_SYS_INPUTDEVICESCHANGED, 0));
                    }
                }
                PadAction::Guide => {
                    self.guide_open = !self.guide_open;
                    notes.push((XN_SYS_UI, self.guide_open as u32));
                }
            }
        }
        notes
    }

    /// Fold the live keyboard into the pad: keystrokes for button changes, a new packet for any change.
    pub fn poll_live(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(l) = self.live.clone() else { return };
        let buttons = l.buttons.load(Relaxed) as u16;
        let thumbs: [i16; 4] = std::array::from_fn(|i| l.thumbs[i].load(Relaxed) as i16);
        let trig = [l.triggers[0].load(Relaxed) as u8, l.triggers[1].load(Relaxed) as u8];
        if (buttons, thumbs, trig) == self.live_seen {
            return;
        }
        let changed = self.live_seen.0 ^ buttons;
        for bit in 0..16 {
            let b = 1u16 << bit;
            if changed & b != 0 {
                let flags = if buttons & b != 0 { KEYDOWN } else { KEYUP };
                self.keystrokes.push_back(Keystroke { vk: button_vk(b), flags });
            }
        }
        self.live_seen = (buttons, thumbs, trig);
        self.packet = self.packet.wrapping_add(1);
    }

    pub fn pad_buttons(&self) -> u16 {
        self.buttons | self.live_seen.0
    }
    pub fn pad_thumbs(&self) -> [i16; 4] {
        std::array::from_fn(|i| if self.live_seen.1[i] != 0 { self.live_seen.1[i] } else { self.thumbs[i] })
    }
    pub fn pad_triggers(&self) -> (u8, u8) {
        (self.lt.max(self.live_seen.2[0]), self.rt.max(self.live_seen.2[1]))
    }

    pub fn next_keystroke(&mut self) -> Option<Keystroke> {
        self.keystrokes.pop_front()
    }
}

fn user_ok(user: u32) -> bool {
    user == 0 || user == 0xFF
}

impl Kernel {
    /// Apply scripted pad events that have come due (called from the scheduler loop).
    pub fn xam_tick(&mut self) {
        for (id, param) in self.xam.advance(self.vsyncs) {
            self.objects.broadcast(id, param);
        }
    }

    /// XAM calls: returns false if `name` isn't one of ours.
    pub fn xam_call(&mut self, name: &str, cpu: &mut Cpu, mem: &mut GuestMemory) -> bool {
        let (a, b, c) = (cpu.gpr[3] as u32, cpu.gpr[4] as u32, cpu.gpr[5] as u32);
        self.xam_tick();
        match name {
            // ---- controller ----
            "XamInputGetState" => {
                // (user, flags, XINPUT_STATE *out)
                if user_ok(a) && self.xam.pad_connected {
                    self.xam.poll_live();
                    let x = &self.xam;
                    let (lt, rt) = x.pad_triggers();
                    mem.write32(c, x.packet);
                    mem.write16(c + 4, x.pad_buttons());
                    mem.write8(c + 6, lt);
                    mem.write8(c + 7, rt);
                    for (i, v) in x.pad_thumbs().iter().enumerate() {
                        mem.write16(c + 8 + i as u32 * 2, *v as u16);
                    }
                    cpu.gpr[3] = 0;
                } else {
                    cpu.gpr[3] = ERROR_DEVICE_NOT_CONNECTED;
                }
            }
            "XamInputGetCapabilities" => {
                // (user, flags, XINPUT_CAPABILITIES *out): a standard gamepad with everything present
                if user_ok(a) && self.xam.pad_connected {
                    for i in 0..6 {
                        mem.write32(c + i * 4, 0);
                    }
                    mem.write8(c, 1); // XINPUT_DEVTYPE_GAMEPAD
                    mem.write8(c + 1, 1); // XINPUT_DEVSUBTYPE_GAMEPAD
                    mem.write16(c + 4, 0xF3FF); // every button
                    mem.write8(c + 6, 0xFF);
                    mem.write8(c + 7, 0xFF);
                    for i in 0..4 {
                        mem.write16(c + 8 + i * 2, 0xFFFF);
                    }
                    mem.write16(c + 16, 0xFFFF); // vibration motors
                    mem.write16(c + 18, 0xFFFF);
                    cpu.gpr[3] = 0;
                } else {
                    cpu.gpr[3] = ERROR_DEVICE_NOT_CONNECTED;
                }
            }
            "XamInputSetState" => {
                // (user, flags, XINPUT_VIBRATION *in)
                if user_ok(a) && self.xam.pad_connected {
                    self.xam.vibration = (mem.read16(c), mem.read16(c + 2));
                    cpu.gpr[3] = 0;
                } else {
                    cpu.gpr[3] = ERROR_DEVICE_NOT_CONNECTED;
                }
            }
            "XamInputGetKeystroke" | "XamInputGetKeystrokeEx" => {
                // Ex takes a pointer to the user index (0xFF = any) and writes back the real one.
                self.xam.poll_live();
                let user = if name == "XamInputGetKeystrokeEx" { mem.read32(a) } else { a };
                let out = c;
                match (user_ok(user) && self.xam.pad_connected, self.xam.next_keystroke()) {
                    (true, Some(k)) => {
                        mem.write16(out, k.vk);
                        mem.write16(out + 2, 0); // no unicode for pad buttons
                        mem.write16(out + 4, k.flags);
                        mem.write8(out + 6, 0);
                        mem.write8(out + 7, 0);
                        if name == "XamInputGetKeystrokeEx" {
                            mem.write32(a, 0);
                        }
                        cpu.gpr[3] = 0;
                    }
                    (true, None) => cpu.gpr[3] = ERROR_EMPTY,
                    (false, _) => cpu.gpr[3] = ERROR_DEVICE_NOT_CONNECTED,
                }
            }
            // ---- profile ----
            "XamUserGetSigninState" => {
                // 0 not signed in, 1 signed in locally, 2 signed in to Live
                cpu.gpr[3] = (a == 0 && self.xam.signed_in) as u64;
            }
            "XamUserGetSigninInfo" => {
                // (user, flags, XUSER_SIGNIN_INFO *out): xuid u64, flags, state, guest num, sponsor, name[16]
                if a == 0 && self.xam.signed_in {
                    for off in (0..0x28).step_by(4) {
                        mem.write32(c + off, 0);
                    }
                    mem.write32(c, (XUID >> 32) as u32);
                    mem.write32(c + 4, XUID as u32);
                    mem.write32(c + 0x0C, 1); // signed in locally
                    for (i, ch) in GAMERTAG.bytes().enumerate() {
                        mem.write8(c + 0x18 + i as u32, ch);
                    }
                    cpu.gpr[3] = 0;
                } else {
                    if c != 0 {
                        for off in (0..0x28).step_by(4) {
                            mem.write32(c + off, 0);
                        }
                    }
                    cpu.gpr[3] = E_NO_SUCH_USER;
                }
            }
            "XamUserGetXUID" => {
                // (user, type mask, u64 *out)
                if a == 0 && self.xam.signed_in {
                    mem.write32(c, (XUID >> 32) as u32);
                    mem.write32(c + 4, XUID as u32);
                    cpu.gpr[3] = 0;
                } else {
                    cpu.gpr[3] = E_NO_SUCH_USER;
                }
            }
            "XamUserGetIndexFromXUID" => {
                // (u64 *xuid, flags, u32 *index out)
                let xuid = ((mem.read32(a) as u64) << 32) | mem.read32(a + 4) as u64;
                if xuid == XUID && self.xam.signed_in {
                    mem.write32(c, 0);
                    cpu.gpr[3] = 0;
                } else {
                    cpu.gpr[3] = E_NO_SUCH_USER;
                }
            }
            "XamUserGetName" => {
                // (user, char *buf, length)
                if a == 0 && self.xam.signed_in && c > 0 {
                    let n = GAMERTAG.len().min(c as usize - 1);
                    for (i, ch) in GAMERTAG.bytes().take(n).enumerate() {
                        mem.write8(b + i as u32, ch);
                    }
                    mem.write8(b + n as u32, 0);
                    cpu.gpr[3] = 0;
                } else {
                    cpu.gpr[3] = E_NO_SUCH_USER;
                }
            }
            "XamUserGetGamerTag" => {
                // (user, wchar_t *buf, length in chars)
                if a == 0 && self.xam.signed_in && c > 0 {
                    let n = GAMERTAG.len().min(c as usize - 1);
                    for (i, ch) in GAMERTAG.bytes().take(n).enumerate() {
                        mem.write16(b + i as u32 * 2, ch as u16);
                    }
                    mem.write16(b + n as u32 * 2, 0);
                    cpu.gpr[3] = 0;
                } else {
                    cpu.gpr[3] = E_NO_SUCH_USER;
                }
            }
            "XamUserGetUserIndexMask" | "XamUserGetRequestedUserIndexMask" => {
                cpu.gpr[3] = self.xam.signin_mask() as u64;
            }
            "XamUserCheckPrivilege" => {
                // (user, privilege, BOOL *out): offline profile with every privilege granted
                mem.write32(c, (a == 0 && self.xam.signed_in) as u32);
                cpu.gpr[3] = 0;
            }
            "XamUserIsOnlineEnabled" | "XamUserIsGuest" | "XamUserIsParentalControlled" | "XamUserIsPartial" => {
                cpu.gpr[3] = 0;
            }
            "XamUserReadProfileSettings" => {
                // (title id, user index, xuid count, xuids*, setting count, setting ids*, u32 *buffer size,
                //  buffer*, overlapped* [stack]). With no (or a too small) buffer it reports the size needed
                // and fails with ERROR_INSUFFICIENT_BUFFER; the title then allocates and calls again.
                // Layout (reconstructed, unverified): header {count, entries*} followed by one 0x28-byte
                // entry per setting: +0 source (0 = no value / 1 = default / 2 = user), +8 user index,
                // +0x10 setting id, +0x14 data type (0 = none). Every setting is reported as having no value.
                let (count, ids, size_ptr, buf) = (cpu.gpr[7] as u32, cpu.gpr[8] as u32, cpu.gpr[9] as u32, cpu.gpr[10] as u32);
                const ENTRY: u32 = 0x28;
                let needed = 8 + count * ENTRY;
                let have = if size_ptr != 0 { mem.read32(size_ptr) } else { 0 };
                if size_ptr != 0 {
                    mem.write32(size_ptr, needed);
                }
                if buf == 0 || have < needed {
                    cpu.gpr[3] = ERROR_INSUFFICIENT_BUFFER;
                } else if !(b == 0 || b == 0xFF) {
                    cpu.gpr[3] = E_NO_SUCH_USER;
                } else {
                    for off in (0..needed).step_by(4) {
                        mem.write32(buf + off, 0);
                    }
                    mem.write32(buf, count);
                    mem.write32(buf + 4, buf + 8);
                    for i in 0..count {
                        let e = buf + 8 + i * ENTRY;
                        mem.write32(e + 0x10, mem.read32(ids + i * 4));
                    }
                    let overlapped = mem.read32(cpu.gpr[1] as u32 + 0x54);
                    if overlapped != 0 {
                        // complete immediately: Internal = status, InternalHigh = size, signal the event
                        mem.write32(overlapped, 0);
                        mem.write32(overlapped + 4, needed);
                        let event = mem.read32(overlapped + 0x10);
                        if let Some(addr) = self.objects.dispatcher_addr(event) {
                            crate::dispatcher::set_event(mem, addr);
                        }
                        cpu.gpr[3] = ERROR_IO_PENDING;
                    } else {
                        cpu.gpr[3] = 0;
                    }
                }
            }
            "XamUserCreateAchievementEnumerator" => {
                // (title id, user index, xuid, flags, offset, count, u32 *buffer size out, HANDLE *out)
                // There are no achievements for an offline profile: a valid but empty enumerator.
                let (count, size_ptr, out) = (cpu.gpr[8] as u32, cpu.gpr[9] as u32, cpu.gpr[10] as u32);
                const ACHIEVEMENT_SIZE: u32 = 0x40; // XUSER_ACHIEVEMENT record (size reconstructed)
                if size_ptr != 0 {
                    mem.write32(size_ptr, ACHIEVEMENT_SIZE * count.max(1));
                }
                let h = self.objects.create(crate::objects::KObject::Enumerator { item_size: ACHIEVEMENT_SIZE, items: Default::default() });
                mem.write32(out, h);
                cpu.gpr[3] = 0;
            }
            "XamContentCreateEnumerator" => {
                // (user index, device id, content type, flags, items per enumerate, u32 *buffer size out,
                //  HANDLE *out). Lists saved content on the device; no saves exist yet, so it is empty.
                // Each record is an XCONTENT_DATA (0x134 bytes).
                let (per, size_ptr, out) = (cpu.gpr[7] as u32, cpu.gpr[8] as u32, cpu.gpr[9] as u32);
                const CONTENT_DATA_SIZE: u32 = 0x134;
                if size_ptr != 0 {
                    mem.write32(size_ptr, CONTENT_DATA_SIZE * per.max(1));
                }
                let h = self.objects.create(crate::objects::KObject::Enumerator { item_size: CONTENT_DATA_SIZE, items: Default::default() });
                mem.write32(out, h);
                cpu.gpr[3] = 0;
            }
            "XamEnumerate" => {
                // (enumerator, flags, buffer, buffer size, u32 *items returned, overlapped*)
                let (h, buf, size, ret, overlapped) = (a, c, cpu.gpr[6] as u32, cpu.gpr[7] as u32, cpu.gpr[8] as u32);
                match self.objects.enumerator_take(h, usize::MAX) {
                    None => cpu.gpr[3] = ERROR_INVALID_HANDLE,
                    Some((item_size, items)) => {
                        let fit = if item_size == 0 { 0 } else { (size / item_size) as usize };
                        let n = items.len().min(fit);
                        for (i, rec) in items.iter().take(n).enumerate() {
                            for (k, byte) in rec.iter().enumerate() {
                                mem.write8(buf + i as u32 * item_size + k as u32, *byte);
                            }
                        }
                        if ret != 0 {
                            mem.write32(ret, n as u32);
                        }
                        let status = if n == 0 { ERROR_NO_MORE_FILES } else { 0 };
                        if overlapped != 0 {
                            mem.write32(overlapped, status as u32);
                            mem.write32(overlapped + 4, n as u32);
                            if let Some(addr) = self.objects.dispatcher_addr(mem.read32(overlapped + 0x10)) {
                                crate::dispatcher::set_event(mem, addr);
                            }
                            cpu.gpr[3] = ERROR_IO_PENDING;
                        } else {
                            cpu.gpr[3] = status;
                        }
                    }
                }
            }
            // ---- notifications ----
            "XNotifyGetNext" => {
                // (listener, match id, DWORD *id out, DWORD *param out) -> BOOL
                match self.objects.listener_pop(a, b) {
                    Some((id, param)) => {
                        mem.write32(c, id);
                        if cpu.gpr[6] != 0 {
                            mem.write32(cpu.gpr[6] as u32, param);
                        }
                        cpu.gpr[3] = 1;
                    }
                    None => cpu.gpr[3] = 0,
                }
            }
            "XNotifyBroadcast" => {
                // (id, param)
                self.objects.broadcast(a, b);
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
    fn script_parsing() {
        let s = parse_script("10:+A, 20:-A,30:tap=START,40:lx=-100,50:signout,60:guide").unwrap();
        assert_eq!(s[0], (10, PadAction::Press(0x1000)));
        assert_eq!(s[1], (20, PadAction::Release(0x1000)));
        assert_eq!(s[2], (30, PadAction::Press(0x10)));
        assert_eq!(s[3], (33, PadAction::Release(0x10)));
        assert_eq!(s[4], (40, PadAction::Axis(Axis::Lx, -100)));
        assert!(parse_script("5:bogus").is_err());
        assert!(parse_script("5:+Z").is_err());
        assert!(parse_script("nope").is_err());
    }

    #[test]
    fn buttons_generate_keystrokes_and_packets() {
        let mut x = XamState::with_script(parse_script("5:tap=A").unwrap());
        assert!(x.advance(4).is_empty() && x.buttons == 0 && x.packet == 1);
        x.advance(5);
        assert_eq!(x.buttons, 0x1000);
        assert_eq!(x.next_keystroke(), Some(Keystroke { vk: 0x5800, flags: KEYDOWN }));
        x.advance(8);
        assert_eq!(x.buttons, 0);
        assert_eq!(x.next_keystroke(), Some(Keystroke { vk: 0x5800, flags: KEYUP }));
        assert_eq!(x.next_keystroke(), None);
        assert_eq!(x.packet, 3);
    }

    #[test]
    fn signin_and_guide_produce_notifications() {
        let mut x = XamState::with_script(parse_script("1:signout,2:signout,3:signin,4:guide,5:guide").unwrap());
        assert_eq!(x.advance(1), vec![(XN_SYS_SIGNINCHANGED, 0)]);
        assert!(x.advance(2).is_empty(), "no change, no notification");
        assert_eq!(x.advance(3), vec![(XN_SYS_SIGNINCHANGED, 1)]);
        assert_eq!(x.advance(5), vec![(XN_SYS_UI, 1), (XN_SYS_UI, 0)]);
    }
}
