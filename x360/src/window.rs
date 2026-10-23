//! Live preview window (X360_WINDOW=1): shows each presented frame and turns the keyboard into the
//! controller. It runs on its own thread, so it never slows the emulation; frames are handed over
//! through `Shared`, and key state comes back through the same struct.
//!
//! Keys: arrows = D-pad, Z/X/C/V = A/B/X/Y, Enter = Start, Backspace = Back, Q/E = LB/RB,
//!       W/A/S/D = left stick, I/J/K/L = right stick, 1/2 = left/right trigger, Tab = Guide (no effect yet).

use minifb::{Key, Window, WindowOptions};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// 0x00RRGGBB pixels
    pub pixels: Vec<u32>,
}

#[derive(Default)]
pub struct Shared {
    pub frame: Mutex<Option<Frame>>,
    pub frame_counter: AtomicU32,
    /// XINPUT_GAMEPAD button bits currently held on the keyboard
    pub buttons: AtomicU32,
    pub thumbs: [AtomicI32; 4],
    pub triggers: [AtomicU32; 2],
    /// Set when the window is closed, so the run can stop.
    pub closed: AtomicBool,
}

impl Shared {
    pub fn publish(&self, frame: Frame) {
        *self.frame.lock().unwrap() = Some(frame);
        self.frame_counter.fetch_add(1, Relaxed);
    }
}

/// XInput button bits for the held keys, plus axes in [-32767, 32767] and triggers 0..255.
fn read_keys(w: &Window) -> (u32, [i32; 4], [u32; 2]) {
    let map: &[(Key, u32)] = &[
        (Key::Up, 0x0001),
        (Key::Down, 0x0002),
        (Key::Left, 0x0004),
        (Key::Right, 0x0008),
        (Key::Enter, 0x0010),
        (Key::Backspace, 0x0020),
        (Key::Q, 0x0100),
        (Key::E, 0x0200),
        (Key::Z, 0x1000),
        (Key::X, 0x2000),
        (Key::C, 0x4000),
        (Key::V, 0x8000),
    ];
    let mut buttons = 0;
    for (k, bit) in map {
        if w.is_key_down(*k) {
            buttons |= bit;
        }
    }
    let axis = |neg: Key, pos: Key| -> i32 { (w.is_key_down(pos) as i32 - w.is_key_down(neg) as i32) * 32767 };
    let trig = |k: Key| if w.is_key_down(k) { 255 } else { 0 };
    (
        buttons,
        [axis(Key::A, Key::D), axis(Key::S, Key::W), axis(Key::J, Key::L), axis(Key::K, Key::I)],
        [trig(Key::Key1), trig(Key::Key2)],
    )
}

pub fn spawn(shared: Arc<Shared>) {
    std::thread::spawn(move || {
        let opts = WindowOptions { resize: true, scale_mode: minifb::ScaleMode::AspectRatioStretch, ..WindowOptions::default() };
        let mut window = match Window::new("x360 - GTA IV", 1280, 720, opts) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("window: could not open ({e}); continuing without it");
                return;
            }
        };
        window.set_target_fps(60);
        let mut shown = 0u32;
        let mut current: Option<Frame> = None;
        while window.is_open() && !window.is_key_down(Key::Escape) {
            let (buttons, thumbs, triggers) = read_keys(&window);
            shared.buttons.store(buttons, Relaxed);
            for (i, t) in thumbs.iter().enumerate() {
                shared.thumbs[i].store(*t, Relaxed);
            }
            for (i, t) in triggers.iter().enumerate() {
                shared.triggers[i].store(*t, Relaxed);
            }
            let n = shared.frame_counter.load(Relaxed);
            if n != shown {
                shown = n;
                current = shared.frame.lock().unwrap().take().or(current);
            }
            match &current {
                Some(f) => {
                    let _ = window.update_with_buffer(&f.pixels, f.width, f.height);
                }
                None => window.update(),
            }
        }
        shared.closed.store(true, Relaxed);
    });
}
