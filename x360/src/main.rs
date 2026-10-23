mod audio;
mod crt;
mod cstdio;
mod disasm;
mod display;
mod dispatcher;
mod exceptions;
mod export_table;
mod fs;
mod gpu;
mod hostfs;
mod hostio;
mod iocalls;
mod iso;
mod kernel;
mod loader;
mod memory;
mod objects;
mod sched;
mod snapshot;
mod vm;
mod window;
mod xam;
mod xenos_draw;
mod xenos_tex;
mod xenos_vm;
mod xex;

use kernel::Next;
use memory::GuestMemory;
use std::fs::File;
use std::io::BufReader;
use xenon_cpu::interp::Trap;
use xenon_cpu::memory::Bus;
use xenon_cpu::Cpu;

/// Default instructions a thread runs before the scheduler considers switching
/// (override with X360_SLICE to test timing-sensitive behaviour).
const DEFAULT_SLICE: u32 = 20_000;

fn load_xex_bytes(path: &str) -> Result<Vec<u8>, String> {
    if path.to_lowercase().ends_with(".xex") {
        return std::fs::read(path).map_err(|e| e.to_string());
    }
    let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut disc = iso::Gdf::open(BufReader::new(f)).map_err(|e| e.to_string())?;
    println!("disc: {} entries", disc.files.len());
    disc.read_file("default.xex").map_err(|e| e.to_string())
}

/// Walk the stack's back-chain: each frame stores the caller's sp at [sp], and the prologue
/// (`stw r12,-8(r1)` before `stwu`) leaves the saved LR at [caller_sp - 8].
fn backtrace(cpu: &Cpu, mem: &GuestMemory) -> String {
    let mut out = format!("  #0 pc={:#010x}\n  #1 lr={:#010x}\n", cpu.pc, cpu.lr);
    let mut sp = cpu.gpr[1] as u32;
    for depth in 2..14 {
        let caller_sp = mem.read32(sp);
        if caller_sp <= sp || caller_sp == 0 || !mem.is_mapped(caller_sp - 8) {
            break;
        }
        out.push_str(&format!("  #{depth} {:#010x}\n", mem.read32(caller_sp - 8)));
        sp = caller_sp;
    }
    out
}

fn crash_report(cpu: &Cpu, mem: &GuestMemory, thread_id: u32, headline: &str) {
    println!("\n==== CRASH (thread {thread_id}): {headline}");
    print!("-- backtrace:\n{}", backtrace(cpu, mem));
    println!("pc={:#010x} lr={:#010x} ctr={:#x} cr={:08x} xer={:#x}", cpu.pc, cpu.lr, cpu.ctr, cpu.cr, cpu.xer);
    for (i, regs) in cpu.gpr.chunks(4).enumerate() {
        let s: Vec<String> = regs.iter().enumerate().map(|(j, v)| format!("r{:<2}={:016x}", i * 4 + j, v)).collect();
        println!("{}", s.join("  "));
    }
    println!("-- last instructions:");
    for pc in cpu.recent_pcs(16) {
        let inst = mem.read32(pc);
        println!("{:08x}: {:08x}  {}", pc, inst, disasm::disasm(pc, inst));
    }
    // X360_CRASH_DUMP="0xADDR,LEN" hex-dumps guest memory at the moment of the crash.
    if let Ok(spec) = std::env::var("X360_CRASH_DUMP") {
        if let Some((a, n)) = spec.split_once(',') {
            let a = u32::from_str_radix(a.trim_start_matches("0x"), 16).unwrap_or(0);
            let n: u32 = n.parse().unwrap_or(64);
            println!("-- memory at {a:#010x}:");
            for row in 0..n.div_ceil(16) {
                let bytes: Vec<u8> = (0..16).map(|i| mem.read8(a + row * 16 + i)).collect();
                let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
                let txt: String = bytes.iter().map(|&b| if (32..127).contains(&b) { b as char } else { '.' }).collect();
                println!("{:08x}  {:<48} {}", a + row * 16, hex.join(" "), txt);
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: x360 <game.iso | default.xex> [max_steps]");
        return;
    };
    let max_steps: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000_000);

    // X360_HEXDUMP="audio.rpf,0,64" prints `len` bytes of a disc file from `offset` and exits.
    if let Ok(spec) = std::env::var("X360_HEXDUMP") {
        let parts: Vec<&str> = spec.split(',').collect();
        if let (Some(name), Some(off), Some(len)) = (parts.first(), parts.get(1), parts.get(2)) {
            let (off, len): (u64, usize) = (off.parse().unwrap_or(0), len.parse().unwrap_or(64));
            let f = File::open(&path).expect("open disc");
            let disc = iso::Gdf::open(BufReader::new(f)).expect("read disc");
            let mut disc = disc;
            match disc.find(name).cloned() {
                Some(e) => {
                    let data = disc.read_range(&e, off, len).expect("read");
                    for (i, row) in data.chunks(16).enumerate() {
                        let hex: Vec<String> = row.iter().map(|b| format!("{b:02x}")).collect();
                        let txt: String = row.iter().map(|&b| if (32..127).contains(&b) { b as char } else { '.' }).collect();
                        println!("{:08x}  {:<48} {}", off as usize + i * 16, hex.join(" "), txt);
                    }
                }
                None => println!("no such file on disc: {name}"),
            }
        }
        return;
    }

    // X360_LIST_DISC=1 prints every file/directory on the disc and exits.
    if std::env::var_os("X360_LIST_DISC").is_some() {
        match File::open(&path).map_err(|e| e.to_string()).and_then(|f| iso::Gdf::open(BufReader::new(f)).map_err(|e| e.to_string())) {
            Ok(disc) => {
                for f in &disc.files {
                    println!("{} {:>12}  {}", if f.is_dir { "d" } else { "-" }, f.size, f.path);
                }
            }
            Err(e) => eprintln!("could not read disc: {e}"),
        }
        return;
    }

    let bytes = load_xex_bytes(&path).unwrap_or_else(|e| {
        eprintln!("load failed: {e}");
        std::process::exit(1);
    });
    let xex = xex::Xex::parse(bytes).unwrap_or_else(|e| {
        eprintln!("XEX parse failed: {e}");
        std::process::exit(1);
    });
    println!(
        "xex: load={:#010x} image_size={:#x} encrypted={} compression={:?}",
        xex.load_address, xex.image_size, xex.encrypted, xex.compression
    );

    let mut mem = GuestMemory::new();
    let module = loader::load(&xex, &mut mem).unwrap_or_else(|e| {
        eprintln!("module load failed: {e}");
        std::process::exit(1);
    });
    println!(
        "module: base={:#010x} size={:#x} entry={:#010x} imports={} variable_imports={}",
        module.base, module.size, module.entry, module.imports.len(), module.variable_imports.len()
    );

    // X360_LIST_IMPORTS=1 prints every kernel import the title uses and exits.
    if std::env::var_os("X360_LIST_IMPORTS").is_some() {
        for i in &module.imports {
            let name = export_table::lookup(&i.library, i.ordinal).map(|e| e.0).unwrap_or("<unknown>");
            println!("{} {:#06x} {name}", i.library, i.ordinal);
        }
        for (lib, ordinal, rec) in &module.variable_imports {
            let name = export_table::lookup(lib, *ordinal).map(|e| e.0).unwrap_or("<unknown>");
            println!("VARIABLE {lib} {ordinal:#06x} {name} record at {rec:#010x}");
        }
        return;
    }

    // TLS info (key 0x20104, 4 dwords stored inline at the header's offset):
    // slot_count, raw_data_address, data_size, raw_data_size.
    let tls = xex.opt.get(&0x0002_0104).map(|&off| {
        let o = off as usize;
        let w = |i: usize| u32::from_be_bytes(xex.file[o + i..o + i + 4].try_into().unwrap());
        kernel::TlsInfo { slot_count: w(0), raw_data_addr: w(4), data_size: w(8), raw_data_size: w(12) }
    });
    println!("tls: {tls:?}");

    // X360_DUMP="0x829a0800,64" prints that many words of guest memory after load, disassembled.
    if let Ok(spec) = std::env::var("X360_DUMP") {
        if let Some((a, n)) = spec.split_once(',') {
            let a = u32::from_str_radix(a.trim_start_matches("0x"), 16).unwrap_or(0);
            for i in 0..n.parse::<u32>().unwrap_or(0) {
                let w = mem.read32(a + i * 4);
                println!("{:08x}: {:08x}  {}", a + i * 4, w, disasm::disasm(a + i * 4, w));
            }
        }
    }

    let started = std::time::Instant::now();
    let mut kernel = kernel::Kernel::new();
    // X360_WINDOW=1: live preview window + keyboard as the controller
    let live = std::env::var_os("X360_WINDOW").map(|_| {
        let shared = std::sync::Arc::new(window::Shared::default());
        window::spawn(shared.clone());
        kernel.gpu.window = Some(shared.clone());
        kernel.xam.live = Some(shared.clone());
        shared
    });
    kernel.libraries = module.libraries.clone();
    kernel.exec_module = module.module_info;
    kernel.system_flags = xex.opt.get(&0x0003_0000).copied().unwrap_or(0);
    kernel.tls_info = tls;
    if !path.to_lowercase().ends_with(".xex") {
        kernel.vfs = fs::Vfs::open_disc(&path).ok();
    }

    let main_idx = kernel
        .spawn_thread(&mut mem, module.entry, 0, 0, module.stack_size, false)
        .unwrap_or_else(|e| {
            eprintln!("could not create main thread: {e}");
            std::process::exit(1);
        });
    let slice_len: u32 = std::env::var("X360_SLICE").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SLICE);
    let mut cur = main_idx;
    let mut cpu = kernel.take_cpu(cur);
    let mut slice = 0u32;
    let mut steps = 0u64;
    let mut switches = 0u64;
    // X360_LOAD_STATE=file: resume a saved machine instead of the fresh boot above (the max-instructions argument
    // then means "this many more"). X360_SAVE_STATE=file,steps: write one when the run reaches `steps`.
    let mut max_steps = max_steps;
    if let Ok(file) = std::env::var("X360_LOAD_STATE") {
        let t0 = std::time::Instant::now();
        match snapshot::load(&file, &mut mem) {
            Ok((k2, saved_steps)) => {
                let vfs = kernel.vfs.take();
                kernel = k2;
                kernel.vfs = vfs;
                kernel.trace_calls = std::env::var_os("X360_TRACE_KERNEL").is_some();
                kernel.syscall_cycles = std::env::var("X360_SYSCALL_CYCLES").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
                kernel.gpu.reload_env();
                if let Some(l) = &live {
                    kernel.gpu.window = Some(l.clone());
                    kernel.xam.live = Some(l.clone());
                }
                cur = kernel.cur;
                cpu = kernel.take_cpu(cur);
                steps = saved_steps;
                max_steps = saved_steps + max_steps;
                println!("loaded state {file} ({:.1}G instructions in) in {:.1}s", saved_steps as f64 / 1e9, t0.elapsed().as_secs_f64());
            }
            Err(e) => {
                eprintln!("could not load state {file}: {e}");
                std::process::exit(1);
            }
        }
    }
    let mut save_request: Option<(String, u64)> = std::env::var("X360_SAVE_STATE").ok().and_then(|v| {
        let (f, n) = v.rsplit_once(',')?;
        Some((f.to_string(), n.trim().parse().ok()?))
    });

    // X360_WATCH="0x827e2078,0x827e2098" prints registers each time any thread reaches one of these PCs.
    let watch: Vec<u64> = std::env::var("X360_WATCH")
        .map(|s| s.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect())
        .unwrap_or_default();
    // JIT: translate hot blocks to native code (X360_NOJIT=1 forces the interpreter, needed for exact
    // fault pcs and for the per-pc watch tools).
    // X360_COUNT_PCS="0xPC,0xPC": count visits to those addresses (no logging, cheap even for hot functions)
    let count_pcs: Vec<u64> = std::env::var("X360_COUNT_PCS")
        .ok()
        .map(|s| s.split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect())
        .unwrap_or_default();
    let mut pc_counts: Vec<u64> = vec![0; count_pcs.len()];
    let mut jit = (std::env::var_os("X360_NOJIT").is_none() && !std::env::var_os("X360_OPSTATS").is_some())
        .then(|| xenon_cpu::jit::Jit::<GuestMemory>::new_mode(module.base, module.base + module.size, true));
    if let Some(j) = jit.as_mut() {
        for &w in &watch {
            j.add_break(w as u32);
        }
        for &w in &count_pcs {
            j.add_break(w as u32);
        }
    }
    let mut at_entry = true;
    let mut next_cp_poll: u64 = 0;
    // X360_TRACE_CALLS="N,M": for the Nth and Mth visits to the (first) X360_WATCH pc, write an instruction
    // trace of the next 30000 instructions to trace_<N>.txt (runs those instructions in the interpreter).
    let trace_calls: Vec<u32> = std::env::var("X360_TRACE_CALLS")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    let mut trace_left: u32 = 0;
    // X360_WATCH_LIVE=1: print a line the moment each X360_WATCH pc is reached (to interleave with other live logs)
    let watch_live = std::env::var_os("X360_WATCH_LIVE").is_some();
    // X360_WATCH_KEEP=<n>: how many X360_WATCH hits to keep (the most recent n), default 60
    let watch_keep: usize = std::env::var("X360_WATCH_KEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    // X360_CP_DELAY=<instructions>: how long freshly written ring commands take to reach the command processor
    let cp_delay: u64 = std::env::var("X360_CP_DELAY").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut cp_due: u64 = 0;
    // X360_TRACE_TID=<id>: which thread X360_TRACE_CALLS records (default: the main thread, 1)
    let trace_tid: u32 = std::env::var("X360_TRACE_TID").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut trace_out: Option<std::io::BufWriter<std::fs::File>> = None;
    let mut watch_hits = 0u32;
    let mut watch_log: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    // X360_WATCH_MEM="0xB7B0B000" reports the guest PC of every 32-bit write to that address.
    mem.watch_addr = std::env::var("X360_WATCH_MEM")
        .ok()
        .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);
    let mut mem_watch_hits = 0u32;
    let mut mem_watch_log: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut exit_code: Option<u32> = None;
    // self-profile: host time spent in kernel calls and the GPU command processor
    let (mut kernel_time, mut gpu_time, mut kernel_calls) = (std::time::Duration::ZERO, std::time::Duration::ZERO, 0u64);
    // X360_TRACE_AFTER=<steps>: switch kernel-call tracing on only once that many instructions have run
    let trace_after: u64 = std::env::var("X360_TRACE_AFTER").ok().and_then(|v| v.parse().ok()).unwrap_or(u64::MAX);
    // X360_OPSTATS=1: histogram of executed instruction kinds, printed at exit (drives JIT priorities)
    let opstats = std::env::var_os("X360_OPSTATS").is_some();
    let mut op_hist: std::collections::HashMap<(u32, u32), u64> = std::collections::HashMap::new();
    // X360_STATUS_EVERY=<instructions>: print a progress line at that interval (is the game moving or looping?)
    let status_every: u64 = std::env::var("X360_STATUS_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut next_status = status_every;
    while steps < max_steps {
        let pc_before = cpu.pc;
        if save_request.as_ref().is_some_and(|(_, at)| steps >= *at) {
            let (file, _) = save_request.take().unwrap();
            let t0 = std::time::Instant::now();
            kernel.put_cpu(cur, cpu);
            match snapshot::save(&file, &kernel, &mem, steps) {
                Ok(()) => println!("saved state {file} at {:.1}G instructions in {:.1}s", steps as f64 / 1e9, t0.elapsed().as_secs_f64()),
                Err(e) => eprintln!("could not save state {file}: {e}"),
            }
            cpu = kernel.take_cpu(cur);
        }
        if status_every != 0 && steps >= next_status {
            next_status = steps - steps % status_every + status_every;
            let live = kernel.threads.iter().filter(|t| !t.terminated).count();
            // CPU share of the busiest threads since the last status line (id[kind]: share of instructions)
            let total: u64 = kernel.threads.iter().map(|t| t.executed).sum::<u64>().max(1);
            let mut top: Vec<(u32, u64, &str)> = kernel
                .threads
                .iter()
                .filter(|t| t.executed > 0)
                .map(|t| (t.id, t.executed, if t.is_interrupt { "irq" } else if t.is_audio { "audio" } else { "" }))
                .collect();
            top.sort_by(|a, b| b.1.cmp(&a.1));
            let _ = total;
            let share: Vec<String> = top.iter().take(6).map(|(id, n, k)| format!("t{id}{k}:{:.1}G", *n as f64 / 1e9)).collect();
            println!("[status] busiest threads (cumulative instrs): {}", share.join(" "));
            println!(
                "[status] {:.1}G instrs | vsync {} frames {} | gpu packets {} draws {} | kernel calls {} | io {:.1} MB | live threads {} | virtual {:.0}s",
                steps as f64 / 1e9, kernel.vsyncs, kernel.gpu.frames, kernel.gpu.packets,
                kernel.gpu.opcode_counts.get(&0x36).copied().unwrap_or(0) + kernel.gpu.opcode_counts.get(&0x22).copied().unwrap_or(0),
                kernel_calls, kernel.io_bytes as f64 / 1e6, live, kernel.global_cycles as f64 / 3.2e9
            );
        }
        if opstats {
            let inst = mem.read32(cpu.pc as u32);
            let opcd = inst >> 26;
            let sub = match opcd {
                31 => (inst >> 1) & 0x3FF,
                19 => (inst >> 1) & 0x3FF,
                4 => inst & 0x7FF,
                58 | 62 => inst & 3,
                59 | 63 => (inst >> 1) & 0x1F,
                30 => (inst >> 2) & 7,
                _ => 0,
            };
            *op_hist.entry((opcd, sub)).or_default() += 1;
        }
        if steps >= trace_after && !kernel.trace_calls {
            kernel.trace_calls = true;
        }
        if trace_left > 0 && kernel.threads[cur].id == trace_tid {
            use std::io::Write;
            if let Some(w) = trace_out.as_mut() {
                let inst = mem.read32(cpu.pc as u32);
                let _ = writeln!(w, "{:08x}: {:08x}  {}   r3={:x} r4={:x} r23={:x} r30={:x} FILE[ptr={:x} cnt={:x}]", cpu.pc, inst, disasm::disasm(cpu.pc as u32, inst), cpu.gpr[3], cpu.gpr[4], cpu.gpr[23], cpu.gpr[30], mem.read32(0x4003f720), mem.read32(0x4003f724));
            }
            trace_left -= 1;
            if trace_left == 0 {
                trace_out = None;
            }
        }
        if !count_pcs.is_empty() {
            if let Some(i) = count_pcs.iter().position(|&p| p == cpu.pc) {
                pc_counts[i] += 1;
            }
        }
        if !watch.is_empty() && watch.contains(&cpu.pc) {
            if let Some(&n) = trace_calls.iter().find(|&&n| n == watch_hits + 1) {
                trace_left = std::env::var("X360_TRACE_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(60000);
                trace_out = std::fs::File::create(format!("trace_{n}.txt")).ok().map(std::io::BufWriter::new);
                at_entry = false;
            }
            if watch_live {
                println!("[mark] {:#x} t{}", cpu.pc, kernel.threads[cur].id);
            }
            // keep the LAST 40 hits (the ones nearest a crash), printed when the run ends
            watch_hits += 1;
            while watch_log.len() >= watch_keep {
                watch_log.pop_front();
            }
            watch_log.push_back(format!(
                "[watch #{watch_hits} t{} {:#010x}] r3={:#x} r4={:#x} r5={:#x} r6={:#x} r9={:#x} r10={:#x} r11={:#x} r25={:#x} r27={:#x} r28={:#x} r29={:#x} r30={:#x} r31={:#x} lr={:#x} sp={:#x}",
                kernel.threads[cur].id, cpu.pc, cpu.gpr[3], cpu.gpr[4], cpu.gpr[5], cpu.gpr[6], cpu.gpr[9], cpu.gpr[10], cpu.gpr[11],
                cpu.gpr[25], cpu.gpr[27], cpu.gpr[28], cpu.gpr[29], cpu.gpr[30], cpu.gpr[31], cpu.lr, cpu.gpr[1]
            ));
            // also what r4 and r6 point at (8 words each), when they look like addresses
            // if r3 points at a struct whose first word points at text (a string "FILE"), show that text
            {
                let a = cpu.gpr[3] as u32;
                if a >= 0x1000 && mem.is_mapped(a) {
                    let p = mem.read32(a);
                    if p >= 0x1000 && mem.is_mapped(p) {
                        let text: String = (0..96u32).map(|i| mem.read8(p + i)).take_while(|b| *b != 0).map(|b| if (32..127).contains(&b) { b as char } else { '.' }).collect();
                        watch_log.push_back(format!("    [text at [r3]={p:#x}] {text:?}"));
                        let raw: Vec<String> = (0..48u32).map(|i| format!("{:02x}", mem.read8(p + i))).collect();
                        let file: Vec<String> = (0..8u32).map(|i| format!("{:08x}", mem.read32(a + i * 4))).collect();
                        watch_log.push_back(format!("    [raw bytes] {}
    [FILE struct @r3] {}", raw.join(" "), file.join(" ")));
                    }
                }
            }
            for (name, r) in [("r4", cpu.gpr[4]), ("r6", cpu.gpr[6])] {
                let a = r as u32;
                if a >= 0x1000 && mem.is_mapped(a) && mem.is_mapped(a + 28) {
                    let w: Vec<String> = (0..8).map(|i| format!("{:08x}", mem.read32(a + i * 4))).collect();
                    watch_log.push_back(format!("    [{name}]={a:#x}: {}", w.join(" ")));
                }
            }
        }
        let r = match jit.as_mut() {
            Some(j) if at_entry && trace_left == 0 => match j.run(&mut cpu, &mut mem) {
                xenon_cpu::jit::Outcome::Ran(n) => {
                    kernel.threads[cur].executed += n;
                    steps += n;
                    slice += n as u32;
                    Ok(())
                }
                xenon_cpu::jit::Outcome::Trapped(n, t) => {
                    kernel.threads[cur].executed += n;
                    steps += n;
                    slice += n as u32;
                    at_entry = true;
                    Err(t)
                }
                xenon_cpu::jit::Outcome::Interpret => {
                    let r = cpu.step(&mut mem);
                    kernel.threads[cur].executed += 1;
                    steps += 1;
                    slice += 1;
                    at_entry = cpu.pc != pc_before.wrapping_add(4);
                    r
                }
            },
            _ => {
                let r = cpu.step(&mut mem);
                steps += 1;
                slice += 1;
                at_entry = cpu.pc != pc_before.wrapping_add(4);
                r
            }
        };
        let tid = kernel.threads[cur].id;
        // One flag test covers faults, memory watches and GPU register writes.
        if mem.attn.swap(false, std::sync::atomic::Ordering::Relaxed) {
            if let Some((a, v)) = mem.watch_hit.take() {
                // keep the LAST hits (nearest a crash); printed when the run ends
                mem_watch_hits += 1;
                while mem_watch_log.len() >= 40 {
                    mem_watch_log.pop_front();
                }
                mem_watch_log.push_back(format!(
                    "[watchmem #{mem_watch_hits} t{tid}] {a:#010x} <- {v:#010x} by pc={pc_before:#010x} (lr={:#010x}) at step {steps} sp={:#x}",
                    cpu.lr, cpu.gpr[1]
                ));
            }
            if mem.watch_read.swap(0, std::sync::atomic::Ordering::Relaxed) != 0 {
                mem_watch_hits += 1;
                while mem_watch_log.len() >= 40 {
                    mem_watch_log.pop_front();
                }
                mem_watch_log.push_back(format!(
                    "[watchREAD #{mem_watch_hits} t{tid}] {:#010x} = {:#010x} read by block at pc={pc_before:#010x} (lr={:#010x}) at step {steps}",
                    mem.watch_addr, mem.read32_unwatched(mem.watch_addr), cpu.lr
                ));
            }
            if mem.gpu_dirty.swap(false, std::sync::atomic::Ordering::Relaxed) {
                if cp_delay == 0 {
                    let t0 = std::time::Instant::now();
                    run_gpu(&mut kernel, &mut mem);
                    gpu_time += t0.elapsed();
                } else if cp_due == 0 {
                    // New work reaches the CP after a short latency, not within the same instruction
                    cp_due = steps + cp_delay;
                }
            }
            if let Some(f) = mem.take_fault() {
                let kind = if f.write { "write to" } else { "read from" };
                crash_report(&cpu, &mem, tid, &format!("{kind} unmapped address {:#010x}", f.addr));
                break;
            }
        }
        // A command processor blocked on WAIT_REG_MEM is released by the CPU writing memory, and the CPU may be
        // spinning waiting for the CP in turn. Looking only at scheduling points made every such hand-off cost a
        // whole time slice; polling every few hundred instructions keeps the pair in step.
        if (kernel.gpu.stalled || cp_due != 0) && steps >= next_cp_poll {
            next_cp_poll = steps + 400;
            if cp_due == 0 || steps >= cp_due {
                cp_due = 0;
                run_gpu(&mut kernel, &mut mem);
            }
        }
        let mut irq_preempt = false;
        if !kernel.gpu.pending_interrupts.is_empty() {
            // A real CP interrupt pre-empts whatever is running at once; waiting for the slice to end
            // made every CP hand-off cost a whole slice of the other threads spinning.
            irq_preempt = std::env::var_os("X360_IRQ_PREEMPT").is_some();
            if let Err(e) = kernel.deliver_gpu_interrupts(&mut mem) {
                crash_report(&cpu, &mem, tid, &e);
                break;
            }
        }
        let mut reschedule = false;
        match r {
            Ok(()) => reschedule = slice >= slice_len && kernel.threads[cur].irql < 2,
            Err(Trap::Syscall) => match {
                let t0 = std::time::Instant::now();
                let r = kernel.dispatch(&mut cpu, &mut mem);
                kernel_time += t0.elapsed();
                kernel_calls += 1;
                r
            } {
                Ok(Next::Continue) => reschedule = slice >= slice_len && kernel.threads[cur].irql < 2,
                Ok(Next::Reschedule) => reschedule = true,
                Ok(Next::Exit(code)) => {
                    // Run the registered title-termination notifications before really exiting.
                    if exit_code.is_none() {
                        exit_code = Some(code);
                        match kernel.begin_shutdown(&mut mem) {
                            Ok(n) if n > 0 => {
                                println!("\ntitle is exiting (code {code:#x}): running {n} terminate notification(s)");
                                reschedule = true;
                            }
                            _ => {
                                println!("\nmain thread returned {code:#x}: title exited cleanly");
                                break;
                            }
                        }
                    } else {
                        reschedule = true;
                    }
                }
                Err(e) => {
                    crash_report(&cpu, &mem, tid, &e);
                    break;
                }
            },
            // `twui 31,r0,20` is the debugger's OutputDebugString: r3 = text, r4 = length.
            Err(Trap::ProgramTrap { pc, inst: 0x0FE0_0014 }) => {
                let (p, n) = (cpu.gpr[3] as u32, (cpu.gpr[4] as u32).min(1024));
                let text: String = (0..n).map(|i| mem.read8(p + i) as char).collect();
                println!("[guest debug t{tid} pc={pc:#x} lr={:#x} r3={:#x} r4={:#x}] {:?}", cpu.lr, cpu.gpr[3], cpu.gpr[4], text);
                cpu.pc = pc + 4;
            }
            Err(Trap::ProgramTrap { pc, inst }) => {
                crash_report(&cpu, &mem, tid, &format!("guest trap (tw/td) fired at {pc:#010x}, inst {inst:08x}"));
                break;
            }
            Err(Trap::Unimplemented { pc, inst }) => {
                crash_report(&cpu, &mem, tid,
                    &format!(
                        "unimplemented instruction {inst:08x} at {pc:#010x} (primary opcode {}, xo(31-form) {})",
                        inst >> 26,
                        (inst >> 1) & 0x3FF
                    ),
                );
                break;
            }
        }
        if exit_code.is_some() && !kernel.shutdown_threads_alive() {
            println!("terminate notifications finished: title exited cleanly (code {:#x})", exit_code.unwrap());
            break;
        }
        if irq_preempt && kernel.threads[cur].irql < 2 {
            reschedule = true;
        }
        if reschedule {
            if live.as_ref().is_some_and(|l| l.closed.load(std::sync::atomic::Ordering::Relaxed)) {
                println!("
window closed: stopping");
                break;
            }
            // A CP blocked on WAIT_REG_MEM can only be released by the CPU writing memory, so
            // re-check it at every scheduling point.
            if kernel.gpu.stalled {
                run_gpu(&mut kernel, &mut mem);
            }
            kernel.put_cpu(cur, cpu);
            match kernel.pick_next(&mut mem) {
                Ok(i) => {
                    if i != cur {
                        switches += 1;
                    }
                    cur = i;
                    cpu = kernel.take_cpu(cur);
                    slice = 0;
                    at_entry = true;
                }
                Err(e) => {
                    println!("\n==== {e}");
                    println!("executed {steps} instructions, {switches} thread switches");
                    return;
                }
            }
        }
    }
    if steps >= max_steps {
        let tid = kernel.threads[cur].id;
        crash_report(&cpu, &mem, tid, "step limit reached (still running - spinning, or just busy?)");
    }
    for l in watch_log.iter().chain(mem_watch_log.iter()) {
        println!("{l}");
    }
    if opstats {
        let total: u64 = op_hist.values().sum();
        let mut v: Vec<_> = op_hist.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        println!("instruction mix ({total} sampled):");
        let mut cum = 0u64;
        for ((op, sub), n) in v.iter().take(60) {
            cum += **n;
            println!("  op {op:>2} sub {sub:>4}: {:>5.2}%  (cum {:>5.1}%)", **n as f64 * 100.0 / total as f64, cum as f64 * 100.0 / total as f64);
        }
    }
    println!("executed {steps} instructions, {switches} thread switches, {} threads", kernel.threads.len());
    let secs = started.elapsed().as_secs_f64();
    println!("wall time {secs:.1}s => {:.1} M instructions/sec", steps as f64 / secs / 1e6);
    println!("profile: {kernel_calls} kernel calls took {:.2}s, GPU command processing {:.2}s", kernel_time.as_secs_f64(), gpu_time.as_secs_f64());
    if let Some(j) = &jit {
        println!("{}", j.summary());
    }
    for (p, n) in count_pcs.iter().zip(&pc_counts) {
        println!("[count] pc {p:#x} visited {n} times");
    }
    println!("scheduler: {} picks, {:.2}s in pick_next", kernel.picks, kernel.pick_time.as_secs_f64());
    println!("vsync interrupts fired: {}, CP interrupts: {}", kernel.vsyncs, kernel.cp_interrupts);
    println!("threads at exit:\n{}", kernel.thread_summary(Some((cur, cpu.pc))));
    // X360_BACKTRACES=1: where every live thread is, as a chain of return addresses.
    if std::env::var_os("X360_BACKTRACES").is_some() {
        for (i, t) in kernel.threads.iter().enumerate() {
            if t.terminated {
                continue;
            }
            let c = if i == cur { &cpu } else { &t.cpu };
            if matches!(t.wait, sched::Wait::Multi { .. }) {
                eprintln!("  thread {} {}", t.id, kernel.describe_wait(&mem, i));
            }
            let mut chain = format!("pc={:#x} lr={:#x}", c.pc, c.lr);
            let mut sp = c.gpr[1] as u32;
            for _ in 0..14 {
                let back = mem.read32(sp);
                if back <= sp || !mem.is_mapped(back) || !mem.is_mapped(back - 8) {
                    break;
                }
                chain.push_str(&format!(" <- {:#x}", mem.read32(back - 8)));
                sp = back;
            }
            println!("  bt thread {:>2}: {chain}", t.id);
        }
    }
    println!("{}", kernel.display.summary());
    println!("{}", kernel.audio.summary());
    kernel.audio.finish();
    println!("{}", kernel.gpu.summary());
    let _ = mem.read8(0); // keep Bus import used
}

/// Run the command processor, unless it is held back until the interrupt handler its last INTERRUPT packet raised
/// has finished (the handler reads and acknowledges state the CP would otherwise run past).
fn run_gpu(kernel: &mut kernel::Kernel, mem: &mut memory::GuestMemory) {
    if kernel.gpu.hold {
        if kernel.threads.iter().any(|t| t.is_interrupt && !t.terminated) {
            kernel.gpu.stalled = true; // keep polling
            return;
        }
        kernel.gpu.hold = false;
    }
    kernel.gpu.process(mem);
}
