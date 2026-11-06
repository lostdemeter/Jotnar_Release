//! Playable DOOM on the Rust RV32IM core, using SDL2 through its C ABI.
//!
//! No SDL headers or Rust SDL crate are needed: the binary dynamically loads
//! `libSDL2-2.0.so.0`, like the original ctypes player.

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use jotnar::rv32::Rv32;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_LAZY: c_int = 1;
const SDL_INIT_VIDEO: u32 = 0x20;
const WINDOW_SHOWN: u32 = 0x4;
const RENDERER_ACCELERATED: u32 = 0x2;
const PIXFMT_ARGB8888: u32 = 0x1636_2004;
const TEXTURE_STREAMING: c_int = 1;
const SDL_QUIT: u32 = 0x100;
const SDL_KEYDOWN: u32 = 0x300;
const SDL_KEYUP: u32 = 0x301;

type SdlInit = unsafe extern "C" fn(u32) -> c_int;
type CreateWindow =
    unsafe extern "C" fn(*const c_char, c_int, c_int, c_int, c_int, u32) -> *mut c_void;
type CreateRenderer = unsafe extern "C" fn(*mut c_void, c_int, u32) -> *mut c_void;
type CreateTexture = unsafe extern "C" fn(*mut c_void, u32, c_int, c_int, c_int) -> *mut c_void;
type UpdateTexture =
    unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, c_int) -> c_int;
type RenderCopy =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, *const c_void) -> c_int;
type RenderPresent = unsafe extern "C" fn(*mut c_void);
type PollEvent = unsafe extern "C" fn(*mut c_void) -> c_int;
type DestroyWindow = unsafe extern "C" fn(*mut c_void);
type RaiseWindow = unsafe extern "C" fn(*mut c_void);
type DestroyRenderer = unsafe extern "C" fn(*mut c_void);
type DestroyTexture = unsafe extern "C" fn(*mut c_void);
type Quit = unsafe extern "C" fn();

struct Sdl {
    init: SdlInit,
    create_window: CreateWindow,
    create_renderer: CreateRenderer,
    create_texture: CreateTexture,
    update_texture: UpdateTexture,
    render_copy: RenderCopy,
    render_present: RenderPresent,
    poll_event: PollEvent,
    destroy_window: DestroyWindow,
    raise_window: RaiseWindow,
    destroy_renderer: DestroyRenderer,
    destroy_texture: DestroyTexture,
    quit: Quit,
}

unsafe fn sym<T: Copy>(handle: *mut c_void, name: &str) -> T {
    let c = CString::new(name).unwrap();
    let p = dlsym(handle, c.as_ptr());
    if p.is_null() {
        panic!("SDL2 symbol missing: {name}");
    }
    std::mem::transmute_copy(&p)
}

impl Sdl {
    unsafe fn load() -> Result<Self, String> {
        let name = CString::new("libSDL2-2.0.so.0").unwrap();
        let h = dlopen(name.as_ptr(), RTLD_LAZY);
        if h.is_null() {
            return Err("could not load libSDL2-2.0.so.0".into());
        }
        Ok(Sdl {
            init: sym(h, "SDL_Init"),
            create_window: sym(h, "SDL_CreateWindow"),
            create_renderer: sym(h, "SDL_CreateRenderer"),
            create_texture: sym(h, "SDL_CreateTexture"),
            update_texture: sym(h, "SDL_UpdateTexture"),
            render_copy: sym(h, "SDL_RenderCopy"),
            render_present: sym(h, "SDL_RenderPresent"),
            poll_event: sym(h, "SDL_PollEvent"),
            destroy_window: sym(h, "SDL_DestroyWindow"),
            raise_window: sym(h, "SDL_RaiseWindow"),
            destroy_renderer: sym(h, "SDL_DestroyRenderer"),
            destroy_texture: sym(h, "SDL_DestroyTexture"),
            quit: sym(h, "SDL_Quit"),
        })
    }
}

fn map_key(sym: i32) -> Option<u8> {
    Some(match sym {
        0x4000_0052 => 0xAD, // up
        0x4000_0051 => 0xAF, // down
        0x4000_0050 => 0xAC, // left
        0x4000_004F => 0xAE, // right
        0x1B => 27,
        0x0D => 13,
        0x20 => 0xA2,
        0x09 => 9,
        0x7F => 0x7F,
        0x4000_00E0 | 0x4000_00E4 => 0xA3, // ctrl -> fire
        0x4000_00E1 | 0x4000_00E5 => 0xB6, // shift -> run
        c if (32..127).contains(&c) => (c as u8),
        _ => return None,
    })
}

fn usage() -> &'static str {
    "jotnar-doom-player — playable DOOM on the Rust RV32IM core\n\
     \n\
     usage: jotnar-doom-player --image IMAGE --wad WAD\n\
     \n\
     Arrows move, ctrl fires, space uses, shift runs, esc menu, q quits.\n"
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut image = None;
    let mut wad = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--image" => image = Some(PathBuf::from(it.next().ok_or("--image needs a path")?)),
            "--wad" => wad = Some(PathBuf::from(it.next().ok_or("--wad needs a path")?)),
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}\n\n{}", usage())),
        }
    }
    let image = std::fs::read(image.ok_or("need --image")?).map_err(|e| e.to_string())?;
    let wad = std::fs::read(wad.ok_or("need --wad")?).map_err(|e| e.to_string())?;
    let mut cpu = Rv32::new(64 << 20);
    cpu.load_doom(&image, &wad).map_err(|e| e.to_string())?;

    let sdl = unsafe { Sdl::load()? };
    unsafe {
        if (sdl.init)(SDL_INIT_VIDEO) != 0 {
            return Err("SDL_Init failed (is DISPLAY set?)".into());
        }
        let title = CString::new("Jotnar RV32IM").unwrap();
        let win = (sdl.create_window)(title.as_ptr(), 80, 80, 960, 600, WINDOW_SHOWN);
        if win.is_null() {
            return Err("could not create SDL window".into());
        }
        let mut ren = (sdl.create_renderer)(win, -1, RENDERER_ACCELERATED);
        if ren.is_null() {
            ren = (sdl.create_renderer)(win, -1, 0);
        }
        if ren.is_null() {
            return Err("could not create SDL renderer".into());
        }
        let tex = (sdl.create_texture)(ren, PIXFMT_ARGB8888, TEXTURE_STREAMING, 320, 200);
        if tex.is_null() {
            return Err("could not create SDL texture".into());
        }
        (sdl.raise_window)(win);
        let mut ev = [0u8; 56];
        let start = Instant::now();
        let mut next = start;
        let frame_dt = Duration::from_nanos(16_666_667);
        println!(
            "DOOM running. Arrows move, ctrl fires, space uses, shift runs, esc menu, q quits."
        );
        let mut running = true;
        while running {
            let now = Instant::now();
            cpu.set_ms(now.duration_since(start).as_millis() as u32);
            cpu.run_one_frame();
            let fb = cpu.framebuffer();
            (sdl.update_texture)(tex, std::ptr::null(), fb.as_ptr() as *const c_void, 320 * 4);
            (sdl.render_copy)(ren, tex, std::ptr::null(), std::ptr::null());
            (sdl.render_present)(ren);
            while (sdl.poll_event)(ev.as_mut_ptr() as *mut c_void) > 0 {
                let t = u32::from_le_bytes([ev[0], ev[1], ev[2], ev[3]]);
                if t == SDL_QUIT {
                    running = false;
                } else if t == SDL_KEYDOWN || t == SDL_KEYUP {
                    let sym = i32::from_le_bytes([ev[20], ev[21], ev[22], ev[23]]);
                    if let Some(k) = map_key(sym) {
                        cpu.key(t == SDL_KEYDOWN, k);
                    }
                    if t == SDL_KEYDOWN && sym == b'q' as i32 {
                        running = false;
                    }
                }
            }
            next += frame_dt;
            let sleep = next.saturating_duration_since(Instant::now());
            if !sleep.is_zero() {
                std::thread::sleep(sleep);
            } else {
                next = Instant::now();
            }
        }
        (sdl.destroy_texture)(tex);
        (sdl.destroy_renderer)(ren);
        (sdl.destroy_window)(win);
        (sdl.quit)();
    }
    Ok(())
}
