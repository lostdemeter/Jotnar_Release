//! SDL2 viewer for the two-node DOOM coordinator.
//!
//! The coordinator runs the RV32IM core and the GPU stage. This process only
//! displays the stitched RGB frames and forwards keypresses back over TCP. It
//! loads SDL2 through its C ABI, so no SDL headers or Rust SDL crate are
//! needed.

use std::ffi::{c_char, c_int, c_void, CString};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use serde_json::{json, Value as Json};

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_LAZY: c_int = 1;
const SDL_INIT_VIDEO: u32 = 0x20;
const WINDOW_SHOWN: u32 = 0x4;
const RENDERER_ACCELERATED: u32 = 0x2;
const PIXFMT_RGB24: u32 = 0x1710_1803;
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
type SetWindowTitle = unsafe extern "C" fn(*mut c_void, *const c_char);
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
    set_window_title: SetWindowTitle,
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
            set_window_title: sym(h, "SDL_SetWindowTitle"),
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
    "jotnar-doom-view — display two-node DOOM and forward keys\n\
     \n\
     usage: jotnar-doom-view [--addr ADDR] [--title TEXT]\n\
     \n\
     The coordinator must be started with --viewer-listen ADDR.\n\
     Arrows move, ctrl fires, space uses, shift runs, esc menu, q quits.\n"
}

fn connect(addr: &str) -> Result<TcpStream, String> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match TcpStream::connect(addr) {
            Ok(s) => {
                let _ = s.set_nodelay(true);
                return Ok(s);
            }
            Err(e) => {
                if Instant::now() >= deadline {
                    return Err(format!("connect {addr}: {e}"));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

fn send_json(stream: &mut TcpStream, value: &Json) -> Result<(), String> {
    let body = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    stream
        .write_all(&(body.len() as u32).to_be_bytes())
        .map_err(|e| e.to_string())?;
    stream.write_all(&body).map_err(|e| e.to_string())
}

fn read_frame(stream: &mut TcpStream) -> Result<(Json, Vec<u8>), String> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).map_err(|e| e.to_string())?;
    let n = u32::from_be_bytes(len) as usize;
    if n == 0 || n > 64 * 1024 {
        return Err(format!("bad frame header length {n}"));
    }
    let mut hbuf = vec![0u8; n];
    stream.read_exact(&mut hbuf).map_err(|e| e.to_string())?;
    let header: Json = serde_json::from_slice(&hbuf).map_err(|e| e.to_string())?;
    let width = header.get("width").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let height = header.get("height").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let bytes = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or("bad frame size")?;
    let mut body = vec![0u8; bytes];
    stream.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok((header, body))
}

unsafe fn present(sdl: &Sdl, tex: *mut c_void, ren: *mut c_void, body: &[u8], width: usize) {
    (sdl.update_texture)(
        tex,
        std::ptr::null(),
        body.as_ptr() as *const c_void,
        (width * 3) as c_int,
    );
    (sdl.render_copy)(ren, tex, std::ptr::null(), std::ptr::null());
    (sdl.render_present)(ren);
}

fn run() -> Result<(), String> {
    let mut addr = "127.0.0.1:10000".to_string();
    let mut title = "Jotnar DOOM - two nodes".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--addr" => addr = it.next().ok_or("--addr needs an address")?,
            "--title" => title = it.next().ok_or("--title needs text")?,
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}\n\n{}", usage())),
        }
    }
    let mut stream = connect(&addr)?;
    let (mut header, mut body) = read_frame(&mut stream)?;
    let width = header.get("width").and_then(|x| x.as_u64()).unwrap_or(640) as usize;
    let height = header.get("height").and_then(|x| x.as_u64()).unwrap_or(400) as usize;
    let win_w = 960usize;
    let win_h = (win_w * height / width.max(1)).max(1);
    let sdl = unsafe { Sdl::load()? };
    unsafe {
        if (sdl.init)(SDL_INIT_VIDEO) != 0 {
            return Err("SDL_Init failed (is DISPLAY set?)".into());
        }
        let ctitle = CString::new(title.clone()).unwrap();
        let win = (sdl.create_window)(
            ctitle.as_ptr(),
            80,
            80,
            win_w as c_int,
            win_h as c_int,
            WINDOW_SHOWN,
        );
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
        let tex = (sdl.create_texture)(
            ren,
            PIXFMT_RGB24,
            TEXTURE_STREAMING,
            width as c_int,
            height as c_int,
        );
        if tex.is_null() {
            return Err("could not create SDL texture".into());
        }
        let mut tex = tex;
        let mut tex_w = width;
        let mut tex_h = height;
        (sdl.raise_window)(win);
        present(&sdl, tex, ren, &body, width);
        let mut ev = [0u8; 56];
        let mut running = true;
        while running {
            if let Some(mode) = header.get("mode").and_then(|x| x.as_str()) {
                let frame = header.get("frame").and_then(|x| x.as_u64()).unwrap_or(0);
                let ms = header.get("total_ms").and_then(|x| x.as_u64()).unwrap_or(0);
                let text = format!("{title} | {mode} | frame {frame} | {ms} ms");
                if let Ok(c) = CString::new(text) {
                    (sdl.set_window_title)(win, c.as_ptr());
                }
            }
            while (sdl.poll_event)(ev.as_mut_ptr() as *mut c_void) > 0 {
                let t = u32::from_le_bytes([ev[0], ev[1], ev[2], ev[3]]);
                if t == SDL_QUIT {
                    let _ = send_json(&mut stream, &json!({"type": "quit"}));
                    running = false;
                } else if t == SDL_KEYDOWN || t == SDL_KEYUP {
                    let sym = i32::from_le_bytes([ev[20], ev[21], ev[22], ev[23]]);
                    if let Some(k) = map_key(sym) {
                        let _ = send_json(
                            &mut stream,
                            &json!({"type": "key", "pressed": t == SDL_KEYDOWN, "key": k}),
                        );
                    }
                    if t == SDL_KEYDOWN && sym == b'q' as i32 {
                        let _ = send_json(&mut stream, &json!({"type": "quit"}));
                        running = false;
                    }
                }
            }
            if !running {
                break;
            }
            match read_frame(&mut stream) {
                Ok((h, b)) => {
                    header = h;
                    body = b;
                    let nw = header
                        .get("width")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(tex_w as u64) as usize;
                    let nh = header
                        .get("height")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(tex_h as u64) as usize;
                    if nw != tex_w || nh != tex_h {
                        (sdl.destroy_texture)(tex);
                        tex = (sdl.create_texture)(
                            ren,
                            PIXFMT_RGB24,
                            TEXTURE_STREAMING,
                            nw as c_int,
                            nh as c_int,
                        );
                        if tex.is_null() {
                            return Err("could not resize SDL texture".into());
                        }
                        tex_w = nw;
                        tex_h = nh;
                    }
                }
                Err(e) => {
                    eprintln!("viewer: {e}");
                    break;
                }
            }
            present(&sdl, tex, ren, &body, tex_w);
        }
        (sdl.destroy_texture)(tex);
        (sdl.destroy_renderer)(ren);
        (sdl.destroy_window)(win);
        (sdl.quit)();
    }
    Ok(())
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
