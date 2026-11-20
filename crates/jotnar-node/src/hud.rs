//! Tiny bitmap HUD for the live DOOM viewer.
//!
//! The viewer gets a 640x400 game frame plus a top and bottom status band.
//! Text uses a dependency-free 3x5 font scaled 2x, so the live overlay does
//! not need SDL_ttf or a font file. The raw frames written for the GIF stay
//! clean; this module only builds the viewer frame.

pub const HUD_TOP: usize = 44;
pub const HUD_BOTTOM: usize = 22;

#[derive(Clone, Debug)]
pub struct HudInfo {
    pub frame: u32,
    pub mode: String,
    pub cuda_addr: String,
    pub hip_addr: String,
    pub cuda_cols: usize,
    pub hip_cols: usize,
    pub cuda_ms: u64,
    pub hip_ms: u64,
    pub total_ms: u64,
    pub budget_ms: u64,
    pub p50: u64,
    pub p95: u64,
    pub cuda_ok: bool,
    pub hip_ok: bool,
    pub note: String,
}

/// One tile line in the wall HUD.
#[derive(Clone, Debug)]
pub struct WallTileInfo {
    pub row: usize,
    pub col: usize,
    pub target: String,
    pub cols: usize,
    pub ms: u64,
    pub attempts: u32,
    pub failover_from: Option<String>,
}

/// Wall HUD: per-tile placement, latency, and the running frame percentiles.
#[derive(Clone, Debug)]
pub struct WallHudInfo {
    pub frame: u64,
    pub fps: u32,
    pub work_ms: u64,
    pub budget_ms: u64,
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub tiles: Vec<WallTileInfo>,
    pub note: String,
}

/// Render a 3x5 glyph. Rows are top-to-bottom, bits are left-to-right.
fn glyph(c: char) -> [u8; 5] {
    let c = c.to_ascii_uppercase();
    match c {
        ' ' => [0, 0, 0, 0, 0],
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'O' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b110, 0b011],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b001, 0b001, 0b001],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '%' => [0b101, 0b001, 0b010, 0b100, 0b101],
        '(' => [0b001, 0b010, 0b010, 0b010, 0b001],
        ')' => [0b100, 0b010, 0b010, 0b010, 0b100],
        '[' => [0b011, 0b010, 0b010, 0b010, 0b011],
        ']' => [0b110, 0b010, 0b010, 0b010, 0b110],
        '|' => [0b010, 0b010, 0b010, 0b010, 0b010],
        '+' => [0b000, 0b010, 0b111, 0b010, 0b000],
        ',' => [0b000, 0b000, 0b000, 0b010, 0b100],
        '_' => [0b000, 0b000, 0b000, 0b000, 0b111],
        '=' => [0b000, 0b111, 0b000, 0b111, 0b000],
        '#' => [0b101, 0b111, 0b101, 0b111, 0b101],
        '?' => [0b111, 0b001, 0b011, 0b000, 0b010],
        '!' => [0b010, 0b010, 0b010, 0b000, 0b010],
        '*' => [0b101, 0b010, 0b111, 0b010, 0b101],
        '>' => [0b100, 0b010, 0b001, 0b010, 0b100],
        '<' => [0b001, 0b010, 0b100, 0b010, 0b001],
        '\'' => [0b010, 0b010, 0b000, 0b000, 0b000],
        _ => [0b111, 0b001, 0b011, 0b000, 0b010],
    }
}

fn fill_rect(buf: &mut [u8], w: usize, x: usize, y: usize, rw: usize, rh: usize, c: [u8; 3]) {
    for yy in y..y + rh {
        for xx in x..x + rw {
            if xx < w {
                let i = (yy * w + xx) * 3;
                if i + 2 < buf.len() {
                    buf[i..i + 3].copy_from_slice(&c);
                }
            }
        }
    }
}

fn draw_text(
    buf: &mut [u8],
    w: usize,
    x: usize,
    y: usize,
    text: &str,
    color: [u8; 3],
    scale: usize,
) -> usize {
    let mut cx = x;
    for ch in text.chars() {
        let rows = glyph(ch);
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..3 {
                if bits & (1 << (2 - col)) != 0 {
                    fill_rect(
                        buf,
                        w,
                        cx + col * scale,
                        y + row * scale,
                        scale,
                        scale,
                        color,
                    );
                }
            }
        }
        cx += 4 * scale;
    }
    cx
}

pub fn render(
    frame: &[u8],
    width: usize,
    height: usize,
    info: &HudInfo,
) -> Result<Vec<u8>, String> {
    if frame.len() != width * height * 3 {
        return Err(format!(
            "HUD frame has {} bytes, expected {}",
            frame.len(),
            width * height * 3
        ));
    }
    let out_h = height + HUD_TOP + HUD_BOTTOM;
    let mut out = vec![10u8; width * out_h * 3];
    for y in 0..height {
        let src = y * width * 3;
        let dst = (y + HUD_TOP) * width * 3;
        out[dst..dst + width * 3].copy_from_slice(&frame[src..src + width * 3]);
    }
    let white = [232, 232, 232];
    let dim = [150, 150, 150];
    let cyan = [0, 220, 255];
    let yellow = [255, 210, 90];
    let red = [255, 80, 80];
    let green = [120, 255, 120];

    let line1 = format!(
        "JOTNAR DOOM  MODE {}  FRAME {}  TOTAL {}MS/{}MS",
        info.mode.to_ascii_uppercase(),
        info.frame,
        info.total_ms,
        info.budget_ms
    );
    draw_text(&mut out, width, 6, 2, &line1, white, 2);

    let cuda = if !info.cuda_ok {
        "B CUDA DOWN".to_string()
    } else if info.cuda_cols == 0 {
        format!("B CUDA FULL {}MS", info.cuda_ms)
    } else {
        format!("B CUDA {}C {}MS", info.cuda_cols, info.cuda_ms)
    };
    let hip = if !info.hip_ok {
        "C HIP DOWN".to_string()
    } else if info.hip_cols == 0 {
        format!("C HIP FULL {}MS", info.hip_ms)
    } else {
        format!("C HIP {}C {}MS", info.hip_cols, info.hip_ms)
    };
    let mut x = draw_text(&mut out, width, 6, 16, "A RV32IM", white, 2);
    x = draw_text(
        &mut out,
        width,
        x + 8,
        16,
        &cuda,
        if info.cuda_ok { cyan } else { red },
        2,
    );
    let _ = draw_text(
        &mut out,
        width,
        x + 8,
        16,
        &hip,
        if info.hip_ok { yellow } else { red },
        2,
    );
    let addr_line = format!("B {}   C {}", info.cuda_addr, info.hip_addr);
    draw_text(&mut out, width, 6, 31, &addr_line, dim, 1);

    let note_color = if info.cuda_ok && info.hip_ok {
        green
    } else {
        red
    };
    let bottom = format!(
        "P50 {}MS  P95 {}MS  |  {}",
        info.p50,
        info.p95,
        info.note.to_ascii_uppercase()
    );
    draw_text(
        &mut out,
        width,
        6,
        HUD_TOP + height + 5,
        &bottom,
        note_color,
        2,
    );
    Ok(out)
}

/// Render the wall HUD: a top band with the clock/percentiles and one line per
/// tile (placement, owned columns, latency, failover origin), and a bottom band
/// with the target set and health note. The wall pixels are copied through.
pub fn render_wall(
    frame: &[u8],
    width: usize,
    height: usize,
    info: &WallHudInfo,
) -> Result<Vec<u8>, String> {
    if frame.len() != width * height * 3 {
        return Err(format!(
            "wall HUD frame has {} bytes, expected {}",
            frame.len(),
            width * height * 3
        ));
    }
    let out_h = height + HUD_TOP + HUD_BOTTOM;
    let mut out = vec![10u8; width * out_h * 3];
    for y in 0..height {
        let src = y * width * 3;
        let dst = (y + HUD_TOP) * width * 3;
        out[dst..dst + width * 3].copy_from_slice(&frame[src..src + width * 3]);
    }
    let white = [232, 232, 232];
    let cyan = [0, 220, 255];
    let yellow = [255, 210, 90];
    let green = [120, 255, 120];
    let red = [255, 80, 80];

    let line1 = format!(
        "JOTNAR WALL  {} FPS  FRAME {}  WORK {}MS/{}MS  P50 {}MS  P95 {}MS  P99 {}MS",
        info.fps, info.frame, info.work_ms, info.budget_ms, info.p50, info.p95, info.p99
    );
    draw_text(&mut out, width, 6, 2, &line1, white, 2);

    let mut x = 6;
    for tile in &info.tiles {
        let label = match &tile.failover_from {
            Some(from) => format!(
                "R{}C{} {}<-{} {}C {}MS",
                tile.row, tile.col, tile.target, from, tile.cols, tile.ms
            ),
            None => format!(
                "R{}C{} {} {}C {}MS",
                tile.row, tile.col, tile.target, tile.cols, tile.ms
            ),
        };
        let color = if tile.failover_from.is_some() {
            yellow
        } else {
            cyan
        };
        x = draw_text(&mut out, width, x, 16, &label, color, 2) + 8;
    }
    let _ = x;

    let mut targets: Vec<String> = Vec::new();
    for tile in &info.tiles {
        if !targets.iter().any(|target| target == &tile.target) {
            targets.push(tile.target.clone());
        }
    }
    let note_lower = info.note.to_ascii_lowercase();
    let note_color = if note_lower.contains("failover") || note_lower.contains("recovered") {
        yellow
    } else if note_lower.contains("failed") || note_lower.contains("down") {
        red
    } else {
        green
    };
    let bottom = format!(
        "TARGETS {}  |  {}",
        targets.join("+"),
        info.note.to_ascii_uppercase()
    );
    draw_text(
        &mut out,
        width,
        6,
        HUD_TOP + height + 5,
        &bottom,
        note_color,
        2,
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_adds_bands_and_draws_text() {
        let w = 64;
        let h = 32;
        let frame = vec![42u8; w * h * 3];
        let info = HudInfo {
            frame: 7,
            mode: "split".to_string(),
            cuda_addr: "1.2.3.4:9".to_string(),
            hip_addr: "5.6.7.8:9".to_string(),
            cuda_cols: 100,
            hip_cols: 220,
            cuda_ms: 40,
            hip_ms: 20,
            total_ms: 45,
            budget_ms: 66,
            p50: 44,
            p95: 50,
            cuda_ok: true,
            hip_ok: true,
            note: "all nodes healthy".to_string(),
        };
        let out = render(&frame, w, h, &info).unwrap();
        assert_eq!(out.len(), w * (h + HUD_TOP + HUD_BOTTOM) * 3);
        // Game area is copied through.
        let game = (HUD_TOP * w + 10) * 3;
        assert_eq!(&out[game..game + 3], &[42, 42, 42]);
        // Some HUD pixels are not the background.
        assert!(out[..HUD_TOP * w * 3].iter().any(|b| *b > 20));
    }

    #[test]
    fn wall_hud_draws_tiles_and_keeps_the_wall() {
        let w = 64;
        let h = 32;
        let frame = vec![9u8; w * h * 3];
        let info = WallHudInfo {
            frame: 12,
            fps: 15,
            work_ms: 27,
            budget_ms: 66,
            p50: 25,
            p95: 28,
            p99: 29,
            tiles: vec![
                WallTileInfo {
                    row: 0,
                    col: 0,
                    target: "cuda".to_string(),
                    cols: 320,
                    ms: 39,
                    attempts: 1,
                    failover_from: None,
                },
                WallTileInfo {
                    row: 0,
                    col: 1,
                    target: "cuda".to_string(),
                    cols: 320,
                    ms: 18,
                    attempts: 2,
                    failover_from: Some("hip".to_string()),
                },
            ],
            note: "failover hip".to_string(),
        };
        let out = render_wall(&frame, w, h, &info).unwrap();
        assert_eq!(out.len(), w * (h + HUD_TOP + HUD_BOTTOM) * 3);
        let game = HUD_TOP * w * 3;
        assert_eq!(&out[game..game + w * h * 3], &frame[..]);
        assert!(out[..game].iter().any(|b| *b > 20));
        assert!(out[(HUD_TOP + h) * w * 3..].iter().any(|b| *b > 20));
    }
}
