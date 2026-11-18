//! Pure helpers for the two-node DOOM frame pipeline.
//!
//! The coordinator splits a 320x200 RV32IM framebuffer into two 160x200 RGB
//! halves, each node runs the same Jotnar upscale program, and the coordinator
//! stitches the two 320x400 results into one 640x400 frame. Keeping the pixel
//! transforms here makes the network path testable without a GPU.

use sha2::{Digest, Sha256};

pub const FB_W: usize = 320;
pub const FB_H: usize = 200;
pub const FB_BYTES: usize = FB_W * FB_H * 4;
pub const HALF_W: usize = FB_W / 2;
pub const OUT_W: usize = FB_W * 2;
pub const OUT_H: usize = FB_H * 2;
/// The two nodes overlap by one input column so bilinear upscaling is
/// continuous at the split. A 161-column window becomes 322 output columns;
/// the coordinator keeps the 320 columns that belong to each side.
pub const SEND_W: usize = HALF_W + 1;
pub const SEND_OUT_W: usize = SEND_W * 2;
pub const HALF_OUT_W: usize = HALF_W * 2;
pub const CHANNELS: usize = 3;

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Convert a horizontal window of the native BGRA framebuffer to packed RGB
/// F32 values.
pub fn rgb_window(fb: &[u8], x0: usize, width: usize) -> Result<Vec<f32>, String> {
    if fb.len() != FB_BYTES {
        return Err(format!(
            "framebuffer has {} bytes, expected {FB_BYTES}",
            fb.len()
        ));
    }
    if width == 0 || x0.checked_add(width).map(|e| e > FB_W).unwrap_or(true) {
        return Err(format!("window {x0}+{width} is outside {FB_W}"));
    }
    let mut out = Vec::with_capacity(width * FB_H * CHANNELS);
    for y in 0..FB_H {
        for x in x0..x0 + width {
            let i = (y * FB_W + x) * 4;
            // The RV32 framebuffer is little-endian BGRA.
            out.push(fb[i + 2] as f32);
            out.push(fb[i + 1] as f32);
            out.push(fb[i] as f32);
        }
    }
    Ok(out)
}

/// Convert one logical half (with one overlap column) of the framebuffer.
pub fn rgb_half(fb: &[u8], half: usize) -> Result<Vec<f32>, String> {
    if half > 1 {
        return Err(format!("half must be 0 or 1, got {half}"));
    }
    let x0 = if half == 0 { 0 } else { FB_W - SEND_W };
    rgb_window(fb, x0, SEND_W)
}

/// Decode a Jotnar F32 RGB output into tightly packed RGB8.
pub fn f32_rgb_to_u8(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    let values = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(CHANNELS))
        .ok_or("image shape overflows")?;
    if bytes.len() != values * 4 {
        return Err(format!(
            "F32 RGB body has {} bytes, expected {}",
            bytes.len(),
            values * 4
        ));
    }
    let mut out = Vec::with_capacity(values);
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(v.round().clamp(0.0, 255.0) as u8);
    }
    Ok(out)
}

/// Join two overlapping 322x400 RGB halves into one 640x400 RGB frame.
///
/// The left window is columns 0..=160 and contributes output 0..319. The right
/// window is columns 159..=319 and contributes its output 2..321 to output
/// 320..639. The one-column overlap makes the bilinear result continuous.
pub fn stitch_overlap(left: &[u8], right: &[u8]) -> Result<Vec<u8>, String> {
    let send_bytes = SEND_OUT_W * OUT_H * CHANNELS;
    if left.len() != send_bytes || right.len() != send_bytes {
        return Err(format!(
            "halves need {send_bytes} bytes each, got {} and {}",
            left.len(),
            right.len()
        ));
    }
    let mut out = vec![0u8; OUT_W * OUT_H * CHANNELS];
    let row_bytes = SEND_OUT_W * CHANNELS;
    let out_row = OUT_W * CHANNELS;
    let half_row = HALF_OUT_W * CHANNELS;
    for y in 0..OUT_H {
        let src = y * row_bytes;
        let dst = y * out_row;
        out[dst..dst + half_row].copy_from_slice(&left[src..src + half_row]);
        let rsrc = src + 2 * CHANNELS;
        out[dst + half_row..dst + out_row].copy_from_slice(&right[rsrc..rsrc + half_row]);
    }
    Ok(out)
}

/// Median, p95, and max of a latency sample set.
pub fn latency_summary(samples: &[u64]) -> (u64, u64, u64) {
    if samples.is_empty() {
        return (0, 0, 0);
    }
    let mut s = samples.to_vec();
    s.sort_unstable();
    let p50 = s[s.len() / 2];
    let p95 = s[(s.len() * 95 / 100).min(s.len() - 1)];
    let max = *s.last().unwrap();
    (p50, p95, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn framebuffer() -> Vec<u8> {
        let mut fb = vec![0u8; FB_BYTES];
        for y in 0..FB_H {
            for x in 0..FB_W {
                let i = (y * FB_W + x) * 4;
                fb[i] = (x % 256) as u8; // B
                fb[i + 1] = (y % 256) as u8; // G
                fb[i + 2] = ((x + y) % 256) as u8; // R
                fb[i + 3] = 255;
            }
        }
        fb
    }

    #[test]
    fn halves_are_rgb_and_stitch_back() {
        let fb = framebuffer();
        let left = rgb_half(&fb, 0).unwrap();
        let right = rgb_half(&fb, 1).unwrap();
        assert_eq!(left.len(), SEND_W * FB_H * CHANNELS);
        // First left pixel: B=0,G=0,R=0.
        assert_eq!(&left[0..3], &[0.0, 0.0, 0.0]);
        // Last left-window pixel is x=160,y=199.
        assert_eq!(
            &left[(SEND_W * (FB_H - 1) + (SEND_W - 1)) * CHANNELS..][..3],
            &[103.0, 199.0, 160.0]
        );
        // First right-window pixel is global x=159.
        assert_eq!(&right[0..3], &[159.0, 0.0, 159.0]);

        // Round-trip through the F32 upscale shape by duplicating each pixel
        // 2x2; this is the shape the GPU returns.
        let up = |half: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(SEND_OUT_W * OUT_H * CHANNELS);
            for y in 0..FB_H {
                let mut row = Vec::with_capacity(SEND_OUT_W * CHANNELS);
                for x in 0..SEND_W {
                    let i = (y * SEND_W + x) * CHANNELS;
                    let px = [half[i] as u8, half[i + 1] as u8, half[i + 2] as u8];
                    row.extend_from_slice(&px);
                    row.extend_from_slice(&px);
                }
                out.extend_from_slice(&row);
                out.extend_from_slice(&row);
            }
            out
        };
        let l8 = up(&left);
        let r8 = up(&right);
        let stitched = stitch_overlap(&l8, &r8).unwrap();
        assert_eq!(stitched.len(), OUT_W * OUT_H * CHANNELS);
        assert_eq!(&stitched[0..3], &[0, 0, 0]);
        assert_eq!(&stitched[HALF_OUT_W * CHANNELS..][..3], &[160, 0, 160]);
        assert_eq!(&stitched[(OUT_W - 1) * CHANNELS..][..3], &[63, 0, 63]);
        let last = ((OUT_H - 1) * OUT_W + OUT_W - 1) * CHANNELS;
        assert_eq!(&stitched[last..last + 3], &[6, 199, 63]);
    }

    #[test]
    fn f32_output_is_clamped_and_rounded() {
        let vals = [-1.0f32, 0.4, 12.6, 300.0];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = f32_rgb_to_u8(&bytes, 2, 1).unwrap_err();
        assert!(out.contains("expected 24"));
        let mut bytes = Vec::new();
        for _ in 0..6 {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for (i, v) in vals.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let out = f32_rgb_to_u8(&bytes, 2, 1).unwrap();
        assert_eq!(out, vec![0, 0, 13, 255, 0, 0]);
    }

    #[test]
    fn latency_summary_is_ordered() {
        assert_eq!(latency_summary(&[]), (0, 0, 0));
        assert_eq!(latency_summary(&[5, 1, 3, 9, 7]), (5, 9, 9));
    }
}
