//! Pure helpers for the two-node DOOM frame pipeline.
//!
//! The coordinator splits a 320x200 RV32IM framebuffer into two horizontal
//! windows, each node runs the same Jotnar upscale program, and the coordinator
//! stitches the results into one 640x400 frame. The windows overlap by one
//! input column so bilinear upscaling is continuous at the split.
//!
//! The split is not fixed: the coordinator chooses it from measured per-node
//! throughput. Keeping the geometry here makes the network path testable
//! without a GPU.

use sha2::{Digest, Sha256};

pub const FB_W: usize = 320;
pub const FB_H: usize = 200;
pub const FB_BYTES: usize = FB_W * FB_H * 4;
pub const HALF_W: usize = FB_W / 2;
pub const OUT_W: usize = FB_W * 2;
pub const OUT_H: usize = FB_H * 2;
pub const CHANNELS: usize = 3;

/// Even split, with the one-column overlap the GPU windows need.
pub const SEND_W: usize = HALF_W + 1;
pub const SEND_OUT_W: usize = SEND_W * 2;
pub const HALF_OUT_W: usize = HALF_W * 2;

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// A horizontal split of the 320-column framebuffer.
///
/// `left_cols + right_cols == FB_W`. Each side sends `cols + 1` input columns
/// (one overlap column) and returns `(cols + 1) * 2` output columns. The
/// coordinator keeps `cols * 2` output columns per side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    pub left_cols: usize,
    pub right_cols: usize,
}

impl Split {
    pub const EVEN: Self = Self {
        left_cols: HALF_W,
        right_cols: HALF_W,
    };

    pub fn new(left_cols: usize, right_cols: usize) -> Result<Self, String> {
        if left_cols == 0 || right_cols == 0 {
            return Err("split needs at least one column per side".to_string());
        }
        if left_cols + right_cols != FB_W {
            return Err(format!("split {} + {} != {FB_W}", left_cols, right_cols));
        }
        Ok(Self {
            left_cols,
            right_cols,
        })
    }

    pub fn from_left(left_cols: usize) -> Result<Self, String> {
        if left_cols == 0 || left_cols >= FB_W {
            return Err(format!("left split {left_cols} is outside 1..{FB_W}"));
        }
        Self::new(left_cols, FB_W - left_cols)
    }

    /// `(x0, width)` for the left and right input windows.
    pub fn windows(&self) -> ((usize, usize), (usize, usize)) {
        let left = (0, self.left_cols + 1);
        let right = (FB_W - self.right_cols - 1, self.right_cols + 1);
        (left, right)
    }

    pub fn left_out_width(&self) -> usize {
        (self.left_cols + 1) * 2
    }

    pub fn right_out_width(&self) -> usize {
        (self.right_cols + 1) * 2
    }

    pub fn left_share(&self) -> f64 {
        self.left_cols as f64 / FB_W as f64
    }

    /// Join the two overlapping F32-decoded RGB outputs into one RGB frame.
    pub fn stitch(&self, left: &[u8], right: &[u8]) -> Result<Vec<u8>, String> {
        let left_bytes = self.left_out_width() * OUT_H * CHANNELS;
        let right_bytes = self.right_out_width() * OUT_H * CHANNELS;
        if left.len() != left_bytes || right.len() != right_bytes {
            return Err(format!(
                "split {}:{} needs {left_bytes}+{right_bytes} bytes, got {}+{}",
                self.left_cols,
                self.right_cols,
                left.len(),
                right.len()
            ));
        }
        let mut out = vec![0u8; OUT_W * OUT_H * CHANNELS];
        let left_row = self.left_out_width() * CHANNELS;
        let right_row = self.right_out_width() * CHANNELS;
        let out_row = OUT_W * CHANNELS;
        let keep_left = self.left_cols * 2 * CHANNELS;
        let keep_right = self.right_cols * 2 * CHANNELS;
        for y in 0..OUT_H {
            let lsrc = y * left_row;
            let rsrc = y * right_row + 2 * CHANNELS;
            let dst = y * out_row;
            out[dst..dst + keep_left].copy_from_slice(&left[lsrc..lsrc + keep_left]);
            out[dst + keep_left..dst + out_row].copy_from_slice(&right[rsrc..rsrc + keep_right]);
        }
        Ok(out)
    }
}

/// Pick a split from two measured probe times. A node that takes half as long
/// for the same probe gets twice the columns, clamped to `min_cols..=max_cols`.
pub fn choose_split(left_ms: u64, right_ms: u64, min_cols: usize, max_cols: usize) -> Split {
    let min_cols = min_cols.clamp(1, FB_W / 2);
    let max_cols = max_cols.clamp(FB_W / 2, FB_W - min_cols);
    if left_ms == 0 || right_ms == 0 || left_ms == right_ms {
        return Split::EVEN;
    }
    // throughput is 1/time, so the left share is right_time / total_time.
    let share = right_ms as f64 / (left_ms + right_ms) as f64;
    let left = (FB_W as f64 * share).round() as usize;
    let left = left.clamp(min_cols, max_cols);
    Split::from_left(left).unwrap_or(Split::EVEN)
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

/// Convert one logical half of the even split.
pub fn rgb_half(fb: &[u8], half: usize) -> Result<Vec<f32>, String> {
    if half > 1 {
        return Err(format!("half must be 0 or 1, got {half}"));
    }
    let (left, right) = Split::EVEN.windows();
    let (x0, w) = if half == 0 { left } else { right };
    rgb_window(fb, x0, w)
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

/// Backward-compatible even-split stitch.
pub fn stitch_overlap(left: &[u8], right: &[u8]) -> Result<Vec<u8>, String> {
    Split::EVEN.stitch(left, right)
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

    fn upscale(half: &[f32], width: usize) -> Vec<u8> {
        let out_w = width * 2;
        let mut out = Vec::with_capacity(out_w * OUT_H * CHANNELS);
        for y in 0..FB_H {
            let mut row = Vec::with_capacity(out_w * CHANNELS);
            for x in 0..width {
                let i = (y * width + x) * CHANNELS;
                let px = [half[i] as u8, half[i + 1] as u8, half[i + 2] as u8];
                row.extend_from_slice(&px);
                row.extend_from_slice(&px);
            }
            out.extend_from_slice(&row);
            out.extend_from_slice(&row);
        }
        out
    }

    #[test]
    fn even_split_is_rgb_and_stitches_back() {
        let fb = framebuffer();
        let ((lx, lw), (rx, rw)) = Split::EVEN.windows();
        assert_eq!((lx, lw), (0, 161));
        assert_eq!((rx, rw), (159, 161));
        let left = rgb_window(&fb, lx, lw).unwrap();
        let right = rgb_window(&fb, rx, rw).unwrap();
        assert_eq!(left.len(), lw * FB_H * CHANNELS);
        assert_eq!(&left[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(
            &left[(lw * (FB_H - 1) + (lw - 1)) * CHANNELS..][..3],
            &[103.0, 199.0, 160.0]
        );
        assert_eq!(&right[0..3], &[159.0, 0.0, 159.0]);

        let stitched = Split::EVEN
            .stitch(&upscale(&left, lw), &upscale(&right, rw))
            .unwrap();
        assert_eq!(stitched.len(), OUT_W * OUT_H * CHANNELS);
        assert_eq!(&stitched[0..3], &[0, 0, 0]);
        assert_eq!(&stitched[HALF_OUT_W * CHANNELS..][..3], &[160, 0, 160]);
        assert_eq!(&stitched[(OUT_W - 1) * CHANNELS..][..3], &[63, 0, 63]);
        let last = ((OUT_H - 1) * OUT_W + OUT_W - 1) * CHANNELS;
        assert_eq!(&stitched[last..last + 3], &[6, 199, 63]);
    }

    #[test]
    fn weighted_split_moves_the_boundary() {
        let split = Split::new(80, 240).unwrap();
        let ((lx, lw), (rx, rw)) = split.windows();
        assert_eq!((lx, lw), (0, 81));
        assert_eq!((rx, rw), (79, 241));
        assert_eq!(split.left_out_width(), 162);
        assert_eq!(split.right_out_width(), 482);
        let fb = framebuffer();
        let left = rgb_window(&fb, lx, lw).unwrap();
        let right = rgb_window(&fb, rx, rw).unwrap();
        let stitched = split
            .stitch(&upscale(&left, lw), &upscale(&right, rw))
            .unwrap();
        assert_eq!(stitched.len(), OUT_W * OUT_H * CHANNELS);
        // Output column 160 is the first column owned by the right node.
        assert_eq!(&stitched[160 * CHANNELS..][..3], &[80, 0, 80]);
    }

    #[test]
    fn split_rejects_bad_geometry() {
        assert!(Split::new(0, 320).is_err());
        assert!(Split::new(160, 161).is_err());
        assert!(Split::from_left(320).is_err());
    }

    #[test]
    fn split_follows_measured_throughput() {
        // CUDA 70 ms, HIP 20 ms for the same probe. HIP is 3.5x faster,
        // so it should get roughly 3.5x the columns.
        let s = choose_split(70, 20, 32, 288);
        assert_eq!(s.left_cols + s.right_cols, FB_W);
        assert!((55..=75).contains(&s.left_cols), "left {}", s.left_cols);
        let even = choose_split(20, 20, 32, 288);
        assert_eq!(even, Split::EVEN);
        let clamped = choose_split(10_000, 1, 32, 288);
        assert_eq!(clamped.left_cols, 32);
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
