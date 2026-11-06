use std::path::Path;

use jotnar::rv32::Rv32;

#[test]
fn adds_and_halts() {
    let mut cpu = Rv32::new(1 << 16);
    let prog: [u32; 4] = [
        0x0030_0093, // addi x1, x0, 3
        0x0040_0113, // addi x2, x0, 4
        0x0020_81B3, // add x3, x1, x2
        0x0010_0073, // ebreak
    ];
    for (i, w) in prog.iter().enumerate() {
        cpu.write32(i as u32 * 4, *w);
    }
    cpu.run_to_frame(1, 100);
    assert!(cpu.halted);
    assert_eq!(cpu.x[3], 7);
    assert_eq!(cpu.pc, 16);
}

#[test]
fn checkpoint_roundtrip() {
    let mut cpu = Rv32::new(1 << 16);
    cpu.write32(0, 0x0050_0093); // addi x1, x0, 5
    cpu.step();
    let dir = std::env::temp_dir().join(format!("jotnar_rv32_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("cp");
    cpu.save_checkpoint(&prefix).unwrap();
    let mut other = Rv32::new(1 << 16);
    other.load_checkpoint(&prefix).unwrap();
    assert_eq!(other.x[1], 5);
    assert_eq!(other.pc, cpu.pc);
    assert_eq!(other.cycles, cpu.cycles);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn framebuffer_png_roundtrip() {
    let mut cpu = Rv32::new(64 << 20);
    // Write two RGBA words into the framebuffer.
    cpu.write32(jotnar::rv32::FB_ADDR, 0x00112233);
    cpu.write32(jotnar::rv32::FB_ADDR + 4, 0x44556677);
    let p = Path::new("/tmp/opencode/rv32_test_fb.png");
    jotnar::rv32::save_fb_png(cpu.framebuffer(), p).unwrap();
    let got = jotnar::rv32::read_fb_png(p).unwrap();
    for i in 0..2 {
        assert_eq!(&got[i * 4..i * 4 + 3], &cpu.framebuffer()[i * 4..i * 4 + 3]);
    }
}

#[test]
fn interactive_mmio_helpers() {
    let mut cpu = Rv32::new(64 << 20);
    cpu.set_ms(1234);
    assert_eq!(cpu.read32(jotnar::rv32::CLOCK_ADDR), 1234);
    cpu.key(true, 0x1B);
    let tail = cpu.read32(0x03FF_0100 + 4);
    let ev = cpu.read32(0x03FF_0100 + 8);
    assert_eq!(tail, 1);
    assert_eq!(ev, (1 << 8) | 0x1B);
}
