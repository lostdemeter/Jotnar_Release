use std::collections::HashMap;

use serde_json::Value as Json;

use jotnar::verilog::{emit, Memory, Sim};

#[test]
fn cpu8_verilog_matches_reference_trace() {
    let text = include_str!("fixtures/cpu8.asm");
    let fixture: Json = serde_json::from_str(include_str!("fixtures/cpu8_trace.json")).unwrap();
    let words: Vec<u64> = fixture["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let mem = Memory {
        name: "RAM".into(),
        depth: 256,
        width: 16,
        raddr: "PC".into(),
        rout: "INSTR".into(),
        we: "WE".into(),
        waddr: "ARG".into(),
        wdata: "WDATA".into(),
        init: words,
        raddr2: Some("ARG".into()),
        rout2: Some("DATA".into()),
    };
    let mut state = HashMap::new();
    state.insert("PC".to_string(), (16usize, 0u64, "PC_NEXT".to_string()));
    state.insert("ACC".to_string(), (16, 0, "ACC_NEXT".to_string()));
    state.insert("Z".to_string(), (16, 0, "Z_NEXT".to_string()));
    state.insert("HALT".to_string(), (16, 0, "HALT_NEXT".to_string()));

    let widths = HashMap::new();
    let v = emit(text, &widths, "cpu8", &state, &[mem.clone()]).unwrap();
    assert!(v.contains("module cpu8"));
    assert!(v.contains("always @(posedge clk)"));

    let sim = Sim::compile(text, &widths, &state, &[mem]).unwrap();
    let got = sim.simulate_cycles(&HashMap::new(), 90, 1, true).unwrap();
    let want = fixture["trace"].as_array().unwrap();
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for k in ["PC", "ACC", "Z", "HALT"] {
            assert_eq!(g[k] as u64, w[k].as_u64().unwrap(), "cycle {i} {k}");
        }
        let mem = w["mem"].as_array().unwrap();
        for (addr, val) in mem.iter().enumerate() {
            assert_eq!(
                g[&format!("mem:RAM:{addr}")] as u64,
                val.as_u64().unwrap(),
                "cycle {i} mem[{addr}]"
            );
        }
    }
    assert_eq!(got.last().unwrap()["ACC"], 55);
    assert_eq!(got.last().unwrap()["HALT"], 1);
}

#[test]
fn combinational_comparisons_work() {
    let text = "IN X\nIN Y\nIN A\nIN B\nEQ = IEQ(X, Y)\nLT = ILT(X, Y)\nS = ISELECT(LT, A, B)\n";
    let mut widths = HashMap::new();
    for n in ["X", "Y", "A", "B", "EQ", "LT", "S"] {
        widths.insert(n.to_string(), (1usize, 8usize));
    }
    let v = emit(text, &widths, "cmp", &HashMap::new(), &[]).unwrap();
    assert!(v.contains("module cmp"));
    let sim = Sim::compile(text, &widths, &HashMap::new(), &[]).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("X".to_string(), 3);
    inputs.insert("Y".to_string(), 4);
    inputs.insert("A".to_string(), 0xAA);
    inputs.insert("B".to_string(), 0x55);
    let got = sim.simulate(&inputs).unwrap();
    assert_eq!(got["EQ"], 0);
    assert_eq!(got["LT"], 1);
    assert_eq!(got["S"], 0xAA);
}
