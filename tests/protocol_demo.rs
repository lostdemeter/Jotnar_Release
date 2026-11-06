use jotnar::showcase::protocol_demo::{run, ProtocolOptions};

#[test]
fn ordering_demo_is_byte_exact() {
    let opts = ProtocolOptions {
        nodes: 2,
        jobs: 5,
        permutations: 2,
        latency: false,
    };
    run(&opts).unwrap();
}
