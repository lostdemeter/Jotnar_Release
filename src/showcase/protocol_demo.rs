//! Ordering/determinism exhibit: model-blind nodes, in-process transport.
//!
//! Same listing, same input bytes, different submission orders and node
//! assignments. The nodes only execute Jotnar; the controller owns the
//! workload and placement.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::json;

use crate::error::{Error, Result};
use crate::payload::Feeds;
use crate::protocol::{decode_frame, encode_stream, exec_outputs, request, NodeHandle};
use crate::value::Value;

const LISTING: &str = include_str!("../../showcase/language/matmul2.asm");
const OUTS: [&str; 4] = ["c00", "c01", "c10", "c11"];

#[derive(Clone, Debug)]
pub struct ProtocolOptions {
    pub nodes: usize,
    pub jobs: usize,
    pub permutations: usize,
    pub latency: bool,
}

impl Default for ProtocolOptions {
    fn default() -> Self {
        ProtocolOptions {
            nodes: 4,
            jobs: 12,
            permutations: 3,
            latency: true,
        }
    }
}

#[derive(Clone, Debug)]
struct Job {
    id: usize,
    vals: [i64; 8],
    expected: BTreeMap<String, Vec<u8>>,
}

fn scalar(v: i64) -> Value {
    Value::I64(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), v))
}

fn local_outputs(vals: &[i64; 8]) -> Result<BTreeMap<String, Vec<u8>>> {
    let names = ["a00", "a01", "a10", "a11", "b00", "b01", "b10", "b11"];
    let mut feeds = Feeds::new();
    for (n, v) in names.iter().zip(vals) {
        feeds.insert((*n).to_string(), scalar(*v));
    }
    let got = crate::runtime::run_text(LISTING, &crate::asm::AsmOptions::new("."), &feeds)?;
    let mut out = BTreeMap::new();
    for o in OUTS {
        let (_, bytes) = encode_stream(got.get(o).unwrap())?;
        out.insert(o.to_string(), bytes);
    }
    Ok(out)
}

fn make_jobs(n: usize) -> Result<Vec<Job>> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i as i64;
        let vals = [
            base,
            base + 1,
            base + 2,
            base + 3,
            base + 4,
            base + 5,
            base + 6,
            base + 7,
        ];
        out.push(Job {
            id: i,
            vals,
            expected: local_outputs(&vals)?,
        });
    }
    Ok(out)
}

fn exec_frame(id: u64, job: &Job) -> Result<Vec<u8>> {
    let names = ["a00", "a01", "a10", "a11", "b00", "b01", "b10", "b11"];
    let mut payload = Vec::with_capacity(8 * 8);
    let mut inputs = serde_json::Map::new();
    for (i, n) in names.iter().enumerate() {
        let off = payload.len();
        payload.extend_from_slice(&job.vals[i].to_le_bytes());
        inputs.insert(
            (*n).to_string(),
            json!({"kind": "I", "dtype": "i64", "shape": [1],
                   "off": off, "len": 8}),
        );
    }
    request(
        id,
        "exec",
        json!({"handle": "matmul2", "inputs": inputs, "outputs": OUTS}),
        &payload,
    )
}

#[derive(Clone, Debug)]
struct BatchResult {
    tag: String,
    order: Vec<usize>,
    outputs: BTreeMap<usize, BTreeMap<String, Vec<u8>>>,
    placement: Vec<(usize, usize)>,
    elapsed_ms: u128,
}

fn run_batch(
    tag: &str,
    nodes: &[NodeHandle],
    jobs: &[Job],
    order: &[usize],
    node_for: impl Fn(usize) -> usize,
    latency: bool,
) -> Result<BatchResult> {
    let (tx, rx) = mpsc::channel::<(usize, usize, BTreeMap<String, Vec<u8>>)>();
    let mut hs = Vec::new();
    let t0 = Instant::now();
    for (slot, &job_id) in order.iter().enumerate() {
        let node_i = node_for(job_id) % nodes.len();
        let node = nodes[node_i].clone();
        let job = jobs[job_id].clone();
        let tx = tx.clone();
        hs.push(thread::spawn(move || {
            if latency {
                // Simulated network/jitter, not a claim about real latency.
                let ms = ((job.id * 7 + slot * 3) % 11) as u64;
                thread::sleep(Duration::from_millis(ms));
            }
            let frame = exec_frame(1000 + job.id as u64, &job).unwrap();
            let resp = node.request(frame).unwrap();
            let out = exec_outputs(&resp).unwrap();
            tx.send((job.id, node_i, out)).unwrap();
        }));
    }
    drop(tx);
    let mut outputs = BTreeMap::new();
    let mut completion = Vec::new();
    let mut placement = Vec::new();
    for (job_id, node_i, out) in rx {
        completion.push(job_id);
        placement.push((job_id, node_i));
        outputs.insert(job_id, out);
    }
    for h in hs {
        h.join()
            .map_err(|_| Error::runtime("protocol demo worker panicked"))?;
    }
    let elapsed_ms = t0.elapsed().as_millis();
    Ok(BatchResult {
        tag: tag.to_string(),
        order: completion,
        outputs,
        placement,
        elapsed_ms,
    })
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut s = seed;
    for i in (1..n).rev() {
        let j = (lcg(&mut s) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

fn check(
    tag: &str,
    b: &BatchResult,
    jobs: &[Job],
    reference: &BTreeMap<usize, BTreeMap<String, Vec<u8>>>,
) -> Result<()> {
    if b.outputs.len() != jobs.len() {
        return Err(Error::runtime(format!("{tag}: missing jobs")));
    }
    for j in jobs {
        let got = b
            .outputs
            .get(&j.id)
            .ok_or_else(|| Error::runtime(format!("{tag}: missing {}", j.id)))?;
        for o in OUTS {
            if got.get(o) != j.expected.get(o) {
                return Err(Error::runtime(format!(
                    "{tag}: job {} output {o} differs from local reference",
                    j.id
                )));
            }
        }
    }
    if &b.outputs != reference {
        return Err(Error::runtime(format!("{tag}: batch bytes differ")));
    }
    Ok(())
}

pub fn run(opts: &ProtocolOptions) -> Result<()> {
    if opts.nodes == 0 || opts.jobs == 0 {
        return Err(Error::runtime("protocol demo needs nodes > 0 and jobs > 0"));
    }
    let jobs = make_jobs(opts.jobs)?;
    let reference: BTreeMap<usize, BTreeMap<String, Vec<u8>>> =
        jobs.iter().map(|j| (j.id, j.expected.clone())).collect();

    let nodes: Vec<NodeHandle> = (0..opts.nodes)
        .map(|i| NodeHandle::spawn(i, &["lattice"]))
        .collect();
    for (i, n) in nodes.iter().enumerate() {
        let ping = request(1, "ping", json!({}), &[])?;
        let (h, _) = decode_frame(&n.request(ping)?)?;
        if h["ok"] != true {
            return Err(Error::runtime(format!("node {i} ping failed: {h}")));
        }
        let prog = request(
            2,
            "program",
            json!({"name": "matmul2", "backend": "lattice"}),
            LISTING.as_bytes(),
        )?;
        let (h, _) = decode_frame(&n.request(prog)?)?;
        if h["ok"] != true {
            return Err(Error::runtime(format!("node {i} program failed: {h}")));
        }
    }
    println!(
        "nodes: {} model-blind lattice workers; program: matmul2 ({} ops)",
        opts.nodes,
        LISTING.lines().filter(|l| l.contains('=')).count()
    );

    let mut orders = vec![
        (
            "original".to_string(),
            (0..opts.jobs).collect::<Vec<_>>(),
            0usize,
        ),
        (
            "reverse".to_string(),
            (0..opts.jobs).rev().collect::<Vec<_>>(),
            1usize,
        ),
    ];
    for p in 0..opts.permutations.saturating_sub(2) {
        orders.push((
            format!("shuffle-{p}"),
            shuffled(opts.jobs, 0x5eed + p as u64),
            p + 2,
        ));
    }

    let mut batches = Vec::new();
    for (tag, order, seed) in orders {
        let b = run_batch(
            &tag,
            &nodes,
            &jobs,
            &order,
            |job_id| {
                if seed == 1 {
                    // reverse node placement as well
                    (opts.nodes - 1) - (job_id % opts.nodes)
                } else if seed >= 2 {
                    ((job_id * 5 + seed) % opts.nodes) as usize
                } else {
                    job_id % opts.nodes
                }
            },
            opts.latency,
        )?;
        check(&tag, &b, &jobs, &reference)?;
        println!(
            "{tag}: completion={:?} ms={} exact={}/{}",
            b.order,
            b.elapsed_ms,
            jobs.len(),
            jobs.len()
        );
        batches.push(b);
    }

    let base = &batches[0].outputs;
    for b in &batches[1..] {
        if &b.outputs != base {
            return Err(Error::runtime(format!(
                "{}: outputs are not identical to {}",
                b.tag, batches[0].tag
            )));
        }
    }
    println!(
        "determinism: {} batches, {} jobs, {} node assignments, byte-identical",
        batches.len(),
        opts.jobs,
        opts.nodes
    );
    println!("placement original: {:?}", batches[0].placement);
    println!("RESULT: ALL OK");
    Ok(())
}
