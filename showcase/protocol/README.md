# Ordering and determinism without sockets

This exhibit keeps the protocol contract and drops the network transport.
Nodes are real worker threads behind the same frame codec; the controller
sends byte frames through channels, exactly as it would send them over TCP.

```text
controller (smart, owns workload + placement)
        |
        |  u32BE || JSON header || payload
        v
node 0..N (model-blind: listing + named blobs + typed streams)
```

A node never sees "Futhark", "matmul", or a model name. It receives:

- a listing to assemble,
- named blobs on a shelf,
- an `EXEC` binding input stream names to blobs or inline byte segments,
- output stream names to return.

## Run

```sh
cargo run --release --bin jotnar-demo-protocol
```

Default: 4 nodes, 12 independent jobs, 3 batches.

The workload is the small `matmul2` listing. Each job is a different 2x2
matrix pair. The controller computes a local reference, then runs the same
jobs in:

- original order,
- reverse order,
- deterministic shuffled orders,

with different node placements and simulated jitter. The completion order
changes; the per-job output bytes do not.

Reference output:

```text
original: completion=[0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] exact=12/12
reverse:  completion=[11, 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8] exact=12/12
shuffle:  completion=[4, 0, 3, 7, 5, 6, 11, 10, 1, 9, 8, 2] exact=12/12
determinism: 3 batches, 12 jobs, 4 node assignments, byte-identical
RESULT: ALL OK
```

## Why no sockets (here)

The interesting contract is not TCP; it is:

> same program + same input bytes = same output bytes, independent of
> placement, submission order, and completion order.

Channels preserve the frame bytes and the model-blind boundary. The TCP
transport now exists as `jotnar-node` wrapping the same `Node::handle`,
`EXEC`, blob shelf, and determinism gate — see [`../runtime/`](../runtime/).
This exhibit keeps the channel form because it isolates the ordering
claim from network jitter.

The nodes are intentionally one-request-at-a-time. Fan-out lives in the
controller, matching the original one-client-per-server policy.
