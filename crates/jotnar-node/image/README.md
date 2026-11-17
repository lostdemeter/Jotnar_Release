# Build

`build-node-image.sh` produces the node image. It is intentionally a single
script with pinned inputs so the build can be repeated without a toolchain
project.

## Host requirements

- Linux x86-64 with Rust (stable) and the `x86_64-unknown-linux-musl` target
  (`rustup target add x86_64-unknown-linux-musl`).
- `git`, `curl`, `sha256sum`, `cpio`, `gzip`.
- GRUB EFI tools: `grub-mkstandalone` (Debian/Ubuntu `grub-efi-amd64-bin`).
- `sgdisk`, `mtools` (`mcopy`, `mmd`), `mkfs.vfat`, `mkfs.ext4`, `depmod`.
- `qemu-img` only for the disposable QEMU overlay used by the test harness.
- Network access to `dl-cdn.alpinelinux.org` and GitHub on the first build.

`crates/jotnar-node/test/qemu_smoke.py` additionally needs QEMU (`qemu-system-x86`), OVMF 4M
(`ovmf`), and KVM on the machine that runs it.

## Pinned inputs

```text
Alpine minirootfs  3.24.4 x86_64
Alpine linux-virt  6.18.63-r0
Dropbear           2026.91-r0  (development SSH)
utmps-libs         0.1.3.3-r0
zlib               1.4-r0
skalibs-libs       2.15.0.0-r0
Jotnar runtime     this workspace (commit recorded in artifacts/build-info.txt)
```

The Alpine SHA-256 values are constants in the script; a mismatch is fatal.
The node binaries and control client are built from this workspace; the
runtime commit is recorded in the build manifest.

## Environment

```sh
JOTNAR_TOKEN=...            explicit test token (default: reuse/generate artifacts/node.token)
JOTNAR_SSH=0|1              enable key-only development SSH (default: 1 if a key exists)
SSH_AUTHORIZED_KEYS=FILE    public keys for root/jotnar (default: artifacts/authorized_keys)
NODE_HOSTNAME=...           node hostname (default: jotnar-node)
OUT=DIR                     output directory (default: <repo>/artifacts)
CACHE=DIR                   download/build cache (default: OUT/cache)
REBUILD=1                   re-download pinned Alpine inputs
ESP_MB=128                  EFI system partition size
DATA_MB=384                 jotnar-data filesystem size (disk gets extra slack)
```

If `artifacts/authorized_keys` does not exist, the script copies
`~/.ssh/id_ed25519.pub` (or `id_rsa.pub`). For QEMU SSH tests from the Mint,
concatenate the Mint key into that file as well; the file itself is gitignored
and only its hash is recorded.

## Outputs

```text
artifacts/jotnar-node-x86_64.img                 bootable GPT disk image
artifacts/jotnar-node-x86_64.img.sha256
artifacts/jotnar-node-x86_64.initramfs.cpio.gz   for inspection
artifacts/jotnar-node-x86_64.vmlinuz
artifacts/node.token                             test token, mode 0600
artifacts/authorized_keys                        SSH public keys (gitignored)
artifacts/jotnar-payload                         controller CLI for payload/event/console/module commands
artifacts/jotnar-ctl                             host control client (ping/program/store/exec/jobs)
artifacts/jotnar-osd                             emitted control-plane step harness (M7b/M7c)
artifacts/console-alpine-minirootfs-3.24.4.tar.gz  M4 console bundle (pinned Alpine)
artifacts/build-info.txt                         sizes, hashes, versions
```

`artifacts/` is gitignored.

## Boot chain

Short version: UEFI → ESP →
GRUB standalone → Alpine `vmlinuz-virt` + initramfs (minirootfs) → `/init` →
DHCP + data mount + SSH + `jotnar-node` + console shell.

## Node configuration

Build defaults live in `/etc/jotnar/node.conf`:

```sh
JOTNAR_HOSTNAME=jotnar-node
JOTNAR_BIND=0.0.0.0
JOTNAR_PORT=9999
JOTNAR_ALLOW=lattice
JOTNAR_MAX_FRAME=67108864
JOTNAR_TOKEN_FILE=/etc/jotnar/node.token
JOTNAR_DATA_DIR=/var/lib/jotnar
JOTNAR_LOG_DIR=/var/lib/jotnar/log
JOTNAR_SSH=1
JOTNAR_SSH_PORT=22
JOTNAR_SUPERVISOR=1
```

A persistent override at `/var/lib/jotnar/etc/node.conf` is applied last and
survives reflashes only if the data partition is preserved (a full image write
replaces it). `JOTNAR_BIND` accepts either `host:port` or a bare host plus
`JOTNAR_PORT`.

## Service and SSH (development)

```sh
jotnar-service start|stop|restart|status|logs
```

The wrapper is installed at `/usr/local/sbin/jotnar-service` and symlinked into
`/usr/bin` and `/usr/local/bin` so Dropbear non-login shells find it.

Built-in system modules are copied to
`/usr/local/share/jotnar/modules/<name>/`; the module host loads them, and their
state is written under `/var/lib/jotnar/modules/<name>/state.ndjson`.

`jotnar-node` logs to `$JOTNAR_LOG_DIR/node.log`; init logs to `init.log`.
When `JOTNAR_SSH=1`, Dropbear listens on port 22 with **public-key auth only**
(no passwords). Key logins are allowed for `root` and the unprivileged `jotnar`
user. This is a trusted-LAN development feature, not a production security
model; disable it with `JOTNAR_SSH=0`.

The Dropbear host key is generated on first boot and stored at
`/var/lib/jotnar/etc/dropbear/dropbear_ed25519_host_key`, so it is stable across
reboots while the data partition is mounted. A persistent key override can be
placed at `/var/lib/jotnar/etc/ssh/authorized_keys`.

## Payload supervisor

`/usr/local/bin/jotnar-supervisord` runs from init when
`JOTNAR_SUPERVISOR=1`. It owns payload threads, stores manifests/programs under
`/var/lib/jotnar/payloads/<name>/`, and serves `payload-*`/`event-*` frames on
`/run/jotnar/supervisor.sock` (0600). The TCP node binary is
`/usr/local/bin/jotnar-node` (the `jotnar-noded` build) and forwards those
frames to the supervisor. The controller CLI is `/usr/local/bin/jotnar-payload`
(or `artifacts/jotnar-payload` on the development host). See
`../README.md`.

