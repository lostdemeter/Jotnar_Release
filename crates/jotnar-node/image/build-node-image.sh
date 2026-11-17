#!/bin/sh
# Build a bootable x86-64 JotnarOS node image.
#
#   ./crates/jotnar-node/image/build-node-image.sh
#
# Outputs (under artifacts/, gitignored):
#   jotnar-node-x86_64.img       GPT disk: ESP + jotnar-data ext4
#   jotnar-node-x86_64.img.sha256
#   jotnar-node-x86_64.initramfs.cpio.gz
#   jotnar-node-x86_64.vmlinuz
#   node.token                    test token (mode 0600; never commit)
#   build-info.txt
#
# Environment:
#   JOTNAR_TOKEN=...   explicit test token (otherwise generated/reused)
#   JOTNAR_SSH=0|1     enable key-only development SSH (default: 1 if a key exists)
#   SSH_AUTHORIZED_KEYS=FILE  public keys for root/jotnar (default: OUT/authorized_keys)
#   NODE_HOSTNAME=...  node hostname (default: jotnar-node)
#   OUT=DIR            output directory (default: <repo>/artifacts)
#   REBUILD=1          ignore cached Alpine/kernel downloads
#
# Pinned inputs:
#   Alpine minirootfs 3.24.4 x86_64
#   Alpine linux-virt 6.18.63-r0
#
# The node binary is built for x86_64-unknown-linux-musl so it runs inside
# the Alpine root filesystem without any dynamic libraries.

set -eu

CACHE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NODE_DIR=$(CDPATH= cd -- "$CACHE_DIR/.." && pwd)
PROJECT=$(CDPATH= cd -- "$NODE_DIR/../.." && pwd)
OUT=${OUT:-"$PROJECT/artifacts"}
CACHE=${CACHE:-"$OUT/cache"}
REBUILD=${REBUILD:-0}

ALPINE_VERSION=3.24.4
KVER=6.18.63-0-virt
KERNEL_APK=linux-virt-6.18.63-r0.apk
DROPBEAR_APK=dropbear-2026.91-r0.apk
UTMPS_APK=utmps-libs-0.1.3.3-r0.apk
ZLIB_APK=zlib-1.4-r0.apk
SKALIBS_APK=skalibs-libs-2.15.0.0-r0.apk
MINIROOT=$CACHE"/alpine-minirootfs-$ALPINE_VERSION-x86_64.tar.gz"
KERNEL=$CACHE/$KERNEL_APK
DROPBEAR=$CACHE/$DROPBEAR_APK
UTMPS=$CACHE/$UTMPS_APK
ZLIB=$CACHE/$ZLIB_APK
SKALIBS=$CACHE/$SKALIBS_APK

MINIROOT_SHA256=cdaa38cb8de3537967ee5b6c9b6cd00cc962da2b018e6eea48696bb274923012
KERNEL_SHA256=3a67e6674348ecf160ba85b1aaf8419170095125b0a6b9e6a8341794d1158db1
DROPBEAR_SHA256=cba11fe37aa85d7259ae44d494544e869bb58879e35124d0ba33b32c3b1201d7
UTMPS_SHA256=497f0f1cb51f0e6c1acc338fded0dc7722b1d43ea17bdb749f0d556794f8b2d8
ZLIB_SHA256=7b05bc778b8b25cd880835b40114182bd4d1581717504d0129c810c0a14291cb
SKALIBS_SHA256=3301f6ccbeaea49cdd966c7e636898a8612fb8fe964a15f24f6d204b3944c3f3

ESP_MB=${ESP_MB:-128}
DATA_MB=${DATA_MB:-384}
IMAGE=$OUT/jotnar-node-x86_64.img
ESP=$CACHE/esp.fat
DATA=$CACHE/jotnar-data.ext4
ROOTFS=$CACHE/rootfs
KERNELDIR=$CACHE/kernel
INITRAMFS=$OUT/jotnar-node-x86_64.initramfs.cpio.gz
VMLINUZ=$OUT/jotnar-node-x86_64.vmlinuz
TOKEN_FILE=$OUT/node.token

say() { echo "==> $*"; }
die() { echo "error: $*" >&2; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"
}

for tool in cargo git curl sha256sum cpio gzip grub-mkstandalone sgdisk \
            mkfs.vfat mkfs.ext4 mcopy mmd depmod; do
    need "$tool"
done

mkdir -p "$CACHE" "$OUT"

# --- pinned inputs --------------------------------------------------------
fetch() {
    url=$1
    dest=$2
    want=$3
    if [ -f "$dest" ] && [ "$REBUILD" != "1" ]; then
        got=$(sha256sum "$dest" | awk '{print $1}')
        [ "$got" = "$want" ] || die "cached $(basename "$dest") has wrong hash ($got); set REBUILD=1"
        return
    fi
    say "downloading $(basename "$dest")"
    curl -fL --retry 3 -o "$dest.tmp" "$url"
    got=$(sha256sum "$dest.tmp" | awk '{print $1}')
    [ "$got" = "$want" ] || die "hash mismatch for $(basename "$dest"): $got != $want"

    mv "$dest.tmp" "$dest"
}

fetch "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION%.*}/releases/x86_64/alpine-minirootfs-$ALPINE_VERSION-x86_64.tar.gz" \
    "$MINIROOT" "$MINIROOT_SHA256"
fetch "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION%.*}/main/x86_64/$KERNEL_APK" \
    "$KERNEL" "$KERNEL_SHA256"
fetch "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION%.*}/main/x86_64/$DROPBEAR_APK" \
    "$DROPBEAR" "$DROPBEAR_SHA256"
fetch "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION%.*}/main/x86_64/$UTMPS_APK" \
    "$UTMPS" "$UTMPS_SHA256"
fetch "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION%.*}/main/x86_64/$ZLIB_APK" \
    "$ZLIB" "$ZLIB_SHA256"
fetch "https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION%.*}/main/x86_64/$SKALIBS_APK" \
    "$SKALIBS" "$SKALIBS_SHA256"

# The M4 console payload bundle is the pinned Alpine minirootfs itself.
cp "$MINIROOT" "$OUT/console-alpine-minirootfs-$ALPINE_VERSION.tar.gz"

# --- runtime binaries -----------------------------------------------------
RUNTIME_COMMIT=$(git -C "$PROJECT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
say "runtime commit $RUNTIME_COMMIT"

say "building node daemons (musl static)"
(
    cd "$PROJECT"
    cargo build --release --target x86_64-unknown-linux-musl -p jotnar-node-tools \
        --bin jotnar-noded --bin jotnar-supervisord --bin jotnar-payload --bin jotnar-osd >/dev/null
)
TOOLS=$PROJECT/target/x86_64-unknown-linux-musl/release
NODE_BIN=$TOOLS/jotnar-noded
SUPERVISOR_BIN=$TOOLS/jotnar-supervisord
PAYLOAD_CTL_BIN=$TOOLS/jotnar-payload
OSD_BIN=$TOOLS/jotnar-osd
for b in "$NODE_BIN" "$SUPERVISOR_BIN" "$PAYLOAD_CTL_BIN" "$OSD_BIN"; do
    [ -f "$b" ] || die "node tool not found at $b"
done
cp "$PAYLOAD_CTL_BIN" "$OUT/jotnar-payload"
cp "$OSD_BIN" "$OUT/jotnar-osd"

say "building host control client"
(
    cd "$PROJECT"
    cargo build --release --bin jotnar-ctl >/dev/null
)
CTL_BIN=$PROJECT/target/release/jotnar-ctl
[ -f "$CTL_BIN" ] || die "control client not found at $CTL_BIN"
cp "$CTL_BIN" "$OUT/jotnar-ctl"

# --- token ----------------------------------------------------------------
if [ -n "${JOTNAR_TOKEN:-}" ]; then
    TOKEN=$JOTNAR_TOKEN
elif [ -f "$TOKEN_FILE" ]; then
    TOKEN=$(cat "$TOKEN_FILE")
else
    TOKEN=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
fi
[ -n "$TOKEN" ] || die "token must not be empty"
umask 077
printf '%s\n' "$TOKEN" >"$TOKEN_FILE"
umask 022

# --- development SSH keys -------------------------------------------------
# Key-only SSH is a development convenience on the trusted LAN. It is enabled
# by default when a public key is available; set JOTNAR_SSH=0 to disable.
SSH_ENABLE=${JOTNAR_SSH:-1}
AUTH_KEYS=${SSH_AUTHORIZED_KEYS:-"$OUT/authorized_keys"}
if [ ! -f "$AUTH_KEYS" ]; then
    for k in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub"; do
        if [ -f "$k" ]; then
            cp "$k" "$AUTH_KEYS"
            break
        fi
    done
fi
if [ ! -s "$AUTH_KEYS" ]; then
    say "warning: no SSH public key found; disabling SSH (set SSH_AUTHORIZED_KEYS=FILE)"
    SSH_ENABLE=0
    AUTH_KEYS=
fi

# --- root filesystem ------------------------------------------------------
say "assembling Alpine root filesystem"
rm -rf "$ROOTFS" "$KERNELDIR"
mkdir -p "$ROOTFS" "$KERNELDIR"
tar -xzf "$MINIROOT" -C "$ROOTFS"
tar -xzf "$KERNEL" -C "$KERNELDIR" 2>/dev/null

cp "$KERNELDIR/boot/vmlinuz-virt" "$VMLINUZ"
cp "$NODE_BIN" "$ROOTFS/usr/local/bin/jotnar-node"
cp "$SUPERVISOR_BIN" "$ROOTFS/usr/local/bin/jotnar-supervisord"
cp "$PAYLOAD_CTL_BIN" "$ROOTFS/usr/local/bin/jotnar-payload"
cp "$OSD_BIN" "$ROOTFS/usr/local/bin/jotnar-osd"
chmod 0755 "$ROOTFS/usr/local/bin/jotnar-node" \
           "$ROOTFS/usr/local/bin/jotnar-supervisord" \
           "$ROOTFS/usr/local/bin/jotnar-payload" \
           "$ROOTFS/usr/local/bin/jotnar-osd"

mkdir -p "$ROOTFS/etc/jotnar" "$ROOTFS/var/log" "$ROOTFS/var/lib/jotnar"
cp "$CACHE_DIR/rootfs/init" "$ROOTFS/init"
chmod 0755 "$ROOTFS/init"
mkdir -p "$ROOTFS/usr/local/lib/jotnar" "$ROOTFS/usr/local/sbin"
cp "$CACHE_DIR/rootfs/usr/local/lib/jotnar/config.sh" "$ROOTFS/usr/local/lib/jotnar/config.sh"
cp "$CACHE_DIR/rootfs/usr/local/sbin/jotnar-service" "$ROOTFS/usr/local/sbin/jotnar-service"
chmod 0755 "$ROOTFS/usr/local/lib/jotnar/config.sh" "$ROOTFS/usr/local/sbin/jotnar-service"
mkdir -p "$ROOTFS/usr/local/share/jotnar"
cp -a "$CACHE_DIR/rootfs/usr/local/share/jotnar/modules" "$ROOTFS/usr/local/share/jotnar/"
# Dropbear non-login shells use /usr/sbin:/usr/bin:/sbin:/bin. Put the service
# wrapper on that PATH as well as in /usr/local/sbin.
ln -sf /usr/local/sbin/jotnar-service "$ROOTFS/usr/bin/jotnar-service"
ln -sf /usr/local/sbin/jotnar-service "$ROOTFS/usr/local/bin/jotnar-service"

# Kernel modules. Decompress them so busybox modprobe does not depend on
# gzip-seamless support, then regenerate modules.dep for the new names.
mkdir -p "$ROOTFS/lib/modules/$KVER"
cp -a "$KERNELDIR/lib/modules/$KVER/." "$ROOTFS/lib/modules/$KVER/"
find "$ROOTFS/lib/modules/$KVER" -type f -name '*.ko.gz' -exec gunzip -f {} \;
depmod -b "$ROOTFS" "$KVER" 2>/dev/null || die "depmod failed"

# Dropbear + its runtime libraries. This is a development SSH service, not a
# production remote-access story: key auth only, trusted LAN only.
if [ "$SSH_ENABLE" = "1" ]; then
    say "installing dropbear"
    PKGDIR=$CACHE/pkgs
    rm -rf "$PKGDIR"
    mkdir -p "$PKGDIR"
    for apk in "$DROPBEAR" "$UTMPS" "$ZLIB" "$SKALIBS"; do
        name=$(basename "$apk" .apk)
        mkdir -p "$PKGDIR/$name"
        tar -xzf "$apk" -C "$PKGDIR/$name" 2>/dev/null
    done
    cp "$PKGDIR/${DROPBEAR_APK%.apk}/usr/sbin/dropbear" "$ROOTFS/usr/sbin/dropbear"
    cp "$PKGDIR/${DROPBEAR_APK%.apk}/usr/bin/dropbearkey" "$ROOTFS/usr/bin/dropbearkey"
    chmod 0755 "$ROOTFS/usr/sbin/dropbear" "$ROOTFS/usr/bin/dropbearkey"
    for p in "$UTMPS_APK" "$ZLIB_APK" "$SKALIBS_APK"; do
        cp -a "$PKGDIR/${p%.apk}/usr/lib/." "$ROOTFS/usr/lib/"
    done
    mkdir -p "$ROOTFS/etc/dropbear" "$ROOTFS/home/jotnar/.ssh" "$ROOTFS/root/.ssh"
    chmod 0700 "$ROOTFS/home/jotnar/.ssh" "$ROOTFS/root/.ssh"
    cp "$AUTH_KEYS" "$ROOTFS/home/jotnar/.ssh/authorized_keys"
    cp "$AUTH_KEYS" "$ROOTFS/root/.ssh/authorized_keys"
    chmod 0600 "$ROOTFS/home/jotnar/.ssh/authorized_keys" \
                "$ROOTFS/root/.ssh/authorized_keys"
    if ! grep -q '^jotnar:' "$ROOTFS/etc/passwd"; then
        echo 'jotnar:x:1000:1000:Jotnar development:/home/jotnar:/bin/sh' >>"$ROOTFS/etc/passwd"
        echo 'jotnar:!::0:::::' >>"$ROOTFS/etc/shadow"
        echo 'jotnar:x:1000:' >>"$ROOTFS/etc/group"
    fi
fi

cat >"$ROOTFS/etc/jotnar/node.conf" <<EOF
# JotnarOS node configuration (generated by build-node-image.sh).
# A persistent override may be placed at /var/lib/jotnar/etc/node.conf.
JOTNAR_HOSTNAME=${NODE_HOSTNAME:-jotnar-node}
JOTNAR_BIND=0.0.0.0
JOTNAR_PORT=9999
JOTNAR_ALLOW=lattice
JOTNAR_MAX_FRAME=67108864
JOTNAR_MDNS=1
JOTNAR_TOKEN_FILE=/etc/jotnar/node.token
JOTNAR_DATA_DIR=/var/lib/jotnar
JOTNAR_LOG_DIR=/var/lib/jotnar/log
JOTNAR_SSH=${SSH_ENABLE}
JOTNAR_SSH_PORT=22
JOTNAR_SUPERVISOR=1
JOTNAR_OSD=1
JOTNAR_MODULE_SOCKET=/run/jotnar/osd.sock
JOTNAR_CONSOLE_AUTOSTART=1
EOF
chmod 0644 "$ROOTFS/etc/jotnar/node.conf"
umask 077
printf '%s\n' "$TOKEN" >"$ROOTFS/etc/jotnar/node.token"
umask 022
chmod 0600 "$ROOTFS/etc/jotnar/node.token"

cat >"$ROOTFS/etc/hostname" <<EOF
${NODE_HOSTNAME:-jotnar-node}
EOF
cat >"$ROOTFS/etc/hosts" <<EOF
127.0.0.1 localhost ${NODE_HOSTNAME:-jotnar-node}
::1 localhost
EOF

say "building initramfs"
(
    cd "$ROOTFS"
    find . -print0 | cpio --null -o -H newc --owner=0:0 2>/dev/null | gzip -9
) >"$INITRAMFS"

# --- EFI system partition -------------------------------------------------
say "building GRUB EFI image"
GRUB_STAGE=$CACHE/grub
rm -rf "$GRUB_STAGE"
mkdir -p "$GRUB_STAGE"
cp "$CACHE_DIR/grub.cfg" "$GRUB_STAGE/grub.cfg"
grub-mkstandalone -O x86_64-efi \
    -o "$GRUB_STAGE/BOOTX64.EFI" \
    --modules="part_gpt part_msdos fat search search_fs_file search_label linux normal configfile echo test" \
    --locales="" --fonts="" --themes="" \
    "boot/grub/grub.cfg=$GRUB_STAGE/grub.cfg" >/dev/null

say "building ${ESP_MB} MiB FAT ESP"
rm -f "$ESP"
truncate -s "${ESP_MB}M" "$ESP"
mkfs.vfat -F 32 -n JOTNARESP "$ESP" >/dev/null
mmd -i "$ESP" ::/EFI ::/EFI/BOOT ::/boot
mcopy -i "$ESP" "$GRUB_STAGE/BOOTX64.EFI" ::/EFI/BOOT/BOOTX64.EFI
mcopy -i "$ESP" "$VMLINUZ" ::/boot/vmlinuz
mcopy -i "$ESP" "$INITRAMFS" ::/boot/initramfs.cpio.gz

# --- data partition -------------------------------------------------------
say "building ${DATA_MB} MiB jotnar-data partition"
rm -f "$DATA"
truncate -s "${DATA_MB}M" "$DATA"
mkfs.ext4 -q -F -L jotnar-data -m 0 "$DATA"

# --- disk image -----------------------------------------------------------
TOTAL_MB=$((ESP_MB + DATA_MB + 4))
say "building GPT disk image (${TOTAL_MB} MiB)"
rm -f "$IMAGE"
truncate -s "${TOTAL_MB}M" "$IMAGE"
sgdisk --clear \
    --set-alignment=2048 \
    --new=1:2048:+"${ESP_MB}"M --typecode=1:EF00 --change-name=1:JOTNARESP \
    --new=2:0:0 --typecode=2:8300 --change-name=2:jotnar-data \
    "$IMAGE" >/dev/null

P1=$(sgdisk -i 1 "$IMAGE" | sed -n 's/^First sector: \([0-9][0-9]*\).*/\1/p')
P2=$(sgdisk -i 2 "$IMAGE" | sed -n 's/^First sector: \([0-9][0-9]*\).*/\1/p')
[ -n "$P1" ] && [ -n "$P2" ] || die "could not read partition offsets"

dd if="$ESP" of="$IMAGE" bs=512 seek="$P1" conv=notrunc status=none
dd if="$DATA" of="$IMAGE" bs=512 seek="$P2" conv=notrunc status=none
sync

# --- manifest -------------------------------------------------------------
IMG_SHA=$(sha256sum "$IMAGE" | awk '{print $1}')
INITRAMFS_SHA=$(sha256sum "$INITRAMFS" | awk '{print $1}')
VMLINUZ_SHA=$(sha256sum "$VMLINUZ" | awk '{print $1}')
if [ -n "$AUTH_KEYS" ] && [ -f "$AUTH_KEYS" ]; then
    AUTH_KEYS_SHA=$(sha256sum "$AUTH_KEYS" | awk '{print $1}')
else
    AUTH_KEYS_SHA=none
fi

printf '%s  %s\n' "$IMG_SHA" "$IMAGE" >"$IMAGE.sha256"

cat >"$OUT/build-info.txt" <<EOF
built:            $(date -u +%Y-%m-%dT%H:%M:%SZ)
host:             $(uname -srm)
alpine:           $ALPINE_VERSION (minirootfs, x86_64)
kernel:           $KVER (Alpine linux-virt $KERNEL_APK)
runtime_commit:   $RUNTIME_COMMIT
node_build:       x86_64-unknown-linux-musl, release
image:            $IMAGE
image_sha256:     $IMG_SHA
initramfs:        $INITRAMFS
initramfs_sha256: $INITRAMFS_SHA
vmlinuz:          $VMLINUZ
vmlinuz_sha256:   $VMLINUZ_SHA
partitions:       p1 JOTNARESP (${ESP_MB} MiB FAT32), p2 jotnar-data (${DATA_MB} MiB ext4)
ssh:              $([ "$SSH_ENABLE" = 1 ] && echo "enabled (port 22, key auth: root, jotnar)" || echo "disabled")
authorized_keys:  ${AUTH_KEYS_SHA}
token_file:       $TOKEN_FILE
EOF

say "done"
cat "$OUT/build-info.txt"
