#!/bin/sh
# Flash a verified JotnarOS node image to a USB disk.
#
#   sudo ./crates/jotnar-node/image/flash-usb.sh /dev/sdX
#
# The script refuses non-USB devices, the root disk, a too-small target, and an
# image whose hash does not match its .sha256 sidecar. It requires the word
# ERASE, unmounts the target's partitions, wipes old signatures, and writes the
# image.
set -eu

IMAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT=${JOTNAR_PROJECT:-$(CDPATH= cd -- "$IMAGE_DIR/../.." && pwd)}
IMAGE=${IMAGE:-"$PROJECT/artifacts/jotnar-node-x86_64.img"}
DEV=${1:-}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[ "$(id -u)" -eq 0 ] || die "run with sudo: sudo $0 /dev/sdX"
[ -n "$DEV" ] || die "usage: sudo $0 /dev/sdX"
[ -b "$DEV" ] || die "$DEV is not a block device"
DEV=$(readlink -f "$DEV")

case "$DEV" in
    /dev/sd?|/dev/mmcblk?|/dev/nvme?n?) ;;
    *) die "refusing non-whole-disk device name: $DEV" ;;
esac

# Never erase the disk that holds the running system.
ROOT_SRC=$(findmnt -n -o SOURCE /)
ROOT_DISK=$(lsblk -no PKNAME "$ROOT_SRC" 2>/dev/null | head -1 || true)
if [ -n "$ROOT_DISK" ] && [ "/dev/$ROOT_DISK" = "$DEV" ]; then
    die "refusing to erase the root disk $DEV"
fi

TRAN=$(lsblk -dno TRAN "$DEV")
MODEL=$(lsblk -dno MODEL "$DEV" | sed 's/ *$//')
SIZE=$(blockdev --getsize64 "$DEV")

[ "$TRAN" = "usb" ] || die "$DEV is not a USB device (TRAN=$TRAN)"
[ -f "$IMAGE" ] || die "missing image: $IMAGE"

# Prefer an explicit hash, otherwise use the builder's .sha256 sidecar. This
# keeps the check tied to the current image instead of a stale hardcoded value.
if [ -z "${EXPECTED:-}" ]; then
    [ -f "$IMAGE.sha256" ] || die "missing $IMAGE.sha256 (or set EXPECTED)"
    EXPECTED=$(awk 'NR==1 {print $1}' "$IMAGE.sha256")
fi
[ -n "$EXPECTED" ] || die "empty expected hash for $IMAGE"

IMG_SIZE=$(stat -c %s "$IMAGE")
if [ "$SIZE" -lt "$IMG_SIZE" ]; then
    die "target too small: $SIZE bytes < image $IMG_SIZE bytes"
fi

ACTUAL=$(sha256sum "$IMAGE" | awk '{print $1}')
if [ "$ACTUAL" != "$EXPECTED" ]; then
    die "image hash mismatch ($ACTUAL != $EXPECTED); re-copy the verified image"
fi

cat <<EOF
WARNING: this erases the whole USB disk.

  target : $DEV
  model  : $MODEL
  size   : $(numfmt --to=iec "$SIZE")
  image  : $IMAGE
  sha256 : $ACTUAL

Type ERASE to continue:
EOF
read -r reply
[ "$reply" = "ERASE" ] || { echo "aborted"; exit 1; }

# Unmount every partition on the target (it may have auto-mounted).
for p in $(lsblk -lnpo NAME "$DEV" | tail -n +2); do
    umount "$p" 2>/dev/null || true
done

wipefs -a "$DEV"
dd if="$IMAGE" of="$DEV" bs=4M conv=fsync status=progress
sync
blockdev --flushbufs "$DEV"
udevadm settle 2>/dev/null || true

echo
echo "Done."
lsblk -o NAME,SIZE,TYPE,TRAN,MODEL,MOUNTPOINTS "$DEV"
echo "Flashed sha256: $ACTUAL"
echo "Remove the USB and boot the Dell from it."
