#!/usr/bin/env python3
"""Deterministic texture and halftone generator for the Jotnar Field Manual."""
from pathlib import Path
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "assets" / "textures"
IMG = ROOT / "assets" / "img"
TEX.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
random.seed(1977)


def paper(path, base, n=512, grain=7, fibre=90):
    img = Image.new("RGB", (n, n), base)
    px = img.load()
    for y in range(n):
        for x in range(n):
            r, g, b = px[x, y]
            d = int(random.gauss(0, grain))
            px[x, y] = (max(0, min(255, r + d)),
                        max(0, min(255, g + d)),
                        max(0, min(255, b + d)))
    # faint fibres
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(fibre):
        x, y = random.randrange(n), random.randrange(n)
        ln = random.randrange(4, 22)
        a = random.randrange(0, 360)
        x2 = x + int(ln * math.cos(math.radians(a)))
        y2 = y + int(ln * math.sin(math.radians(a)))
        d.line((x, y, x2, y2), fill=(30, 24, 18, random.randrange(4, 14)), width=1)
    # a few specks
    for _ in range(120):
        x, y = random.randrange(n), random.randrange(n)
        d.ellipse((x, y, x + 1, y + 1), fill=(20, 16, 12, random.randrange(10, 35)))
    img.save(path, optimize=True)


paper(TEX / "paper.png", (242, 235, 221), grain=6, fibre=110)
paper(TEX / "paper-red.png", (219, 64, 53), grain=7, fibre=80)
paper(TEX / "paper-dark.png", (18, 16, 14), grain=5, fibre=40)

# halftone tile
tile = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
d = ImageDraw.Draw(tile)
d.ellipse((3, 3, 8, 8), fill=(15, 13, 11, 90))
tile.save(TEX / "halftone.png")

# grain overlay
g = Image.new("RGBA", (256, 256))
p = g.load()
for y in range(256):
    for x in range(256):
        v = random.randrange(0, 255)
        p[x, y] = (v, v, v, 16)
g.save(TEX / "grain.png")

# blueprint grid
grid = Image.new("RGBA", (48, 48), (0, 0, 0, 0))
d = ImageDraw.Draw(grid)
d.line((0, 0, 0, 48), fill=(20, 18, 15, 28), width=1)
d.line((0, 0, 48, 0), fill=(20, 18, 15, 28), width=1)
grid.save(TEX / "grid.png")


def halftone(src, dst, width, cell=5, ink=(18, 16, 14), bg=(242, 235, 221),
             crop=None, dot=0.72):
    img = Image.open(src).convert("L")
    if crop:
        img = img.crop(crop)
    h = max(1, int(img.height * width / img.width))
    img = img.resize((width, h), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    sw, sh = max(1, width // cell), max(1, h // cell)
    small = img.resize((sw, sh), Image.BOX)
    out = Image.new("RGB", (sw * cell, sh * cell), bg)
    d = ImageDraw.Draw(out)
    for y in range(sh):
        for x in range(sw):
            v = small.getpixel((x, y)) / 255.0
            r = (1.0 - v) * (cell * 0.5) * dot
            if r > 0.2:
                cx, cy = x * cell + cell / 2, y * cell + cell / 2
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ink)
    # paper speckle
    for _ in range(out.width * out.height // 900):
        x, y = random.randrange(out.width), random.randrange(out.height)
        d.point((x, y), fill=(30, 25, 20))
    out.save(dst, optimize=True)


halftone(IMG / "portrait.png", IMG / "portrait-halftone.png", 520, cell=3,
         ink=(18, 16, 14), bg=(242, 235, 221), dot=0.8)
halftone(IMG / "doom-title.png", IMG / "doom-halftone.png", 720, cell=5,
         ink=(18, 16, 14), bg=(242, 235, 221), dot=0.78)

# four-panel analog-filter duotone (TARGET | LAPLACE | RC | BEST PM .08)
a = Image.open(IMG / "analog-filter.png").convert("L")
a = ImageOps.autocontrast(a, cutoff=1)
a = ImageOps.colorize(a, black=(28, 19, 16), white=(239, 222, 198),
                      mid=(206, 66, 51))
a.save(IMG / "analog-filter-duotone.png", optimize=True)


def image_targets():
    """INPUT | C / STRIP | CUDA / UNSHARP, labelled for the field note."""
    items = [
        ("INPUT", IMG / "image-input.png"),
        ("C / STRIP", IMG / "image-c-strip.png"),
        ("CUDA / UNSHARP", IMG / "image-cuda.png"),
    ]
    height = 220
    gap = 18
    margin = 18
    top = 44
    panels = []
    for label, path in items:
        im = Image.open(path).convert("RGB")
        w = int(im.width * height / im.height)
        panels.append((label, im.resize((w, height), Image.LANCZOS)))
    width = margin * 2 + gap * (len(panels) - 1) + sum(p.width for _, p in panels)
    out = Image.new("RGB", (width, top + height + margin), (242, 235, 221))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, width, top), fill=(23, 20, 17))
    font = ImageFont.truetype(str(ROOT / "assets" / "fonts" / "BarlowCondensed-Bold.ttf"), 28)
    x = margin
    for label, im in panels:
        d.text((x + im.width / 2, top / 2), label, font=font, fill=(242, 235, 221), anchor="mm")
        out.paste(im, (x, top))
        d.rectangle((x, top, x + im.width - 1, top + height - 1), outline=(23, 20, 17), width=2)
        x += im.width + gap
    out.save(IMG / "image-targets.png", optimize=True)


# five-panel analog-field duotone (TARGET | DIGITAL | ANALOG | DAC4 | DAC8)
f = Image.open(IMG / "analog-field.png").convert("L")
f = ImageOps.autocontrast(f, cutoff=1)
f = ImageOps.colorize(f, black=(28, 19, 16), white=(239, 222, 198),
                      mid=(206, 66, 51))
f.save(IMG / "analog-field-duotone.png", optimize=True)

image_targets()

print("assets written")
