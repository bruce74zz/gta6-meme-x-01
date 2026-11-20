#!/usr/bin/env python3
"""Build the gta6-meme-x-01 meme picture.

Takes the AI-generated Vice City living-room scene in assets/, cuts the
official make.fun ghost mascot out of the press mark (assets/mark-1024.png),
sits it on the empty half of the couch, adds a neon make.fun sign in the sky
and the caption in big condensed letters. Output: meme.png in the repo root.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SCENE_FILE = ASSETS / "media-generation-scene-base-0-532d0231-319d-475e-8ef4-8b90bb2631cf.png"
MARK_FILE = ASSETS / "mark-1024.png"
OUT_FILE = ROOT / "meme.png"
GHOST_FILE = ASSETS / "ghost_cutout.png"

CANVAS = 1200
CAPTION_FONT = "/usr/share/fonts/truetype/noto/NotoSansDisplay-CondensedBlack.ttf"
SIGN_FONT = "/usr/share/fonts/truetype/noto/NotoSansDisplay-Black.ttf"

TOP_TEXT = "WAITED 13 YEARS FOR GTA VI"
BOTTOM_LINES = ["MY AI AGENT ALREADY GOT PAID", "FOR A GTA VI MEME"]
MAGENTA = (196, 28, 131)


def largest_white_blob(mark: Image.Image) -> Image.Image:
    """Cut the ghost (the biggest white shape in the mark) out of the badge."""
    arr = np.asarray(mark.convert("RGBA")).astype(int)
    white = (arr[..., 0] > 205) & (arr[..., 1] > 205) & (arr[..., 2] > 205)

    # Label connected white regions with a simple two-pass union-find.
    h, w = white.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nxt = 0
    for y in range(h):
        for x in range(w):
            if not white[y, x]:
                continue
            seen = []
            if x > 0 and labels[y, x - 1]:
                seen.append(labels[y, x - 1])
            if y > 0 and labels[y - 1, x]:
                seen.append(labels[y - 1, x])
            if not seen:
                nxt += 1
                parent.append(nxt)
                labels[y, x] = nxt
            else:
                keep = min(seen)
                labels[y, x] = keep
                for other in seen:
                    ra, rb = find(keep), find(other)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([0] + [find(i) for i in range(1, nxt + 1)])
    flat = roots[labels.ravel()].reshape(h, w)
    sizes = np.bincount(flat[white], minlength=nxt + 1)
    ghost = flat == int(np.argmax(sizes))

    # Grow the blob a little so the ghost keeps its thick black outline,
    # then knock out any magenta badge pixels that growth reached into.
    grown = Image.fromarray((ghost * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(21)
    )
    keep = np.asarray(grown) > 0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    magenta = (r > 140) & (g < 130) & (b > 70) & (r > g + 60)
    keep &= ~magenta

    out = np.asarray(mark.convert("RGBA")).copy()
    out[..., 3] = np.where(keep, out[..., 3], 0)
    ys, xs = np.where(keep)
    cut = Image.fromarray(out).crop(
        (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    )
    cut.save(GHOST_FILE)
    return cut


def fit_font(draw: ImageDraw.ImageDraw, text: str, path: str, max_w: int,
             start: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > 20:
        font = ImageFont.truetype(path, size)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 4
    return ImageFont.truetype(path, size)


def caption(draw: ImageDraw.ImageDraw, cx: int, y: int, text: str,
            font: ImageFont.FreeTypeFont, stroke: int) -> int:
    draw.text((cx, y), text, font=font, fill="white", anchor="ma",
              stroke_width=stroke, stroke_fill="black")
    return int(font.size * 1.08)


def neon_sign(base: Image.Image, text: str) -> None:
    """A pink neon make.fun sign glowing in the sky, top right."""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = ImageFont.truetype(SIGN_FONT, 74)
    x, y = CANVAS - 46, 40
    d.text((x, y), text, font=font, fill=MAGENTA + (255,), anchor="ra")
    glow = layer.filter(ImageFilter.GaussianBlur(14))
    base.alpha_composite(glow)
    base.alpha_composite(glow)
    base.alpha_composite(layer)
    d2 = ImageDraw.Draw(base)
    d2.text((x, y), text, font=font, fill=(255, 214, 240, 255), anchor="ra")


def main() -> None:
    scene = Image.open(SCENE_FILE).convert("RGBA")
    scene = scene.resize((CANVAS, CANVAS), Image.LANCZOS)

    ghost = largest_white_blob(Image.open(MARK_FILE))
    gh = 470
    gw = int(ghost.width * gh / ghost.height)
    ghost = ghost.resize((gw, gh), Image.LANCZOS)
    # Sit the ghost on the right-hand couch cushion, bottom sunk into the seat,
    # with a soft shadow under it so it rests in the scene instead of floating.
    gx, gy = CANVAS - gw - 90, 955 - gh
    shadow = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.ellipse((gx + 30, 955 - 46, gx + gw - 30, 955 + 26), fill=(0, 0, 0, 110))
    scene.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(16)))
    scene.alpha_composite(ghost, (gx, gy))

    neon_sign(scene, "make.fun")

    draw = ImageDraw.Draw(scene)
    top_font = fit_font(draw, TOP_TEXT, CAPTION_FONT, int(CANVAS * 0.94), 118)
    caption(draw, CANVAS // 2, 118, TOP_TEXT, top_font, 9)

    y = CANVAS - 268
    for line in BOTTOM_LINES:
        font = fit_font(draw, line, CAPTION_FONT, int(CANVAS * 0.94), 108)
        y += caption(draw, CANVAS // 2, y, line, font, 9)

    scene.convert("RGB").save(OUT_FILE)
    print("wrote", OUT_FILE)


if __name__ == "__main__":
    main()
