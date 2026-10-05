#!/usr/bin/env python3
"""Cinematic procedural fire renderer for animated GitHub profile banners.

Every banner is rendered frame by frame and encoded as a seamlessly looping
animated WebP:

* fire     - periodic FFT noise volume (x, y, time), advected upward and
             domain-warped, coloured with a blackbody ramp in HDR
* air      - fire-lit smoke, heat haze, motion-blurred embers, bokeh sparks
* optics   - multi-radius bloom, filmic (ACES) tone mapping, vignette
* type     - forged-metal lettering lit from below by the flames, red-hot at
             the base; frosted glass that really blurs the frame behind it

Usage:
    python render_fire.py OUT_DIR --font Cinzel.ttf [--only header] [--still]
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import fft as sfft
from scipy import ndimage


# --------------------------------------------------------------------------- scenes
@dataclass
class Line:
    text: str
    size: float
    y: float                      # vertical centre of the capitals, px
    weight: int = 900
    track: float = 0.08           # letter spacing, em
    color: tuple = (1.0, 0.9, 0.8)


@dataclass
class Scene:
    name: str
    W: int
    H: int
    T: int                        # frames per loop
    fps: int
    flame_h: float                # mean flame height, px
    title: Line | None = None     # forged-metal lettering
    lines: list = field(default_factory=list)  # flat lettering (on glass)
    glass: tuple | None = None    # x0, y0, x1, y1
    embers: int = 60
    bokeh: int = 4
    seed: int = 1
    k: int = 1                    # noise periods travelled upward per loop
    s: float = 0.35               # noise samples per px
    stretch: float = 0.55         # < 1 makes flame tongues taller
    haze: float = 2.4             # heat-haze displacement, px
    front: float = 0.5            # flames drawn in front of the title
    radius: int = 18
    gain: float = 0.94            # fire temperature scale
    hvar: float = 0.3             # flame-height variation along x
    hot: float = 0.62             # red-hot fraction of the letter height
    shine: float = 1.25           # firelight reflected by the letters
    q: int = 72                   # WebP quality


def scenes():
    out = [
        Scene("header", 1280, 480, 72, 24, flame_h=255, seed=7, q=82, embers=75, bokeh=6,
              title=Line("MUKTADI", 150, 168, weight=900, track=0.09),
              glass=(300, 316, 980, 404),
              lines=[Line("FULL-STACK AI DEVELOPER", 24, 346, weight=700, track=0.26, color=(1.0, 0.92, 0.85)),
                     Line("CSE · SOUTHEAST UNIVERSITY · DHAKA", 13.5, 379, weight=700, track=0.34, color=(1.0, 0.6, 0.36))]),
        Scene("footer", 1280, 320, 60, 24, flame_h=175, seed=23, q=80, embers=60, bokeh=4, front=0.3,
              hot=1.5, shine=2.2, gain=0.88, title=Line("SHIP IT · SECURE IT · MAKE IT SMARTER", 36, 100, weight=900, track=0.12)),
    ]
    for i, (slug, label) in enumerate([("whoami", "WHOAMI"), ("arsenal", "ARSENAL"), ("projects", "PROJECTS"),
                                        ("stats", "BATTLE STATS"), ("now", "CURRENTLY"), ("connect", "SUMMON ME")]):
        out.append(Scene(f"sec-{slug}", 1280, 150, 40, 20, flame_h=62, seed=101 + i * 17, embers=26, bokeh=2,
                         title=Line(label, 46, 58, weight=900, track=0.3), front=0.2, haze=1.2, radius=14, q=70,
                         s=0.5, gain=0.8, hvar=0.22, hot=1.5, shine=2.4))
    return out


# --------------------------------------------------------------------------- helpers
def periodic_noise(shape, beta, t_sigma, seed):
    """Zero-mean, unit-variance noise that is periodic on every axis (T, Y, X)."""
    T, Y, X = shape
    rng = np.random.default_rng(seed)
    spec = sfft.rfftn(rng.standard_normal(shape, dtype=np.float32), workers=-1)
    ft = (sfft.fftfreq(T) * T)[:, None, None]            # cycles per loop
    fy = sfft.fftfreq(Y)[None, :, None]
    fx = sfft.rfftfreq(X)[None, None, :]
    fr = np.sqrt(fx ** 2 + fy ** 2)
    amp = (fr + 2.0 / X) ** (-beta) * np.exp(-(fr / 0.22) ** 2) * np.exp(-(ft / t_sigma) ** 2)
    amp *= 1 - np.exp(-(fr * X / 7.0) ** 2)               # no frame-sized swells: steady overall height
    vol = sfft.irfftn(spec * amp.astype(np.float32), s=shape, workers=-1).astype(np.float32)
    vol -= vol.mean(axis=(1, 2), keepdims=True)
    vol /= vol.std()
    return vol


def sample(slc, xs, ys):
    return ndimage.map_coordinates(slc, [ys, xs], order=1, mode="grid-wrap")


RAMP_T = np.array([0.0, 0.12, 0.28, 0.45, 0.62, 0.78, 0.92, 1.0], np.float32)
RAMP_C = np.array([(0, 0, 0), (.06, .004, 0), (.42, .035, .002), (1.1, .2, .012), (1.85, .55, .055),
                   (2.45, 1.05, .17), (2.95, 1.62, .42), (3.3, 2.25, .85)], np.float32)


def blackbody(t):
    return np.stack([np.interp(t, RAMP_T, RAMP_C[:, c]) for c in range(3)], -1).astype(np.float32)


def blur_big(img, sigma):
    """Gaussian blur for large radii: blur at reduced resolution, upsample."""
    f = max(1, int(sigma // 4))
    H, W = img.shape[:2]
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        ch = np.ascontiguousarray(img[..., c], dtype=np.float32)
        small = np.asarray(Image.fromarray(ch).resize((max(1, W // f), max(1, H // f)), Image.BILINEAR))
        small = np.ascontiguousarray(ndimage.gaussian_filter(small, sigma / f, mode="nearest"), dtype=np.float32)
        out[..., c] = np.asarray(Image.fromarray(small).resize((W, H), Image.BILINEAR))
    return out


def text_mask(W, H, line, font_path, ss=3):
    """Antialiased mask of a line of capitals centred at line.y; also returns (top, bottom)."""
    font = ImageFont.truetype(font_path, int(round(line.size * ss)))
    try:
        font.set_variation_by_axes([line.weight])
    except Exception:
        pass
    track = line.track * line.size * ss
    widths = [font.getlength(c) for c in line.text]
    total = sum(widths) + track * (len(line.text) - 1)
    cap_h = -font.getbbox("H", anchor="ls")[1]
    img = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(img)
    x, base = (W * ss - total) / 2, line.y * ss + cap_h / 2
    for c, w in zip(line.text, widths):
        d.text((x, base), c, font=font, fill=255, anchor="ls")
        x += w + track
    m = np.asarray(img.resize((W, H), Image.LANCZOS), np.float32) / 255
    return m, (line.y - cap_h / ss / 2, line.y + cap_h / ss / 2)


def rounded_mask(W, H, box, r, ss=4, outline=False):
    img = Image.new("L", (W * ss, H * ss), 0)
    x0, y0, x1, y1 = box
    ImageDraw.Draw(img).rounded_rectangle((x0 * ss, y0 * ss, x1 * ss - 1, y1 * ss - 1), r * ss,
                                          fill=None if outline else 255, outline=255 if outline else None,
                                          width=ss if outline else 0)
    return np.asarray(img.resize((W, H), Image.LANCZOS), np.float32) / 255


def aces(x):
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055)


# --------------------------------------------------------------------------- scene state (shared with workers)
G = {}


def prepare(sc, font):
    W, H = sc.W, sc.H
    Nx = int(round(W * sc.s))
    Ny = max(64, int(round(H * sc.s * sc.stretch)) // 2 * 2)
    G["sc"] = sc
    G["vol"] = periodic_noise((sc.T, Ny, Nx), beta=1.55, t_sigma=3.0, seed=sc.seed)
    G["volw"] = ndimage.gaussian_filter(G["vol"], (0, 3, 3), mode="wrap")
    G["volw"] /= G["volw"].std()
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    G["X"], G["Y"] = X, Y
    G["Ny"] = Ny

    # ---- forged-metal title: static geometry
    if sc.title:
        M, (top, bot) = text_mask(W, H, sc.title, font)
        size = sc.title.size
        hgt = ndimage.gaussian_filter(M, max(0.8, size / 70))
        gy, gx = ndimage.sobel(hgt, 0), ndimage.sobel(hgt, 1)
        norm = np.percentile(np.hypot(gx, gy)[M > 0.05], 97) + 1e-6
        rng = np.random.default_rng(sc.seed + 1)
        brushed = ndimage.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), (0.4, 5))
        grain = ndimage.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), 0.7)
        tex = 1 + 0.14 * brushed / (brushed.std() + 1e-6) + 0.22 * grain / (grain.std() + 1e-6)
        cap = bot - top
        G["M"] = M
        G["down"] = np.clip(-gy / norm, 0, 1) * M          # bevels facing the fire
        G["up"] = np.clip(gy / norm, 0, 1) * M             # bevels facing the dark sky
        G["side"] = np.clip(np.abs(gx) / norm, 0, 1) * M
        G["heat"] = np.clip((Y - (bot - cap * sc.hot)) / (cap * sc.hot), 0, 1) ** 2.2 * M
        G["tex"] = np.clip(tex, 0.3, 2.0).astype(np.float32)
        G["halo"] = ndimage.gaussian_filter(M, size / 7)
        G["halo_n"] = np.clip(ndimage.gaussian_filter(M, size / 18) * 2.5, 0, 1)
        G["front_mask"] = np.clip(ndimage.gaussian_filter(M, 2.0) * 1.6, 0, 1)
    # ---- flat lettering + glass
    G["lines"] = [(text_mask(W, H, ln, font)[0], np.array(ln.color, np.float32)) for ln in sc.lines]
    if sc.glass:
        G["glass"] = rounded_mask(W, H, sc.glass, 16)
        G["glass_edge"] = rounded_mask(W, H, sc.glass, 16, outline=True)
        gy0, gy1 = sc.glass[1], sc.glass[3]
        G["glass_sheen"] = np.clip(1 - (Y - gy0) / (gy1 - gy0), 0, 1) ** 2
    G["frame"] = rounded_mask(W, H, (0, 0, W, H), sc.radius)
    cx, cy = (X - W / 2) / (W / 2), (Y - H * 0.55) / (H * 0.75)
    G["vignette"] = np.clip(1 - 0.32 * (cx ** 2 + cy ** 2), 0.45, 1)[..., None]
    G["dither"] = (np.random.default_rng(3).random((H, W, 1)).astype(np.float32) - 0.5) / 255

    G.pop("coverage", None)
    G["coverage"] = float(np.mean([_fire_field(t, 0.0, 1.0)[0].mean() for t in range(0, sc.T, max(1, sc.T // 12))]))

    # ---- particles (deterministic, looping)
    rng = np.random.default_rng(sc.seed + 2)
    n = sc.embers
    G["emb"] = dict(x0=rng.uniform(-40, W + 40, n), p0=rng.random(n), m=np.ones(n, int),
                    rise=rng.uniform(0.45, 1.0, n) * H, drift=rng.uniform(-90, 90, n) * H / 480,
                    sway=rng.uniform(4, 22, n), cyc=rng.uniform(0.6, 2.2, n), ph=rng.random(n),
                    fl=rng.integers(5, 15, n), fph=rng.random(n), r=rng.uniform(0.6, 1.6, n) ** 1.5, br=rng.uniform(0.35, 1.0, n),
                    start=H - rng.uniform(0, 0.35, n) * sc.flame_h)
    nb = sc.bokeh
    G["bok"] = dict(x0=rng.uniform(0, W, nb), p0=rng.random(nb), r=rng.uniform(5, 12, nb) * min(1, H / 300 + 0.4),
                    drift=rng.uniform(-60, 60, nb))


def fire_field(t, xoff, hscale):
    """Fire intensity for frame t, with the flame height gently held near the loop average."""
    fire, hx = _fire_field(t, xoff, hscale)
    target = G.get("coverage")
    if target:
        fire, hx = _fire_field(t, xoff, hscale * (target / max(fire.mean(), 1e-3)) ** 0.55)
    return fire, hx


def _fire_field(t, xoff, hscale):
    sc, vol, volw, X, Y, Ny = G["sc"], G["vol"][t], G["volw"][t], G["X"], G["Y"], G["Ny"]
    ph = t / sc.T
    adv = ph * Ny * sc.k
    xs = X * sc.s + xoff
    ys = Y * sc.s * sc.stretch
    wx = sample(volw, xs + 37.1, ys + adv)
    wy = sample(volw, xs + 11.7, ys + adv + Ny * 0.5)
    n1 = sample(vol, xs + 5.5 * wx, ys + adv + 5.5 * wy)
    n2 = sample(vol, xs * 2.13 + 71.0 + 3 * wx, ys * 2.13 + 2 * adv + 3 * wy)
    n = 0.78 * n1 + 0.42 * n2
    row = np.clip(sample(volw, X[0] * sc.s * 0.3 + 19.0, np.full(sc.W, 7.0, np.float32)), -1.6, 1.6)
    hx = sc.flame_h * hscale * (1 + sc.hvar * row)[None, :]
    base = 1 - (sc.H - Y) / hx
    v = 0.9 * base + 0.36 * n * (0.55 + 0.45 * np.clip(base, 0, 1))
    fire = np.clip(v, 0, 1)
    fire = fire * fire * (3 - 2 * fire)                       # smoothstep: crisp tongue edges
    return fire, hx


def draw_segment(buf, p0, p1, r, col, power=2.0):
    H, W = buf.shape[:2]
    m = 3 * r + 1
    x0, x1 = int(max(0, min(p0[0], p1[0]) - m)), int(min(W, max(p0[0], p1[0]) + m + 1))
    y0, y1 = int(max(0, min(p0[1], p1[1]) - m)), int(min(H, max(p0[1], p1[1]) + m + 1))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.array(p1) - np.array(p0)
    L2 = float(d @ d) + 1e-6
    u = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L2, 0, 1)
    dist = np.hypot(xx - (p0[0] + u * d[0]), yy - (p0[1] + u * d[1]))
    buf[y0:y1, x0:x1] += np.exp(-(dist / r) ** power)[..., None] * col


def ember_pos(e, i, a):
    x = e["x0"][i] + e["drift"][i] * a + e["sway"][i] * np.sin(2 * np.pi * (a * e["cyc"][i] + e["ph"][i]))
    y = e["start"][i] - a * e["rise"][i]
    return x, y


def render_frame(t):
    sc, X, Y = G["sc"], G["X"], G["Y"]
    W, H = sc.W, sc.H
    ph = t / sc.T
    Ny = G["Ny"]

    fire, hx = fire_field(t, 0.0, 1.0)
    temp = fire ** 1.3 * sc.gain
    em = blackbody(temp)
    yb = H - Y

    # ---- background: dark air lit by the fire, with drifting smoke
    lum = em.mean(-1, keepdims=True)
    glow = blur_big(np.repeat(lum, 3, -1), 55)[..., :1]
    bg = np.empty((H, W, 3), np.float32)
    k = (Y / H)[..., None]
    bg[:] = (1 - k) * np.array([0.004, 0.003, 0.003]) + k * np.array([0.05, 0.011, 0.005])
    bg += glow * np.array([0.42, 0.13, 0.04], np.float32)
    vw = G["volw"][t]
    smoke = sample(vw, X * sc.s * 0.6 + 53.0, Y * sc.s + ph * Ny * sc.k)
    smask = np.clip((yb - hx * 0.55) / (hx * 0.5), 0, 1)
    sd = (np.clip(0.5 + 0.35 * smoke, 0, 1) ** 2 * smask)[..., None]
    scol = np.array([0.03, 0.022, 0.02], np.float32) + glow * np.array([0.3, 0.12, 0.05], np.float32)
    bg = bg * (1 - 0.6 * sd) + scol * 0.6 * sd

    comp = bg + em

    # ---- forged-metal title
    if sc.title:
        M = G["M"][..., None]
        band = (Y > H - sc.flame_h * 1.1)
        colmean = (temp * band).sum(0) / band.sum(0).clip(1)
        light = np.clip(ndimage.gaussian_filter1d(colmean, 45) / 0.3, 0.45, 1.6)[None, :, None]
        tex = G["tex"][..., None]
        metal = np.array([0.016, 0.013, 0.012], np.float32) * tex
        refl = (G["down"] + 0.22 * G["side"])[..., None] * light * np.array([1.55, 0.5, 0.11], np.float32) * sc.shine
        cool = G["up"][..., None] * np.array([0.035, 0.04, 0.05], np.float32)
        heat = G["heat"][..., None] * light * np.array([1.5, 0.28, 0.03], np.float32) * 0.75 * (0.75 + 0.25 * tex)
        col = metal + refl * (0.8 + 0.2 * tex) + cool + heat
        comp = comp + G["halo"][..., None] * light * np.array([0.5, 0.12, 0.02], np.float32) * 0.45
        comp = comp * (1 - M) + col * M

    # ---- heat haze over everything behind the front flames
    if sc.haze > 0:
        vol = G["vol"][t]
        xs, ys = X * sc.s * 2.6 + 13.0, Y * sc.s * 2.6 + 2 * ph * Ny * sc.k
        dx, dy = sample(vol, xs, ys), sample(vol, xs + 40.0, ys + 21.0)
        hm = np.clip(1 - np.abs(yb - hx) / (0.65 * hx), 0, 1) * sc.haze
        if sc.title:
            hm = hm * (1 - 0.75 * G["halo_n"])
        cy_, cx_ = Y + dy * hm, X + dx * hm
        comp = np.stack([ndimage.map_coordinates(comp[..., c], [cy_, cx_], order=1, mode="nearest")
                         for c in range(3)], -1)

    # ---- flame tongues in front of the letters
    if sc.title and sc.front > 0:
        f2, _ = fire_field(t, 211.0, 0.92)
        comp = comp + blackbody(f2 ** 1.15) * (G["front_mask"][..., None] * sc.front)

    # ---- embers with motion blur, cooling as they rise
    e = G["emb"]
    sp = np.zeros((H, W, 3), np.float32)
    dt = 0.55 / sc.T
    for i in range(len(e["x0"])):
        a = (ph * e["m"][i] + e["p0"][i]) % 1.0
        b = np.sin(np.pi * a) ** 0.7 * (0.6 + 0.4 * np.sin(2 * np.pi * (ph * e["fl"][i] + e["fph"][i])) ** 2)
        p1 = ember_pos(e, i, a)
        p0 = ember_pos(e, i, max(0.0, a - dt * e["m"][i]))
        col = blackbody(np.array(0.97 - 0.45 * a, np.float32)) * b * 1.7 * e["br"][i]
        draw_segment(sp, p0, p1, e["r"][i], col)
    bk = G["bok"]
    for i in range(len(bk["x0"])):
        a = (ph + bk["p0"][i]) % 1.0
        x = bk["x0"][i] + bk["drift"][i] * a
        y = H + 20 - a * (H + 40)
        col = np.array([0.9, 0.35, 0.08], np.float32) * np.sin(np.pi * a) * 0.35
        draw_segment(sp, (x, y), (x, y), bk["r"][i], col, power=6.0)
    comp = comp + sp

    # ---- bloom, tone mapping
    bright = np.clip(comp - 1.2, 0, None)
    comp = comp + 0.28 * blur_big(bright, 4) + 0.18 * blur_big(bright, 14) + 0.14 * blur_big(bright, 42)
    disp = to_srgb(aces(comp * 0.86)).astype(np.float32)

    # ---- frosted glass (display space: it blurs exactly what is behind it)
    if sc.glass:
        g = G["glass"][..., None]
        frosted = ndimage.gaussian_filter(disp, (11, 11, 0)) * 0.58 + np.array([0.035, 0.016, 0.012], np.float32)
        frosted += G["glass_sheen"][..., None] * 0.045
        disp = disp * (1 - g) + frosted * g
        disp += G["glass_edge"][..., None] * (0.12 + 0.18 * G["glass_sheen"][..., None]) * np.array([1, .85, .75])
    for m, c in G["lines"]:
        disp = disp * (1 - m[..., None]) + c * m[..., None]

    disp = np.clip(disp * G["vignette"] + G["dither"], 0, 1)
    rgba = np.concatenate([disp, G["frame"][..., None]], -1)
    return (rgba * 255 + 0.5).astype(np.uint8)


def _work(args):
    t, folder = args
    Image.fromarray(render_frame(t), "RGBA").save(os.path.join(folder, f"{t:04d}.png"), compress_level=1)
    return t


def render(sc, out_dir, font, still=False, workers=10):
    prepare(sc, font)
    if still:
        for t in (0, sc.T // 2):
            Image.fromarray(render_frame(t), "RGBA").save(os.path.join(out_dir, f"{sc.name}-{t}.png"))
        print(f"{sc.name}: stills written")
        return
    tmp = tempfile.mkdtemp(prefix=f"fire-{sc.name}-")
    try:
        with Pool(workers, initializer=prepare, initargs=(sc, font)) as pool:
            list(pool.imap_unordered(_work, [(t, tmp) for t in range(sc.T)]))
        out = os.path.join(out_dir, f"{sc.name}.webp")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(sc.fps), "-i", os.path.join(tmp, "%04d.png"),
                        "-c:v", "libwebp_anim", "-pix_fmt", "yuva420p", "-q:v", str(sc.q),
                        "-compression_level", "6", "-loop", "0", out], check=True)
        shutil.copy(os.path.join(tmp, "0000.png"), os.path.join(out_dir, f"{sc.name}-0.png"))
        shutil.copy(os.path.join(tmp, f"{sc.T - 1:04d}.png"), os.path.join(out_dir, f"{sc.name}-last.png"))
        print(f"{sc.name}: {os.path.getsize(out) / 1024:.0f} KB, {sc.T} frames @ {sc.fps} fps")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--font", required=True, help="Cinzel variable TTF (Google Fonts)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--still", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sc in scenes():
        if not a.only or sc.name in a.only:
            render(sc, a.out, a.font, still=a.still)
