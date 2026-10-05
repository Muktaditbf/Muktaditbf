# 🔥 Procedural fire renderer

Every banner on my profile is rendered by [`render_fire.py`](render_fire.py): pure Python, NumPy and SciPy, with no stock footage and no game engine. Each banner is rendered frame by frame and encoded as a seamlessly looping animated WebP.

## How it works

| Stage | Technique |
|---|---|
| **Flames** | A 3D noise volume (x, y, time) built by FFT-filtering white noise with a power-law spectrum. It's periodic on every axis, so the loop has no seam. The noise is advected upward, domain-warped and shaped by a height falloff to form tongues of flame. |
| **Colour** | A blackbody temperature ramp in HDR (deep red → orange → gold), tone-mapped with a filmic ACES curve |
| **Air** | Fire-lit smoke, heat-haze refraction, and motion-blurred embers that cool from yellow to red as they rise, plus out-of-focus bokeh sparks |
| **Optics** | Multi-radius bloom, a vignette and dithering |
| **Lettering** | Cinzel capitals as forged metal: an emboss from the glyph's height field, bevels lit by the fire below and red-hot at the base. Flame tongues pass in front of the letters. |
| **Glass** | Frosted glass that really blurs the rendered frame behind it, with a specular edge |

The loop is checked numerically: the difference between the last frame and the first frame matches the difference between any two neighbouring frames, so there's no visible jump when it restarts.

## Run it

```bash
pip install numpy scipy pillow
# Cinzel variable font: https://fonts.google.com/specimen/Cinzel
python render_fire.py out/ --font Cinzel-VariableFont_wght.ttf            # all banners
python render_fire.py out/ --font Cinzel-VariableFont_wght.ttf --only header --still   # quick stills
```

You also need `ffmpeg` built with `libwebp`.
