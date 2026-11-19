"""Tiny dependency-free SVG charting.

Just enough to render equity curves and drawdown fills straight to an ``.svg``
string -- no matplotlib, no binary assets.  GitHub renders SVG inline in
READMEs, so the charts stay crisp and version-controllable.
"""
from __future__ import annotations

import numpy as np

# dark "terminal" palette
BG = "#0d1117"
GRID = "#21262d"
AXIS = "#8b949e"
TEXT = "#c9d1d9"
PALETTE = ["#58a6ff", "#3fb950", "#f778ba", "#d29922", "#a371f7", "#ff7b72"]
LOSS = "#f85149"


def _scale(vals, lo, hi, a, b):
    if hi == lo:
        return np.full_like(np.asarray(vals, float), (a + b) / 2)
    return a + (np.asarray(vals, float) - lo) * (b - a) / (hi - lo)


def _poly(xs, ys):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def equity_chart(curves: dict[str, np.ndarray], title: str = "",
                 width: int = 960, height: int = 420) -> str:
    """``curves``: name -> equity array (all same length, start=1.0)."""
    pad_l, pad_r, pad_t, pad_b = 54, 212, 40, 34
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    all_vals = np.concatenate([np.asarray(c, float) for c in curves.values()])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    span = hi - lo or 1.0
    lo, hi = lo - span * 0.05, hi + span * 0.05
    n = len(next(iter(curves.values())))
    xs = _scale(np.arange(n), 0, n - 1, pad_l, pad_l + pw)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="SFMono-Regular,Consolas,monospace">']
    svg.append(f'<rect width="{width}" height="{height}" fill="{BG}" rx="10"/>')
    if title:
        svg.append(f'<text x="{pad_l}" y="24" fill="{TEXT}" font-size="15" '
                   f'font-weight="700">{title}</text>')
    # horizontal gridlines + y labels (as growth multiple)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + ph * frac
        val = hi - (hi - lo) * frac
        svg.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+pw}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" fill="{AXIS}" font-size="11" '
                   f'text-anchor="end">{val:.2f}x</text>')
    # the 1.0 (breakeven) line
    y0 = _scale([1.0], lo, hi, pad_t + ph, pad_t)[0]
    svg.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{pad_l+pw}" y2="{y0:.1f}" '
               f'stroke="{AXIS}" stroke-width="1" stroke-dasharray="4 3" opacity="0.5"/>')
    # series + legend
    for i, (name, curve) in enumerate(curves.items()):
        color = PALETTE[i % len(PALETTE)]
        ys = _scale(curve, lo, hi, pad_t + ph, pad_t)
        svg.append(f'<polyline points="{_poly(xs, ys)}" fill="none" '
                   f'stroke="{color}" stroke-width="2"/>')
        ly = pad_t + 6 + i * 22
        svg.append(f'<rect x="{pad_l+pw+16}" y="{ly}" width="12" height="12" '
                   f'rx="2" fill="{color}"/>')
        mult = float(curve[-1])
        label = name if len(name) <= 17 else name[:16] + "…"
        svg.append(f'<text x="{pad_l+pw+34}" y="{ly+11}" fill="{TEXT}" '
                   f'font-size="12">{label}  {mult:.2f}x</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def drawdown_chart(dd: np.ndarray, title: str = "", color: str = LOSS,
                   width: int = 820, height: int = 200) -> str:
    """``dd``: drawdown series (<= 0)."""
    pad_l, pad_r, pad_t, pad_b = 54, 24, 36, 28
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    lo = float(min(dd.min(), -1e-6))
    n = len(dd)
    xs = _scale(np.arange(n), 0, n - 1, pad_l, pad_l + pw)
    ys = _scale(dd, lo, 0.0, pad_t + ph, pad_t)
    y_zero = pad_t
    area = f"{xs[0]:.1f},{y_zero:.1f} " + _poly(xs, ys) + f" {xs[-1]:.1f},{y_zero:.1f}"

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="SFMono-Regular,Consolas,monospace">']
    svg.append(f'<rect width="{width}" height="{height}" fill="{BG}" rx="10"/>')
    if title:
        svg.append(f'<text x="{pad_l}" y="22" fill="{TEXT}" font-size="14" '
                   f'font-weight="700">{title}</text>')
    for frac in (0, 0.5, 1.0):
        y = pad_t + ph * frac
        val = 0.0 - (0.0 - lo) * frac
        svg.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+pw}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" fill="{AXIS}" font-size="11" '
                   f'text-anchor="end">{val*100:.0f}%</text>')
    svg.append(f'<polygon points="{area}" fill="{color}" opacity="0.35"/>')
    svg.append(f'<polyline points="{_poly(xs, ys)}" fill="none" stroke="{color}" '
               f'stroke-width="1.5"/>')
    svg.append("</svg>")
    return "\n".join(svg)
