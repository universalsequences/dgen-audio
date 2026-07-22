#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / sum(
        (x - x_bar) ** 2 for x in xs
    )
    return slope, y_bar - slope * x_bar


def points(values: list[tuple[float, float]], xmap, ymap) -> str:
    return " ".join(f"{xmap(x):.2f},{ymap(y):.2f}" for x, y in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()

    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    delta = [float(row["delta"]) for row in rows]
    loss = [float(row["loss"]) for row in rows]
    line_slope, line_intercept = fit_line(delta, loss)
    detrended = [(y - (line_slope * x + line_intercept)) * 1e6 for x, y in zip(delta, loss)]
    midpoint = [(a + b) * 0.5 for a, b in zip(delta[:-1], delta[1:])]
    local_slope = [(b - a) / (xb - xa) for a, b, xa, xb in zip(
        loss[:-1], loss[1:], delta[:-1], delta[1:]
    )]

    width, height = 1080, 720
    left, right = 92, 28
    top1, bottom1 = 76, 326
    top2, bottom2 = 410, 660
    plot_width = width - left - right
    xmin, xmax = min(delta), max(delta)

    def xmap(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_width

    def make_ymap(values: list[float], top: float, bottom: float):
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.08, 1e-9)
        lo, hi = lo - pad, hi + pad
        return lambda y: bottom - (y - lo) / (hi - lo) * (bottom - top), lo, hi

    y1, y1min, y1max = make_ymap(detrended, top1, bottom1)
    slope_values = local_slope + [4.461658]
    y2, y2min, y2max = make_ymap(slope_values, top2, bottom2)

    grid = []
    labels = []
    for i in range(6):
        x = left + plot_width * i / 5
        value = xmin + (xmax - xmin) * i / 5
        grid.append(f'<line x1="{x:.2f}" y1="{top1}" x2="{x:.2f}" y2="{bottom2}" class="grid"/>')
        labels.append(f'<text x="{x:.2f}" y="686" text-anchor="middle">{value:.1e}</text>')
    for top, bottom, lo, hi, mapper in [
        (top1, bottom1, y1min, y1max, y1),
        (top2, bottom2, y2min, y2max, y2),
    ]:
        for i in range(5):
            value = lo + (hi - lo) * i / 4
            y = mapper(value)
            grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>')
            labels.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end">{value:.2f}</text>')

    slope_path = []
    for index, (x, y) in enumerate(zip(midpoint, local_slope)):
        px, py = xmap(x), y2(y)
        if index == 0:
            slope_path.append(f"M {px:.2f} {py:.2f}")
        else:
            previous_y = y2(local_slope[index - 1])
            slope_path.append(f"L {px:.2f} {previous_y:.2f} L {px:.2f} {py:.2f}")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">Production log-magnitude L1 loss along shape</title>
<desc id="desc">A dense 201-point sweep across plus or minus five times ten to the minus four. The detrended loss and local secant slope show many deterministic slope breaks.</desc>
<style>
  text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; fill: #30343b; }}
  .title {{ font-size: 19px; font-weight: 500; }}
  .subtitle {{ font-size: 13px; fill: #606770; }}
  .axis-title {{ font-size: 14px; font-weight: 500; }}
  .grid {{ stroke: #c7ccd1; stroke-width: 1; opacity: .45; }}
  .axis {{ stroke: #697078; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="28" class="title">Production log-magnitude L1 loss along shape</text>
<text x="{left}" y="50" class="subtitle">201 points across ±5×10⁻⁴; local slope breaks remain visible at 5×10⁻⁶ spacing</text>
{''.join(grid)}
<line x1="{left}" y1="{bottom1}" x2="{width-right}" y2="{bottom1}" class="axis"/>
<line x1="{left}" y1="{bottom2}" x2="{width-right}" y2="{bottom2}" class="axis"/>
<polyline points="{points(list(zip(delta, detrended)), xmap, y1)}" fill="none" stroke="#31688e" stroke-width="1.6"/>
<path d="{' '.join(slope_path)}" fill="none" stroke="#b24a3b" stroke-width="1.3"/>
<line x1="{left}" y1="{y2(4.461658):.2f}" x2="{width-right}" y2="{y2(4.461658):.2f}" stroke="#30343b" stroke-width="1.2" stroke-dasharray="6 5"/>
<text x="{width-right-6}" y="{y2(4.461658)-7:.2f}" text-anchor="end">autograd at center = 4.461658</text>
<text x="22" y="{(top1+bottom1)/2}" transform="rotate(-90 22 {(top1+bottom1)/2})" text-anchor="middle" class="axis-title">Loss minus linear fit (×10⁻⁶)</text>
<text x="22" y="{(top2+bottom2)/2}" transform="rotate(-90 22 {(top2+bottom2)/2})" text-anchor="middle" class="axis-title">Local secant slope</text>
<text x="{(left+width-right)/2}" y="712" text-anchor="middle" class="axis-title">Transformed shape offset from 0.35</text>
{''.join(labels)}
</svg>
'''
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(svg)


if __name__ == "__main__":
    main()
