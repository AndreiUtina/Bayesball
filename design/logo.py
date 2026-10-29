"""Draws the Bayesball crest logo as SVG.

    uv run python design/logo.py

writes design/logo-crest.svg (the full crest, with Bayes' theorem around the ring and the name
along the bottom) and app/static/logo.svg (the compact crest used in the site header and as the
browser-tab icon). The PNGs (app/static/logo-crest.png for the README and
app/static/apple-touch-icon.png) are screenshots of these, rendered with the real fonts:
STIX Two Text Italic and Barlow Condensed Bold, both from Google Fonts.
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAVY = "#13171F"
MATH = "'STIX Two Text', 'Times New Roman', serif"
CONDENSED = "'Barlow Condensed', 'Arial Narrow', sans-serif"
CX, CY = 180, 150  # centre of the crest in the full 360×300 drawing

GRADIENTS = """<radialGradient id="ball" cx="0.36" cy="0.3" r="0.75">
    <stop offset="0" stop-color="#FFE3C4"/>
    <stop offset="0.22" stop-color="#FDA05A"/>
    <stop offset="0.62" stop-color="#EA580C"/>
    <stop offset="1" stop-color="#9F1D14"/>
  </radialGradient>
  <linearGradient id="heat" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#FB923C"/>
    <stop offset="1" stop-color="#DC2626"/>
  </linearGradient>"""


def pentagon(x: float, y: float, radius: float, rotation: float) -> str:
    return " ".join(
        f"{x + radius * math.cos(math.radians(rotation + 72 * k)):.1f},"
        f"{y + radius * math.sin(math.radians(rotation + 72 * k)):.1f}"
        for k in range(5)
    )


def football(cx: float, cy: float, r: float) -> str:
    """An orange football: a centre pentagon, five cut-off ones at the edge, and seams."""
    patches, seams = [pentagon(cx, cy, r * 0.33, -90)], []
    for k in range(5):
        a = math.radians(-54 + 72 * k)  # between two corners of the centre pentagon
        x, y = cx + r * 0.98 * math.cos(a), cy + r * 0.98 * math.sin(a)
        patches.append(pentagon(x, y, r * 0.36, math.degrees(a) + 144))
        c, s = math.cos(math.radians(-90 + 72 * k)), math.sin(math.radians(-90 + 72 * k))
        x1, y1, x2, y2 = cx + r * 0.33 * c, cy + r * 0.33 * s, cx + r * 0.66 * c, cy + r * 0.66 * s
        seams.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    shine_x, shine_y = cx - r * 0.34, cy - r * 0.42
    polygons = "".join(f'<polygon points="{p}"/>' for p in patches)
    return f"""<clipPath id="ball-edge"><circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#ball)"/>
  <g clip-path="url(#ball-edge)" fill="#7A1A12" opacity="0.88">{polygons}</g>
  <g clip-path="url(#ball-edge)" stroke="#7A1A12" stroke-width="{r / 28:.1f}" opacity="0.7">
    {"".join(seams)}
  </g>
  <ellipse cx="{shine_x:.1f}" cy="{shine_y:.1f}" rx="{r * 0.26:.1f}" ry="{r * 0.14:.1f}"
    fill="#fff" opacity="0.45" transform="rotate(-32 {shine_x:.1f} {shine_y:.1f})"/>"""


def rings() -> str:
    return f"""<circle cx="{CX}" cy="{CY}" r="134" fill="none"
    stroke="url(#heat)" stroke-width="6"/>
  <circle cx="{CX}" cy="{CY}" r="94" fill="none"
    stroke="#EA580C" stroke-opacity="0.55" stroke-width="2"/>"""


def full_crest() -> str:
    """The crest with Bayes' theorem around the top and the name along the bottom, on navy."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 300" role="img"
  aria-label="Bayesball: a football in a crest, with Bayes' theorem around the ring">
  <defs>{GRADIENTS}
    <path id="top-arc" d="M {CX - 108},{CY} A 108,108 0 0 1 {CX + 108},{CY}"/>
    <path id="bottom-arc" d="M {CX - 121},{CY} A 121,121 0 0 0 {CX + 121},{CY}"/></defs>
  <rect width="360" height="300" rx="28" fill="{NAVY}"/>
  {rings()}
  <text font-family="{MATH}" font-style="italic" font-size="19" fill="#FDBA74"
    letter-spacing="0.3"><textPath href="#top-arc" startOffset="50%"
    text-anchor="middle">P(A|B) = P(B|A) · P(A) / P(B)</textPath></text>
  <text font-family="{CONDENSED}" font-weight="700" font-size="21" fill="#F3EDE7"
    letter-spacing="5"><textPath href="#bottom-arc" startOffset="50%"
    text-anchor="middle">· BAYESBALL ·</textPath></text>
  {football(CX, CY, 62)}
</svg>
"""


def compact_crest() -> str:
    """The ball in its rings, no text and no background: for small sizes on any page colour."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bayesball"
  viewBox="{CX - 138} {CY - 138} 276 276">
  <defs>{GRADIENTS}</defs>
  {rings()}
  {football(CX, CY, 82)}
</svg>
"""


if __name__ == "__main__":
    (ROOT / "design" / "logo-crest.svg").write_text(full_crest())
    (ROOT / "app" / "static" / "logo.svg").write_text(compact_crest())
    print("wrote design/logo-crest.svg and app/static/logo.svg")
