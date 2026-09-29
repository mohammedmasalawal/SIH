"""Colour-distinguishability checks for the industrial sub-type palette.

    python -m training.colour_check

Same maths and thresholds as the dataviz skill's validate_palette (which is not vendored here):
OKLab ΔE ×100; colour-blind simulation with the Machado-Oliveira-Fernandes (2009) severity-1.0 matrices
(protan, deutan, tritan); a lightness band, a chroma floor, and WCAG contrast against the surface. The
surface is the dark basemap's land colour (CARTO dark-matter, #0e0e0e). Thresholds (dark mode):

    lightness band   OKLCH L 0.48-0.67
    chroma floor     OKLCH C >= 0.10   (the muted blue-grey is *meant* to fail this, like the grey
                                        "unclassified" class colour: it must read as "no type")
    CVD separation   min(protan, deutan) ΔE >= 8 target, 6-8 only with a secondary cue
    normal vision    ΔE >= 15 between any two colours that can touch
    contrast         >= 3:1 against the surface

Which pairs are checked matters: "adjacent" is panel order (a legend or stack); "all" is every pair (a
scatter or map, where any two points can touch). Six blue/cyan/violet hues plus a blue-grey cannot all be
pairwise separated inside the dark lightness band (README lists the pairs), so the palette is fitted to
pass every check on adjacent pairs and to keep the large groups apart on the map.
"""

from __future__ import annotations

import itertools
import math

from training.industrial_subtype import ALL_SUBTYPES, SUBTYPE_COLORS

SURFACE = "#0e0e0e"  # CARTO dark-matter land colour
BAND = (0.48, 0.67)
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR, CONTRAST_MIN = 8.0, 6.0, 15.0, 3.0
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def _linear(hex_colour: str) -> tuple[float, float, float]:
    channels = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)


def _oklab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(hex_colour: str) -> tuple[float, float, float]:
    """(lightness, chroma, hue in degrees)"""
    lightness, a, b = _oklab(_linear(hex_colour))
    return lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def simulate(hex_colour: str, kind: str | None) -> tuple[float, float, float]:
    rgb = _linear(hex_colour)
    if kind is None:
        return rgb
    return tuple(max(0.0, min(1.0, sum(w * c for w, c in zip(row, rgb)))) for row in MACHADO[kind])


def delta_e(a: str, b: str, kind: str | None = None) -> float:
    """OKLab distance ×100, under normal vision or a simulated colour-vision deficiency."""
    return 100 * math.dist(_oklab(simulate(a, kind)), _oklab(simulate(b, kind)))


def contrast(a: str, b: str) -> float:
    luminance = lambda h: (lambda r, g, bl: 0.2126 * r + 0.7152 * g + 0.0722 * bl)(*_linear(h))
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def pair_table(palette: list[str]) -> list[dict]:
    """Every pair with its ΔE under normal vision and protan / deutan / tritan simulation."""
    return [
        {"i": i, "j": j, "adjacent": j == i + 1, "normal": delta_e(palette[i], palette[j]),
         "protan": delta_e(palette[i], palette[j], "protan"), "deutan": delta_e(palette[i], palette[j], "deutan"),
         "tritan": delta_e(palette[i], palette[j], "tritan")}
        for i, j in itertools.combinations(range(len(palette)), 2)
    ]


def weak_pairs(palette: list[str]) -> list[dict]:
    """Pairs below the normal-vision floor (15) or the protan/deutan target (8)."""
    return [p for p in pair_table(palette) if p["normal"] < NORMAL_FLOOR or min(p["protan"], p["deutan"]) < CVD_TARGET]


def check(palette: list[str], surface: str = SURFACE, chromatic: int | None = None) -> dict:
    """Measurements for `palette` in legend order. `chromatic` = how many leading entries must meet the
    lightness/chroma rules (the trailing muted blue-grey is exempt from the chroma floor by design)."""
    chromatic = len(palette) if chromatic is None else chromatic
    lch = [oklch(c) for c in palette]
    pairs = pair_table(palette)
    adjacent = [p for p in pairs if p["adjacent"]]
    return {
        "off_band": [c for c, (l, _, _) in zip(palette, lch) if not BAND[0] <= l <= BAND[1]],
        "low_chroma": [c for c, (_, ch, _) in zip(palette[:chromatic], lch) if ch < CHROMA_FLOOR],
        "low_contrast": [c for c in palette if contrast(c, surface) < CONTRAST_MIN],
        "adjacent_worst_cvd": min(min(p["protan"], p["deutan"]) for p in adjacent),
        "adjacent_worst_normal": min(p["normal"] for p in adjacent),
        "all_worst_cvd": min(min(p["protan"], p["deutan"]) for p in pairs),
        "all_worst_normal": min(p["normal"] for p in pairs),
        "adjacent_tritan_worst": min(p["tritan"] for p in adjacent),
        "all_tritan_worst": min(p["tritan"] for p in pairs),
    }


def subtype_palette() -> list[str]:
    return [SUBTYPE_COLORS[name] for name in ALL_SUBTYPES]


if __name__ == "__main__":
    palette = subtype_palette()
    report = check(palette, chromatic=len(palette) - 1)
    for key, value in report.items():
        print(f"{key:>22}: {value if isinstance(value, list) else round(value, 1)}")
    print("\npairs under normal-vision 15 or protan/deutan 8:")
    for p in weak_pairs(palette):
        print(f"  {ALL_SUBTYPES[p['i']]:<32} - {ALL_SUBTYPES[p['j']]:<32} normal {p['normal']:5.1f}  "
              f"protan {p['protan']:5.1f}  deutan {p['deutan']:5.1f}  tritan {p['tritan']:5.1f}{'  (adjacent)' if p['adjacent'] else ''}")
