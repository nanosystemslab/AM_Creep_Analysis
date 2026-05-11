
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt

# Canonical output directory for thesis figures
THESIS_FIGURES_DIR = Path(__file__).resolve().parent.parent / "out" / "figures" / "ch3"

from _figure_style import apply_style, TOL_COLORS as _TOL_COLORS, MARKERS as _MARKERS

# Nanoindent folders
_NANO_STYLES = {
    '103':              (_TOL_COLORS[0],  'o'),   # blue, circle
    '103-t1500':        (_TOL_COLORS[1],  's'),   # orange, square
    '103-pc-con':       (_TOL_COLORS[2],  '^'),   # teal, triangle-up
    '103-t1500-con':    (_TOL_COLORS[3],  'D'),   # red, diamond
    '44-500':           (_TOL_COLORS[4],  'v'),   # cyan, triangle-down
    '44-1000':          (_TOL_COLORS[5],  'P'),   # magenta, plus-filled
    '44-2000':          (_TOL_COLORS[8],  'X'),   # mint, x-filled
    '44-3000':          (_TOL_COLORS[9],  'h'),   # wine, hexagon
    '44-5000':          (_TOL_COLORS[10], 'p'),   # indigo, pentagon
    '44-7000':          (_TOL_COLORS[11], '*'),    # light blue, star
    '44-10000':         (_TOL_COLORS[12], '<'),    # dark pink, left-tri
    '50fast':           (_TOL_COLORS[13], '>'),    # dark green, right-tri
    '50slow':           (_TOL_COLORS[14], '8'),    # sand, octagon
    '75':               (_TOL_COLORS[15], 'd'),    # olive, thin-diamond
}

# Bulk stress levels (MPa) — same palette, cycling markers
_BULK_STRESSES = [15, 17.5, 18, 20, 21, 22.5, 24, 25, 27, 30, 33, 36, 37.5, 39]
_BULK_STYLES = {}
for _i, _s in enumerate(_BULK_STRESSES):
    _color = _TOL_COLORS[_i % len(_TOL_COLORS)]
    _marker = _MARKERS[_i % len(_MARKERS)]
    # Store under several equivalent keys
    _BULK_STYLES[str(_s)] = (_color, _marker)
    # Integer form if it's a whole number
    if _s == int(_s):
        _BULK_STYLES[str(int(_s))] = (_color, _marker)
    # MPa suffixed variants
    _BULK_STYLES[f'{_s}mpa'] = (_color, _marker)
    if _s == int(_s):
        _BULK_STYLES[f'{int(_s)}mpa'] = (_color, _marker)

# Merge into one dict
FOLDER_STYLES = {}
FOLDER_STYLES.update(_NANO_STYLES)
FOLDER_STYLES.update(_BULK_STYLES)


def _normalize_folder_name(name: str) -> str:
    """Strip path components, trailing slashes, and lowercase."""
    # Take last path component
    name = name.rstrip('/').rstrip('\\')
    if '/' in name:
        name = name.rsplit('/', 1)[-1]
    if '\\' in name:
        name = name.rsplit('\\', 1)[-1]
    return name.lower().strip()


def get_folder_style(name: str, index: int | None = None) -> tuple[str, str]:
    """Return (color, marker) for a folder name, deterministic for unknowns.

    If *index* is given and the name is not in the predefined styles,
    use it to cycle through colors/markers (guarantees unique styles
    for each folder in a plot).
    """
    key = _normalize_folder_name(name)
    if key in FOLDER_STYLES:
        return FOLDER_STYLES[key]
    if index is not None:
        color = _TOL_COLORS[index % len(_TOL_COLORS)]
        marker = _MARKERS[index % len(_MARKERS)]
        return (color, marker)
    # Hash-based fallback — same name always gets the same style
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    color = _TOL_COLORS[h % len(_TOL_COLORS)]
    marker = _MARKERS[h % len(_MARKERS)]
    return (color, marker)


def get_style_by_index(index: int) -> tuple[str, str]:
    """Return (color, marker) for a flat index, cycling through all styles."""
    color = _TOL_COLORS[index % len(_TOL_COLORS)]
    marker = _MARKERS[index % len(_MARKERS)]
    return (color, marker)


def get_stress_style(stress_value) -> tuple[str, str]:
    """Return (color, marker) for a numeric stress value (int or float)."""
    stress = float(stress_value)
    # Try exact key first
    key = str(stress) if stress != int(stress) else str(int(stress))
    if key in FOLDER_STYLES:
        return FOLDER_STYLES[key]
    # Try float form
    if str(stress) in FOLDER_STYLES:
        return FOLDER_STYLES[str(stress)]
    # Fallback via get_folder_style hash
    return get_folder_style(key)


# ─────────────────────────────────────────────────────────────────────────
# Publication method names  (internal ID → figure label)
# ─────────────────────────────────────────────────────────────────────────
METHOD_DISPLAY_NAMES = {
    "19":  "Method 1",
    "41":  "Method 2",
    "44":  "Method 3",
    "50":     "Method 4",
    "50slow": "Method 4 (slow)",
    "50fast": "Method 4 (fast)",
    "75":  "Method 5",
    "103": "Method 6",
}


def get_method_label(method_id: str) -> str:
    """Return publication-friendly method name, e.g. '103' → 'Method 5'."""
    return METHOD_DISPLAY_NAMES.get(str(method_id), f"Method {method_id}")


def get_display_label(folder_name: str) -> str:
    """Convert folder name to publication label.

    Examples:
        '19-8000'  → 'Method 1 (8000 µN)'
        '103'      → 'Method 5'
        '44-500'   → 'Method 3 (500 µN)'
        'unknown'  → 'unknown'  (no match)
    """
    name = folder_name.strip().rstrip('/')
    # Try splitting on '-' to get method + load
    if '-' in name:
        parts = name.split('-', 1)
        method_id = parts[0]
        suffix = parts[1]
        if method_id in METHOD_DISPLAY_NAMES:
            return f"{METHOD_DISPLAY_NAMES[method_id]} ({suffix} µN)"
    # Try splitting on '_' to get method + suffix
    if '_' in name:
        parts = name.split('_', 1)
        method_id = parts[0]
        suffix = parts[1]
        if method_id in METHOD_DISPLAY_NAMES:
            return f"{METHOD_DISPLAY_NAMES[method_id]} ({suffix} µN)"
    # No dash — try bare method ID
    if name in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[name]
    return folder_name


def apply_journal_style():
    apply_style()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{bm}"
        r"\renewcommand{\seriesdefault}{b}"   # default text series = bold
        r"\boldmath"                           # bold math (tick labels)
    )
