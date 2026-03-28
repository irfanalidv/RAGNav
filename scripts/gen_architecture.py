"""
RAGNav architecture diagram — clean monochromatic version.
Monochrome: white background, near-black text, grey tones for boxes.

Dependency: Pillow. Run from repo root:
  python3 scripts/gen_architecture.py

Writes: assets/ragnav-architecture.png
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "assets" / "ragnav-architecture.png"

# ── Canvas (height fits content; avoids empty space below dashed boxes) ─────
W, H = 3200, 1080

# ── Palette (pure monochrome) ────────────────────────────────────────────────
BLACK = "#0D0D0D"
DARK = "#1A1A1A"  # step-number circles, border
MID = "#3A3A3A"  # main box fill (dark text on top)
LITE = "#F0F0F0"  # light box fill (dark text inside)
BORDER = "#1A1A1A"
RULE = "#BBBBBB"  # horizontal rule / divider lines
DASH = "#999999"  # dashed boxes
WHITE = "#FFFFFF"
GREY50 = "#7A7A7A"  # sub-labels


def _font_candidates(bold: bool) -> list[str]:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/{name}",
        f"/usr/share/fonts/dejavu/{name}",
    ]
    if platform.system() == "Darwin":
        candidates.extend(
            [
                f"/Library/Fonts/{name}",
                f"/opt/homebrew/share/fonts/dejavu/{name}",
                *(
                    (
                        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                        "/Library/Fonts/Arial Bold.ttf",
                    )
                    if bold
                    else (
                        "/System/Library/Fonts/Supplemental/Arial.ttf",
                        "/Library/Fonts/Arial.ttf",
                    )
                ),
            ]
        )
    return candidates


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = _font_candidates(bold)
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


F_TITLE = font(72, bold=True)
F_SUB = font(36)
F_STEP = font(28, bold=True)
F_HEAD = font(30, bold=True)
F_BODY = font(24)
F_SMALL = font(20)
F_STEP_MARK = font(22, bold=True)


def text_center(draw, text, cx, cy, fnt, color=BLACK, anchor="mm"):
    draw.text((cx, cy), text, font=fnt, fill=color, anchor=anchor)


def rect(draw, x1, y1, x2, y2, fill, outline=BORDER, radius=14, width=2):
    draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width
    )


def circle(draw, cx, cy, r, fill=DARK, outline=DARK):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline)


def arrow(draw, x1, y1, x2, y2, color=DARK, width=3):
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    hs = 14
    draw.polygon([(x2, y2), (x2 - hs, y2 - hs // 2), (x2 - hs, y2 + hs // 2)], fill=color)


def dashed_rect(draw, x1, y1, x2, y2, color=DASH, width=2, dash=14, gap=8):
    for x in range(x1, x2, dash + gap):
        x_end = min(x + dash, x2)
        draw.line([(x, y1), (x_end, y1)], fill=color, width=width)
        draw.line([(x, y2), (x_end, y2)], fill=color, width=width)
    for y in range(y1, y2, dash + gap):
        y_end = min(y + dash, y2)
        draw.line([(x1, y), (x1, y_end)], fill=color, width=width)
        draw.line([(x2, y), (x2, y_end)], fill=color, width=width)


def divider(draw, x1, x2, y, color=RULE):
    draw.line([(x1, y), (x2, y)], fill=color, width=1)


def main() -> None:
    img = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(img)

    # ── Title ────────────────────────────────────────────────────────────────
    text_center(d, "RAGNav", W // 2, 70, F_TITLE, BLACK)
    text_center(d, "Hybrid Structure-Aware Retrieval", W // 2, 140, font(48, bold=True), DARK)
    text_center(
        d,
        "ingest  →  index  →  retrieve  →  rank  →  answer",
        W // 2,
        195,
        F_SUB,
        GREY50,
    )

    d.line([(120, 225), (W - 120, 225)], fill=RULE, width=1)

    # ── Layout constants ────────────────────────────────────────────────────
    TOP = 260
    BOT = 860
    MH = BOT - TOP

    cols = [
        (120, 280),
        (420, 270),
        (710, 320),
        (1050, 270),
        (1340, 310),
        (1670, 280),
        (1970, 280),
        (2270, 380),
    ]

    def col_box(i):
        x1, w = cols[i]
        return x1, TOP, x1 + w, BOT

    def col_cx(i):
        x1, w = cols[i]
        return x1 + w // 2

    # "R" = optional rerank between ⑤ and ⑥ (avoids reading ↺ as a second "five")
    step_labels = ["①", "②", "③", "④", "⑤", "R", "⑥", "⑦"]
    step_shown = [True, True, True, True, True, True, True, True]

    NUM_R = 20
    step_circle_y = {
        4: TOP - 50,
        5: TOP - 26,
        6: TOP - 50,
    }
    for i, (label, show) in enumerate(zip(step_labels, step_shown)):
        if not show:
            continue
        cx = col_cx(i)
        cy = step_circle_y.get(i, TOP - 38)
        circle(d, cx, cy, NUM_R, fill=DARK)
        fmark = F_STEP_MARK if label == "R" else F_SMALL
        text_center(d, label, cx, cy, fmark, WHITE)

    # ── Box 0: Input ────────────────────────────────────────────────────────
    x1, y1, x2, y2 = col_box(0)
    rect(d, x1, y1, x2, y2, LITE, BORDER)
    cx = col_cx(0)
    text_center(d, "INPUT", cx, y1 + 36, F_HEAD, DARK)
    divider(d, x1 + 16, x2 - 16, y1 + 60)
    inputs = [
        "PDF URL / file",
        "Markdown",
        "HTML",
        "Email thread",
        "Legal document",
        "Chat log",
        "",
        "User question",
    ]
    ty = y1 + 82
    for ln in inputs:
        if ln == "":
            divider(d, x1 + 16, x2 - 16, ty + 4)
            ty += 20
            continue
        text_center(d, ln, cx, ty, F_BODY, DARK)
        ty += 34

    # ── Box 1: Download + Cache ─────────────────────────────────────────────
    x1, y1, x2, y2 = col_box(1)
    rect(d, x1, y1, x2, y2, LITE, BORDER)
    cx = col_cx(1)
    text_center(d, "DOWNLOAD", cx, y1 + 36, F_HEAD, DARK)
    text_center(d, "+ CACHE", cx, y1 + 66, F_HEAD, DARK)
    divider(d, x1 + 16, x2 - 16, y1 + 90)
    items = [
        "download_pdf(url)",
        "→ bytes",
        "",
        "optional disk",
        "cache",
        "",
        "SqliteKV",
        "EmbeddingCache",
        "RetrievalCache",
    ]
    ty = y1 + 108
    for ln in items:
        if ln == "":
            ty += 14
            continue
        f = (
            F_SMALL
            if ln.startswith("Sqlite") or ln.endswith("Cache")
            else F_BODY
        )
        text_center(d, ln, cx, ty, f, DARK)
        ty += 32

    # ── Box 2: Ingest ───────────────────────────────────────────────────────
    x1, y1, x2, y2 = col_box(2)
    rect(d, x1, y1, x2, y2, MID, BORDER)
    cx = col_cx(2)
    text_center(d, "INGEST", cx, y1 + 36, F_HEAD, WHITE)
    divider(d, x1 + 16, x2 - 16, y1 + 60, GREY50)

    formats = [
        "ingest_pdf  (paper mode)",
        "ingest_markdown",
        "ingest_html",
        "ingest_email",
        "ingest_legal",
        "ingest_chat",
    ]
    ty = y1 + 80
    for ln in formats:
        text_center(d, ln, cx, ty, F_BODY, WHITE)
        ty += 34

    divider(d, x1 + 16, x2 - 16, ty + 8, GREY50)
    ty += 28

    for ln in [
        "Blocks  (page + anchors)",
        "Edges: parent / next / link_to",
        "Provenance: block_id",
    ]:
        text_center(d, ln, cx, ty, F_SMALL, "#C8C8C8")
        ty += 30

    # ── Box 3: Index ────────────────────────────────────────────────────────
    x1, y1, x2, y2 = col_box(3)
    rect(d, x1, y1, x2, y2, MID, BORDER)
    cx = col_cx(3)
    text_center(d, "INDEX", cx, y1 + 36, F_HEAD, WHITE)
    divider(d, x1 + 16, x2 - 16, y1 + 60, GREY50)

    items3 = [
        ("BM25", "rank-bm25"),
        ("", ""),
        ("Vector index", "all-MiniLM-L6-v2"),
        ("", "(optional)"),
    ]
    ty = y1 + 84
    for head, sub in items3:
        if head:
            text_center(d, head, cx, ty, F_STEP, WHITE)
            ty += 32
        if sub:
            text_center(d, sub, cx, ty, F_SMALL, "#C8C8C8")
            ty += 34
        else:
            ty += 20

    # ── Box 4: Retrieve ───────────────────────────────────────────────────
    x1, y1, x2, y2 = col_box(4)
    rect(d, x1, y1, x2, y2, MID, BORDER)
    cx = col_cx(4)
    text_center(d, "RETRIEVE", cx, y1 + 36, F_HEAD, WHITE)
    divider(d, x1 + 16, x2 - 16, y1 + 60, GREY50)

    steps5 = [
        "Route pages",
        "Search within",
        "routed pages",
        "",
        "Expand coherence",
        "(parent + next)",
        "",
        "fusion: RRF (default)",
        "| weighted",
        "",
        "bm25_weight",
        "vector_weight",
    ]
    ty = y1 + 82
    for ln in steps5:
        if ln == "":
            ty += 14
            continue
        text_center(d, ln, cx, ty, F_BODY, WHITE)
        ty += 32

    # ── Box 5: Rerank (optional) ───────────────────────────────────────────
    x1, y1, x2, y2 = col_box(5)
    rect(d, x1, y1 + 60, x2, y2 - 60, LITE, BORDER, radius=12)
    cx = col_cx(5)
    text_center(d, "RERANK", cx, y1 + 100, F_HEAD, DARK)
    text_center(d, "(optional)", cx, y1 + 132, F_SMALL, GREY50)
    divider(d, x1 + 16, x2 - 16, y1 + 154)

    ry = y1
    for ln, fnt in [
        ("CrossEncoder", F_BODY),
        ("Reranker", F_BODY),
        ("", None),
        ("ms-marco-MiniLM", F_SMALL),
        ("L-6-v2", F_SMALL),
        ("", None),
        ("≥ 50 candidates", F_SMALL),
    ]:
        if fnt is None:
            ry += 14
            continue
        ry += 36
        text_center(d, ln, cx, ry + 140, fnt, DARK)

    # ── Box 6: Graph Expand ─────────────────────────────────────────────────
    x1, y1, x2, y2 = col_box(6)
    rect(d, x1, y1 + 60, x2, y2 - 60, LITE, BORDER, radius=12)
    cx = col_cx(6)
    text_center(d, "GRAPH", cx, y1 + 100, F_HEAD, DARK)
    text_center(d, "EXPAND", cx, y1 + 132, F_HEAD, DARK)
    text_center(d, "(optional)", cx, y1 + 162, F_SMALL, GREY50)
    divider(d, x1 + 16, x2 - 16, y1 + 186)

    gy = y1
    for ln in [
        "Follow refs:",
        "Figure / Table",
        "Appendix / Section",
        "",
        "link_to edges",
        "",
        "multi-hop",
        "traversal",
    ]:
        if ln == "":
            gy += 14
            continue
        gy += 34
        text_center(d, ln, cx, gy + 195, F_BODY, DARK)

    # ── Box 7: Output ───────────────────────────────────────────────────────
    x1, y1, x2, y2 = col_box(7)
    rect(d, x1, y1, x2, y2, MID, BORDER)
    cx = col_cx(7)
    text_center(d, "OUTPUT", cx, y1 + 36, F_HEAD, WHITE)
    divider(d, x1 + 16, x2 - 16, y1 + 60, GREY50)

    out_items = [
        ("Evidence blocks", F_STEP, WHITE),
        ("block_id provenance", F_SMALL, "#C8C8C8"),
        ("", None, None),
        ("Answer", F_STEP, WHITE),
        ("", None, None),
        ("Citations", F_BODY, WHITE),
        ("[[block_id]] per sentence", F_SMALL, "#C8C8C8"),
        ("", None, None),
        ("ConfidenceLevel", F_BODY, WHITE),
        ("HIGH / MEDIUM / LOW", F_SMALL, "#C8C8C8"),
        ("", None, None),
        ("CostReport", F_BODY, WHITE),
        ("(optional)", F_SMALL, "#C8C8C8"),
    ]
    ty = y1 + 82
    for ln, fnt, color in out_items:
        if fnt is None:
            ty += 18
            continue
        text_center(d, ln, cx, ty, fnt, color)
        ty += 30

    # ── Arrows between boxes ────────────────────────────────────────────────
    for i in range(len(cols) - 1):
        x1_end = cols[i][0] + cols[i][1]
        x2_start = cols[i + 1][0]
        mid_y = TOP + MH // 2
        arrow(d, x1_end + 6, mid_y, x2_start - 6, mid_y, DARK, 3)

    # ── Bottom section (tight dashed frames) ─────────────────────────────
    BOTY = BOT + 36
    DASH_BOT = BOTY + 92
    foot_y = DASH_BOT + 36

    gx1, gx2 = 120, 1900
    dashed_rect(d, gx1, BOTY, gx2, DASH_BOT, DASH, 2)
    gcy = (BOTY + DASH_BOT) // 2
    text_center(d, "GraphRAG  (optional)", (gx1 + gx2) // 2, gcy - 22, F_HEAD, DARK)
    text_center(
        d,
        "extract entities / relations  →  build entity graph  →  multi-hop retrieve  (provenance)",
        (gx1 + gx2) // 2,
        gcy + 14,
        F_BODY,
        GREY50,
    )

    qx1, qx2 = 1940, 3080
    dashed_rect(d, qx1, BOTY, qx2, DASH_BOT, DASH, 2)
    qcy = (BOTY + DASH_BOT) // 2
    text_center(d, "QueryFallback  (optional)", (qx1 + qx2) // 2, qcy - 36, F_HEAD, DARK)
    text_center(d, "LOW / MEDIUM confidence", (qx1 + qx2) // 2, qcy - 2, F_BODY, GREY50)
    text_center(d, "→  LLM query variations  →  best result", (qx1 + qx2) // 2, qcy + 28, F_BODY, GREY50)

    ret_x = cols[4][0] + cols[4][1] // 2
    qcx = (qx1 + qx2) // 2
    d.line([(qcx, BOTY), (qcx, BOT + 32)], fill=DASH, width=2)
    d.line([(ret_x, BOT + 32), (qcx, BOT + 32)], fill=DASH, width=2)
    arrow(d, ret_x, BOT + 32, ret_x, BOT + 6, DASH, 2)
    text_center(d, "retry", ret_x + 36, BOT + 24, F_SMALL, DASH)

    text_center(
        d,
        "Offline benchmarks: squad_benchmark  ·  cuad_benchmark  ·  scorecard  ·  paper_eval  ·  security_eval",
        W // 2,
        foot_y,
        F_SMALL,
        GREY50,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH, "PNG", dpi=(144, 144))
    print(f"saved → {OUT_PATH}  ({W}×{H})")


if __name__ == "__main__":
    main()
