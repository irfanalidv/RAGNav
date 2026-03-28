#!/usr/bin/env python3
"""
Regenerate assets/ragnav-architecture.png — RAGNav hybrid retrieval pipeline.
Run from repo root: python scripts/generate_architecture_diagram.py
Requires: matplotlib
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- Style (match legacy diagram) ---
BG = "#c5dce8"
GRID = "#ffffff"
NAVY = "#1a3352"
NAVY_TEXT = "#ffffff"
ORANGE = "#d9781c"
RERANK_BLUE = "#4a8fb8"
RERANK_BLUE_LIGHT = "#6ba3c7"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "assets" / "ragnav-architecture.png"

W, H = 16, 9  # inches
DPI = 200  # 3200×1800 px


def grid_bg(ax):
    ax.set_facecolor(BG)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    for g in range(0, int(W * 10) + 1):
        x = g / 10
        ax.axvline(x, color=GRID, linewidth=0.35, alpha=0.85, zorder=0)
    for g in range(0, int(H * 10) + 1):
        y = g / 10
        ax.axhline(y, color=GRID, linewidth=0.35, alpha=0.85, zorder=0)


def rounded_box(ax, xy, w, h, facecolor, edgecolor="none", linewidth=0, zorder=2):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


def dashed_box(ax, xy, w, h, zorder=2):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#d0e8f0",
        edgecolor=NAVY,
        linewidth=1.8,
        linestyle="--",
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


def arrow(ax, xy1, xy2, color=NAVY, lw=1.4):
    a = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linewidth=lw,
        zorder=3,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


def circled_num(ax, cx, cy, n: str, r=0.22):
    circ = plt.Circle((cx, cy), r, facecolor=NAVY, edgecolor="white", linewidth=1.2, zorder=5)
    ax.add_patch(circ)
    ax.text(
        cx,
        cy,
        n,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        zorder=6,
        fontfamily="sans-serif",
    )


def main():
    fig, ax = plt.subplots(1, 1, figsize=(W, H), dpi=DPI)
    grid_bg(ax)

    # Title
    ax.text(
        W / 2,
        H - 0.55,
        "RAGNav: Hybrid Structure-Aware Retrieval",
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        fontfamily="sans-serif",
    )
    ax.text(
        W / 2,
        H - 1.05,
        "ingest → index → retrieve → rank → answer",
        ha="center",
        va="top",
        fontsize=12,
        color=NAVY,
        fontfamily="sans-serif",
    )

    y_main = 2.35
    h_small = 2.85
    h_ingest = 4.35
    h_out = 4.1

    # Column centers (x) for step labels above boxes
    cols = {
        1: (0.85, 1.85),
        2: (2.15, 3.35),
        3: (3.55, 5.95),
        4: (6.35, 7.35),
        5: (7.65, 9.05),
        "r": (9.2, 9.95),
        6: (10.15, 11.25),
        7: (11.55, 15.2),
    }

    # ① Input
    x1, x1b = cols[1]
    rounded_box(ax, (x1, y_main), x1b - x1, h_small, NAVY)
    circled_num(ax, x1 - 0.35, y_main + h_small + 0.35, "1")
    ax.text(
        (x1 + x1b) / 2,
        y_main + h_small - 0.35,
        "PDF URL / file\nUser question",
        ha="center",
        va="top",
        fontsize=8.5,
        color=NAVY_TEXT,
        linespacing=1.25,
        fontfamily="sans-serif",
    )
    ax.text(
        (x1 + x1b) / 2,
        y_main + 0.55,
        "Markdown · HTML · Email\nLegal · Chat",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#b8d4ea",
        linespacing=1.2,
        fontfamily="sans-serif",
    )

    # ② Download + Cache
    x2, x2b = cols[2]
    rounded_box(ax, (x2, y_main), x2b - x2, h_small, NAVY)
    circled_num(ax, x2 - 0.35, y_main + h_small + 0.35, "2")
    ax.text(
        (x2 + x2b) / 2,
        y_main + h_small - 0.28,
        "download_pdf(url) → bytes\noptional disk cache",
        ha="center",
        va="top",
        fontsize=8,
        color=NAVY_TEXT,
        linespacing=1.2,
        fontfamily="sans-serif",
    )
    ax.text(
        (x2 + x2b) / 2,
        y_main + 0.45,
        "SqliteKV: EmbeddingCache /\nRetrievalCache",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#b8d4ea",
        linespacing=1.15,
        fontfamily="sans-serif",
    )

    # ③ Ingest
    x3, x3b = cols[3]
    rounded_box(ax, (x3, y_main), x3b - x3, h_ingest, NAVY)
    circled_num(ax, x3 - 0.35, y_main + h_ingest + 0.35, "3")
    ingest_lines = (
        "ingest_pdf (paper mode)\n"
        "ingest_markdown · ingest_html\n"
        "ingest_email · ingest_legal\n"
        "ingest_chat\n"
        "—\n"
        "Blocks (page + anchors)\n"
        "Edges: parent / next / link_to\n"
        "Provenance: block_id"
    )
    ax.text(
        (x3 + x3b) / 2,
        y_main + h_ingest - 0.3,
        ingest_lines,
        ha="center",
        va="top",
        fontsize=7.4,
        color=NAVY_TEXT,
        linespacing=1.22,
        fontfamily="sans-serif",
    )

    # ④ Index
    x4, x4b = cols[4]
    rounded_box(ax, (x4, y_main), x4b - x4, h_small, NAVY)
    circled_num(ax, x4 - 0.35, y_main + h_small + 0.35, "4")
    ax.text(
        (x4 + x4b) / 2,
        y_main + h_small - 0.35,
        "BM25 index\n\nVector index\n(optional embeddings)\nall-MiniLM-L6-v2 (default)",
        ha="center",
        va="top",
        fontsize=8,
        color=NAVY_TEXT,
        linespacing=1.2,
        fontfamily="sans-serif",
    )

    # ⑤ Retrieve
    x5, x5b = cols[5]
    rounded_box(ax, (x5, y_main), x5b - x5, h_small, NAVY)
    circled_num(ax, x5 - 0.35, y_main + h_small + 0.35, "5")
    ax.text(
        (x5 + x5b) / 2,
        y_main + h_small - 0.28,
        "Route pages\nSearch within routed pages\nExpand coherence (parent + next)",
        ha="center",
        va="top",
        fontsize=7.8,
        color=NAVY_TEXT,
        linespacing=1.18,
        fontfamily="sans-serif",
    )
    ax.text(
        (x5 + x5b) / 2,
        y_main + 0.42,
        "fusion: RRF (default) | weighted\nbm25_weight / vector_weight",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#b8d4ea",
        linespacing=1.12,
        fontfamily="sans-serif",
    )

    # Rerank (optional) — between 5 and 6
    xr, xrb = cols["r"]
    hr = h_small * 0.92
    yr = y_main + (h_small - hr) / 2
    rounded_box(ax, (xr, yr), xrb - xr, hr, RERANK_BLUE_LIGHT, edgecolor=RERANK_BLUE, linewidth=1.5)
    ax.text(
        (xr + xrb) / 2,
        yr + hr + 0.32,
        "Rerank (optional)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=NAVY,
        fontfamily="sans-serif",
    )
    ax.text(
        (xr + xrb) / 2,
        yr + hr - 0.28,
        "CrossEncoderReranker\nms-marco-MiniLM-L-6-v2\n≥50 candidates",
        ha="center",
        va="top",
        fontsize=7.2,
        color=NAVY,
        linespacing=1.15,
        fontfamily="sans-serif",
    )

    # ⑥ Graph expand
    x6, x6b = cols[6]
    rounded_box(ax, (x6, y_main), x6b - x6, h_small * 0.95, ORANGE)
    circled_num(ax, x6 - 0.35, y_main + h_small * 0.95 + 0.35, "6")
    ax.text(
        (x6 + x6b) / 2,
        y_main + h_small * 0.95 - 0.3,
        "Follow refs:\nFigure / Table /\nAppendix / Section\n(link_to)",
        ha="center",
        va="top",
        fontsize=8,
        color="white",
        linespacing=1.15,
        fontfamily="sans-serif",
    )

    # ⑦ Output
    x7, x7b = cols[7]
    rounded_box(ax, (x7, y_main), x7b - x7, h_out, NAVY)
    circled_num(ax, x7 - 0.35, y_main + h_out + 0.35, "7")
    out_txt = (
        "Evidence blocks\n"
        "→ Answer\n"
        "→ Answer + citations [[block_id]]\n"
        "→ ConfidenceLevel HIGH / MEDIUM / LOW\n"
        "→ CostReport (optional)"
    )
    ax.text(
        (x7 + x7b) / 2,
        y_main + h_out - 0.32,
        out_txt,
        ha="center",
        va="top",
        fontsize=8,
        color=NAVY_TEXT,
        linespacing=1.22,
        fontfamily="sans-serif",
    )

    # Arrows along main flow
    arrow(ax, (x1b, y_main + h_small * 0.55), (x2, y_main + h_small * 0.55))
    arrow(ax, (x2b, y_main + h_small * 0.55), (x3, y_main + h_ingest * 0.55))
    arrow(ax, (x3b, y_main + h_ingest * 0.55), (x4, y_main + h_small * 0.55))
    arrow(ax, (x4b, y_main + h_small * 0.55), (x5, y_main + h_small * 0.55))
    arrow(ax, (x5b, y_main + h_small * 0.55), (xr, yr + hr * 0.5))
    arrow(ax, (xrb, yr + hr * 0.5), (x6, y_main + h_small * 0.48))
    arrow(ax, (x6b, y_main + h_small * 0.48), (x7, y_main + h_out * 0.5))

    # GraphRAG dashed (bottom) — under ingest → output region
    gx, gy, gw, gh = 3.45, 0.38, 9.35, 1.42
    dashed_box(ax, (gx, gy), gw, gh, zorder=2)
    ax.text(
        gx + gw / 2,
        gy + gh - 0.22,
        "GraphRAG (optional): extract entities/relations → build entity graph → multi-hop retrieve (provenance)",
        ha="center",
        va="top",
        fontsize=8.2,
        color=NAVY,
        fontfamily="sans-serif",
    )
    # Dashed connectors: ingest to GraphRAG, GraphRAG toward output
    ax.plot([x3 + (x3b - x3) * 0.5, gx + gw * 0.35], [y_main, gy + gh], color=NAVY, linestyle="--", linewidth=1, zorder=2)
    ax.plot([gx + gw * 0.72, x7 + (x7b - x7) * 0.4], [gy + gh, y_main], color=NAVY, linestyle="--", linewidth=1, zorder=2)

    # QueryFallback dashed (bottom right)
    qx, qy, qw, qh = 12.95, 0.38, 2.85, 1.42
    dashed_box(ax, (qx, qy), qw, qh, zorder=2)
    ax.text(
        qx + qw / 2,
        qy + qh - 0.2,
        "QueryFallback (optional)",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=NAVY,
        fontfamily="sans-serif",
    )
    ax.text(
        qx + qw / 2,
        qy + qh - 0.55,
        "LOW / MEDIUM confidence →\nLLM query variations →\nbest result",
        ha="center",
        va="top",
        fontsize=7,
        color=NAVY,
        linespacing=1.12,
        fontfamily="sans-serif",
    )

    # Curved retry arrow: QueryFallback → Retrieve
    retry = FancyArrowPatch(
        (qx + qw * 0.15, qy + qh),
        (x5 + (x5b - x5) * 0.85, y_main),
        connectionstyle="arc3,rad=-0.35",
        arrowstyle="-|>",
        mutation_scale=11,
        color=NAVY,
        linewidth=1.3,
        linestyle="--",
        zorder=4,
    )
    ax.add_patch(retry)
    ax.text(
        x5 + 1.15,
        y_main - 0.15,
        "retry",
        fontsize=7,
        color=NAVY,
        style="italic",
        fontfamily="sans-serif",
    )

    # Bench line
    ax.text(
        W / 2,
        0.12,
        "Offline benches: squad_benchmark / cuad_benchmark / scorecard / paper_eval / security_eval (optional)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=NAVY,
        fontfamily="sans-serif",
    )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor=BG, edgecolor="none", bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
