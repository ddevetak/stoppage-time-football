#!/usr/bin/env python3
"""Step 11: the paper's five figures, drawn from figdata/ and the public results only.

  python3 figures.py            writes ../paper/figures/fig{1..5}_*.png (600 dpi) and .pdf

One visual grammar throughout:
  - gray is before the change, blue is after it (open and filled markers where the two are compared
    point by point, so they also differ without colour);
  - every figure is drawn on a canvas exactly as wide as the page's text (6.5 in) and saved uncropped,
    so a 7.5-point label is 7.5 points on the page in every figure;
  - each panel has a letter, a title and, below it, the unit of its vertical axis; no rotated labels;
  - grid and axes are hairlines, text is never in a data colour, intervals are thin and lighter than
    the estimate they belong to.

The typeface is Helvetica Neue where the system has it (macOS), else Arial, else DejaVu Sans. The
version of this file used for the first draft is kept as figures_v1.py.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "figdata")
OUT = os.path.join(HERE, "..", "figures")

BLUE, BLUE_M, BLUE_L = "#1c5cab", "#86b6ef", "#cde2fb"     # after: mark, interval, wash
GRAY, GRAY_M, GRAY_L = "#898781", "#c9c8c0", "#e6e5df"     # before: mark, interval or context, wash
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, ZONE = "#e9e8e2", "#b9b8af", "#f4f3ef"
W = 6.5                                                    # text width of the page, inches
GUT = 0.30                                                 # room left of an axis for its tick labels, inches
PAD = 0.02                                                 # white at the left edge of the canvas, inches


def typeface():
    """Helvetica Neue in four weights, taken out of the macOS font collection (matplotlib reads only
    the first face of a .ttc file). Falls back to Arial where the collection is not there."""
    src = "/System/Library/Fonts/HelveticaNeue.ttc"
    want = {"Helvetica Neue", "Helvetica Neue Medium", "Helvetica Neue Bold", "Helvetica Neue Light", "Helvetica Neue Italic"}
    try:
        from fontTools.ttLib import TTCollection
        cache = os.path.join(os.path.expanduser("~"), "Library", "Caches", "stoppage-time-figures")
        os.makedirs(cache, exist_ok=True)
        got = 0
        for face in TTCollection(src).fonts:
            full = face["name"].getDebugName(4)
            if full in want:
                path = os.path.join(cache, full.replace(" ", "") + ".ttf")
                if not os.path.exists(path):
                    face.save(path)
                fm.fontManager.addfont(path)
                got += 1
        if got == len(want):
            return ["Helvetica Neue", "Arial", "DejaVu Sans"]
    except Exception:
        pass
    return ["Arial", "Helvetica", "DejaVu Sans"]


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": typeface(), "font.size": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.labelsize": 7.5, "axes.labelpad": 5,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.major.size": 2.5, "ytick.major.size": 0,
    "xtick.major.width": 0.6, "xtick.major.pad": 3, "ytick.major.pad": 4,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 7.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "savefig.dpi": 600, "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
})

LINE = dict(lw=1.2, marker="o", ms=2.6, mec="none")                                     # a series over time
BEFORE = dict(color=GRAY, lw=1.1, marker="o", ms=4.3, mfc="white", mec=GRAY, mew=1.1)    # point-by-point comparisons
AFTER = dict(color=BLUE, lw=1.3, marker="o", ms=4.3, mfc=BLUE, mec="white", mew=0.7)


def canvas(height):
    return plt.figure(figsize=(W, height))


def axes(fig, left, bottom, width, height, **kw):
    """An axes placed in inches from the lower left corner of the canvas."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([(PAD + left) / fw, bottom / fh, width / fw, height / fh], **kw)


def header(ax, letter, title, unit=None, gutter=GUT):
    """Panel letter (bold), title (medium) and the unit of the vertical axis (muted), set flush with
    the left edge of the tick labels."""
    fig = ax.figure
    fw, fh = fig.get_size_inches()
    box = ax.get_position()
    x = box.x0 - gutter / fw
    y = box.y1 + ((0.215 if unit else 0.075) / fh)
    if letter:
        fig.text(x, y, letter, ha="left", va="baseline", fontsize=9, fontweight="bold", color=INK)
    fig.text(x + ((0.15 if letter else 0) / fw), y, title, ha="left", va="baseline", fontsize=9, fontweight=500, color=INK)
    if unit:
        fig.text(x, box.y1 + 0.068 / fh, unit, ha="left", va="baseline", fontsize=7.5, color=INK2)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def sg(v, fmt="+.0f"):
    """A signed number with a typographic minus."""
    return format(v, fmt).replace("-", "−")


def period_legend(ax, loc="upper left", before="Before the change", after="After", **kw):
    h = [Line2D([], [], label=before, **BEFORE), Line2D([], [], label=after, **AFTER)]
    kw.setdefault("handlelength", 1.9)
    kw.setdefault("borderaxespad", 0.3)
    kw.setdefault("labelspacing", 0.45)
    return ax.legend(handles=h, loc=loc, labelcolor=INK2, **kw)


# ---------------------------------------------------------------- Figure 1
def fig1():
    t = pd.read_csv(os.path.join(FD, "fig1_minutes.csv"))
    part = json.load(open(os.path.join(FD, "figdata.json")))["fig1"]["per_1000_matches"]
    half2 = (t.half == 2).values
    x = np.arange(len(t)) + np.where(half2, 2.0, 0.0)          # a small gap between the halves
    stop = (t.added > 0).values
    z1, z2 = x[stop & ~half2], x[stop & half2]
    r1, r2 = x[~stop & ~half2], x[~stop & half2]

    fig = canvas(4.10)
    a = axes(fig, GUT, 2.27, W - PAD - GUT - 0.06, 1.47)
    b = axes(fig, GUT, 0.45, W - PAD - GUT - 0.06, 1.32, sharex=a)
    for ax in (a, b):
        for z in (z1, z2):
            ax.axvspan(z[0] - 0.5, z[-1] + 0.5, color=ZONE, lw=0, zorder=0)

    for m in (~half2, half2):
        xb, yb, ya = x[m], t.before[m].values, t.after[m].values
        a.fill_between(xb, yb, ya, where=ya >= yb, interpolate=True, color=BLUE_L, lw=0, zorder=1)
        a.fill_between(xb, yb, ya, where=ya < yb, interpolate=True, color=GRAY_L, lw=0, zorder=1)
        a.plot(xb, yb, color=GRAY, lw=1.2, zorder=2)
        a.plot(xb, ya, color=BLUE, lw=1.2, zorder=3)
    a.set_ylim(0, 40)
    a.set_yticks([0, 10, 20, 30])
    a.tick_params(labelbottom=False)
    for z in (z1, z2):
        a.text(z.mean(), 39.0, "stoppage\ntime", ha="center", va="top", color=INK2, fontsize=7, linespacing=1.15)
    a.legend(handles=[Line2D([], [], color=GRAY, lw=1.2, label="Before the change (2020/21 to 2022/23)"),
                      Line2D([], [], color=BLUE, lw=1.2, label="After (2023/24 to 2025/26)")],
             loc="lower left", bbox_to_anchor=(0.045, 0.035), handlelength=1.9, labelcolor=INK2, labelspacing=0.35)
    k = np.where(stop & half2)[0][4]                           # 90+5, where the gap between the curves is widest
    a.annotate("extra goals\nafter the change", xy=(x[k], (t.before[k] + t.after[k]) / 2), xytext=(z2[0] - 4.5, 13.5),
               ha="right", va="center", color=INK2, fontsize=7, linespacing=1.15,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=3, shrinkB=0))
    header(a, "a", "Goals in each minute of the match", "Goals per 1,000 matches")

    reg = ~stop
    b.axhline(0, color=AXIS, lw=0.7, zorder=1)
    b.vlines(x[reg], t.diff_lo[reg], t.diff_hi[reg], color=GRAY_M, lw=0.6, zorder=2)
    b.vlines(x[stop], t.diff_lo[stop], t.diff_hi[stop], color=BLUE_M, lw=0.8, zorder=2)
    b.plot(x[reg], t["diff"][reg], lw=0, marker="o", ms=2.3, mfc=GRAY, mec="none", zorder=3)
    b.plot(x[stop], t["diff"][stop], lw=0, marker="o", ms=3.0, mfc=BLUE, mec="none", zorder=4)
    b.set_ylim(-7.5, 18.6)
    b.set_yticks([-5, 0, 5, 10])
    b.yaxis.set_major_formatter(lambda v, _: sg(v, ".0f") if v else "0")

    def bracket(xs, key, strong):
        p = part[key]
        x0, x1, yb = xs[0] - 0.5, xs[-1] + 0.5, 12.3
        b.plot([x0, x0, x1, x1], [yb - 0.6, yb, yb, yb - 0.6], color=BLUE if strong else AXIS, lw=0.8,
               solid_joinstyle="miter", solid_capstyle="butt", zorder=3)
        xc = (x0 + x1) / 2
        b.text(xc, yb + 3.1, sg(p["diff"]), ha="center", va="baseline", fontsize=8.5,
               fontweight="bold" if strong else 500, color=INK if strong else INK2)
        b.text(xc, yb + 0.85, f"{sg(p['lo'])} to {sg(p['hi'])}", ha="center", va="baseline", fontsize=7,
               color=INK2 if strong else MUTED)
    bracket(r1, "first_regular", False)
    bracket(z1, "first_stoppage", True)
    bracket(r2, "second_regular", False)
    bracket(z2, "second_stoppage", True)

    ticks, labels = [], []
    for xi, r in zip(x, t.itertuples()):
        if r.added == 0 and r.minute in (1, 15, 30, 45, 60, 75, 90):
            ticks.append(xi); labels.append(str(int(r.minute)))
        elif r.added in (5, 10) and not (r.half == 1 and r.added == 10):
            ticks.append(xi); labels.append(f"+{int(r.added)}")
    b.set_xticks(ticks, labels)
    b.set_xlim(x[0] - 1.6, x[-1] + 0.9)
    b.set_xlabel("Minute of the match")
    header(b, "b", "After minus before",
           "Change in goals per 1,000 matches, with 95% intervals. Brackets: the change summed over each stretch.")
    save(fig, "fig1_minutes")


# ---------------------------------------------------------------- Figure 2
def month_x(ym):
    y, mth = int(ym[:4]), int(ym[5:7])
    return y + (mth - 0.5) / 12


def fig2():
    mon = pd.read_csv(os.path.join(FD, "fig2_monthly.csv"))
    hs = pd.read_csv(os.path.join(FD, "fig2_half_seasons.csv"))
    mon["x"] = mon.ym.map(month_x)
    mon["season"] = mon.ym.map(lambda s: int(s[:4]) if int(s[5:7]) >= 7 else int(s[:4]) - 1)
    hs["x"] = [2000 + int(c[:2]) + (0.79 if c.endswith("a") else 1.21) for c in hs.cell]
    hs["after"] = [int(c[:2]) >= 23 for c in hs.cell]

    fig = canvas(4.86)
    gut, hgt, pitch = 0.33, 1.02, 1.49
    panels = [axes(fig, gut, 0.50 + i * pitch, W - PAD - gut - 0.06, hgt) for i in (2, 1, 0)]
    spec = [("board", "a", "Announced second-half stoppage time", "Minutes, by month", (3.6, 6.6), [4, 5, 6]),
            ("length", "b", "From kick-off to the final whistle", "Minutes, interval included, by month", (112.6, 119.7), [113, 115, 117, 119]),
            ("goals", "c", "Goals in second-half stoppage time", "Per match, by half-season, with 95% intervals", (0.125, 0.245), [0.14, 0.18, 0.22])]
    for ax, (col, letter, title, unit, ylim, yt) in zip(panels, spec):
        ax.axvspan(2022 + 10.6 / 12, 2022 + 11.6 / 12, color=ZONE, lw=0, zorder=0)
        ax.axvline(2023.5, color=AXIS, lw=0.6, zorder=1)
        if col != "goals":
            for season, g in mon[mon[f"n_{col}"] >= 300].groupby("season"):
                c = BLUE if season >= 2023 else GRAY
                ax.plot(g.x, g[col], color=c, mfc=c, zorder=3, **LINE)
        else:
            for after, g in hs.groupby("after"):
                c, tint = (BLUE, BLUE_M) if after else (GRAY, GRAY_M)
                ax.vlines(g.x, g.goals_lo, g.goals_hi, color=tint, lw=1.0, zorder=2)
                ax.plot(g.x, g.goals, lw=0, marker="o", ms=4.4, mfc=c, mec="white", mew=0.7, zorder=3)
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.set_xlim(2020.5, 2026.5)
        ax.set_xticks([2020.5 + i for i in range(7)], [""] * 7)
        header(ax, letter, title, unit, gutter=gut)
    a, b, c = panels
    aug23 = mon[mon.ym == "2023-08"].iloc[0]
    a.plot([aug23.x], [aug23.board], lw=0, marker="o", ms=5.0, mfc=BLUE, mec="white", mew=0.8, zorder=4)
    a.annotate(f"August 2023: {aug23.board:.1f}", xy=(aug23.x, aug23.board), xytext=(7, 0), textcoords="offset points",
               ha="left", va="center", color=INK, fontsize=7.5)
    a.text(2022 + 11.1 / 12, 6.52, "World Cup", ha="center", va="top", color=INK2, fontsize=7)
    a.legend(handles=[Line2D([], [], color=GRAY, mfc=GRAY, label="Before the change", **LINE),
                      Line2D([], [], color=BLUE, mfc=BLUE, label="After", **LINE)],
             loc="upper left", bbox_to_anchor=(0.012, 1.02), ncol=2, columnspacing=1.6, handlelength=1.9, labelcolor=INK2,
             borderaxespad=0.2)
    c.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    c.set_xticks([2021 + i for i in range(6)], [f"{2020 + i}/{21 + i}" for i in range(6)], minor=True)
    c.tick_params(axis="x", which="minor", length=0, labelsize=7.5, labelcolor=INK2, pad=4)
    c.set_xlabel("Season (August to May)")
    save(fig, "fig2_timing")


# ---------------------------------------------------------------- Figure 3
FIG3 = [("china", "China"), ("korea1", "South Korea"), ("france1", "France"), ("germany1", "Germany"),
        ("premier", "England"), ("usa1", "United States"), ("brazil1", "Brazil"), ("spain1", "Spain"),
        ("holland1", "Netherlands"), ("italy1", "Italy"), ("saudi1", "Saudi Arabia"), ("turkey1", "Turkey")]


def fig3():
    p = pd.read_csv(os.path.join(FD, "fig3_league_paths.csv"))
    s = pd.read_csv(os.path.join(FD, "fig3_league_summary.csv")).set_index("league")
    n_panel = json.load(open(os.path.join(FD, "figdata.json")))["fig3"]["leagues"]
    allp = p[p.league == "ALL"].sort_values("x")
    allp = allp[allp.n >= 1500]
    order = sorted(FIG3, key=lambda r: -s.loc[r[0], "now"])

    gut, gap, hgt, pitch, base = 0.22, 0.17, 0.90, 1.33, 0.52
    fig = canvas(base + 2 * pitch + hgt + 0.72)
    wdt = (W - PAD - gut - 0.06 - 3 * gap) / 4
    fw, fh = fig.get_size_inches()
    for i, (lg, name) in enumerate(order):
        row, col = divmod(i, 4)
        ax = axes(fig, gut + col * (wdt + gap), base + (2 - row) * pitch, wdt, hgt)
        g = p[p.league == lg].sort_values("x")
        summer = g.calendar.iloc[0] == "summer"
        start = 2023.0 if summer else 2023.5
        pre, post = g[g.x < start], g[g.x >= start]
        ax.plot(allp.x, allp.board, color=GRAY_M, lw=1.0, zorder=2)
        ax.plot(pre.x, pre.board, color=GRAY, mfc=GRAY, zorder=3, **LINE)
        join = pd.concat([pre.tail(1), post])
        ax.plot(join.x, join.board, color=BLUE, lw=LINE["lw"], zorder=3.5)
        ax.plot(post.x, post.board, color=BLUE, mfc=BLUE, zorder=4, **{**LINE, "lw": 0})
        ax.set_ylim(3, 9)
        ax.set_yticks([3, 5, 7, 9])
        ax.set_xlim(2020.4, 2026.6)
        ax.set_xticks([2021, 2023, 2025], ["2021", "2023", "2025"])
        ax.tick_params(labelleft=(col == 0), labelbottom=(row == 2))
        box = ax.get_position()
        fig.text(box.x0, box.y1 + 0.205 / fh, name, ha="left", va="baseline", fontsize=8.5, fontweight=500, color=INK)
        d = s.loc[lg, "now"]
        fig.text(box.x0, box.y1 + 0.065 / fh, f"{sg(d, '+.1f')} min since {'2022' if summer else '2022/23'}",
                 ha="left", va="baseline", fontsize=7, color=INK2)
    fig.text(PAD / fw, 1 - 0.125 / fh, "Announced second-half stoppage time in twelve top divisions",
             ha="left", va="baseline", fontsize=9, fontweight=500, color=INK)
    fig.text(PAD / fw, 1 - 0.272 / fh, "Minutes, by half-year", ha="left", va="baseline", fontsize=7.5, color=INK2)
    fig.legend(handles=[Line2D([], [], color=GRAY, mfc=GRAY, label="The league before the change", **LINE),
                        Line2D([], [], color=BLUE, mfc=BLUE, label="and after it", **LINE),
                        Line2D([], [], color=GRAY_M, lw=1.0, label=f"All {n_panel} leagues of the panel")],
               loc="lower left", bbox_to_anchor=((PAD + gut) / fw - 0.008, 0.0), ncol=3, handlelength=1.9, labelcolor=INK2,
               columnspacing=2.0, borderaxespad=0.25)
    save(fig, "fig3_leagues")


# ---------------------------------------------------------------- Figure 4
LABELS4 = {"china": ("China", 0, 8, "center"), "korea1": ("South Korea", -6, 1, "right"),
           "premier": ("England", 0, 8, "center"), "turkey1": ("Turkey", -1, 8, "right"), "italy1": ("Italy", 6, -5, "left"),
           "usa1": ("United States", 7, -6, "left")}


def fig4():
    d = pd.read_csv(os.path.join(HERE, "league_dose.csv"))
    d = d[d.half == 2]
    mech = json.load(open(os.path.join(HERE, "results_mechanism.json")))["dose"]["2"]
    bb = pd.read_csv(os.path.join(FD, "fig4_by_board.csv"))

    fig = canvas(3.62)
    gut = 0.33
    wa = 2.72
    left_r = gut + wa + 0.30 + GUT
    wr = W - PAD - left_r - 0.06
    a = axes(fig, gut, 0.44, wa, 2.84)
    b = axes(fig, left_r, 1.96, wr, 1.32)
    c = axes(fig, left_r, 0.44, wr, 0.98, sharex=b)

    xs = np.array([-0.6, 2.65])
    rate = mech["pre_rate_per_minute"]
    a.grid(axis="x", color=GRID, lw=0.5)
    a.axhline(0, color=AXIS, lw=0.7, zorder=1)
    a.axvline(0, color=AXIS, lw=0.7, zorder=1)
    a.plot(xs, rate * xs, color=GRAY, lw=1.1, zorder=2)
    a.plot(d.d_board, d.d_goals, lw=0, marker="o", ms=5.0, mfc=BLUE, mec="white", mew=0.8, zorder=3)
    a.set_xlim(*xs)
    a.set_ylim(-0.04, 0.125)
    a.set_yticks([0, 0.04, 0.08, 0.12])
    a.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    a.spines["bottom"].set_visible(False)
    a.tick_params(axis="x", length=0)
    a.set_xlabel("Change in announced second-half stoppage time (minutes)")
    for r in d.itertuples():
        if r.league in LABELS4:
            name, dx, dy, ha = LABELS4[r.league]
            a.annotate(name, xy=(r.d_board, r.d_goals), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                       color=INK2, fontsize=7.5)
    # the label of the line runs along it, below it on the right, where no league sits
    p0, p1 = a.transData.transform((1.0, rate * 1.0)), a.transData.transform((2.0, rate * 2.0))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    a.annotate("the rate before the change", xy=(2.04, rate * 2.04), xytext=(0, -5.5), textcoords="offset points",
               rotation=angle, rotation_mode="anchor", ha="center", va="top", color=INK2, fontsize=7)
    header(a, "a", f"More minutes, more goals: {len(d)} leagues", "Change in second-half stoppage-time goals per match", gutter=gut)

    for per, st, dx in (("before", BEFORE, -0.07), ("after", AFTER, 0.07)):
        g = bb[bb.period == per]
        b.vlines(g.v + dx, g.rate_lo, g.rate_hi, color=GRAY_M if per == "before" else BLUE_M, lw=0.9, zorder=2)
        b.plot(g.v + dx, g.rate, zorder=3.5 if per == "after" else 3, **st)
    b.set_ylim(0, 0.56)
    b.set_yticks([0, 0.2, 0.4])
    b.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}" if v else "0")
    b.tick_params(labelbottom=False)
    period_legend(b, loc="upper left", bbox_to_anchor=(0.01, 1.0))
    header(b, "b", "Goals at each announced length", "Second-half stoppage-time goals per match, with 95% intervals")

    wbar = 0.30
    gb, ga = bb[bb.period == "before"], bb[bb.period == "after"]
    c.bar(gb.v - wbar / 2 - 0.03, 100 * gb.share, width=wbar, color=GRAY, lw=0, zorder=3)
    c.bar(ga.v + wbar / 2 + 0.03, 100 * ga.share, width=wbar, color=BLUE, lw=0, zorder=3)
    c.set_ylim(0, 35)
    c.set_yticks([0, 10, 20, 30])
    c.set_xticks(range(1, 11), [str(i) for i in range(1, 10)] + ["10+"])
    c.set_xlim(0.4, 10.6)
    c.set_xlabel("Announced second-half stoppage time (minutes)")
    header(c, "c", "How often each length is announced", "% of matches")
    save(fig, "fig4_mechanism")


# ---------------------------------------------------------------- Figure 5
def fig5():
    t = pd.read_csv(os.path.join(FD, "fig5_by_board.csv"))
    res = json.load(open(os.path.join(HERE, "results_decides.json")))["leads"]
    fig = canvas(3.05)
    gap = 0.24
    wdt = (W - PAD - GUT - 0.06 - gap) / 2
    panels = [axes(fig, GUT + i * (wdt + gap), 0.62, wdt, 2.06) for i in (0, 1)]
    spec = [("one-goal lead not held", "a", "A one-goal lead at 90:00 is not held", (0.965, 0.675, "right")),
            ("level match gets a winner", "b", "A level match at 90:00 gets a winner", (0.04, 0.80, "left"))]
    for i, (ax, (key, letter, title, (tx, ty, ha))) in enumerate(zip(panels, spec)):
        q = t[(t.outcome == key) & (t.n >= 100)]
        for per, st, dx in (("before", BEFORE, -0.08), ("after", AFTER, 0.08)):
            g = q[q.period == per]
            ax.vlines(g.v + dx, 100 * g.lo, 100 * g.hi, color=GRAY_M if per == "before" else BLUE_M, lw=0.9, zorder=2)
            ax.plot(g.v + dx, 100 * g.share, zorder=3.5 if per == "after" else 3, **st)
        ax.set_xticks(range(2, 10), ["2 or\nless"] + [str(v) for v in range(3, 9)] + ["9 or\nmore"])
        ax.set_xlim(1.5, 9.5)
        ax.set_ylim(0, 42)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.tick_params(labelleft=(i == 0))
        r = res[key]
        ax.text(tx, ty, f"{sg(r['per_board_minute_pp'], '+.1f')} percentage points\nper announced minute", transform=ax.transAxes,
                ha=ha, va="top", color=INK, fontsize=7.5, linespacing=1.3)
        header(ax, letter, title, "% of such matches, with 95% intervals", gutter=GUT if i == 0 else 0.0)
    period_legend(panels[0], loc="upper left", bbox_to_anchor=(0.01, 0.935))
    fw, fh = fig.get_size_inches()
    fig.text((PAD + GUT + (W - PAD - GUT - 0.06) / 2) / fw, 0.07 / fh, "Announced second-half stoppage time (minutes)",
             ha="center", va="baseline", fontsize=7.5, color=INK2)
    save(fig, "fig5_decides")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
