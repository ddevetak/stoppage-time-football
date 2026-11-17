"""Reproduce every table and figure of the stoppage-time study from
data/league_season_aggregates.csv alone.

    pip install matplotlib
    python analysis/reproduce.py

Prints Table 1 (overall), Table 2 (by tier), the event study and the
league count, and writes figures/fig1_event_study.pdf, fig2_tiers.pdf,
fig3_leagues.pdf.
"""
import csv
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "league_season_aggregates.csv"))))
INT = ["season_rel", "matches_total", "matches_excluded", "matches", "goals", "normal_time_goals",
       "stoppage_goals_1h", "stoppage_goals_2h", "late_equaliser_draws"]
for r in rows:
    for k in INT:
        r[k] = int(r[k])
    r["stoppage"] = r["stoppage_goals_1h"] + r["stoppage_goals_2h"]


def total(rs):
    t = defaultdict(int)
    for r in rs:
        for k in INT[1:] + ["stoppage"]:
            t[k] += r[k]
    return t


def rr(pre, post, key):
    a, b = pre[key], post[key]
    ratio = (b / post["matches"]) / (a / pre["matches"])
    se = math.sqrt(1 / a + 1 / b)
    return ratio, ratio * math.exp(-1.96 * se), ratio * math.exp(1.96 * se)


def fmt(t):
    return f"{t[0]:.3f} ({t[1]:.3f}-{t[2]:.3f})"


pre = total(r for r in rows if r["period"] == "pre")
post = total(r for r in rows if r["period"] == "post")
excl = pre["matches_excluded"] + post["matches_excluded"]
print(f"Matches used: {pre['matches']} pre, {post['matches']} post; excluded {excl} "
      f"({100 * excl / (pre['matches_total'] + post['matches_total']):.1f}%)")
print("\nTable 1  per match: pre / post / RR (95% CI)")
for key, label in (("goals", "All goals"), ("normal_time_goals", "Normal time"),
                   ("stoppage_goals_1h", "Stoppage, 1st half"), ("stoppage_goals_2h", "Stoppage, 2nd half"),
                   ("stoppage", "Stoppage, total")):
    print(f"  {label:20} {pre[key] / pre['matches']:.3f} / {post[key] / post['matches']:.3f} / {fmt(rr(pre, post, key))}")
print(f"  Late-equaliser draws  {100 * pre['late_equaliser_draws'] / pre['matches']:.2f}% / "
      f"{100 * post['late_equaliser_draws'] / post['matches']:.2f}%")

TIERS = [("1", "Top division"), ("2", "Second tier"), ("3", "Third tier"), ("4-6", "Tiers 4-6"),
         ("women", "Women's top divisions"), ("youth", "Youth (placebo)")]
print("\nTable 2  stoppage goals per match by tier")
tier_rr = {}
for t, label in TIERS:
    a = total(r for r in rows if r["tier"] == t and r["period"] == "pre")
    b = total(r for r in rows if r["tier"] == t and r["period"] == "post")
    tier_rr[t] = (rr(a, b, "stoppage"), rr(a, b, "normal_time_goals"))
    print(f"  {label:22} n={a['matches']}/{b['matches']}  {a['stoppage'] / a['matches']:.3f} -> "
          f"{b['stoppage'] / b['matches']:.3f}  RR {fmt(tier_rr[t][0])}  normal-time RR {tier_rr[t][1][0]:.3f}")

print("\nEvent study: stoppage goals per match by relative season")
es = {}
for cal in ("winter", "summer"):
    for k in range(-3, 3):
        t = total(r for r in rows if r["calendar"] == cal and r["season_rel"] == k)
        es[(cal, k)] = t["stoppage"] / t["matches"]
        print(f"  {cal:6} {k:+d}  {es[(cal, k)]:.4f}  (n={t['matches']})")

leagues = []
for lg in sorted({r["league"] for r in rows}):
    a = total(r for r in rows if r["league"] == lg and r["period"] == "pre")
    b = total(r for r in rows if r["league"] == lg and r["period"] == "post")
    if a["matches"] >= 150 and b["matches"] >= 150:
        tier = next(r["tier"] for r in rows if r["league"] == lg)
        leagues.append((lg, tier, a["stoppage"] / a["matches"], b["stoppage"] / b["matches"]))
up = sum(1 for x in leagues if x[3] > x[2])
print(f"\nLeagues with >=150 matches in each period: {len(leagues)}; stoppage goals up in {up}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\nmatplotlib not installed; skipping figures")
    raise SystemExit
out = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for cal, col, lab in (("winter", "#1f4e79", "Autumn-spring leagues (0 = 2023/24)"),
                      ("summer", "#c0504d", "Calendar-year leagues (0 = 2023)")):
    ax.plot(range(-3, 3), [es[(cal, k)] for k in range(-3, 3)], marker="o", color=col, label=lab, lw=1.6, ms=4)
ax.axvline(-0.5, color="grey", ls="--", lw=0.8)
ax.set_xlabel("Season relative to the IFAB decision (0 = first season after)"); ax.set_ylabel("Stoppage-time goals per match")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_event_study.pdf"))

fig, ax = plt.subplots(figsize=(4.6, 2.9))
for i, (t, label) in enumerate(TIERS):
    y = len(TIERS) - 1 - i
    for (r_, lo, hi), col, off, mk in ((tier_rr[t][0], "#1f4e79", 0.12, "o"), (tier_rr[t][1], "#999999", -0.12, "s")):
        ax.errorbar(r_, y + off, xerr=[[r_ - lo], [hi - r_]], fmt=mk, color=col, ms=4, capsize=2, lw=1)
ax.axvline(1, color="black", lw=0.7)
ax.set_yticks(range(len(TIERS))); ax.set_yticklabels([l for _, l in reversed(TIERS)])
ax.set_xlabel("Rate ratio, post / pre (95% CI)")
ax.plot([], [], "o", color="#1f4e79", label="Stoppage-time goals"); ax.plot([], [], "s", color="#999999", label="Goals in normal time")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_tiers.pdf"))

cols = {"1": "#1f4e79", "2": "#4f81bd", "3": "#9bbb59", "4-6": "#f79646", "women": "#c0504d", "youth": "#7f7f7f"}
fig, ax = plt.subplots(figsize=(3.6, 3.4))
for t, c in cols.items():
    pts = [x for x in leagues if x[1] == t]
    ax.scatter([p[2] for p in pts], [p[3] for p in pts], s=14, color=c, label=t, alpha=0.85, edgecolor="none")
ax.plot([0.05, 0.55], [0.05, 0.55], color="black", lw=0.7)
ax.set_xlim(0.05, 0.55); ax.set_ylim(0.05, 0.55)
ax.set_xlabel("Stoppage goals per match, pre"); ax.set_ylabel("Stoppage goals per match, post")
ax.legend(frameon=False, fontsize=7, loc="upper left", title="Tier", title_fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_leagues.pdf"))
print("\nFigures written to figures/")
