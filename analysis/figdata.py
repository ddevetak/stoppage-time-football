#!/usr/bin/env python3
"""Step 10: the numbers behind the paper's figures, as small public tables under figdata/.

Reads the match table (../data/matches.csv, ../data/durations.csv, ../data/inc_matches.csv)
and the public results; writes only aggregates. figures.py draws from figdata/ and nothing else.

  fig1_minutes.csv        goals per 1,000 matches in each minute of the clock, before and after,
                          with every league given the same weight in both periods
  fig2_monthly.csv        announced second-half minutes, real match length and second-half stoppage
                          goals by month, league-adjusted (European-calendar leagues of the panel)
  fig2_half_seasons.csv   the same three by half-season, with intervals (leagues resampled)
  fig3_league_paths.csv   announced second-half minutes by league and half-year
  fig4_by_board.csv       second-half stoppage goals per match and share of matches at each value
                          of the board, before and after
  fig5_by_board.csv       what happens to the score at 90:00 at each value of the board
  figdata.json            panels, weights and the few scalars the figures print
"""
import json
import os

import numpy as np
import pandas as pd

from decides import score_at_90

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "figdata")
RNG = np.random.default_rng(20261111)
BOOT = 2000
OUT = {}

FIRST_TAIL, SECOND_TAIL = 9, 13      # 45+9 and 90+13 collect everything later


def minute_slots():
    """The clock as an ordered list of (half, minute, added); added 0 is regular time."""
    s = [(1, t, 0) for t in range(1, 46)] + [(1, 45, a) for a in range(1, FIRST_TAIL + 1)]
    s += [(2, t, 0) for t in range(46, 91)] + [(2, 90, a) for a in range(1, SECOND_TAIL + 1)]
    return s


def slot_label(h, t, a):
    if a == 0:
        return str(t)
    tail = FIRST_TAIL if h == 1 else SECOND_TAIL
    return f"{t}+{a}" + ("+" if a == tail else "")


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig1(m):
    slots = minute_slots()
    idx = {s: i for i, s in enumerate(slots)}
    leagues = sorted(m.league.unique())
    li = {lg: i for i, lg in enumerate(leagues)}
    G = np.zeros((len(leagues), 2, len(slots)))
    N = np.zeros((len(leagues), 2))
    dropped = 0
    for lg, post, first, second in zip(m.league, m.post, m["first"].fillna(""), m["second"].fillna("")):
        i = li[lg]
        N[i, post] += 1
        for t in map(int, first.split()):
            G[i, post, idx[(1, t, 0)] if t <= 45 else idx[(1, 45, min(t - 45, FIRST_TAIL))]] += 1
        for t in map(int, second.split()):
            if t < 46:                      # a second-half goal with a first-half minute: a recording error
                dropped += 1
                continue
            G[i, post, idx[(2, t, 0)] if t <= 90 else idx[(2, 90, min(t - 90, SECOND_TAIL))]] += 1
    both = (N[:, 0] > 0) & (N[:, 1] > 0)
    G, N = G[both], N[both]
    L = len(N)

    def standardised(sel):
        w = N[sel].sum(axis=1)
        w = w / w.sum()
        return 1000 * np.einsum("l,lps->ps", w, G[sel] / N[sel][:, :, None])

    r = standardised(np.arange(L))
    diff = r[1] - r[0]
    cum = np.cumsum(diff)
    bd, bc = np.empty((BOOT, len(slots))), np.empty((BOOT, len(slots)))
    for b in range(BOOT):
        rb = standardised(RNG.integers(0, L, L))
        bd[b] = rb[1] - rb[0]
        bc[b] = np.cumsum(bd[b])
    raw = 1000 * G.sum(axis=0) / N.sum(axis=0)[:, None]
    t = pd.DataFrame(dict(
        half=[s[0] for s in slots], minute=[s[1] for s in slots], added=[s[2] for s in slots],
        label=[slot_label(*s) for s in slots],
        before=r[0], after=r[1], diff=diff, diff_lo=np.percentile(bd, 2.5, axis=0), diff_hi=np.percentile(bd, 97.5, axis=0),
        cum_diff=cum, cum_lo=np.percentile(bc, 2.5, axis=0), cum_hi=np.percentile(bc, 97.5, axis=0),
        before_unweighted=raw[0], after_unweighted=raw[1]))
    t.round(4).to_csv(os.path.join(OUTDIR, "fig1_minutes.csv"), index=False)
    stop = (t.added > 0).values
    seg = dict(first_regular=(t.half == 1) & ~stop, first_stoppage=(t.half == 1) & stop,
               second_regular=(t.half == 2) & ~stop, second_stoppage=(t.half == 2) & stop)
    part = {}
    for k_, sel in seg.items():
        v = bd[:, sel.values].sum(axis=1)
        part[k_] = dict(before=float(t.before[sel].sum()), after=float(t.after[sel].sum()), diff=float(t["diff"][sel].sum()),
                        lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
    v_all, v_stop = bd.sum(axis=1), bd[:, stop].sum(axis=1)
    part["all"] = dict(before=float(t.before.sum()), after=float(t.after.sum()), diff=float(t["diff"].sum()),
                       lo=float(np.percentile(v_all, 2.5)), hi=float(np.percentile(v_all, 97.5)))
    part["stoppage"] = dict(diff=float(t["diff"][stop].sum()), lo=float(np.percentile(v_stop, 2.5)), hi=float(np.percentile(v_stop, 97.5)))
    part["regular"] = dict(diff=float(t["diff"][~stop].sum()), lo=float(np.percentile(v_all - v_stop, 2.5)), hi=float(np.percentile(v_all - v_stop, 97.5)))
    share = v_stop / v_all
    part["stoppage_share_of_rise"] = dict(share=float(part["stoppage"]["diff"] / part["all"]["diff"]),
                                          lo=float(np.percentile(share, 2.5)), hi=float(np.percentile(share, 97.5)))
    OUT["fig1"] = dict(leagues=int(L), matches_before=int(N[:, 0].sum()), matches_after=int(N[:, 1].sum()),
                       second_half_goals_with_first_half_minute=int(dropped), per_1000_matches=part)
    print(f"fig1: {L} leagues, standardised. Per 1,000 matches, after minus before:")
    for k_ in ("first_regular", "first_stoppage", "second_regular", "second_stoppage", "regular", "stoppage", "all"):
        p = part[k_]
        print(f"   {k_:<16} {p['diff']:+7.1f}  [{p['lo']:+.1f}, {p['hi']:+.1f}]")
    p = part["stoppage_share_of_rise"]
    print(f"   stoppage time's share of the rise in all goals: {p['share']:.0%} [{p['lo']:.0%}, {p['hi']:.0%}]")


def league_adjust(d, col):
    """Each league's own mean removed and the overall mean put back."""
    return d[col] - d.groupby("league")[col].transform("mean") + d[col].mean()


def boot_cells(d, col, cell):
    """League-adjusted mean of col in each cell, with an interval from resampling leagues."""
    d = d[d[col].notna()].copy()
    d["adj"] = league_adjust(d, col)
    s = d.groupby(["league", cell]).adj.agg(["sum", "size"]).unstack(fill_value=0)
    S, C = s["sum"].values, s["size"].values
    cells = list(s["sum"].columns)
    est = S.sum(axis=0) / C.sum(axis=0)
    L = len(S)
    bs = np.empty((BOOT, len(cells)))
    for b in range(BOOT):
        i = RNG.integers(0, L, L)
        bs[b] = S[i].sum(axis=0) / np.maximum(C[i].sum(axis=0), 1)
    return pd.DataFrame(dict(cell=cells, n=C.sum(axis=0), value=est, lo=np.percentile(bs, 2.5, axis=0), hi=np.percentile(bs, 97.5, axis=0)))


def fig2(m, panel2):
    w = m[(m.calendar == "winter") & m.league.isin(panel2)].copy()
    w["ym"] = w.date.dt.to_period("M").astype(str)
    w["cell"] = w.season.str[2:4] + "/" + w.season.str[7:9] + np.where(w.date.dt.month >= 7, "a", "b")
    leagues = sorted(w.league.unique())
    OUT["fig2"] = dict(leagues=leagues, n_leagues=len(leagues), matches=int(len(w)),
                       with_board=int(w.board2.notna().sum()), with_length=int(w.total.notna().sum()))
    mon = None
    for col, name in (("board2", "board"), ("total", "length"), ("s2", "goals")):
        d = w[w[col].notna()].copy()
        d["adj"] = league_adjust(d, col)
        g = d.groupby("ym").adj.agg(["mean", "size"]).rename(columns={"mean": name, "size": f"n_{name}"})
        mon = g if mon is None else mon.join(g, how="outer")
    mon = mon[mon.n_goals >= 300].reset_index()
    mon.round(4).to_csv(os.path.join(OUTDIR, "fig2_monthly.csv"), index=False)
    hs = None
    for col, name in (("board2", "board"), ("total", "length"), ("s2", "goals")):
        t = boot_cells(w, col, "cell").rename(columns={"value": name, "lo": f"{name}_lo", "hi": f"{name}_hi", "n": f"n_{name}"})
        hs = t if hs is None else hs.merge(t, on="cell")
    hs.round(4).to_csv(os.path.join(OUTDIR, "fig2_half_seasons.csv"), index=False)
    print(f"\nfig2: {len(leagues)} European-calendar leagues, {len(w)} matches ({OUT['fig2']['with_board']} with a board, {OUT['fig2']['with_length']} with a usable length)")
    print(hs.round(3).to_string(index=False))


def fig3(m, panel2):
    p = m[m.league.isin(panel2) & m.board2.notna()].copy()
    p["year"] = p.date.dt.year
    p["h"] = np.where(p.date.dt.month >= 7, 2, 1)
    p["x"] = p.year + np.where(p.h == 1, 0.25, 0.75)
    g = p.groupby(["league", "calendar", "year", "h", "x"]).board2.agg(["mean", "std", "size"]).reset_index()
    g = g[g["size"] >= 30]
    g["se"] = g["std"] / np.sqrt(g["size"])
    g = g.rename(columns={"mean": "board", "size": "n"}).drop(columns="std")
    # the same half-years for all panel leagues together, league-adjusted
    p["adj"] = league_adjust(p, "board2")
    allp = p.groupby(["year", "h", "x"]).adj.agg(["mean", "size"]).reset_index().rename(columns={"mean": "board", "size": "n"})
    allp.insert(0, "calendar", "all")
    allp.insert(0, "league", "ALL")
    allp["se"] = np.nan
    pd.concat([g, allp]).round(4).to_csv(os.path.join(OUTDIR, "fig3_league_paths.csv"), index=False)
    # one line per league for the text: last season before, first autumn after, latest season
    rows = []
    for lg, q in p.groupby("league"):
        cal = q.calendar.iloc[0]
        rows.append(dict(league=lg, calendar=cal,
                         last_before=q[q.k == -1].board2.mean(), first_after=q[q.k == 0].board2.mean(), second_after=q[q.k == 1].board2.mean(),
                         latest=q[q.k == 2].board2.mean(), n_latest=int((q.k == 2).sum())))
    t = pd.DataFrame(rows)
    t["jump"] = t.first_after - t.last_before
    t["now"] = t.latest - t.last_before
    t.round(3).to_csv(os.path.join(OUTDIR, "fig3_league_summary.csv"), index=False)
    OUT["fig3"] = dict(leagues=int(t.league.nunique()),
                       jumped_half_minute=int((t.jump >= 0.5).sum()), still_half_minute_up=int((t.now >= 0.5).sum()),
                       back_within_quarter=int((t.now.abs() < 0.25).sum()), below=int((t.now <= -0.25).sum()))
    print(f"\nfig3: {len(t)} leagues; 2023/24 at least half a minute above 2022/23: {OUT['fig3']['jumped_half_minute']}; "
          f"2025/26 at least half a minute above: {OUT['fig3']['still_half_minute_up']}; within a quarter of a minute: {OUT['fig3']['back_within_quarter']}")
    print(t.sort_values("now").round(2).to_string(index=False))


def fig4(m):
    b = m[m.board2.notna()].copy()
    b["v"] = b.board2.clip(upper=10).astype(int)
    g = b.groupby(["v", "post"]).s2.agg(["sum", "size"]).reset_index().rename(columns={"sum": "goals", "size": "n"})
    g["share"] = g.n / g.groupby("post").n.transform("sum")
    g["rate"] = g.goals / g.n
    g["rate_lo"] = np.maximum(g.goals - 1.96 * np.sqrt(g.goals), 0) / g.n
    g["rate_hi"] = (g.goals + 1.96 * np.sqrt(g.goals)) / g.n
    g["period"] = np.where(g.post == 1, "after", "before")
    g.drop(columns="post").round(5).to_csv(os.path.join(OUTDIR, "fig4_by_board.csv"), index=False)
    print("\nfig4: second-half stoppage goals per match and share of matches by board value")
    print(g.pivot(index="v", columns="period", values=["n", "rate", "share"]).round(3).to_string())


def fig5(x):
    x = x[x.h90x.notna() & x.board2.notna()].copy()
    x["lead90"] = (x.h90x - x.a90x).abs()
    fin = x.ft_h - x.ft_a
    leader_won = np.where(x.h90x > x.a90x, fin > 0, fin < 0)
    x["v"] = x.board2.clip(lower=2, upper=9).astype(int)
    rows = []
    for name, sel, y in (("one-goal lead not held", x.lead90 == 1, ~leader_won),
                         ("level match gets a winner", x.lead90 == 0, (fin != 0).values)):
        q = x[sel].assign(y=np.asarray(y)[sel.values].astype(int))
        for (v, post), g in q.groupby(["v", "post"]):
            lo, hi = wilson(g.y.sum(), len(g))
            rows.append(dict(outcome=name, v=int(v), period="after" if post else "before", n=int(len(g)), k=int(g.y.sum()),
                             share=float(g.y.mean()), lo=lo, hi=hi))
        for v, g in q.groupby("v"):
            lo, hi = wilson(g.y.sum(), len(g))
            rows.append(dict(outcome=name, v=int(v), period="all", n=int(len(g)), k=int(g.y.sum()), share=float(g.y.mean()), lo=lo, hi=hi))
    t = pd.DataFrame(rows)
    t.round(5).to_csv(os.path.join(OUTDIR, "fig5_by_board.csv"), index=False)
    print("\nfig5: share by board value (2 = two or less, 9 = nine or more)")
    print(t.pivot(index=["outcome", "v"], columns="period", values=["n", "share"]).round(3).to_string())


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    x = score_at_90()
    m = x.copy()
    m["date"] = pd.to_datetime(m.month)
    d = pd.read_csv(os.path.join(HERE, "..", "data", "durations.csv"))
    d["total"] = d.total_s / 60
    d.loc[~(d.total.between(100, 135) & (d.last_change_is_status == 1)), "total"] = np.nan
    m = m.merge(d[["event_id", "total"]].dropna().drop_duplicates("event_id"), on="event_id", how="left")
    panel2 = json.load(open(os.path.join(HERE, "results_mechanism.json")))["panel"]["2"]["names"]
    OUT["panel2"] = panel2
    fig1(m)
    fig2(m, panel2)
    fig3(m, panel2)
    fig4(m)
    fig5(x)
    json.dump(OUT, open(os.path.join(OUTDIR, "figdata.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
