#!/usr/bin/env python3
"""Step 8: how long matches really last, from the clock times of the season listings.

  python3 durations.py analyse      joins ../data/matches.csv and ../data/durations.csv, prints, writes results_durations.json

Each listed match carries three clock times: the scheduled kick-off, the start of the second half
(where it has not been overwritten at the final whistle) and the last change to the record, which is
the final whistle when that last change was the status turning to "Ended". ../data/durations.csv gives
their differences in seconds: total_s (kick-off to final whistle, interval included) and h2_s (the real
length of the second half). Both are kept only when plausible, and summarised by medians.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def analyse():
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    out = {}
    d = pd.read_csv(os.path.join(HERE, "..", "data", "durations.csv"))
    m = pd.read_csv(os.path.join(HERE, "..", "data", "matches.csv"), dtype={"first": str, "second": str, "tier": str})
    m = m.merge(d, on="event_id", how="inner")
    m["post"] = (m.k >= 0).astype(int)
    m["total"] = m.total_s / 60          # kick-off to final whistle, interval included
    m["h2"] = m.h2_s / 60          # real length of the second half
    m["to_h2"] = (m.total_s - m.h2_s) / 60        # kick-off to the second-half kick-off
    ok_total = m.total.between(100, 135) & (m.last_change_is_status == 1)
    ok_h2 = m.h2.between(45, 70) & m.to_h2.between(55, 80) & (m.last_change_is_status == 1)
    m["ok_total"], m["ok_h2"] = ok_total, ok_h2
    out["coverage"] = dict(linked=int(len(m)), total_ok=int(ok_total.sum()), h2_ok=int(ok_h2.sum()),
                           total_ok_by_k=m.groupby("k").ok_total.mean().round(3).to_dict(),
                           h2_ok_by_k=m.groupby("k").ok_h2.mean().round(3).to_dict())
    print(f"linked matches {len(m)}; usable kick-off-to-whistle {int(ok_total.sum())} ({ok_total.mean():.1%}); usable second-half length {int(ok_h2.sum())} ({ok_h2.mean():.1%})")
    print("  share usable by season index, kick-off to whistle:", out["coverage"]["total_ok_by_k"])
    print("  share usable by season index, second-half length:  ", out["coverage"]["h2_ok_by_k"])

    # panel: leagues where the duration is usable for 80%+ of matches in every season
    t = m[ok_total]
    cnt = t.groupby(["league", "season"]).size().unstack()
    panel = [lg for lg in cnt.index if cnt.loc[lg].notna().sum() == 6 and (cnt.loc[lg].dropna() >= 50).all()]
    out["panel"] = panel
    p = t[t.league.isin(panel)].copy()
    print(f"\npanel: {len(panel)} leagues with 50+ usable matches in each of the six seasons; {len(p)} matches")
    by = p.groupby("k").agg(n=("total", "size"), median=("total", "median"), mean=("total", "mean"),
                            board=("board2", "mean"), board1=("board1", "mean"))
    # league-adjusted season effects on the duration (median regression is heavy; use league-demeaned means on trimmed data)
    kd = pd.get_dummies(p.k.astype(int), prefix="k", dtype=float)
    X = pd.concat([kd.drop(columns="k_-1"), pd.get_dummies(p.league, prefix="fe", dtype=float)], axis=1)
    fit = sm.OLS(p.total, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.league)[0]})
    print("   k     matches  median  mean   league-adjusted difference from 2022/23 (min)   second-half board  first-half board")
    out["by_season"] = {}
    for k, r in by.iterrows():
        diff = 0.0 if k == -1 else float(fit.params[f"k_{k}"])
        se = 0.0 if k == -1 else float(fit.bse[f"k_{k}"])
        out["by_season"][int(k)] = dict(n=int(r.n), median=float(r["median"]), mean=float(r["mean"]), diff=diff, se=se,
                                        board2=float(r.board), board1=float(r.board1))
        print(f"  {k:+d}   {int(r.n):>7}   {r['median']:.1f}   {r['mean']:.1f}   {diff:+.2f} ({se:.2f})                                   {r.board:.2f}              {r.board1:.2f}")
    Xp = pd.concat([p[["post"]].astype(float), pd.get_dummies(p.league, prefix="fe", dtype=float)], axis=1)
    fp = sm.OLS(p.total, Xp).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.league)[0]})
    out["post_minus_pre_minutes"] = dict(diff=float(fp.params["post"]), se=float(fp.bse["post"]))
    print(f"  after against before, league effects: {fp.params['post']:+.2f} min (se {fp.bse['post']:.2f})")

    # does a minute on the board become a minute on the clock?
    q = p[p.board1.notna() & p.board2.notna()].copy()
    Xb = pd.concat([q[["board1", "board2"]].astype(float), pd.get_dummies(q.league, prefix="fe", dtype=float)], axis=1)
    fb = sm.OLS(q.total, Xb).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(q.league)[0]})
    out["minutes_per_board_minute"] = dict(first_half=float(fb.params["board1"]), first_half_se=float(fb.bse["board1"]),
                                           second_half=float(fb.params["board2"]), second_half_se=float(fb.bse["board2"]), n=int(len(q)))
    print(f"  real minutes per announced minute: first half {fb.params['board1']:.2f} (se {fb.bse['board1']:.2f}), second half {fb.params['board2']:.2f} (se {fb.bse['board2']:.2f}); n={len(q)}")

    # the second half's real length against its board (seasons where the second-half start survives)
    h = m[ok_h2 & m.board2.notna()].copy()
    h["over"] = h.h2 - 45 - h.board2
    if len(h):
        g = h.groupby("k").agg(n=("over", "size"), h2=("h2", "median"), board=("board2", "mean"), over_mean=("over", "mean"), over_median=("over", "median"))
        out["second_half_length"] = {int(k): dict(n=int(r.n), median_h2=float(r.h2), board=float(r.board), over_mean=float(r.over_mean), over_median=float(r.over_median)) for k, r in g.iterrows()}
        print("\nsecond half: real length minus 45 minus the board (played beyond the announced minimum)")
        print(g.round(2).to_string())
        if (h.post == 0).sum() > 500 and (h.post == 1).sum() > 500:
            keep = h.groupby("league").post.transform("nunique") == 2
            hh = h[keep]
            Xo = pd.concat([hh[["post"]].astype(float), pd.get_dummies(hh.league, prefix="fe", dtype=float)], axis=1)
            fo = sm.OLS(hh.over, Xo).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(hh.league)[0]})
            out["over_post"] = dict(diff=float(fo.params["post"]), se=float(fo.bse["post"]), n=int(len(hh)), leagues=int(hh.league.nunique()))
            print(f"  beyond the board, after against before (league effects, {hh.league.nunique()} leagues, {len(hh)} matches): {fo.params['post']:+.2f} min (se {fo.bse['post']:.2f})")

    json.dump(out, open(os.path.join(HERE, "results_durations.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    analyse()
