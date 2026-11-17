#!/usr/bin/env python3
"""Step 7: checks of the weak points (../data/matches.csv, inc_matches.csv). Prints; writes results_checks.json.

1. Timing inside seasons: is the pre-2023 "drift" a drift, or early steps (after the 2022 World Cup)?
2. The same curve before and after: stoppage goals per match at each value of the board.
3. The goal rate minute by minute inside second-half stoppage time.
4. Close games and home teams: does the board depend on who needs a goal, before and after?
5. Is a missing board a short board?
6. How much of the rise in all goals is stoppage-time goals.
7. The distribution of the board.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def fe_ols(d, y, cols, w=None):
    X = pd.concat([d[cols].astype(float), pd.get_dummies(d.league, prefix="fe", dtype=float)], axis=1)
    groups = pd.factorize(d.league)[0]
    mod = sm.WLS(d[y].astype(float), X, weights=(w if w is not None else np.ones(len(d))))
    return mod.fit(cov_type="cluster", cov_kwds={"groups": groups})


def fe_pois(d, y, cols, exposure=None):
    X = pd.concat([d[cols].astype(float), pd.get_dummies(d.league, prefix="fe", dtype=float)], axis=1)
    groups = pd.factorize(d.league)[0]
    off = np.log(d[exposure].astype(float)) if exposure else None
    return sm.GLM(d[y].astype(float), X, family=sm.families.Poisson(), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": groups})


def main():
    m = pd.read_csv(os.path.join(HERE, "..", "data", "matches.csv"), dtype={"first": str, "second": str, "tier": str})
    m["date"] = pd.to_datetime(m.month)
    m["post"] = (m.k >= 0).astype(int)
    m["s"] = m.s1 + m.s2
    m["ym"] = m.date.dt.to_period("M")

    # ---------- 1. timing inside seasons
    print("1. TIMING INSIDE SEASONS")
    w = m[m.calendar == "winter"].copy()
    w["half"] = np.where(w.date.dt.month >= 7, "a", "b")       # a = July-December, b = January-June
    w["cell"] = w.season.str[2:4] + "/" + w.season.str[7:9] + w.half
    cells = sorted(w.cell.unique())
    ref = "22/23a"
    dums = pd.get_dummies(w.cell, dtype=float)
    cols = [c for c in cells if c != ref]
    wd = pd.concat([w, dums[cols]], axis=1)
    # (i) stoppage goals, every winter league, FC minutes
    fit = fe_pois(wd, "s", cols)
    t1 = {c: dict(rr=float(np.exp(fit.params[c])), lo=float(np.exp(fit.params[c] - 1.96 * fit.bse[c])),
                  hi=float(np.exp(fit.params[c] + 1.96 * fit.bse[c]))) for c in cols}
    t1[ref] = dict(rr=1.0, lo=1.0, hi=1.0)
    # (ii) the second-half board, leagues with a board
    wb = wd[wd.board2.notna()]
    fitb = fe_ols(wb, "board2", cols)
    base = float(wb[wb.cell == ref].board2.mean())
    t2 = {c: dict(diff=float(fitb.params[c]), se=float(fitb.bse[c])) for c in cols}
    t2[ref] = dict(diff=0.0, se=0.0)
    n = w.groupby("cell").size()
    print(f"   winter-calendar leagues; reference {ref} (August-December 2022, before and during the World Cup break); board there {base:.2f} min")
    print("   half-season   matches   stoppage goals (ratio)        second-half board (min, difference)")
    for c in cells:
        print(f"   {c:<10} {n[c]:>8}     {t1[c]['rr']:.3f} [{t1[c]['lo']:.3f}, {t1[c]['hi']:.3f}]        {t2[c]['diff']:+.2f} ({t2[c]['se']:.2f})")
    OUT["half_seasons_winter"] = dict(reference=ref, board_at_reference=base, goals=t1, board=t2, matches={k_: int(v) for k_, v in n.items()})
    # the step across the World Cup, against the same step in other seasons (difference in differences)
    wd["second_part"] = (wd.half == "b").astype(float)
    wd["wc_season"] = (wd.season == "2022-2023").astype(float)
    wd["wc_x_second"] = wd.second_part * wd.wc_season
    pre = wd[wd.k < 0]
    seas = pd.get_dummies(pre.season, dtype=float).iloc[:, 1:]
    pre2 = pd.concat([pre, seas], axis=1)
    f1 = fe_pois(pre2, "s", ["second_part", "wc_x_second"] + list(seas.columns))
    f2 = fe_ols(pre2[pre2.board2.notna()], "board2", ["second_part", "wc_x_second"] + list(seas.columns))
    OUT["world_cup_step"] = dict(
        goals_rr=float(np.exp(f1.params["wc_x_second"])), goals_lo=float(np.exp(f1.params["wc_x_second"] - 1.96 * f1.bse["wc_x_second"])),
        goals_hi=float(np.exp(f1.params["wc_x_second"] + 1.96 * f1.bse["wc_x_second"])),
        board_diff=float(f2.params["wc_x_second"]), board_se=float(f2.bse["wc_x_second"]),
        usual_second_part_board=float(f2.params["second_part"]), usual_second_part_goals_rr=float(np.exp(f1.params["second_part"])))
    o = OUT["world_cup_step"]
    print(f"   after the World Cup (January-June 2023) against the usual spring step of 2020/21 and 2021/22: board {o['board_diff']:+.2f} min (se {o['board_se']:.2f}); "
          f"stoppage goals x{o['goals_rr']:.3f} [{o['goals_lo']:.3f}, {o['goals_hi']:.3f}]  (usual spring step: board {o['usual_second_part_board']:+.2f}, goals x{o['usual_second_part_goals_rr']:.3f})")

    # monthly, league-adjusted second-half board, winter leagues with a board in every season
    cover = wb.groupby(["league", "season"]).size().unstack().notna().sum(axis=1)
    keep = cover[cover == 6].index
    mb = wb[wb.league.isin(keep)].copy()
    mb["adj"] = mb.board2 - mb.groupby("league").board2.transform("mean") + mb.board2.mean()
    mon = mb.groupby("ym").agg(board=("adj", "mean"), n=("adj", "size"))
    mon = mon[mon.n >= 300]
    OUT["monthly_board_winter"] = {str(k_): dict(board=float(v.board), n=int(v.n)) for k_, v in mon.iterrows()}
    print("   monthly second-half board, league-adjusted (months with 300+ matches):")
    line = []
    for k_, v in mon.iterrows():
        line.append(f"{str(k_)[2:]}:{v.board:.2f}")
    for i in range(0, len(line), 12):
        print("     " + "  ".join(line[i:i + 12]))

    s_ = m[m.calendar == "summer"].copy()
    s_["half"] = np.where(s_.date.dt.month >= 7, "b", "a")     # a = January-June, b = July-December
    s_["cell"] = s_.season + s_.half
    cells = sorted(s_.cell.unique())
    ref = "2022b"
    cols = [c for c in cells if c != ref]
    sd = pd.concat([s_, pd.get_dummies(s_.cell, dtype=float)[cols]], axis=1)
    fit = fe_pois(sd, "s", cols)
    sb = sd[sd.board2.notna()]
    fitb = fe_ols(sb, "board2", cols)
    print(f"   summer-calendar leagues; reference {ref} (July-December 2022); board there {sb[sb.cell == ref].board2.mean():.2f} min")
    OUT["half_seasons_summer"] = {}
    for c in cells:
        rr_ = 1.0 if c == ref else float(np.exp(fit.params[c]))
        df_ = 0.0 if c == ref else float(fitb.params[c])
        OUT["half_seasons_summer"][c] = dict(rr=rr_, board_diff=df_, matches=int((s_.cell == c).sum()))
        print(f"   {c:<8} {int((s_.cell == c).sum()):>6}   goals x{rr_:.3f}    board {df_:+.2f}")

    # ---------- 2. the same curve before and after
    print("\n2. STOPPAGE GOALS AT EACH VALUE OF THE BOARD (second half; all matches with a board)")
    b = m[m.board2.notna()].copy()
    b["bcat"] = b.board2.clip(upper=10).astype(int)
    tab = b.groupby(["bcat", "post"]).agg(n=("s2", "size"), g=("s2", "mean")).unstack()
    print("   board   matches before / after     goals per match before / after    per announced minute before / after")
    curve = {}
    for v in range(0, 11):
        if v not in tab.index:
            continue
        nb, na_, gb, ga = tab.loc[v, ("n", 0)], tab.loc[v, ("n", 1)], tab.loc[v, ("g", 0)], tab.loc[v, ("g", 1)]
        curve[v] = dict(n_pre=int(nb), n_post=int(na_), pre=float(gb), post=float(ga))
        pm = f"{1000 * gb / v:6.1f} / {1000 * ga / v:6.1f}" if v > 0 else "     -"
        print(f"   {('10+' if v == 10 else v)!s:>4}   {int(nb):>7} / {int(na_):<7}      {gb:.4f} / {ga:.4f}                {pm}")
    OUT["curve_by_board"] = curve
    bd = pd.get_dummies(b.bcat, prefix="b", dtype=float).iloc[:, 1:]
    bb = pd.concat([b, bd], axis=1)
    fit = fe_pois(bb, "s2", ["post"] + list(bd.columns))
    fit0 = fe_pois(bb, "s2", ["post"])
    bb["log_board"] = np.log(bb.board2.clip(lower=0.5))
    fitl = fe_pois(bb, "s2", ["post", "log_board"])
    OUT["match_level"] = dict(
        post_unconditional=float(np.exp(fit0.params["post"])),
        post_given_board=float(np.exp(fit.params["post"])), lo=float(np.exp(fit.params["post"] - 1.96 * fit.bse["post"])),
        hi=float(np.exp(fit.params["post"] + 1.96 * fit.bse["post"])),
        elasticity_match_level=float(fitl.params["log_board"]), elasticity_se=float(fitl.bse["log_board"]), n=int(len(bb)))
    o = OUT["match_level"]
    print(f"   match level, league effects ({o['n']} matches): after/before {o['post_unconditional']:.3f} without the board; "
          f"{o['post_given_board']:.3f} [{o['lo']:.3f}, {o['hi']:.3f}] at a given board; elasticity to the board across matches {o['elasticity_match_level']:.2f} (se {o['elasticity_se']:.2f})")

    # ---------- 3. minute by minute inside second-half stoppage time
    print("\n3. GOAL RATE BY ADDED MINUTE (goals in minute 90+x per 1,000 matches whose board reaches x)")
    sec = b.second.fillna("").map(lambda s: [int(x) for x in s.split()] if s else [])
    haz = {}
    for x in range(1, 9):
        reach = b.board2 >= x
        g = sec.map(lambda t: sum(1 for u in t if u == 90 + x))
        r = {}
        for p in (0, 1):
            sel = reach & (b.post == p)
            r[p] = (int(sel.sum()), float(1000 * g[sel].mean()))
        haz[x] = dict(n_pre=r[0][0], pre=r[0][1], n_post=r[1][0], post=r[1][1])
        print(f"   90+{x}:  before {r[0][1]:5.1f} (n={r[0][0]})   after {r[1][1]:5.1f} (n={r[1][0]})")
    g_all = sec.map(lambda t: sum(1 for u in t if u > 90))
    g_in = pd.Series([sum(1 for u in t if 90 < u <= 90 + bd_) for t, bd_ in zip(sec, b.board2)], index=b.index)
    for p, lab in ((0, "before"), (1, "after")):
        sel = b.post == p
        haz[f"beyond_share_{lab}"] = float(1 - g_in[sel].sum() / g_all[sel].sum())
        haz[f"per_1000_announced_within_{lab}"] = float(1000 * g_in[sel].sum() / b.board2[sel].sum())
    OUT["hazard"] = haz
    print(f"   goals later than the board allows: {haz['beyond_share_before']:.1%} before, {haz['beyond_share_after']:.1%} after; "
          f"goals inside the announced minutes per 1,000 of them: {haz['per_1000_announced_within_before']:.1f} -> {haz['per_1000_announced_within_after']:.1f}")

    # ---------- 4. who needs a goal
    print("\n4. THE BOARD BY THE SCORE AT 90:00 (incident sample, weighted; seconds against a level match)")
    inc = pd.read_csv(os.path.join(HERE, "..", "data", "inc_matches.csv"), dtype={"tier": str})
    inc = inc[inc.group.isin(["stoppage_goal", "control"]) & (inc.goal_sequence_ok == 1) & (inc.has_extra_time == 0) & inc.ss_board2.notna()]
    inc = inc.merge(m[["event_id", "post"]], on="event_id")
    inc["wt"] = 1 / inc.p_select
    inc["hm"] = (inc.h90 - inc.a90).clip(-3, 3)
    fav = {}
    names = {-3: "home 3+ behind", -2: "home 2 behind", -1: "home 1 behind", 1: "home 1 ahead", 2: "home 2 ahead", 3: "home 3+ ahead"}
    for v in names:
        inc[f"hm_{v}"] = (inc.hm == v).astype(float)
    ctrl = ["ss_reg_goals_2", "sub_stops_2", "yellow_2", "red_2", "var_2", "pens_2", "injury_subs_2"]
    for p, lab in ((0, "before"), (1, "after")):
        q = inc[(inc.post == p) & (inc.has_subs == 1) & (inc.has_cards == 1)]
        fit = fe_ols(q, "ss_board2", [f"hm_{v}" for v in names] + ctrl, w=q.wt)
        fav[lab] = {names[v]: dict(sec=float(60 * fit.params[f"hm_{v}"]), se=float(60 * fit.bse[f"hm_{v}"])) for v in names}
        t = fit.t_test("hm_-1 - hm_1 = 0")
        fav[lab]["home_behind_minus_home_ahead_by_one"] = dict(sec=float(60 * t.effect[0]), se=float(60 * t.sd[0][0]), p=float(t.pvalue), n=int(len(q)))
    OUT["favouritism"] = fav
    print("   score at 90:00        before            after")
    for v in names:
        a, c = fav["before"][names[v]], fav["after"][names[v]]
        print(f"   {names[v]:<16}  {a['sec']:+6.1f} ({a['se']:.1f})   {c['sec']:+6.1f} ({c['se']:.1f})")
    a, c = fav["before"]["home_behind_minus_home_ahead_by_one"], fav["after"]["home_behind_minus_home_ahead_by_one"]
    print(f"   home one behind minus home one ahead: {a['sec']:+.1f} s (se {a['se']:.1f}, p={a['p']:.3f}, n={a['n']}) before; {c['sec']:+.1f} s (se {c['se']:.1f}, p={c['p']:.3f}, n={c['n']}) after")

    # ---------- 5. is a missing board a short board
    print("\n5. MATCHES WITHOUT A BOARD")
    mm = m.copy()
    mm["has2"] = mm.board2.notna().astype(float)
    mm["has1"] = mm.board1.notna().astype(float)
    mix = mm.groupby(["league", "season"]).has2.transform("mean")
    part = mm[(mix > 0.2) & (mix < 0.95)].copy()
    part["ls"] = part.league + part.season
    X = pd.concat([part[["has2"]], pd.get_dummies(part.ls, prefix="fe", dtype=float)], axis=1)
    fit = sm.GLM(part.s2.astype(float), X, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(part.league)[0]})
    both = mm[mm.board2.notna()].copy()
    both["ls"] = both.league + both.season
    X1 = pd.concat([both[["has1"]], pd.get_dummies(both.ls, prefix="fe", dtype=float)], axis=1)
    fit1 = sm.GLM(both.s1.astype(float), X1, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(both.league)[0]})
    OUT["missing_board"] = dict(
        s2_with_vs_without_board2=float(np.exp(fit.params["has2"])), lo2=float(np.exp(fit.params["has2"] - 1.96 * fit.bse["has2"])),
        hi2=float(np.exp(fit.params["has2"] + 1.96 * fit.bse["has2"])), n2=int(len(part)),
        s1_with_vs_without_board1=float(np.exp(fit1.params["has1"])), lo1=float(np.exp(fit1.params["has1"] - 1.96 * fit1.bse["has1"])),
        hi1=float(np.exp(fit1.params["has1"] + 1.96 * fit1.bse["has1"])), share_board1_missing_given_board2=float(1 - both.has1.mean()),
        zero_board1=int((m.board1 == 0).sum()), zero_board2=int((m.board2 == 0).sum()))
    o = OUT["missing_board"]
    print(f"   second-half stoppage goals, matches with a board against matches without, same league-season ({o['n2']} matches in partly covered league-seasons): "
          f"x{o['s2_with_vs_without_board2']:.3f} [{o['lo2']:.3f}, {o['hi2']:.3f}]")
    print(f"   first-half stoppage goals, with a first-half board against without (among matches with a second-half board; {o['share_board1_missing_given_board2']:.1%} lack it): "
          f"x{o['s1_with_vs_without_board1']:.3f} [{o['lo1']:.3f}, {o['hi1']:.3f}]; boards of zero: first half {o['zero_board1']}, second half {o['zero_board2']}")

    # ---------- 6. share of the rise in all goals
    t = m.groupby("post").agg(n=("match_id", "size"), goals=("goals", "sum"), s=("s", "sum"))
    d_all = t.goals[1] / t.n[1] - t.goals[0] / t.n[0]
    d_s = t.s[1] / t.n[1] - t.s[0] / t.n[0]
    OUT["share_of_rise"] = dict(all_goals_change=float(d_all), stoppage_change=float(d_s), share=float(d_s / d_all))
    print(f"\n6. goals per match rose by {d_all:.3f}; stoppage goals by {d_s:.3f}: {d_s / d_all:.0%} of the rise")

    # ---------- 7. the distribution of the board
    print("\n7. THE DISTRIBUTION OF THE BOARD (all matches with a board)")
    dist = {}
    for h in ("2", "1"):
        bq = m[m[f"board{h}"].notna()]
        r = {}
        for p, lab in ((0, "before"), (1, "after")):
            x = bq[bq.post == p][f"board{h}"]
            r[lab] = dict(n=int(len(x)), mean=float(x.mean()), median=float(x.median()),
                          **{f"share_ge_{c}": float((x >= c).mean()) for c in ((6, 8, 10) if h == "2" else (3, 5, 7))})
        dist[h] = r
        ks = [k_ for k_ in r["before"] if k_.startswith("share")]
        print(f"   half {h}: mean {r['before']['mean']:.2f} -> {r['after']['mean']:.2f}; median {r['before']['median']:.0f} -> {r['after']['median']:.0f}; "
              + "; ".join(f"{k_.replace('share_ge_', '')}+ min: {r['before'][k_]:.1%} -> {r['after'][k_]:.1%}" for k_ in ks))
    OUT["board_distribution"] = dist

    json.dump(OUT, open(os.path.join(HERE, "results_checks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
