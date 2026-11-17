#!/usr/bin/env python3
"""Step 4: the mechanism. Do stoppage goals follow the announced added minutes?

From league_season.csv (aggregates only). The added-time board is the "+N" shown at the end of each
half (SofaScore, Stage 1). The panel is the set of leagues with a board for at least 80% of their
matches in every season (argument 1 changes the share) and with seasons on both sides of 2023.

Writes results_mechanism.json and league_dose.csv, and prints the tables.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from models import fe_poisson, rr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
import sys

# share of a league's matches that must carry a board in every season; 0.8 is the main panel
PANEL_MIN_SHARE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8


def panel_leagues(d, half):
    share = d[f"n_board{half}"] / d.matches
    ok = []
    for lg, g in d.assign(share=share).groupby("league"):
        if (g.share >= PANEL_MIN_SHARE).all() and (g.k < 0).any() and (g.k >= 0).any():
            ok.append(lg)
    return ok


def main():
    d = pd.read_csv(os.path.join(HERE, "league_season.csv"), dtype={"tier": str})
    d["post"] = (d.k >= 0).astype(float)
    for k in (-3, -2, 0, 1, 2):
        d[f"k{k}"] = (d.k == k).astype(float)
    kcols = ["k-3", "k-2", "k0", "k1", "k2"]

    # ---- A. coverage and the board by season
    OUT["coverage"] = {h: dict(matches_with_board=int(d[f"n_board{h}"].sum()), share=float(d[f"n_board{h}"].sum() / d.matches.sum()))
                       for h in ("1", "2")}
    panel = {h: panel_leagues(d, h) for h in ("1", "2")}
    OUT["panel"] = {h: dict(leagues=len(panel[h]), names=panel[h]) for h in ("1", "2")}
    print(f"board coverage: first half {OUT['coverage']['1']['share']:.1%}, second half {OUT['coverage']['2']['share']:.1%}; "
          f"panel leagues: first half {len(panel['1'])}, second half {len(panel['2'])}")

    spec = {"2": ("s2_with_board", "g_76_90_with_board2", "minutes 76-90"),
            "1": ("s1_with_board", "g_31_45_with_board1", "minutes 31-45")}
    OUT["by_season"] = {}
    for h in ("2", "1"):
        y, ref, ref_lab = spec[h]
        p = d[d.league.isin(panel[h])]
        OUT[f"panel_matches_{h}"] = int(p[f"n_board{h}"].sum())
        t = p.groupby("k")[[f"n_board{h}", f"sum_board{h}", y, ref]].sum()
        t["board"] = t[f"sum_board{h}"] / t[f"n_board{h}"]
        t["goals_pm"] = t[y] / t[f"n_board{h}"]
        t["per_1000_min"] = 1000 * t[y] / t[f"sum_board{h}"]
        t["ref_per_1000_min"] = 1000 * t[ref] / (15 * t[f"n_board{h}"])
        OUT["by_season"][h] = t[["board", "goals_pm", "per_1000_min", "ref_per_1000_min"]].round(4).reset_index().to_dict("records")
        print(f"\nA. half {h}, panel of {len(panel[h])} leagues ({int(p[f'n_board{h}'].sum())} matches with a board)")
        print("   k   board  stoppage goals/match  per 1,000 announced min   " + ref_lab + " per 1,000 min")
        for k, r in t.iterrows():
            print(f"  {k:+d}   {r.board:.2f}   {r.goals_pm:.4f}                {r.per_1000_min:.1f}                      {r.ref_per_1000_min:.1f}")
        tp = p.groupby("post")[[f"n_board{h}", f"sum_board{h}", y, ref]].sum()
        OUT[f"period_{h}"] = {("post" if q else "pre"): dict(
            board=float(tp.loc[q, f"sum_board{h}"] / tp.loc[q, f"n_board{h}"]),
            goals_pm=float(tp.loc[q, y] / tp.loc[q, f"n_board{h}"]),
            per_1000_min=float(1000 * tp.loc[q, y] / tp.loc[q, f"sum_board{h}"]),
            ref_per_1000_min=float(1000 * tp.loc[q, ref] / (15 * tp.loc[q, f"n_board{h}"]))) for q in (0.0, 1.0)}
        a, b = OUT[f"period_{h}"]["pre"], OUT[f"period_{h}"]["post"]
        print(f"  before -> after: board {a['board']:.2f} -> {b['board']:.2f} ({b['board'] / a['board'] - 1:+.1%}); goals/match "
              f"{a['goals_pm']:.4f} -> {b['goals_pm']:.4f} ({b['goals_pm'] / a['goals_pm'] - 1:+.1%}); per 1,000 min "
              f"{a['per_1000_min']:.1f} -> {b['per_1000_min']:.1f} ({b['per_1000_min'] / a['per_1000_min'] - 1:+.1%})")

    # ---- B. regressions on the panel
    OUT["models"] = {}
    for h in ("2", "1"):
        y = spec[h][0]
        p = d[d.league.isin(panel[h])].copy()
        p["n"] = p[f"n_board{h}"]
        p["mins"] = p[f"sum_board{h}"]
        p["log_board"] = np.log(p.mins / p.n)
        res = {}
        # (1) after/before in goals per match, same matches
        fit, _ = fe_poisson(p, y, "n", ["post"])
        res["goals_per_match_post"] = rr(fit, "post")
        # (2) after/before in the board itself (log-linear, league effects, clustered)
        X = pd.concat([p[["post"]], pd.get_dummies(p.league, prefix="fe", dtype=float)], axis=1)
        ols = sm.WLS(p.log_board, X, weights=p.n).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p.league)[0]})
        res["board_post"] = dict(rr=float(np.exp(ols.params["post"])), lo=float(np.exp(ols.conf_int().loc["post", 0])),
                                 hi=float(np.exp(ols.conf_int().loc["post", 1])))
        # (3) after/before in goals per announced minute
        fit, _ = fe_poisson(p, y, "mins", ["post"])
        res["per_minute_post"] = rr(fit, "post")
        fit, _ = fe_poisson(p, y, "mins", kcols)
        res["per_minute_event_study"] = {c: rr(fit, c) for c in kcols}
        res["per_minute_event_study"]["all_seasons_equal_p"] = float(
            fit.wald_test(", ".join(f"({c} = 0)" for c in kcols), scalar=True).pvalue)
        # (4) elasticity of goals to the board: league effects only; plus the after dummy; plus season effects
        fit, _ = fe_poisson(p, y, "n", ["log_board"])
        res["elasticity"] = dict(b=float(fit.params["log_board"]), se=float(fit.bse["log_board"]),
                                 p_equals_1=float(fit.wald_test("log_board = 1", scalar=True).pvalue))
        fit, _ = fe_poisson(p, y, "n", ["log_board", "post"])
        res["elasticity_with_post"] = dict(b=float(fit.params["log_board"]), se=float(fit.bse["log_board"]),
                                           p_equals_1=float(fit.wald_test("log_board = 1", scalar=True).pvalue),
                                           post=rr(fit, "post"))
        fit, _ = fe_poisson(p, y, "n", ["log_board"] + kcols)
        res["elasticity_two_way"] = dict(b=float(fit.params["log_board"]), se=float(fit.bse["log_board"]),
                                         p_equals_1=float(fit.wald_test("log_board = 1", scalar=True).pvalue),
                                         p_equals_0=float(fit.pvalues["log_board"]),
                                         seasons_jointly_zero_p=float(fit.wald_test(", ".join(f"({c} = 0)" for c in kcols), scalar=True).pvalue))
        OUT["models"][h] = res
        print(f"\nB. half {h} (league fixed effects, clustered by league; {len(p)} league-seasons)")
        print(f"  goals per match, after/before             {res['goals_per_match_post']['rr']:.3f} [{res['goals_per_match_post']['lo']:.3f}, {res['goals_per_match_post']['hi']:.3f}]")
        print(f"  announced minutes, after/before           {res['board_post']['rr']:.3f} [{res['board_post']['lo']:.3f}, {res['board_post']['hi']:.3f}]")
        print(f"  goals per announced minute, after/before  {res['per_minute_post']['rr']:.3f} [{res['per_minute_post']['lo']:.3f}, {res['per_minute_post']['hi']:.3f}]")
        es = res["per_minute_event_study"]
        print("  per-minute rate by season (ref -1): " + "  ".join(f"{c[1:]}: {es[c]['rr']:.3f} [{es[c]['lo']:.2f},{es[c]['hi']:.2f}]" for c in kcols)
              + f"   all equal p={es['all_seasons_equal_p']:.2f}")
        for key, lab in (("elasticity", "league effects"), ("elasticity_with_post", "league effects + after dummy"),
                         ("elasticity_two_way", "league + season effects")):
            e = res[key]
            extra = f"; after dummy {e['post']['rr']:.3f} [{e['post']['lo']:.3f}, {e['post']['hi']:.3f}]" if "post" in e else ""
            extra += f"; season effects jointly zero p={e['seasons_jointly_zero_p']:.2f}" if "seasons_jointly_zero_p" in e else ""
            print(f"  elasticity of goals to the board, {lab:<28} {e['b']:.2f} (se {e['se']:.2f}), test of 1: p={e['p_equals_1']:.2f}{extra}")

    # ---- C. league by league: change in the board against change in stoppage goals
    rows = []
    for h in ("2", "1"):
        y = spec[h][0]
        for lg, g in d[d.league.isin(panel[h])].groupby("league"):
            a, b = g[g.post == 0], g[g.post == 1]
            na, nb = a[f"n_board{h}"].sum(), b[f"n_board{h}"].sum()
            rows.append(dict(half=h, league=lg, tier=g.tier.iloc[0], n_pre=int(na), n_post=int(nb),
                             board_pre=a[f"sum_board{h}"].sum() / na, board_post=b[f"sum_board{h}"].sum() / nb,
                             goals_pre=a[y].sum() / na, goals_post=b[y].sum() / nb))
    L = pd.DataFrame(rows)
    L["d_board"] = L.board_post - L.board_pre
    L["d_goals"] = L.goals_post - L.goals_pre
    if PANEL_MIN_SHARE == 0.8:
        L.to_csv(os.path.join(HERE, "league_dose.csv"), index=False)
    OUT["dose"] = {}
    print("\nC. across leagues: change in the board against change in stoppage goals per match")
    for h in ("2", "1"):
        x = L[L.half == h]
        w = 1 / (1 / x.n_pre + 1 / x.n_post)
        X = sm.add_constant(x.d_board)
        fit = sm.WLS(x.d_goals, X, weights=w).fit(cov_type="HC1")
        pre_rate = OUT[f"period_{h}"]["pre"]["per_1000_min"] / 1000
        r_w = float(np.cov(x.d_board, x.d_goals, aweights=w)[0, 1] / np.sqrt(np.cov(x.d_board, aweights=w) * np.cov(x.d_goals, aweights=w)))
        OUT["dose"][h] = dict(leagues=len(x), corr=float(x.d_board.corr(x.d_goals)), corr_weighted=r_w,
                              slope=float(fit.params["d_board"]), slope_se=float(fit.bse["d_board"]),
                              intercept=float(fit.params["const"]), intercept_se=float(fit.bse["const"]),
                              pre_rate_per_minute=pre_rate,
                              board_up=int((x.d_board > 0).sum()), d_board_min=float(x.d_board.min()), d_board_max=float(x.d_board.max()))
        o = OUT["dose"][h]
        print(f"  half {h}: {o['leagues']} leagues, board up in {o['board_up']}; change from {o['d_board_min']:+.2f} to {o['d_board_max']:+.2f} min; "
              f"correlation {o['corr']:.2f} (weighted {o['corr_weighted']:.2f}); slope {o['slope']:.4f} (se {o['slope_se']:.4f}) goals per added minute, "
              f"against {o['pre_rate_per_minute']:.4f} per announced minute before 2023; intercept {o['intercept']:+.4f} (se {o['intercept_se']:.4f})")

    # ---- D. by stoppage minute, all leagues (FC's goal minutes)
    print("\nD. stoppage goals by minute, all leagues (per 1,000 matches; ratio with league fixed effects)")
    OUT["by_minute"] = {}
    t = d.groupby("post")[["matches", "s2_1_3", "s2_4_6", "s2_7p", "s1_1", "s1_2", "s1_3", "s1_4p", "g_45", "g_90", "g_41_44", "g_86_89", "g_76_90", "g_31_45"]].sum()
    for name, lab in (("s2_1_3", "90+1 to 90+3"), ("s2_4_6", "90+4 to 90+6"), ("s2_7p", "90+7 and later"),
                      ("s1_1", "45+1"), ("s1_2", "45+2"), ("s1_3", "45+3"), ("s1_4p", "45+4 and later"),
                      ("g_86_89", "minutes 86-89"), ("g_90", "minute 90 exactly"), ("g_41_44", "minutes 41-44"), ("g_45", "minute 45 exactly"),
                      ("g_76_90", "minutes 76-90"), ("g_31_45", "minutes 31-45")):
        fit, _ = fe_poisson(d, name, "matches", ["post"])
        r = dict(rr(fit, "post"), pre=float(1000 * t.loc[0, name] / t.loc[0, "matches"]), post=float(1000 * t.loc[1, name] / t.loc[1, "matches"]))
        OUT["by_minute"][name] = r
        print(f"  {lab:<18} {r['pre']:6.1f} -> {r['post']:6.1f}   {r['rr']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
    # the same minute-90 check by tier (a recording artefact would show as a fall after 2023)
    OUT["minute_90_by_tier"] = {}
    for tier, g in d.groupby("tier"):
        tt = g.groupby("post")[["matches", "g_90", "g_45"]].sum()
        OUT["minute_90_by_tier"][tier] = dict(g90_pre=float(1000 * tt.loc[0, "g_90"] / tt.loc[0, "matches"]), g90_post=float(1000 * tt.loc[1, "g_90"] / tt.loc[1, "matches"]),
                                              g45_pre=float(1000 * tt.loc[0, "g_45"] / tt.loc[0, "matches"]), g45_post=float(1000 * tt.loc[1, "g_45"] / tt.loc[1, "matches"]))
    print("  minute 90 exactly, per 1,000 matches by tier:", {k: f"{v['g90_pre']:.0f}->{v['g90_post']:.0f}" for k, v in OUT["minute_90_by_tier"].items()})
    print("  minute 45 exactly, per 1,000 matches by tier:", {k: f"{v['g45_pre']:.0f}->{v['g45_post']:.0f}" for k, v in OUT["minute_90_by_tier"].items()})

    OUT["panel_min_share"] = PANEL_MIN_SHARE
    name = "results_mechanism.json" if PANEL_MIN_SHARE == 0.8 else f"results_mechanism_{int(100 * PANEL_MIN_SHARE)}.json"
    json.dump(OUT, open(os.path.join(HERE, name), "w"), indent=1)


if __name__ == "__main__":
    main()
