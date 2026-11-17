#!/usr/bin/env python3
"""Step 3: the before/after models, from league_season.csv (aggregates only).

Poisson regressions of goal counts on league fixed effects, with matches as exposure and standard
errors clustered by league. A Poisson model with league effects and cell-level covariates gives the
same estimates from league-season cells as from single matches, so nothing per-match is needed.

Writes results_models.json and prints the tables.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import TIER_LABEL, TIER_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def fe_poisson(d, y, exposure, cols, fe="league"):
    """Poisson GLM of d[y] on d[cols] plus `fe` dummies, offset log(exposure), clustered by league."""
    d = d[(d[exposure] > 0)].copy()
    X = pd.concat([d[cols].astype(float), pd.get_dummies(d[fe], prefix="fe", dtype=float)], axis=1)
    groups = pd.factorize(d.league)[0]
    fit = sm.GLM(d[y].astype(float), X, family=sm.families.Poisson(), offset=np.log(d[exposure].astype(float))).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    return fit, d


def rr(fit, name):
    b, se = fit.params[name], fit.bse[name]
    return dict(rr=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)),
                b=float(b), se=float(se), p=float(fit.pvalues[name]))


def show(label, r):
    print(f"  {label:<46} {r['rr']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]")


def main():
    d = pd.read_csv(os.path.join(HERE, "league_season.csv"), dtype={"tier": str})
    d["s"] = d.s1 + d.s2
    d["other"] = d.goals - d.s
    d["post"] = (d.k >= 0).astype(float)
    for k in (-3, -2, 0, 1, 2):
        d[f"k{k}"] = (d.k == k).astype(float)
    OUT["sample"] = dict(matches=int(d.matches.sum()), leagues=int(d.league.nunique()), league_seasons=len(d),
                         matches_pre=int(d[d.post == 0].matches.sum()), matches_post=int(d[d.post == 1].matches.sum()))
    print(OUT["sample"])

    # ---- 1. raw rates and the plain (unclustered) ratio, for comparison with the September run
    t = d.groupby("post")[["matches", "goals", "s", "s1", "s2", "other", "goalless"]].sum()
    raw = {}
    for name in ("s", "s1", "s2", "other", "goals"):
        a, b = t.loc[0, name], t.loc[1, name]
        r = (b / t.loc[1, "matches"]) / (a / t.loc[0, "matches"])
        se = np.sqrt(1 / a + 1 / b)
        raw[name] = dict(pre=float(a / t.loc[0, "matches"]), post=float(b / t.loc[1, "matches"]), rr=float(r),
                         lo_plain=float(r * np.exp(-1.96 * se)), hi_plain=float(r * np.exp(1.96 * se)))
    raw["goalless_share"] = dict(pre=float(t.loc[0, "goalless"] / t.loc[0, "matches"]),
                                 post=float(t.loc[1, "goalless"] / t.loc[1, "matches"]))
    OUT["raw"] = raw
    print("\n1. per match, before -> after (plain Poisson interval)")
    for name, lab in (("s", "stoppage goals"), ("s1", "  first half"), ("s2", "  second half"),
                      ("other", "all other goals"), ("goals", "all goals")):
        r = raw[name]
        print(f"  {lab:<18} {r['pre']:.4f} -> {r['post']:.4f}   ratio {r['rr']:.3f} [{r['lo_plain']:.3f}, {r['hi_plain']:.3f}]")
    print(f"  goalless matches   {raw['goalless_share']['pre']:.2%} -> {raw['goalless_share']['post']:.2%}")

    # ---- 2. league fixed effects, clustered by league
    print("\n2. rate ratio after/before, league fixed effects, clustered by league")
    OUT["fe"] = {}
    for name, lab in (("s", "stoppage goals"), ("s1", "first-half stoppage goals"), ("s2", "second-half stoppage goals"),
                      ("other", "all other goals (placebo)"), ("goals", "all goals")):
        fit, _ = fe_poisson(d, name, "matches", ["post"])
        OUT["fe"][name] = rr(fit, "post")
        show(lab, OUT["fe"][name])
    # dispersion of the main model (Pearson chi2 / df)
    fit, dd = fe_poisson(d, "s", "matches", ["post"])
    OUT["fe"]["s"]["pearson_dispersion"] = float(fit.pearson_chi2 / fit.df_resid)
    print(f"  Pearson dispersion of the stoppage-goal model: {fit.pearson_chi2 / fit.df_resid:.2f}")

    # ---- 3. event study, reference season k = -1
    print("\n3. event study (season index, reference -1 = 2022/23 or 2022)")
    kcols = ["k-3", "k-2", "k0", "k1", "k2"]
    OUT["event_study"] = {}
    for name in ("s", "s1", "s2", "other"):
        fit, _ = fe_poisson(d, name, "matches", kcols)
        es = {c: rr(fit, c) for c in kcols}
        # joint test of the two leads; and of a linear pre-trend (k-3 = 2 * k-2)
        w = fit.wald_test("(k-3 = 0), (k-2 = 0)", scalar=True)
        lin = fit.wald_test("(k-3 - 2*k-2 = 0)", scalar=True)
        es["leads_joint_p"] = float(w.pvalue)
        es["pretrend_linear_p"] = float(lin.pvalue)
        OUT["event_study"][name] = es
        print(f"  {name}: " + "  ".join(f"k={c[1:]}: {es[c]['rr']:.3f} [{es[c]['lo']:.3f},{es[c]['hi']:.3f}]" for c in kcols)
              + f"   leads jointly zero p={w.pvalue:.2g}")
    for cal in ("winter", "summer"):
        fit, _ = fe_poisson(d[d.calendar == cal], "s", "matches", kcols)
        OUT["event_study"][f"s_{cal}"] = {c: rr(fit, c) for c in kcols}
        print(f"  s, {cal} leagues: " + "  ".join(f"k={c[1:]}: {np.exp(fit.params[c]):.3f}" for c in kcols))

    # ---- 4. the 2023 step measured against the earlier drift
    print("\n4. the step in 2023 against the drift before it (stoppage goals)")
    fit, _ = fe_poisson(d, "s", "matches", kcols)
    b = fit.params
    V = fit.cov_params()
    # (a) against the season before: k0
    step_prev = rr(fit, "k0")
    # (b) against the pre-2023 line through k=-3..-1 (least squares on the three points 0-ref):
    #     slope per season from k-3 and k-2 with k-1 = 0:  fitted line through (-3,b3), (-2,b2), (-1,0)
    #     OLS slope = (b(-1) - b(-3)) / 2 = -b3/2 ; fitted value at -1 = mean - slope*... compute generally
    ks = np.array([-3.0, -2.0, -1.0])
    names = ["k-3", "k-2"]
    A = np.zeros((3, len(b)))
    A[0, list(b.index).index("k-3")] = 1
    A[1, list(b.index).index("k-2")] = 1          # row 2 (k=-1) is the reference: zero
    xc = ks - ks.mean()
    slope_w = xc / (xc ** 2).sum()                 # weights giving the OLS slope from the 3 points
    mean_w = np.ones(3) / 3
    out4 = {"vs_previous_season": step_prev}
    for kk in (0, 1, 2):
        # contrast: beta_k - [mean + slope * (kk - mean(ks))]
        c = np.zeros(len(b))
        c[list(b.index).index(f"k{kk}")] = 1
        c -= (mean_w + slope_w * (kk - ks.mean())) @ A
        est = float(c @ b.values)
        se = float(np.sqrt(c @ V.values @ c))
        out4[f"vs_trend_k{kk}"] = dict(rr=float(np.exp(est)), lo=float(np.exp(est - 1.96 * se)),
                                       hi=float(np.exp(est + 1.96 * se)), b=est, se=se)
    slope = float((slope_w @ A) @ b.values)
    slope_se = float(np.sqrt((slope_w @ A) @ V.values @ (slope_w @ A)))
    out4["pre_slope_per_season"] = dict(rr=float(np.exp(slope)), lo=float(np.exp(slope - 1.96 * slope_se)),
                                        hi=float(np.exp(slope + 1.96 * slope_se)))
    OUT["step_vs_trend"] = out4
    show("2023/24 against 2022/23", step_prev)
    show("drift per season before 2023", out4["pre_slope_per_season"])
    for kk in (0, 1, 2):
        show(f"season {kk:+d} against the extended pre-2023 line", out4[f"vs_trend_k{kk}"])

    # ---- 5. robustness of the before/after ratio
    print("\n5. robustness (stoppage goals, league fixed effects, clustered)")
    rob = {}
    variants = {
        "all seasons": d,
        "without 2020/21 and 2020": d[d.k != -3],
        "2022/23 against 2023/24 only": d[d.k.isin([-1, 0])],
        "winter-calendar leagues": d[d.calendar == "winter"],
        "summer-calendar leagues": d[d.calendar == "summer"],
        "men's senior leagues only": d[~d.tier.isin(["women", "youth"])],
        "leagues with all six seasons": d[d.groupby("league").k.transform("nunique") == 6],
    }
    for lab, sub in variants.items():
        fit, sd = fe_poisson(sub, "s", "matches", ["post"])
        rob[lab] = dict(rr(fit, "post"), matches=int(sd.matches.sum()), leagues=int(sd.league.nunique()))
        show(f"{lab} (n={rob[lab]['matches']})", rob[lab])
    OUT["robustness"] = rob

    # ---- 6. by tier
    print("\n6. by tier")
    bt = {}
    for t_ in TIER_ORDER:
        sub = d[d.tier == t_]
        if sub.league.nunique() < 2:
            continue
        fit, sd = fe_poisson(sub, "s", "matches", ["post"])
        tt = sub.groupby("post")[["matches", "s"]].sum()
        bt[t_] = dict(rr(fit, "post"), leagues=int(sub.league.nunique()), matches=int(sub.matches.sum()),
                      pre=float(tt.loc[0, "s"] / tt.loc[0, "matches"]), post=float(tt.loc[1, "s"] / tt.loc[1, "matches"]))
        print(f"  {TIER_LABEL[t_]:<24} leagues {bt[t_]['leagues']:>2}  {bt[t_]['pre']:.3f} -> {bt[t_]['post']:.3f}  "
              f"{bt[t_]['rr']:.3f} [{bt[t_]['lo']:.3f}, {bt[t_]['hi']:.3f}]")
    OUT["by_tier"] = bt

    # ---- 7. league by league, and how much leagues differ (random-effects meta-analysis)
    rows = []
    for lg, g in d.groupby("league"):
        a = g[g.post == 0][["matches", "s"]].sum()
        b_ = g[g.post == 1][["matches", "s"]].sum()
        if a.matches < 150 or b_.matches < 150 or a.s == 0 or b_.s == 0:
            continue
        lr = np.log((b_.s / b_.matches) / (a.s / a.matches))
        rows.append(dict(league=lg, tier=g.tier.iloc[0], pre_m=int(a.matches), post_m=int(b_.matches),
                         pre=float(a.s / a.matches), post=float(b_.s / b_.matches), log_rr=float(lr),
                         se=float(np.sqrt(1 / a.s + 1 / b_.s))))
    L = pd.DataFrame(rows)
    w = 1 / L.se ** 2
    mu_f = (w * L.log_rr).sum() / w.sum()
    Q = (w * (L.log_rr - mu_f) ** 2).sum()
    tau2 = max(0.0, (Q - (len(L) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (L.se ** 2 + tau2)
    mu_r = (wr * L.log_rr).sum() / wr.sum()
    se_r = np.sqrt(1 / wr.sum())
    L["shrunk"] = (L.log_rr / L.se ** 2 + mu_r / tau2) / (1 / L.se ** 2 + 1 / tau2) if tau2 > 0 else mu_r
    sig_up = int(((L.log_rr - 1.96 * L.se) > 0).sum())
    sig_down = int(((L.log_rr + 1.96 * L.se) < 0).sum())
    OUT["leagues"] = dict(n=len(L), up=int((L.log_rr > 0).sum()), down=int((L.log_rr <= 0).sum()),
                          significantly_up=sig_up, significantly_down=sig_down,
                          mean_rr=float(np.exp(mu_r)), lo=float(np.exp(mu_r - 1.96 * se_r)), hi=float(np.exp(mu_r + 1.96 * se_r)),
                          tau=float(np.sqrt(tau2)), Q=float(Q), Q_p=float(stats.chi2.sf(Q, len(L) - 1)),
                          I2=float(max(0, (Q - (len(L) - 1)) / Q)),
                          range_95=[float(np.exp(mu_r - 1.96 * np.sqrt(tau2))), float(np.exp(mu_r + 1.96 * np.sqrt(tau2)))],
                          down_leagues=L[L.log_rr <= 0].league.tolist())
    L.to_csv(os.path.join(HERE, "league_effects.csv"), index=False)
    o = OUT["leagues"]
    print(f"\n7. league by league: {o['up']} of {o['n']} leagues up ({o['significantly_up']} with an interval above 1, "
          f"{o['significantly_down']} below); random-effects mean {o['mean_rr']:.3f} [{o['lo']:.3f}, {o['hi']:.3f}]; "
          f"between-league SD of the log ratio {o['tau']:.3f} (I2 {o['I2']:.0%}); 95% of leagues between "
          f"{o['range_95'][0]:.2f} and {o['range_95'][1]:.2f}; down: {o['down_leagues']}")

    # ---- 8. where the break falls: each league's own data
    print("\n8. the break season located by the data")
    cand = [-2, -1, 0, 1, 2]
    best = {}
    for lg, g in d.groupby("league"):
        if g.k.nunique() < 6:
            continue
        ll = {}
        for c in cand:
            ll_c = 0.0
            for part in (g[g.k < c], g[g.k >= c]):
                lam = part.s.sum() / part.matches.sum()
                ll_c += float((part.s * np.log(lam * part.matches) - lam * part.matches).sum()) if lam > 0 else 0.0
            ll[c] = ll_c
        best[lg] = max(ll, key=ll.get)
    counts = pd.Series(best).value_counts().reindex(cand, fill_value=0)
    pooled = {}
    for c in cand:
        d["step"] = (d.k >= c).astype(float)
        fit, _ = fe_poisson(d, "s", "matches", ["step"])
        pooled[c] = float(fit.deviance)
    OUT["break"] = dict(per_league_best=counts.to_dict(), leagues=len(best),
                        pooled_deviance={str(k): v for k, v in pooled.items()},
                        pooled_best=int(min(pooled, key=pooled.get)))
    print("  best single break per league (season index: leagues):", counts.to_dict(), f"of {len(best)}")
    print("  pooled deviance by break:", {k: round(v, 1) for k, v in pooled.items()}, "-> best", OUT["break"]["pooled_best"])

    # ---- 9. results settled in stoppage time
    print("\n9. results settled late (share of matches, league fixed effects)")
    OUT["settled"] = {}
    for name, lab in (("any_s", "a stoppage goal in the match"), ("last_s2", "last goal in second-half stoppage"),
                      ("late_eq_draw", "draw sealed by a stoppage equaliser"),
                      ("only_goal_s2", "1:0 with the goal in stoppage time"),
                      ("margin1_last_s2", "one-goal margin, last goal in stoppage"), ("draws", "draws (all)")):
        fit, _ = fe_poisson(d, name, "matches", ["post"])
        tt = d.groupby("post")[["matches", name]].sum()
        r = dict(rr(fit, "post"), pre=float(tt.loc[0, name] / tt.loc[0, "matches"]), post=float(tt.loc[1, name] / tt.loc[1, "matches"]))
        OUT["settled"][name] = r
        print(f"  {lab:<40} {r['pre']:.2%} -> {r['post']:.2%}   {r['rr']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")

    # ---- a bound for recording: count every goal written at minute 90 as a stoppage-time goal, in
    # both periods. Where reports wrote "90" for 90+x and stopped doing so, this removes the gain.
    d["s_or_90"] = d.s1 + d.s2 + d.g_90
    OUT["recording_bound"] = {}
    for lab, sel in (("all", d.tier.notna()), ("4-6", d.tier == "4-6"), ("1-3", d.tier.isin(["1", "2", "3"]))):
        f, _ = fe_poisson(d[sel], "s_or_90", "matches", ["post"])
        g, _ = fe_poisson(d[sel], "g_90", "matches", ["post"])
        OUT["recording_bound"][lab] = dict(stoppage_or_minute_90=rr(f, "post"), minute_90=rr(g, "post"))
        show(f"stoppage or minute 90, tiers {lab}", OUT["recording_bound"][lab]["stoppage_or_minute_90"])
        show(f"minute 90 alone, tiers {lab}", OUT["recording_bound"][lab]["minute_90"])

    json.dump(OUT, open(os.path.join(HERE, "results_models.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
