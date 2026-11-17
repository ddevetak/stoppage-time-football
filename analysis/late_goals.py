#!/usr/bin/env python3
"""Step 6: what the incident sample adds (../data/inc_matches.csv, inc_goals.csv, matches.csv).

A. FC's goal minutes checked against SofaScore's, an independent source.
B. Who scores in second-half stoppage time (score state before the goal, home or away).
C. Results changed between 90:00 and the final whistle, and the points that moved.
D. What the announced added time is made of: board regressed on the half's events, before and after.
E. Goals scored after the announced minimum had run out.

The sample: every match with a stoppage goal in FC's minutes (probability 1) plus a 9% stratified
random sample of the matches without one that have a second-half board (weights 1 / p_select).
Writes results_incidents.json and prints the tables.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def wls_cluster(y, X, w, groups):
    return sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": groups})


def main():
    m = pd.read_csv(os.path.join(HERE, "..", "data", "matches.csv"), dtype={"first": str, "second": str, "tier": str})
    inc = pd.read_csv(os.path.join(HERE, "..", "data", "inc_matches.csv"), dtype={"tier": str})
    goals = pd.read_csv(os.path.join(HERE, "..", "data", "inc_goals.csv"))

    linked = m[m.event_id.notna()].copy()
    linked["post"] = (linked.k >= 0).astype(int)
    inc = inc[inc.group.isin(["stoppage_goal", "control"])]
    x = inc.merge(linked[["match_id", "event_id", "s1", "s2", "goals", "post", "board1", "board2", "ft_h", "ft_a"]],
                  on="event_id", how="inner", suffixes=("", "_fc"))
    x = x[(x.goal_sequence_ok == 1) & (x.has_extra_time == 0)].copy()
    x["w"] = 1 / x.p_select
    x["fc_s"] = x.s1 + x.s2
    x["ss_s"] = x.ss_s1 + x.ss_s2
    N = linked.groupby("post").size()
    OUT["sample"] = dict(linked_matches=int(len(linked)), linked_pre=int(N[0]), linked_post=int(N[1]),
                         fetched_in_sample=int(len(x)), stoppage_goal_matches=int((x.group == "stoppage_goal").sum()),
                         controls=int((x.group == "control").sum()))
    print(OUT["sample"])

    # ---- A. two sources for the same goals
    sg, ct = x[x.group == "stoppage_goal"], x[x.group == "control"]
    same_score = (x.ss_ft_h == x.ft_h) & (x.ss_ft_a == x.ft_a)
    A = dict(final_score_agrees=float(same_score.mean()),
             counts_agree_exactly=float(((x.s1 == x.ss_s1) & (x.s2 == x.ss_s2)).mean()),
             fc_has_goal_sofascore_none=float((sg.ss_s == 0).mean()),
             fc_none_sofascore_has_goal=float((ct.ss_s > 0).mean()))
    for p, lab in ((0, "pre"), (1, "post")):
        A[f"fc_has_goal_sofascore_none_{lab}"] = float((sg[sg.post == p].ss_s == 0).mean())
        A[f"fc_none_sofascore_has_goal_{lab}"] = float((ct[ct.post == p].ss_s > 0).mean())
    A["by_tier"] = {}
    for t, g in x.groupby("tier"):
        A["by_tier"][t] = {}
        for p, lab in ((0, "pre"), (1, "post")):
            gg = g[g.post == p]
            A["by_tier"][t][lab] = dict(
                n_goal=int((gg.group == "stoppage_goal").sum()), n_control=int((gg.group == "control").sum()),
                fc_goal_ss_none=float((gg[gg.group == "stoppage_goal"].ss_s == 0).mean()),
                fc_none_ss_goal=float((gg[gg.group == "control"].ss_s > 0).mean()) if (gg.group == "control").any() else None)
    # the headline from each source on one population: matches with a second-half board
    pop = x[x.board2.notna()]
    two = {}
    for p, lab in ((0, "pre"), (1, "post")):
        q = pop[pop.post == p]
        n = q.w.sum()
        two[lab] = dict(matches_weighted=float(n), fc=float((q.w * q.fc_s).sum() / n), sofascore=float((q.w * q.ss_s).sum() / n),
                        fc_s2=float((q.w * q.s2).sum() / n), sofascore_s2=float((q.w * q.ss_s2).sum() / n),
                        fc_s1=float((q.w * q.s1).sum() / n), sofascore_s1=float((q.w * q.ss_s1).sum() / n))
    two["ratio_fc"] = two["post"]["fc"] / two["pre"]["fc"]
    two["ratio_sofascore"] = two["post"]["sofascore"] / two["pre"]["sofascore"]
    A["two_sources"] = two
    OUT["validation"] = A
    print(f"\nA. FC against SofaScore, {len(x)} matches: final score agrees {A['final_score_agrees']:.2%}; stoppage-goal counts by half agree exactly {A['counts_agree_exactly']:.2%}")
    print(f"   FC has a stoppage goal, SofaScore none: {A['fc_has_goal_sofascore_none']:.2%} (before {A['fc_has_goal_sofascore_none_pre']:.2%}, after {A['fc_has_goal_sofascore_none_post']:.2%})")
    print(f"   FC has none, SofaScore has one: {A['fc_none_sofascore_has_goal']:.2%} (before {A['fc_none_sofascore_has_goal_pre']:.2%}, after {A['fc_none_sofascore_has_goal_post']:.2%})")
    for t, v in A["by_tier"].items():
        print(f"   tier {t}: " + "; ".join(f"{lab}: FC goal/SS none {v[lab]['fc_goal_ss_none']:.1%} (n={v[lab]['n_goal']}), FC none/SS goal "
                                         + (f"{v[lab]['fc_none_ss_goal']:.1%}" if v[lab]['fc_none_ss_goal'] is not None else "-") + f" (n={v[lab]['n_control']})" for lab in ("pre", "post")))
    print(f"   stoppage goals per match on matches with a board, FC {two['pre']['fc']:.4f} -> {two['post']['fc']:.4f} (ratio {two['ratio_fc']:.3f}); "
          f"SofaScore {two['pre']['sofascore']:.4f} -> {two['post']['sofascore']:.4f} (ratio {two['ratio_sofascore']:.3f})")

    # ---- B. who scores in second-half stoppage time (SofaScore's goals, all fetched matches of the sample)
    g2 = goals[(goals.half == 2) & (goals.stoppage == 1)].merge(x[["event_id", "post", "w", "tier", "board2", "ss_board2"]], on="event_id")
    g2["state"] = pd.cut(g2.scorer_margin_before, [-99, -2, -1, 0, 99], labels=["trailing by 2+", "trailing by 1", "level", "leading"])
    B = {}
    for p, lab in ((0, "pre"), (1, "post")):
        q = g2[g2.post == p]
        tot = q.w.sum()
        B[lab] = dict(goals_weighted=float(tot), per_1000_matches=float(1000 * tot / N[p]),
                      by_state={s: dict(share=float(q[q.state == s].w.sum() / tot), per_1000_matches=float(1000 * q[q.state == s].w.sum() / N[p]))
                                for s in ["trailing by 2+", "trailing by 1", "level", "leading"]},
                      home_share=float(q[q.side == "home"].w.sum() / tot),
                      penalty_share=float(q[q.cls == "penalty"].w.sum() / tot),
                      own_goal_share=float(q[q.cls == "ownGoal"].w.sum() / tot))
    OUT["who_scores"] = B
    print("\nB. second-half stoppage goals by the scorer's position before the goal (share; per 1,000 matches)")
    for s in ["trailing by 1", "level", "leading", "trailing by 2+"]:
        a, b = B["pre"]["by_state"][s], B["post"]["by_state"][s]
        print(f"   {s:<15} {a['share']:.1%} -> {b['share']:.1%}    {a['per_1000_matches']:.1f} -> {b['per_1000_matches']:.1f}  ({b['per_1000_matches'] / a['per_1000_matches'] - 1:+.0%})")
    print(f"   home share {B['pre']['home_share']:.1%} -> {B['post']['home_share']:.1%}; penalties {B['pre']['penalty_share']:.1%} -> {B['post']['penalty_share']:.1%}")

    # ---- C. results changed after 90:00 and the points that moved
    C = {}
    x["d_home"] = x.pts_home_ft - x.pts_home_90
    x["d_away"] = x.pts_away_ft - x.pts_away_90
    x["kind"] = np.select([(x.pts_home_90 == 1) & (x.result_changed_s2 == 1),
                           (x.pts_home_90 != 1) & (x.pts_home_ft == 1) & (x.result_changed_s2 == 1),
                           (x.result_changed_s2 == 1)], ["draw became a win", "win became a draw", "defeat became a win"], "")
    for p, lab in ((0, "pre"), (1, "post")):
        q = x[x.post == p]
        ch = q[q.result_changed_s2 == 1]
        C[lab] = dict(matches=int(N[p]), changed_share=float(ch.w.sum() / N[p]),
                      kinds={k: float(ch[ch.kind == k].w.sum() / N[p]) for k in ["draw became a win", "win became a draw", "defeat became a win"]},
                      points_moved_per_team_match=float(((ch.d_home.abs() + ch.d_away.abs()) * ch.w).sum() / (2 * N[p])),
                      points_gained_per_match=float(((ch.d_home.clip(lower=0) + ch.d_away.clip(lower=0)) * ch.w).sum() / N[p]),
                      home_net_points_per_match=float((ch.d_home * ch.w).sum() / N[p]),
                      away_net_points_per_match=float((ch.d_away * ch.w).sum() / N[p]))
    C["ratio_changed"] = C["post"]["changed_share"] / C["pre"]["changed_share"]
    OUT["results_changed"] = C
    print("\nC. result different at the final whistle from 90:00 (share of all linked matches)")
    print(f"   any change   {C['pre']['changed_share']:.2%} -> {C['post']['changed_share']:.2%}  (ratio {C['ratio_changed']:.3f}); one match in {1 / C['pre']['changed_share']:.1f} -> one in {1 / C['post']['changed_share']:.1f}")
    for k in ["draw became a win", "win became a draw", "defeat became a win"]:
        print(f"   {k:<20} {C['pre']['kinds'][k]:.2%} -> {C['post']['kinds'][k]:.2%}")
    print(f"   points moved per team per match {C['pre']['points_moved_per_team_match']:.4f} -> {C['post']['points_moved_per_team_match']:.4f}; over 34 matches "
          f"{34 * C['pre']['points_moved_per_team_match']:.2f} -> {34 * C['post']['points_moved_per_team_match']:.2f}")
    print(f"   net points after 90:00 per match: home {C['pre']['home_net_points_per_match']:+.4f} -> {C['post']['home_net_points_per_match']:+.4f}; "
          f"away {C['pre']['away_net_points_per_match']:+.4f} -> {C['post']['away_net_points_per_match']:+.4f}")

    # ---- D. what the board is made of
    D = {}
    for h in ("2", "1"):
        q = x[x[f"ss_board{h}"].notna() & (x.has_subs == 1) & (x.has_cards == 1)].copy()
        if h == "1":
            q = q[q.board2.notna()]          # the control frame was defined on the second-half board
        # the margin when the board goes up: referees add less to a match that is already decided
        margin = (q.h90 - q.a90).abs()
        cols = [f"ss_reg_goals_{h}", f"sub_stops_{h}", f"yellow_{h}", f"red_{h}", f"var_{h}", f"pens_{h}", f"injury_subs_{h}"]
        names = ["goal", "substitution stop", "yellow card", "red card", "VAR decision", "penalty", "injury substitution (extra)"]
        if h == "2":
            q["level_at_90"] = (margin == 0).astype(float)
            q["one_goal_at_90"] = (margin == 1).astype(float)
            q["two_goals_at_90"] = (margin == 2).astype(float)
            cols += ["level_at_90", "one_goal_at_90", "two_goals_at_90"]
            names += ["level at 90:00 (against 3+ goals)", "one goal in it at 90:00", "two goals in it at 90:00"]
        res = dict(n=int(len(q)), leagues=int(q.league.nunique()))
        for p, lab in ((0, "pre"), (1, "post")):
            qq = q[q.post == p]
            X = pd.concat([qq[cols].astype(float), pd.get_dummies(qq.league, prefix="fe", dtype=float)], axis=1)
            fit = wls_cluster(qq[f"ss_board{h}"].astype(float), X, qq.w, pd.factorize(qq.league)[0])
            means = {c: float(np.average(qq[c], weights=qq.w)) for c in cols}
            board = float(np.average(qq[f"ss_board{h}"], weights=qq.w))
            res[lab] = dict(n=int(len(qq)), board_mean=board,
                            seconds_per_event={nm: dict(sec=float(60 * fit.params[c]), se=float(60 * fit.bse[c])) for c, nm in zip(cols, names)},
                            events_per_match={nm: means[c] for c, nm in zip(cols, names)},
                            explained_minutes={nm: float(fit.params[c] * means[c]) for c, nm in zip(cols, names)},
                            r2=float(fit.rsquared))
            res[lab]["base_minutes"] = board - sum(res[lab]["explained_minutes"].values())
        # one model with everything interacted with the period (league effects too): the change in
        # each coefficient is then exactly after minus before, with a standard error clustered by league
        qq = q.copy()
        inter = []
        for c in cols:
            qq[c + "_post"] = qq[c] * qq.post
            inter.append(c + "_post")
        fe = pd.get_dummies(qq.league, prefix="fe", dtype=float)
        fe_post = fe.mul(qq.post.astype(float), axis=0).add_suffix("_post")
        fe_post = fe_post.loc[:, fe_post.sum() > 0]
        X = pd.concat([qq[cols + inter].astype(float), fe, fe_post], axis=1)
        fit = wls_cluster(qq[f"ss_board{h}"].astype(float), X, qq.w, pd.factorize(qq.league)[0])
        res["change_seconds"] = {nm: dict(sec=float(60 * fit.params[c + "_post"]), se=float(60 * fit.bse[c + "_post"]),
                                          p=float(fit.pvalues[c + "_post"])) for c, nm in zip(cols, names)}
        # the same question with one set of league effects: a single shift for the whole period
        X1 = pd.concat([qq[cols + inter + ["post"]].astype(float), fe], axis=1)
        fit1 = wls_cluster(qq[f"ss_board{h}"].astype(float), X1, qq.w, pd.factorize(qq.league)[0])
        res["post_shift_seconds"] = dict(sec=float(60 * fit1.params["post"]), se=float(60 * fit1.bse["post"]), p=float(fit1.pvalues["post"]))
        # the change in the mean board in three parts: more events at the old seconds each; more
        # seconds for the events of today; and the rest, time not tied to any counted event
        pre_s, post_s = res["pre"]["seconds_per_event"], res["post"]["seconds_per_event"]
        pre_n, post_n = res["pre"]["events_per_match"], res["post"]["events_per_match"]
        d_events = sum(pre_s[nm]["sec"] / 60 * (post_n[nm] - pre_n[nm]) for nm in names)
        d_price = sum((post_s[nm]["sec"] - pre_s[nm]["sec"]) / 60 * post_n[nm] for nm in names)
        change = res["post"]["board_mean"] - res["pre"]["board_mean"]
        res["decomposition"] = dict(board_change=change, from_more_events=float(d_events), from_more_seconds_per_event=float(d_price),
                                    from_the_base=float(change - d_events - d_price),
                                    from_more_time_per_event_and_base=float(change - d_events))
        D[h] = res
        print(f"\nD. half {h}: the board regressed on the half's events ({res['n']} matches, {res['leagues']} leagues; league effects; weighted)")
        print(f"   mean board {res['pre']['board_mean']:.2f} -> {res['post']['board_mean']:.2f} min;  R2 {res['pre']['r2']:.2f} / {res['post']['r2']:.2f}")
        print("   event                         seconds each: before -> after (change, p)        per match: before -> after")
        for nm in names:
            a, b, c = res["pre"]["seconds_per_event"][nm], res["post"]["seconds_per_event"][nm], res["change_seconds"][nm]
            print(f"   {nm:<34} {a['sec']:6.1f} ({a['se']:.1f}) -> {b['sec']:6.1f} ({b['se']:.1f})   ({c['sec']:+.1f}, p={c['p']:.3f})      "
                  f"{res['pre']['events_per_match'][nm]:.2f} -> {res['post']['events_per_match'][nm]:.2f}")
        print(f"   time not tied to a counted event: {res['pre']['base_minutes']:.2f} -> {res['post']['base_minutes']:.2f} min (shift in the pooled model {res['post_shift_seconds']['sec']:+.0f} s, p={res['post_shift_seconds']['p']:.3f})")
        dc = res["decomposition"]
        print(f"   change in the board {dc['board_change']:+.2f} min = {dc['from_more_events']:+.2f} from more events + {dc['from_more_seconds_per_event']:+.2f} from more seconds per event + {dc['from_the_base']:+.2f} not tied to a counted event")
    OUT["board_model"] = D

    # ---- D2. the same question season by season: did the events change, or the time given for them?
    q = x[x.ss_board2.notna() & (x.has_subs == 1) & (x.has_cards == 1)].copy()
    q = q.merge(m[["event_id", "k"]].rename(columns={"k": "k_fc"}), on="event_id", how="left")
    cnt = q.groupby(["league", "k_fc"]).size().unstack()
    keep = cnt.index[(cnt.notna().sum(axis=1) == 6) & (cnt.min(axis=1) >= 25)]
    q = q[q.league.isin(keep)].copy()
    G = pd.factorize(q.league)[0]
    seasons = [-3, -2, 0, 1, 2]
    kd = pd.get_dummies(q.k_fc.astype(int), prefix="k", dtype=float).drop(columns="k_-1")
    fe = pd.get_dummies(q.league, prefix="fe", dtype=float)

    def season_effects(col, controls=None):
        X = pd.concat(([q[controls].astype(float)] if controls else []) + [kd, fe], axis=1)
        f = wls_cluster(q[col].astype(float), X, q.w, G)
        return {int(k): dict(diff=float(f.params[f"k_{k}"]), se=float(f.bse[f"k_{k}"])) for k in seasons}

    margin = (q.h90 - q.a90).abs()
    for nm, v in (("lvl", 0), ("one", 1), ("two", 2)):
        q[nm] = (margin == v).astype(float)
    ev = ["ss_reg_goals_2", "sub_stops_2", "yellow_2", "red_2", "var_2", "pens_2", "injury_subs_2"]
    ref = q[q.k_fc == -1]
    S = dict(leagues=int(len(keep)), matches=int(len(q)),
             level_2022_23={c: float(np.average(ref[c], weights=ref.w)) for c in ["ss_board2", "subs_2"] + ev},
             board=season_effects("ss_board2"), board_given_events=season_effects("ss_board2", ev + ["lvl", "one", "two"]),
             events={c: season_effects(c) for c in ["subs_2"] + ev})
    OUT["board_by_season"] = S
    print(f"\nD2. second half, season by season against 2022/23 ({S['leagues']} leagues with all six seasons, {S['matches']} matches; league effects; weighted)")
    for lab, r in (("board (min)", S["board"]), ("board at given events and margin", S["board_given_events"]),
                   ("substitutions", S["events"]["subs_2"]), ("stoppages for substitutions", S["events"]["sub_stops_2"]),
                   ("injury substitutions", S["events"]["injury_subs_2"]), ("VAR decisions", S["events"]["var_2"]),
                   ("yellow cards", S["events"]["yellow_2"]), ("goals in minutes 46-90", S["events"]["ss_reg_goals_2"])):
        print(f"   {lab:<34}" + "  ".join(f"{k:+d}: {r[k]['diff']:+.3f} ({r[k]['se']:.3f})" for k in seasons))
    print("   levels in 2022/23:", {k_: round(v, 3) for k_, v in S["level_2022_23"].items()})

    # ---- E. goals after the announced minimum had run out
    e = g2[g2.ss_board2.notna()].copy()
    e["beyond"] = (e.added > e.ss_board2).astype(int)
    E = {}
    for p, lab in ((0, "pre"), (1, "post")):
        q = e[e.post == p]
        E[lab] = dict(goals=int(len(q)), beyond_share=float(np.average(q.beyond, weights=q.w)),
                      mean_added_minute=float(np.average(q.added, weights=q.w)), mean_board=float(np.average(q.ss_board2, weights=q.w)))
    OUT["beyond_board"] = E
    print(f"\nE. second-half stoppage goals scored after the announced minutes had run out: {E['pre']['beyond_share']:.1%} -> {E['post']['beyond_share']:.1%} "
          f"(mean added minute of a stoppage goal {E['pre']['mean_added_minute']:.2f} -> {E['post']['mean_added_minute']:.2f}; board in those matches {E['pre']['mean_board']:.2f} -> {E['post']['mean_board']:.2f})")

    json.dump(OUT, open(os.path.join(HERE, "results_incidents.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
