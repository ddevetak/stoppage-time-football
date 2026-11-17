#!/usr/bin/env python3
"""Step 9: what stoppage time decides. The score at 90:00 for every match, then two questions.

The score when second-half stoppage time begins is the final score when FC's minutes show no
second-half stoppage goal, and comes from the incident sample (which holds every match with one)
otherwise. Matches where neither is known are dropped and counted.

A. How safe is a lead at 90:00, by the length of the board, before and after?
B. League tables at 90:00 against the final tables: first place, bottom places, points moved.

Reads ../data/matches.csv and ../data/inc_matches.csv. Writes results_decides.json and, for the
tables, ../data/tables_at_90.csv (club level). Prints.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def pts(a, b):
    return np.where(a > b, 3, np.where(a == b, 1, 0))


def score_at_90():
    """The match table with the score at 90:00 (h90x, a90x; missing where it is not known)."""
    m = pd.read_csv(os.path.join(HERE, "..", "data", "matches.csv"), dtype={"first": str, "second": str, "tier": str})
    inc = pd.read_csv(os.path.join(HERE, "..", "data", "inc_matches.csv"))
    inc = inc[(inc.goal_sequence_ok == 1) & (inc.has_extra_time == 0)][["event_id", "h90", "a90", "ss_ft_h", "ss_ft_a"]]
    m = m.merge(inc, on="event_id", how="left")
    m["post"] = (m.k >= 0).astype(int)
    same_final = (m.ss_ft_h == m.ft_h) & (m.ss_ft_a == m.ft_a)
    m["h90x"] = np.where(m.s2 == 0, m.ft_h, np.where(same_final, m.h90, np.nan))
    m["a90x"] = np.where(m.s2 == 0, m.ft_a, np.where(same_final, m.a90, np.nan))
    return m


def main():
    m = score_at_90()
    known = m.h90x.notna()
    OUT["known_state"] = dict(matches=int(len(m)), known=int(known.sum()), share=float(known.mean()),
                              unknown_by_period={("post" if p else "pre"): int((~known & (m.post == p)).sum()) for p in (0, 1)})
    print(f"score at 90:00 known for {known.sum()} of {len(m)} matches ({known.mean():.2%}); unknown: {OUT['known_state']['unknown_by_period']}")
    x = m[known].copy()
    x["lead90"] = (x.h90x - x.a90x).abs()
    x["leader_home"] = x.h90x > x.a90x
    fin = x.ft_h - x.ft_a
    x["leader_won"] = np.where(x.leader_home, fin > 0, fin < 0)
    x["leader_lost"] = np.where(x.leader_home, fin < 0, fin > 0)
    x["winner_after_90"] = (x.lead90 == 0) & (fin != 0)

    # ---------- A. how safe is a lead
    print("\nA. LEADS AT 90:00")
    A = {}
    for lab, sel, col in (("one-goal lead not held", x.lead90 == 1, "not_held"), ("two-goal lead not held", x.lead90 == 2, "not_held"),
                          ("level match gets a winner", x.lead90 == 0, "winner_after_90")):
        q = x[sel].copy()
        q["not_held"] = (~q.leader_won).astype(int)
        q["y"] = q[col].astype(int)
        r = {}
        for p, pl in ((0, "pre"), (1, "post")):
            qq = q[q.post == p]
            r[pl] = dict(n=int(len(qq)), share=float(qq.y.mean()))
        X = pd.concat([q[["post"]].astype(float), pd.get_dummies(q.league, prefix="fe", dtype=float)], axis=1)
        f = sm.OLS(q.y.astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(q.league)[0]})
        r["post_diff_pp"] = float(100 * f.params["post"])
        r["post_diff_se"] = float(100 * f.bse["post"])
        b = q[q.board2.notna()]
        Xb = pd.concat([b[["board2"]].astype(float), pd.get_dummies(b.league, prefix="fe", dtype=float)], axis=1)
        fb = sm.OLS(b.y.astype(float), Xb).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(b.league)[0]})
        Xc = pd.concat([b[["board2", "post"]].astype(float), pd.get_dummies(b.league, prefix="fe", dtype=float)], axis=1)
        fc = sm.OLS(b.y.astype(float), Xc).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(b.league)[0]})
        r["per_board_minute_pp"] = float(100 * fb.params["board2"])
        r["per_board_minute_se"] = float(100 * fb.bse["board2"])
        r["post_given_board_pp"] = float(100 * fc.params["post"])
        r["post_given_board_se"] = float(100 * fc.bse["post"])
        r["by_board"] = {}
        b = b.assign(bc=b.board2.clip(lower=2, upper=9).astype(int))
        for v, g in b.groupby("bc"):
            r["by_board"][int(v)] = dict(n=int(len(g)), share=float(g.y.mean()))
        A[lab] = r
        print(f"  {lab}: {r['pre']['share']:.2%} (n={r['pre']['n']}) -> {r['post']['share']:.2%} (n={r['post']['n']}); "
              f"after-before {r['post_diff_pp']:+.2f} pp (se {r['post_diff_se']:.2f}); per announced minute {r['per_board_minute_pp']:+.2f} pp (se {r['per_board_minute_se']:.2f}); "
              f"after-before at a given board {r['post_given_board_pp']:+.2f} pp (se {r['post_given_board_se']:.2f})")
        print("     by board (2 or less ... 9 or more): " + "  ".join(f"{v}:{d['share']:.1%}" for v, d in r["by_board"].items()))
    q = x[x.lead90 == 1]
    for p, pl in ((0, "pre"), (1, "post")):
        qq = q[q.post == p]
        A[f"one_goal_lead_{pl}"] = dict(drawn=float(((qq.ft_h - qq.ft_a) == 0).mean()), lost=float(qq.leader_lost.mean()), extended=float(((qq.ft_h - qq.ft_a).abs() >= 2).mean()))
    print(f"  one-goal lead at 90:00 ends level {A['one_goal_lead_pre']['drawn']:.2%} -> {A['one_goal_lead_post']['drawn']:.2%}; ends in defeat {A['one_goal_lead_pre']['lost']:.2%} -> {A['one_goal_lead_post']['lost']:.2%}; "
          f"is extended {A['one_goal_lead_pre']['extended']:.2%} -> {A['one_goal_lead_post']['extended']:.2%}")
    sh = x.groupby("post").apply(lambda g: pd.Series(dict(level=(g.lead90 == 0).mean(), one=(g.lead90 == 1).mean(), two=(g.lead90 == 2).mean())))
    A["state_shares"] = {("post" if p else "pre"): sh.loc[p].round(4).to_dict() for p in (0, 1)}
    print("  share of matches level / one goal / two goals in it at 90:00:", A["state_shares"])
    OUT["leads"] = A

    # ---------- B. tables at 90:00 against the final tables
    print("\nB. LEAGUE TABLES AT 90:00 AGAINST THE FINAL TABLES")
    # all regular-season results, including the few the minute rule drops; where the 90:00 score
    # is unknown the final score stands in, and the league-season is used only if that is rare
    from common import REL, TIER
    ra = pd.read_csv(os.path.join(HERE, "..", "data", "results_all.csv"))
    ra = ra.merge(m[["match_id", "h90x", "a90x"]], on="match_id", how="left")
    ra["tier"] = ra.league.map(TIER)
    ra["k"] = ra.season.astype(str).map(REL)
    ra["unknown"] = ra.h90x.isna()
    comp = ra.groupby(["league", "season"]).agg(n=("match_id", "size"), unknown=("unknown", "sum"))
    full = comp[comp.unknown <= 2].index
    OUT["tables_input"] = dict(results=int(len(ra)), unknown_90=int(ra.unknown.sum()), league_seasons=int(len(comp)),
                               league_seasons_used=int(len(full)), with_no_unknown=int((comp.unknown == 0).sum()))
    t = ra.set_index(["league", "season"]).loc[full].reset_index()
    unk = t.groupby(["league", "season"]).unknown.sum().rename("n_unknown")
    t["h90x"] = t.h90x.fillna(t.ft_h)
    t["a90x"] = t.a90x.fillna(t.ft_a)
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        gf, ga = (t.ft_h, t.ft_a) if side == "home" else (t.ft_a, t.ft_h)
        gf9, ga9 = (t.h90x, t.a90x) if side == "home" else (t.a90x, t.h90x)
        rows.append(pd.DataFrame(dict(league=t.league, season=t.season, tier=t.tier, k=t.k, club=t[side], played=1,
                                      pts=pts(gf, ga), gd=gf - ga, gf=gf, pts90=pts(gf9, ga9), gd90=gf9 - ga9, gf90=gf9)))
    c = pd.concat(rows).groupby(["league", "season", "tier", "k", "club"]).sum().reset_index()
    out_rows, per_ls = [], []
    for (lg, se), g in c.groupby(["league", "season"]):
        # a table needs a round-robin: keep league-seasons where clubs played a similar number of games
        if g.played.min() < 0.8 * g.played.max() or len(g) < 8:
            continue
        g = g.copy()
        g["rank"] = g.sort_values(["pts", "gd", "gf"], ascending=False).assign(r=lambda d: np.arange(1, len(d) + 1)).r
        g["rank90"] = g.sort_values(["pts90", "gd90", "gf90"], ascending=False).assign(r=lambda d: np.arange(1, len(d) + 1)).r
        g["d_pts"] = g.pts - g.pts90
        n = len(g)
        first, first90 = g.loc[g["rank"] == 1, "club"].iloc[0], g.loc[g.rank90 == 1, "club"].iloc[0]
        bottom = lambda col, z: set(g.loc[g[col] > n - z, "club"])
        per_ls.append(dict(league=lg, season=se, tier=g.tier.iloc[0], k=int(g.k.iloc[0]), clubs=n, games=int(g.played.max()),
                           n_unknown=int(unk.get((lg, se), 0)),
                           first=first, first_at_90=first90, first_changed=int(first != first90),
                           bottom2_changed=int(bottom("rank", 2) != bottom("rank90", 2)),
                           bottom3_changed=int(bottom("rank", 3) != bottom("rank90", 3)),
                           top4_changed=int(set(g.loc[g["rank"] <= 4, "club"]) != set(g.loc[g.rank90 <= 4, "club"])),
                           clubs_moved=int((g["rank"] != g.rank90).sum()), mean_abs_rank_move=float((g["rank"] - g.rank90).abs().mean()),
                           mean_abs_pts=float(g.d_pts.abs().mean()), max_gain=int(g.d_pts.max()), max_loss=int(g.d_pts.min()),
                           gap_first=int(g.sort_values("rank").pts.iloc[0] - g.sort_values("rank").pts.iloc[1])))
        out_rows.append(g)
    T = pd.DataFrame(per_ls)
    T["post"] = (T.k >= 0).astype(int)
    pd.concat(out_rows).to_csv(os.path.join(HERE, "tables_at_90.csv"), index=False)
    B = dict(league_seasons=int(len(T)))
    for p, pl in ((0, "pre"), (1, "post")):
        g = T[T.post == p]
        B[pl] = dict(n=int(len(g)), first_changed=float(g.first_changed.mean()), bottom2_changed=float(g.bottom2_changed.mean()),
                     bottom3_changed=float(g.bottom3_changed.mean()), top4_changed=float(g.top4_changed.mean()),
                     share_of_clubs_moved=float(g.clubs_moved.sum() / g.clubs.sum()), mean_abs_rank_move=float(g.mean_abs_rank_move.mean()),
                     mean_abs_pts_per_club=float(g.mean_abs_pts.mean()), mean_abs_pts_per_34=float((34 * g.mean_abs_pts / g.games).mean()),
                     any_of_first_bottom3=float(((g.first_changed + g.bottom3_changed) > 0).mean()))
    OUT["tables"] = B
    print(f"  league-seasons with a full table and a known 90:00 score: {len(T)}")
    for key, lab in (("first_changed", "first place differs"), ("top4_changed", "top four differ"), ("bottom3_changed", "bottom three differ"),
                     ("bottom2_changed", "bottom two differ"), ("any_of_first_bottom3", "first place or bottom three differ"),
                     ("share_of_clubs_moved", "clubs in a different position"), ("mean_abs_pts_per_34", "points moved per club per 34 games (net, absolute)")):
        a, b = B["pre"][key], B["post"][key]
        fmt = (lambda v: f"{v:.2f}") if "pts" in key else (lambda v: f"{v:.1%}")
        print(f"  {lab:<52} {fmt(a)} (n={B['pre']['n']}) -> {fmt(b)} (n={B['post']['n']})")
    top = T[(T.tier == "1") & (T.first_changed == 1)].sort_values(["k", "league"])
    OUT["first_changed_top_divisions"] = top[["league", "season", "first", "first_at_90", "gap_first", "n_unknown"]].to_dict("records")
    print("  top divisions where first place differs (league season: final first | first at 90:00 | final gap in points | matches with unknown 90:00):")
    for r in top.itertuples():
        print(f"    {r.league} {r.season}: {r.first} | {r.first_at_90} | {r.gap_first} | {r.n_unknown}")
    big = pd.concat(out_rows)
    big = big[big.tier == "1"].assign(k=lambda d: d.k)
    ext = big.sort_values("d_pts")
    print("  largest net points gained after 90:00 by a top-division club in one season:",
          [(r.league, r.season, r.club, int(r.d_pts)) for r in ext.tail(5).itertuples()][::-1])
    print("  largest net points lost:", [(r.league, r.season, r.club, int(r.d_pts)) for r in ext.head(5).itertuples()])
    json.dump(OUT, open(os.path.join(HERE, "results_decides.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
