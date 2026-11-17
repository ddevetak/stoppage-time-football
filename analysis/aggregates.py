#!/usr/bin/env python3
"""Step 2: public aggregates from ../data/matches.csv. No per-match rows leave this script.

Writes, next to this file:
  league_season.csv   one row per league-season: matches, goals, stoppage goals by half and by
                      minute band, results settled late, and the added-time board (count, sum)
  goal_clock.csv      goals by minute of play, per calendar type and season index (all leagues pooled)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def minutes(s):
    return [int(x) for x in str(s).split()] if isinstance(s, str) and s.strip() else []


def main():
    m = pd.read_csv(os.path.join(HERE, "..", "data", "matches.csv"), dtype={"first": str, "second": str})
    first = m["first"].map(minutes)
    second = m["second"].map(minutes)

    def count(lists, lo, hi):
        return lists.map(lambda t: sum(1 for x in t if lo <= x <= hi))

    m["g_1_15"] = count(first, 1, 15)
    m["g_16_30"] = count(first, 16, 30)
    m["g_31_45"] = count(first, 31, 45)
    m["g_46_60"] = count(second, 46, 60)
    m["g_61_75"] = count(second, 61, 75)
    m["g_76_90"] = count(second, 76, 90)
    # goals written at exactly the last regular minute: if stoppage goals had once been recorded as
    # "45" or "90", these would be inflated before 2023 and fall after
    m["g_45"] = count(first, 45, 45)
    m["g_90"] = count(second, 90, 90)
    m["g_41_44"] = count(first, 41, 44)
    m["g_86_89"] = count(second, 86, 89)
    # second-half stoppage by minute band (90+1..90+3, 90+4..90+6, 90+7 and later)
    m["s2_1_3"] = count(second, 91, 93)
    m["s2_4_6"] = count(second, 94, 96)
    m["s2_7p"] = count(second, 97, 200)
    # first-half stoppage (45+1, 45+2, 45+3, 45+4 and later)
    m["s1_1"] = count(first, 46, 46)
    m["s1_2"] = count(first, 47, 47)
    m["s1_3"] = count(first, 48, 48)
    m["s1_4p"] = count(first, 49, 200)

    draw = m.ft_h == m.ft_a
    margin1 = (m.ft_h - m.ft_a).abs() == 1
    m["goalless"] = (m.goals == 0).astype(int)
    m["draws"] = draw.astype(int)
    m["any_s"] = ((m.s1 + m.s2) > 0).astype(int)
    m["any_s2"] = (m.s2 > 0).astype(int)
    m["last_s2"] = m.last_is_s2
    m["late_eq_draw"] = (draw & (m.last_is_s2 == 1)).astype(int)        # a win became a draw
    m["margin1_last_s2"] = (margin1 & (m.last_is_s2 == 1)).astype(int)  # late winner or consolation
    m["only_goal_s2"] = ((m.goals == 1) & (m.s2 == 1)).astype(int)      # 1:0 settled in stoppage time
    for h in ("1", "2"):
        has = m[f"board{h}"].notna()
        m[f"n_board{h}"] = has.astype(int)
        m[f"sum_board{h}"] = m[f"board{h}"].fillna(0)
        m[f"s{h}_with_board"] = np.where(has, m[f"s{h}"], 0)
    m["g_76_90_with_board2"] = np.where(m.board2.notna(), m.g_76_90, 0)
    m["g_31_45_with_board1"] = np.where(m.board1.notna(), m.g_31_45, 0)

    sums = ["goals", "goals_1h", "goals_2h", "s1", "s2", "goalless", "draws", "any_s", "any_s2", "last_s2",
            "late_eq_draw", "margin1_last_s2", "only_goal_s2",
            "g_1_15", "g_16_30", "g_31_45", "g_46_60", "g_61_75", "g_76_90", "g_45", "g_90", "g_41_44", "g_86_89",
            "s2_1_3", "s2_4_6", "s2_7p", "s1_1", "s1_2", "s1_3", "s1_4p",
            "n_board1", "sum_board1", "s1_with_board", "g_31_45_with_board1",
            "n_board2", "sum_board2", "s2_with_board", "g_76_90_with_board2"]
    keys = ["league", "tier", "calendar", "season", "k", "period"]
    ls = m.groupby(keys)[sums].sum().reset_index()
    ls.insert(6, "matches", m.groupby(keys).size().values)
    ls = ls.sort_values(["league", "k"])
    ls.to_csv(os.path.join(HERE, "league_season.csv"), index=False)

    rows = []
    for (cal, k), g in m.groupby(["calendar", "k"]):
        n = len(g)
        c1, c2 = {}, {}
        for t in g["first"].map(minutes):
            for x in t:
                c1[x] = c1.get(x, 0) + 1
        for t in g["second"].map(minutes):
            for x in t:
                c2[x] = c2.get(x, 0) + 1
        for x, c in sorted(c1.items()):
            rows.append(dict(calendar=cal, k=k, matches=n, half=1, minute=min(x, 45), added=max(x - 45, 0), goals=c))
        for x, c in sorted(c2.items()):
            rows.append(dict(calendar=cal, k=k, matches=n, half=2, minute=min(x, 90), added=max(x - 90, 0), goals=c))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "goal_clock.csv"), index=False)

    print(f"league_season.csv: {len(ls)} rows, {ls.league.nunique()} leagues, {int(ls.matches.sum())} matches")
    t = ls.groupby("period")[["matches"] + sums].sum()
    for p in ("pre", "post"):
        r = t.loc[p]
        print(f"{p}: matches {int(r.matches)}  goals/match {r.goals / r.matches:.3f}  stoppage/match "
              f"{(r.s1 + r.s2) / r.matches:.4f}  1H {r.s1 / r.matches:.4f}  2H {r.s2 / r.matches:.4f}  "
              f"other goals/match {(r.goals - r.s1 - r.s2) / r.matches:.3f}  goalless {r.goalless / r.matches:.2%}")


if __name__ == "__main__":
    main()
