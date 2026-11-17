#!/usr/bin/env python3
# NOT RUNNABLE HERE. This script made the files in ../../data/ from inputs that are not in the repository
# (the results database export with club names and dates; the downloaded match timelines). It is included
# so that the sample rule and the reading of the timelines can be checked line by line.
"""Step 1: the match table for the SSAC27 paper, from one export of FC's finished matches.

  python3 build_matches.py --fm-csv private/fm_export_<source>_<date>.csv [--extra-csv more_rows.csv]

Input: a CSV with id, league, season, match_date, home_team, away_team, ft_result, ht_result,
all_goal_times, league_phase_status (FC's FinishedMatch). `--extra-csv` appends rows from a second
export for leagues the first one lacks (used for the youth league kept only in the July copy).
Also reads ../sofascore_board/board_matches.csv (the added-time board, Stage 1).

Output, all under private/ (per-match history, not for the public repository):
  matches.csv      one row per match in the sample
  exclusions.csv   one row per excluded match with the reason
and prints the exclusion table. The public aggregates are built from matches.csv by aggregates.py.
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import REL, SEASONS, TIER, classify

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.join(HERE, "..", "sofascore_board", "board_matches.csv")
BOARD_MAX = 30  # minutes; nothing above it in Stage 1


def load_board():
    b = pd.read_csv(BOARD, dtype={"fc_match_id": "Int64"}, low_memory=False)
    b = b[b.fc_match_id.notna()].copy()
    b["has2"] = b.injury_time_2.notna()
    b["has1"] = b.injury_time_1.notna()
    # One row per FC match: prefer a row with a board (12 FC matches have two finished events).
    b = b.sort_values(["has2", "has1", "sofascore_event_id"], ascending=[False, False, True])
    b = b.drop_duplicates("fc_match_id")
    return b.set_index("fc_match_id")[["sofascore_event_id", "injury_time_1", "injury_time_2"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm-csv", required=True)
    ap.add_argument("--extra-csv")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "private"))
    args = ap.parse_args()

    fm = pd.read_csv(args.fm_csv, dtype=str)
    if args.extra_csv:
        extra = pd.read_csv(args.extra_csv, dtype=str)
        extra = extra[~extra.league.isin(set(fm.league))]
        fm = pd.concat([fm, extra], ignore_index=True)
    fm["id"] = fm.id.astype("int64")
    n_in = len(fm)

    reason = pd.Series("", index=fm.index)
    reason[~fm.season.isin(SEASONS)] = "season_outside_window"
    reason[(reason == "") & ~fm.league.isin(TIER)] = "league_not_in_paper"
    # The six leagues added in October 2026 (usa1, slovakia1, slovakia2, belgium2, wengland1, argentina1)
    # label the stage "regular_season"; every older league has "Regular Season".
    reason[(reason == "") & ~fm.league_phase_status.fillna("Regular Season").isin(["Regular Season", "regular_season"])] = "not_regular_season"

    board = load_board()
    fm = fm.join(board, on="id")

    # A match stored twice under two spellings of one club: same league, season, date, scores and
    # goal minutes, and the same home or the same away name. Keep the row linked to the board.
    live = fm[reason == ""].copy()
    live["linked"] = live.sofascore_event_id.notna()
    key = ["league", "season", "match_date", "ft_result", "ht_result", "all_goal_times"]
    dup_ids = []
    for _, g in live[live.duplicated(key, keep=False)].groupby(key):
        g = g.sort_values(["linked", "id"], ascending=[False, True])
        kept = []
        for r in g.itertuples():
            if any(r.home_team == k.home_team or r.away_team == k.away_team for k in kept):
                dup_ids.append(r.id)
            else:
                kept.append(r)
    reason[fm.id.isin(dup_ids) & (reason == "")] = "stored_twice"

    # Every regular-season result with a full-time score, whatever the state of its goal minutes:
    # league tables (decides.py) need all of them, not only the matches the minute rule keeps.
    from common import parse_score
    base = fm[reason == ""].copy()
    sc = base.ft_result.map(parse_score)
    base = base[sc.notna()]
    sc = sc[sc.notna()]
    res_all = pd.DataFrame(dict(id=base.id.values, league=base.league.values, season=base.season.values,
                                match_date=base.match_date.values, home=base.home_team.values, away=base.away_team.values,
                                ft_h=[a for a, _ in sc], ft_a=[b for _, b in sc],
                                sofascore_event_id=base.sofascore_event_id.values))

    rows = []
    for i, r in fm[reason == ""].iterrows():
        why, c = classify(r.ft_result, r.ht_result, r.all_goal_times)
        if why:
            reason[i] = why
            continue
        ft = r.ft_result.replace("-", ":").split(":")
        ht = r.ht_result.replace("-", ":").split(":")
        second = c["second"]
        first = c["first"]
        rows.append(dict(
            id=r.id, league=r.league, tier=TIER[r.league], calendar="winter" if "-" in r.season else "summer",
            season=r.season, k=REL[r.season], match_date=r.match_date, home=r.home_team, away=r.away_team,
            ft_h=int(ft[0]), ft_a=int(ft[1]), ht_h=int(ht[0]), ht_a=int(ht[1]),
            goals=c["goals"], goals_1h=c["goals_1h"], goals_2h=c["goals_2h"], s1=c["s1"], s2=c["s2"],
            first=" ".join(map(str, first)), second=" ".join(map(str, second)),
            # the last goal of the match is a second-half stoppage goal
            last_is_s2=int(bool(second) and second[-1] > 90),
            sofascore_event_id=r.sofascore_event_id, board1=r.injury_time_1, board2=r.injury_time_2))
    m = pd.DataFrame(rows)
    m["period"] = np.where(m.k >= 0, "post", "pre")
    for c in ("board1", "board2"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
        m.loc[m[c] > BOARD_MAX, c] = np.nan
    # A second-half board of 0 is a missing value, not "no added time": SofaScore writes almost none
    # before 2023/24 (8 matches) and 379 after, and those matches score in stoppage time like matches
    # with a board of 3 to 4 minutes. First-half zeros are kept (they score like the shortest boards).
    m.loc[m.board2 == 0, "board2"] = np.nan
    m["sofascore_event_id"] = m.sofascore_event_id.astype("Int64")

    os.makedirs(args.out_dir, exist_ok=True)
    m.to_csv(os.path.join(args.out_dir, "matches.csv"), index=False)
    res_all.to_csv(os.path.join(args.out_dir, "results_all.csv"), index=False)
    ex = fm[reason != ""][["id", "league", "season", "match_date", "ft_result", "ht_result",
                           "all_goal_times", "league_phase_status"]].copy()
    ex["reason"] = reason[reason != ""]
    ex.to_csv(os.path.join(args.out_dir, "exclusions.csv"), index=False)

    print(f"rows in {n_in}; in the sample {len(m)}; leagues {m.league.nunique()}; league-seasons "
          f"{m.groupby(['league', 'season']).ngroups}")
    in_window = ex[~ex.reason.isin(["season_outside_window", "league_not_in_paper"])].copy()
    in_window["period"] = np.where(in_window.season.map(REL) >= 0, "post", "pre")
    tab = pd.crosstab(in_window.reason, in_window.period, margins=True)
    print("\nexcluded, by reason and period:")
    print(tab.to_string())
    kept = m.groupby("period").size()
    for p in ("pre", "post"):
        n_ex = int((in_window.period == p).sum())
        print(f"{p}: kept {kept[p]}, excluded {n_ex} ({n_ex / (n_ex + kept[p]):.2%})")
    print("\nsample by period:")
    print(m.groupby("period").agg(matches=("id", "size"), goals=("goals", "sum"), s1=("s1", "sum"), s2=("s2", "sum"),
                                  goalless=("goals", lambda x: int((x == 0).sum())),
                                  board2=("board2", "count")).to_string())


if __name__ == "__main__":
    main()
