#!/usr/bin/env python3
# NOT RUNNABLE HERE. This script made the files in ../../data/ from inputs that are not in the repository
# (the results database export with club names and dates; the downloaded match timelines). It is included
# so that the sample rule and the reading of the timelines can be checked line by line.
"""Step 5: parse the Stage 2 incident files into two private tables.

Reads ../sofascore_board/stage2/incidents_*.jsonl.gz and ../sofascore_board/stage2_events.csv
(which matches were fetched and with what sampling probability). Sends no requests.

Writes under private/:
  inc_matches.csv  one row per fetched match: the board from the incident feed, events by half
                   (goals, substitutions, cards, VAR decisions, penalties), stoppage goals as
                   SofaScore records them, and the result at 45:00 / 90:00 against the final result
  inc_goals.csv    one row per goal: half, minute, added minute, scoring side, score before it

SofaScore writes a first-half stoppage event as time 45 with addedTime x, a second-half one as time 90
with addedTime x, so the half is unambiguous here (FC's own minutes store 45+3 as 48).
"""
import glob
import gzip
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.join(HERE, "..", "sofascore_board")


def half_of(i):
    t = i.get("time")
    if t is None:
        return None
    return 1 if t <= 45 else 2


def in_stoppage(i):
    return bool(i.get("addedTime")) and i.get("time") in (45, 90) and i.get("addedTime", 0) < 900


def points(h, a):
    return (3, 0) if h > a else ((1, 1) if h == a else (0, 3))


def parse(event_id, incidents):
    row = dict(sofascore_event_id=event_id)
    goals = sorted((i for i in incidents if i.get("incidentType") == "goal"
                    and i.get("homeScore") is not None and i.get("awayScore") is not None),
                   key=lambda i: i["homeScore"] + i["awayScore"])
    # boards: the announced added time per half (the largest if announced more than once)
    for h, t in ((1, 45), (2, 90)):
        b = [i.get("length") for i in incidents if i.get("incidentType") == "injuryTime" and i.get("time") == t
             and i.get("length") is not None]
        row[f"ss_board{h}"] = max(b) if b else None
        row[f"ss_board{h}_n"] = len(b)
    ft = [i for i in incidents if i.get("incidentType") == "period" and i.get("text") == "FT"]
    ht = [i for i in incidents if i.get("incidentType") == "period" and i.get("text") == "HT"]
    row["ss_ft_h"] = ft[0].get("homeScore") if ft else None
    row["ss_ft_a"] = ft[0].get("awayScore") if ft else None
    row["ss_ht_h"] = ht[0].get("homeScore") if ht else None
    row["ss_ht_a"] = ht[0].get("awayScore") if ht else None
    row["has_extra_time"] = int(any(i.get("incidentType") == "period" and str(i.get("text", "")).startswith(("ET", "AET", "PEN", "AP"))
                                    for i in incidents) or any((i.get("time") or 0) > 90 for i in incidents))

    grows, ph, pa, ok = [], 0, 0, True
    for g in goals:
        h, a = g["homeScore"], g["awayScore"]
        if (h - ph, a - pa) not in ((1, 0), (0, 1)):
            ok = False
        side = "home" if h - ph == 1 else "away"
        before = (ph, pa)
        diff = (ph - pa) if side == "home" else (pa - ph)   # scorer's margin before the goal
        grows.append(dict(sofascore_event_id=event_id, half=half_of(g), minute=g.get("time"),
                          added=g.get("addedTime") if in_stoppage(g) else 0, stoppage=int(in_stoppage(g)),
                          side=side, cls=g.get("incidentClass"), h_before=before[0], a_before=before[1],
                          scorer_margin_before=diff))
        ph, pa = h, a
    row["goal_sequence_ok"] = int(ok and (row["ss_ft_h"] is None or (ph, pa) == (row["ss_ft_h"], row["ss_ft_a"])))
    row["ss_goals"] = len(grows)
    for h in (1, 2):
        row[f"ss_s{h}"] = sum(1 for g in grows if g["half"] == h and g["stoppage"])
        row[f"ss_reg_goals_{h}"] = sum(1 for g in grows if g["half"] == h and not g["stoppage"])

    def n(kind, h, stop, test=lambda i: True):
        return sum(1 for i in incidents if i.get("incidentType") == kind and half_of(i) == h
                   and in_stoppage(i) == stop and test(i))

    for h in (1, 2):
        row[f"subs_{h}"] = n("substitution", h, False)
        row[f"subs_stop_{h}"] = n("substitution", h, True)
        wins = {(i.get("isHome"), i.get("time")) for i in incidents if i.get("incidentType") == "substitution"
                and half_of(i) == h and not in_stoppage(i)}
        row[f"sub_windows_{h}"] = len(wins)
        row[f"injury_subs_{h}"] = n("substitution", h, False, lambda i: bool(i.get("injury")))
        # substitutions made during play: a change at the interval (written as minute 46) stops nothing
        inplay = [i for i in incidents if i.get("incidentType") == "substitution" and half_of(i) == h
                  and not in_stoppage(i) and i.get("time") != 46]
        row[f"subs_inplay_{h}"] = len(inplay)
        row[f"sub_stops_{h}"] = len({(i.get("isHome"), i.get("time")) for i in inplay})
        row[f"yellow_{h}"] = n("card", h, False, lambda i: i.get("incidentClass") == "yellow")
        row[f"red_{h}"] = n("card", h, False, lambda i: i.get("incidentClass") in ("red", "yellowRed"))
        row[f"cards_stop_{h}"] = n("card", h, True)
        row[f"var_{h}"] = n("varDecision", h, False)
        row[f"var_stop_{h}"] = n("varDecision", h, True)
        row[f"pens_{h}"] = n("inGamePenalty", h, False) + sum(
            1 for g in grows if g["half"] == h and not g["stoppage"] and g["cls"] == "penalty")
    row["has_subs"] = int(any(i.get("incidentType") == "substitution" for i in incidents))
    row["has_cards"] = int(any(i.get("incidentType") == "card" for i in incidents))

    # the result when second-half stoppage time began, against the final result
    h90 = sum(1 for g in grows if g["side"] == "home" and not (g["half"] == 2 and g["stoppage"]))
    a90 = sum(1 for g in grows if g["side"] == "away" and not (g["half"] == 2 and g["stoppage"]))
    row["h90"], row["a90"] = h90, a90
    p90, pft = points(h90, a90), points(ph, pa)
    row["pts_home_90"], row["pts_away_90"] = p90
    row["pts_home_ft"], row["pts_away_ft"] = pft
    row["result_changed_s2"] = int(p90 != pft)
    return row, grows


def main():
    ev = pd.read_csv(os.path.join(S2, "stage2_events.csv"))
    seen = {}
    for f in sorted(glob.glob(os.path.join(S2, "stage2", "incidents_*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                seen[r["event_id"]] = r["incidents"]
    mrows, grows = [], []
    for eid, inc in seen.items():
        r, g = parse(eid, inc)
        mrows.append(r)
        grows.extend(g)
    m = pd.DataFrame(mrows).merge(
        ev[["sofascore_event_id", "fc_match_id", "group", "p_select", "panel_league", "league", "fc_season", "tier",
            "season_rel", "s1", "s2", "injury_time_1", "injury_time_2"]], on="sofascore_event_id", how="inner")
    m = m.rename(columns={"fc_season": "season", "season_rel": "k", "s1": "fc_s1_stage2list", "s2": "fc_s2_stage2list"})
    os.makedirs(os.path.join(HERE, "private"), exist_ok=True)
    m.to_csv(os.path.join(HERE, "private", "inc_matches.csv"), index=False)
    g = pd.DataFrame(grows)
    g.to_csv(os.path.join(HERE, "private", "inc_goals.csv"), index=False)
    print(f"matches {len(m)}; goals {len(g)}; goal sequence consistent {m.goal_sequence_ok.mean():.2%}; "
          f"with subs {m.has_subs.mean():.1%}; with cards {m.has_cards.mean():.1%}; "
          f"board from incidents: first half {m.ss_board1.notna().mean():.1%}, second half {m.ss_board2.notna().mean():.1%}")
    print("second-half board announced more than once:", int((m.ss_board2_n > 1).sum()),
          "| board in incidents equals the Stage 1 board (second half):",
          f"{(m.ss_board2 == m.injury_time_2)[m.ss_board2.notna() & m.injury_time_2.notna()].mean():.2%}")
    print(m.groupby("group").size().to_dict())


if __name__ == "__main__":
    main()
