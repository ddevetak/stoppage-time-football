#!/usr/bin/env python3
"""Step 12: every number the paper prints, formatted once, from the analysis outputs.

  python3 paper_numbers.py  writes numbers.json (values and the three tables) and prints it

The manuscript names these values by key, so that its text, tables and figures are filled from this
one file. A value that starts with a plus sign also exists without it under the key with "_u".
"""
import json
import math
import os
import sys

import pandas as pd

from common import REL

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda name: json.load(open(os.path.join(HERE, name)))
MINUS = "−"
N, T = {}, {}


def num(v, d=2):
    return f"{v:,.{d}f}".replace("-", MINUS)


def sgn(v, d=2):
    return f"{v:+,.{d}f}".replace("-", MINUS)


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def chg(rr, d=0):
    """A ratio as a percentage change."""
    return f"{100 * (rr - 1):+.{d}f}%".replace("-", MINUS)


def ci(lo, hi, d=2):
    return f"{num(lo, d)} to {num(hi, d)}"


def rrci(r, d=2):
    return f"{num(r['rr'], d)} ({ci(r['lo'], r['hi'], d)})"


def main():
    mod, mech, chk = J("results_models.json"), J("results_mechanism.json"), J("results_checks.json")
    inc, dec, dur = J("results_incidents.json"), J("results_decides.json"), J("results_durations.json")
    fd = J(os.path.join("figdata", "figdata.json"))
    mech50, mech95 = J("results_mechanism_50.json"), J("results_mechanism_95.json")

    # ---------------- sample
    s = mod["sample"]
    N.update(n_matches=f"{s['matches']:,}", n_leagues=str(s["leagues"]), n_league_seasons=str(s["league_seasons"]),
             n_pre=f"{s['matches_pre']:,}", n_post=f"{s['matches_post']:,}")
    ex = pd.read_csv(os.path.join(HERE, "..", "data", "exclusions.csv"), dtype=str)
    ex = ex[~ex.reason.isin(["season_outside_window", "league_not_in_paper"])]
    ex["post"] = [REL[x] >= 0 for x in ex.season]
    quality = ex[~ex.reason.isin(["not_regular_season", "stored_twice"])]
    q_pre, q_post = int((~quality.post).sum()), int(quality.post.sum())
    N.update(excl_pre=str(q_pre), excl_post=str(q_post),
             excl_pre_pct=pct(q_pre / (q_pre + s["matches_pre"]), 2), excl_post_pct=pct(q_post / (q_post + s["matches_post"]), 2),
             excl_playoff=f"{int((ex.reason == 'not_regular_season').sum()):,}", excl_twice=str(int((ex.reason == "stored_twice").sum())))
    N.update(board2_matches=f"{mech['coverage']['2']['matches_with_board']:,}", board2_share=pct(mech["coverage"]["2"]["share"], 0),
             board1_matches=f"{mech['coverage']['1']['matches_with_board']:,}", board1_share=pct(mech["coverage"]["1"]["share"], 0),
             panel_leagues=str(mech["panel"]["2"]["leagues"]), panel_matches=f"{mech['panel_matches_2']:,}",
             panel1_leagues=str(mech["panel"]["1"]["leagues"]), panel1_matches=f"{mech['panel_matches_1']:,}",
             panel50_leagues=str(mech50["panel"]["2"]["leagues"]), panel95_leagues=str(mech95["panel"]["2"]["leagues"]))
    N.update(inc_matches=f"{inc['sample']['fetched_in_sample']:,}", inc_goal_matches=f"{inc['sample']['stoppage_goal_matches']:,}",
             inc_controls=f"{inc['sample']['controls']:,}", linked_matches=f"{inc['sample']['linked_matches']:,}",
             val_score=pct(inc["validation"]["final_score_agrees"], 1), val_counts=pct(inc["validation"]["counts_agree_exactly"], 1),
             val_rr_fc=num(inc["validation"]["two_sources"]["ratio_fc"]), val_rr_ss=num(inc["validation"]["two_sources"]["ratio_sofascore"]))
    N.update(dur_leagues=str(len(dur["panel"])), dur_matches=f"{sum(v['n'] for v in dur['by_season'].values()):,}")
    lsn = pd.read_csv(os.path.join(HERE, "league_season.csv"))
    N.update(n_goals=f"{int(lsn.goals.sum()):,}", n_stoppage_goals=f"{int(lsn.s1.sum() + lsn.s2.sum()):,}",
             n_summer=str(lsn[lsn.calendar == "summer"].league.nunique()), n_winter=str(lsn[lsn.calendar == "winter"].league.nunique()))

    # ---------------- the headline (models.py)
    raw, fe = mod["raw"], mod["fe"]
    for key, name in (("s", "s"), ("s1", "s1"), ("s2", "s2"), ("other", "oth"), ("goals", "all")):
        N[f"{name}_pre"] = num(raw[key]["pre"], 3)
        N[f"{name}_post"] = num(raw[key]["post"], 3)
        N[f"{name}_rr"] = num(fe[key]["rr"])
        N[f"{name}_ci"] = ci(fe[key]["lo"], fe[key]["hi"])
        N[f"{name}_chg"] = chg(fe[key]["rr"])
    N.update(s_chg_lo=chg(fe["s"]["lo"]), s_chg_hi=chg(fe["s"]["hi"]), oth_chg1=chg(fe["other"]["rr"], 1),
             oth_chg_lo=chg(fe["other"]["lo"], 1), oth_chg_hi=chg(fe["other"]["hi"], 1),
             goalless_pre=pct(raw["goalless_share"]["pre"]), goalless_post=pct(raw["goalless_share"]["post"]),
             # "one fewer match in every N ends goalless", N to the nearest ten
             goalless_one_in=str(int(round(1 / (raw["goalless_share"]["pre"] - raw["goalless_share"]["post"]), -1))),
             draws_pre=pct(mod["settled"]["draws"]["pre"]), draws_post=pct(mod["settled"]["draws"]["post"]),
             s_dispersion=num(fe["s"]["pearson_dispersion"], 1))
    p = fd["fig1"]["per_1000_matches"]
    N.update(std_leagues=str(fd["fig1"]["leagues"]),
             std_stop=sgn(p["stoppage"]["diff"] / 1000, 3), std_stop_ci=f"{sgn(p['stoppage']['lo'] / 1000, 3)} to {sgn(p['stoppage']['hi'] / 1000, 3)}",
             std_reg=sgn(p["regular"]["diff"] / 1000, 3), std_reg_ci=f"{sgn(p['regular']['lo'] / 1000, 3)} to {sgn(p['regular']['hi'] / 1000, 3)}",
             std_all=sgn(p["all"]["diff"] / 1000, 3), std_all_ci=f"{sgn(p['all']['lo'] / 1000, 3)} to {sgn(p['all']['hi'] / 1000, 3)}",
             std_share=pct(p["stoppage_share_of_rise"]["share"], 0), std_share_lo=pct(p["stoppage_share_of_rise"]["lo"], 0),
             std_s1=sgn(p["first_stoppage"]["diff"], 0), std_s2=sgn(p["second_stoppage"]["diff"], 0),
             std_r1=sgn(p["first_regular"]["diff"], 0), std_r2=sgn(p["second_regular"]["diff"], 0),
             raw_share=pct(chk["share_of_rise"]["share"], 0))
    bm = mech["by_minute"]
    for key in ("s2_1_3", "s2_4_6", "s2_7p", "s1_1", "s1_2", "s1_3", "s1_4p", "g_86_89", "g_41_44", "g_90", "g_45"):
        N[f"min_{key}_rr"] = num(bm[key]["rr"])
        N[f"min_{key}_pre"] = num(bm[key]["pre"], 1)
        N[f"min_{key}_post"] = num(bm[key]["post"], 1)
        N[f"min_{key}_chg"] = chg(bm[key]["rr"])
    # Figure 1: how many of the 90 minutes of regular time have a change whose 95% interval excludes zero
    f1 = pd.read_csv(os.path.join(HERE, "figdata", "fig1_minutes.csv"))
    f1 = f1[f1.added == 0]
    n_out = int(((f1.diff_lo > 0) | (f1.diff_hi < 0)).sum())
    N.update(reg_minutes=str(len(f1)),
             reg_out=["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][n_out] if n_out <= 10 else str(n_out))
    lg = mod["leagues"]
    N.update(lg_n=str(lg["n"]), lg_up=str(lg["up"]), lg_down=str(lg["down"]), lg_sig_up=str(lg["significantly_up"]),
             lg_sig_down=str(lg["significantly_down"]), lg_mean=num(lg["mean_rr"]), lg_mean_ci=ci(lg["lo"], lg["hi"]),
             lg_i2=pct(lg["I2"], 0), lg_range=ci(lg["range_95"][0], lg["range_95"][1]))
    for t, name in (("1", "t1"), ("2", "t2"), ("3", "t3"), ("4-6", "t46"), ("women", "tw"), ("youth", "ty")):
        r = mod["by_tier"][t]
        N[f"{name}_rr"] = rrci(r)
        N[f"{name}_n"] = str(r["leagues"])
    rb = mod["recording_bound"]
    N.update(bound_all=rrci(rb["all"]["stoppage_or_minute_90"]), bound_46=rrci(rb["4-6"]["stoppage_or_minute_90"]),
             bound_all_chg=chg(rb["all"]["stoppage_or_minute_90"]["rr"]),
             g90_46_fall=pct(1 - rb["4-6"]["minute_90"]["rr"], 0), g90_13_fall=pct(1 - rb["1-3"]["minute_90"]["rr"], 0))
    low = mech["minute_90_by_tier"]["4-6"]
    N.update(t46_g90_pre=num(low["g90_pre"], 0), t46_g90_post=num(low["g90_post"], 0))

    # ---------------- timing
    es = mod["event_study"]["s"]
    for k in ("k-3", "k-2", "k0", "k1", "k2"):
        N[f"es_{k.replace('-', 'm')}"] = num(es[k]["rr"])
        N[f"es_{k.replace('-', 'm')}_ci"] = ci(es[k]["lo"], es[k]["hi"])
    st = mod["step_vs_trend"]
    N.update(es_k0_chg=chg(es["k0"]["rr"]), es_k2_chg=chg(es["k2"]["rr"]),
             trend_slope=chg(st["pre_slope_per_season"]["rr"]), trend_k0=rrci(st["vs_trend_k0"]), trend_k0_chg=chg(st["vs_trend_k0"]["rr"]),
             trend_k1=rrci(st["vs_trend_k1"]), trend_k2=rrci(st["vs_trend_k2"]),
             rob_adjacent=rrci(mod["robustness"]["2022/23 against 2023/24 only"]), rob_adjacent_chg=chg(mod["robustness"]["2022/23 against 2023/24 only"]["rr"]),
             rob_no2020=rrci(mod["robustness"]["without 2020/21 and 2020"]), rob_winter=rrci(mod["robustness"]["winter-calendar leagues"]),
             rob_summer=rrci(mod["robustness"]["summer-calendar leagues"]), rob_men=rrci(mod["robustness"]["men's senior leagues only"]),
             rob_six=rrci(mod["robustness"]["leagues with all six seasons"]))
    mon = pd.read_csv(os.path.join(HERE, "figdata", "fig2_monthly.csv")).set_index("ym")
    hs = pd.read_csv(os.path.join(HERE, "figdata", "fig2_half_seasons.csv")).set_index("cell")
    for ym in ("2021-05", "2021-08", "2021-12", "2022-05", "2022-08", "2022-11", "2022-12", "2023-05", "2023-08", "2023-12", "2024-05", "2024-08",
               "2024-12", "2025-05", "2025-08", "2025-12", "2026-05"):
        N[f"mon_{ym.replace('-', '_')}"] = num(mon.loc[ym, "board"], 1)
    N["mon_len_2023_05"] = num(mon.loc["2023-05", "length"], 1)
    N["mon_len_2023_08"] = num(mon.loc["2023-08", "length"], 1)
    N["mon_len_2023_12"] = num(mon.loc["2023-12", "length"], 1)
    spring = lambda y: mon.loc[[f"{y}-0{i}" for i in (2, 3, 4)], "board"].mean()
    for y in (2021, 2022, 2023, 2024, 2025, 2026):
        N[f"spring_{y}"] = num(spring(y), 1)
    for c in hs.index:
        key = c.replace("/", "_")
        N[f"hs_board_{key}"] = num(hs.loc[c, "board"])
        N[f"hs_len_{key}"] = num(hs.loc[c, "length"], 1)
        N[f"hs_goals_{key}"] = num(hs.loc[c, "goals"], 3)
    N.update(fig2_leagues=str(fd["fig2"]["n_leagues"]), fig2_matches=f"{fd['fig2']['matches']:,}",
             wc_step=sgn(chk["world_cup_step"]["board_diff"]), wc_step_se=num(chk["world_cup_step"]["board_se"]),
             summer_2023a=sgn(chk["half_seasons_summer"]["2023a"]["board_diff"], 1), summer_2025b=sgn(chk["half_seasons_summer"]["2025b"]["board_diff"], 1))
    f3 = fd["fig3"]
    N.update(paths_n=str(f3["leagues"]), paths_jumped=str(f3["jumped_half_minute"]), paths_still=str(f3["still_half_minute_up"]),
             paths_back=str(f3["back_within_quarter"]), paths_below=str(f3["below"]))
    ls = pd.read_csv(os.path.join(HERE, "figdata", "fig3_league_summary.csv")).set_index("league")
    for lgk in ("premier", "france1", "germany1", "spain1", "italy1", "holland1", "turkey1", "saudi1", "brazil1", "japan1", "korea1", "china", "usa1"):
        r = ls.loc[lgk]
        N[f"path_{lgk}"] = f"{num(r.last_before, 1)}, {num(r.first_after, 1)} and {num(r.latest, 1)}"
        N[f"path_{lgk}_now"] = sgn(r.now, 1)
        N[f"path_{lgk}_a"], N[f"path_{lgk}_b"], N[f"path_{lgk}_c"] = num(r.last_before, 1), num(r.first_after, 1), num(r.latest, 1)
        N[f"path_{lgk}_m"] = num(r.second_after, 1)
    # the order of the leagues: does a league's announced time in one season say where it stands later?
    # Correlations across the panel leagues of the mean second-half announcement by season (k = -3 … 2)
    pb = lsn[lsn.league.isin(ls.index)].copy()
    pb["board2"] = pb.sum_board2 / pb.n_board2
    wide = pb.pivot(index="league", columns="k", values="board2")
    late = wide.drop(index=["turkey1", "turkey2", "saudi1"])       # their last season before already contains the jump
    N.update(ord_pre=num(wide[-3].corr(wide[-1])), ord_cross2=num(wide[-1].corr(wide[1])), ord_cross3=num(wide[-1].corr(wide[2])),
             ord_pre_x=num(late[-3].corr(late[-1])), ord_cross2_x=num(late[-1].corr(late[1])), ord_cross3_x=num(late[-1].corr(late[2])),
             before_min=num(ls.last_before.min(), 1), before_max=num(ls.last_before.max(), 1),
             latest_min=num(ls.latest.min(), 1), latest_max=num(ls.latest.max(), 1),
             before_gap=num(ls.last_before.max() - ls.last_before.min(), 1), latest_gap=num(ls.latest.max() - ls.latest.min(), 1))
    named = dict(before_min="korea1", before_max="turkey1", latest_min="slovenia1", latest_max="china")
    found = dict(before_min=ls.last_before.idxmin(), before_max=ls.last_before.idxmax(), latest_min=ls.latest.idxmin(), latest_max=ls.latest.idxmax())
    if found != named:
        sys.exit(f"paper.md names the leagues with the shortest and longest announcements as {named}; the data now say {found}")
    # the United States on its own (Major League Soccer); intervals are Poisson, from the goal counts of the one league
    us = pd.read_csv(os.path.join(HERE, "league_effects.csv")).set_index("league").loc["usa1"]
    ud = pd.read_csv(os.path.join(HERE, "league_dose.csv"))
    ud = ud[ud.half == 2].set_index("league").loc["usa1"]
    us_pm = (ud.goals_post / ud.board_post) / (ud.goals_pre / ud.board_pre)
    us_pm_se = math.sqrt(1 / (ud.goals_pre * ud.n_pre) + 1 / (ud.goals_post * ud.n_post))
    N.update(usa_s_pre=num(us.pre, 3), usa_s_post=num(us.post, 3),
             usa_rr=f"{num(math.exp(us.log_rr))} ({ci(math.exp(us.log_rr - 1.96 * us.se), math.exp(us.log_rr + 1.96 * us.se))})",
             usa_board_pre=num(ud.board_pre, 1), usa_board_post=num(ud.board_post, 1),
             usa_pm_pre=num(1000 * ud.goals_pre / ud.board_pre, 0), usa_pm_post=num(1000 * ud.goals_post / ud.board_post, 0),
             usa_pm_rr=f"{num(us_pm)} ({ci(us_pm * math.exp(-1.96 * us_pm_se), us_pm * math.exp(1.96 * us_pm_se))})")
    rate = mech["dose"]["2"]["slope"]
    N.update(price_goals=num(rate, 3), fr_goals_season=num(ls.loc["france1", "now"] * rate * 306, 0),
             en_goals_season=num(ls.loc["premier", "now"] * rate * 380, 0))
    N.update(len_pre=num(dur["by_season"]["-3"]["median"], 1), len_last_pre=num(dur["by_season"]["-1"]["median"], 1),
             len_k0=num(dur["by_season"]["0"]["median"], 1), len_k2=num(dur["by_season"]["2"]["median"], 1),
             len_diff=sgn(dur["post_minus_pre_minutes"]["diff"], 1), len_diff_se=num(dur["post_minus_pre_minutes"]["se"]),
             len_per_min1=num(dur["minutes_per_board_minute"]["first_half"]), len_per_min2=num(dur["minutes_per_board_minute"]["second_half"]),
             over_board=num(dur["second_half_length"]["2"]["over_mean"], 1), over_post=sgn(dur["over_post"]["diff"]), over_post_se=num(dur["over_post"]["se"]))

    # ---------------- mechanism
    a, b = mech["period_2"]["pre"], mech["period_2"]["post"]
    m2 = mech["models"]["2"]
    N.update(board_pre=num(a["board"]), board_post=num(b["board"]), board_chg=chg(m2["board_post"]["rr"]), board_rr=rrci(m2["board_post"]),
             board_diff=sgn(b["board"] - a["board"]),
             pg_pre=num(a["goals_pm"], 3), pg_post=num(b["goals_pm"], 3), pg_chg=chg(m2["goals_per_match_post"]["rr"]), pg_rr=rrci(m2["goals_per_match_post"]),
             pm_pre=num(a["per_1000_min"], 1), pm_post=num(b["per_1000_min"], 1), pm_chg=chg(m2["per_minute_post"]["rr"]), pm_rr=rrci(m2["per_minute_post"]),
             pm_lo=chg(m2["per_minute_post"]["lo"]), pm_hi=chg(m2["per_minute_post"]["hi"]),
             ref_pre=num(a["ref_per_1000_min"], 1), ref_post=num(b["ref_per_1000_min"], 1),
             elas=num(m2["elasticity"]["b"]), elas_se=num(m2["elasticity"]["se"]),
             elas2=num(m2["elasticity_two_way"]["b"]), elas2_se=num(m2["elasticity_two_way"]["se"]),
             pm_seasons_p=num(m2["per_minute_event_study"]["all_seasons_equal_p"]))
    d = mech["dose"]["2"]
    N.update(dose_corr=num(d["corr"]), dose_slope=num(1000 * d["slope"], 1), dose_slope_se=num(1000 * d["slope_se"], 1),
             dose_pre=num(1000 * d["pre_rate_per_minute"], 1), dose_int=sgn(d["intercept"], 3), dose_int_se=num(d["intercept_se"], 3),
             dose_up=str(d["board_up"]), dose_min=sgn(d["d_board_min"], 1), dose_max=sgn(d["d_board_max"], 1))
    ml = chk["match_level"]
    N.update(ml_n=f"{ml['n']:,}", ml_uncond=chg(ml["post_unconditional"]), ml_cond=chg(ml["post_given_board"]),
             ml_cond_ci=f"{chg(ml['lo'])} to {chg(ml['hi'])}", ml_elas=num(ml["elasticity_match_level"]), ml_elas_se=num(ml["elasticity_se"]))
    hz = chk["hazard"]
    rates = [hz[str(i)][p_] for i in range(1, 9) for p_ in ("pre", "post")]
    N.update(hz_lo=num(min(rates), 0), hz_hi=num(max(rates), 0), beyond_pre=pct(hz["beyond_share_before"], 0), beyond_post=pct(hz["beyond_share_after"], 0))
    bd = chk["board_distribution"]["2"]
    N.update(b6_pre=pct(bd["before"]["share_ge_6"], 0), b6_post=pct(bd["after"]["share_ge_6"], 0),
             b8_pre=pct(bd["before"]["share_ge_8"], 1), b8_post=pct(bd["after"]["share_ge_8"], 1),
             b10_pre=pct(bd["before"]["share_ge_10"], 1), b10_post=pct(bd["after"]["share_ge_10"], 1))
    for tag, mm in (("p50", mech50), ("p95", mech95)):
        x2 = mm["models"]["2"]
        N[f"{tag}_goals"] = rrci(x2["goals_per_match_post"])
        N[f"{tag}_board"] = rrci(x2["board_post"])
        N[f"{tag}_pm"] = rrci(x2["per_minute_post"])
    m1 = mech["models"]["1"]
    a1, b1 = mech["period_1"]["pre"], mech["period_1"]["post"]
    N.update(h1_board_pre=num(a1["board"]), h1_board_post=num(b1["board"]), h1_board=rrci(m1["board_post"]), h1_goals=rrci(m1["goals_per_match_post"]),
             h1_pm=rrci(m1["per_minute_post"]), h1_pm_pre=num(a1["per_1000_min"], 1), h1_pm_post=num(b1["per_1000_min"], 1))

    # ---------------- what the minutes are made of
    bmod = inc["board_model"]["2"]
    dc = bmod["decomposition"]
    N.update(bm_n=f"{bmod['n']:,}", bm_leagues=str(bmod["leagues"]), bm_board_pre=num(bmod["pre"]["board_mean"]), bm_board_post=num(bmod["post"]["board_mean"]),
             bm_change=sgn(dc["board_change"]), bm_events=sgn(dc["from_more_events"]), bm_time=sgn(dc["from_more_time_per_event_and_base"]),
             bm_price=sgn(dc["from_more_seconds_per_event"]), bm_base=sgn(dc["from_the_base"]),
             bm_base_pre=num(bmod["pre"]["base_minutes"]), bm_base_post=num(bmod["post"]["base_minutes"]),
             bm_base_share=pct(dc["from_the_base"] / dc["board_change"], 0),
             bm_goal_pre=num(bmod["pre"]["seconds_per_event"]["goal"]["sec"], 0), bm_goal_post=num(bmod["post"]["seconds_per_event"]["goal"]["sec"], 0),
             bm_var_post=num(bmod["post"]["seconds_per_event"]["VAR decision"]["sec"], 0),
             bm_one_pre=num(bmod["pre"]["seconds_per_event"]["one goal in it at 90:00"]["sec"], 0),
             bm_one_post=num(bmod["post"]["seconds_per_event"]["one goal in it at 90:00"]["sec"], 0),
             bm_shift=num(bmod["post_shift_seconds"]["sec"], 0), bm_shift_se=num(bmod["post_shift_seconds"]["se"], 0),
             bm_r2_pre=num(bmod["pre"]["r2"]), bm_r2_post=num(bmod["post"]["r2"]))
    bs = inc["board_by_season"]
    lv = bs["level_2022_23"]
    N.update(bs_leagues=str(bs["leagues"]), bs_matches=f"{bs['matches']:,}",
             bs_subs_first=num(lv["subs_2"] + bs["events"]["subs_2"]["-3"]["diff"], 1), bs_subs_ref=num(lv["subs_2"], 1),
             bs_subs_last=num(lv["subs_2"] + bs["events"]["subs_2"]["2"]["diff"], 1),
             bs_stops_first=num(lv["sub_stops_2"] + bs["events"]["sub_stops_2"]["-3"]["diff"], 1), bs_stops_ref=num(lv["sub_stops_2"], 1),
             bs_stops_last=num(lv["sub_stops_2"] + bs["events"]["sub_stops_2"]["2"]["diff"], 1),
             bs_board_first=sgn(bs["board"]["-3"]["diff"]), bs_board_k0=sgn(bs["board"]["0"]["diff"]), bs_board_last=sgn(bs["board"]["2"]["diff"]),
             bs_given_first=sgn(bs["board_given_events"]["-3"]["diff"]), bs_given_k0=sgn(bs["board_given_events"]["0"]["diff"]),
             bs_given_last=sgn(bs["board_given_events"]["2"]["diff"]))
    fav = chk["favouritism"]
    N.update(fav_pre=sgn(fav["before"]["home_behind_minus_home_ahead_by_one"]["sec"], 0), fav_pre_se=num(fav["before"]["home_behind_minus_home_ahead_by_one"]["se"], 1),
             fav_post=sgn(fav["after"]["home_behind_minus_home_ahead_by_one"]["sec"], 0), fav_post_se=num(fav["after"]["home_behind_minus_home_ahead_by_one"]["se"], 1))

    # ---------------- what it decides
    rc = inc["results_changed"]
    N.update(rc_pre=pct(rc["pre"]["changed_share"]), rc_post=pct(rc["post"]["changed_share"]),
             rc_one_in_pre=num(1 / rc["pre"]["changed_share"], 0), rc_one_in_post=num(1 / rc["post"]["changed_share"], 0),
             rc_dw_pre=pct(rc["pre"]["kinds"]["draw became a win"]), rc_dw_post=pct(rc["post"]["kinds"]["draw became a win"]),
             rc_wd_pre=pct(rc["pre"]["kinds"]["win became a draw"]), rc_wd_post=pct(rc["post"]["kinds"]["win became a draw"]),
             rc_lw_pre=pct(rc["pre"]["kinds"]["defeat became a win"], 2), rc_lw_post=pct(rc["post"]["kinds"]["defeat became a win"], 2),
             pts34_pre=num(34 * rc["pre"]["points_moved_per_team_match"], 1), pts34_post=num(34 * rc["post"]["points_moved_per_team_match"], 1))
    L = dec["leads"]
    for key, name in (("one-goal lead not held", "lead1"), ("level match gets a winner", "level"), ("two-goal lead not held", "lead2")):
        r = L[key]
        N.update({f"{name}_pre": pct(r["pre"]["share"]), f"{name}_post": pct(r["post"]["share"]),
                  f"{name}_n": f"{r['pre']['n'] + r['post']['n']:,}",
                  f"{name}_diff": sgn(r["post_diff_pp"], 1), f"{name}_diff_se": num(r["post_diff_se"], 1),
                  f"{name}_per_min": sgn(r["per_board_minute_pp"], 1), f"{name}_per_min_se": num(r["per_board_minute_se"], 1),
                  f"{name}_given": sgn(r["post_given_board_pp"], 1), f"{name}_given_se": num(r["post_given_board_se"], 1),
                  f"{name}_b3": pct(r["by_board"]["3"]["share"], 0), f"{name}_b5": pct(r["by_board"]["5"]["share"], 0),
                  f"{name}_b7": pct(r["by_board"]["7"]["share"], 0), f"{name}_b9": pct(r["by_board"]["9"]["share"], 0)})
    ss = L["state_shares"]
    N.update(st_level_pre=pct(ss["pre"]["level"]), st_level_post=pct(ss["post"]["level"]), st_one_pre=pct(ss["pre"]["one"]), st_one_post=pct(ss["post"]["one"]),
             known90=pct(dec["known_state"]["share"], 1))
    ws = inc["who_scores"]
    for key, name in (("level", "ws_level"), ("trailing by 1", "ws_trail1"), ("trailing by 2+", "ws_trail2"), ("leading", "ws_lead")):
        N[f"{name}_pre"] = pct(ws["pre"]["by_state"][key]["share"], 0)
        N[f"{name}_post"] = pct(ws["post"]["by_state"][key]["share"], 0)
    tb = dec["tables"]
    N.update(tb_n=str(tb["league_seasons"]), tb_n_pre=str(tb["pre"]["n"]), tb_n_post=str(tb["post"]["n"]),
             tb_first_pre=pct(tb["pre"]["first_changed"], 0), tb_first_post=pct(tb["post"]["first_changed"], 0),
             tb_bottom_pre=pct(tb["pre"]["bottom3_changed"], 0), tb_bottom_post=pct(tb["post"]["bottom3_changed"], 0),
             tb_any_pre=pct(tb["pre"]["any_of_first_bottom3"], 0), tb_any_post=pct(tb["post"]["any_of_first_bottom3"], 0),
             tb_pts_pre=num(tb["pre"]["mean_abs_pts_per_34"], 1), tb_pts_post=num(tb["post"]["mean_abs_pts_per_34"], 1),
             tb_moved_pre=pct(tb["pre"]["share_of_clubs_moved"], 0), tb_moved_post=pct(tb["post"]["share_of_clubs_moved"], 0))

    # ---------------- Table 1: goals per match by part of the match
    T["t1"] = dict(
        title="Goals per match by part of the match, before (2020/21 to 2022/23) and after (2023/24 to 2025/26).",
        note=(f"{N['n_matches']} matches in {N['n_leagues']} leagues. The ratio is from a Poisson regression of league-season goal counts on an "
              "after indicator with league fixed effects and matches as exposure; 95% intervals from standard errors clustered by league."),
        header=["Part of the match", "Before", "After", "Ratio after/before (95% interval)"],
        rows=[["Stoppage time, both halves", N["s_pre"], N["s_post"], f"{N['s_rr']} ({N['s_ci']})"],
              ["   first half (45+)", N["s1_pre"], N["s1_post"], f"{N['s1_rr']} ({N['s1_ci']})"],
              ["   second half (90+)", N["s2_pre"], N["s2_post"], f"{N['s2_rr']} ({N['s2_ci']})"],
              ["Minutes 1 to 90", N["oth_pre"], N["oth_post"], f"{N['oth_rr']} ({N['oth_ci']})"],
              ["All goals", N["all_pre"], N["all_post"], f"{N['all_rr']} ({N['all_ci']})"]])

    # ---------------- Table 2: goals follow the announced minutes
    def row2(label, pre, post, rr):
        return [label, pre, post, rr]
    T["t2"] = dict(
        title=f"Second half: announced stoppage time and the goals scored in it, {N['panel_leagues']} leagues with the announcement recorded for at least 80% of matches in every season.",
        note=(f"{N['panel_matches']} matches. Ratios from the Poisson regression of equation (1), with 95% intervals from standard errors clustered by league. "
              "The last row is a comparison period inside regular time, where nothing changed."),
        header=["", "Before", "After", "Ratio after/before (95% interval)"],
        rows=[row2("Announced stoppage time (minutes)", N["board_pre"], N["board_post"], N["board_rr"]),
              row2("Goals in stoppage time, per match", N["pg_pre"], N["pg_post"], N["pg_rr"]),
              row2("Goals per 1,000 announced minutes", N["pm_pre"], N["pm_post"], N["pm_rr"]),
              row2("Goals per 1,000 minutes, minutes 76 to 90", N["ref_pre"], N["ref_post"], "")])

    # ---------------- Table 3: what the announced minutes are made of
    names = [("goal", "Goal"), ("substitution stop", "Stoppage for substitutions"), ("injury substitution (extra)", "Substitution for an injury (extra)"),
             ("yellow card", "Yellow card"), ("red card", "Red card"), ("penalty", "Penalty"), ("VAR decision", "VAR decision"),
             ("level at 90:00 (against 3+ goals)", "Score level at 90:00"), ("one goal in it at 90:00", "One goal between the teams at 90:00"),
             ("two goals in it at 90:00", "Two goals between the teams at 90:00")]
    rows = []
    for key, lab in names:
        a_, b_, c_ = bmod["pre"]["seconds_per_event"][key], bmod["post"]["seconds_per_event"][key], bmod["change_seconds"][key]
        star = "*" if c_["p"] < 0.05 else ""
        per = "" if "90:00" in key else f"{num(bmod['pre']['events_per_match'][key], 2)} / {num(bmod['post']['events_per_match'][key], 2)}"
        rows.append([lab, per, f"{num(a_['sec'], 0)} ({num(a_['se'], 0)})", f"{num(b_['sec'], 0)} ({num(b_['se'], 0)})", f"{sgn(c_['sec'], 0)}{star}"])
    T["t3"] = dict(
        title="Seconds of announced second-half stoppage time associated with each event of the half, before and after.",
        note=(f"Weighted least squares on {N['bm_n']} matches of the incident sample in {N['bm_leagues']} leagues, league fixed effects, standard errors "
              "clustered by league in parentheses. The last three rows compare with a match in which three or more goals separate the teams at 90:00. "
              "* the change differs from zero at the 5% level."),
        header=["Event in the second half", "Per match, before / after", "Seconds each, before", "Seconds each, after", "Change"],
        rows=rows)

    ev = bmod["pre"]["events_per_match"], bmod["post"]["events_per_match"]
    N.update(bm_stops_pre=num(ev[0]["substitution stop"], 1), bm_stops_post=num(ev[1]["substitution stop"], 1),
             bm_inj_more=num(100 * (ev[1]["injury substitution (extra)"] - ev[0]["injury substitution (extra)"]), 0),
             bm_var_more=num(100 * (ev[1]["VAR decision"] - ev[0]["VAR decision"]), 0),
             lead1_hold_b3=pct(1 - L["one-goal lead not held"]["by_board"]["3"]["share"], 0),
             lead1_hold_b9=pct(1 - L["one-goal lead not held"]["by_board"]["9"]["share"], 0))
    # a value that starts with a plus sign also exists without it (key_u), for use after "rose by"
    for k_, v in list(N.items()):
        if v.startswith("+"):
            N[k_ + "_u"] = v[1:]

    out = os.path.join(HERE, "numbers.json")
    json.dump(dict(values=N, tables=T), open(out, "w"), indent=1, ensure_ascii=False)
    width = max(len(k) for k in N)
    for k, v in N.items():
        print(f"{k:<{width}}  {v}")
    for k, t in T.items():
        print(f"\n{k}: {t['title']}")
        print("   " + " | ".join(t["header"]))
        for r in t["rows"]:
            print("   " + " | ".join(r))
    print(f"\n{len(N)} values, {len(T)} tables -> {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
