# Where the goals went: stoppage time after football's 2023 timekeeping change

Replication materials for the paper *Where the Goals Went: Stoppage Time after Football's 2023
Timekeeping Change in 94 Leagues* (full paper, November 2026): the match-level data behind every
table and figure, and the code that produces them.

The paper compares 151,250 league matches in 94 leagues over six seasons, three before and three
after the game's lawmakers agreed, on 4 March 2023, that competitions should count lost time as at
the 2022 World Cup
([IFAB, 137th AGM](https://www.theifab.com/news/var-decision-communications-trials-confirmed-by-the-ifab/)).
Goals in stoppage time rose by 32% (95% interval 27% to 38%); goals in minutes 1 to 90 did not
change. Announced stoppage time went separate ways league by league, most of the added time is
tied to no event in the match's timeline, and goals per announced minute did not change.

## Reproduce

```bash
pip install -r requirements.txt
sh analysis/run_all.sh
```

About one minute on a laptop (Python 3.12). The run rewrites the result files in `analysis/`, the
five figures in `figures/` and `analysis/numbers.json`, which holds every number printed in the
paper: 624 values and the three tables. The files in this repository are the output of exactly this
run. The figures are set in Helvetica Neue on macOS and in Arial elsewhere, so the type may differ;
the numbers do not.

## Contents

| Path | What it is |
|---|---|
| `data/matches.csv` | One row per match of the sample (151,250). |
| `data/inc_matches.csv`, `data/inc_goals.csv` | The timeline sample: one row per sampled match (40,763) with the events of each half, and one row per goal in those matches (136,529). |
| `data/durations.csv` | Real length of the match and of its second half, in seconds, where the clock times are known. |
| `data/results_all.csv` | Every regular-season result (151,661), with the clubs as numbers, for the league tables stopped at 90:00. |
| `data/exclusions.csv` | The matches left out, each with its reason. |
| `data/leagues.csv` | The 94 leagues: key, name, country, tier, calendar, matches. |
| `analysis/*.py`, `analysis/run_all.sh` | The analysis, in the order of `run_all.sh`. `common.py` holds the sample rule. |
| `analysis/build/` | The two scripts that made the match files from the results database and the downloaded timelines. They cannot run here; they are included so the sample rule and the reading of the timelines can be checked. |
| `analysis/numbers.json` | Every number in the paper, by key, and the three tables. |
| `analysis/results_*.json`, `analysis/*.csv`, `analysis/figdata/` | Outputs of the steps; `figdata/` holds the files the figures are drawn from. |
| `figures/` | The five figures (PNG and PDF). |
| `abstract_2026-09/` | The materials of the September 2026 abstract (league-season counts, 93 leagues, an earlier sample rule). Kept as they were. |

Figures and tables are numbered in the paper by their order in the text:

| In the paper | Here |
|---|---|
| Figure 1, timing | `figures/fig2_timing`, from `analysis/figdata/fig2_monthly.csv` and `fig2_half_seasons.csv` |
| Figure 2, twelve leagues | `figures/fig3_leagues`, from `fig3_league_paths.csv` and `fig3_league_summary.csv` |
| Figure 3, goals by minute | `figures/fig1_minutes`, from `fig1_minutes.csv` |
| Figure 4, goals follow the announced minutes | `figures/fig4_mechanism`, from `analysis/league_dose.csv` and `fig4_by_board.csv` |
| Figure 5, what a minute decides | `figures/fig5_decides`, from `fig5_by_board.csv` |
| Table 1, seconds per event | table `t3` in `numbers.json` (`late_goals.py`) |
| Table 2, goals by part of the match | table `t1` (`models.py`) |
| Table 3, announced minutes and goals in the panel | table `t2` (`mechanism.py`) |

## The data files

**What is not in them.** No club names and no match days. Every identifier is a random number
with no meaning outside this repository; clubs are numbers within their league season; the date
is given as the month; the clock times are given as lengths.

### `data/matches.csv`

| Column | Meaning |
|---|---|
| `match_id` | Random number of the match. Joins `results_all.csv`. |
| `event_id` | Random number of the match's record at the timeline source; empty for the 1.5% of matches that could not be linked. Joins `inc_matches.csv`, `inc_goals.csv` and `durations.csv`. |
| `league`, `tier`, `calendar` | League key (see `leagues.csv`); `1`, `2`, `3`, `4-6`, `women`, `youth`; `winter` (autumn to spring) or `summer` (calendar year). |
| `season`, `k`, `period` | Season; its index relative to the first season that began after the recommendation (`0` = 2023/24 or 2023); `pre` (`k` < 0) or `post`. |
| `month` | Month of the match, `YYYY-MM`. |
| `ft_h`, `ft_a` | Final score, home and away. |
| `goals`, `goals_1h`, `goals_2h` | Goals in the match and in each half. |
| `s1`, `s2` | Goals in first-half and in second-half stoppage time. |
| `first`, `second` | Minutes of the goals of each half, separated by spaces. A stoppage-time goal is stored as 45 or 90 plus the added minute: 45+2 is `47`, 90+3 is `93`. |
| `last_is_s2` | 1 if the last goal of the match came in second-half stoppage time. |
| `board1`, `board2` | Stoppage time announced at the end of each half, in minutes; empty where it is not recorded. |

The sample rule (regular-season matches only, a match counted once, what happens to a match whose
score and goal minutes disagree) is the function `classify` in `analysis/common.py`.

### `data/inc_matches.csv`

One row per match whose timeline was read. `group` says why it is in the sample: `stoppage_goal`
(every linked match with a stoppage-time goal in `matches.csv`), `control` (a random draw from the
other matches with a recorded announcement, stratified by league and season) or `no_local_minutes`
(matches of two leagues read before their goal minutes were available; used only for the score at
90:00). `p_select` is the probability with which the match was drawn; estimates weight by its
inverse.

| Columns | Meaning |
|---|---|
| `ss_board1`, `ss_board2`, `ss_board1_n`, `ss_board2_n` | Announced stoppage time in the timeline, and how many times it was announced. |
| `ss_ft_h`, `ss_ft_a`, `ss_ht_h`, `ss_ht_a` | Final and half-time score in the timeline. |
| `has_extra_time`, `goal_sequence_ok` | 1 if the timeline shows extra time or a penalty shoot-out (such matches are not used); 1 if its goals add up, one at a time, to its final score. |
| `ss_goals`, `ss_s1`, `ss_s2`, `ss_reg_goals_1`, `ss_reg_goals_2` | Goals in the timeline: all, in stoppage time by half, in regular time by half. |
| `subs_h`, `subs_stop_h`, `subs_inplay_h`, `sub_windows_h`, `sub_stops_h`, `injury_subs_h` | Substitutions in regular time of half `h`; in its stoppage time; in regular time without those made at the interval; the breaks they were made in, with and without the interval; substitutions marked as for an injury. |
| `yellow_h`, `red_h`, `cards_stop_h` | Yellow and red cards in regular time of the half; cards in its stoppage time. |
| `var_h`, `var_stop_h`, `pens_h` | Video-review decisions in regular time and in stoppage time; penalties in regular time, scored or not. |
| `has_subs`, `has_cards` | 1 if the timeline lists any substitution, any card. |
| `h90`, `a90` | Score when second-half stoppage time began. |
| `pts_home_90`, `pts_away_90`, `pts_home_ft`, `pts_away_ft`, `result_changed_s2` | Points on the score at 90:00 and on the final score; 1 if they differ. |
| `league`, `season`, `tier`, `k` | As in `matches.csv`. |

### `data/inc_goals.csv`

One row per goal of those matches: `half`, `minute`, `added` (the added minute, 0 in regular time),
`stoppage` (1 if in stoppage time), `side` (`home` or `away`), `cls` (the kind of goal, for example
`penalty`), `h_before`, `a_before` (the score before it) and `scorer_margin_before` (the scoring
side's lead before it; negative when behind).

### `data/durations.csv`

`total_s` is the time from the scheduled kick-off to the last change of the match record, in
seconds; `last_change_is_status` is 1 when that last change was the status turning to "ended", so
that `total_s` is kick-off to final whistle, interval included. `h2_s` is the same for the second
half alone, where its start time survives. The analysis keeps a length only when it is plausible
(`analysis/durations.py`).

### `data/results_all.csv`

`match_id`, `league`, `season`, `home`, `away` (club numbers within the league season), `ft_h`,
`ft_a`. It holds the matches of `matches.csv` and the few regular-season matches the minute rule
leaves out, so that a league table has all its results.

### `data/exclusions.csv`

`league`, `season`, `reason`: one row per match of the 94 leagues and six seasons that is left out.
`not_regular_season` (play-offs, split rounds, relegation ties) and `stored_twice` are left out by
design; the other reasons are scores or goal minutes that are missing or contradict each other.

## Sources

Results, half-time scores and goal minutes come from the
[Football Charts](https://www.football-charts.com) match database. Recent seasons are open under
CC BY 4.0 ([DOI 10.5281/zenodo.22295583](https://doi.org/10.5281/zenodo.22295583)); the full
history back to 2020 is available through the
[Football Charts API](https://www.football-charts.com/developers), free for academic institutions
on request. Announced stoppage time, match timelines and clock times were read from SofaScore's
public match pages in November 2026.

## Licence

Data files: CC BY 4.0. Code: MIT. Please cite the paper and Football Charts.
