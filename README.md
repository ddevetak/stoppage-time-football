# Where the goals went: stoppage time after football's 2023 timekeeping change

Replication materials for *Where the goals went: stoppage time after football's 2023 timekeeping change in 93 leagues*
(D. Devetak, Football Charts, 2026).

After the 2022 World Cup, IFAB guidance encouraged a more accurate calculation of added
time (applied in English football from 2023/24). Testing a break at the first post-World Cup
season (2023/24; 2023 for calendar-year leagues), across 136,558 matches in 93 leagues and 42 countries, goals in normal time
did not change (2.629 → 2.626 per match), while goals in stoppage time rose 31%
(0.223 → 0.292 per match). The rise is about twice as large in tiers 4–6 and in women's top
divisions as in tiers 1–3; youth competitions show no increase.

## Contents

| Path | What it is |
|---|---|
| `data/league_season_aggregates.csv` | One row per league and season: match counts and goal counts by period (normal time, first-half stoppage, second-half stoppage). Counts only — no teams, no individual matches. |
| `analysis/reproduce.py` | Rebuilds every table and figure of the paper from the aggregates file alone. |
| `analysis/extract_aggregates.py` | How the aggregates were produced from the Football Charts match database (for transparency; needs database access). |
| `figures/` | The three figures, as produced by `reproduce.py`. |

## Reproduce

```bash
pip install matplotlib
python analysis/reproduce.py
```

## Columns of `league_season_aggregates.csv`

| Column | Meaning |
|---|---|
| `league`, `league_name`, `country` | Football Charts league key, display name, country |
| `tier` | `1`, `2`, `3`, `4-6`, `women`, `youth` |
| `calendar` | `winter` (autumn–spring season, e.g. `2023-2024`) or `summer` (calendar-year season, e.g. `2023`) |
| `season`, `season_rel` | Season, and its position relative to the first season of the added-time regime (0 = 2023/24 or 2023) |
| `period` | `pre` (season_rel < 0) or `post` |
| `matches_total` | Matches in the archive for that league-season |
| `matches_excluded` | Matches dropped because the goal-minute list does not match the full-time score, or the half-time score exceeds it |
| `matches` | Matches used |
| `goals`, `normal_time_goals` | All goals; goals outside stoppage time |
| `stoppage_goals_1h`, `stoppage_goals_2h` | First-half stoppage goals (identified via the half-time score); second-half goals recorded after minute 90 |
| `late_equaliser_draws` | Matches that ended level with the last goal in second-half stoppage time |

## Match-level data

Match-level results and goal minutes are provided by
[Football Charts](https://www.football-charts.com): recent seasons are open under CC BY 4.0
([DOI 10.5281/zenodo.22295583](https://doi.org/10.5281/zenodo.22295583)); full history back to
2020 is available through the [Football Charts API](https://www.football-charts.com/developers),
free for academic institutions on request.

## Licence

Aggregated data: CC BY 4.0. Code: MIT. Please cite the paper and Football Charts.
