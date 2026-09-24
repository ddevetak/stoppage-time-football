"""Extract league x season aggregates for the stoppage-time study.

Run inside the Football Charts backend (needs the match database):
    python manage.py shell < extract_aggregates.py
Writes /tmp/league_season_aggregates.csv. Output contains counts only:
no teams, no individual matches, no dates.
"""
import csv
from collections import defaultdict

from charts.models import FinishedMatch, LeagueStatus

TIER = {}
for l in ("algir1 australia austria1 azer belgium1 brazil1 bulgaria1 china czech1 denmark1 egypt estonia "
          "finland1 france1 germany1 greece1 holland1 hungary1 india1 ireland1 italy1 japan1 korea1 kuwait "
          "latvia lithuania morocco norway poland1 portugal1 premier romania1 russia saudi1 scot-premier "
          "serbia1 slovenia1 spain1 sweden1 swiss1 thai1 turkey1").split():
    TIER[l] = "1"
for l in ("austria2 brazil2 cha czech2 denmark2 finland2 france2 germany2 greece2 holland2 hungary2 ireland2 "
          "italy2 japan2 korea2 poland2 portugal2 saudi2 scot-champ spain2 sweden2 swiss2 turkey2").split():
    TIER[l] = "2"
for l in "czech3a czech3b eng1 france3 germany3 holland3 italy3a italy3b primera1 primera2 scot1".split():
    TIER[l] = "3"
for l in "eng2 germany-north national north south scot2 segunda1 segunda2 segunda3 segunda4 segunda5".split():
    TIER[l] = "4-6"
for l in "wbelgium1 wfrance1 wgermany1 wsweden".split():
    TIER[l] = "women"
for l in "develop21 italy19".split():
    TIER[l] = "youth"

WINTER = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
SUMMER = ["2020", "2021", "2022", "2023", "2024", "2025"]
REL = {s: i - 3 for i, s in enumerate(WINTER)}
REL.update({s: i - 3 for i, s in enumerate(SUMMER)})


def parse(score):
    try:
        a, b = str(score).replace("-", ":").split(":")[:2]
        return int(a), int(b)
    except (ValueError, AttributeError):
        return None


names = dict(LeagueStatus.objects.values_list("league", "display_name"))
countries = {}
agg = defaultdict(lambda: defaultdict(int))
qs = (FinishedMatch.objects.filter(season__in=WINTER + SUMMER)
      .values_list("league", "country", "season", "ft_result", "ht_result", "all_goal_times"))
for league, country, season, ft, ht, gt in qs.iterator(chunk_size=5000):
    if league not in TIER:
        continue
    countries.setdefault(league, (country or "").title())
    a = agg[(league, season)]
    a["matches_total"] += 1
    f, h = parse(ft), parse(ht)
    try:
        times = sorted(int(float(x)) for x in (gt or "").split(",") if x.strip())
    except ValueError:
        times = None
    if not f or not h or times is None:
        a["matches_excluded"] += 1
        continue
    total, n1 = f[0] + f[1], h[0] + h[1]
    if len(times) != total or n1 > total:
        a["matches_excluded"] += 1
        continue
    first, second = times[:n1], times[n1:]
    s1 = sum(1 for t in first if t > 45)
    s2 = sum(1 for t in second if t > 90)
    a["matches"] += 1
    a["goals"] += total
    a["stoppage_goals_1h"] += s1
    a["stoppage_goals_2h"] += s2
    a["normal_time_goals"] += total - s1 - s2
    if second and second[-1] > 90 and f[0] == f[1]:
        a["late_equaliser_draws"] += 1

cols = ["league", "league_name", "country", "tier", "calendar", "season", "season_rel", "period",
        "matches_total", "matches_excluded", "matches", "goals", "normal_time_goals",
        "stoppage_goals_1h", "stoppage_goals_2h", "late_equaliser_draws"]
with open("/tmp/league_season_aggregates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for (league, season) in sorted(agg, key=lambda k: (k[0], REL[k[1]])):
        a = agg[(league, season)]
        w.writerow({"league": league, "league_name": names.get(league, league),
                    "country": countries.get(league, ""), "tier": TIER[league],
                    "calendar": "winter" if "-" in season else "summer", "season": season,
                    "season_rel": REL[season], "period": "post" if REL[season] >= 0 else "pre",
                    **{c: a.get(c, 0) for c in cols[8:]}})
print("rows", len(agg))
