"""Shared definitions for the SSAC27 stoppage-time analysis (version 2, November 2026).

The sample rule, in one place. It differs from the September run (abstract_2026-09/) in four ways,
each found on 2026-11-06 / 2026-11-11:

1. A goalless match counts. FC stores a 0:0 match's goal minutes as the sentinel '111'; the 09-24
   script read that as one goal, saw it did not match the score and dropped every 0:0.
2. A match whose only stored minute is '111' despite a score has unknown minutes and is excluded
   (the 09-24 script counted a stoppage goal for the four such matches with exactly one goal).
3. Regular-season games only. Split rounds, play-offs and relegation ties are stored for 2025/26 and
   for no earlier season, so keeping them would change the mix of games after 2023.
4. A match stored twice under two spellings of one club (168 pairs, all in 2025/26) counts once.
"""
import re

TIER = {}
for _l in ("algir1 australia austria1 azer belgium1 brazil1 bulgaria1 china czech1 denmark1 egypt estonia "
           "finland1 france1 germany1 greece1 holland1 hungary1 india1 ireland1 italy1 japan1 korea1 kuwait "
           "latvia lithuania morocco norway poland1 portugal1 premier romania1 russia saudi1 scot-premier "
           "serbia1 slovenia1 spain1 sweden1 swiss1 thai1 turkey1 usa1").split():
    TIER[_l] = "1"
for _l in ("austria2 brazil2 cha czech2 denmark2 finland2 france2 germany2 greece2 holland2 hungary2 ireland2 "
           "italy2 japan2 korea2 poland2 portugal2 saudi2 scot-champ spain2 sweden2 swiss2 turkey2").split():
    TIER[_l] = "2"
for _l in "czech3a czech3b eng1 france3 germany3 holland3 italy3a italy3b primera1 primera2 scot1".split():
    TIER[_l] = "3"
for _l in "eng2 germany-north national north south scot2 segunda1 segunda2 segunda3 segunda4 segunda5".split():
    TIER[_l] = "4-6"
for _l in "wbelgium1 wfrance1 wgermany1 wsweden".split():
    TIER[_l] = "women"
for _l in "develop21 italy19".split():
    TIER[_l] = "youth"

TIER_ORDER = ["1", "2", "3", "4-6", "women", "youth"]
TIER_LABEL = {"1": "Top divisions", "2": "Second tiers", "3": "Third tiers", "4-6": "Tiers 4-6",
              "women": "Women's top divisions", "youth": "Youth"}

WINTER = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
SUMMER = ["2020", "2021", "2022", "2023", "2024", "2025"]
# Season index: 0 = the first season after the IFAB meeting of 4 March 2023 (2023/24, or 2023).
REL = {s: i - 3 for i, s in enumerate(WINTER)}
REL.update({s: i - 3 for i, s in enumerate(SUMMER)})
SEASONS = WINTER + SUMMER

_SCORE = re.compile(r"\s*(\d+)\s*[:\-]\s*(\d+)\s*")


def parse_score(s):
    """'2:1' -> (2, 1). Anything else (cancelled, awarded, after extra time or penalties) -> None."""
    m = _SCORE.fullmatch(str(s))
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_minutes(s):
    try:
        return sorted(int(float(x)) for x in str(s).split(",") if x.strip())
    except ValueError:
        return None


def classify(ft, ht, goal_times):
    """Apply the sample rule to one match.

    Returns (reason, None) when the match is excluded, else (None, dict) with
      goals, goals_1h, goals_2h, s1, s2 (stoppage goals by half), first (first-half minutes),
      second (second-half minutes).
    FC stores 45+2 as 47 and 90+3 as 93. The first (half-time goals) minutes in order are the first
    half's; a first-half stoppage goal is one of those above 45, a second-half one is one of the rest
    above 90. The counts s1 and s2 are exact under this rule even when a 45+3 goal (48) and a 47th- or
    48th-minute second-half goal share a match; the split of single minutes across the break is not.
    """
    f, h = parse_score(ft), parse_score(ht)
    if f is None:
        return "no_full_time_score", None
    if h is None:
        return "no_half_time_score", None
    total, n1 = f[0] + f[1], h[0] + h[1]
    if n1 > total or h[0] > f[0] or h[1] > f[1]:
        return "half_time_above_full_time", None
    if total == 0:
        return None, dict(goals=0, goals_1h=0, goals_2h=0, s1=0, s2=0, first=[], second=[])
    times = parse_minutes(goal_times)
    if times is None:
        return "minutes_unreadable", None
    if times == [111]:
        return "minutes_unknown", None
    if len(times) != total:
        return "minutes_count_differs", None
    first, second = times[:n1], times[n1:]
    if first and first[-1] > 60:
        return "half_time_score_inconsistent", None
    return None, dict(goals=total, goals_1h=n1, goals_2h=total - n1,
                      s1=sum(1 for t in first if t > 45), s2=sum(1 for t in second if t > 90),
                      first=first, second=second)
