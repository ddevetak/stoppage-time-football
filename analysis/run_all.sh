#!/bin/sh
# Rebuilds every result file, the five figures and numbers.json from ../data/. A few minutes on a laptop.
set -e
cd "$(dirname "$0")"
python3 aggregates.py
python3 models.py
python3 mechanism.py 0.5      # the wider panel (announcement recorded for 50% of matches in every season)
python3 mechanism.py 0.95     # the stricter panel
python3 mechanism.py          # the panel of the paper (80%); run last, it writes league_dose.csv
python3 late_goals.py
python3 checks.py
python3 durations.py analyse
python3 decides.py
python3 figdata.py
python3 figures.py
python3 paper_numbers.py
