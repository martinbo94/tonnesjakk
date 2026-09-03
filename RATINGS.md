# Tønnesjakk rating list

Maximum-likelihood Elo over 7200 round-robin games
(paired openings, both colours; draws = ½). Anchor: `heur-100ms` = 1500.
All configurations run under the current binary — code-era differences
(e.g. pre-2026-08-27 root-search bugs) are not reproduced, only knob/net/TB eras.

| # | player | Elo | games |
|---|---|---|---|
| 1 | `net3-tb-100ms` | **1850** | 1600 |
| 2 | `net3-100ms` | **1792** | 1600 |
| 3 | `net2-100ms` | **1785** | 1600 |
| 4 | `net3-100ms-oldsearch` | **1731** | 1600 |
| 5 | `net1b-100ms` | **1729** | 1600 |
| 6 | `net1-100ms` | **1699** | 1600 |
| 7 | `heur-100ms` | **1500** | 1600 |
| 8 | `heur-100ms-oldsearch` | **1496** | 1600 |
| 9 | `heur-d4` | **1058** | 1600 |

_Updated 2026-08-31 10:42. Extend: add a PlayerDef and re-run `python scripts/rating_tournament.py` — only missing pairs are played._
