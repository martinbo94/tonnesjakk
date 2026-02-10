# TODO - Tønnesjakk AI

## Høy prioritet

### Generer nytt treningssett og tren NNUE
1. [ ] **Generer treningsdata** (~2.6 timer):
   ```
   .venv/Scripts/python.exe -u -m tonnesjakk.nnue --games 10000 --depth 7 --save-data training_data_10k_d7.npz --no-compare --arch 64 32
   ```
2. [ ] **Tren 16x8 modell** (beste speed/strength tradeoff):
   ```
   .venv/Scripts/python.exe -m tonnesjakk.nnue --load-data training_data_10k_d7.npz --arch 16 8 --epochs 80 --output nnue_16x8 --no-compare
   ```
3. [ ] **Sammenlign mot heuristikk**: `--compare nnue_16x8/nnue_weights.json heuristic --compare-games 50 --depth 5`

### NNUE Self-Play Forbedring
Neste steg er en iterativ self-play forbedringssyklus. Se [SELF_PLAY_PLAN.md](SELF_PLAY_PLAN.md) for full plan.

- [ ] **Self-play loop** - Bruk beste NNUE til å generere treningsdata, tren ny modell, promoter kun hvis den slår forrige
- [ ] **Gatekeeper-matcher** - Automatisk sammenligning mellom ny og gammel modell (50+ partier)
- [ ] **Generasjonsbasert trening** - Bygg på data fra stadig sterkere modeller

### Modellsammenligning
Tre arkitekturer er trent og kan sammenlignes:

| Arkitektur | Val Loss | Status |
|------------|----------|--------|
| 147→64→32→1 | 0.0120 | Trent |
| 147→32→16→1 | 0.0137 | Trent |
| 147→16→8→1 | 0.0157 | Trent |

- [ ] **Tidsmatchet benchmark** - Sammenlign arkitekturer med lik tidskontroll
- [ ] **Velg produksjonsarkitektur** - Balanse mellom styrke og hastighet

### Web UI
- [ ] **Animasjoner** - Smooth animasjon av tønne-bevegelser
- [ ] **Undo/Redo** - Mulighet til å angre trekk
- [ ] **Spillhistorikk** - Vis liste over alle trekk
- [ ] **AI styrke-velger** - Slider for å justere dybde/vanskelighet

---

## Medium prioritet

### Søkoptimalisering
- [ ] **Parallell søk (Lazy SMP)** - Multi-threaded search for moderne CPUer
- [ ] **Razoring** - Aggressiv pruning nær bladnoder
- [ ] **SEE (Static Exchange Evaluation)** - Bedre vurdering av "capture" trekk
- [ ] **Countermove Heuristic** - Lagre beste svar på hvert trekk

### Åpningsbok
- [ ] **Generere åpningsbok** - Spill tusenvis av partier, lagre gode åpninger
- [ ] **Polyglot format** - Standard format for sjakk-åpningsbøker
- [ ] **Random valg** - Velg tilfeldig blant gode åpninger for variasjon

### Sluttspill
- [ ] **Endgame tablebase** - Perfekt spill når få brikker gjenstår
- [ ] **Generere tablebases** - Retrograd analyse for alle enkle sluttspill

---

## Lav prioritet / Eksperimentelt

### Alternativ AI
- [ ] **MCTS (Monte Carlo Tree Search)** - AlphaZero-lignende tilnærming
- [ ] **Hybrid MCTS + Minimax** - Kombiner styrker fra begge
- [ ] **Reinforcement Learning** - Tren med self-play uten menneskelig kunnskap

### Analyse
- [ ] **PV (Principal Variation) visning** - Vis beste trekk-sekvens i UI
- [ ] **Multi-PV** - Vis flere alternative linjer
- [ ] **Analyse-modus** - La brukeren utforske stillinger

### Infrastruktur
- [ ] **WebSocket** - Real-time oppdateringer i stedet for polling
- [ ] **Persistent lagring** - Database for pågående spill
- [ ] **Multiplayer** - Spill mot andre mennesker online
- [ ] **ELO-rating** - Rangeringssystem for spillere

---

## Fullført

### BitBoard & Søk (Januar 2025)
- [x] **BitBoard representasjon** - u64 bitboards for rask posisjon-håndtering
- [x] **Prekalkulerte tabeller** - ADJACENT, JUMP_OVER, JUMP_LANDING (const)
- [x] **BitMove pakket format** - 32-bit kompakt trekk-representasjon
- [x] **Make/Unmake med UndoInfo** - Unngå brett-kloning i søk
- [x] **History Heuristic** - Sporer cutoff-foråsakende trekk, depth2-bonus
- [x] **Futility Pruning** - Skipper håpløse trekk ved shallow depth
- [x] **Evaluation Cache** - 64K entries for posisjon-evaluering
- [x] **Forbedret TT Replacement** - Depth-preferred med generasjons-aging
- [x] **Engine wrapper** - BitBoardEngine eksponert via Python-kompatibel Engine

### NNUE (Januar 2025)
- [x] **IncrementalNNUE** - Accumulator-basert inkrementell evaluering
- [x] **SIMD akselerasjon** - f32x8 vektorisert layer 1 (wide crate)
- [x] **Accumulator stack** - 32-nivå stack for søketre
- [x] **Transponert vektlayout** - Bedre cache-lokalitet for inkrementell oppdatering

### Søkoptimalisering (Februar 2025)
- [x] **Aspiration Windows** - Smalt søkevindu rundt forventet score
- [x] **Principal Variation Search (PVS)** - Null-window søk for ikke-PV trekk
- [x] **Late Move Reductions (LMR)** - Reduser dybde for trekk sent i listen
- [x] **Quiescence Search** - Fortsett søk i taktiske stillinger (nær mål)
- [x] **Null Move Pruning** - Gi motstander "gratis" trekk for å teste om posisjon er god nok
- [x] **Forbedret heuristikk** - Trussel-bonus, blokkerings-evaluering, bedre pail-vurdering
- [x] **Quiescence Depth Limit** - Maks 8 nivåer for å forhindre stack overflow

### NNUE Trening & Pipeline (Februar 2025)
- [x] **147 input-features** - 144 base + 3 relasjonelle features (fremgang, avstand, senter)
- [x] **Treningspipeline** - `train_nnue.py` med selvspill datagenerering
- [x] **Arkitektursøk** - Trent og sammenlignet 64x32, 32x16, 16x8
- [x] **Benchmark-verktøy** - `benchmark_nnue.py`, `compare_time_matched.py`
- [x] **Fjernet QuantizedNNUE** - Kun IncrementalNNUE (float SIMD) brukes nå
- [x] **Self-play plan** - [SELF_PLAY_PLAN.md](SELF_PLAY_PLAN.md) utarbeidet

### Grunnleggende (Tidligere)
- [x] Grunnleggende spillogikk (6x6 brett, tønner, melkespann)
- [x] Alpha-Beta pruning
- [x] Transposition Table med Zobrist hashing
- [x] Killer Moves
- [x] Håndlaget evaluering (fremgang, senter, mobilitet)
- [x] NNUE-arkitektur og trening (PyTorch)
- [x] NNUE eksport til JSON
- [x] NNUE inference i Rust
- [x] FastAPI web-backend
- [x] Interaktivt web-grensesnitt
- [x] Stockfish-lignende terminal output
- [x] PyO3 Rust-Python bindings
- [x] README dokumentasjon

---

## Bugs / Kjente problemer

- [ ] **Andretreksfordel** - Svart ser ut til å ha betydelig fordel ved perfekt spill
- [ ] **NNUE bias** - Modellen er trent på ubalansert data (bare svart-seire)
- [ ] **PyO3 enum comparison** - Må bruke `repr()` for enum-sammenligning

## Fikset bugs

- [x] **Engine crash etter flere spill** - STATUS_ILLEGAL_INSTRUCTION (0xC000001D) på Windows
  - **Årsak**: Ubegrenset quiescence search forårsaket stack overflow
  - **Løsning**: Lagt til `MAX_QSEARCH_DEPTH = 8` grense i `quiesce()` funksjonen
  - Se `debug.md` for full analyse

---

## Ytelsestall

### Nåværende (Februar 2025 - med Aspiration/PVS/LMR/Quiescence)
| Dybde | Tid | Noder | NPS |
|-------|-----|-------|-----|
| 6 | 76ms | 40K | 535K |
| 8 | 59ms | 40K | 676K |
| 10 | 490ms | 369K | 753K |
| 12 | 2.8s | 2.5M | 908K |
| 13 | 31s | 24M | 777K |

### BitBoardEngine før søkoptimalisering (Januar 2025)
| Dybde | Tid | Noder | NPS |
|-------|-----|-------|-----|
| 4 | 45ms | 38K | 860K |
| 6 | 219ms | 306K | 1.4M |
| 8 | 2.7s | 2.0M | 730K |

### Gammel Engine (for sammenligning)
| Dybde | Tid | Noder | NPS |
|-------|-----|-------|-----|
| 4 | 1.44s | 12K | 8K |
| 6 | 8.85s | 84K | 9.5K |
| 8 | 9.59s | 343K | 36K |

**Total forbedring: Kan nå søke dybde 12 på ~3s (før: dybde 8 tok 2.7s)**

---

## Viktige filer

- `src/lib.rs` - Rust-kjerne (~4000 linjer, kun IncrementalNNUE)
- `python/tonnesjakk/nnue.py` - NNUE modell og trening
- `train_nnue.py` - Hovedskript for selvspill + trening
- `benchmark_nnue.py` - Modell-benchmark
- `compare_time_matched.py` - Tidsmatchet sammenligning
- `web/server.py` - FastAPI backend
- `nnue_weights.json` - Eksporterte NNUE-vekter
- `SELF_PLAY_PLAN.md` - Plan for self-play forbedringssyklus
