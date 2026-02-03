# Tønnesjakk AI 🛢️

En komplett AI for det norske TV-spillet **Tønnesjakk** fra NRK, implementert som en Rust/Python hybrid med moderne søkealgoritmer og neural network evaluering.

## Spillregler

Tønnesjakk spilles på et 6×6 brett hvor to spillere (hvit og svart) konkurrerer om å få flest tønner over til motsatt side.

- Hver spiller har **4 tønner** og **1 melkespann**
- Tønner starter **utenfor brettet** og plasseres fra egen side
- Målet er å få tønner over til **motstanderens startrad** (da fjernes de og teller som poeng)
- Tønner beveger seg **ett felt** ortogonalt, eller kan **hoppe over** andre tønner
- **Melkespannet** kan plasseres én gang per spill, og fungerer som en blokker
- Første spiller med flest tønner på andre siden vinner

## Arkitektur

```
┌─────────────────────────────────────────────────────────────┐
│                      Python (web/server.py)                 │
│                    FastAPI + Uvicorn                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Rust Core (src/lib.rs)                    │
│  Board ◄──► BitBoard │ Engine ◄──► BitBoardEngine │ NNUE   │
└─────────────────────────────────────────────────────────────┘
```

### Rust-kjernen (PyO3)

- **Board**: Python-vennlig spillbrett med all spillogikk
- **BitBoard**: Optimalisert brettrepresentasjon (u64 bitboards)
- **Engine**: Python wrapper rundt BitBoardEngine
- **BitBoardEngine**: Rask søkemotor med NNUE evaluering
- **NNUE**: SIMD-akselerert neural network evaluator
- **Eksponert via PyO3** for bruk fra Python

### Python

- **FastAPI backend** for web-grensesnitt
- **NNUE trening** med PyTorch
- **Selvspill** for datagenerering

---

## Søkealgoritmer

### 1. Minimax med Alpha-Beta Pruning
Grunnleggende tre-søk som utforsker alle mulige trekk rekursivt. Alpha-beta pruning kutter bort grener som garantert er verre enn allerede funnet alternativer.

```
Forbedring: ~95% færre noder vs ren minimax
```

### 2. Transposition Table (TT)
En hash-tabell som lagrer evaluerte posisjoner med Zobrist-hashing. Samme posisjon evalueres aldri to ganger.

```
Størrelse: 1M entries (~32 MB)
Innhold: Score, dybde, flag (exact/upper/lower), beste trekk
Forbedring: Depth-preferred replacement med generasjons-aging
```

### 3. Killer Moves
Lagrer trekk som ga "cutoffs" på hver dybde. Disse prøves tidlig fordi de ofte er gode i lignende posisjoner.

```
Implementasjon: 2 killer moves per dybde
```

### 4. History Heuristic
Sporer hvilke trekk som historisk har forårsaket beta-cutoffs. Brukes til å sortere trekk slik at de beste prøves først.

```rust
history[from][to] += depth * depth;  // Bonus for cutoff
// Aging: history *= 0.9 hver søkerunde
```

### 5. Futility Pruning
Ved grunne dybder (≤3), hvis statisk evaluering er langt under alpha, hoppes "håpløse" trekk over.

```rust
if depth <= 3 && static_eval + margin < alpha {
    continue;  // Skip this move
}
```

### 6. Evaluation Cache
Cache for evaluerte posisjoner som ikke lagres i TT (f.eks. bladnoder).

```
Størrelse: 64K entries
Hit rate: Typisk 10-30%
```

---

## Brettrepresentasjon

### BitBoard (u64)
Hver brikketype har en 64-bit integer hvor bit 0-35 representerer de 36 feltene.

```
Bit-mapping (6×6 = 36 bits):
Rad 0: bits  0-5
Rad 1: bits  6-11
...
Rad 5: bits 30-35

sq(row, col) = row * 6 + col
```

### Prekalkulerte tabeller (const)
- `ADJACENT[36]` - Naboer for hvert felt
- `JUMP_OVER[36][4]` - Felt som hoppes over per retning
- `JUMP_LANDING[36][4]` - Landingsfelt etter hopp
- `ROW_MASK[6]` - Bitmask for hver rad

### Make/Unmake Move
I stedet for å klone brettet for hvert trekk, bruker vi `UndoInfo` for å angre trekk effektivt.

---

## Evaluering

### NNUE - Neural Network Evaluering

NNUE (Efficiently Updatable Neural Network) er en lett neural network-evaluator inspirert av Stockfish.

```
Arkitektur:
Input (144) → FC(64) → ReLU → FC(32) → ReLU → FC(1) → Tanh

Input: One-hot encoding av brettet
       6×6×4 = 144 (4 kanaler: hvit tønne, svart tønne, hvit spann, svart spann)

Output: Score mellom -1 (svart vinner) og +1 (hvit vinner)

Parametre: ~11,000 (veldig lite!)
```

#### SIMD Akselerasjon
Layer 1 (144→64) er akselerert med SIMD (f32x8) for ~4x raskere evaluering.

#### Inkrementell Oppdatering
Accumulator-stack cacher layer 1 output. Ved trekk oppdateres kun endrede features (~2-4 per trekk i stedet for 144).

### Heuristisk Fallback

Når NNUE ikke er lastet, brukes håndlaget evaluering:

| Faktor | Vekt | Beskrivelse |
|--------|------|-------------|
| **Fremgang** | 100 | Hvor langt tønnene har kommet mot mål |
| **Scoret** | 500 | Tønner som har nådd mål |
| **Senter-spann** | 10 | Melkespann nær sentrum |

---

## Ytelse

Typisk ytelse på moderne CPU (single-threaded):

| Dybde | Tid | Noder | NPS |
|-------|-----|-------|-----|
| 4 | 45ms | 38K | 860K |
| 5 | 96ms | 103K | 1.1M |
| 6 | 219ms | 306K | 1.4M |
| 7 | 972ms | 804K | 830K |
| 8 | 2.7s | 2.0M | 730K |

**Sammenligning med gammel implementasjon:**

| Dybde | Gammel | Ny | Speedup |
|-------|--------|-----|---------|
| 4 | 1.44s | 45ms | **32x** |
| 6 | 8.85s | 219ms | **40x** |
| 8 | 9.59s | 2.75s | **3.5x** |

---

## Kjøring

### Installasjon

```bash
# Klon repo
git clone <repo>
cd tonnesjakk

# Opprett virtuelt miljø
python -m venv .venv
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Installer Rust-pakken
pip install maturin
maturin develop --release

# Installer Python-avhengigheter
pip install fastapi uvicorn torch
```

### Start webserver

```bash
cd web
python server.py
# Åpne http://localhost:8000
```

### Kommandolinje

```python
from tonnesjakk import Board, Engine, Player

board = Board()
engine = Engine()

# Søk beste trekk på dybde 6
result = engine.search(board, 6)
print(f"Beste trekk: {result.best_move}")
print(f"Score: {result.score}")
print(f"Noder: {result.nodes_searched}")

# Utfør trekket
board.make_move(result.best_move)
```

---

## Filstruktur

```
tønnesjakk/
├── src/
│   └── lib.rs          # Rust-kjerne (~3200 linjer)
│                       # - BitBoard, BitMove, UndoInfo
│                       # - BitBoardEngine (søk)
│                       # - IncrementalNNUE (SIMD-akselerert)
│                       # - Board, Engine (Python API)
├── python/
│   └── tonnesjakk/
│       ├── __init__.py
│       ├── nnue.py     # NNUE trening (PyTorch)
│       └── export_nnue.py
├── web/
│   ├── server.py       # FastAPI backend
│   └── index.html      # Web-grensesnitt
├── Cargo.toml
└── pyproject.toml
```

---

## Fremtidige forbedringer

Se [TODO.md](TODO.md) for full liste.

Høyest prioritet:
- [ ] **Aspiration Windows** - Smalt søkevindu rundt forventet score
- [ ] **Principal Variation Search** - Null-window søk for ikke-PV trekk
- [ ] **Null Move Pruning** - Skip søk når posisjon er overveldende god
- [ ] **Late Move Reductions** - Reduser dybde for senere trekk
- [ ] **Quiescence Search** - Fortsett søk i taktiske stillinger
- [ ] **Parallell søk (Lazy SMP)** - Multi-threaded search

---

## Teknologier

- **Rust** - Kjernemotor (hastighet + sikkerhet)
- **PyO3** - Rust ↔ Python bindings
- **maturin** - Bygg og pakking
- **wide** - Portable SIMD (f32x8)
- **PyTorch** - Neural network trening
- **FastAPI** - Web backend
- **Vanilla JS** - Enkel web frontend

---

## Lisens

MIT License

---

*Inspirert av Stockfish, Leela Chess Zero, og NRKs klassiske Tønnesjakk* 🛢️
