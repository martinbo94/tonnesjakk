use pyo3::prelude::*;
use std::time::{Duration, Instant};


use crate::board::*;
use crate::nnue::*;

/// Størrelse på transposition table (antall entries)
pub const TT_SIZE: usize = 1 << 20; // ~1 million entries

// ============================================================================
// AI: EVALUERING OG SØK
// ============================================================================

/// Flag for transposition table entry
#[derive(Clone, Copy, PartialEq, Eq)]
enum TTFlag {
    Exact,      // Eksakt score
    LowerBound, // Score er minst denne (beta cutoff)
    UpperBound, // Score er høyst denne (alpha cutoff)
}

/// Entry i transposition table
#[derive(Clone)]
struct TTEntry {
    hash: u64,        // Verifiser at det er riktig posisjon
    depth: u8,        // Hvor dypt vi søkte
    score: i32,       // Resultatet
    flag: TTFlag,     // Type score
    generation: u8,   // Søke-generasjon (for aging)
    best_move: Option<Move>,  // Beste trekk (for move ordering)
}

/// ═══════════════════════════════════════════════════════════════
/// TT Clustering: 3 entries per bucket
/// ═══════════════════════════════════════════════════════════════
/// Instead of 1 entry per hash slot, we store 3 entries per cluster.
/// This reduces hash collisions and effectively triples the TT capacity.
/// Replacement uses age+depth priority: priority = depth - age * 8.
#[derive(Clone)]
struct TTCluster {
    entries: [Option<TTEntry>; 3],
}

/// Transposition Table - cache av evaluerte posisjoner
/// Bruker clustered replacement med 3 entries per bucket og age+depth prioritet
struct TranspositionTable {
    clusters: Vec<TTCluster>,
    hits: u64,
    misses: u64,
    generation: u8,   // Økes hver gang søket starter
}

impl TranspositionTable {
    fn new(size: usize) -> Self {
        TranspositionTable {
            clusters: vec![TTCluster { entries: [None, None, None] }; size],
            hits: 0,
            misses: 0,
            generation: 0,
        }
    }

    /// Øk generasjonen (kall ved starten av hvert søk)
    fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    fn index(&self, hash: u64) -> usize {
        (hash as usize) % self.clusters.len()
    }

    fn probe(&mut self, hash: u64) -> Option<&TTEntry> {
        let idx = self.index(hash);
        let cluster = &self.clusters[idx];
        // Search all 3 entries in the cluster for a hash match.
        // No generation check — older entries are still useful for move ordering.
        for entry in &cluster.entries {
            if let Some(ref e) = entry {
                if e.hash == hash {
                    self.hits += 1;
                    return Some(e);
                }
            }
        }
        self.misses += 1;
        None
    }

    fn store(&mut self, hash: u64, depth: u8, score: i32, flag: TTFlag, best_move: Option<Move>) {
        let idx = self.index(hash);
        let cluster = &mut self.clusters[idx];

        // Find best slot: prefer empty, then same hash, then lowest priority
        let mut replace_idx = 0;
        let mut worst_priority = i32::MAX;

        for i in 0..3 {
            match &cluster.entries[i] {
                None => { replace_idx = i; break; }
                Some(e) if e.hash == hash => { replace_idx = i; break; }
                Some(e) => {
                    // Priority = depth - age_penalty
                    // Old entries (high age) get low priority and are replaced first
                    let age = self.generation.wrapping_sub(e.generation) as i32;
                    let priority = e.depth as i32 - age * 8;
                    if priority < worst_priority {
                        worst_priority = priority;
                        replace_idx = i;
                    }
                }
            }
        }

        cluster.entries[replace_idx] = Some(TTEntry {
            hash,
            depth,
            score,
            flag,
            generation: self.generation,
            best_move,
        });
    }

    fn clear(&mut self) {
        // O(1) clear using generation counter - stale entries will have low
        // priority and be replaced first. Probe still works for move ordering.
        self.generation = self.generation.wrapping_add(1);
        self.hits = 0;
        self.misses = 0;
    }
}


/// Resultat fra et søk - inneholder beste trekk og score
#[pyclass]
#[derive(Clone, Debug)]
pub struct SearchResult {
    #[pyo3(get)]
    pub best_move: Option<Move>,
    #[pyo3(get)]
    pub score: i32,
    #[pyo3(get)]
    pub nodes_searched: u64,
    #[pyo3(get)]
    pub cutoffs: u64,
    #[pyo3(get)]
    pub tt_hits: u64,
    #[pyo3(get)]
    pub quiesce_nodes: u64,
    #[pyo3(get)]
    pub depth: u8,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(score={}, depth={}, nodes={}, cutoffs={}, tt_hits={})",
            self.score, self.depth, self.nodes_searched, self.cutoffs, self.tt_hits
        )
    }

    /// Cutoff ratio - høyere er bedre (betyr move ordering fungerer)
    fn cutoff_ratio(&self) -> f64 {
        if self.nodes_searched == 0 {
            0.0
        } else {
            self.cutoffs as f64 / self.nodes_searched as f64
        }
    }

    /// TT hit ratio
    fn tt_hit_ratio(&self) -> f64 {
        if self.nodes_searched == 0 {
            0.0
        } else {
            self.tt_hits as f64 / self.nodes_searched as f64
        }
    }
}

/// AI-motor - wrapper rundt BitBoardEngine for Python-kompatibilitet
#[pyclass]
pub struct Engine {
    /// Den faktiske motoren (bitboard-basert)
    inner: BitBoardEngine,
}

#[pymethods]
impl Engine {
    #[new]
    fn new() -> Self {
        Engine {
            inner: BitBoardEngine::new(),
        }
    }

    /// Tøm transposition table (mellom spill)
    fn clear_tt(&mut self) {
        self.inner.clear_tt();
    }

    /// Full reset - tøm alle caches og tabeller (mellom spill)
    fn full_reset(&mut self) {
        self.inner.full_reset();
    }

    /// Hent TT statistikk
    fn get_tt_stats(&self) -> (u64, u64) {
        self.inner.get_tt_stats()
    }

    /// Søk etter beste trekk
    /// Konverterer Board til BitBoard, kjører BitBoardEngine.search(), og returnerer SearchResult
    fn search(&mut self, board: &Board, depth: u8) -> SearchResult {
        // Konverter Board til BitBoard
        let bb = BitBoard::from_board(board);

        // Søk med BitBoardEngine
        let (score, best_bitmove) = self.inner.search(&bb, depth);

        // Konverter BitMove til Move
        let best_move = best_bitmove.map(|bm| bm.to_move());

        SearchResult {
            best_move,
            score,
            nodes_searched: self.inner.nodes_searched,
            cutoffs: self.inner.cutoffs,
            tt_hits: self.inner.tt_hits,
            quiesce_nodes: self.inner.quiesce_nodes,
            depth,
        }
    }

    /// Iterative deepening: Søk gradvis dypere
    fn search_iterative(&mut self, board: &Board, max_depth: u8) -> SearchResult {
        let bb = BitBoard::from_board(board);
        let mut best_result = SearchResult {
            best_move: None,
            score: 0,
            nodes_searched: 0,
            cutoffs: 0,
            tt_hits: 0,
            quiesce_nodes: 0,
            depth: 0,
        };

        for depth in 1..=max_depth {
            let (score, best_bitmove) = self.inner.search(&bb, depth);
            let best_move = best_bitmove.map(|bm| bm.to_move());

            best_result = SearchResult {
                best_move,
                score,
                nodes_searched: best_result.nodes_searched + self.inner.nodes_searched,
                cutoffs: best_result.cutoffs + self.inner.cutoffs,
                tt_hits: best_result.tt_hits + self.inner.tt_hits,
                quiesce_nodes: best_result.quiesce_nodes + self.inner.quiesce_nodes,
                depth,
            };

            // Stopp tidlig hvis vi fant en vinnersekvens
            if score.abs() > 90_000 {
                break;
            }
        }

        best_result
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Vekter - pass-through til inner BitBoardEngine
    // ─────────────────────────────────────────────────────────────────────────────

    #[getter]
    fn weight_progress(&self) -> i32 {
        self.inner.weight_progress
    }

    #[setter]
    fn set_weight_progress(&mut self, value: i32) {
        self.inner.weight_progress = value;
    }

    #[getter]
    fn weight_center_pail(&self) -> i32 {
        self.inner.weight_center_pail
    }

    #[setter]
    fn set_weight_center_pail(&mut self, value: i32) {
        self.inner.weight_center_pail = value;
    }

    #[getter]
    fn weight_blocking(&self) -> i32 {
        self.inner.weight_blocking
    }

    #[setter]
    fn set_weight_blocking(&mut self, value: i32) {
        self.inner.weight_blocking = value;
    }

    #[getter]
    fn weight_scored(&self) -> i32 {
        self.inner.weight_scored
    }

    #[setter]
    fn set_weight_scored(&mut self, value: i32) {
        self.inner.weight_scored = value;
    }

    #[getter]
    fn weight_threat(&self) -> i32 {
        self.inner.weight_threat
    }

    #[setter]
    fn set_weight_threat(&mut self, value: i32) {
        self.inner.weight_threat = value;
    }

    /// Load NNUE weights from JSON file
    /// After loading, the engine will use NNUE for evaluation instead of heuristics
    fn load_nnue(&mut self, path: &str) -> PyResult<()> {
        self.inner.load_nnue(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load NNUE: {}", e))
        })
    }

    /// Check if NNUE is loaded
    fn has_nnue(&self) -> bool {
        self.inner.nnue.is_some() || self.inner.quantized_nnue.is_some() || self.inner.halfpail_nnue.is_some()
    }

    /// Clear NNUE (revert to heuristic evaluation)
    fn clear_nnue(&mut self) {
        self.inner.clear_nnue();
    }

    /// Skip relational features during NNUE evaluation (for benchmarking)
    fn set_skip_relational(&mut self, skip: bool) {
        self.inner.skip_relational = skip;
    }

    /// Time-based search: iterative deepening that stops when time runs out.
    /// Returns SearchResult with the best move from the last fully completed depth.
    fn search_timed(&mut self, board: &Board, time_ms: u64) -> SearchResult {
        let bb = BitBoard::from_board(board);
        let (score, best_bitmove, depth_reached) = self.inner.search_timed(&bb, time_ms);
        let best_move = best_bitmove.map(|bm| bm.to_move());

        SearchResult {
            best_move,
            score,
            nodes_searched: self.inner.nodes_searched,
            cutoffs: self.inner.cutoffs,
            tt_hits: self.inner.tt_hits,
            quiesce_nodes: self.inner.quiesce_nodes,
            depth: depth_reached,
        }
    }

    /// Expose heuristic evaluation to Python (for MCTS leaf evaluation).
    /// Always uses the hand-crafted heuristic, not NNUE.
    /// Returns score from White's perspective.
    fn evaluate_position(&self, board: &Board) -> i32 {
        let bb = BitBoard::from_board(board);
        self.inner.evaluate_heuristic(&bb)
    }
}


// ============================================================================
// BITBOARD ENGINE - Søk med bitboards og inkrementell NNUE
// ============================================================================

/// AI-motor som bruker BitBoard og inkrementell NNUE
pub struct BitBoardEngine {
    // Statistikk
    pub nodes_searched: u64,
    pub cutoffs: u64,
    pub tt_hits: u64,
    pub quiesce_nodes: u64,
    pub eval_cache_hits: u64,

    // Transposition Table
    tt: TranspositionTable,

    // Evaluation Cache
    eval_cache: EvalCache,

    // Killer moves
    killer_moves: [[Option<BitMove>; 2]; MAX_DEPTH],

    // History heuristic: [from_sq][to_sq] -> bonus score
    // Tracks which moves historically cause cutoffs
    history: [[i32; NUM_SQUARES]; NUM_SQUARES],

    // Continuation history: [prev_to_sq][curr_to_sq] -> score
    // Tracks which moves are good responses to a given previous move
    cont_history: [[i32; NUM_SQUARES]; NUM_SQUARES],

    // Previous move (for continuation history indexing)
    prev_move: Option<BitMove>,

    // NNUE evaluator (f32)
    nnue: Option<IncrementalNNUE>,

    // Accumulator stack (f32)
    acc_stack: AccumulatorStack,

    // Working accumulator for evaluation (reused to avoid allocations)
    eval_acc: Accumulator,

    // Quantized NNUE evaluator (i16) — takes priority over f32 when loaded
    quantized_nnue: Option<QuantizedNNUE>,

    // Quantized accumulator stack
    qacc_stack: QAccumulatorStack,

    // HalfPail NNUE evaluator — takes priority over f32/quantized when loaded
    halfpail_nnue: Option<HalfPailNNUE>,

    // Dual accumulator stack for HalfPail
    dual_acc_stack: DualAccumulatorStack,

    // Skip relational features (for benchmarking)
    skip_relational: bool,

    // Time-based search: deadline for when to abort, checked every 1024 nodes
    deadline: Option<Instant>,
    nodes_since_check: u32,
    search_stopped: bool,
    last_completed_depth: u8,

    // LMR reduction table: lmr_table[depth][move_count] = reduction
    // Precomputed using ln(depth) * ln(move_count) / 2.5
    lmr_table: [[u8; 64]; 32],

    // Fallback heuristisk vekter
    pub weight_progress: i32,
    pub weight_center_pail: i32,
    pub weight_blocking: i32,
    pub weight_scored: i32,
    pub weight_threat: i32,
}

impl Default for BitBoardEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BitBoardEngine {
    pub fn new() -> Self {
        // Precompute LMR reduction table: R = ln(depth) * ln(move_count)
        // Divisor 1.0 tuned for 6x6 board (shallower depths than standard chess)
        let mut lmr_table = [[0u8; 64]; 32];
        for d in 1..32 {
            for m in 1..64 {
                lmr_table[d][m] = ((d as f64).ln() * (m as f64).ln() / 1.0) as u8;
            }
        }

        BitBoardEngine {
            nodes_searched: 0,
            cutoffs: 0,
            tt_hits: 0,
            quiesce_nodes: 0,
            eval_cache_hits: 0,
            tt: TranspositionTable::new(TT_SIZE),
            eval_cache: EvalCache::new(),
            killer_moves: std::array::from_fn(|_| [None, None]),
            history: [[0; NUM_SQUARES]; NUM_SQUARES],
            cont_history: [[0i32; NUM_SQUARES]; NUM_SQUARES],
            prev_move: None,
            nnue: None,
            acc_stack: AccumulatorStack::new(),
            eval_acc: Accumulator::default(),
            quantized_nnue: None,
            qacc_stack: QAccumulatorStack::new(),
            halfpail_nnue: None,
            dual_acc_stack: DualAccumulatorStack::new(),
            skip_relational: false,
            deadline: None,
            nodes_since_check: 0,
            search_stopped: false,
            last_completed_depth: 0,
            lmr_table,
            weight_progress: 80,
            weight_center_pail: 15,
            weight_blocking: 20,
            weight_scored: 700,
            weight_threat: 150,
        }
    }

    /// Clear history table (call between games)
    pub fn clear_history(&mut self) {
        self.history = [[0; NUM_SQUARES]; NUM_SQUARES];
        self.cont_history = [[0i32; NUM_SQUARES]; NUM_SQUARES];
        self.prev_move = None;
    }

    /// Age history table (reduce values to prevent overflow and adapt to position)
    fn age_history(&mut self) {
        for from in 0..NUM_SQUARES {
            for to in 0..NUM_SQUARES {
                self.history[from][to] /= 2;
                self.cont_history[from][to] /= 2;
            }
        }
    }

    /// Update history on beta cutoff
    #[inline]
    fn update_history(&mut self, mv: &BitMove, depth: u8) {
        if mv.is_placement() {
            return; // Skip placements for history
        }
        if let Some(from_sq) = mv.barrel_from() {
            let to_sq = mv.barrel_to();
            let bonus = (depth as i32) * (depth as i32);
            // Cap history values to prevent overflow
            self.history[from_sq as usize][to_sq as usize] =
                (self.history[from_sq as usize][to_sq as usize] + bonus).min(10_000);
        }
    }

    /// Last NNUE-modell (auto-detects halfpail vs quantized vs f32 format)
    pub fn load_nnue(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json_str = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&json_str)?;

        if value.get("halfpail").and_then(|v| v.as_bool()).unwrap_or(false) {
            // HalfPail dual-perspective sparse format
            let json: HalfPailJson = serde_json::from_value(value)?;
            let hp = HalfPailNNUE::from_json(json)?;
            self.halfpail_nnue = Some(hp);
            self.nnue = None;
            self.quantized_nnue = None;
        } else if value.get("quantized").and_then(|v| v.as_bool()).unwrap_or(false) {
            // Quantized int16 format
            let json: QuantizedNNUEJson = serde_json::from_value(value)?;
            let qnnue = QuantizedNNUE::from_json(json)?;
            self.quantized_nnue = Some(qnnue);
            self.nnue = None;
            self.halfpail_nnue = None;
        } else {
            // Standard f32 format
            let model: NNUEModel = serde_json::from_value(value)?;
            let fc1_weight: Vec<f32> = model.weights.fc1_weight.into_iter().flatten().collect();
            let fc2_weight: Vec<f32> = model.weights.fc2_weight.into_iter().flatten().collect();
            let fc3_weight: Vec<f32> = model.weights.fc3_weight.into_iter().flatten().collect();
            let input_size = fc1_weight.len() / model.hidden1;
            let hidden1 = model.hidden1;
            let mut fc1_weight_t = vec![0.0f32; input_size * hidden1];
            for neuron in 0..hidden1 {
                for feature in 0..input_size {
                    fc1_weight_t[feature * hidden1 + neuron] = fc1_weight[neuron * input_size + feature];
                }
            }
            let nnue = IncrementalNNUE {
                fc1_weight,
                fc1_weight_t,
                fc1_bias: model.weights.fc1_bias,
                fc2_weight,
                fc2_bias: model.weights.fc2_bias,
                fc3_weight,
                fc3_bias: model.weights.fc3_bias[0],
                hidden1,
                hidden2: model.hidden2,
                input_size,
            };
            self.nnue = Some(nnue);
            self.quantized_nnue = None;
            self.halfpail_nnue = None;
        }
        Ok(())
    }

    /// Clear NNUE (revert to heuristic evaluation)
    pub fn clear_nnue(&mut self) {
        self.nnue = None;
        self.quantized_nnue = None;
        self.halfpail_nnue = None;
    }

    /// Tøm TT
    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    /// Full reset - tøm alle caches og tabeller (mellom spill)
    pub fn full_reset(&mut self) {
        self.tt.clear();
        self.eval_cache.clear();
        self.clear_history();
        self.killer_moves = std::array::from_fn(|_| [None, None]);
        self.acc_stack.reset();
        self.qacc_stack.reset();
        self.dual_acc_stack.reset();
        self.nodes_searched = 0;
        self.cutoffs = 0;
        self.tt_hits = 0;
        self.quiesce_nodes = 0;
        self.eval_cache_hits = 0;
    }

    /// Hent TT statistikk
    pub fn get_tt_stats(&self) -> (u64, u64) {
        (self.tt.hits, self.tt.misses)
    }

    /// Heuristisk evaluering (fallback når NNUE ikke er lastet)
    pub fn evaluate_heuristic(&self, bb: &BitBoard) -> i32 {
        if let Some(winner) = bb.check_winner() {
            return match winner {
                Player::White => 100_000,
                Player::Black => -100_000,
            };
        }

        let mut score = 0;

        // Poeng for scorede tønner (big bonus)
        score += (bb.white_scored as i32 - bb.black_scored as i32) * self.weight_scored;

        // Fremgang + trussel-bonus for tønner nær mål
        let mut white_progress = 0i32;
        let mut white_threats = 0i32; // Tønner på rad 1 (kan score neste trekk)
        let mut bb_white = bb.white_barrels;
        while bb_white != 0 {
            let sq = bb_white.trailing_zeros() as usize;
            let (row, _) = sq_to_coords(sq);
            let dist_to_goal = row; // White's goal is row 0
            white_progress += (BOARD_SIZE - 1 - row) as i32;
            if dist_to_goal == 1 {
                white_threats += 1; // Barrel can score next move
            }
            bb_white &= bb_white - 1;
        }

        let mut black_progress = 0i32;
        let mut black_threats = 0i32; // Tønner på rad 4 (kan score neste trekk)
        let mut bb_black = bb.black_barrels;
        while bb_black != 0 {
            let sq = bb_black.trailing_zeros() as usize;
            let (row, _) = sq_to_coords(sq);
            let dist_to_goal = (BOARD_SIZE - 1) - row; // Black's goal is row 5
            black_progress += row as i32;
            if dist_to_goal == 1 {
                black_threats += 1; // Barrel can score next move
            }
            bb_black &= bb_black - 1;
        }

        score += (white_progress - black_progress) * self.weight_progress;
        score += (white_threats - black_threats) * self.weight_threat; // Immediate threats are valuable

        // Pail-posisjon: senterkontroll + blokkering
        // White's ideal pail position is in opponent's half (rows 0-2), centered
        if bb.white_pail != 0 {
            let sq = bb.white_pail.trailing_zeros() as usize;
            let (row, col) = sq_to_coords(sq);
            // Center is (2.5, 2.5), use integer approximation
            let center_dist = ((row as i32 - 2).abs() + (col as i32 - 2).abs()) as i32;
            score += (6 - center_dist) * self.weight_center_pail;

            // Blocking bonus: pail in front of black barrels
            let mut blocking_bonus = 0i32;
            let mut bb_opp = bb.black_barrels;
            while bb_opp != 0 {
                let opp_sq = bb_opp.trailing_zeros() as usize;
                let (opp_row, opp_col) = sq_to_coords(opp_sq);
                // Pail blocks if same column and ahead of opponent
                if col == opp_col && row > opp_row {
                    blocking_bonus += self.weight_blocking;
                }
                bb_opp &= bb_opp - 1;
            }
            score += blocking_bonus;
        }
        // Black's pail (symmetric to white)
        if bb.black_pail != 0 {
            let sq = bb.black_pail.trailing_zeros() as usize;
            let (row, col) = sq_to_coords(sq);
            let center_dist = ((row as i32 - 3).abs() + (col as i32 - 3).abs()) as i32;
            score -= (6 - center_dist) * self.weight_center_pail;

            // Blocking bonus: pail in front of white barrels
            let mut blocking_bonus = 0i32;
            let mut bb_opp = bb.white_barrels;
            while bb_opp != 0 {
                let opp_sq = bb_opp.trailing_zeros() as usize;
                let (opp_row, opp_col) = sq_to_coords(opp_sq);
                // Pail blocks if same column and ahead of opponent
                if col == opp_col && row < opp_row {
                    blocking_bonus += self.weight_blocking;
                }
                bb_opp &= bb_opp - 1;
            }
            score -= blocking_bonus;
        }

        score
    }

    /// Evaluer posisjon (bruker NNUE hvis tilgjengelig, med caching)
    fn evaluate(&mut self, bb: &BitBoard) -> i32 {
        let hash = bb.hash;

        // Check eval cache first (works for both NNUE and heuristic)
        if let Some(score) = self.eval_cache.probe(hash) {
            self.eval_cache_hits += 1;
            return score;
        }

        // HalfPail NNUE (highest priority — best feature representation)
        if self.halfpail_nnue.is_some() {
            let dual_acc = self.dual_acc_stack.current();
            let hp = self.halfpail_nnue.as_ref().unwrap();
            let score = hp.evaluate_from_dual_acc(bb, dual_acc);
            self.eval_cache.store(hash, score);
            score
        }
        // Quantized NNUE (preferred — faster)
        else if self.quantized_nnue.is_some() {
            let base_acc = self.qacc_stack.current();
            let qnnue = self.quantized_nnue.as_ref().unwrap();
            let score = qnnue.evaluate_from_qacc(bb, base_acc);
            self.eval_cache.store(hash, score);
            score
        }
        // Float NNUE (fallback)
        else if self.nnue.is_some() {
            let base_acc = self.acc_stack.current();
            let pre_activation = base_acc.pre_activation;

            let nnue = self.nnue.as_ref().unwrap();
            let eval_acc = &mut self.eval_acc;

            eval_acc.pre_activation = pre_activation;

            if !self.skip_relational && nnue.input_size > BASE_FEATURES {
                nnue.add_relational_features(bb, eval_acc);
            }

            eval_acc.apply_relu();
            let score = (nnue.evaluate_from_accumulator(eval_acc) * 1000.0) as i32;

            self.eval_cache.store(hash, score);
            score
        } else {
            // Heuristisk eval (cache already checked above)
            let score = self.evaluate_heuristic(bb);
            self.eval_cache.store(hash, score);
            score
        }
    }

    /// Hent eval cache statistikk
    pub fn get_eval_cache_stats(&self) -> (u64, f64) {
        (self.eval_cache.hits, self.eval_cache.hit_ratio())
    }

    /// Tøm eval cache
    pub fn clear_eval_cache(&mut self) {
        self.eval_cache.clear();
    }

    /// Score et trekk for move ordering
    fn score_move(&self, mv: &BitMove, player: Player, depth: usize, tt_move: Option<&BitMove>) -> i32 {
        let mut score = 0;
        let goal_row = match player {
            Player::White => 0,
            Player::Black => BOARD_SIZE - 1,
        };

        // TT-trekk har høyest prioritet
        if let Some(tt_mv) = tt_move {
            if mv.packed == tt_mv.packed {
                return 10_000;
            }
        }

        // Killer moves
        if depth < MAX_DEPTH {
            if let Some(ref k1) = self.killer_moves[depth][0] {
                if mv.packed == k1.packed {
                    score += 5_000;
                }
            }
            if let Some(ref k2) = self.killer_moves[depth][1] {
                if mv.packed == k2.packed {
                    score += 4_000;
                }
            }
        }

        // History heuristic - add historical cutoff bonus
        if !mv.is_placement() {
            if let Some(from_sq) = mv.barrel_from() {
                let to_sq = mv.barrel_to() as usize;
                // Scale history to be below killer moves but significant
                score += self.history[from_sq as usize][to_sq] / 10;
            }
        }

        // Continuation history - weight 2x relative to butterfly history
        if let Some(ref pm) = self.prev_move {
            let prev_to = pm.barrel_to() as usize;
            let curr_to = mv.barrel_to() as usize;
            score += 2 * self.cont_history[prev_to][curr_to] / 10;
        }

        let to_sq = mv.barrel_to() as usize;
        let (to_row, to_col) = sq_to_coords(to_sq);

        // Når mål
        if to_row == goal_row {
            score += 500;
        }

        if mv.is_placement() {
            score += 50;
            let center_col = BOARD_SIZE / 2;
            let col_dist = (to_col as i32 - center_col as i32).abs();
            score += (3 - col_dist) * 10;
        } else {
            if let Some(from_sq) = mv.barrel_from() {
                let (from_row, _) = sq_to_coords(from_sq as usize);
                let forward = match player {
                    Player::White => from_row as i32 - to_row as i32,
                    Player::Black => to_row as i32 - from_row as i32,
                };
                score += forward * 100;
            }

            // Hopp-bonus
            let path_len = mv.path_len();
            if path_len > 1 {
                score += (path_len as i32 - 1) * 50;
            }
        }

        // Pail-plassering bonus
        if mv.pail_pos().is_some() {
            score += 20;
        }

        score
    }

    /// Lagre killer move
    fn store_killer(&mut self, mv: &BitMove, depth: usize) {
        if depth >= MAX_DEPTH {
            return;
        }

        if let Some(ref k1) = self.killer_moves[depth][0] {
            if mv.packed == k1.packed {
                return;
            }
        }

        self.killer_moves[depth][1] = self.killer_moves[depth][0];
        self.killer_moves[depth][0] = Some(*mv);
    }

    /// Sorter trekk
    fn order_moves(&self, mut moves: Vec<BitMove>, player: Player, depth: usize, tt_move: Option<&BitMove>) -> Vec<BitMove> {
        moves.sort_by(|a, b| {
            let score_a = self.score_move(a, player, depth, tt_move);
            let score_b = self.score_move(b, player, depth, tt_move);
            score_b.cmp(&score_a)
        });
        moves
    }

    /// Søk etter beste trekk med Aspiration Windows
    pub fn search(&mut self, bb: &BitBoard, depth: u8) -> (i32, Option<BitMove>) {
        self.search_with_aspiration(bb, depth, None)
    }

    /// Søk med aspiration windows - smalt vindu rundt forventet score
    fn search_with_aspiration(&mut self, bb: &BitBoard, depth: u8, prev_score: Option<i32>) -> (i32, Option<BitMove>) {
        self.nodes_searched = 0;
        self.cutoffs = 0;
        self.tt_hits = 0;
        self.quiesce_nodes = 0;

        // Ny søke-generasjon (for TT aging) - kun ved første søk
        if prev_score.is_none() {
            self.tt.new_search();
        }

        // Nullstill killer moves
        for km in &mut self.killer_moves {
            km[0] = None;
            km[1] = None;
        }

        // Age history (don't clear - accumulated knowledge is valuable)
        self.age_history();

        // Initialiser accumulator med full evaluering
        self.acc_stack.reset();
        self.qacc_stack.reset();
        self.dual_acc_stack.reset();
        if let Some(ref hp) = self.halfpail_nnue {
            let dual_acc = self.dual_acc_stack.current_mut();
            hp.init_accumulators(bb, dual_acc);
        } else if let Some(ref qnnue) = self.quantized_nnue {
            let acc = self.qacc_stack.current_mut();
            qnnue.init_accumulator(bb, acc);
        } else if let Some(ref nnue) = self.nnue {
            let acc = self.acc_stack.current_mut();
            for i in 0..nnue.hidden1 {
                acc.pre_activation[i] = nnue.fc1_bias[i];
            }
            nnue.add_features_from_bitboard(bb, acc);
            acc.apply_relu();
        }

        let maximizing = bb.current_player == Player::White;

        // ═══════════════════════════════════════════════════════════════
        // ASPIRATION WINDOWS
        // ═══════════════════════════════════════════════════════════════
        // Start med smalt vindu rundt forventet score, utvid ved fail
        const ASPIRATION_WINDOW: i32 = 50;

        let (mut alpha, mut beta) = match prev_score {
            Some(score) => (score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW),
            None => (i32::MIN + 1, i32::MAX - 1),
        };

        let mut best_move;
        let mut score;

        loop {
            let (s, mv) = self.minimax(bb, depth, alpha, beta, maximizing);
            score = s;
            best_move = mv;

            // Sjekk om score er innenfor vinduet
            if score <= alpha {
                // Fail low - utvid nedre grense
                alpha = i32::MIN + 1;
            } else if score >= beta {
                // Fail high - utvid øvre grense
                beta = i32::MAX - 1;
            } else {
                // Score innenfor vinduet - ferdig
                break;
            }

            // Hvis begge grenser er utvidet, bruk fullt vindu
            if alpha == i32::MIN + 1 && beta == i32::MAX - 1 {
                break;
            }
        }

        (score, best_move)
    }

    /// Quiescence Search - fortsett søk i taktiske stillinger ved depth 0
    /// Søker kun "spennende" trekk: tønner som når/nærmer seg mål
    /// qsdepth: current quiescence depth (starts at 0, max MAX_QSEARCH_DEPTH)
    fn quiesce(&mut self, bb: &BitBoard, mut alpha: i32, beta: i32, maximizing: bool, qsdepth: u8) -> i32 {
        const MAX_QSEARCH_DEPTH: u8 = 8; // Prevent stack overflow

        self.quiesce_nodes += 1;

        // Stand-pat: kan vi bare evaluere og returnere?
        let stand_pat = self.evaluate(bb);

        // Sjekk for vinner
        if bb.check_winner().is_some() {
            return stand_pat;
        }

        // Prevent stack overflow from unbounded quiescence search
        if qsdepth >= MAX_QSEARCH_DEPTH {
            return stand_pat;
        }

        if maximizing {
            if stand_pat >= beta {
                return beta; // Beta cutoff
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        } else {
            if stand_pat <= alpha {
                return alpha; // Alpha cutoff
            }
            // For minimizing, we use stand_pat as upper bound
        }

        // Finn "taktiske" trekk: tønner nær mål, eller fremover-trekk ved dist 2
        // Expanded from dist<=1 to also catch 2-step scoring sequences
        let player = bb.current_player;

        let moves = bb.generate_moves();
        let tactical_moves: Vec<BitMove> = moves
            .into_iter()
            .filter(|mv| {
                let to_sq = mv.barrel_to() as usize;
                let (to_row, _) = sq_to_coords(to_sq);
                let dist_to_goal = if player == Player::White {
                    to_row // White's goal is row 0
                } else {
                    BOARD_SIZE - 1 - to_row // Black's goal is row 5
                };

                // Always include barrels within 1 step of goal
                if dist_to_goal <= 1 {
                    return true;
                }

                // At distance 2: only include if barrel moved forward toward goal
                if dist_to_goal == 2 {
                    if let Some(from_sq) = mv.barrel_from() {
                        let (from_row, _) = sq_to_coords(from_sq as usize);
                        let from_dist = if player == Player::White {
                            from_row
                        } else {
                            BOARD_SIZE - 1 - from_row
                        };
                        // Only tactical if moving closer to goal
                        return from_dist > dist_to_goal;
                    }
                }

                false
            })
            .collect();

        // Ingen taktiske trekk - returner stand-pat
        if tactical_moves.is_empty() {
            return stand_pat;
        }

        if maximizing {
            let mut best = stand_pat;
            for mv in tactical_moves {
                let mut new_bb = *bb;
                new_bb.make_move(&mv);

                // Oppdater accumulator (halfpail, quantized, or float)
                if self.halfpail_nnue.is_some() {
                    let hp = self.halfpail_nnue.as_ref().unwrap();
                    let (w_deltas, b_deltas, w_recomp, b_recomp) = hp.compute_move_deltas(bb, &mv);
                    self.dual_acc_stack.push();
                    if w_recomp || b_recomp {
                        let hp = self.halfpail_nnue.as_ref().unwrap();
                        hp.init_accumulators(&new_bb, self.dual_acc_stack.current_mut());
                    } else {
                        let hp = self.halfpail_nnue.as_ref().unwrap();
                        let dual_acc = self.dual_acc_stack.current_mut();
                        hp.apply_deltas(&mut dual_acc.white_pre, &w_deltas);
                        hp.apply_deltas(&mut dual_acc.black_pre, &b_deltas);
                    }
                } else if let Some(ref qnnue) = self.quantized_nnue {
                    let deltas = compute_nnue_move_deltas(bb, &mv);
                    self.qacc_stack.push();
                    let acc = self.qacc_stack.current_mut();
                    qnnue.apply_deltas_q(acc, &deltas);
                } else if self.nnue.is_some() {
                    let nnue = self.nnue.as_ref().unwrap();
                    let deltas = nnue.compute_move_deltas(bb, &mv);
                    self.acc_stack.push();
                    let acc = self.acc_stack.current_mut();
                    nnue.apply_deltas(acc, &deltas);
                }

                let score = self.quiesce(&new_bb, alpha, beta, false, qsdepth + 1);

                if self.halfpail_nnue.is_some() {
                    self.dual_acc_stack.pop();
                } else if self.quantized_nnue.is_some() {
                    self.qacc_stack.pop();
                } else if self.nnue.is_some() {
                    self.acc_stack.pop();
                }

                best = best.max(score);
                alpha = alpha.max(score);
                if alpha >= beta {
                    break; // Beta cutoff
                }
            }
            best
        } else {
            let mut best = stand_pat;
            for mv in tactical_moves {
                let mut new_bb = *bb;
                new_bb.make_move(&mv);

                // Oppdater accumulator (halfpail, quantized, or float)
                if self.halfpail_nnue.is_some() {
                    let hp = self.halfpail_nnue.as_ref().unwrap();
                    let (w_deltas, b_deltas, w_recomp, b_recomp) = hp.compute_move_deltas(bb, &mv);
                    self.dual_acc_stack.push();
                    if w_recomp || b_recomp {
                        let hp = self.halfpail_nnue.as_ref().unwrap();
                        hp.init_accumulators(&new_bb, self.dual_acc_stack.current_mut());
                    } else {
                        let hp = self.halfpail_nnue.as_ref().unwrap();
                        let dual_acc = self.dual_acc_stack.current_mut();
                        hp.apply_deltas(&mut dual_acc.white_pre, &w_deltas);
                        hp.apply_deltas(&mut dual_acc.black_pre, &b_deltas);
                    }
                } else if let Some(ref qnnue) = self.quantized_nnue {
                    let deltas = compute_nnue_move_deltas(bb, &mv);
                    self.qacc_stack.push();
                    let acc = self.qacc_stack.current_mut();
                    qnnue.apply_deltas_q(acc, &deltas);
                } else if self.nnue.is_some() {
                    let nnue = self.nnue.as_ref().unwrap();
                    let deltas = nnue.compute_move_deltas(bb, &mv);
                    self.acc_stack.push();
                    let acc = self.acc_stack.current_mut();
                    nnue.apply_deltas(acc, &deltas);
                }

                let score = self.quiesce(&new_bb, alpha, beta, true, qsdepth + 1);

                if self.halfpail_nnue.is_some() {
                    self.dual_acc_stack.pop();
                } else if self.quantized_nnue.is_some() {
                    self.qacc_stack.pop();
                } else if self.nnue.is_some() {
                    self.acc_stack.pop();
                }

                best = best.min(score);
                if score <= alpha {
                    break; // Alpha cutoff
                }
            }
            best
        }
    }

    /// Minimax med alpha-beta, PVS, og LMR
    fn minimax(
        &mut self,
        bb: &BitBoard,
        depth: u8,
        mut alpha: i32,
        mut beta: i32,
        maximizing: bool,
    ) -> (i32, Option<BitMove>) {
        self.nodes_searched += 1;

        // Time check: abort search if deadline exceeded
        if self.should_stop() {
            return (0, None);
        }

        let original_alpha = alpha;
        let mut depth = depth; // Make depth mutable for IIR

        // TT lookup - extract data first to avoid borrow issues
        let hash = bb.hash;
        let mut tt_move: Option<BitMove> = None;

        let tt_result = if let Some(entry) = self.tt.probe(hash) {
            self.tt_hits += 1;
            // Clone the move so we can use it after the borrow ends
            let mv_clone = entry.best_move.clone();
            Some((entry.depth, entry.score, entry.flag, mv_clone))
        } else {
            None
        };

        if let Some((tt_depth, tt_score, tt_flag, ref tt_mv_opt)) = tt_result {
            // Now convert the move outside the borrow
            if let Some(ref mv) = tt_mv_opt {
                tt_move = Some(Self::move_to_bitmove_static(mv));
            }

            if tt_depth >= depth {
                match tt_flag {
                    TTFlag::Exact => {
                        return (tt_score, tt_move);
                    }
                    TTFlag::LowerBound => alpha = alpha.max(tt_score),
                    TTFlag::UpperBound => beta = beta.min(tt_score),
                }

                if alpha >= beta {
                    return (tt_score, tt_move);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // IIR: Internal Iterative Reduction
        // ═══════════════════════════════════════════════════════════════
        // Without a TT move, the search has no good first move for PVS.
        // Reduce depth by 1 — the shallower result will populate the TT
        // for the next iterative deepening iteration.
        if tt_move.is_none() && depth >= 4 {
            depth -= 1;
        }

        // ═══════════════════════════════════════════════════════════════
        // ENDGAME DETECTION: fewer barrels = more tactical, less pruning
        // ═══════════════════════════════════════════════════════════════
        // When few barrels remain on the board, every move is critical.
        // Disable or reduce aggressive pruning to avoid missing winning moves.
        let total_remaining = (4u8.saturating_sub(bb.white_scored)) + (4u8.saturating_sub(bb.black_scored));
        let is_endgame = total_remaining <= 3;

        // Terminal node
        if bb.check_winner().is_some() {
            return (self.evaluate(bb), None);
        }

        // ═══════════════════════════════════════════════════════════════
        // QUIESCENCE SEARCH ved depth 0
        // ═══════════════════════════════════════════════════════════════
        if depth == 0 {
            let score = self.quiesce(bb, alpha, beta, maximizing, 0);
            return (score, None);
        }

        // Static evaluation for pruning decisions
        let static_eval = self.evaluate(bb);

        // ═══════════════════════════════════════════════════════════════
        // RAZORING
        // ═══════════════════════════════════════════════════════════════
        // When static eval is far below alpha (or above beta for minimizer),
        // drop to quiescence search. If even qsearch can't save the
        // position, prune the entire subtree.
        if depth <= 3 && !is_endgame {
            let razor_margin = 200 + 150 * depth as i32;
            if maximizing && static_eval + razor_margin < alpha {
                let qscore = self.quiesce(bb, alpha, beta, maximizing, 0);
                if qscore < alpha {
                    return (qscore, None);
                }
            }
            if !maximizing && static_eval - razor_margin > beta {
                let qscore = self.quiesce(bb, alpha, beta, maximizing, 0);
                if qscore > beta {
                    return (qscore, None);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // NULL MOVE PRUNING (tuned: R=2-3 + depth/eval boosts)
        // ═══════════════════════════════════════════════════════════════
        // If giving opponent a free move still results in a beta cutoff,
        // this position is so good we can prune.
        // Only use when position is already favorable (otherwise unlikely to cutoff)
        let nmp_margin = 50; // Only try NMP if we're at least this much better
        let nmp_allowed = depth >= 4
            && !is_endgame
            && static_eval.abs() < 90_000
            && !bb.has_barrel_near_goal()
            && beta.abs() < 90_000
            && if maximizing {
                static_eval >= beta - nmp_margin
            } else {
                static_eval <= alpha + nmp_margin
            };

        if nmp_allowed {
            // Base reduction: R=2 shallow, R=3 deeper (proven for 6x6)
            let mut r: u8 = if depth >= 6 { 3 } else { 2 };
            // Depth-scaling boost: at high depths we have more margin
            if depth >= 8 {
                r += 1;
            }
            // Eval-based boost: if eval strongly exceeds the bound, prune harder
            if maximizing && static_eval >= beta + 150 {
                r += 1;
            }
            if !maximizing && static_eval <= alpha - 150 {
                r += 1;
            }
            let null_depth = (depth as i16 - r as i16 - 1).max(1) as u8;

            // Make null move (swap sides without moving)
            let mut new_bb = *bb;
            new_bb.make_null_move();

            // Search with null window around beta
            let (null_score, _) = if maximizing {
                // White is maximizing - after null move, black searches to minimize
                self.minimax(&new_bb, null_depth, beta - 1, beta, false)
            } else {
                // Black is minimizing - after null move, white searches to maximize
                self.minimax(&new_bb, null_depth, alpha, alpha + 1, true)
            };

            // Check for cutoff
            if maximizing && null_score >= beta {
                return (beta, None);
            }
            if !maximizing && null_score <= alpha {
                return (alpha, None);
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // FUTILITY PRUNING (extended to depth 8)
        // ═══════════════════════════════════════════════════════════════
        // At shallow-to-medium depths, if the static evaluation is far
        // below alpha, we can skip searching most moves (they won't raise
        // alpha). Margins scale super-linearly with depth.
        const FUTILITY_MARGINS: [i32; 9] = [0, 80, 160, 250, 350, 450, 600, 750, 950];
        // Use half the futility margin in endgame (prune less aggressively)
        let margin = if is_endgame { FUTILITY_MARGINS[depth.min(8) as usize] / 2 } else { FUTILITY_MARGINS[depth.min(8) as usize] };
        let futility_pruning = depth <= 8
            && static_eval.abs() < 90_000 // Not near mate
            && if maximizing {
                static_eval + margin < alpha
            } else {
                static_eval - margin > beta
            };

        // Generate and order moves
        let moves = bb.generate_moves();
        if moves.is_empty() {
            return (static_eval, None);
        }

        let sorted_moves = self.order_moves(moves, bb.current_player, depth as usize, tt_move.as_ref());

        let mut best_move = None;
        let mut best_score = if maximizing { i32::MIN + 1 } else { i32::MAX - 1 };
        let mut moves_searched = 0;

        // Save previous move for continuation history
        let prev_mv = self.prev_move;

        for mv in sorted_moves {
            // ═══════════════════════════════════════════════════════════════
            // FUTILITY PRUNING - Skip futile moves
            // ═══════════════════════════════════════════════════════════════
            if futility_pruning && moves_searched > 0 {
                // Don't prune moves that reach goal (high tactical value)
                let to_sq = mv.barrel_to() as usize;
                let (to_row, _) = sq_to_coords(to_sq);
                let goal_row = bb.goal_row(bb.current_player);
                if to_row != goal_row {
                    continue; // Prune this move
                }
            }

            // Make move
            let mut new_bb = *bb;
            let _undo = new_bb.make_move(&mv);

            // Oppdater accumulator inkrementelt (halfpail, quantized, or float)
            if self.halfpail_nnue.is_some() {
                let hp = self.halfpail_nnue.as_ref().unwrap();
                let (w_deltas, b_deltas, w_recompute, b_recompute) = hp.compute_move_deltas(bb, &mv);
                self.dual_acc_stack.push();
                let dual_acc = self.dual_acc_stack.current_mut();
                if w_recompute || b_recompute {
                    // One or both perspectives need full recompute (pail placement)
                    let hp = self.halfpail_nnue.as_ref().unwrap();
                    if w_recompute && b_recompute {
                        hp.init_accumulators(&new_bb, dual_acc);
                    } else if w_recompute {
                        // Recompute white, apply deltas to black
                        let hp = self.halfpail_nnue.as_ref().unwrap();
                        // Reset white to bias and rebuild
                        for i in 0..hp.hidden1 {
                            dual_acc.white_pre[i] = hp.fc1_bias[i];
                        }
                        // Re-init white perspective from new_bb
                        let w_bucket = if new_bb.white_pail != 0 {
                            new_bb.white_pail.trailing_zeros() as usize
                        } else { 36 };
                        let mut barrels = new_bb.white_barrels;
                        while barrels != 0 {
                            let sq = barrels.trailing_zeros() as usize;
                            hp.add_feature(&mut dual_acc.white_pre,
                                halfpail_feature_index(w_bucket, sq, 0) as usize);
                            barrels &= barrels - 1;
                        }
                        barrels = new_bb.black_barrels;
                        while barrels != 0 {
                            let sq = barrels.trailing_zeros() as usize;
                            hp.add_feature(&mut dual_acc.white_pre,
                                halfpail_feature_index(w_bucket, sq, 1) as usize);
                            barrels &= barrels - 1;
                        }
                        if new_bb.black_pail != 0 {
                            let sq = new_bb.black_pail.trailing_zeros() as usize;
                            hp.add_feature(&mut dual_acc.white_pre,
                                halfpail_feature_index(w_bucket, sq, 2) as usize);
                        }
                        // Apply deltas to black
                        hp.apply_deltas(&mut dual_acc.black_pre, &b_deltas);
                    } else {
                        // b_recompute: recompute black, apply deltas to white
                        let hp = self.halfpail_nnue.as_ref().unwrap();
                        for i in 0..hp.hidden1 {
                            dual_acc.black_pre[i] = hp.fc1_bias[i];
                        }
                        let b_bucket = if new_bb.black_pail != 0 {
                            new_bb.black_pail.trailing_zeros() as usize
                        } else { 36 };
                        let mut barrels = new_bb.black_barrels;
                        while barrels != 0 {
                            let sq = barrels.trailing_zeros() as usize;
                            hp.add_feature(&mut dual_acc.black_pre,
                                halfpail_feature_index(b_bucket, sq, 0) as usize);
                            barrels &= barrels - 1;
                        }
                        barrels = new_bb.white_barrels;
                        while barrels != 0 {
                            let sq = barrels.trailing_zeros() as usize;
                            hp.add_feature(&mut dual_acc.black_pre,
                                halfpail_feature_index(b_bucket, sq, 1) as usize);
                            barrels &= barrels - 1;
                        }
                        if new_bb.white_pail != 0 {
                            let sq = new_bb.white_pail.trailing_zeros() as usize;
                            hp.add_feature(&mut dual_acc.black_pre,
                                halfpail_feature_index(b_bucket, sq, 2) as usize);
                        }
                        // Apply deltas to white
                        hp.apply_deltas(&mut dual_acc.white_pre, &w_deltas);
                    }
                } else {
                    let hp = self.halfpail_nnue.as_ref().unwrap();
                    hp.apply_deltas(&mut dual_acc.white_pre, &w_deltas);
                    hp.apply_deltas(&mut dual_acc.black_pre, &b_deltas);
                }
            } else if let Some(ref qnnue) = self.quantized_nnue {
                let deltas = compute_nnue_move_deltas(bb, &mv);
                self.qacc_stack.push();
                let acc = self.qacc_stack.current_mut();
                qnnue.apply_deltas_q(acc, &deltas);
            } else if self.nnue.is_some() {
                let nnue = self.nnue.as_ref().unwrap();
                let deltas = nnue.compute_move_deltas(bb, &mv);
                self.acc_stack.push();
                let acc = self.acc_stack.current_mut();
                nnue.apply_deltas(acc, &deltas);
            }

            let score;

            // Set prev_move for continuation history in child nodes
            self.prev_move = Some(mv);

            if moves_searched == 0 {
                // ═══════════════════════════════════════════════════════════════
                // PVS: Første trekk - fullt vindu (Principal Variation)
                // ═══════════════════════════════════════════════════════════════
                let (s, _) = self.minimax(&new_bb, depth - 1, alpha, beta, !maximizing);
                score = s;
            } else {
                // ═══════════════════════════════════════════════════════════════
                // LMR: Late Move Reductions (logarithmic table + history modulation)
                // ═══════════════════════════════════════════════════════════════
                // Precomputed table gives graduated reductions based on depth and move index
                let mut reduction: u8 = 0;
                if depth >= 3 && moves_searched >= 2 {
                    reduction = self.lmr_table[depth.min(31) as usize][moves_searched.min(63) as usize];
                    // History modulation: good moves get less reduction, bad moves get more
                    if let Some(from) = mv.barrel_from() {
                        let to = mv.barrel_to() as usize;
                        let from = from as usize;
                        if self.history[from][to] > 1000 { reduction = reduction.saturating_sub(1); }
                        if self.history[from][to] < -500 { reduction += 1; }
                        // Don't reduce goal-reaching moves
                        let (to_row, _) = sq_to_coords(to);
                        if to_row == bb.goal_row(bb.current_player) { reduction = 0; }
                    }
                    // In endgame, reduce LMR reductions (every move matters)
                    if is_endgame { reduction = reduction.saturating_sub(1); }
                    // Don't reduce more than depth-2
                    reduction = reduction.min(depth.saturating_sub(2));
                }

                // ═══════════════════════════════════════════════════════════════
                // PVS: Null-window søk for ikke-PV trekk
                // ═══════════════════════════════════════════════════════════════
                let search_depth = depth.saturating_sub(1 + reduction);

                let (null_score, _) = if maximizing {
                    self.minimax(&new_bb, search_depth, alpha, alpha + 1, false)
                } else {
                    self.minimax(&new_bb, search_depth, beta - 1, beta, true)
                };

                // Sjekk om vi trenger re-search
                let needs_research = if maximizing {
                    null_score > alpha && (null_score < beta || reduction > 0)
                } else {
                    null_score < beta && (null_score > alpha || reduction > 0)
                };

                if needs_research {
                    // Re-search med fullt vindu og full dybde
                    let (full_score, _) = self.minimax(&new_bb, depth - 1, alpha, beta, !maximizing);
                    score = full_score;
                } else {
                    score = null_score;
                }
            }

            // Pop accumulator
            if self.halfpail_nnue.is_some() {
                self.dual_acc_stack.pop();
            } else if self.quantized_nnue.is_some() {
                self.qacc_stack.pop();
            } else if self.nnue.is_some() {
                self.acc_stack.pop();
            }

            moves_searched += 1;

            if maximizing {
                if score > best_score {
                    best_score = score;
                    best_move = Some(mv);
                }
                alpha = alpha.max(score);
                if beta <= alpha {
                    // Beta cutoff - update killer moves, history, and cont_history
                    self.store_killer(&mv, depth as usize);
                    self.update_history(&mv, depth);
                    if let Some(pm) = prev_mv {
                        let prev_to = pm.barrel_to() as usize;
                        let curr_to = mv.barrel_to() as usize;
                        let bonus = (depth as i32) * (depth as i32);
                        self.cont_history[prev_to][curr_to] += bonus;
                        self.cont_history[prev_to][curr_to] =
                            self.cont_history[prev_to][curr_to].clamp(-32000, 32000);
                    }
                    self.cutoffs += 1;
                    break;
                }
            } else {
                if score < best_score {
                    best_score = score;
                    best_move = Some(mv);
                }
                beta = beta.min(score);
                if beta <= alpha {
                    // Beta cutoff - update killer moves, history, and cont_history
                    self.store_killer(&mv, depth as usize);
                    self.update_history(&mv, depth);
                    if let Some(pm) = prev_mv {
                        let prev_to = pm.barrel_to() as usize;
                        let curr_to = mv.barrel_to() as usize;
                        let bonus = (depth as i32) * (depth as i32);
                        self.cont_history[prev_to][curr_to] += bonus;
                        self.cont_history[prev_to][curr_to] =
                            self.cont_history[prev_to][curr_to].clamp(-32000, 32000);
                    }
                    self.cutoffs += 1;
                    break;
                }
            }
        }

        // Restore prev_move for continuation history context
        self.prev_move = prev_mv;

        // Store in TT
        let flag = if best_score <= original_alpha {
            TTFlag::UpperBound
        } else if best_score >= beta {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };

        let tt_best_move = best_move.map(|m| m.to_move());
        self.tt.store(hash, depth, best_score, flag, tt_best_move);

        (best_score, best_move)
    }

    /// Konverter Move til BitMove (statisk versjon)
    fn move_to_bitmove_static(mv: &Move) -> BitMove {
        let barrel_to = sq(mv.barrel_to.row as usize, mv.barrel_to.col as usize) as u8;
        let pail_pos = mv.place_pail.map(|p| sq(p.row as usize, p.col as usize) as u8);

        if mv.is_barrel_placement {
            BitMove::new_placement(barrel_to, pail_pos)
        } else {
            let from = mv.barrel_from.unwrap();
            let barrel_from = sq(from.row as usize, from.col as usize) as u8;
            let path: Vec<u8> = mv.barrel_path
                .iter()
                .map(|p| sq(p.row as usize, p.col as usize) as u8)
                .collect();
            BitMove::new_move(barrel_from, barrel_to, &path, pail_pos)
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // TIME-BASED SEARCH
    // ═══════════════════════════════════════════════════════════════
    // Check deadline every 1024 nodes to avoid expensive Instant::now() calls.

    /// Check if the search should stop due to time limit.
    /// Uses a sticky `search_stopped` flag so that once the deadline is hit,
    /// all subsequent calls return true immediately without rechecking the clock.
    /// Only calls Instant::now() every 1024 nodes to minimize overhead.
    #[inline]
    fn should_stop(&mut self) -> bool {
        if self.search_stopped {
            return true;
        }
        if let Some(deadline) = self.deadline {
            self.nodes_since_check += 1;
            if self.nodes_since_check >= 1024 {
                self.nodes_since_check = 0;
                if Instant::now() >= deadline {
                    self.search_stopped = true;
                    return true;
                }
            }
        }
        false
    }

    /// Time-based search: iterative deepening up to max depth, stopping when time runs out.
    /// Returns the best result from the last fully completed depth iteration.
    pub fn search_timed(&mut self, bb: &BitBoard, time_ms: u64) -> (i32, Option<BitMove>, u8) {
        self.deadline = Some(Instant::now() + Duration::from_millis(time_ms));
        self.nodes_since_check = 0;
        self.search_stopped = false;
        let (score, mv) = self.search_iterative(bb, 30);
        let depth_reached = self.last_completed_depth;
        self.deadline = None;
        self.search_stopped = false;
        (score, mv, depth_reached)
    }

    /// Iterative deepening search med aspiration windows
    pub fn search_iterative(&mut self, bb: &BitBoard, max_depth: u8) -> (i32, Option<BitMove>) {
        let mut best_score = 0;
        let mut best_move = None;
        let mut total_nodes = 0u64;
        let mut total_quiesce = 0u64;
        let mut total_cutoffs = 0u64;
        let mut total_tt_hits = 0u64;
        let mut prev_score: Option<i32> = None;
        self.last_completed_depth = 0;

        for depth in 1..=max_depth {
            // Check deadline before starting a new depth iteration
            if self.deadline.is_some() && self.should_stop() {
                break; // Use best result from previous completed depth
            }

            // Bruk forrige score for aspiration windows
            let (score, mv) = self.search_with_aspiration(bb, depth, prev_score);

            total_nodes += self.nodes_searched;
            total_quiesce += self.quiesce_nodes;
            total_cutoffs += self.cutoffs;
            total_tt_hits += self.tt_hits;

            // If search was aborted mid-iteration, discard partial results
            if self.deadline.is_some() && self.should_stop() {
                break;
            }

            // This depth completed fully — update best results
            prev_score = Some(score);
            best_score = score;
            if mv.is_some() {
                best_move = mv;
            }
            self.last_completed_depth = depth;

            // Stopp tidlig ved vinnersekvens
            if score.abs() > 90_000 {
                break;
            }
        }

        self.nodes_searched = total_nodes;
        self.quiesce_nodes = total_quiesce;
        self.cutoffs = total_cutoffs;
        self.tt_hits = total_tt_hits;
        (best_score, best_move)
    }
}
