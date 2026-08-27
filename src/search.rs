use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};


use crate::board::*;
use crate::nnue::*;

/// Størrelse på transposition table (antall entries)
pub const TT_SIZE: usize = 1 << 20; // ~1 million entries
const CORRECTION_TABLE_SIZE: usize = 16384; // 64KB correction history

/// Win scoring: a won position at distance d from the root scores
/// WIN_SCORE - d (White perspective). Scores beyond WIN_BOUND are wins.
pub const WIN_SCORE: i32 = 100_000;
pub const WIN_BOUND: i32 = 90_000;

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
    best_move: Option<BitMove>,  // Beste trekk (for move ordering); Copy, no heap
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

    fn store(&mut self, hash: u64, depth: u8, score: i32, flag: TTFlag, best_move: Option<BitMove>) {
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

    /// True wipe: remove all entries. Used between games so results are
    /// deterministic and independent of earlier games (O(1) clear keeps old
    /// entries visible to probe, which contaminates cross-game measurement).
    fn wipe(&mut self) {
        for cluster in &mut self.clusters {
            cluster.entries = [None, None, None];
        }
        self.generation = 0;
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

    /// True if the search was aborted by an external stop signal
    #[pyo3(get)]
    pub was_stopped: bool,
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

/// Handle for cancelling a running search from Python.
/// Holds a shared reference to the engine's atomic stop flag.
/// Call stop() from any thread to abort the current search within ~1024 nodes.
#[pyclass]
pub struct StopHandle {
    flag: Arc<AtomicBool>,
}

#[pymethods]
impl StopHandle {
    /// Signal the engine to stop searching. Takes effect within ~1024 nodes.
    fn stop(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Check if stop has been signalled.
    fn is_stopped(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
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

    /// Get a StopHandle for cancelling searches from another thread.
    /// The handle shares the engine's atomic stop flag.
    fn get_stop_handle(&self) -> StopHandle {
        StopHandle {
            flag: self.inner.stop_flag.clone(),
        }
    }

    /// Søk etter beste trekk
    /// Releases the GIL during search so Python threads (HTTP handlers) stay responsive.
    /// Use get_stop_handle() to cancel from another thread.
    fn search(&mut self, py: Python, board: &Board, depth: u8) -> SearchResult {
        let bb = BitBoard::from_board(board);

        // Clear stop flag before starting
        self.inner.stop_flag.store(false, Ordering::Relaxed);
        self.inner.search_stopped = false;
        self.inner.nodes_since_check = 0;

        // Release GIL during Rust computation
        let (score, best_bitmove) = py.allow_threads(|| {
            self.inner.search(&bb, depth)
        });

        let best_move = best_bitmove.map(|bm| bm.to_move());
        let was_stopped = self.inner.search_stopped;

        SearchResult {
            best_move,
            score,
            nodes_searched: self.inner.nodes_searched,
            cutoffs: self.inner.cutoffs,
            tt_hits: self.inner.tt_hits,
            quiesce_nodes: self.inner.quiesce_nodes,
            depth,
            was_stopped,
        }
    }

    /// Iterative deepening: Søk gradvis dypere (releases GIL per depth)
    fn search_iterative(&mut self, py: Python, board: &Board, max_depth: u8) -> SearchResult {
        let bb = BitBoard::from_board(board);

        // Clear stop flag before starting
        self.inner.stop_flag.store(false, Ordering::Relaxed);
        self.inner.search_stopped = false;
        self.inner.nodes_since_check = 0;

        let mut best_result = SearchResult {
            best_move: None,
            score: 0,
            nodes_searched: 0,
            cutoffs: 0,
            tt_hits: 0,
            quiesce_nodes: 0,
            depth: 0,
            was_stopped: false,
        };

        for depth in 1..=max_depth {
            let (score, best_bitmove) = py.allow_threads(|| {
                self.inner.search(&bb, depth)
            });

            // If stopped mid-depth, discard partial results
            if self.inner.search_stopped {
                best_result.was_stopped = true;
                break;
            }

            let best_move = best_bitmove.map(|bm| bm.to_move());

            best_result = SearchResult {
                best_move,
                score,
                nodes_searched: best_result.nodes_searched + self.inner.nodes_searched,
                cutoffs: best_result.cutoffs + self.inner.cutoffs,
                tt_hits: best_result.tt_hits + self.inner.tt_hits,
                quiesce_nodes: best_result.quiesce_nodes + self.inner.quiesce_nodes,
                depth,
                was_stopped: false,
            };

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

    #[getter]
    fn weight_threat2(&self) -> i32 {
        self.inner.weight_threat2
    }

    #[setter]
    fn set_weight_threat2(&mut self, value: i32) {
        self.inner.weight_threat2 = value;
    }

    #[getter]
    fn weight_adj_blocking(&self) -> i32 {
        self.inner.weight_adj_blocking
    }

    #[setter]
    fn set_weight_adj_blocking(&mut self, value: i32) {
        self.inner.weight_adj_blocking = value;
    }

    #[getter]
    fn weight_mobility(&self) -> i32 {
        self.inner.weight_mobility
    }

    #[setter]
    fn set_weight_mobility(&mut self, value: i32) {
        self.inner.weight_mobility = value;
    }

    #[getter]
    fn weight_passed(&self) -> i32 {
        self.inner.weight_passed
    }

    #[setter]
    fn set_weight_passed(&mut self, value: i32) {
        self.inner.weight_passed = value;
    }

    #[getter]
    fn weight_trapped(&self) -> i32 {
        self.inner.weight_trapped
    }

    #[setter]
    fn set_weight_trapped(&mut self, value: i32) {
        self.inner.weight_trapped = value;
    }

    #[getter]
    fn weight_score_accel(&self) -> i32 {
        self.inner.weight_score_accel
    }

    #[setter]
    fn set_weight_score_accel(&mut self, value: i32) {
        self.inner.weight_score_accel = value;
    }

    #[getter]
    fn weight_eg_threat(&self) -> i32 {
        self.inner.weight_eg_threat
    }

    #[setter]
    fn set_weight_eg_threat(&mut self, value: i32) {
        self.inner.weight_eg_threat = value;
    }

    #[getter]
    fn weight_jump(&self) -> i32 {
        self.inner.weight_jump
    }

    #[setter]
    fn set_weight_jump(&mut self, value: i32) {
        self.inner.weight_jump = value;
    }

    #[getter]
    fn weight_race(&self) -> i32 {
        self.inner.weight_race
    }

    #[setter]
    fn set_weight_race(&mut self, value: i32) {
        self.inner.weight_race = value;
    }

    #[getter]
    fn rfp_margin(&self) -> i32 {
        self.inner.rfp_margin
    }

    #[setter]
    fn set_rfp_margin(&mut self, value: i32) {
        self.inner.rfp_margin = value;
    }

    #[getter]
    fn lmp_base(&self) -> i32 {
        self.inner.lmp_base
    }

    #[setter]
    fn set_lmp_base(&mut self, value: i32) {
        self.inner.lmp_base = value;
    }

    #[getter]
    fn qs_mode(&self) -> i32 {
        self.inner.qs_mode
    }

    #[setter]
    fn set_qs_mode(&mut self, value: i32) {
        self.inner.qs_mode = value;
    }

    #[getter]
    fn asp_delta(&self) -> i32 { self.inner.asp_delta }
    #[setter]
    fn set_asp_delta(&mut self, value: i32) { self.inner.asp_delta = value; }

    #[getter]
    fn razor_base(&self) -> i32 { self.inner.razor_base }
    #[setter]
    fn set_razor_base(&mut self, value: i32) { self.inner.razor_base = value; }

    #[getter]
    fn razor_slope(&self) -> i32 { self.inner.razor_slope }
    #[setter]
    fn set_razor_slope(&mut self, value: i32) { self.inner.razor_slope = value; }

    #[getter]
    fn nmp_margin(&self) -> i32 { self.inner.nmp_margin }
    #[setter]
    fn set_nmp_margin(&mut self, value: i32) { self.inner.nmp_margin = value; }

    #[getter]
    fn nmp_boost_margin(&self) -> i32 { self.inner.nmp_boost_margin }
    #[setter]
    fn set_nmp_boost_margin(&mut self, value: i32) { self.inner.nmp_boost_margin = value; }

    #[getter]
    fn fut_scale(&self) -> i32 { self.inner.fut_scale }
    #[setter]
    fn set_fut_scale(&mut self, value: i32) { self.inner.fut_scale = value; }

    #[getter]
    fn lmr_div(&self) -> i32 { self.inner.lmr_div }
    #[setter]
    fn set_lmr_div(&mut self, value: i32) { self.inner.set_lmr_div(value); }

    #[getter]
    fn lmr_hist_good(&self) -> i32 { self.inner.lmr_hist_good }
    #[setter]
    fn set_lmr_hist_good(&mut self, value: i32) { self.inner.lmr_hist_good = value; }

    #[getter]
    fn lmr_hist_bad(&self) -> i32 { self.inner.lmr_hist_bad }
    #[setter]
    fn set_lmr_hist_bad(&mut self, value: i32) { self.inner.lmr_hist_bad = value; }

    #[getter]
    fn iir_depth(&self) -> i32 { self.inner.iir_depth }
    #[setter]
    fn set_iir_depth(&mut self, value: i32) { self.inner.iir_depth = value; }

    /// Load NNUE weights from JSON file
    /// After loading, the engine will use NNUE for evaluation instead of heuristics
    fn load_nnue(&mut self, path: &str) -> PyResult<()> {
        self.inner.load_nnue(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load NNUE: {}", e))
        })
    }

    /// Check if NNUE is loaded
    fn has_nnue(&self) -> bool {
        self.inner.nnue.is_some()
    }

    /// Clear NNUE (revert to heuristic evaluation)
    fn clear_nnue(&mut self) {
        self.inner.clear_nnue();
    }

    /// Static NNUE evaluation of a position (White perspective, centipawns),
    /// computed from scratch. For Python<->Rust parity checks and debugging.
    fn nnue_eval(&self, board: &Board) -> PyResult<i32> {
        let bb = BitBoard::from_board(board);
        match self.inner.nnue {
            Some(ref net) => Ok(net.evaluate_from_scratch(&bb)),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err("no NNUE loaded")),
        }
    }

    /// Architecture description of the loaded NNUE (or None).
    fn nnue_info(&self) -> Option<String> {
        self.inner.nnue.as_ref().map(|n| format!("{:?}", n.config))
    }

    /// Evaluate raw 164-float training rows (N*164 flat) with the loaded NNUE,
    /// from scratch. Lets the trainer verify Python<->Rust parity on the exact
    /// data it trained on (scored counts included, which `Board` can't express).
    fn nnue_eval_rows(&self, rows: Vec<f32>) -> PyResult<Vec<i32>> {
        let net = self.inner.nnue.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no NNUE loaded"))?;
        if rows.len() % 164 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("rows must be N*164 floats"));
        }
        Ok(rows.chunks_exact(164)
            .map(|r| net.evaluate_from_scratch(&bitboard_from_dense164(r)))
            .collect())
    }

    /// Time-based search: iterative deepening that stops when time runs out.
    /// Returns SearchResult with the best move from the last fully completed depth.
    fn search_timed(&mut self, py: Python, board: &Board, time_ms: u64) -> SearchResult {
        let bb = BitBoard::from_board(board);

        // Clear external stop flag (deadline is managed internally)
        self.inner.stop_flag.store(false, Ordering::Relaxed);

        let (score, best_bitmove, depth_reached) = py.allow_threads(|| {
            self.inner.search_timed(&bb, time_ms)
        });

        let best_move = best_bitmove.map(|bm| bm.to_move());

        SearchResult {
            best_move,
            score,
            nodes_searched: self.inner.nodes_searched,
            cutoffs: self.inner.cutoffs,
            tt_hits: self.inner.tt_hits,
            quiesce_nodes: self.inner.quiesce_nodes,
            depth: depth_reached,
            was_stopped: self.inner.search_stopped,
        }
    }

    /// Expose heuristic evaluation to Python (for MCTS leaf evaluation).
    /// Always uses the hand-crafted heuristic, not NNUE.
    /// Returns score from White's perspective.
    fn evaluate_position(&self, board: &Board) -> i32 {
        let bb = BitBoard::from_board(board);
        self.inner.evaluate_heuristic(&bb)
    }

    /// Load all solved tablebase phases from a directory (tb_{w}v{b}.bin files).
    fn load_tablebases(&mut self, dir: &str) -> PyResult<Vec<(usize, usize)>> {
        let tb = crate::tablebase::Tablebase::load_dir(std::path::Path::new(dir))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("load tablebases: {}", e)))?;
        let phases = tb.loaded_phases();
        self.inner.tablebase = Some(tb);
        Ok(phases)
    }

    /// Exact tablebase value of a position if its phase is solved:
    /// ("white"|"black"|"draw", distance_to_win) or None.
    fn tablebase_probe(&self, board: &Board) -> Option<(String, u32)> {
        let bb = BitBoard::from_board(board);
        let tb = self.inner.tablebase.as_ref()?;
        let v = tb.value(&bb)?;
        use crate::tablebase::{is_black_win, is_white_win, win_dist};
        Some(if is_white_win(v) {
            ("white".to_string(), win_dist(v) as u32)
        } else if is_black_win(v) {
            ("black".to_string(), win_dist(v) as u32)
        } else {
            ("draw".to_string(), 0)
        })
    }

    #[getter]
    fn tb_hits(&self) -> u64 {
        self.inner.tb_hits
    }

    /// Single-agent race distances (white, black): exact min-moves for each
    /// side alone to score all remaining barrels. For play analysis.
    fn race_distances(&self, board: &Board) -> (u16, u16) {
        let bb = BitBoard::from_board(board);
        (
            crate::race::RACE_TABLE.side_distance(&bb, Player::White),
            crate::race::RACE_TABLE.side_distance(&bb, Player::Black),
        )
    }

    /// Handcrafted heuristic evaluation of raw 164-float rows (N*164 flat),
    /// for side-by-side comparison with nnue_eval_rows.
    fn heuristic_eval_rows(&self, rows: Vec<f32>) -> PyResult<Vec<i32>> {
        if rows.len() % 164 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("rows must be N*164 floats"));
        }
        Ok(rows.chunks_exact(164)
            .map(|r| self.inner.evaluate_heuristic(&bitboard_from_dense164(r)))
            .collect())
    }

    /// Set the game's position history (Zobrist hashes of positions BEFORE
    /// the current one). The search scores a repetition of any of these as a
    /// draw. Only positions since the last irreversible event (placement,
    /// pail placement, scoring) can repeat, so passing just that suffix is
    /// both correct and fastest. Call before each search; cleared by full_reset.
    fn set_game_history(&mut self, hashes: Vec<u64>) {
        self.inner.game_history = hashes;
    }

    /// Contempt in centipawns: positive = the engine avoids draws
    /// (a repetition/no-progress draw is scored -contempt for the root player).
    #[getter]
    fn contempt(&self) -> i32 {
        self.inner.contempt
    }

    #[setter]
    fn set_contempt(&mut self, value: i32) {
        self.inner.contempt = value;
    }

    /// Plies without an irreversible event before the position is scored as
    /// a draw in search (0 = disabled). Default 60.
    #[getter]
    fn no_progress_limit(&self) -> u16 {
        self.inner.no_progress_limit
    }

    #[setter]
    fn set_no_progress_limit(&mut self, value: u16) {
        self.inner.no_progress_limit = value;
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

    // NNUE evaluator (generic over feature set; None = handcrafted eval)
    nnue: Option<SparseNNUE>,

    // Endgame tablebases (solved phases); probed at non-root nodes
    tablebase: Option<crate::tablebase::Tablebase>,
    pub tb_hits: u64,

    // Dual-perspective accumulator stack for incremental NNUE updates
    dual_acc_stack: DualAccumulatorStack,

    // Time-based search: deadline for when to abort, checked every 1024 nodes
    deadline: Option<Instant>,
    nodes_since_check: u32,
    search_stopped: bool,
    last_completed_depth: u8,

    // External stop flag: set from Python (via StopHandle) to cancel search.
    // Arc allows sharing between Engine wrapper and StopHandle objects.
    // Checked every 1024 nodes alongside the deadline.
    stop_flag: Arc<AtomicBool>,

    // LMR reduction table: lmr_table[depth][move_count] = reduction
    // Precomputed using ln(depth) * ln(move_count) / 2.5
    lmr_table: [[u8; 64]; 32],

    // Correction history: tracks (search_score - static_eval) error per hash bucket.
    // Applied to static_eval before pruning decisions (razoring, NMP, futility).
    correction_history: [i32; CORRECTION_TABLE_SIZE],

    // ═══ Draw rules ═══
    // Hashes of positions earlier in the actual game (set from Python before
    // each search). Only positions since the last irreversible event matter —
    // the Zobrist off-board keys guarantee no collision across such events.
    pub game_history: Vec<u64>,
    // Hashes of positions on the current search path (root..parent).
    path_hashes: Vec<u64>,
    // A single repetition of a game/path position is scored as a draw.
    // Draw score is 0 adjusted by contempt: positive contempt makes the
    // engine (the root player) avoid draws.
    pub contempt: i32,
    // Plies without an irreversible event before the game is a draw (0 = off).
    pub no_progress_limit: u16,
    // Side to move at the search root (for contempt sign).
    root_player: Player,

    // Fallback heuristisk vekter
    pub weight_progress: i32,
    pub weight_center_pail: i32,
    pub weight_blocking: i32,
    pub weight_scored: i32,
    pub weight_threat: i32,

    // New eval terms (default 0 = no-op until tuned)
    pub weight_threat2: i32,       // Barrels at dist==2 from goal
    pub weight_adj_blocking: i32,  // Pail in adjacent column ahead of enemy barrel
    pub weight_mobility: i32,      // Forward empty squares per barrel

    // Research-based eval terms (default 0 = no-op until tuned)
    pub weight_passed: i32,        // Bonus per barrel with clear column path to goal
    pub weight_trapped: i32,       // Penalty per barrel with zero empty adjacent squares
    pub weight_score_accel: i32,   // Non-linear scoring: extra reward for 2+ scored barrels
    pub weight_eg_threat: i32,     // Endgame threat amplification (scaled by game phase)
    pub weight_jump: i32,          // Bonus per barrel with forward jump available
    pub weight_race: i32,          // Single-agent race distance difference (Roschke & Sturtevant)

    // ═══ Search knobs (A/B-able via match.py --set) ═══
    // Reverse futility pruning: static eval beats beta by margin*depth → cutoff
    // (0 = off; inconclusive at 120, retries pending).
    pub rfp_margin: i32,
    // Late move pruning: skip quiets after lmp_base + depth² ordered moves.
    pub lmp_base: i32,
    // Quiescence: 1 = scoring moves + moves to the row before goal, cap 6;
    // 2 = scoring moves only, cap 4 (default; beat mode 1 by +29 Elo).
    pub qs_mode: i32,
    // The remaining pruning constants, exposed for SPSA with the NNUE loaded
    // (all were tuned by hand against the heuristic eval).
    pub asp_delta: i32,        // initial aspiration half-window
    pub razor_base: i32,       // razoring margin = base + slope * depth
    pub razor_slope: i32,
    pub nmp_margin: i32,       // try NMP only if eval >= beta - margin
    pub nmp_boost_margin: i32, // extra reduction when eval >= beta + this
    pub fut_scale: i32,        // percent scale applied to the futility margin table
    pub lmr_div: i32,          // LMR table divisor x100: R = ln(d) ln(m) / (lmr_div/100)
    pub lmr_hist_good: i32,    // history above this: one less reduction
    pub lmr_hist_bad: i32,     // history below this: one more reduction
    pub iir_depth: i32,        // internal iterative reduction from this depth (no TT move)
}

fn build_lmr_table(div_x100: i32) -> [[u8; 64]; 32] {
    // R = ln(depth) * ln(move_count) / div. Divisor 1.0 was tuned for the 6x6
    // board (shallower depths than standard chess).
    let div = (div_x100.max(1) as f64) / 100.0;
    let mut t = [[0u8; 64]; 32];
    for d in 1..32 {
        for m in 1..64 {
            t[d][m] = ((d as f64).ln() * (m as f64).ln() / div) as u8;
        }
    }
    t
}

impl Default for BitBoardEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BitBoardEngine {
    pub fn new() -> Self {
        let lmr_table = build_lmr_table(101);

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
            tablebase: None,
            tb_hits: 0,
            dual_acc_stack: DualAccumulatorStack::new(),
            deadline: None,
            nodes_since_check: 0,
            search_stopped: false,
            last_completed_depth: 0,
            stop_flag: Arc::new(AtomicBool::new(false)),
            lmr_table,
            correction_history: [0i32; CORRECTION_TABLE_SIZE],
            game_history: Vec::new(),
            path_hashes: Vec::with_capacity(64),
            contempt: 0,
            no_progress_limit: 60,
            root_player: Player::White,
            // SPSA-tuned 2026-08-20 (scripts/results/spsa_tune.json):
            // +24 Elo @ d5 [+4,+45], +25 @ d7 [+2,+49] vs previous defaults.
            weight_progress: 77,
            weight_center_pail: 15,
            weight_blocking: 19,
            weight_scored: 700,
            weight_threat: 144,
            weight_threat2: 101,
            weight_adj_blocking: 0,
            weight_mobility: 13,
            weight_passed: 80,
            weight_trapped: 1,
            weight_score_accel: -3,
            weight_eg_threat: -1,
            weight_jump: 63,
            // Validated 2026-08-24: +36 Elo @ d5 [+6,+67], +34 @ 50ms [+2,+66],
            // +63 @ d7 [+24,+104] vs 0. Sweep: 40 +13(ns), 120 +5(ns) → 80.
            weight_race: 80,
            // Search knobs: SPSA-tuned 2026-08-27 with net-3 loaded, 100 ms
            // (scripts/results/spsa_search_net3.json). Tuned vs previous
            // defaults, same net both sides: +60 [+40,+79] @ 100ms (600 games),
            // +54 [+32,+77] @ 200ms (400 games). Previous values in comments.
            rfp_margin: 63,  // was 0 (SPRT vs heuristic eval inconclusive at 120)
            // LMP: SPRT PASS @ 50ms (+27 [+10,+45]); @ 200ms accepted on CI
            // (+14 [+3,+26] over 1600 games; two independent LTC runs +16/+14).
            lmp_base: 7,     // was 6
            // qs_mode 2 PASSED both gates: +53 [+28,+78] @ 50ms, +62 [+35,+91]
            // @ 200ms vs legacy; beats qs_mode 1 head-to-head +29 [+11,+47].
            qs_mode: 2,
            asp_delta: 29,          // was 30
            razor_base: 198,        // was 200
            razor_slope: 137,       // was 150
            nmp_margin: 49,         // was 50
            nmp_boost_margin: 161,  // was 150
            fut_scale: 104,         // was 100
            lmr_div: 101,           // was 100
            lmr_hist_good: 877,     // was 1000
            lmr_hist_bad: -559,     // was -500
            iir_depth: 3,           // was 4
        }
    }

    pub fn set_lmr_div(&mut self, div_x100: i32) {
        self.lmr_div = div_x100;
        self.lmr_table = build_lmr_table(div_x100);
    }

    /// Clear history table (call between games)
    pub fn clear_history(&mut self) {
        self.history = [[0; NUM_SQUARES]; NUM_SQUARES];
        self.cont_history = [[0i32; NUM_SQUARES]; NUM_SQUARES];
        self.correction_history = [0i32; CORRECTION_TABLE_SIZE];
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
        for bucket in 0..CORRECTION_TABLE_SIZE {
            self.correction_history[bucket] /= 2;
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

    /// Load an NNUE (v2 sparse format or legacy HalfPail JSON)
    pub fn load_nnue(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.nnue = Some(SparseNNUE::load(path)?);
        self.eval_cache.clear();
        Ok(())
    }

    /// Clear NNUE (revert to heuristic evaluation)
    pub fn clear_nnue(&mut self) {
        self.nnue = None;
        self.eval_cache.clear();
    }

    /// Tøm TT
    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }

    /// Full reset - tøm alle caches og tabeller (mellom spill)
    pub fn full_reset(&mut self) {
        self.tt.wipe();
        self.eval_cache.clear();
        self.clear_history();
        self.killer_moves = std::array::from_fn(|_| [None, None]);
        self.game_history.clear();
        self.path_hashes.clear();
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

        // Non-linear scoring acceleration: extra reward for having 2+ scored barrels.
        // A player with 3 scored (1 away from winning) deserves exponentially more credit.
        // Table: [0, 0, 1, 3] means: 0 extra for 0-1 scored, 1x accel for 2, 3x for 3.
        if self.weight_score_accel != 0 {
            const ACCEL: [i32; 4] = [0, 0, 1, 3];
            let ws = (bb.white_scored as usize).min(3);
            let bs = (bb.black_scored as usize).min(3);
            score += (ACCEL[ws] - ACCEL[bs]) * self.weight_score_accel;
        }

        // Fremgang + trussel-bonus for tønner nær mål
        let mut white_progress = 0i32;
        let mut white_threats = 0i32; // Tønner på rad 1 (kan score neste trekk)
        let mut white_threats2 = 0i32; // Tønner på rad 2 (dist==2 from goal)
        let mut white_mobility = 0i32; // Forward empty squares
        let mut white_passed = 0i32;  // Barrels with clear column path to goal
        let mut white_trapped = 0i32; // Barrels with zero empty adjacent squares
        let mut white_jumps = 0i32;  // Forward jumps available
        let occupied = bb.occupied;
        // Obstacles for passed barrel detection: enemy barrels + enemy pail
        let white_obstacles = bb.black_barrels | bb.black_pail;
        let black_obstacles = bb.white_barrels | bb.white_pail;
        let empty = !occupied & ((1u64 << NUM_SQUARES) - 1);
        let mut bb_white = bb.white_barrels;
        while bb_white != 0 {
            let sq = bb_white.trailing_zeros() as usize;
            let (row, col) = sq_to_coords(sq);
            let dist_to_goal = row; // White's goal is row 0
            white_progress += (BOARD_SIZE - 1 - row) as i32;
            if dist_to_goal == 1 {
                white_threats += 1; // Barrel can score next move
            } else if dist_to_goal == 2 {
                white_threats2 += 1;
            }
            // Passed barrel: no enemy barrels/pail in this column ahead (rows 0..row-1)
            if self.weight_passed != 0 && row > 0 {
                // Mask: all squares in this column with row < current row
                let col_ahead = COL_MASK[col] & ((1u64 << (row * BOARD_SIZE)) - 1);
                if col_ahead & white_obstacles == 0 {
                    white_passed += 1;
                }
            }
            // Trapped barrel: zero empty adjacent squares
            if self.weight_trapped != 0 && ADJACENT[sq] & empty == 0 {
                white_trapped += 1;
            }
            // Mobility: count empty adjacent squares toward goal (row - 1 for white)
            if self.weight_mobility != 0 {
                let adj = ADJACENT[sq] & empty;
                let mut adj_bits = adj;
                while adj_bits != 0 {
                    let adj_sq = adj_bits.trailing_zeros() as usize;
                    let (adj_row, _) = sq_to_coords(adj_sq);
                    if adj_row < row { // toward goal for white
                        white_mobility += 1;
                    }
                    adj_bits &= adj_bits - 1;
                }
            }
            // Jump: count forward jumps over any piece (except enemy pail) to empty landing
            if self.weight_jump != 0 {
                let jumpable = bb.white_barrels | bb.black_barrels | bb.white_pail;
                for dir in 0..NUM_JUMP_DIRS {
                    let over = JUMP_OVER[sq][dir];
                    let landing = JUMP_LANDING[sq][dir];
                    if over >= 0 && landing >= 0 {
                        let over_bit = 1u64 << over;
                        let landing_bit = 1u64 << landing;
                        if (jumpable & over_bit) != 0
                            && (bb.black_pail & over_bit) == 0
                            && (occupied & landing_bit) == 0
                        {
                            let (land_row, _) = sq_to_coords(landing as usize);
                            if land_row < row {
                                white_jumps += 1;
                            }
                        }
                    }
                }
            }
            bb_white &= bb_white - 1;
        }

        let mut black_progress = 0i32;
        let mut black_threats = 0i32; // Tønner på rad 4 (kan score neste trekk)
        let mut black_threats2 = 0i32;
        let mut black_mobility = 0i32;
        let mut black_passed = 0i32;
        let mut black_trapped = 0i32;
        let mut black_jumps = 0i32;
        let mut bb_black = bb.black_barrels;
        while bb_black != 0 {
            let sq = bb_black.trailing_zeros() as usize;
            let (row, col) = sq_to_coords(sq);
            let dist_to_goal = (BOARD_SIZE - 1) - row; // Black's goal is row 5
            black_progress += row as i32;
            if dist_to_goal == 1 {
                black_threats += 1; // Barrel can score next move
            } else if dist_to_goal == 2 {
                black_threats2 += 1;
            }
            // Passed barrel: no enemy barrels/pail in this column ahead (rows row+1..5)
            if self.weight_passed != 0 && row < BOARD_SIZE - 1 {
                let col_ahead = COL_MASK[col] & !((1u64 << ((row + 1) * BOARD_SIZE)) - 1);
                if col_ahead & black_obstacles == 0 {
                    black_passed += 1;
                }
            }
            // Trapped barrel: zero empty adjacent squares
            if self.weight_trapped != 0 && ADJACENT[sq] & empty == 0 {
                black_trapped += 1;
            }
            // Mobility: count empty adjacent squares toward goal (row + 1 for black)
            if self.weight_mobility != 0 {
                let adj = ADJACENT[sq] & empty;
                let mut adj_bits = adj;
                while adj_bits != 0 {
                    let adj_sq = adj_bits.trailing_zeros() as usize;
                    let (adj_row, _) = sq_to_coords(adj_sq);
                    if adj_row > row { // toward goal for black
                        black_mobility += 1;
                    }
                    adj_bits &= adj_bits - 1;
                }
            }
            // Jump: count forward jumps over any piece (except enemy pail) to empty landing
            if self.weight_jump != 0 {
                let jumpable = bb.white_barrels | bb.black_barrels | bb.black_pail;
                for dir in 0..NUM_JUMP_DIRS {
                    let over = JUMP_OVER[sq][dir];
                    let landing = JUMP_LANDING[sq][dir];
                    if over >= 0 && landing >= 0 {
                        let over_bit = 1u64 << over;
                        let landing_bit = 1u64 << landing;
                        if (jumpable & over_bit) != 0
                            && (bb.white_pail & over_bit) == 0
                            && (occupied & landing_bit) == 0
                        {
                            let (land_row, _) = sq_to_coords(landing as usize);
                            if land_row > row {
                                black_jumps += 1;
                            }
                        }
                    }
                }
            }
            bb_black &= bb_black - 1;
        }

        score += (white_progress - black_progress) * self.weight_progress;
        score += (white_threats - black_threats) * self.weight_threat; // Immediate threats are valuable
        score += (white_threats2 - black_threats2) * self.weight_threat2;
        score += (white_mobility - black_mobility) * self.weight_mobility;
        score += (white_passed - black_passed) * self.weight_passed;
        score -= (white_trapped - black_trapped) * self.weight_trapped; // Penalty: more trapped = worse
        score += (white_jumps - black_jumps) * self.weight_jump;

        // Endgame threat amplification: threats become more valuable as game progresses.
        // Phase = total scored barrels (0..8). At phase 0, no extra bonus.
        // At phase 6 (e.g., 3-3), threats get weight_eg_threat * 6/4 extra per threat diff.
        if self.weight_eg_threat != 0 {
            let phase = (bb.white_scored + bb.black_scored) as i32;
            score += (white_threats - black_threats) * self.weight_eg_threat * phase / 4;
        }

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
                } else if (col as i32 - opp_col as i32).abs() == 1 && row > opp_row {
                    // Adjacent-column partial blocking
                    blocking_bonus += self.weight_adj_blocking;
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
                } else if (col as i32 - opp_col as i32).abs() == 1 && row < opp_row {
                    // Adjacent-column partial blocking
                    blocking_bonus += self.weight_adj_blocking;
                }
                bb_opp &= bb_opp - 1;
            }
            score -= blocking_bonus;
        }

        // Single-agent race distance: exact min-plies for each side alone to
        // score all remaining barrels (jump chains over own barrels included).
        // The difference is the strongest known race-eval term in this game
        // family (Chinese checkers: Roschke & Sturtevant 2013).
        if self.weight_race != 0 {
            let wd = crate::race::RACE_TABLE.side_distance(bb, Player::White) as i32;
            let bd = crate::race::RACE_TABLE.side_distance(bb, Player::Black) as i32;
            score += (bd - wd) * self.weight_race;
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

        let score = if let Some(ref net) = self.nnue {
            net.evaluate(bb, self.dual_acc_stack.current())
        } else {
            self.evaluate_heuristic(bb)
        };
        self.eval_cache.store(hash, score);
        score
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

    /// Sorter trekk — precompute scores once, then sort on keys
    fn order_moves(&self, moves: Vec<BitMove>, player: Player, depth: usize, tt_move: Option<&BitMove>) -> Vec<BitMove> {
        let mut scored: Vec<(i32, BitMove)> = moves
            .into_iter()
            .map(|mv| (self.score_move(&mv, player, depth, tt_move), mv))
            .collect();
        scored.sort_unstable_by_key(|(s, _)| -*s);
        scored.into_iter().map(|(_, mv)| mv).collect()
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

        // Nullstill killer moves per iteration. (Persisting them across
        // iterations was SPRT-null here: +1 Elo in 2000 games.)
        for km in &mut self.killer_moves {
            km[0] = None;
            km[1] = None;
        }

        // Age history (don't clear - accumulated knowledge is valuable)
        self.age_history();

        // Draw-rule state: fresh search path, contempt sign follows root player
        self.path_hashes.clear();
        self.root_player = bb.current_player;

        // Initialize accumulator
        self.dual_acc_stack.reset();
        if let Some(ref net) = self.nnue {
            net.init_accumulators(bb, self.dual_acc_stack.current_mut());
        }

        let maximizing = bb.current_player == Player::White;

        // ═══════════════════════════════════════════════════════════════
        // ASPIRATION WINDOWS
        // ═══════════════════════════════════════════════════════════════
        // Start ±30 around the previous score; on a fail, widen ONLY the
        // failed side geometrically (a fail-low says nothing about beta).
        // SPRT-passed vs fixed ±50 + full-window re-search: +42 @ 50ms,
        // +31 @ 200ms.
        let asp_delta = self.asp_delta.max(1);

        let (mut alpha, mut beta) = match prev_score {
            Some(score) => (score - asp_delta, score + asp_delta),
            None => (i32::MIN + 1, i32::MAX - 1),
        };

        let mut best_move;
        let mut score;
        let mut delta: i32 = asp_delta;

        loop {
            let (s, mv) = self.minimax(bb, depth, alpha, beta, maximizing);
            score = s;
            best_move = mv;

            if score <= alpha {
                // Fail low - utvid nedre grense
                delta *= 2;
                alpha = if delta > 1000 { i32::MIN + 1 } else { score - delta };
            } else if score >= beta {
                // Fail high - utvid øvre grense
                delta *= 2;
                beta = if delta > 1000 { i32::MAX - 1 } else { score + delta };
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
        let max_qsearch_depth: u8 = if self.qs_mode == 1 { 6 } else { 4 };

        self.quiesce_nodes += 1;

        // Sjekk for vinner (før eval: vinn-score er avstands-justert)
        if let Some(winner) = bb.check_winner() {
            return Self::win_score(winner, self.ply() + qsdepth as i32);
        }

        // Stand-pat: kan vi bare evaluere og returnere?
        let stand_pat = self.evaluate(bb);

        // Prevent stack overflow from unbounded quiescence search
        if qsdepth >= max_qsearch_depth {
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
        let qs_mode = self.qs_mode;

        let moves = bb.generate_moves();
        // Collect (sort_key, move): scoring moves first, then threats.
        let mut tactical: Vec<(u8, BitMove)> = moves
            .into_iter()
            .filter_map(|mv| {
                // Pail placements are strategic, never tactical-scoring moves
                // (their target square is the pail square, not a barrel).
                if mv.is_pail_placement() {
                    return None;
                }
                let to_sq = mv.barrel_to() as usize;
                let (to_row, _) = sq_to_coords(to_sq);
                let dist_to_goal = if player == Player::White {
                    to_row // White's goal is row 0
                } else {
                    BOARD_SIZE - 1 - to_row // Black's goal is row 5
                };

                // Scoring move: always tactical, searched first
                if dist_to_goal == 0 {
                    return Some((0u8, mv));
                }
                // Mode 1 also takes moves to the row before goal (immediate
                // scoring threat); mode 2 (default) is scoring moves only.
                if qs_mode == 1 && dist_to_goal == 1 {
                    return Some((1u8, mv));
                }
                None
            })
            .collect();

        // Ingen taktiske trekk - returner stand-pat
        if tactical.is_empty() {
            return stand_pat;
        }
        tactical.sort_by_key(|(k, _)| *k);
        let tactical_moves: Vec<BitMove> = tactical.into_iter().map(|(_, mv)| mv).collect();

        if maximizing {
            let mut best = stand_pat;
            for mv in tactical_moves {
                let mut new_bb = *bb;
                new_bb.make_move(&mv);

                // Incremental NNUE accumulator update (generic feature diff)
                if let Some(ref net) = self.nnue {
                    self.dual_acc_stack.push();
                    net.update(bb, &new_bb, self.dual_acc_stack.current_mut());
                }

                let qs_child_maximizing = new_bb.current_player == Player::White;
                let score = self.quiesce(&new_bb, alpha, beta, qs_child_maximizing, qsdepth + 1);

                if self.nnue.is_some() {
                    self.dual_acc_stack.pop();
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

                // Incremental NNUE accumulator update (generic feature diff)
                if let Some(ref net) = self.nnue {
                    self.dual_acc_stack.push();
                    net.update(bb, &new_bb, self.dual_acc_stack.current_mut());
                }

                let qs_child_maximizing = new_bb.current_player == Player::White;
                let score = self.quiesce(&new_bb, alpha, beta, qs_child_maximizing, qsdepth + 1);

                if self.nnue.is_some() {
                    self.dual_acc_stack.pop();
                }

                best = best.min(score);
                if score <= alpha {
                    break; // Alpha cutoff
                }
            }
            best
        }
    }

    /// Draw score from White's perspective, adjusted by contempt.
    /// Positive contempt = the root player dislikes draws.
    #[inline]
    fn draw_score(&self) -> i32 {
        match self.root_player {
            Player::White => -self.contempt,
            Player::Black => self.contempt,
        }
    }

    /// Current node's distance from the search root (wrapper pushes the
    /// node's hash before minimax_impl runs, so root = 0).
    #[inline]
    fn ply(&self) -> i32 {
        (self.path_hashes.len() as i32 - 1).max(0)
    }

    /// Win score from White's perspective, preferring FASTER wins:
    /// a win at distance d from the root scores WIN_SCORE - d.
    /// Without this the engine can't tell "win in 2" from "win in 12" and
    /// may shuffle a won position into a repetition draw.
    #[inline]
    fn win_score(winner: Player, dist: i32) -> i32 {
        match winner {
            Player::White => WIN_SCORE - dist,
            Player::Black => -(WIN_SCORE - dist),
        }
    }

    /// Win scores are root-relative, so they must be stored in the TT as
    /// node-relative and converted back on probe (standard mate-score
    /// handling), otherwise a "win in N" cached at one ply corrupts nodes
    /// probed at another.
    #[inline]
    fn value_to_tt(v: i32, ply: i32) -> i32 {
        if v > WIN_BOUND { v + ply } else if v < -WIN_BOUND { v - ply } else { v }
    }

    #[inline]
    fn value_from_tt(v: i32, ply: i32) -> i32 {
        if v > WIN_BOUND { v - ply } else if v < -WIN_BOUND { v + ply } else { v }
    }

    /// Minimax wrapper: draw-rule detection (repetition + no-progress clock)
    /// and search-path bookkeeping. Checked BEFORE the TT probe so repetition
    /// draws are never masked by (or stored into) the transposition table.
    fn minimax(
        &mut self,
        bb: &BitBoard,
        depth: u8,
        alpha: i32,
        beta: i32,
        maximizing: bool,
    ) -> (i32, Option<BitMove>) {
        // Only at non-root nodes (root must always return a move)
        if !self.path_hashes.is_empty() {
            if self.no_progress_limit > 0 && bb.halfmove_clock >= self.no_progress_limit {
                return (self.draw_score(), None);
            }
            let h = bb.hash;
            // One repetition (in search path or actual game) is scored as a
            // draw: if repeating is best, the position is at best drawn.
            if self.path_hashes.contains(&h) || self.game_history.contains(&h) {
                return (self.draw_score(), None);
            }
            // Tablebase probe: exact value with distance (root-relative)
            if let Some(ref tb) = self.tablebase {
                if let Some(v) = tb.value(bb) {
                    self.tb_hits += 1;
                    let ply = self.path_hashes.len() as i32;
                    use crate::tablebase::{is_black_win, is_white_win, win_dist};
                    let score = if is_white_win(v) {
                        WIN_SCORE - (ply + win_dist(v) as i32)
                    } else if is_black_win(v) {
                        -(WIN_SCORE - (ply + win_dist(v) as i32))
                    } else {
                        self.draw_score()
                    };
                    return (score, None);
                }
            }
        }

        self.path_hashes.push(bb.hash);
        let result = self.minimax_impl(bb, depth, alpha, beta, maximizing);
        self.path_hashes.pop();
        result
    }

    /// Minimax med alpha-beta, PVS, og LMR
    fn minimax_impl(
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

        let node_ply = self.ply();
        let tt_result = if let Some(entry) = self.tt.probe(hash) {
            self.tt_hits += 1;
            // Convert node-relative win scores back to root-relative
            Some((entry.depth, Self::value_from_tt(entry.score, node_ply), entry.flag, entry.best_move))
        } else {
            None
        };

        if let Some((tt_depth, tt_score, tt_flag, tt_mv_opt)) = tt_result {
            tt_move = tt_mv_opt;

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
        if tt_move.is_none() && depth as i32 >= self.iir_depth {
            depth -= 1;
        }

        // ═══════════════════════════════════════════════════════════════
        // ENDGAME DETECTION: fewer barrels = more tactical, less pruning
        // ═══════════════════════════════════════════════════════════════
        // When few barrels remain on the board, every move is critical.
        // Disable or reduce aggressive pruning to avoid missing winning moves.
        let total_remaining = (4u8.saturating_sub(bb.white_scored)) + (4u8.saturating_sub(bb.black_scored));
        let is_endgame = total_remaining <= 3;

        // Mid-turn detection: a pail sub-move was just made and the barrel
        // move is pending. Disable NMP/razoring/futility here — passing or
        // pruning a half-completed turn is not meaningful. (Pail placement
        // itself is optional on any turn, so "pail in hand" is otherwise a
        // normal position.)
        let is_pail_position = bb.awaiting_barrel;

        // Terminal node: prefer faster wins / slower losses
        if let Some(winner) = bb.check_winner() {
            return (Self::win_score(winner, self.ply()), None);
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

        // Apply correction history: adjust static eval based on historical error
        // at this hash bucket. Improves pruning decisions when static eval is
        // systematically biased for positions that hash to this bucket.
        let hash_bucket = (bb.hash as usize) & (CORRECTION_TABLE_SIZE - 1);
        let corrected_eval = static_eval + self.correction_history[hash_bucket] / 256;

        // ═══════════════════════════════════════════════════════════════
        // RAZORING
        // ═══════════════════════════════════════════════════════════════
        // When static eval is far below alpha (or above beta for minimizer),
        // drop to quiescence search. If even qsearch can't save the
        // position, prune the entire subtree.
        if depth <= 3 && !is_endgame && !is_pail_position {
            let razor_margin = self.razor_base + self.razor_slope * depth as i32;
            if maximizing && corrected_eval + razor_margin < alpha {
                let qscore = self.quiesce(bb, alpha, beta, maximizing, 0);
                if qscore < alpha {
                    return (qscore, None);
                }
            }
            if !maximizing && corrected_eval - razor_margin > beta {
                let qscore = self.quiesce(bb, alpha, beta, maximizing, 0);
                if qscore > beta {
                    return (qscore, None);
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // REVERSE FUTILITY PRUNING (static null move)
        // ═══════════════════════════════════════════════════════════════
        // If the static eval already beats the bound by a depth-scaled
        // margin, trust it and cut. Cheap sibling of NMP; standard in all
        // top engines. Gated on rfp_margin (0 = off) for A/B testing.
        if self.rfp_margin > 0
            && depth <= 8
            && !is_endgame
            && !is_pail_position
            && corrected_eval.abs() < WIN_BOUND
            && beta.abs() < WIN_BOUND
            && alpha.abs() < WIN_BOUND
        {
            let margin = self.rfp_margin * depth as i32;
            if maximizing && corrected_eval - margin >= beta {
                return (corrected_eval - margin, None);
            }
            if !maximizing && corrected_eval + margin <= alpha {
                return (corrected_eval + margin, None);
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // NULL MOVE PRUNING (tuned: R=2-3 + depth/eval boosts)
        // ═══════════════════════════════════════════════════════════════
        // If giving opponent a free move still results in a beta cutoff,
        // this position is so good we can prune.
        // Only use when position is already favorable (otherwise unlikely to cutoff)
        let nmp_margin = self.nmp_margin; // Only try NMP if we're at least this much better
        let nmp_allowed = depth >= 4
            && !is_endgame
            && !is_pail_position
            && corrected_eval.abs() < 90_000
            && !bb.has_barrel_near_goal()
            && beta.abs() < 90_000
            && if maximizing {
                corrected_eval >= beta - nmp_margin
            } else {
                corrected_eval <= alpha + nmp_margin
            };

        if nmp_allowed {
            // Base reduction: R=2 shallow, R=3 deeper (proven for 6x6)
            let mut r: u8 = if depth >= 6 { 3 } else { 2 };
            // Depth-scaling boost: at high depths we have more margin
            if depth >= 8 {
                r += 1;
            }
            // Eval-based boost: if eval strongly exceeds the bound, prune harder
            if maximizing && corrected_eval >= beta + self.nmp_boost_margin {
                r += 1;
            }
            if !maximizing && corrected_eval <= alpha - self.nmp_boost_margin {
                r += 1;
            }
            let null_depth = (depth as i16 - r as i16 - 1).max(1) as u8;

            // Make null move (swap sides without moving)
            let mut new_bb = *bb;
            new_bb.make_null_move();

            // Search with null window around beta
            let null_child_maximizing = new_bb.current_player == Player::White;
            let (null_score, _) = if maximizing {
                // White is maximizing - after null move, black searches to minimize
                self.minimax(&new_bb, null_depth, beta - 1, beta, null_child_maximizing)
            } else {
                // Black is minimizing - after null move, white searches to maximize
                self.minimax(&new_bb, null_depth, alpha, alpha + 1, null_child_maximizing)
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
        let base_margin = FUTILITY_MARGINS[depth.min(8) as usize] * self.fut_scale / 100;
        let margin = if is_endgame { base_margin / 2 } else { base_margin };
        let futility_pruning = depth <= 8
            && !is_pail_position
            && corrected_eval.abs() < 90_000 // Not near mate
            && if maximizing {
                corrected_eval + margin < alpha
            } else {
                corrected_eval - margin > beta
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

        // Late move pruning threshold: after this many ordered moves at low
        // depth, remaining quiets are skipped outright (0 = off).
        let lmp_threshold: usize = if self.lmp_base > 0 && depth <= 6 && !is_endgame && !is_pail_position {
            self.lmp_base as usize + (depth as usize) * (depth as usize)
        } else {
            usize::MAX
        };

        for mv in sorted_moves {
            // ═══════════════════════════════════════════════════════════════
            // LATE MOVE PRUNING - movecount-based skip of late quiets
            // ═══════════════════════════════════════════════════════════════
            if moves_searched >= lmp_threshold && !mv.is_pail_placement() {
                let (to_row, _) = sq_to_coords(mv.barrel_to() as usize);
                if to_row != bb.goal_row(bb.current_player) {
                    continue;
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // FUTILITY PRUNING - Skip futile moves
            // ═══════════════════════════════════════════════════════════════
            if futility_pruning && moves_searched > 0 && !mv.is_pail_placement() {
                // Don't prune moves that reach goal (high tactical value).
                // Pail placements are never futility-pruned either: a well-
                // placed pail can swing far more than the futility margin.
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

            // Incremental NNUE accumulator update (generic feature diff;
            // recomputes a perspective only when its bucket changed)
            if let Some(ref net) = self.nnue {
                self.dual_acc_stack.push();
                net.update(bb, &new_bb, self.dual_acc_stack.current_mut());
            }

            let score;

            // Set prev_move for continuation history in child nodes
            self.prev_move = Some(mv);

            // Derive child_maximizing from board state (handles pail sub-moves correctly)
            let child_maximizing = new_bb.current_player == Player::White;

            if moves_searched == 0 {
                // ═══════════════════════════════════════════════════════════════
                // PVS: Første trekk - fullt vindu (Principal Variation)
                // ═══════════════════════════════════════════════════════════════
                let (s, _) = self.minimax(&new_bb, depth - 1, alpha, beta, child_maximizing);
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
                        if self.history[from][to] > self.lmr_hist_good { reduction = reduction.saturating_sub(1); }
                        if self.history[from][to] < self.lmr_hist_bad { reduction += 1; }
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
                    self.minimax(&new_bb, search_depth, alpha, alpha + 1, child_maximizing)
                } else {
                    self.minimax(&new_bb, search_depth, beta - 1, beta, child_maximizing)
                };

                // Sjekk om vi trenger re-search
                let needs_research = if maximizing {
                    null_score > alpha && (null_score < beta || reduction > 0)
                } else {
                    null_score < beta && (null_score > alpha || reduction > 0)
                };

                if needs_research {
                    // Re-search med fullt vindu og full dybde
                    let (full_score, _) = self.minimax(&new_bb, depth - 1, alpha, beta, child_maximizing);
                    score = full_score;
                } else {
                    score = null_score;
                }
            }

            // Pop accumulator
            if self.nnue.is_some() {
                self.dual_acc_stack.pop();
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
                    // (only for barrel moves, not pail-only sub-moves)
                    if !mv.is_pail_placement() {
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
                    if !mv.is_pail_placement() {
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
                    }
                    self.cutoffs += 1;
                    break;
                }
            }
        }

        // Restore prev_move for continuation history context
        self.prev_move = prev_mv;

        // Update correction history: track (search_score - static_eval) error.
        // Only at depth >= 3 and non-mate scores. Clamped to ±8192.
        if depth >= 3 && best_score.abs() < 90_000 && static_eval.abs() < 90_000 {
            let error = (best_score - static_eval) * depth as i32;
            self.correction_history[hash_bucket] =
                (self.correction_history[hash_bucket] + error).clamp(-8192, 8192);
        }

        // Store in TT
        let flag = if best_score <= original_alpha {
            TTFlag::UpperBound
        } else if best_score >= beta {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };

        self.tt.store(hash, depth, Self::value_to_tt(best_score, node_ply), flag, best_move);

        (best_score, best_move)
    }

    // ═══════════════════════════════════════════════════════════════
    // TIME-BASED SEARCH
    // ═══════════════════════════════════════════════════════════════
    // Check deadline every 1024 nodes to avoid expensive Instant::now() calls.

    /// Check if the search should stop (time limit or external stop flag).
    /// Uses a sticky `search_stopped` flag so that once triggered,
    /// all subsequent calls return true immediately.
    /// Checks external stop flag and clock every 1024 nodes to minimize overhead.
    #[inline]
    fn should_stop(&mut self) -> bool {
        if self.search_stopped {
            return true;
        }
        self.nodes_since_check += 1;
        if self.nodes_since_check >= 1024 {
            self.nodes_since_check = 0;
            // Check external stop flag (set from Python via StopHandle)
            if self.stop_flag.load(Ordering::Relaxed) {
                self.search_stopped = true;
                return true;
            }
            // Check time deadline
            if let Some(deadline) = self.deadline {
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
            // Check stop flag / deadline before starting a new depth
            if self.should_stop() {
                break;
            }

            // Bruk forrige score for aspiration windows
            let (score, mv) = self.search_with_aspiration(bb, depth, prev_score);

            total_nodes += self.nodes_searched;
            total_quiesce += self.quiesce_nodes;
            total_cutoffs += self.cutoffs;
            total_tt_hits += self.tt_hits;

            // If search was aborted mid-iteration, discard partial results
            if self.should_stop() {
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
