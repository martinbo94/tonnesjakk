//! Rust MCTS for Tonnesjakk.
//!
//! Arena-based MCTS with PUCT selection, supporting three evaluation modes:
//! - Heuristic: entirely in Rust, no Python calls (very fast)
//! - Network: calls a Python function for leaf evaluation (policy + value)
//! - ONNX: runs ONNX Runtime inference entirely in Rust (no Python needed)
//!
//! Also provides full game loops (self-play and evaluation matches) that
//! run entirely in Rust, returning training data to Python.

use pyo3::prelude::*;

use crate::{BitBoard, BitBoardEngine, BitMove, Board, Move, Player, NUM_SQUARES, sq_to_coords};

// ---------------------------------------------------------------------------
// ONNX Runtime session wrapper (pure Rust inference, no Python needed)
// ---------------------------------------------------------------------------

/// Wrapper around an ONNX Runtime session for neural network inference in Rust.
/// Eliminates Python/GIL overhead from self-play by running inference entirely in Rust.
#[pyclass]
pub struct OnnxSession {
    session: ort::session::Session,
}

#[pymethods]
impl OnnxSession {
    /// Load an ONNX model from disk.
    /// use_coreml: if true, attempts CoreML execution provider (Apple Neural Engine).
    /// Falls back to CPU if CoreML is unavailable.
    #[new]
    #[pyo3(signature = (path, use_coreml=false))]
    fn new(path: &str, use_coreml: bool) -> PyResult<Self> {
        let session = if use_coreml {
            ort::session::Session::builder()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX session builder error: {e}")))?
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default()
                        .with_subgraphs(true)
                        .build(),
                ])
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX CoreML EP error: {e}")))?
                .commit_from_file(path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX load error: {e}")))?
        } else {
            ort::session::Session::builder()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX session builder error: {e}")))?
                .commit_from_file(path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX load error: {e}")))?
        };
        Ok(OnnxSession { session })
    }
}

impl OnnxSession {
    /// Run inference on a batch of board planes.
    /// Input: Vec of flat planes (each 5*36 = 180 floats).
    /// Returns: (batch_policies, batch_values) — policies as Vec<Vec<f32>>, values as Vec<f32>.
    fn infer_batch(&mut self, batch_planes: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let batch_size = batch_planes.len();
        if batch_size == 0 {
            return Ok((vec![], vec![]));
        }

        let plane_size = 5 * NUM_SQUARES; // 180
        // Build contiguous [batch, 5, 6, 6] tensor
        let mut flat = Vec::with_capacity(batch_size * plane_size);
        for planes in batch_planes {
            if planes.len() != plane_size {
                return Err(format!("Expected {} planes, got {}", plane_size, planes.len()));
            }
            flat.extend_from_slice(planes);
        }

        // Create ONNX tensor from shape + flat data
        let shape = vec![batch_size as i64, 5, 6, 6];
        let input_tensor = ort::value::Tensor::from_array((shape, flat))
            .map_err(|e| format!("ONNX tensor error: {e}"))?;

        let outputs = self.session.run(
            ort::inputs![input_tensor],
        ).map_err(|e| format!("ONNX run error: {e}"))?;

        // Output 0: policy logits [batch, 1332]
        // Output 1: value [batch]
        let (_policy_shape, policy_flat) = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| format!("ONNX policy extract error: {e}"))?;
        let (_value_shape, value_flat) = outputs[1].try_extract_tensor::<f32>()
            .map_err(|e| format!("ONNX value extract error: {e}"))?;

        let policy_size = POLICY_SIZE; // 1332
        let mut batch_policies = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * policy_size;
            let end = start + policy_size;
            batch_policies.push(policy_flat[start..end].to_vec());
        }

        let batch_values: Vec<f32> = value_flat.to_vec();

        Ok((batch_policies, batch_values))
    }
}

/// Policy output size: 37 from-indices (0-35 board + 36 off-board) x 36 to-indices
pub const POLICY_SIZE: usize = 37 * 36; // 1332

const SCORE_SCALING: f32 = 600.0;

/// Precomputed policy vertical-flip table for Black's relative encoding.
/// Maps each policy index through vertical square flip: row r → row 5-r.
static POLICY_VFLIP: std::sync::LazyLock<[u16; POLICY_SIZE]> = std::sync::LazyLock::new(|| {
    let mut table = [0u16; POLICY_SIZE];
    for from_sq in 0..37usize {
        for to_sq in 0..36usize {
            let idx = from_sq * 36 + to_sq;
            let new_from = if from_sq == 36 { 36 } else { vflip_sq(from_sq) };
            let new_to = vflip_sq(to_sq);
            table[idx] = (new_from * 36 + new_to) as u16;
        }
    }
    table
});

// ---------------------------------------------------------------------------
// Move encoding
// ---------------------------------------------------------------------------

/// Convert a BitMove to a policy index in [0, 1331].
#[inline]
pub fn policy_index(bm: &BitMove) -> u16 {
    let to_idx = bm.barrel_to() as u16;
    let from_idx = if bm.is_placement() {
        36u16
    } else {
        bm.barrel_from().unwrap_or(36) as u16
    };
    from_idx * 36 + to_idx
}

/// Vertically flip a square index on the 6x6 board: row r → row 5-r.
#[inline]
fn vflip_sq(sq: usize) -> usize {
    let row = sq / 6;
    let col = sq % 6;
    (5 - row) * 6 + col
}

/// Convert a BitBoard position to 5x6x6 float planes (relative encoding).
/// Planes: [my_barrels, opp_barrels, my_pail, opp_pail, bias]
/// For Black, pieces are swapped (my=black, opp=white) and all squares are
/// vertically flipped so that Black's goal direction matches White's.
pub fn bb_to_planes(bb: &BitBoard) -> Vec<f32> {
    let mut planes = vec![0.0f32; 5 * NUM_SQUARES];
    let is_white = bb.current_player == Player::White;

    let (my_barrels, opp_barrels, my_pail, opp_pail) = if is_white {
        (bb.white_barrels, bb.black_barrels, bb.white_pail, bb.black_pail)
    } else {
        (bb.black_barrels, bb.white_barrels, bb.black_pail, bb.white_pail)
    };

    for sq in 0..NUM_SQUARES {
        let mask = 1u64 << sq;
        let out_sq = if is_white { sq } else { vflip_sq(sq) };
        if my_barrels & mask != 0   { planes[out_sq] = 1.0; }
        if opp_barrels & mask != 0  { planes[NUM_SQUARES + out_sq] = 1.0; }
        if my_pail & mask != 0      { planes[2 * NUM_SQUARES + out_sq] = 1.0; }
        if opp_pail & mask != 0     { planes[3 * NUM_SQUARES + out_sq] = 1.0; }
    }
    // Plane 4: bias (always 1)
    for i in 4 * NUM_SQUARES..5 * NUM_SQUARES {
        planes[i] = 1.0;
    }
    planes
}

// ---------------------------------------------------------------------------
// MCTS Node (arena-allocated)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MCTSNode {
    parent: u32,          // Index in arena (u32::MAX = root / no parent)
    children_start: u32,  // First child index in arena
    children_count: u16,  // Number of children (0 = leaf)
    bitmove: BitMove,     // Move that led here (default for root)
    visit_count: u32,
    total_value: f32,     // Cumulative value (White's perspective)
    prior: f32,
    is_terminal: bool,
    terminal_value: f32,
    player_is_white: bool,
}

impl MCTSNode {
    fn root(is_white: bool) -> Self {
        MCTSNode {
            parent: u32::MAX,
            children_start: u32::MAX,
            children_count: 0,
            bitmove: BitMove::new_placement(0, None), // dummy
            visit_count: 0,
            total_value: 0.0,
            prior: 1.0,
            is_terminal: false,
            terminal_value: 0.0,
            player_is_white: is_white,
        }
    }

    #[inline]
    fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }

    fn is_expanded(&self) -> bool {
        self.children_count > 0 || self.is_terminal
    }
}

// ---------------------------------------------------------------------------
// MCTS Search Result (exposed to Python)
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct MCTSSearchResult {
    #[pyo3(get)]
    pub best_move: Option<Move>,
    #[pyo3(get)]
    pub visits: u32,
    #[pyo3(get)]
    pub policy_target: Vec<f32>,
    #[pyo3(get)]
    pub root_value: f32,
}

#[pymethods]
impl MCTSSearchResult {
    fn __repr__(&self) -> String {
        format!(
            "MCTSSearchResult(visits={}, root_value={:.3}, has_move={})",
            self.visits,
            self.root_value,
            self.best_move.is_some()
        )
    }
}

impl MCTSSearchResult {
    fn empty() -> Self {
        MCTSSearchResult {
            best_move: None,
            visits: 0,
            policy_target: vec![0.0; POLICY_SIZE],
            root_value: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Training data from self-play games (exposed to Python)
// ---------------------------------------------------------------------------

/// Training example from a single position: (planes, policy_target, value_target).
#[pyclass]
#[derive(Clone)]
pub struct TrainingExample {
    #[pyo3(get)]
    pub planes: Vec<f32>,        // 180 floats (5x6x6)
    #[pyo3(get)]
    pub policy_target: Vec<f32>, // 1332 floats
    #[pyo3(get)]
    pub value_target: f32,       // outcome in [-1, +1]
    #[pyo3(get)]
    pub search_score: f32,       // search eval at this position (current player perspective)
}

/// Result of a complete self-play game.
#[pyclass]
#[derive(Clone)]
pub struct SelfPlayResult {
    #[pyo3(get)]
    pub examples: Vec<TrainingExample>,
    #[pyo3(get)]
    pub winner: String,          // "white", "black", or "draw"
    #[pyo3(get)]
    pub move_count: u32,
}

/// Result of an evaluation match (multiple games).
#[pyclass]
#[derive(Clone)]
pub struct EvalMatchResult {
    #[pyo3(get)]
    pub wins: u32,
    #[pyo3(get)]
    pub draws: u32,
    #[pyo3(get)]
    pub losses: u32,
}

// ---------------------------------------------------------------------------
// Simple xorshift64 RNG (avoids adding rand crate dependency)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new() -> Self {
        // Seed from a mix of address space and a constant
        let seed = 0xdeadbeef_12345678u64
            ^ (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64);
        Rng(if seed == 0 { 1 } else { seed })
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Random usize in [0, n)
    fn usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Random float in [0, 1)
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Sample from a probability distribution. Returns index.
    fn sample_distribution(&mut self, probs: &[f64]) -> usize {
        let r = self.f64();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Standard normal via Box-Muller transform.
    fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-30); // avoid log(0)
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Gamma variate using Marsaglia & Tsang's method (alpha >= 1).
    /// For alpha < 1, uses Ahrens-Dieter boost: Gamma(a) = Gamma(a+1) * U^(1/a).
    fn gamma_variate(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            // Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
            let g = self.gamma_variate(alpha + 1.0);
            let u = self.f64().max(1e-30);
            return g * u.powf(1.0 / alpha);
        }

        // Marsaglia & Tsang for alpha >= 1
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();

        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.f64();
            // Accept/reject
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Sample from Dirichlet(alpha, ..., alpha) with n components.
    fn dirichlet(&mut self, alpha: f64, n: usize) -> Vec<f32> {
        let mut samples: Vec<f64> = (0..n)
            .map(|_| self.gamma_variate(alpha).max(1e-30))
            .collect();
        let sum: f64 = samples.iter().sum();
        for s in &mut samples {
            *s /= sum;
        }
        samples.into_iter().map(|s| s as f32).collect()
    }

    /// Gumbel(0,1) variate: -ln(-ln(U)) where U ~ Uniform(0,1).
    fn gumbel(&mut self) -> f64 {
        let u = self.f64().max(1e-30).min(1.0 - 1e-10);
        -((-u.ln()).ln())
    }
}

// ---------------------------------------------------------------------------
// MCTS Engine (exposed to Python)
// ---------------------------------------------------------------------------

#[pyclass]
pub struct MCTSEngine {
    simulations: u32,
    c_puct: f32,
    fpu_reduction: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    nodes: Vec<MCTSNode>,
    engine: BitBoardEngine, // For heuristic evaluation
    rng: Rng,
    heuristic_cache: std::collections::HashMap<u64, f32>,
    network_cache: std::collections::HashMap<u64, (Vec<f32>, f32)>,
    /// Lambda for blending value targets: value = lambda * game_outcome + (1-lambda) * search_score.
    /// 1.0 = pure game outcome (original AlphaZero), 0.0 = pure search score.
    /// Default 0.5 to give dense per-position signal from heuristic eval.
    value_blend_lambda: f32,
    /// Adjudicate games as decisive when |heuristic_eval| exceeds this threshold
    /// (in tanh-scaled units, i.e. 0.0-1.0) after adjudication_move has been reached.
    /// 0.0 = disabled. Default 0.6 (~360cp in raw heuristic score).
    adjudication_threshold: f32,
    /// Minimum move count before adjudication can trigger. Default 30.
    adjudication_min_moves: u32,
    /// Use Gumbel AlphaZero search at root (Sequential Halving + completed-Q policy targets).
    /// Only active during self-play. Eval games use standard PUCT.
    use_gumbel: bool,
    /// Forward-only mode: filter out backward barrel moves during self-play.
    /// White can only move to same or lower row (toward goal row 0),
    /// Black can only move to same or higher row (toward goal row 5).
    /// Reduces branching factor and forces decisive games.
    forward_only: bool,
    /// Penalty applied to MCTS leaf value when the position's hash appears in
    /// the game-level position history.  Discourages the search from choosing
    /// moves that revisit earlier positions (shuffle-loops).
    /// The penalty is subtracted from the absolute value, pushing it toward 0
    /// (draw-like) for each repetition.  0.0 = disabled.  Default 0.2.
    repetition_penalty: f32,
    /// Game-level position history (hashes).  Set by game loops before each
    /// search call so the tree can penalise revisiting earlier positions.
    game_position_history: Vec<u64>,
}

#[pymethods]
impl MCTSEngine {
    #[new]
    #[pyo3(signature = (simulations=200, c_puct=1.0, fpu_reduction=0.3, dirichlet_alpha=0.5, dirichlet_epsilon=0.25, value_blend_lambda=0.5, adjudication_threshold=0.6, adjudication_min_moves=30, use_gumbel=false, forward_only=false, repetition_penalty=0.0))]
    fn new(
        simulations: u32, c_puct: f32, fpu_reduction: f32,
        dirichlet_alpha: f32, dirichlet_epsilon: f32,
        value_blend_lambda: f32, adjudication_threshold: f32, adjudication_min_moves: u32,
        use_gumbel: bool, forward_only: bool, repetition_penalty: f32,
    ) -> Self {
        MCTSEngine {
            simulations,
            c_puct,
            fpu_reduction,
            dirichlet_alpha,
            dirichlet_epsilon,
            nodes: Vec::with_capacity(8192),
            engine: BitBoardEngine::new(),
            rng: Rng::new(),
            heuristic_cache: std::collections::HashMap::new(),
            network_cache: std::collections::HashMap::new(),
            value_blend_lambda,
            adjudication_threshold,
            adjudication_min_moves,
            use_gumbel,
            forward_only,
            repetition_penalty,
            game_position_history: Vec::new(),
        }
    }

    /// Pure-Rust MCTS with heuristic leaf evaluation. No Python calls.
    fn search_heuristic(&mut self, board: &Board) -> MCTSSearchResult {
        let bb = BitBoard::from_board(board);
        self.search_impl(&bb, None, false)
    }

    /// MCTS with neural network evaluation via Python callback (one-at-a-time).
    /// eval_fn(planes: list[float]) -> (policy: list[float], value: float)
    fn search_network(
        &mut self,
        py: Python<'_>,
        board: &Board,
        eval_fn: PyObject,
    ) -> PyResult<MCTSSearchResult> {
        let bb = BitBoard::from_board(board);
        Ok(self.search_impl(&bb, Some((py, &eval_fn)), false))
    }

    /// MCTS with batched neural network evaluation (much faster for large networks).
    /// eval_fn(batch_planes: list[list[float]]) -> (batch_policy: list[list[float]], batch_values: list[float])
    /// Uses virtual loss to select multiple leaves per batch, reducing Python round-trips.
    #[pyo3(signature = (board, eval_fn, batch_size=8))]
    fn search_network_batched(
        &mut self,
        py: Python<'_>,
        board: &Board,
        eval_fn: PyObject,
        batch_size: u32,
    ) -> PyResult<MCTSSearchResult> {
        let bb = BitBoard::from_board(board);
        Ok(self.search_batched_impl(&bb, py, &eval_fn, batch_size, false))
    }

    /// Get planes for a board position (for Python-side network inference).
    #[staticmethod]
    fn board_planes(board: &Board) -> Vec<f32> {
        let bb = BitBoard::from_board(board);
        bb_to_planes(&bb)
    }

    /// Get policy index for a move.
    #[staticmethod]
    fn move_policy_index(m: &Move) -> u16 {
        if m.is_pail_only {
            let pail = m.place_pail.as_ref().unwrap();
            return 36 * 36 + (pail.row * 6 + pail.col) as u16;
        }
        let to_idx = (m.barrel_to.row * 6 + m.barrel_to.col) as u16;
        let from_idx = if m.is_barrel_placement {
            36u16
        } else if let Some(pos) = &m.barrel_from {
            (pos.row * 6 + pos.col) as u16
        } else {
            36u16
        };
        from_idx * 36 + to_idx
    }

    // -----------------------------------------------------------------------
    // #1: Full heuristic self-play game (entirely in Rust, no Python calls)
    // -----------------------------------------------------------------------

    /// Play one complete self-play game using heuristic MCTS. Returns training data.
    /// No Python calls at all -- runs ~10x faster than the Python game loop.
    #[pyo3(signature = (random_opening=6, max_moves=80, temp_moves=10, full_search_fraction=1.0, cheap_sims=50))]
    fn play_heuristic_game(
        &mut self,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        full_search_fraction: f32,
        cheap_sims: u32,
    ) -> SelfPlayResult {
        self.play_heuristic_game_impl(random_opening, max_moves, temp_moves, full_search_fraction, cheap_sims)
    }

    /// Play N heuristic self-play games. Returns a list of SelfPlayResult.
    #[pyo3(signature = (n_games, random_opening=6, max_moves=80, temp_moves=10, full_search_fraction=1.0, cheap_sims=50))]
    fn play_heuristic_games(
        &mut self,
        n_games: usize,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        full_search_fraction: f32,
        cheap_sims: u32,
    ) -> Vec<SelfPlayResult> {
        (0..n_games)
            .map(|_| self.play_heuristic_game_impl(random_opening, max_moves, temp_moves, full_search_fraction, cheap_sims))
            .collect()
    }

    // -----------------------------------------------------------------------
    // #1b: Alpha-beta self-play games (for bootstrapping)
    // -----------------------------------------------------------------------

    /// Play N alpha-beta self-play games. Returns a list of SelfPlayResult.
    /// Uses alpha-beta search (not MCTS) for both sides, producing high-quality
    /// training data with multi-move policy targets via softmax on scores.
    #[pyo3(signature = (n_games, depth=5, random_opening=4, max_moves=80))]
    fn play_alphabeta_games(
        &mut self,
        n_games: usize,
        depth: u8,
        random_opening: usize,
        max_moves: usize,
    ) -> Vec<SelfPlayResult> {
        (0..n_games)
            .map(|_| self.play_alphabeta_game_impl(depth, random_opening, max_moves))
            .collect()
    }

    // -----------------------------------------------------------------------
    // #2: Temperature-based move selection (in Rust)
    // -----------------------------------------------------------------------

    /// Search with heuristic and return a move sampled by temperature.
    /// temperature=0 returns the most-visited move (deterministic).
    #[pyo3(signature = (board, temperature=1.0))]
    fn search_heuristic_with_temp(
        &mut self,
        board: &Board,
        temperature: f32,
    ) -> MCTSSearchResult {
        let bb = BitBoard::from_board(board);
        let mut result = self.search_impl(&bb, None, false);
        if temperature > 0.0 && result.best_move.is_some() {
            result.best_move = self.sample_move_by_temp(0, temperature);
        }
        result
    }

    // -----------------------------------------------------------------------
    // #3: Network self-play game loop (game loop in Rust, NN callback to Python)
    // -----------------------------------------------------------------------

    /// Play one complete self-play game using network MCTS with batched evaluation.
    /// The game loop runs in Rust; only the network forward pass calls Python.
    #[pyo3(signature = (eval_fn, batch_size=8, random_opening=4, max_moves=80, temp_moves=15, temperature=1.0, full_search_fraction=1.0, cheap_sims=50))]
    fn play_network_game(
        &mut self,
        py: Python<'_>,
        eval_fn: PyObject,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        temperature: f32,
        full_search_fraction: f32,
        cheap_sims: u32,
    ) -> PyResult<SelfPlayResult> {
        self.play_network_game_impl(
            py, &eval_fn, batch_size, random_opening, max_moves, temp_moves, temperature,
            full_search_fraction, cheap_sims,
        )
    }

    // -----------------------------------------------------------------------
    // #4: Evaluation match (MCTS vs alpha-beta, game loop in Rust)
    // -----------------------------------------------------------------------

    /// Play N evaluation games: network MCTS vs alpha-beta engine.
    /// Alternates colors. Returns (wins, draws, losses) for MCTS.
    #[pyo3(signature = (eval_fn, num_games=20, opponent_depth=5, batch_size=8, random_opening=2, max_moves=80))]
    fn play_eval_match(
        &mut self,
        py: Python<'_>,
        eval_fn: PyObject,
        num_games: usize,
        opponent_depth: u8,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
    ) -> PyResult<EvalMatchResult> {
        self.play_eval_match_impl(
            py, &eval_fn, num_games, opponent_depth, batch_size, random_opening, max_moves,
        )
    }

    // -----------------------------------------------------------------------
    // #5: ONNX self-play (pure Rust, no Python callbacks)
    // -----------------------------------------------------------------------

    /// Play one self-play game using ONNX inference (pure Rust, no Python/GIL).
    #[pyo3(signature = (onnx_session, batch_size=8, random_opening=4, max_moves=80, temp_moves=15, temperature=1.0, full_search_fraction=1.0, cheap_sims=50))]
    fn play_network_game_onnx(
        &mut self,
        onnx_session: &mut OnnxSession,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        temperature: f32,
        full_search_fraction: f32,
        cheap_sims: u32,
    ) -> PyResult<SelfPlayResult> {
        self.play_network_game_onnx_impl(
            onnx_session, batch_size, random_opening, max_moves, temp_moves, temperature,
            full_search_fraction, cheap_sims,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Play N self-play games using ONNX inference (pure Rust, no Python/GIL).
    #[pyo3(signature = (onnx_session, n_games, batch_size=8, random_opening=4, max_moves=80, temp_moves=15, temperature=1.0, full_search_fraction=1.0, cheap_sims=50))]
    fn play_network_games_onnx(
        &mut self,
        onnx_session: &mut OnnxSession,
        n_games: usize,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        temperature: f32,
        full_search_fraction: f32,
        cheap_sims: u32,
    ) -> PyResult<Vec<SelfPlayResult>> {
        let mut results = Vec::with_capacity(n_games);
        for _ in 0..n_games {
            let r = self.play_network_game_onnx_impl(
                onnx_session, batch_size, random_opening, max_moves, temp_moves, temperature,
                full_search_fraction, cheap_sims,
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            results.push(r);
        }
        Ok(results)
    }

    /// Play one training game: network MCTS vs alpha-beta heuristic.
    /// Collects training examples from the network's turns only.
    /// The network alternates between white and black across calls.
    #[pyo3(signature = (onnx_session, opponent_depth=5, batch_size=8, random_opening=2, max_moves=80, temp_moves=3, temperature=0.8, mcts_is_white=true))]
    fn play_mixed_game_onnx(
        &mut self,
        onnx_session: &mut OnnxSession,
        opponent_depth: u8,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        temperature: f32,
        mcts_is_white: bool,
    ) -> PyResult<SelfPlayResult> {
        self.play_mixed_game_onnx_impl(
            onnx_session, opponent_depth, batch_size, random_opening, max_moves,
            temp_moves, temperature, mcts_is_white,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Play N evaluation games using ONNX inference: MCTS vs alpha-beta engine.
    #[pyo3(signature = (onnx_session, num_games=20, opponent_depth=5, batch_size=8, random_opening=2, max_moves=80))]
    fn play_eval_match_onnx(
        &mut self,
        onnx_session: &mut OnnxSession,
        num_games: usize,
        opponent_depth: u8,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
    ) -> PyResult<EvalMatchResult> {
        self.play_eval_match_onnx_impl(
            onnx_session, num_games, opponent_depth, batch_size, random_opening, max_moves,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

// ---------------------------------------------------------------------------
// Core MCTS implementation
// ---------------------------------------------------------------------------

impl MCTSEngine {
    /// Generate moves, optionally filtering backward barrel moves in forward-only mode.
    /// Placements, pail moves, sideways moves, and jumps toward/across the goal are kept.
    /// Only simple one-step backward barrel moves are removed.
    fn gen_moves(&self, bb: &BitBoard) -> Vec<BitMove> {
        let mut moves = bb.generate_moves();
        if self.forward_only {
            let player = bb.current_player;
            moves.retain(|m| {
                // Pail-only sub-moves always allowed
                if m.is_pail_placement() {
                    return true;
                }
                // Placements from off-board always allowed (on starting row)
                if m.is_placement() {
                    return true;
                }
                // Barrel moves: check direction
                if let Some(from_sq) = m.barrel_from() {
                    let (from_row, _) = sq_to_coords(from_sq as usize);
                    let (to_row, _) = sq_to_coords(m.barrel_to() as usize);
                    match player {
                        // White moves toward row 0 (forward = to_row <= from_row)
                        Player::White => to_row <= from_row,
                        // Black moves toward row 5 (forward = to_row >= from_row)
                        Player::Black => to_row >= from_row,
                    }
                } else {
                    true // no from_sq means placement, keep it
                }
            });
        }
        moves
    }

    fn search_impl(
        &mut self,
        bb: &BitBoard,
        eval_fn: Option<(Python<'_>, &PyObject)>,
        add_noise: bool,
    ) -> MCTSSearchResult {
        let is_white = bb.current_player == Player::White;

        // Reuse tree if available, otherwise initialize fresh
        let reused = !self.nodes.is_empty()
            && self.nodes[0].children_count > 0
            && self.nodes.len() < 500_000;
        if !reused {
            self.nodes.clear();
            self.nodes.push(MCTSNode::root(is_white));

            // Expand root
            let _root_value = self.evaluate_and_expand(0, bb, &eval_fn);
        }

        if self.nodes[0].children_count == 0 {
            return MCTSSearchResult::empty();
        }

        // Apply Dirichlet noise at root for self-play exploration
        if add_noise {
            self.apply_dirichlet_noise(0);
        }

        // Single legal move: no search needed
        if self.nodes[0].children_count == 1 {
            let child_idx = self.nodes[0].children_start as usize;
            self.nodes[child_idx].visit_count = 1;
            return self.build_result(0);
        }

        // Run simulations
        for _ in 0..self.simulations {
            let mut node_idx: usize = 0;
            let mut sim_bb = *bb; // BitBoard is Copy

            // SELECT: walk to unexpanded leaf
            while self.nodes[node_idx].is_expanded()
                && !self.nodes[node_idx].is_terminal
            {
                node_idx = self.select_child(node_idx);
                sim_bb.make_move(&self.nodes[node_idx].bitmove);
            }

            // EVALUATE + EXPAND
            let value = if self.nodes[node_idx].is_terminal {
                self.nodes[node_idx].terminal_value
            } else {
                self.evaluate_and_expand(node_idx, &sim_bb, &eval_fn)
            };

            // BACKPROPAGATE
            let mut idx = node_idx;
            loop {
                self.nodes[idx].visit_count += 1;
                self.nodes[idx].total_value += value;
                if self.nodes[idx].parent == u32::MAX {
                    break;
                }
                idx = self.nodes[idx].parent as usize;
            }
        }

        self.build_result(0)
    }

    /// PUCT child selection with FPU reduction.
    fn select_child(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        let sqrt_parent = (node.visit_count as f32).sqrt();

        // FPU: unvisited children use parent's average minus fpu_reduction
        let parent_q = node.q_value();
        let parent_q_adj = if node.player_is_white { parent_q } else { -parent_q };

        let mut best_idx = node.children_start as usize;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..node.children_count as usize {
            let child_idx = node.children_start as usize + i;
            let child = &self.nodes[child_idx];

            let q_adjusted = if child.visit_count == 0 {
                // FPU: assume unvisited is slightly worse than parent average
                parent_q_adj - self.fpu_reduction
            } else {
                let q = child.q_value();
                if node.player_is_white { q } else { -q }
            };

            let exploration =
                self.c_puct * child.prior * sqrt_parent / (1.0 + child.visit_count as f32);
            let score = q_adjusted + exploration;

            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }

        best_idx
    }

    /// Evaluate leaf and expand node with children.
    fn evaluate_and_expand(
        &mut self,
        node_idx: usize,
        bb: &BitBoard,
        eval_fn: &Option<(Python<'_>, &PyObject)>,
    ) -> f32 {
        // Check terminal
        if let Some(winner) = bb.check_winner() {
            self.nodes[node_idx].is_terminal = true;
            let v = if winner == Player::White { 1.0 } else { -1.0 };
            self.nodes[node_idx].terminal_value = v;
            return v;
        }

        let moves = self.gen_moves(&bb);
        if moves.is_empty() {
            self.nodes[node_idx].is_terminal = true;
            self.nodes[node_idx].terminal_value = 0.0;
            return 0.0;
        }

        let (value, priors) = match eval_fn {
            Some((py, func)) => {
                // Network evaluation: call Python
                match self.call_network(*py, func, bb, &moves) {
                    Ok(result) => result,
                    Err(_) => {
                        // Fallback to heuristic on error
                        let v = (self.engine.evaluate_heuristic(bb) as f32 / SCORE_SCALING).tanh();
                        let uniform = 1.0 / moves.len() as f32;
                        (v, vec![uniform; moves.len()])
                    }
                }
            }
            None => {
                // Heuristic evaluation with cache
                let hash = bb.hash;
                let v = if let Some(&cached_v) = self.heuristic_cache.get(&hash) {
                    cached_v
                } else {
                    let v = (self.engine.evaluate_heuristic(bb) as f32 / SCORE_SCALING).tanh();
                    if self.heuristic_cache.len() < 100_000 {
                        self.heuristic_cache.insert(hash, v);
                    } else {
                        self.heuristic_cache.clear();
                        self.heuristic_cache.insert(hash, v);
                    }
                    v
                };
                let uniform = 1.0 / moves.len() as f32;
                (v, vec![uniform; moves.len()])
            }
        };

        // Expand: create children
        // Derive child_is_white from board state: after pail sub-move, same player acts
        let child_is_white = if !moves.is_empty() && moves[0].is_pail_placement() {
            bb.current_player == Player::White // same player for pail sub-move children
        } else {
            bb.current_player != Player::White // opposite player for barrel moves
        };
        let children_start = self.nodes.len() as u32;
        let children_count = moves.len() as u16;

        for (bm, prior) in moves.iter().zip(priors.iter()) {
            self.nodes.push(MCTSNode {
                parent: node_idx as u32,
                children_start: u32::MAX,
                children_count: 0,
                bitmove: *bm,
                visit_count: 0,
                total_value: 0.0,
                prior: *prior,
                is_terminal: false,
                terminal_value: 0.0,
                player_is_white: child_is_white,
            });
        }

        self.nodes[node_idx].children_start = children_start;
        self.nodes[node_idx].children_count = children_count;

        // Penalise revisiting positions from earlier in the game
        self.apply_repetition_penalty(value, bb.hash)
    }

    /// Call Python network for evaluation.
    fn call_network(
        &self,
        py: Python<'_>,
        eval_fn: &PyObject,
        bb: &BitBoard,
        moves: &[BitMove],
    ) -> PyResult<(f32, Vec<f32>)> {
        let planes = bb_to_planes(bb);
        let result = eval_fn.call1(py, (planes,))?;
        let (raw_logits, value): (Vec<f32>, f32) = result.extract(py)?;

        // For Black positions, un-flip policy logits from relative to physical coords
        let policy_logits = if bb.current_player != Player::White {
            Self::vflip_policy(&raw_logits)
        } else {
            raw_logits
        };

        // Masked softmax over legal moves
        let mut max_logit = f32::NEG_INFINITY;
        for bm in moves {
            let idx = policy_index(bm) as usize;
            if idx < policy_logits.len() && policy_logits[idx] > max_logit {
                max_logit = policy_logits[idx];
            }
        }

        let mut exp_sum = 0.0f32;
        let mut priors: Vec<f32> = moves
            .iter()
            .map(|bm| {
                let idx = policy_index(bm) as usize;
                let logit = if idx < policy_logits.len() {
                    policy_logits[idx]
                } else {
                    0.0
                };
                let e = (logit - max_logit).exp();
                exp_sum += e;
                e
            })
            .collect();

        // Normalize
        if exp_sum > 0.0 {
            for p in &mut priors {
                *p /= exp_sum;
            }
        }

        Ok((value, priors))
    }

    /// Build result from root node.
    fn build_result(&self, root_idx: usize) -> MCTSSearchResult {
        let root = &self.nodes[root_idx];

        // Build policy target from visit counts
        let mut policy_target = vec![0.0f32; POLICY_SIZE];
        let total_visits: u32 = (0..root.children_count as usize)
            .map(|i| self.nodes[root.children_start as usize + i].visit_count)
            .sum();

        if total_visits > 0 {
            for i in 0..root.children_count as usize {
                let child = &self.nodes[root.children_start as usize + i];
                let idx = policy_index(&child.bitmove) as usize;
                if idx < POLICY_SIZE {
                    policy_target[idx] += child.visit_count as f32 / total_visits as f32;
                }
            }
        }

        // Find most-visited child
        let mut best_visits = 0u32;
        let mut best_child_idx = root.children_start as usize;
        for i in 0..root.children_count as usize {
            let child_idx = root.children_start as usize + i;
            if self.nodes[child_idx].visit_count > best_visits {
                best_visits = self.nodes[child_idx].visit_count;
                best_child_idx = child_idx;
            }
        }

        let best_move = if root.children_count > 0 {
            Some(self.nodes[best_child_idx].bitmove.to_move())
        } else {
            None
        };

        MCTSSearchResult {
            best_move,
            visits: total_visits,
            policy_target,
            root_value: root.q_value(),
        }
    }

    // -----------------------------------------------------------------------
    // Batched network MCTS (virtual loss)
    // -----------------------------------------------------------------------

    /// Virtual loss constant: applied to discourage re-selecting same path.
    const VIRTUAL_LOSS: f32 = 3.0;

    /// Batched MCTS: select multiple leaves, evaluate in one Python call, backprop all.
    fn search_batched_impl(
        &mut self,
        bb: &BitBoard,
        py: Python<'_>,
        eval_fn: &PyObject,
        batch_size: u32,
        add_noise: bool,
    ) -> MCTSSearchResult {
        let is_white = bb.current_player == Player::White;

        // Reuse tree if available, otherwise initialize fresh
        let reused = !self.nodes.is_empty()
            && self.nodes[0].children_count > 0
            && self.nodes.len() < 500_000;
        if !reused {
            self.nodes.clear();
            self.nodes.push(MCTSNode::root(is_white));

            // Expand root with single eval
            let root_planes = bb_to_planes(bb);
            let root_moves = self.gen_moves(&bb);
            if root_moves.is_empty() {
                return MCTSSearchResult::empty();
            }

            // Evaluate root
            // Derive child_is_white: after pail sub-move, same player; after barrel, opposite
            let root_child_is_white = if !root_moves.is_empty() && root_moves[0].is_pail_placement() {
                bb.current_player == Player::White
            } else {
                bb.current_player != Player::White
            };
            match self.call_network_single(py, eval_fn, &root_planes) {
                Ok((raw_logits, value)) => {
                    // For Black positions, un-flip policy logits from relative to physical coords
                    let policy_logits = if !is_white { Self::vflip_policy(&raw_logits) } else { raw_logits };
                    let priors = Self::masked_softmax(&policy_logits, &root_moves);
                    self.expand_node(0, &root_moves, &priors, root_child_is_white);
                    // Network returns current-player perspective; convert to White perspective
                    let value_white = if is_white { value } else { -value };
                    self.nodes[0].visit_count += 1;
                    self.nodes[0].total_value += value_white;
                }
                Err(_) => {
                    let v = (self.engine.evaluate_heuristic(bb) as f32 / SCORE_SCALING).tanh();
                    let uniform = 1.0 / root_moves.len() as f32;
                    let priors = vec![uniform; root_moves.len()];
                    self.expand_node(0, &root_moves, &priors, root_child_is_white);
                    self.nodes[0].visit_count += 1;
                    self.nodes[0].total_value += v;
                }
            }
        }

        if self.nodes[0].children_count == 0 {
            return MCTSSearchResult::empty();
        }

        // Apply Dirichlet noise at root for self-play exploration
        if add_noise {
            self.apply_dirichlet_noise(0);
        }

        if self.nodes[0].children_count == 1 {
            let child_idx = self.nodes[0].children_start as usize;
            self.nodes[child_idx].visit_count = 1;
            return self.build_result(0);
        }

        // Run batched simulations
        let mut remaining = self.simulations;
        while remaining > 0 {
            let n = std::cmp::min(batch_size, remaining) as usize;

            // Phase 1: Select leaves with virtual loss
            struct PendingLeaf {
                node_idx: usize,
                bb: BitBoard,
                moves: Vec<BitMove>,
            }
            let mut pending: Vec<PendingLeaf> = Vec::new();
            let mut terminal_backprops: Vec<(usize, f32)> = Vec::new();

            for _ in 0..n {
                let mut node_idx: usize = 0;
                let mut sim_bb = *bb;

                // SELECT: walk to unexpanded leaf
                while self.nodes[node_idx].is_expanded()
                    && !self.nodes[node_idx].is_terminal
                {
                    node_idx = self.select_child(node_idx);
                    sim_bb.make_move(&self.nodes[node_idx].bitmove);
                }

                // Terminal node
                if self.nodes[node_idx].is_terminal {
                    terminal_backprops.push((node_idx, self.nodes[node_idx].terminal_value));
                    continue;
                }

                // Check if this leaf is terminal
                if let Some(winner) = sim_bb.check_winner() {
                    self.nodes[node_idx].is_terminal = true;
                    let v = if winner == Player::White { 1.0 } else { -1.0 };
                    self.nodes[node_idx].terminal_value = v;
                    terminal_backprops.push((node_idx, v));
                    continue;
                }

                let moves = self.gen_moves(&sim_bb);
                if moves.is_empty() {
                    self.nodes[node_idx].is_terminal = true;
                    self.nodes[node_idx].terminal_value = 0.0;
                    terminal_backprops.push((node_idx, 0.0));
                    continue;
                }

                // Check network cache before adding to batch
                let hash = sim_bb.hash;
                if let Some((cached_logits, cached_value)) = self.network_cache.get(&hash).cloned() {
                    let priors = Self::masked_softmax(&cached_logits, &moves);
                    let child_is_white = if !moves.is_empty() && moves[0].is_pail_placement() {
                        sim_bb.current_player == Player::White
                    } else {
                        sim_bb.current_player != Player::White
                    };
                    self.expand_node(node_idx, &moves, &priors, child_is_white);
                    let penalized_cv = self.apply_repetition_penalty(cached_value, hash);
                    self.backpropagate(node_idx, penalized_cv);
                    continue;
                }

                // Apply virtual loss along path to root
                let mut vl_idx = node_idx;
                loop {
                    self.nodes[vl_idx].visit_count += Self::VIRTUAL_LOSS as u32;
                    self.nodes[vl_idx].total_value -= Self::VIRTUAL_LOSS;
                    if self.nodes[vl_idx].parent == u32::MAX {
                        break;
                    }
                    vl_idx = self.nodes[vl_idx].parent as usize;
                }

                pending.push(PendingLeaf {
                    node_idx,
                    bb: sim_bb,
                    moves,
                });
            }

            // Phase 2: Batch evaluate all pending leaves
            if !pending.is_empty() {
                let batch_planes: Vec<Vec<f32>> = pending
                    .iter()
                    .map(|p| bb_to_planes(&p.bb))
                    .collect();

                let batch_results = self.call_network_batch(py, eval_fn, &batch_planes);

                for (i, leaf) in pending.iter().enumerate() {
                    // Remove virtual loss
                    let mut vl_idx = leaf.node_idx;
                    loop {
                        self.nodes[vl_idx].visit_count -= Self::VIRTUAL_LOSS as u32;
                        self.nodes[vl_idx].total_value += Self::VIRTUAL_LOSS;
                        if self.nodes[vl_idx].parent == u32::MAX {
                            break;
                        }
                        vl_idx = self.nodes[vl_idx].parent as usize;
                    }

                    // Get network output (or fallback to heuristic)
                    // Network returns current-player perspective; convert to White for MCTS tree
                    let leaf_is_white = leaf.bb.current_player == Player::White;
                    let (value, priors) = match &batch_results {
                        Ok((policies, values)) => {
                            let value_white = if leaf_is_white { values[i] } else { -values[i] };
                            // For Black, un-flip policy logits from relative to physical coords
                            let policy_logits = if !leaf_is_white {
                                Self::vflip_policy(&policies[i])
                            } else {
                                policies[i].clone()
                            };
                            // Cache in White perspective with physical-coord logits
                            let hash = leaf.bb.hash;
                            if self.network_cache.len() < 100_000 {
                                self.network_cache.insert(hash, (policy_logits.clone(), value_white));
                            } else {
                                self.network_cache.clear();
                                self.network_cache.insert(hash, (policy_logits.clone(), value_white));
                            }
                            let priors = Self::masked_softmax(&policy_logits, &leaf.moves);
                            (value_white, priors)
                        }
                        Err(_) => {
                            let v = (self.engine.evaluate_heuristic(&leaf.bb) as f32
                                / SCORE_SCALING)
                                .tanh();
                            let uniform = 1.0 / leaf.moves.len() as f32;
                            (v, vec![uniform; leaf.moves.len()])
                        }
                    };

                    // Expand node
                    let child_is_white = if !leaf.moves.is_empty() && leaf.moves[0].is_pail_placement() {
                        leaf.bb.current_player == Player::White
                    } else {
                        leaf.bb.current_player != Player::White
                    };
                    self.expand_node(leaf.node_idx, &leaf.moves, &priors, child_is_white);

                    // Penalise revisiting positions from earlier in the game
                    let value = self.apply_repetition_penalty(value, leaf.bb.hash);

                    // Backpropagate
                    self.backpropagate(leaf.node_idx, value);
                }
            }

            // Phase 3: Backprop terminal nodes
            for (node_idx, value) in &terminal_backprops {
                self.backpropagate(*node_idx, *value);
            }

            remaining -= n as u32;
        }

        self.build_result(0)
    }

    /// Expand a node with children (used by batched search).
    fn expand_node(
        &mut self,
        node_idx: usize,
        moves: &[BitMove],
        priors: &[f32],
        child_is_white: bool,
    ) {
        let children_start = self.nodes.len() as u32;
        let children_count = moves.len() as u16;

        for (bm, prior) in moves.iter().zip(priors.iter()) {
            self.nodes.push(MCTSNode {
                parent: node_idx as u32,
                children_start: u32::MAX,
                children_count: 0,
                bitmove: *bm,
                visit_count: 0,
                total_value: 0.0,
                prior: *prior,
                is_terminal: false,
                terminal_value: 0.0,
                player_is_white: child_is_white,
            });
        }

        self.nodes[node_idx].children_start = children_start;
        self.nodes[node_idx].children_count = children_count;
    }

    /// Backpropagate value from node to root.
    fn backpropagate(&mut self, start_idx: usize, value: f32) {
        let mut idx = start_idx;
        loop {
            self.nodes[idx].visit_count += 1;
            self.nodes[idx].total_value += value;
            if self.nodes[idx].parent == u32::MAX {
                break;
            }
            idx = self.nodes[idx].parent as usize;
        }
    }

    /// Permute a policy logit vector through the vertical flip mapping.
    /// Converts from relative (flipped) coordinates to physical coordinates, or vice versa.
    fn vflip_policy(logits: &[f32]) -> Vec<f32> {
        let table = &*POLICY_VFLIP;
        let mut out = vec![0.0f32; POLICY_SIZE];
        for i in 0..POLICY_SIZE.min(logits.len()) {
            out[table[i] as usize] = logits[i];
        }
        out
    }

    /// Masked softmax: extract priors for legal moves from full policy logits.
    fn masked_softmax(policy_logits: &[f32], moves: &[BitMove]) -> Vec<f32> {
        let mut max_logit = f32::NEG_INFINITY;
        for bm in moves {
            let idx = policy_index(bm) as usize;
            if idx < policy_logits.len() && policy_logits[idx] > max_logit {
                max_logit = policy_logits[idx];
            }
        }

        let mut exp_sum = 0.0f32;
        let mut priors: Vec<f32> = moves
            .iter()
            .map(|bm| {
                let idx = policy_index(bm) as usize;
                let logit = if idx < policy_logits.len() {
                    policy_logits[idx]
                } else {
                    0.0
                };
                let e = (logit - max_logit).exp();
                exp_sum += e;
                e
            })
            .collect();

        if exp_sum > 0.0 {
            for p in &mut priors {
                *p /= exp_sum;
            }
        }
        priors
    }

    /// Compute log-softmax (log-priors) for legal moves from full policy logits.
    /// Returns log(softmax(logit_i)) for each legal move.
    fn masked_log_softmax(policy_logits: &[f32], moves: &[BitMove]) -> Vec<f32> {
        let mut max_logit = f32::NEG_INFINITY;
        for bm in moves {
            let idx = policy_index(bm) as usize;
            if idx < policy_logits.len() && policy_logits[idx] > max_logit {
                max_logit = policy_logits[idx];
            }
        }

        let mut shifted: Vec<f32> = moves
            .iter()
            .map(|bm| {
                let idx = policy_index(bm) as usize;
                if idx < policy_logits.len() {
                    policy_logits[idx] - max_logit
                } else {
                    -30.0
                }
            })
            .collect();

        let log_sum_exp = shifted.iter().map(|s| s.exp()).sum::<f32>().ln();
        for s in &mut shifted {
            *s -= log_sum_exp;
        }
        shifted
    }

    /// Call Python network for a single position (used for root expansion).
    fn call_network_single(
        &self,
        py: Python<'_>,
        eval_fn: &PyObject,
        planes: &[f32],
    ) -> PyResult<(Vec<f32>, f32)> {
        // Wrap in a batch of 1
        let batch = vec![planes.to_vec()];
        let result = eval_fn.call1(py, (batch,))?;
        let (batch_policy, batch_values): (Vec<Vec<f32>>, Vec<f32>) = result.extract(py)?;
        Ok((batch_policy.into_iter().next().unwrap_or_default(),
            batch_values.into_iter().next().unwrap_or(0.0)))
    }

    /// Call Python network for a batch of positions.
    fn call_network_batch(
        &self,
        py: Python<'_>,
        eval_fn: &PyObject,
        batch_planes: &[Vec<f32>],
    ) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        let result = eval_fn.call1(py, (batch_planes.to_vec(),))?;
        let (batch_policy, batch_values): (Vec<Vec<f32>>, Vec<f32>) = result.extract(py)?;
        Ok((batch_policy, batch_values))
    }

    // -----------------------------------------------------------------------
    // Temperature-based move selection from MCTS tree
    // -----------------------------------------------------------------------

    /// Sample a BitMove from the root's children visit counts using temperature.
    /// Returns None if root has no children.
    fn sample_bitmove_by_temp(&mut self, root_idx: usize, temperature: f32) -> Option<BitMove> {
        let root = &self.nodes[root_idx];
        if root.children_count == 0 {
            return None;
        }

        let n_children = root.children_count as usize;
        let start = root.children_start as usize;

        let mut probs = Vec::with_capacity(n_children);
        for i in 0..n_children {
            let visits = self.nodes[start + i].visit_count as f64;
            if temperature <= 0.01 {
                probs.push(visits);
            } else {
                probs.push(visits.powf(1.0 / temperature as f64));
            }
        }

        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            return Some(self.nodes[start].bitmove);
        }
        for p in &mut probs {
            *p /= total;
        }

        let idx = self.rng.sample_distribution(&probs);
        Some(self.nodes[start + idx].bitmove)
    }

    /// Sample a Move (Python-facing) from the root's children visit counts using temperature.
    /// Returns None if root has no children.
    fn sample_move_by_temp(&mut self, root_idx: usize, temperature: f32) -> Option<Move> {
        self.sample_bitmove_by_temp(root_idx, temperature)
            .map(|bm| bm.to_move())
    }

    /// Select a move from a BitMove list using a policy target distribution.
    fn select_move_by_policy(
        &mut self,
        moves: &[BitMove],
        policy_target: &[f32],
        temperature: f32,
    ) -> BitMove {
        let mut probs: Vec<f64> = moves
            .iter()
            .map(|bm| {
                let idx = policy_index(bm) as usize;
                let p = if idx < policy_target.len() {
                    policy_target[idx] as f64
                } else {
                    0.0
                };
                if temperature > 0.01 {
                    p.powf(1.0 / temperature as f64)
                } else {
                    p
                }
            })
            .collect();

        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            return moves[self.rng.usize(moves.len())];
        }
        for p in &mut probs {
            *p /= total;
        }

        let idx = self.rng.sample_distribution(&probs);
        moves[idx]
    }

    // -----------------------------------------------------------------------
    // #1: Full heuristic self-play game (entirely in Rust)
    // -----------------------------------------------------------------------

    fn play_heuristic_game_impl(
        &mut self,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        full_search_fraction: f32,
        cheap_sims: u32,
    ) -> SelfPlayResult {
        self.clear_caches();
        let mut bb = BitBoard::new();
        let mut examples: Vec<(Vec<f32>, Vec<f32>, bool, f32)> = Vec::new();
        let original_sims = self.simulations;
        let mut position_history: Vec<u64> = Vec::new();

        // Random opening moves (count only full turns, not pail sub-moves)
        let mut random_turns = 0;
        while random_turns < random_opening {
            if bb.check_winner().is_some() {
                break;
            }
            let moves = self.gen_moves(&bb);
            if moves.is_empty() {
                break;
            }
            let idx = self.rng.usize(moves.len());
            let is_barrel_move = !moves[idx].is_pail_placement();
            bb.make_move(&moves[idx]);
            if is_barrel_move {
                random_turns += 1;
            }
        }

        let mut move_count = 0u32; // counts full turns only
        loop {
            if bb.check_winner().is_some() || (move_count as usize) >= max_moves {
                break;
            }

            // Adjudicate decisive positions early
            if self.should_adjudicate(&bb, move_count) {
                break;
            }

            // 3-fold repetition detection
            if move_count >= 15 && Self::is_threefold(&position_history, bb.hash) {
                break;
            }
            position_history.push(bb.hash);
            self.game_position_history.clone_from(&position_history);

            // Pail placement: skip MCTS, use center-biased random selection
            if let Some(pail_move) = self.random_center_pail(&bb) {
                bb.make_move(&pail_move);
                continue;
            }

            // Playout cap randomization: full search or cheap search
            let is_full_search = full_search_fraction >= 1.0
                || self.rng.f64() < full_search_fraction as f64;
            self.simulations = if is_full_search { original_sims } else { cheap_sims };

            let result = self.search_impl(&bb, None, true); // add_noise=true for self-play
            if result.best_move.is_none() {
                break;
            }

            // Only collect training data from full-search moves
            if is_full_search {
                let is_white = bb.current_player == Player::White;
                let planes = bb_to_planes(&bb);
                // root_value is from White's perspective; convert to current player
                let search_score = if is_white { result.root_value } else { -result.root_value };
                examples.push((planes, result.policy_target.clone(), is_white, search_score));
            }

            // Temperature-based selection for first moves
            let chosen_bitmove = if (move_count as usize) < temp_moves {
                let moves = self.gen_moves(&bb);
                self.select_move_by_policy(&moves, &result.policy_target, 1.0)
            } else {
                // Find the BitMove for the best move
                let root = &self.nodes[0];
                let mut best_visits = 0u32;
                let mut best_bm = self.nodes[root.children_start as usize].bitmove;
                for i in 0..root.children_count as usize {
                    let child = &self.nodes[root.children_start as usize + i];
                    if child.visit_count > best_visits {
                        best_visits = child.visit_count;
                        best_bm = child.bitmove;
                    }
                }
                best_bm
            };

            // Retain subtree for chosen move before making it
            let is_barrel_move = !chosen_bitmove.is_pail_placement();
            self.retain_subtree(&chosen_bitmove);
            bb.make_move(&chosen_bitmove);
            if is_barrel_move {
                move_count += 1;
            }
        }

        // Restore original simulations count
        self.simulations = original_sims;

        // Clear tree at end of game
        self.nodes.clear();

        // Determine outcome (White perspective: +1 White win, -1 Black win)
        let (outcome_white, winner_str) = self.determine_outcome(&bb);

        // Fill in outcomes: blend game outcome with per-position search score
        let training_examples: Vec<TrainingExample> = examples
            .into_iter()
            .map(|(planes, policy, is_white, search_score)| {
                let game_outcome = if is_white { outcome_white } else { -outcome_white };
                let value_target = self.blend_value_target(game_outcome, search_score);
                // For Black positions, flip policy target to relative coords (matching network output)
                let policy_target = if is_white { policy } else { Self::vflip_policy(&policy) };
                TrainingExample {
                    planes,
                    policy_target,
                    value_target,
                    search_score,
                }
            })
            .collect();

        SelfPlayResult {
            examples: training_examples,
            winner: winner_str,
            move_count,
        }
    }

    // -----------------------------------------------------------------------
    // #1b impl: Alpha-beta self-play game
    // -----------------------------------------------------------------------

    fn play_alphabeta_game_impl(
        &mut self,
        depth: u8,
        random_opening: usize,
        max_moves: usize,
    ) -> SelfPlayResult {
        let mut bb = BitBoard::new();
        let mut examples: Vec<(Vec<f32>, Vec<f32>, bool, f32)> = Vec::new();
        let mut position_history: Vec<u64> = Vec::new();

        // Random opening moves (count only full turns)
        let mut random_turns = 0;
        while random_turns < random_opening {
            if bb.check_winner().is_some() {
                break;
            }
            let moves = self.gen_moves(&bb);
            if moves.is_empty() {
                break;
            }
            let idx = self.rng.usize(moves.len());
            let is_barrel_move = !moves[idx].is_pail_placement();
            bb.make_move(&moves[idx]);
            if is_barrel_move {
                random_turns += 1;
            }
        }

        let mut move_count = 0u32; // counts full turns only
        loop {
            if bb.check_winner().is_some() || (move_count as usize) >= max_moves {
                break;
            }

            // Adjudicate decisive positions early
            if self.should_adjudicate(&bb, move_count) {
                break;
            }

            // 3-fold repetition detection
            if move_count >= 15 && Self::is_threefold(&position_history, bb.hash) {
                break;
            }
            position_history.push(bb.hash);
            self.game_position_history.clone_from(&position_history);

            // Pail placement: skip alpha-beta, use center-biased random selection
            if let Some(pail_move) = self.random_center_pail(&bb) {
                bb.make_move(&pail_move);
                continue;
            }

            let moves = self.gen_moves(&bb);
            if moves.is_empty() {
                break;
            }

            let is_white = bb.current_player == Player::White;

            // Full-depth alpha-beta search to pick the move to play
            let (ab_score, best_bitmove) = self.engine.search(&bb, depth);
            let best_bm = match best_bitmove {
                Some(bm) => bm,
                None => break,
            };

            // Cheap heuristic eval of all child positions for soft policy target
            let mut policy_target = vec![0.0f32; 37 * 36]; // POLICY_SIZE
            let mut max_score = f32::NEG_INFINITY;
            let mut move_scores: Vec<(usize, f32)> = Vec::with_capacity(moves.len());

            for (i, mv) in moves.iter().enumerate() {
                let mut child_bb = bb;
                child_bb.make_move(mv);

                let score = if let Some(winner) = child_bb.check_winner() {
                    if winner == Player::White { 100_000.0 } else { -100_000.0 }
                } else {
                    self.engine.evaluate_heuristic(&child_bb) as f32
                };
                // Convert to current player perspective
                let player_score = if is_white { score } else { -score };
                if player_score > max_score {
                    max_score = player_score;
                }
                move_scores.push((i, player_score));
            }

            // Softmax over heuristic scores
            let policy_temp = 150.0f32;
            let mut exp_sum = 0.0f32;
            for &(i, score) in &move_scores {
                let logit = (score - max_score) / policy_temp;
                let exp_val = logit.exp();
                let idx = policy_index(&moves[i]) as usize;
                if idx < policy_target.len() {
                    policy_target[idx] = exp_val;
                    exp_sum += exp_val;
                }
            }
            if exp_sum > 0.0 {
                for v in policy_target.iter_mut() {
                    *v /= exp_sum;
                }
            }

            // Collect training data
            let planes = bb_to_planes(&bb);
            // ab_score is from White's perspective; normalize to [-1, 1] and flip for Black
            let search_score = (ab_score as f32 / 600.0).clamp(-1.0, 1.0);
            let search_score = if is_white { search_score } else { -search_score };
            examples.push((planes, policy_target, is_white, search_score));

            // Play the best move from alpha-beta
            let is_barrel_move = !best_bm.is_pail_placement();
            bb.make_move(&best_bm);
            if is_barrel_move {
                move_count += 1;
            }
        }

        // Determine outcome
        let (outcome_white, winner_str) = self.determine_outcome(&bb);

        let training_examples: Vec<TrainingExample> = examples
            .into_iter()
            .map(|(planes, policy, is_white, search_score)| {
                let game_outcome = if is_white { outcome_white } else { -outcome_white };
                let value_target = self.blend_value_target(game_outcome, search_score);
                // For Black positions, flip policy target to relative coords (matching network output)
                let policy_target = if is_white { policy } else { Self::vflip_policy(&policy) };
                TrainingExample {
                    planes,
                    policy_target,
                    value_target,
                    search_score,
                }
            })
            .collect();

        SelfPlayResult {
            examples: training_examples,
            winner: winner_str,
            move_count,
        }
    }

    // -----------------------------------------------------------------------
    // #3: Network self-play game loop (game loop in Rust, NN in Python)
    // -----------------------------------------------------------------------

    fn play_network_game_impl(
        &mut self,
        py: Python<'_>,
        eval_fn: &PyObject,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        temperature: f32,
        full_search_fraction: f32,
        cheap_sims: u32,
    ) -> PyResult<SelfPlayResult> {
        self.clear_caches();
        let mut bb = BitBoard::new();
        let mut examples: Vec<(Vec<f32>, Vec<f32>, bool, f32)> = Vec::new();
        let original_sims = self.simulations;
        let mut position_history: Vec<u64> = Vec::new();

        // Random opening moves (count only full turns)
        let mut random_turns = 0;
        while random_turns < random_opening {
            if bb.check_winner().is_some() {
                break;
            }
            let moves = self.gen_moves(&bb);
            if moves.is_empty() {
                break;
            }
            let idx = self.rng.usize(moves.len());
            let is_barrel_move = !moves[idx].is_pail_placement();
            bb.make_move(&moves[idx]);
            if is_barrel_move {
                random_turns += 1;
            }
        }

        let mut move_count = 0u32; // counts full turns only
        loop {
            if bb.check_winner().is_some() || (move_count as usize) >= max_moves {
                break;
            }

            // Adjudicate decisive positions early
            if self.should_adjudicate(&bb, move_count) {
                break;
            }

            // 3-fold repetition detection
            if move_count >= 15 && Self::is_threefold(&position_history, bb.hash) {
                break;
            }
            position_history.push(bb.hash);
            self.game_position_history.clone_from(&position_history);

            // Pail placement: skip MCTS, use center-biased random selection
            if let Some(pail_move) = self.random_center_pail(&bb) {
                bb.make_move(&pail_move);
                // No training data for forced-random pail moves
                continue;
            }

            // Playout cap randomization: full search or cheap search
            let is_full_search = full_search_fraction >= 1.0
                || self.rng.f64() < full_search_fraction as f64;
            self.simulations = if is_full_search { original_sims } else { cheap_sims };

            let result = if self.use_gumbel {
                self.search_gumbel_batched_py_impl(&bb, py, eval_fn, batch_size)
            } else {
                self.search_batched_impl(&bb, py, eval_fn, batch_size, true) // add_noise=true
            };
            if result.best_move.is_none() {
                break;
            }

            // Only collect training data from full-search moves
            if is_full_search {
                let is_white = bb.current_player == Player::White;
                let planes = bb_to_planes(&bb);
                // root_value is from White's perspective; convert to current player
                let search_score = if is_white { result.root_value } else { -result.root_value };
                examples.push((planes, result.policy_target.clone(), is_white, search_score));
            }

            // Move selection
            let chosen_bitmove = if self.use_gumbel {
                if (move_count as usize) < temp_moves && temperature > 0.0 {
                    // Temperature sampling from visit counts (not improved policy)
                    self.sample_bitmove_by_temp(0, temperature)
                        .unwrap_or_else(|| self.most_visited_bitmove())
                } else {
                    // Argmax from Gumbel search result
                    let best_move = result.best_move.as_ref().unwrap();
                    let best_pi = Self::move_policy_index(best_move);
                    let root = &self.nodes[0];
                    let s = root.children_start as usize;
                    let mut bm = self.nodes[s].bitmove;
                    for i in 0..root.children_count as usize {
                        if policy_index(&self.nodes[s + i].bitmove) == best_pi {
                            bm = self.nodes[s + i].bitmove;
                            break;
                        }
                    }
                    bm
                }
            } else if (move_count as usize) < temp_moves && temperature > 0.0 {
                self.sample_bitmove_by_temp(0, temperature)
                    .unwrap_or_else(|| self.most_visited_bitmove())
            } else {
                self.most_visited_bitmove()
            };

            let is_barrel_move = !chosen_bitmove.is_pail_placement();
            if !self.use_gumbel {
                self.retain_subtree(&chosen_bitmove);
            }
            bb.make_move(&chosen_bitmove);
            if is_barrel_move {
                move_count += 1;
            }
        }

        // Restore original simulations count
        self.simulations = original_sims;

        // Clear tree at end of game
        self.nodes.clear();

        // Determine outcome (White perspective: +1 White win, -1 Black win)
        let (outcome_white, winner_str) = self.determine_outcome(&bb);

        // Fill in outcomes: blend game outcome with per-position search score
        let training_examples: Vec<TrainingExample> = examples
            .into_iter()
            .map(|(planes, policy, is_white, search_score)| {
                let game_outcome = if is_white { outcome_white } else { -outcome_white };
                let value_target = self.blend_value_target(game_outcome, search_score);
                // For Black positions, flip policy target to relative coords (matching network output)
                let policy_target = if is_white { policy } else { Self::vflip_policy(&policy) };
                TrainingExample {
                    planes,
                    policy_target,
                    value_target,
                    search_score,
                }
            })
            .collect();

        Ok(SelfPlayResult {
            examples: training_examples,
            winner: winner_str,
            move_count,
        })
    }

    // -----------------------------------------------------------------------
    // #4: Evaluation match (MCTS vs alpha-beta)
    // -----------------------------------------------------------------------

    fn play_eval_match_impl(
        &mut self,
        py: Python<'_>,
        eval_fn: &PyObject,
        num_games: usize,
        opponent_depth: u8,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
    ) -> PyResult<EvalMatchResult> {
        let mut wins = 0u32;
        let mut draws = 0u32;
        let mut losses = 0u32;

        for game_idx in 0..num_games {
            let mcts_is_white = game_idx % 2 == 0;

            let mut bb = BitBoard::new();
            self.engine.full_reset();
            self.clear_caches();

            // Random opening (count only full turns)
            let mut random_turns = 0;
            while random_turns < random_opening {
                if bb.check_winner().is_some() {
                    break;
                }
                let moves = self.gen_moves(&bb);
                if moves.is_empty() {
                    break;
                }
                let idx = self.rng.usize(moves.len());
                let is_barrel_move = !moves[idx].is_pail_placement();
                bb.make_move(&moves[idx]);
                if is_barrel_move {
                    random_turns += 1;
                }
            }

            let mut move_count = 0u32; // counts full turns only
            let mut position_history: Vec<u64> = Vec::new();
            loop {
                if bb.check_winner().is_some() || (move_count as usize) >= max_moves {
                    break;
                }

                // 3-fold repetition detection
                if move_count >= 15 && Self::is_threefold(&position_history, bb.hash) {
                    break;
                }
                position_history.push(bb.hash);
                self.game_position_history.clone_from(&position_history);

                // Pail placement: center-biased random for both sides
                if let Some(pail_move) = self.random_center_pail(&bb) {
                    bb.make_move(&pail_move);
                    continue;
                }

                let is_white_turn = bb.current_player == Player::White;
                let is_mcts_turn = is_white_turn == mcts_is_white;

                let chosen_bitmove = if is_mcts_turn {
                    // MCTS move (network) -- use Gumbel if enabled, standard otherwise
                    if self.use_gumbel {
                        let result = self.search_gumbel_batched_py_impl(&bb, py, eval_fn, batch_size);
                        match result.best_move {
                            Some(mv) => {
                                let best_pi = Self::move_policy_index(&mv);
                                let root = &self.nodes[0];
                                let s = root.children_start as usize;
                                let mut bm = self.nodes[s].bitmove;
                                for i in 0..root.children_count as usize {
                                    if policy_index(&self.nodes[s + i].bitmove) == best_pi {
                                        bm = self.nodes[s + i].bitmove;
                                        break;
                                    }
                                }
                                bm
                            }
                            None => break,
                        }
                    } else {
                        let result = self.search_batched_impl(&bb, py, eval_fn, batch_size, false);
                        match result.best_move {
                            Some(_) => self.most_visited_bitmove(),
                            None => break,
                        }
                    }
                } else {
                    // Alpha-beta opponent
                    self.engine.full_reset();
                    let (_, bm_opt) = self.engine.search(&bb, opponent_depth);
                    match bm_opt {
                        Some(bm) => bm,
                        None => break,
                    }
                };

                // Retain subtree for tree reuse
                let is_barrel_move = !chosen_bitmove.is_pail_placement();
                if is_mcts_turn {
                    self.retain_subtree(&chosen_bitmove);
                } else {
                    self.retain_subtree(&chosen_bitmove);
                }
                bb.make_move(&chosen_bitmove);
                if is_barrel_move {
                    move_count += 1;
                }
            }

            // Clear tree at end of game
            self.nodes.clear();

            // Determine result
            match bb.check_winner() {
                None => draws += 1,
                Some(winner) => {
                    let white_won = winner == Player::White;
                    if white_won == mcts_is_white {
                        wins += 1;
                    } else {
                        losses += 1;
                    }
                }
            }
        }

        Ok(EvalMatchResult { wins, draws, losses })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Retain the subtree rooted at the child matching `chosen_bitmove`.
    /// Promotes that child to root, remapping all indices.
    /// Returns true if subtree was found and retained, false otherwise (tree cleared).
    fn retain_subtree(&mut self, chosen_bitmove: &BitMove) -> bool {
        if self.nodes.is_empty() || self.nodes[0].children_count == 0 {
            self.nodes.clear();
            return false;
        }

        // Find the child matching chosen_bitmove
        let root = &self.nodes[0];
        let start = root.children_start as usize;
        let mut found_idx: Option<usize> = None;
        for i in 0..root.children_count as usize {
            if self.nodes[start + i].bitmove == *chosen_bitmove {
                found_idx = Some(start + i);
                break;
            }
        }

        let child_idx = match found_idx {
            Some(idx) => idx,
            None => {
                self.nodes.clear();
                return false;
            }
        };

        // BFS from child_idx, collecting all reachable nodes
        let mut old_to_new: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut new_nodes: Vec<MCTSNode> = Vec::new();
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        // Add the new root
        old_to_new.insert(child_idx, 0);
        new_nodes.push(self.nodes[child_idx].clone());
        queue.push_back(child_idx);

        while let Some(old_idx) = queue.pop_front() {
            let node = &self.nodes[old_idx];
            if node.children_count > 0 {
                let ch_start = node.children_start as usize;
                for i in 0..node.children_count as usize {
                    let old_child = ch_start + i;
                    if old_child < self.nodes.len() {
                        let new_idx = new_nodes.len();
                        old_to_new.insert(old_child, new_idx);
                        new_nodes.push(self.nodes[old_child].clone());
                        queue.push_back(old_child);
                    }
                }
            }
        }

        // Remap parent and children_start
        for new_idx in 0..new_nodes.len() {
            let node = &mut new_nodes[new_idx];
            if node.parent != u32::MAX {
                if let Some(&mapped) = old_to_new.get(&(node.parent as usize)) {
                    node.parent = mapped as u32;
                } else {
                    node.parent = u32::MAX; // orphan -> root
                }
            }
            if node.children_count > 0 && node.children_start != u32::MAX {
                if let Some(&mapped) = old_to_new.get(&(node.children_start as usize)) {
                    node.children_start = mapped as u32;
                } else {
                    // Children not found -- clear them
                    node.children_start = u32::MAX;
                    node.children_count = 0;
                }
            }
        }

        // Set new root's parent to u32::MAX
        new_nodes[0].parent = u32::MAX;

        self.nodes = new_nodes;
        true
    }

    /// Pick a center-biased random pail placement from the legal moves.
    /// Returns None if current position is not a pail placement.
    /// Weight = (6 - manhattan_dist_from_center)^2, so center squares are ~9x more likely than corners.
    fn random_center_pail(&mut self, bb: &BitBoard) -> Option<BitMove> {
        let moves = self.gen_moves(&bb);
        if moves.is_empty() || !moves[0].is_pail_placement() {
            return None;
        }

        // Compute center-biased weights
        let mut weights: Vec<f32> = Vec::with_capacity(moves.len());
        for m in &moves {
            let pail_sq = m.barrel_to() as usize; // pail_sq stored in barrel_to for pail-only moves
            if m.is_pail_placement() {
                let (row, col) = sq_to_coords(pail_sq);
                let dist = (row as f32 - 2.5).abs() + (col as f32 - 2.5).abs();
                let w = (6.0 - dist).max(0.5);
                weights.push(w * w);
            }
        }

        // Weighted random selection
        let total: f32 = weights.iter().sum();
        let mut r = self.rng.f64() as f32 * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return Some(moves[i]);
            }
        }
        Some(*moves.last().unwrap())
    }

    /// Check if a position hash has occurred 3 times (3-fold repetition).
    #[inline]
    fn is_threefold(history: &[u64], hash: u64) -> bool {
        history.iter().filter(|&&h| h == hash).count() >= 2
        // 2 previous occurrences + current = 3-fold
    }

    /// Clear evaluation caches (call between games, not between moves).
    fn clear_caches(&mut self) {
        self.heuristic_cache.clear();
        self.network_cache.clear();
    }

    /// Apply repetition penalty to a leaf value.
    /// If the position hash appears in the game-level history, shift the value
    /// toward 0.0 (draw-like) by `repetition_penalty` per occurrence.
    /// This makes the search avoid revisiting positions from earlier in the game.
    #[inline]
    fn apply_repetition_penalty(&self, value: f32, hash: u64) -> f32 {
        if self.repetition_penalty <= 0.0 || self.game_position_history.is_empty() {
            return value;
        }
        let count = self.game_position_history.iter().filter(|&&h| h == hash).count() as f32;
        if count == 0.0 {
            return value;
        }
        // Shrink value toward 0 by penalty * count
        // e.g. value=0.5, penalty=0.2, count=1 → 0.5 * (1 - 0.2) = 0.4
        // e.g. value=0.5, penalty=0.2, count=2 → 0.5 * (1 - 0.4) = 0.3
        let shrink = (1.0 - self.repetition_penalty * count).max(0.0);
        value * shrink
    }

    /// Apply Dirichlet noise to root node's children priors.
    /// prior = (1 - epsilon) * prior + epsilon * noise[i]
    fn apply_dirichlet_noise(&mut self, root_idx: usize) {
        let n_children = self.nodes[root_idx].children_count as usize;
        if n_children == 0 {
            return;
        }
        let noise = self.rng.dirichlet(self.dirichlet_alpha as f64, n_children);
        let eps = self.dirichlet_epsilon;
        let start = self.nodes[root_idx].children_start as usize;
        for i in 0..n_children {
            let child = &mut self.nodes[start + i];
            child.prior = (1.0 - eps) * child.prior + eps * noise[i];
        }
    }

    /// Get the BitMove of the most-visited root child.
    fn most_visited_bitmove(&self) -> BitMove {
        let root = &self.nodes[0];
        let start = root.children_start as usize;
        let mut best_visits = 0u32;
        let mut best_bm = self.nodes[start].bitmove;
        for i in 0..root.children_count as usize {
            let child = &self.nodes[start + i];
            if child.visit_count > best_visits {
                best_visits = child.visit_count;
                best_bm = child.bitmove;
            }
        }
        best_bm
    }

    /// Determine game outcome: returns (value_target, winner_string).
    fn determine_outcome(&self, bb: &BitBoard) -> (f32, String) {
        match bb.check_winner() {
            Some(Player::White) => (1.0, "white".to_string()),
            Some(Player::Black) => (-1.0, "black".to_string()),
            None => {
                // Draw/adjudicated: use heuristic score as value target
                let score = self.engine.evaluate_heuristic(bb);
                let outcome = (score as f32 / SCORE_SCALING).tanh();
                (outcome, "draw".to_string())
            }
        }
    }

    /// Check if the current position should be adjudicated based on heuristic eval.
    /// Returns true if the heuristic score exceeds the threshold and we're past min moves.
    fn should_adjudicate(&self, bb: &BitBoard, move_count: u32) -> bool {
        if self.adjudication_threshold <= 0.0 || move_count < self.adjudication_min_moves {
            return false;
        }
        let score = self.engine.evaluate_heuristic(bb);
        let scaled = (score as f32 / SCORE_SCALING).tanh().abs();
        scaled >= self.adjudication_threshold
    }

    /// Compute v_mix for a root node (paper equation 6).
    /// v_mix = value_score * (1 - sum_visited_priors) + sum(prior_i * Q_i) for visited children.
    fn compute_v_mix(&self, root_idx: usize) -> f32 {
        let root = &self.nodes[root_idx];
        let n_children = root.children_count as usize;
        let start = root.children_start as usize;

        // Root value from root-player perspective
        let value_score = if root.player_is_white {
            root.q_value()
        } else {
            -root.q_value()
        };

        let mut sum_visited_priors = 0.0f32;
        let mut sum_prior_q = 0.0f32;
        for i in 0..n_children {
            let child = &self.nodes[start + i];
            if child.visit_count > 0 {
                sum_visited_priors += child.prior;
                let q_white = child.q_value();
                let q = if root.player_is_white { q_white } else { -q_white };
                sum_prior_q += child.prior * q;
            }
        }

        value_score * (1.0 - sum_visited_priors) + sum_prior_q
    }

    /// Completed Q-value for a root child, in root-player perspective.
    /// If the child has been visited, returns its Q-value.
    /// If unvisited, returns v_mix (paper §3.1).
    fn completed_q(&self, root_idx: usize, child_idx: usize) -> f32 {
        let root = &self.nodes[root_idx];
        let child = &self.nodes[child_idx];
        if child.visit_count == 0 {
            return self.compute_v_mix(root_idx);
        }
        let q_white = child.q_value();
        if root.player_is_white { q_white } else { -q_white }
    }

    /// Compute adaptive sigma for Gumbel search (paper §3.1).
    /// sigma = (c_visit + max_visit_count) * c_scale
    fn compute_gumbel_sigma(&self, root_idx: usize) -> f32 {
        let root = &self.nodes[root_idx];
        let n_children = root.children_count as usize;
        let start = root.children_start as usize;
        let max_visits = (0..n_children)
            .map(|i| self.nodes[start + i].visit_count)
            .max()
            .unwrap_or(0) as f32;
        (Self::GUMBEL_C_VISIT + max_visits) * Self::GUMBEL_C_SCALE
    }

    /// Normalized completed Q-values for all root children, mapped to [0, 1].
    fn normalized_completed_q_values(&self, root_idx: usize) -> Vec<f32> {
        let root = &self.nodes[root_idx];
        let n_children = root.children_count as usize;
        let start = root.children_start as usize;

        let raw: Vec<f32> = (0..n_children)
            .map(|i| self.completed_q(root_idx, start + i))
            .collect();

        let min_q = raw.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_q = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_q - min_q).max(1e-8);

        raw.iter().map(|&q| (q - min_q) / range).collect()
    }

    /// Build search result for Gumbel AlphaZero.
    /// Policy target = softmax(log_prior + sigma * normalized_completed_Q) over ALL legal actions.
    /// Best move = argmax(gumbel + log_prior + sigma * normalized_completed_Q).
    fn build_gumbel_result(
        &self,
        root_idx: usize,
        gumbels: &[f64],
        log_priors: &[f32],
        sigma: f32,
    ) -> MCTSSearchResult {
        let root = &self.nodes[root_idx];
        let n_children = root.children_count as usize;
        let start = root.children_start as usize;

        // Use normalized Q-values in [0, 1] (paper §3.1)
        let norm_q = self.normalized_completed_q_values(root_idx);

        // Policy target: softmax(log_prior + sigma * normalized_Q) over ALL actions
        let mut policy_target = vec![0.0f32; POLICY_SIZE];
        let mut logits: Vec<f32> = Vec::with_capacity(n_children);
        let mut max_logit = f32::NEG_INFINITY;

        for i in 0..n_children {
            let logit = log_priors[i] + sigma * norm_q[i];
            logits.push(logit);
            if logit > max_logit {
                max_logit = logit;
            }
        }

        let mut exp_sum = 0.0f32;
        let mut exp_vals: Vec<f32> = logits
            .iter()
            .map(|&l| {
                let e = (l - max_logit).exp();
                exp_sum += e;
                e
            })
            .collect();

        if exp_sum > 0.0 {
            for e in &mut exp_vals {
                *e /= exp_sum;
            }
        }

        for (i, &e) in exp_vals.iter().enumerate() {
            let idx = policy_index(&self.nodes[start + i].bitmove) as usize;
            if idx < POLICY_SIZE {
                policy_target[idx] = e;
            }
        }

        // Best move: argmax(gumbel + log_prior + sigma * normalized_Q)
        let mut best_score = f64::NEG_INFINITY;
        let mut best_child_idx = start;
        for i in 0..n_children {
            let score = gumbels[i] + (log_priors[i] + sigma * norm_q[i]) as f64;
            if score > best_score {
                best_score = score;
                best_child_idx = start + i;
            }
        }

        let total_visits: u32 = (0..n_children)
            .map(|i| self.nodes[start + i].visit_count)
            .sum();

        MCTSSearchResult {
            best_move: Some(self.nodes[best_child_idx].bitmove.to_move()),
            visits: total_visits,
            policy_target,
            root_value: root.q_value(),
        }
    }

    /// Compute blended value target for a training example.
    /// Blends game outcome with per-position search score:
    ///   value = lambda * game_outcome + (1 - lambda) * search_score
    fn blend_value_target(&self, game_outcome: f32, search_score: f32) -> f32 {
        let l = self.value_blend_lambda;
        (l * game_outcome + (1.0 - l) * search_score).clamp(-1.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // ONNX inference methods (pure Rust, no Python/GIL)
    // -----------------------------------------------------------------------

    /// Batched MCTS search using ONNX inference (no Python).
    /// Same algorithm as search_batched_impl but calls OnnxSession instead of Python.
    fn search_batched_onnx_impl(
        &mut self,
        bb: &BitBoard,
        onnx: &mut OnnxSession,
        batch_size: u32,
        add_noise: bool,
    ) -> MCTSSearchResult {
        let is_white = bb.current_player == Player::White;

        // Reuse tree if available, otherwise initialize fresh
        let reused = !self.nodes.is_empty()
            && self.nodes[0].children_count > 0
            && self.nodes.len() < 500_000;
        if !reused {
            self.nodes.clear();
            self.nodes.push(MCTSNode::root(is_white));

            // Expand root with single eval
            let root_planes = bb_to_planes(bb);
            let root_moves = self.gen_moves(&bb);
            if root_moves.is_empty() {
                return MCTSSearchResult::empty();
            }

            let root_child_is_white = if !root_moves.is_empty() && root_moves[0].is_pail_placement() {
                bb.current_player == Player::White
            } else {
                bb.current_player != Player::White
            };
            match onnx.infer_batch(&[root_planes]) {
                Ok((policies, values)) if !policies.is_empty() => {
                    let raw_logits = &policies[0];
                    let policy_logits = if !is_white { Self::vflip_policy(raw_logits) } else { raw_logits.clone() };
                    let priors = Self::masked_softmax(&policy_logits, &root_moves);
                    self.expand_node(0, &root_moves, &priors, root_child_is_white);
                    let value_white = if is_white { values[0] } else { -values[0] };
                    self.nodes[0].visit_count += 1;
                    self.nodes[0].total_value += value_white;
                }
                _ => {
                    let v = (self.engine.evaluate_heuristic(bb) as f32 / SCORE_SCALING).tanh();
                    let uniform = 1.0 / root_moves.len() as f32;
                    let priors = vec![uniform; root_moves.len()];
                    self.expand_node(0, &root_moves, &priors, root_child_is_white);
                    self.nodes[0].visit_count += 1;
                    self.nodes[0].total_value += v;
                }
            }
        }

        if self.nodes[0].children_count == 0 {
            return MCTSSearchResult::empty();
        }

        if add_noise {
            self.apply_dirichlet_noise(0);
        }

        if self.nodes[0].children_count == 1 {
            let child_idx = self.nodes[0].children_start as usize;
            self.nodes[child_idx].visit_count = 1;
            return self.build_result(0);
        }

        // Run batched simulations
        let mut remaining = self.simulations;
        while remaining > 0 {
            let n = std::cmp::min(batch_size, remaining) as usize;

            struct PendingLeaf {
                node_idx: usize,
                bb: BitBoard,
                moves: Vec<BitMove>,
            }
            let mut pending: Vec<PendingLeaf> = Vec::new();
            let mut terminal_backprops: Vec<(usize, f32)> = Vec::new();

            for _ in 0..n {
                let mut node_idx: usize = 0;
                let mut sim_bb = *bb;

                while self.nodes[node_idx].is_expanded()
                    && !self.nodes[node_idx].is_terminal
                {
                    node_idx = self.select_child(node_idx);
                    sim_bb.make_move(&self.nodes[node_idx].bitmove);
                }

                if self.nodes[node_idx].is_terminal {
                    terminal_backprops.push((node_idx, self.nodes[node_idx].terminal_value));
                    continue;
                }

                if let Some(winner) = sim_bb.check_winner() {
                    self.nodes[node_idx].is_terminal = true;
                    let v = if winner == Player::White { 1.0 } else { -1.0 };
                    self.nodes[node_idx].terminal_value = v;
                    terminal_backprops.push((node_idx, v));
                    continue;
                }

                let moves = self.gen_moves(&sim_bb);
                if moves.is_empty() {
                    self.nodes[node_idx].is_terminal = true;
                    self.nodes[node_idx].terminal_value = 0.0;
                    terminal_backprops.push((node_idx, 0.0));
                    continue;
                }

                // Check network cache
                let hash = sim_bb.hash;
                if let Some((cached_logits, cached_value)) = self.network_cache.get(&hash).cloned() {
                    let priors = Self::masked_softmax(&cached_logits, &moves);
                    let child_is_white = if !moves.is_empty() && moves[0].is_pail_placement() {
                        sim_bb.current_player == Player::White
                    } else {
                        sim_bb.current_player != Player::White
                    };
                    self.expand_node(node_idx, &moves, &priors, child_is_white);
                    let penalized_cv = self.apply_repetition_penalty(cached_value, hash);
                    self.backpropagate(node_idx, penalized_cv);
                    continue;
                }

                // Virtual loss
                let mut vl_idx = node_idx;
                loop {
                    self.nodes[vl_idx].visit_count += Self::VIRTUAL_LOSS as u32;
                    self.nodes[vl_idx].total_value -= Self::VIRTUAL_LOSS;
                    if self.nodes[vl_idx].parent == u32::MAX {
                        break;
                    }
                    vl_idx = self.nodes[vl_idx].parent as usize;
                }

                pending.push(PendingLeaf {
                    node_idx,
                    bb: sim_bb,
                    moves,
                });
            }

            // Batch evaluate via ONNX
            if !pending.is_empty() {
                let batch_planes: Vec<Vec<f32>> = pending
                    .iter()
                    .map(|p| bb_to_planes(&p.bb))
                    .collect();

                let batch_results = onnx.infer_batch(&batch_planes);

                for (i, leaf) in pending.iter().enumerate() {
                    // Remove virtual loss
                    let mut vl_idx = leaf.node_idx;
                    loop {
                        self.nodes[vl_idx].visit_count -= Self::VIRTUAL_LOSS as u32;
                        self.nodes[vl_idx].total_value += Self::VIRTUAL_LOSS;
                        if self.nodes[vl_idx].parent == u32::MAX {
                            break;
                        }
                        vl_idx = self.nodes[vl_idx].parent as usize;
                    }

                    let leaf_is_white = leaf.bb.current_player == Player::White;
                    let (value, priors) = match &batch_results {
                        Ok((policies, values)) => {
                            let value_white = if leaf_is_white { values[i] } else { -values[i] };
                            let policy_logits = if !leaf_is_white {
                                Self::vflip_policy(&policies[i])
                            } else {
                                policies[i].clone()
                            };
                            let hash = leaf.bb.hash;
                            if self.network_cache.len() < 100_000 {
                                self.network_cache.insert(hash, (policy_logits.clone(), value_white));
                            } else {
                                self.network_cache.clear();
                                self.network_cache.insert(hash, (policy_logits.clone(), value_white));
                            }
                            let priors = Self::masked_softmax(&policy_logits, &leaf.moves);
                            (value_white, priors)
                        }
                        Err(_) => {
                            let v = (self.engine.evaluate_heuristic(&leaf.bb) as f32
                                / SCORE_SCALING)
                                .tanh();
                            let uniform = 1.0 / leaf.moves.len() as f32;
                            (v, vec![uniform; leaf.moves.len()])
                        }
                    };

                    let child_is_white = if !leaf.moves.is_empty() && leaf.moves[0].is_pail_placement() {
                        leaf.bb.current_player == Player::White
                    } else {
                        leaf.bb.current_player != Player::White
                    };
                    self.expand_node(leaf.node_idx, &leaf.moves, &priors, child_is_white);
                    let value = self.apply_repetition_penalty(value, leaf.bb.hash);
                    self.backpropagate(leaf.node_idx, value);
                }
            }

            // Backprop terminals
            for (node_idx, value) in terminal_backprops {
                self.backpropagate(node_idx, value);
            }

            remaining = remaining.saturating_sub(batch_size);
        }

        self.build_result(0)
    }

    // -----------------------------------------------------------------------
    // Gumbel AlphaZero search (Sequential Halving at root, PUCT below)
    // -----------------------------------------------------------------------

    /// Gumbel AlphaZero adaptive sigma constants (paper §3.1, Appendix D).
    /// sigma(root) = (c_visit + max_visit_count) * c_scale
    /// DeepMind's reference impl (mctx) uses maxvisit_init=50, value_scale=0.1.
    /// Paper text says c_scale=1.0 but that's before min-max Q normalization to [0,1];
    /// the mctx code applies value_scale=0.1 AFTER normalization.
    /// With 400 sims: sigma = (50 + 400) * 0.1 = 45 → entropy ~0.64, top-1 ~0.71.
    const GUMBEL_C_VISIT: f32 = 50.0;
    const GUMBEL_C_SCALE: f32 = 0.1;

    /// Gumbel AlphaZero search with Python NN batched evaluation.
    /// Uses Sequential Halving at root with Gumbel noise instead of PUCT + Dirichlet.
    /// Policy targets are softmax(log_prior + sigma * completed_Q) instead of visit counts.
    fn search_gumbel_batched_py_impl(
        &mut self,
        bb: &BitBoard,
        py: Python<'_>,
        eval_fn: &PyObject,
        batch_size: u32,
    ) -> MCTSSearchResult {
        let is_white = bb.current_player == Player::White;

        // Always fresh tree for Gumbel search (no reuse)
        self.nodes.clear();
        self.nodes.push(MCTSNode::root(is_white));

        // Expand root
        let root_planes = bb_to_planes(bb);
        let root_moves = self.gen_moves(&bb);
        if root_moves.is_empty() {
            return MCTSSearchResult::empty();
        }

        let root_child_is_white = if !root_moves.is_empty() && root_moves[0].is_pail_placement() {
            bb.current_player == Player::White
        } else {
            bb.current_player != Player::White
        };

        // Evaluate root via network
        let log_priors = match self.call_network_single(py, eval_fn, &root_planes) {
            Ok((raw_logits, value)) => {
                let policy_logits = if !is_white { Self::vflip_policy(&raw_logits) } else { raw_logits };
                let priors = Self::masked_softmax(&policy_logits, &root_moves);
                let log_priors = Self::masked_log_softmax(&policy_logits, &root_moves);
                self.expand_node(0, &root_moves, &priors, root_child_is_white);
                // No Dirichlet noise: Gumbel noise replaces it (paper §3)
                let value_white = if is_white { value } else { -value };
                self.nodes[0].visit_count += 1;
                self.nodes[0].total_value += value_white;
                log_priors
            }
            Err(_) => {
                let v = (self.engine.evaluate_heuristic(bb) as f32 / SCORE_SCALING).tanh();
                let n = root_moves.len();
                let uniform = 1.0 / n as f32;
                let priors = vec![uniform; n];
                self.expand_node(0, &root_moves, &priors, root_child_is_white);
                self.nodes[0].visit_count += 1;
                self.nodes[0].total_value += v;
                vec![(uniform as f64).ln() as f32; n]
            }
        };

        let n_children = self.nodes[0].children_count as usize;
        let start = self.nodes[0].children_start as usize;

        if n_children == 0 {
            return MCTSSearchResult::empty();
        }
        if n_children == 1 {
            self.nodes[start].visit_count = 1;
            return self.build_result(0);
        }

        // Sample Gumbel(0,1) noise for each action
        let gumbels: Vec<f64> = (0..n_children).map(|_| self.rng.gumbel()).collect();

        // Sequential Halving: select top-m, then halve in phases
        let m = n_children; // use all actions for small branching factor
        let mut candidates: Vec<usize> = (0..n_children).collect();
        // Sort descending by gumbel + log_prior
        candidates.sort_by(|&a, &b| {
            let sa = gumbels[a] + log_priors[a] as f64;
            let sb = gumbels[b] + log_priors[b] as f64;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(m);

        let n_phases = ((candidates.len() as f64).log2().ceil() as u32).max(1);
        let sims_per_phase = self.simulations / n_phases;

        for phase in 0..n_phases {
            let n_remaining = candidates.len();
            if n_remaining <= 1 {
                break;
            }

            let sims_per_action = (sims_per_phase / n_remaining as u32).max(1);

            // Run sims for each candidate, forcing root action
            for &cand_i in &candidates.clone() {
                let forced_child = start + cand_i;
                let mut sims_done = 0u32;

                while sims_done < sims_per_action {
                    let batch_n = std::cmp::min(batch_size, sims_per_action - sims_done) as usize;

                    struct PendingLeaf {
                        node_idx: usize,
                        bb: BitBoard,
                        moves: Vec<BitMove>,
                    }
                    let mut pending: Vec<PendingLeaf> = Vec::new();
                    let mut terminal_backprops: Vec<(usize, f32)> = Vec::new();

                    for _ in 0..batch_n {
                        // Force root -> forced_child
                        let mut sim_bb = *bb;
                        sim_bb.make_move(&self.nodes[forced_child].bitmove);
                        let mut node_idx = forced_child;

                        // PUCT below forced child
                        while self.nodes[node_idx].is_expanded() && !self.nodes[node_idx].is_terminal {
                            node_idx = self.select_child(node_idx);
                            sim_bb.make_move(&self.nodes[node_idx].bitmove);
                        }

                        if self.nodes[node_idx].is_terminal {
                            terminal_backprops.push((node_idx, self.nodes[node_idx].terminal_value));
                            continue;
                        }

                        if let Some(winner) = sim_bb.check_winner() {
                            self.nodes[node_idx].is_terminal = true;
                            let v = if winner == Player::White { 1.0 } else { -1.0 };
                            self.nodes[node_idx].terminal_value = v;
                            terminal_backprops.push((node_idx, v));
                            continue;
                        }

                        let moves = self.gen_moves(&sim_bb);
                        if moves.is_empty() {
                            self.nodes[node_idx].is_terminal = true;
                            self.nodes[node_idx].terminal_value = 0.0;
                            terminal_backprops.push((node_idx, 0.0));
                            continue;
                        }

                        // Check cache
                        let hash = sim_bb.hash;
                        if let Some((cached_logits, cached_value)) = self.network_cache.get(&hash).cloned() {
                            let priors = Self::masked_softmax(&cached_logits, &moves);
                            let child_is_white = if !moves.is_empty() && moves[0].is_pail_placement() {
                                sim_bb.current_player == Player::White
                            } else {
                                sim_bb.current_player != Player::White
                            };
                            self.expand_node(node_idx, &moves, &priors, child_is_white);
                            self.backpropagate(node_idx, cached_value);
                            continue;
                        }

                        // Virtual loss
                        let mut vl_idx = node_idx;
                        loop {
                            self.nodes[vl_idx].visit_count += Self::VIRTUAL_LOSS as u32;
                            self.nodes[vl_idx].total_value -= Self::VIRTUAL_LOSS;
                            if self.nodes[vl_idx].parent == u32::MAX { break; }
                            vl_idx = self.nodes[vl_idx].parent as usize;
                        }

                        pending.push(PendingLeaf { node_idx, bb: sim_bb, moves });
                    }

                    // Batch evaluate
                    if !pending.is_empty() {
                        let batch_planes: Vec<Vec<f32>> = pending.iter()
                            .map(|p| bb_to_planes(&p.bb)).collect();
                        let batch_results = self.call_network_batch(py, eval_fn, &batch_planes);

                        for (i, leaf) in pending.iter().enumerate() {
                            // Remove virtual loss
                            let mut vl_idx = leaf.node_idx;
                            loop {
                                self.nodes[vl_idx].visit_count -= Self::VIRTUAL_LOSS as u32;
                                self.nodes[vl_idx].total_value += Self::VIRTUAL_LOSS;
                                if self.nodes[vl_idx].parent == u32::MAX { break; }
                                vl_idx = self.nodes[vl_idx].parent as usize;
                            }

                            let leaf_is_white = leaf.bb.current_player == Player::White;
                            let (value, priors) = match &batch_results {
                                Ok((policies, values)) => {
                                    let value_white = if leaf_is_white { values[i] } else { -values[i] };
                                    let policy_logits = if !leaf_is_white {
                                        Self::vflip_policy(&policies[i])
                                    } else {
                                        policies[i].clone()
                                    };
                                    let hash = leaf.bb.hash;
                                    if self.network_cache.len() < 100_000 {
                                        self.network_cache.insert(hash, (policy_logits.clone(), value_white));
                                    } else {
                                        self.network_cache.clear();
                                        self.network_cache.insert(hash, (policy_logits.clone(), value_white));
                                    }
                                    (value_white, Self::masked_softmax(&policy_logits, &leaf.moves))
                                }
                                Err(_) => {
                                    let v = (self.engine.evaluate_heuristic(&leaf.bb) as f32 / SCORE_SCALING).tanh();
                                    (v, vec![1.0 / leaf.moves.len() as f32; leaf.moves.len()])
                                }
                            };

                            let child_is_white = if !leaf.moves.is_empty() && leaf.moves[0].is_pail_placement() {
                                leaf.bb.current_player == Player::White
                            } else {
                                leaf.bb.current_player != Player::White
                            };
                            self.expand_node(leaf.node_idx, &leaf.moves, &priors, child_is_white);
                            let value = self.apply_repetition_penalty(value, leaf.bb.hash);
                            self.backpropagate(leaf.node_idx, value);
                        }
                    }

                    for (node_idx, value) in &terminal_backprops {
                        self.backpropagate(*node_idx, *value);
                    }

                    sims_done += batch_n as u32;
                }
            }

            // Halve: keep top half by gumbel + log_prior + sigma * normalized_Q
            if phase < n_phases - 1 && candidates.len() > 1 {
                let sigma = self.compute_gumbel_sigma(0);
                let norm_q = self.normalized_completed_q_values(0);
                candidates.sort_by(|&a, &b| {
                    let sa = gumbels[a] + (log_priors[a] + sigma * norm_q[a]) as f64;
                    let sb = gumbels[b] + (log_priors[b] + sigma * norm_q[b]) as f64;
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                let half = (candidates.len() + 1) / 2;
                candidates.truncate(half);
            }
        }

        let sigma = self.compute_gumbel_sigma(0);
        self.build_gumbel_result(0, &gumbels, &log_priors, sigma)
    }

    /// Gumbel AlphaZero search with ONNX batched evaluation (pure Rust, no Python).
    fn search_gumbel_batched_onnx_impl(
        &mut self,
        bb: &BitBoard,
        onnx: &mut OnnxSession,
        batch_size: u32,
    ) -> MCTSSearchResult {
        let is_white = bb.current_player == Player::White;

        // Always fresh tree
        self.nodes.clear();
        self.nodes.push(MCTSNode::root(is_white));

        let root_planes = bb_to_planes(bb);
        let root_moves = self.gen_moves(&bb);
        if root_moves.is_empty() {
            return MCTSSearchResult::empty();
        }

        let root_child_is_white = if !root_moves.is_empty() && root_moves[0].is_pail_placement() {
            bb.current_player == Player::White
        } else {
            bb.current_player != Player::White
        };

        let log_priors = match onnx.infer_batch(&[root_planes]) {
            Ok((policies, values)) if !policies.is_empty() => {
                let raw_logits = &policies[0];
                let policy_logits = if !is_white { Self::vflip_policy(raw_logits) } else { raw_logits.clone() };
                let priors = Self::masked_softmax(&policy_logits, &root_moves);
                let log_priors = Self::masked_log_softmax(&policy_logits, &root_moves);
                self.expand_node(0, &root_moves, &priors, root_child_is_white);
                // No Dirichlet noise: Gumbel noise replaces it (paper §3)
                let value_white = if is_white { values[0] } else { -values[0] };
                self.nodes[0].visit_count += 1;
                self.nodes[0].total_value += value_white;
                log_priors
            }
            _ => {
                let v = (self.engine.evaluate_heuristic(bb) as f32 / SCORE_SCALING).tanh();
                let n = root_moves.len();
                let uniform = 1.0 / n as f32;
                let priors = vec![uniform; n];
                self.expand_node(0, &root_moves, &priors, root_child_is_white);
                self.nodes[0].visit_count += 1;
                self.nodes[0].total_value += v;
                vec![(uniform as f64).ln() as f32; n]
            }
        };

        let n_children = self.nodes[0].children_count as usize;
        let start = self.nodes[0].children_start as usize;

        if n_children == 0 {
            return MCTSSearchResult::empty();
        }
        if n_children == 1 {
            self.nodes[start].visit_count = 1;
            return self.build_result(0);
        }

        let gumbels: Vec<f64> = (0..n_children).map(|_| self.rng.gumbel()).collect();

        let m = n_children;
        let mut candidates: Vec<usize> = (0..n_children).collect();
        candidates.sort_by(|&a, &b| {
            let sa = gumbels[a] + log_priors[a] as f64;
            let sb = gumbels[b] + log_priors[b] as f64;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(m);

        let n_phases = ((candidates.len() as f64).log2().ceil() as u32).max(1);
        let sims_per_phase = self.simulations / n_phases;

        for phase in 0..n_phases {
            let n_remaining = candidates.len();
            if n_remaining <= 1 {
                break;
            }

            let sims_per_action = (sims_per_phase / n_remaining as u32).max(1);

            for &cand_i in &candidates.clone() {
                let forced_child = start + cand_i;
                let mut sims_done = 0u32;

                while sims_done < sims_per_action {
                    let batch_n = std::cmp::min(batch_size, sims_per_action - sims_done) as usize;

                    struct PendingLeaf {
                        node_idx: usize,
                        bb: BitBoard,
                        moves: Vec<BitMove>,
                    }
                    let mut pending: Vec<PendingLeaf> = Vec::new();
                    let mut terminal_backprops: Vec<(usize, f32)> = Vec::new();

                    for _ in 0..batch_n {
                        let mut sim_bb = *bb;
                        sim_bb.make_move(&self.nodes[forced_child].bitmove);
                        let mut node_idx = forced_child;

                        while self.nodes[node_idx].is_expanded() && !self.nodes[node_idx].is_terminal {
                            node_idx = self.select_child(node_idx);
                            sim_bb.make_move(&self.nodes[node_idx].bitmove);
                        }

                        if self.nodes[node_idx].is_terminal {
                            terminal_backprops.push((node_idx, self.nodes[node_idx].terminal_value));
                            continue;
                        }

                        if let Some(winner) = sim_bb.check_winner() {
                            self.nodes[node_idx].is_terminal = true;
                            let v = if winner == Player::White { 1.0 } else { -1.0 };
                            self.nodes[node_idx].terminal_value = v;
                            terminal_backprops.push((node_idx, v));
                            continue;
                        }

                        let moves = self.gen_moves(&sim_bb);
                        if moves.is_empty() {
                            self.nodes[node_idx].is_terminal = true;
                            self.nodes[node_idx].terminal_value = 0.0;
                            terminal_backprops.push((node_idx, 0.0));
                            continue;
                        }

                        let hash = sim_bb.hash;
                        if let Some((cached_logits, cached_value)) = self.network_cache.get(&hash).cloned() {
                            let priors = Self::masked_softmax(&cached_logits, &moves);
                            let child_is_white = if !moves.is_empty() && moves[0].is_pail_placement() {
                                sim_bb.current_player == Player::White
                            } else {
                                sim_bb.current_player != Player::White
                            };
                            self.expand_node(node_idx, &moves, &priors, child_is_white);
                            self.backpropagate(node_idx, cached_value);
                            continue;
                        }

                        let mut vl_idx = node_idx;
                        loop {
                            self.nodes[vl_idx].visit_count += Self::VIRTUAL_LOSS as u32;
                            self.nodes[vl_idx].total_value -= Self::VIRTUAL_LOSS;
                            if self.nodes[vl_idx].parent == u32::MAX { break; }
                            vl_idx = self.nodes[vl_idx].parent as usize;
                        }

                        pending.push(PendingLeaf { node_idx, bb: sim_bb, moves });
                    }

                    if !pending.is_empty() {
                        let batch_planes: Vec<Vec<f32>> = pending.iter()
                            .map(|p| bb_to_planes(&p.bb)).collect();
                        let batch_results = onnx.infer_batch(&batch_planes);

                        for (i, leaf) in pending.iter().enumerate() {
                            let mut vl_idx = leaf.node_idx;
                            loop {
                                self.nodes[vl_idx].visit_count -= Self::VIRTUAL_LOSS as u32;
                                self.nodes[vl_idx].total_value += Self::VIRTUAL_LOSS;
                                if self.nodes[vl_idx].parent == u32::MAX { break; }
                                vl_idx = self.nodes[vl_idx].parent as usize;
                            }

                            let leaf_is_white = leaf.bb.current_player == Player::White;
                            let (value, priors) = match &batch_results {
                                Ok((policies, values)) => {
                                    let value_white = if leaf_is_white { values[i] } else { -values[i] };
                                    let policy_logits = if !leaf_is_white {
                                        Self::vflip_policy(&policies[i])
                                    } else {
                                        policies[i].clone()
                                    };
                                    let hash = leaf.bb.hash;
                                    if self.network_cache.len() < 100_000 {
                                        self.network_cache.insert(hash, (policy_logits.clone(), value_white));
                                    } else {
                                        self.network_cache.clear();
                                        self.network_cache.insert(hash, (policy_logits.clone(), value_white));
                                    }
                                    (value_white, Self::masked_softmax(&policy_logits, &leaf.moves))
                                }
                                Err(_) => {
                                    let v = (self.engine.evaluate_heuristic(&leaf.bb) as f32 / SCORE_SCALING).tanh();
                                    (v, vec![1.0 / leaf.moves.len() as f32; leaf.moves.len()])
                                }
                            };

                            let child_is_white = if !leaf.moves.is_empty() && leaf.moves[0].is_pail_placement() {
                                leaf.bb.current_player == Player::White
                            } else {
                                leaf.bb.current_player != Player::White
                            };
                            self.expand_node(leaf.node_idx, &leaf.moves, &priors, child_is_white);
                            let value = self.apply_repetition_penalty(value, leaf.bb.hash);
                            self.backpropagate(leaf.node_idx, value);
                        }
                    }

                    for (node_idx, value) in terminal_backprops {
                        self.backpropagate(node_idx, value);
                    }

                    sims_done += batch_n as u32;
                }
            }

            // Halve
            if phase < n_phases - 1 && candidates.len() > 1 {
                let sigma = self.compute_gumbel_sigma(0);
                let norm_q = self.normalized_completed_q_values(0);
                candidates.sort_by(|&a, &b| {
                    let sa = gumbels[a] + (log_priors[a] + sigma * norm_q[a]) as f64;
                    let sb = gumbels[b] + (log_priors[b] + sigma * norm_q[b]) as f64;
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                let half = (candidates.len() + 1) / 2;
                candidates.truncate(half);
            }
        }

        let sigma = self.compute_gumbel_sigma(0);
        self.build_gumbel_result(0, &gumbels, &log_priors, sigma)
    }

    /// Play one self-play game using ONNX inference (pure Rust).
    fn play_network_game_onnx_impl(
        &mut self,
        onnx: &mut OnnxSession,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        temperature: f32,
        full_search_fraction: f32,
        cheap_sims: u32,
    ) -> Result<SelfPlayResult, String> {
        self.clear_caches();
        let mut bb = BitBoard::new();
        let mut examples: Vec<(Vec<f32>, Vec<f32>, bool, f32)> = Vec::new();
        let original_sims = self.simulations;
        let mut position_history: Vec<u64> = Vec::new();

        // Random opening moves
        let mut random_turns = 0;
        while random_turns < random_opening {
            if bb.check_winner().is_some() { break; }
            let moves = self.gen_moves(&bb);
            if moves.is_empty() { break; }
            let idx = self.rng.usize(moves.len());
            let is_barrel_move = !moves[idx].is_pail_placement();
            bb.make_move(&moves[idx]);
            if is_barrel_move { random_turns += 1; }
        }

        let mut move_count = 0u32;
        loop {
            if bb.check_winner().is_some() || (move_count as usize) >= max_moves {
                break;
            }

            // Adjudicate decisive positions early
            if self.should_adjudicate(&bb, move_count) {
                break;
            }

            // 3-fold repetition detection
            if move_count >= 15 && Self::is_threefold(&position_history, bb.hash) {
                break;
            }
            position_history.push(bb.hash);
            self.game_position_history.clone_from(&position_history);

            // Pail placement: skip MCTS, use center-biased random selection
            if let Some(pail_move) = self.random_center_pail(&bb) {
                bb.make_move(&pail_move);
                continue;
            }

            // Playout cap randomization
            let is_full_search = full_search_fraction >= 1.0
                || self.rng.f64() < full_search_fraction as f64;
            self.simulations = if is_full_search { original_sims } else { cheap_sims };

            let result = if self.use_gumbel {
                self.search_gumbel_batched_onnx_impl(&bb, onnx, batch_size)
            } else {
                self.search_batched_onnx_impl(&bb, onnx, batch_size, true)
            };
            if result.best_move.is_none() { break; }

            // Only collect training data from full-search moves
            if is_full_search {
                let is_white = bb.current_player == Player::White;
                let planes = bb_to_planes(&bb);
                let search_score = if is_white { result.root_value } else { -result.root_value };
                examples.push((planes, result.policy_target.clone(), is_white, search_score));
            }

            // Move selection
            let chosen_bitmove = if self.use_gumbel {
                if (move_count as usize) < temp_moves && temperature > 0.0 {
                    // Temperature sampling from visit counts (not improved policy)
                    self.sample_bitmove_by_temp(0, temperature)
                        .unwrap_or_else(|| self.most_visited_bitmove())
                } else {
                    let best_move = result.best_move.as_ref().unwrap();
                    let best_pi = Self::move_policy_index(best_move);
                    let root = &self.nodes[0];
                    let s = root.children_start as usize;
                    let mut bm = self.nodes[s].bitmove;
                    for i in 0..root.children_count as usize {
                        if policy_index(&self.nodes[s + i].bitmove) == best_pi {
                            bm = self.nodes[s + i].bitmove;
                            break;
                        }
                    }
                    bm
                }
            } else if (move_count as usize) < temp_moves && temperature > 0.0 {
                self.sample_bitmove_by_temp(0, temperature)
                    .unwrap_or_else(|| self.most_visited_bitmove())
            } else {
                self.most_visited_bitmove()
            };

            let is_barrel_move = !chosen_bitmove.is_pail_placement();
            if !self.use_gumbel {
                self.retain_subtree(&chosen_bitmove);
            }
            bb.make_move(&chosen_bitmove);
            if is_barrel_move { move_count += 1; }
        }

        self.simulations = original_sims;
        self.nodes.clear();

        let (outcome_white, winner_str) = self.determine_outcome(&bb);

        let training_examples: Vec<TrainingExample> = examples
            .into_iter()
            .map(|(planes, policy, is_white, search_score)| {
                let game_outcome = if is_white { outcome_white } else { -outcome_white };
                let value_target = self.blend_value_target(game_outcome, search_score);
                let policy_target = if is_white { policy } else { Self::vflip_policy(&policy) };
                TrainingExample {
                    planes,
                    policy_target,
                    value_target,
                    search_score,
                }
            })
            .collect();

        Ok(SelfPlayResult {
            examples: training_examples,
            winner: winner_str,
            move_count,
        })
    }

    /// Play N evaluation games using ONNX: MCTS vs alpha-beta.
    fn play_mixed_game_onnx_impl(
        &mut self,
        onnx: &mut OnnxSession,
        opponent_depth: u8,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        temperature: f32,
        mcts_is_white: bool,
    ) -> Result<SelfPlayResult, String> {
        self.clear_caches();
        let mut bb = BitBoard::new();
        let mut examples: Vec<(Vec<f32>, Vec<f32>, bool, f32)> = Vec::new();
        let mut position_history: Vec<u64> = Vec::new();

        // Random opening moves
        let mut random_turns = 0;
        while random_turns < random_opening {
            if bb.check_winner().is_some() { break; }
            let moves = self.gen_moves(&bb);
            if moves.is_empty() { break; }
            let idx = self.rng.usize(moves.len());
            let is_barrel_move = !moves[idx].is_pail_placement();
            bb.make_move(&moves[idx]);
            if is_barrel_move { random_turns += 1; }
        }

        let mut move_count = 0u32;
        loop {
            if bb.check_winner().is_some() || (move_count as usize) >= max_moves {
                break;
            }

            // Adjudicate decisive positions early
            if self.should_adjudicate(&bb, move_count) {
                break;
            }

            // 3-fold repetition detection
            if move_count >= 15 && Self::is_threefold(&position_history, bb.hash) {
                break;
            }
            position_history.push(bb.hash);
            self.game_position_history.clone_from(&position_history);

            // Pail placement: center-biased random for both sides
            if let Some(pail_move) = self.random_center_pail(&bb) {
                bb.make_move(&pail_move);
                continue;
            }

            let is_white_turn = bb.current_player == Player::White;
            let is_mcts_turn = is_white_turn == mcts_is_white;

            let chosen_bitmove = if is_mcts_turn {
                // Network MCTS turn — collect training data
                let result = if self.use_gumbel {
                    self.search_gumbel_batched_onnx_impl(&bb, onnx, batch_size)
                } else {
                    self.search_batched_onnx_impl(&bb, onnx, batch_size, true)
                };
                if result.best_move.is_none() { break; }

                // Collect training example
                let is_white = bb.current_player == Player::White;
                let planes = bb_to_planes(&bb);
                let search_score = if is_white { result.root_value } else { -result.root_value };
                examples.push((planes, result.policy_target.clone(), is_white, search_score));

                // Move selection
                if self.use_gumbel {
                    if (move_count as usize) < temp_moves && temperature > 0.0 {
                        self.sample_bitmove_by_temp(0, temperature)
                            .unwrap_or_else(|| self.most_visited_bitmove())
                    } else {
                        let best_move = result.best_move.as_ref().unwrap();
                        let best_pi = Self::move_policy_index(best_move);
                        let root = &self.nodes[0];
                        let s = root.children_start as usize;
                        let mut bm = self.nodes[s].bitmove;
                        for i in 0..root.children_count as usize {
                            if policy_index(&self.nodes[s + i].bitmove) == best_pi {
                                bm = self.nodes[s + i].bitmove;
                                break;
                            }
                        }
                        bm
                    }
                } else if (move_count as usize) < temp_moves && temperature > 0.0 {
                    self.sample_bitmove_by_temp(0, temperature)
                        .unwrap_or_else(|| self.most_visited_bitmove())
                } else {
                    self.most_visited_bitmove()
                }
            } else {
                // Heuristic turn
                self.engine.full_reset();
                let (_, bm_opt) = self.engine.search(&bb, opponent_depth);
                match bm_opt {
                    Some(bm) => bm,
                    None => break,
                }
            };

            let is_barrel_move = !chosen_bitmove.is_pail_placement();
            if is_mcts_turn && !self.use_gumbel {
                self.retain_subtree(&chosen_bitmove);
            }
            bb.make_move(&chosen_bitmove);
            if is_barrel_move { move_count += 1; }
        }

        self.nodes.clear();

        let (outcome_white, winner_str) = self.determine_outcome(&bb);

        let training_examples: Vec<TrainingExample> = examples
            .into_iter()
            .map(|(planes, policy, is_white, search_score)| {
                let game_outcome = if is_white { outcome_white } else { -outcome_white };
                let value_target = self.blend_value_target(game_outcome, search_score);
                let policy_target = if is_white { policy } else { Self::vflip_policy(&policy) };
                TrainingExample {
                    planes,
                    policy_target,
                    value_target,
                    search_score,
                }
            })
            .collect();

        Ok(SelfPlayResult {
            examples: training_examples,
            winner: winner_str,
            move_count,
        })
    }

    fn play_eval_match_onnx_impl(
        &mut self,
        onnx: &mut OnnxSession,
        num_games: usize,
        opponent_depth: u8,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
    ) -> Result<EvalMatchResult, String> {
        let mut wins = 0u32;
        let mut draws = 0u32;
        let mut losses = 0u32;

        for game_idx in 0..num_games {
            let mcts_is_white = game_idx % 2 == 0;

            let mut bb = BitBoard::new();
            self.engine.full_reset();
            self.clear_caches();

            // Random opening
            let mut random_turns = 0;
            while random_turns < random_opening {
                if bb.check_winner().is_some() { break; }
                let moves = self.gen_moves(&bb);
                if moves.is_empty() { break; }
                let idx = self.rng.usize(moves.len());
                let is_barrel_move = !moves[idx].is_pail_placement();
                bb.make_move(&moves[idx]);
                if is_barrel_move { random_turns += 1; }
            }

            let mut move_count = 0u32;
            let mut position_history: Vec<u64> = Vec::new();
            loop {
                if bb.check_winner().is_some() || (move_count as usize) >= max_moves {
                    break;
                }

                // 3-fold repetition detection
                if move_count >= 15 && Self::is_threefold(&position_history, bb.hash) {
                    break;
                }
                position_history.push(bb.hash);
                self.game_position_history.clone_from(&position_history);

                // Pail placement: center-biased random for both sides
                if let Some(pail_move) = self.random_center_pail(&bb) {
                    bb.make_move(&pail_move);
                    continue;
                }

                let is_white_turn = bb.current_player == Player::White;
                let is_mcts_turn = is_white_turn == mcts_is_white;

                let chosen_bitmove = if is_mcts_turn {
                    if self.use_gumbel {
                        let result = self.search_gumbel_batched_onnx_impl(&bb, onnx, batch_size);
                        match result.best_move {
                            Some(mv) => {
                                let best_pi = Self::move_policy_index(&mv);
                                let root = &self.nodes[0];
                                let s = root.children_start as usize;
                                let mut bm = self.nodes[s].bitmove;
                                for i in 0..root.children_count as usize {
                                    if policy_index(&self.nodes[s + i].bitmove) == best_pi {
                                        bm = self.nodes[s + i].bitmove;
                                        break;
                                    }
                                }
                                bm
                            }
                            None => break,
                        }
                    } else {
                        let result = self.search_batched_onnx_impl(&bb, onnx, batch_size, false);
                        match result.best_move {
                            Some(_) => self.most_visited_bitmove(),
                            None => break,
                        }
                    }
                } else {
                    self.engine.full_reset();
                    let (_, bm_opt) = self.engine.search(&bb, opponent_depth);
                    match bm_opt {
                        Some(bm) => bm,
                        None => break,
                    }
                };

                let is_barrel_move = !chosen_bitmove.is_pail_placement();
                self.retain_subtree(&chosen_bitmove);
                bb.make_move(&chosen_bitmove);
                if is_barrel_move { move_count += 1; }
            }

            self.nodes.clear();

            match bb.check_winner() {
                None => draws += 1,
                Some(winner) => {
                    let white_won = winner == Player::White;
                    if white_won == mcts_is_white {
                        wins += 1;
                    } else {
                        losses += 1;
                    }
                }
            }
        }

        Ok(EvalMatchResult { wins, draws, losses })
    }
}
