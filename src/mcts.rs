//! Rust MCTS for Tonnesjakk.
//!
//! Arena-based MCTS with PUCT selection, supporting two evaluation modes:
//! - Heuristic: entirely in Rust, no Python calls (very fast)
//! - Network: calls a Python function for leaf evaluation (policy + value)
//!
//! Also provides full game loops (self-play and evaluation matches) that
//! run entirely in Rust, returning training data to Python.

use pyo3::prelude::*;

use crate::{BitBoard, BitBoardEngine, BitMove, Board, Move, Player, NUM_SQUARES};

/// Policy output size: 37 from-indices (0-35 board + 36 off-board) x 36 to-indices
pub const POLICY_SIZE: usize = 37 * 36; // 1332

const SCORE_SCALING: f32 = 600.0;

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

/// Convert a BitBoard position to 6x6x6 float planes.
/// Planes: [white_barrels, black_barrels, white_pail, black_pail, current_player, ones]
pub fn bb_to_planes(bb: &BitBoard) -> Vec<f32> {
    let mut planes = vec![0.0f32; 6 * NUM_SQUARES]; // 216
    for sq in 0..NUM_SQUARES {
        let mask = 1u64 << sq;
        if bb.white_barrels & mask != 0 {
            planes[sq] = 1.0;
        }
        if bb.black_barrels & mask != 0 {
            planes[NUM_SQUARES + sq] = 1.0;
        }
        if bb.white_pail & mask != 0 {
            planes[2 * NUM_SQUARES + sq] = 1.0;
        }
        if bb.black_pail & mask != 0 {
            planes[3 * NUM_SQUARES + sq] = 1.0;
        }
    }
    if bb.current_player == Player::White {
        for i in 4 * NUM_SQUARES..5 * NUM_SQUARES {
            planes[i] = 1.0;
        }
    }
    // Bias plane (always 1)
    for i in 5 * NUM_SQUARES..6 * NUM_SQUARES {
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
    pub planes: Vec<f32>,        // 216 floats (6x6x6)
    #[pyo3(get)]
    pub policy_target: Vec<f32>, // 1332 floats
    #[pyo3(get)]
    pub value_target: f32,       // outcome in [-1, +1]
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
}

// ---------------------------------------------------------------------------
// MCTS Engine (exposed to Python)
// ---------------------------------------------------------------------------

#[pyclass]
pub struct MCTSEngine {
    simulations: u32,
    c_puct: f32,
    nodes: Vec<MCTSNode>,
    engine: BitBoardEngine, // For heuristic evaluation
    rng: Rng,
}

#[pymethods]
impl MCTSEngine {
    #[new]
    #[pyo3(signature = (simulations=200, c_puct=1.4))]
    fn new(simulations: u32, c_puct: f32) -> Self {
        MCTSEngine {
            simulations,
            c_puct,
            nodes: Vec::with_capacity(8192),
            engine: BitBoardEngine::new(),
            rng: Rng::new(),
        }
    }

    /// Pure-Rust MCTS with heuristic leaf evaluation. No Python calls.
    fn search_heuristic(&mut self, board: &Board) -> MCTSSearchResult {
        let bb = BitBoard::from_board(board);
        self.search_impl(&bb, None)
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
        Ok(self.search_impl(&bb, Some((py, &eval_fn))))
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
        Ok(self.search_batched_impl(&bb, py, &eval_fn, batch_size))
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
    #[pyo3(signature = (random_opening=6, max_moves=80, temp_moves=10))]
    fn play_heuristic_game(
        &mut self,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
    ) -> SelfPlayResult {
        self.play_heuristic_game_impl(random_opening, max_moves, temp_moves)
    }

    /// Play N heuristic self-play games. Returns a list of SelfPlayResult.
    #[pyo3(signature = (n_games, random_opening=6, max_moves=80, temp_moves=10))]
    fn play_heuristic_games(
        &mut self,
        n_games: usize,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
    ) -> Vec<SelfPlayResult> {
        (0..n_games)
            .map(|_| self.play_heuristic_game_impl(random_opening, max_moves, temp_moves))
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
        let mut result = self.search_impl(&bb, None);
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
    #[pyo3(signature = (eval_fn, batch_size=8, random_opening=4, max_moves=80, temp_moves=15, temperature=1.0))]
    fn play_network_game(
        &mut self,
        py: Python<'_>,
        eval_fn: PyObject,
        batch_size: u32,
        random_opening: usize,
        max_moves: usize,
        temp_moves: usize,
        temperature: f32,
    ) -> PyResult<SelfPlayResult> {
        self.play_network_game_impl(
            py, &eval_fn, batch_size, random_opening, max_moves, temp_moves, temperature,
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
}

// ---------------------------------------------------------------------------
// Core MCTS implementation
// ---------------------------------------------------------------------------

impl MCTSEngine {
    fn search_impl(
        &mut self,
        bb: &BitBoard,
        eval_fn: Option<(Python<'_>, &PyObject)>,
    ) -> MCTSSearchResult {
        let is_white = bb.current_player == Player::White;

        // Initialize tree
        self.nodes.clear();
        self.nodes.push(MCTSNode::root(is_white));

        // Expand root
        let _root_value = self.evaluate_and_expand(0, bb, &eval_fn);

        if self.nodes[0].children_count == 0 {
            return MCTSSearchResult::empty();
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

    /// PUCT child selection.
    fn select_child(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        let sqrt_parent = (node.visit_count as f32).sqrt();

        let mut best_idx = node.children_start as usize;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..node.children_count as usize {
            let child_idx = node.children_start as usize + i;
            let child = &self.nodes[child_idx];

            let q = child.q_value();
            // Negate Q when parent is Black (Black minimizes White's score)
            let q_adjusted = if node.player_is_white { q } else { -q };

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

        let moves = bb.generate_moves();
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
                // Heuristic evaluation
                let v = (self.engine.evaluate_heuristic(bb) as f32 / SCORE_SCALING).tanh();
                let uniform = 1.0 / moves.len() as f32;
                (v, vec![uniform; moves.len()])
            }
        };

        // Expand: create children
        let child_is_white = !self.nodes[node_idx].player_is_white;
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

        value
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
        let (policy_logits, value): (Vec<f32>, f32) = result.extract(py)?;

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
    ) -> MCTSSearchResult {
        let is_white = bb.current_player == Player::White;

        // Initialize tree
        self.nodes.clear();
        self.nodes.push(MCTSNode::root(is_white));

        // Expand root with single eval
        let root_planes = bb_to_planes(bb);
        let root_moves = bb.generate_moves();
        if root_moves.is_empty() {
            return MCTSSearchResult::empty();
        }

        // Evaluate root
        match self.call_network_single(py, eval_fn, &root_planes) {
            Ok((policy_logits, value)) => {
                let priors = Self::masked_softmax(&policy_logits, &root_moves);
                self.expand_node(0, &root_moves, &priors, is_white);
                // Backprop root value
                self.nodes[0].visit_count += 1;
                self.nodes[0].total_value += value;
            }
            Err(_) => {
                let v = (self.engine.evaluate_heuristic(bb) as f32 / SCORE_SCALING).tanh();
                let uniform = 1.0 / root_moves.len() as f32;
                let priors = vec![uniform; root_moves.len()];
                self.expand_node(0, &root_moves, &priors, is_white);
                self.nodes[0].visit_count += 1;
                self.nodes[0].total_value += v;
            }
        }

        if self.nodes[0].children_count == 0 {
            return MCTSSearchResult::empty();
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

                let moves = sim_bb.generate_moves();
                if moves.is_empty() {
                    self.nodes[node_idx].is_terminal = true;
                    self.nodes[node_idx].terminal_value = 0.0;
                    terminal_backprops.push((node_idx, 0.0));
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
                    let (value, priors) = match &batch_results {
                        Ok((policies, values)) => {
                            let priors = Self::masked_softmax(&policies[i], &leaf.moves);
                            (values[i], priors)
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
                    let child_is_white = !self.nodes[leaf.node_idx].player_is_white;
                    self.expand_node(leaf.node_idx, &leaf.moves, &priors, child_is_white);

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
    ) -> SelfPlayResult {
        let mut bb = BitBoard::new();
        let mut examples: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

        // Random opening moves
        for _ in 0..random_opening {
            if bb.check_winner().is_some() {
                break;
            }
            let moves = bb.generate_moves();
            if moves.is_empty() {
                break;
            }
            let idx = self.rng.usize(moves.len());
            bb.make_move(&moves[idx]);
        }

        let mut move_count = 0u32;
        while bb.check_winner().is_none() && (move_count as usize) < max_moves {
            let result = self.search_impl(&bb, None);
            if result.best_move.is_none() {
                break;
            }

            // Collect training data
            let planes = bb_to_planes(&bb);
            examples.push((planes, result.policy_target.clone()));

            // Temperature-based selection for first moves
            let chosen_bitmove = if (move_count as usize) < temp_moves {
                let moves = bb.generate_moves();
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

            bb.make_move(&chosen_bitmove);
            move_count += 1;
        }

        // Determine outcome
        let (outcome, winner_str) = self.determine_outcome(&bb);

        // Fill in outcomes
        let training_examples: Vec<TrainingExample> = examples
            .into_iter()
            .map(|(planes, policy)| TrainingExample {
                planes,
                policy_target: policy,
                value_target: outcome,
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
    ) -> PyResult<SelfPlayResult> {
        let mut bb = BitBoard::new();
        let mut examples: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

        // Random opening moves
        for _ in 0..random_opening {
            if bb.check_winner().is_some() {
                break;
            }
            let moves = bb.generate_moves();
            if moves.is_empty() {
                break;
            }
            let idx = self.rng.usize(moves.len());
            bb.make_move(&moves[idx]);
        }

        let mut move_count = 0u32;
        while bb.check_winner().is_none() && (move_count as usize) < max_moves {
            let result = self.search_batched_impl(&bb, py, eval_fn, batch_size);
            if result.best_move.is_none() {
                break;
            }

            // Collect training data
            let planes = bb_to_planes(&bb);
            examples.push((planes, result.policy_target.clone()));

            // Temperature-based selection for first moves, then deterministic
            let chosen_bitmove = if (move_count as usize) < temp_moves && temperature > 0.0 {
                // Sample from visit distribution via the tree
                self.sample_bitmove_by_temp(0, temperature)
                    .unwrap_or_else(|| self.most_visited_bitmove())
            } else {
                self.most_visited_bitmove()
            };

            bb.make_move(&chosen_bitmove);
            move_count += 1;
        }

        // Determine outcome
        let (outcome, winner_str) = self.determine_outcome(&bb);

        let training_examples: Vec<TrainingExample> = examples
            .into_iter()
            .map(|(planes, policy)| TrainingExample {
                planes,
                policy_target: policy,
                value_target: outcome,
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

            // Random opening
            for _ in 0..random_opening {
                if bb.check_winner().is_some() {
                    break;
                }
                let moves = bb.generate_moves();
                if moves.is_empty() {
                    break;
                }
                let idx = self.rng.usize(moves.len());
                bb.make_move(&moves[idx]);
            }

            let mut move_count = 0u32;
            while bb.check_winner().is_none() && (move_count as usize) < max_moves {
                let is_white_turn = bb.current_player == Player::White;
                let is_mcts_turn = is_white_turn == mcts_is_white;

                let chosen_bitmove = if is_mcts_turn {
                    // MCTS move (network)
                    let result = self.search_batched_impl(&bb, py, eval_fn, batch_size);
                    match result.best_move {
                        Some(_) => self.most_visited_bitmove(),
                        None => break,
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

                bb.make_move(&chosen_bitmove);
                move_count += 1;
            }

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
                // Draw: use heuristic score as value target
                let score = self.engine.evaluate_heuristic(bb);
                let outcome = (score as f32 / SCORE_SCALING).tanh();
                (outcome, "draw".to_string())
            }
        }
    }
}
