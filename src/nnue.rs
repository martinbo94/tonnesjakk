use serde::Deserialize;
use wide::f32x8;

use crate::board::*;

/// Maximum supported hidden layer sizes (for fixed-size accumulator arrays)
pub const MAX_HIDDEN1: usize = 512;
pub const MAX_HIDDEN2: usize = 128;

// ============================================================================
// HALFPAIL NNUE - Dual-perspective sparse feature evaluation
// ============================================================================

/// HalfPail feature constants
#[allow(dead_code)]
const HALFPAIL_BUCKETS: usize = 37;    // 36 pail squares + 1 for "no pail"
pub const HALFPAIL_PIECE_TYPES: usize = 3; // 0=friendly barrel, 1=enemy barrel, 2=enemy pail
pub const HALFPAIL_FEATURES_PER_BUCKET: usize = NUM_SQUARES * HALFPAIL_PIECE_TYPES; // 108
#[allow(dead_code)]
const HALFPAIL_FEATURES: usize = HALFPAIL_BUCKETS * HALFPAIL_FEATURES_PER_BUCKET; // 3996
pub const HALFPAIL_DENSE: usize = 20;

/// Compute HalfPail feature index
#[inline(always)]
pub const fn halfpail_feature_index(bucket: usize, sq: usize, piece_type: usize) -> u16 {
    (bucket * HALFPAIL_FEATURES_PER_BUCKET + sq * HALFPAIL_PIECE_TYPES + piece_type) as u16
}

/// JSON format for HalfPail NNUE weights
#[derive(Deserialize)]
pub(crate) struct HalfPailWeights {
    fc1_weight: Vec<Vec<f32>>,  // [num_perspective_features][hidden1] - shared embedding
    fc1_bias: Vec<f32>,          // [hidden1]
    fc2_weight: Vec<Vec<f32>>,  // [hidden2][2*hidden1 + dense_size]
    fc2_bias: Vec<f32>,          // [hidden2]
    fc3_weight: Vec<Vec<f32>>,  // [1][hidden2]
    fc3_bias: Vec<f32>,          // [1]
}

#[derive(Deserialize)]
pub(crate) struct HalfPailJson {
    #[allow(dead_code)]
    halfpail: bool,
    hidden1: usize,
    hidden2: usize,
    num_perspective_features: usize,
    #[allow(dead_code)]
    dense_size: usize,
    weights: HalfPailWeights,
}

// ============================================================================
// DENSE FEATURES - 20 relational features used by HalfPail
// ============================================================================

/// Compute 20 relational/dense features from a BitBoard position.
///
/// These are the same features as the legacy NNUE training data
/// (features 144-163 of the 164-feature encoding):
///   [0-3]   White barrel distances to goal (sorted, normalized /5)
///   [4-7]   Black barrel distances to goal (sorted, normalized /5)
///   [8-9]   White/black scored barrels (/4)
///   [10-11] White/black pail placed (0 or 1)
///   [12]    Current player (+1 white, -1 black)
///   [13-14] White/black immediate threats (/4)
///   [15]    Score differential (/4)
///   [16-17] White/black barrels on board (/4)
///   [18-19] White/black pail blocking count (/4)
#[inline]
pub fn compute_relational_features(bb: &BitBoard) -> [f32; HALFPAIL_DENSE] {
    let mut features = [0.0f32; 20];

    // White barrel distances to goal (row 0 is goal, distance = row)
    let mut white_dists = [0.0f32; 4];
    let mut white_rows = [0usize; 4];
    let mut n_white = 0usize;
    let mut white_threats = 0u32;
    let mut barrels = bb.white_barrels;
    while barrels != 0 && n_white < 4 {
        let sq = barrels.trailing_zeros() as usize;
        let row = sq / 6;
        white_dists[n_white] = row as f32 / 5.0;
        white_rows[n_white] = row;
        if row == 1 { white_threats += 1; }
        n_white += 1;
        barrels &= barrels - 1;
    }
    white_dists[..n_white].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n_white {
        features[i] = 1.0 - white_dists[i];
    }

    // Black barrel distances to goal (row 5 is goal, distance = 5 - row)
    let mut black_dists = [0.0f32; 4];
    let mut black_rows = [0usize; 4];
    let mut black_cols = [0usize; 4];
    let mut n_black = 0usize;
    let mut black_threats = 0u32;
    barrels = bb.black_barrels;
    while barrels != 0 && n_black < 4 {
        let sq = barrels.trailing_zeros() as usize;
        let row = sq / 6;
        let col = sq % 6;
        black_dists[n_black] = (5 - row) as f32 / 5.0;
        black_rows[n_black] = row;
        black_cols[n_black] = col;
        if row == 4 { black_threats += 1; }
        n_black += 1;
        barrels &= barrels - 1;
    }
    black_dists[..n_black].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n_black {
        features[4 + i] = 1.0 - black_dists[i];
    }

    // White barrel columns for blocking computation
    let mut white_cols = [0usize; 4];
    {
        let mut i = 0usize;
        let mut b = bb.white_barrels;
        while b != 0 && i < 4 {
            let sq = b.trailing_zeros() as usize;
            white_rows[i] = sq / 6;
            white_cols[i] = sq % 6;
            i += 1;
            b &= b - 1;
        }
    }

    // Scored barrels (normalized by 4)
    features[8] = bb.white_scored as f32 / 4.0;
    features[9] = bb.black_scored as f32 / 4.0;

    // Pails placed
    features[10] = if bb.white_pail != 0 { 1.0 } else { 0.0 };
    features[11] = if bb.black_pail != 0 { 1.0 } else { 0.0 };

    // Current player
    features[12] = match bb.current_player {
        Player::White => 1.0,
        Player::Black => -1.0,
    };

    // Immediate threats (barrels 1 step from scoring)
    features[13] = white_threats as f32 / 4.0;
    features[14] = black_threats as f32 / 4.0;

    // Score differential
    features[15] = (bb.white_scored as f32 - bb.black_scored as f32) / 4.0;

    // Barrels on board
    features[16] = n_white as f32 / 4.0;
    features[17] = n_black as f32 / 4.0;

    // Pail blocking counts
    let mut white_pail_blocks = 0u32;
    if bb.white_pail != 0 {
        let pail_sq = bb.white_pail.trailing_zeros() as usize;
        let pail_row = pail_sq / 6;
        let pail_col = pail_sq % 6;
        for i in 0..n_black {
            if pail_col == black_cols[i] && pail_row > black_rows[i] {
                white_pail_blocks += 1;
            }
        }
    }
    features[18] = white_pail_blocks as f32 / 4.0;

    let mut black_pail_blocks = 0u32;
    if bb.black_pail != 0 {
        let pail_sq = bb.black_pail.trailing_zeros() as usize;
        let pail_row = pail_sq / 6;
        let pail_col = pail_sq % 6;
        for i in 0..n_white {
            if pail_col == white_cols[i] && pail_row < white_rows[i] {
                black_pail_blocks += 1;
            }
        }
    }
    features[19] = black_pail_blocks as f32 / 4.0;

    features
}

// ============================================================================
// DUAL ACCUMULATOR - Stack-based caching for search tree
// ============================================================================

/// Represents a change in HalfPail features for one perspective
#[derive(Clone, Copy, Debug)]
pub struct HalfPailDelta {
    pub index: u16,   // 0-3995: which perspective feature
    pub delta: i8,    // +1 (added) or -1 (removed)
}

/// Dual-perspective accumulator for HalfPail NNUE
#[derive(Clone)]
pub struct DualAccumulator {
    pub white_pre: [f32; MAX_HIDDEN1],  // White perspective pre-activation
    pub black_pre: [f32; MAX_HIDDEN1],  // Black perspective pre-activation
}

impl Default for DualAccumulator {
    fn default() -> Self {
        DualAccumulator {
            white_pre: [0.0; MAX_HIDDEN1],
            black_pre: [0.0; MAX_HIDDEN1],
        }
    }
}

impl DualAccumulator {
    #[inline]
    pub fn copy_from(&mut self, other: &DualAccumulator) {
        self.white_pre = other.white_pre;
        self.black_pre = other.black_pre;
    }
}

/// Stack of dual accumulators for the search tree
pub struct DualAccumulatorStack {
    accumulators: Vec<DualAccumulator>,
    depth: usize,
}

impl Default for DualAccumulatorStack {
    fn default() -> Self {
        let mut accs = Vec::with_capacity(MAX_DEPTH);
        for _ in 0..MAX_DEPTH {
            accs.push(DualAccumulator::default());
        }
        DualAccumulatorStack {
            accumulators: accs,
            depth: 0,
        }
    }
}

impl DualAccumulatorStack {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn current(&self) -> &DualAccumulator {
        &self.accumulators[self.depth]
    }

    #[inline]
    pub fn current_mut(&mut self) -> &mut DualAccumulator {
        &mut self.accumulators[self.depth]
    }

    #[inline]
    pub fn push(&mut self) {
        if self.depth + 1 < MAX_DEPTH {
            let (left, right) = self.accumulators.split_at_mut(self.depth + 1);
            right[0].copy_from(&left[self.depth]);
            self.depth += 1;
        }
    }

    #[inline]
    pub fn pop(&mut self) {
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.depth = 0;
    }
}

// ============================================================================
// HALFPAIL NNUE EVALUATOR
// ============================================================================

/// HalfPail NNUE evaluator
///
/// Architecture:
///   White sparse -> EmbeddingBag(3996, H1) + bias -> ReLU -> acc_white
///   Black sparse -> EmbeddingBag(3996, H1) + bias -> ReLU -> acc_black
///                       (shared weights)
///   concat(acc_white, acc_black, dense_20) -> FC2 -> ReLU -> FC3 -> Tanh
pub struct HalfPailNNUE {
    /// FC1 transposed weights: [HALFPAIL_FEATURES * hidden1]
    fc1_weight_t: Vec<f32>,
    /// FC1 bias: [hidden1]
    pub(crate) fc1_bias: Vec<f32>,
    /// FC2 weights: [hidden2 * (2*hidden1 + HALFPAIL_DENSE)]
    fc2_weight: Vec<f32>,
    /// FC2 bias: [hidden2]
    fc2_bias: Vec<f32>,
    /// FC3 weights: [hidden2]
    fc3_weight: Vec<f32>,
    /// FC3 bias: scalar
    fc3_bias: f32,
    pub(crate) hidden1: usize,
    hidden2: usize,
}

impl HalfPailNNUE {
    /// Load from parsed JSON
    pub(crate) fn from_json(json: HalfPailJson) -> Result<Self, Box<dyn std::error::Error>> {
        let hidden1 = json.hidden1;
        let hidden2 = json.hidden2;
        let num_features = json.num_perspective_features;

        // fc1_weight comes as [num_features][hidden1], transpose to [num_features * hidden1]
        let mut fc1_weight_t = vec![0.0f32; num_features * hidden1];
        for (feat, row) in json.weights.fc1_weight.iter().enumerate() {
            for (neuron, &w) in row.iter().enumerate() {
                fc1_weight_t[feat * hidden1 + neuron] = w;
            }
        }

        // FC2: flatten [hidden2][fc2_input_size]
        let fc2_weight: Vec<f32> = json.weights.fc2_weight.into_iter().flatten().collect();
        let fc3_weight: Vec<f32> = json.weights.fc3_weight.into_iter().flatten().collect();

        Ok(Self {
            fc1_weight_t,
            fc1_bias: json.weights.fc1_bias,
            fc2_weight,
            fc2_bias: json.weights.fc2_bias,
            fc3_weight,
            fc3_bias: json.weights.fc3_bias[0],
            hidden1,
            hidden2,
        })
    }

    /// Add one sparse feature to an accumulator (SIMD)
    #[inline]
    pub(crate) fn add_feature(&self, acc: &mut [f32; MAX_HIDDEN1], feat: usize) {
        let hidden1 = self.hidden1;
        let base_idx = feat * hidden1;
        let mut i = 0;

        while i + 8 <= hidden1 {
            let acc_vec = f32x8::new([
                acc[i], acc[i+1], acc[i+2], acc[i+3],
                acc[i+4], acc[i+5], acc[i+6], acc[i+7],
            ]);
            let wt_vec = f32x8::new([
                self.fc1_weight_t[base_idx+i], self.fc1_weight_t[base_idx+i+1],
                self.fc1_weight_t[base_idx+i+2], self.fc1_weight_t[base_idx+i+3],
                self.fc1_weight_t[base_idx+i+4], self.fc1_weight_t[base_idx+i+5],
                self.fc1_weight_t[base_idx+i+6], self.fc1_weight_t[base_idx+i+7],
            ]);
            let result = acc_vec + wt_vec;
            let arr = result.to_array();
            acc[i] = arr[0]; acc[i+1] = arr[1]; acc[i+2] = arr[2]; acc[i+3] = arr[3];
            acc[i+4] = arr[4]; acc[i+5] = arr[5]; acc[i+6] = arr[6]; acc[i+7] = arr[7];
            i += 8;
        }
    }

    /// Remove one sparse feature from an accumulator (SIMD)
    #[inline]
    fn remove_feature(&self, acc: &mut [f32; MAX_HIDDEN1], feat: usize) {
        let hidden1 = self.hidden1;
        let base_idx = feat * hidden1;
        let mut i = 0;

        while i + 8 <= hidden1 {
            let acc_vec = f32x8::new([
                acc[i], acc[i+1], acc[i+2], acc[i+3],
                acc[i+4], acc[i+5], acc[i+6], acc[i+7],
            ]);
            let wt_vec = f32x8::new([
                self.fc1_weight_t[base_idx+i], self.fc1_weight_t[base_idx+i+1],
                self.fc1_weight_t[base_idx+i+2], self.fc1_weight_t[base_idx+i+3],
                self.fc1_weight_t[base_idx+i+4], self.fc1_weight_t[base_idx+i+5],
                self.fc1_weight_t[base_idx+i+6], self.fc1_weight_t[base_idx+i+7],
            ]);
            let result = acc_vec - wt_vec;
            let arr = result.to_array();
            acc[i] = arr[0]; acc[i+1] = arr[1]; acc[i+2] = arr[2]; acc[i+3] = arr[3];
            acc[i+4] = arr[4]; acc[i+5] = arr[5]; acc[i+6] = arr[6]; acc[i+7] = arr[7];
            i += 8;
        }
    }

    /// Initialize both perspectives from scratch for a position
    pub fn init_accumulators(&self, bb: &BitBoard, dual_acc: &mut DualAccumulator) {
        // Reset to bias
        for i in 0..self.hidden1 {
            dual_acc.white_pre[i] = self.fc1_bias[i];
            dual_acc.black_pre[i] = self.fc1_bias[i];
        }

        let w_bucket = if bb.white_pail != 0 {
            bb.white_pail.trailing_zeros() as usize
        } else {
            36 // no pail placed
        };
        let b_bucket = if bb.black_pail != 0 {
            bb.black_pail.trailing_zeros() as usize
        } else {
            36
        };

        // White barrels: friendly for white perspective, enemy for black
        let mut barrels = bb.white_barrels;
        while barrels != 0 {
            let sq = barrels.trailing_zeros() as usize;
            self.add_feature(&mut dual_acc.white_pre,
                halfpail_feature_index(w_bucket, sq, 0) as usize);  // friendly
            self.add_feature(&mut dual_acc.black_pre,
                halfpail_feature_index(b_bucket, sq, 1) as usize);  // enemy
            barrels &= barrels - 1;
        }

        // Black barrels: enemy for white perspective, friendly for black
        barrels = bb.black_barrels;
        while barrels != 0 {
            let sq = barrels.trailing_zeros() as usize;
            self.add_feature(&mut dual_acc.white_pre,
                halfpail_feature_index(w_bucket, sq, 1) as usize);  // enemy
            self.add_feature(&mut dual_acc.black_pre,
                halfpail_feature_index(b_bucket, sq, 0) as usize);  // friendly
            barrels &= barrels - 1;
        }

        // White pail: enemy pail for black perspective (not in white's own perspective)
        if bb.white_pail != 0 {
            let sq = bb.white_pail.trailing_zeros() as usize;
            self.add_feature(&mut dual_acc.black_pre,
                halfpail_feature_index(b_bucket, sq, 2) as usize);  // enemy pail
        }

        // Black pail: enemy pail for white perspective (not in black's own perspective)
        if bb.black_pail != 0 {
            let sq = bb.black_pail.trailing_zeros() as usize;
            self.add_feature(&mut dual_acc.white_pre,
                halfpail_feature_index(w_bucket, sq, 2) as usize);  // enemy pail
        }
    }

    /// Compute feature deltas for a move, returns (white_deltas, black_deltas, white_recompute, black_recompute)
    ///
    /// When a player places their pail, their own perspective needs full recompute
    /// (the bucket changes from 36 to the pail square).
    pub fn compute_move_deltas(
        &self,
        bb: &BitBoard,
        mv: &BitMove,
    ) -> (Vec<HalfPailDelta>, Vec<HalfPailDelta>, bool, bool) {
        let player = bb.current_player;
        let mut white_deltas = Vec::with_capacity(8);
        let mut black_deltas = Vec::with_capacity(8);
        let mut white_recompute = false;
        let mut black_recompute = false;

        // Current buckets
        let w_bucket = if bb.white_pail != 0 {
            bb.white_pail.trailing_zeros() as usize
        } else { 36 };
        let b_bucket = if bb.black_pail != 0 {
            bb.black_pail.trailing_zeros() as usize
        } else { 36 };

        // 1. Handle pail placement
        if let Some(pail_sq) = mv.pail_pos() {
            let pail_sq = pail_sq as usize;
            match player {
                Player::White => {
                    white_recompute = true;
                    black_deltas.push(HalfPailDelta {
                        index: halfpail_feature_index(b_bucket, pail_sq, 2),
                        delta: 1,
                    });
                }
                Player::Black => {
                    black_recompute = true;
                    white_deltas.push(HalfPailDelta {
                        index: halfpail_feature_index(w_bucket, pail_sq, 2),
                        delta: 1,
                    });
                }
            }
        }

        let w_bucket_for_deltas = w_bucket;
        let b_bucket_for_deltas = b_bucket;

        // 2. Handle barrel movement
        if mv.is_placement() {
            let to_sq = mv.barrel_to() as usize;
            let goal_row = bb.goal_row(player);
            let (to_row, _) = sq_to_coords(to_sq);

            if to_row != goal_row {
                match player {
                    Player::White => {
                        if !white_recompute {
                            white_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(w_bucket_for_deltas, to_sq, 0),
                                delta: 1,
                            });
                        }
                        if !black_recompute {
                            black_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(b_bucket_for_deltas, to_sq, 1),
                                delta: 1,
                            });
                        }
                    }
                    Player::Black => {
                        if !white_recompute {
                            white_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(w_bucket_for_deltas, to_sq, 1),
                                delta: 1,
                            });
                        }
                        if !black_recompute {
                            black_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(b_bucket_for_deltas, to_sq, 0),
                                delta: 1,
                            });
                        }
                    }
                }
            }
        } else {
            // Regular barrel move
            let from_sq = mv.barrel_from().unwrap() as usize;
            let to_sq = mv.barrel_to() as usize;
            let goal_row = bb.goal_row(player);
            let (to_row, _) = sq_to_coords(to_sq);

            match player {
                Player::White => {
                    if !white_recompute {
                        white_deltas.push(HalfPailDelta {
                            index: halfpail_feature_index(w_bucket_for_deltas, from_sq, 0),
                            delta: -1,
                        });
                    }
                    if !black_recompute {
                        black_deltas.push(HalfPailDelta {
                            index: halfpail_feature_index(b_bucket_for_deltas, from_sq, 1),
                            delta: -1,
                        });
                    }
                    if to_row != goal_row {
                        if !white_recompute {
                            white_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(w_bucket_for_deltas, to_sq, 0),
                                delta: 1,
                            });
                        }
                        if !black_recompute {
                            black_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(b_bucket_for_deltas, to_sq, 1),
                                delta: 1,
                            });
                        }
                    }
                }
                Player::Black => {
                    if !white_recompute {
                        white_deltas.push(HalfPailDelta {
                            index: halfpail_feature_index(w_bucket_for_deltas, from_sq, 1),
                            delta: -1,
                        });
                    }
                    if !black_recompute {
                        black_deltas.push(HalfPailDelta {
                            index: halfpail_feature_index(b_bucket_for_deltas, from_sq, 0),
                            delta: -1,
                        });
                    }
                    if to_row != goal_row {
                        if !white_recompute {
                            white_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(w_bucket_for_deltas, to_sq, 1),
                                delta: 1,
                            });
                        }
                        if !black_recompute {
                            black_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(b_bucket_for_deltas, to_sq, 0),
                                delta: 1,
                            });
                        }
                    }
                }
            }
        }

        (white_deltas, black_deltas, white_recompute, black_recompute)
    }

    /// Apply deltas to one perspective accumulator
    #[inline]
    pub fn apply_deltas(&self, acc: &mut [f32; MAX_HIDDEN1], deltas: &[HalfPailDelta]) {
        for d in deltas {
            let feat = d.index as usize;
            if d.delta > 0 {
                self.add_feature(acc, feat);
            } else {
                self.remove_feature(acc, feat);
            }
        }
    }

    /// Evaluate from dual accumulator: apply ReLU, concat, FC2, FC3, tanh
    /// Returns centipawn score
    pub fn evaluate_from_dual_acc(&self, bb: &BitBoard, dual_acc: &DualAccumulator) -> i32 {
        let hidden1 = self.hidden1;
        let fc2_input_size = 2 * hidden1 + HALFPAIL_DENSE;

        // Apply ReLU to both perspectives
        let mut white_post = [0.0f32; MAX_HIDDEN1];
        let mut black_post = [0.0f32; MAX_HIDDEN1];
        {
            let zero = f32x8::ZERO;
            let mut i = 0;
            while i + 8 <= hidden1 {
                let w_vec = f32x8::new([
                    dual_acc.white_pre[i], dual_acc.white_pre[i+1],
                    dual_acc.white_pre[i+2], dual_acc.white_pre[i+3],
                    dual_acc.white_pre[i+4], dual_acc.white_pre[i+5],
                    dual_acc.white_pre[i+6], dual_acc.white_pre[i+7],
                ]);
                let b_vec = f32x8::new([
                    dual_acc.black_pre[i], dual_acc.black_pre[i+1],
                    dual_acc.black_pre[i+2], dual_acc.black_pre[i+3],
                    dual_acc.black_pre[i+4], dual_acc.black_pre[i+5],
                    dual_acc.black_pre[i+6], dual_acc.black_pre[i+7],
                ]);
                let w_relu = w_vec.max(zero);
                let b_relu = b_vec.max(zero);
                let w_arr = w_relu.to_array();
                let b_arr = b_relu.to_array();
                for k in 0..8 {
                    white_post[i+k] = w_arr[k];
                    black_post[i+k] = b_arr[k];
                }
                i += 8;
            }
        }

        // Compute dense features
        let dense = compute_relational_features(bb);

        // FC2: dot product over concat(white_post, black_post, dense)
        let mut hidden2 = [0.0f32; MAX_HIDDEN2];
        for neuron in 0..self.hidden2 {
            let wt_base = neuron * fc2_input_size;
            let mut sum = self.fc2_bias[neuron];

            // White perspective part (SIMD)
            let mut sum_vec = f32x8::ZERO;
            let mut j = 0;
            while j + 8 <= hidden1 {
                let input_vec = f32x8::new([
                    white_post[j], white_post[j+1], white_post[j+2], white_post[j+3],
                    white_post[j+4], white_post[j+5], white_post[j+6], white_post[j+7],
                ]);
                let wt_vec = f32x8::new([
                    self.fc2_weight[wt_base+j], self.fc2_weight[wt_base+j+1],
                    self.fc2_weight[wt_base+j+2], self.fc2_weight[wt_base+j+3],
                    self.fc2_weight[wt_base+j+4], self.fc2_weight[wt_base+j+5],
                    self.fc2_weight[wt_base+j+6], self.fc2_weight[wt_base+j+7],
                ]);
                sum_vec = sum_vec + input_vec * wt_vec;
                j += 8;
            }
            let arr = sum_vec.to_array();
            sum += arr[0]+arr[1]+arr[2]+arr[3]+arr[4]+arr[5]+arr[6]+arr[7];

            // Black perspective part (SIMD)
            let black_offset = hidden1;
            sum_vec = f32x8::ZERO;
            j = 0;
            while j + 8 <= hidden1 {
                let input_vec = f32x8::new([
                    black_post[j], black_post[j+1], black_post[j+2], black_post[j+3],
                    black_post[j+4], black_post[j+5], black_post[j+6], black_post[j+7],
                ]);
                let wt_vec = f32x8::new([
                    self.fc2_weight[wt_base+black_offset+j], self.fc2_weight[wt_base+black_offset+j+1],
                    self.fc2_weight[wt_base+black_offset+j+2], self.fc2_weight[wt_base+black_offset+j+3],
                    self.fc2_weight[wt_base+black_offset+j+4], self.fc2_weight[wt_base+black_offset+j+5],
                    self.fc2_weight[wt_base+black_offset+j+6], self.fc2_weight[wt_base+black_offset+j+7],
                ]);
                sum_vec = sum_vec + input_vec * wt_vec;
                j += 8;
            }
            let arr = sum_vec.to_array();
            sum += arr[0]+arr[1]+arr[2]+arr[3]+arr[4]+arr[5]+arr[6]+arr[7];

            // Dense part (20 values, scalar loop)
            let dense_offset = 2 * hidden1;
            for k in 0..HALFPAIL_DENSE {
                sum += dense[k] * self.fc2_weight[wt_base + dense_offset + k];
            }

            hidden2[neuron] = sum.max(0.0);  // ReLU
        }

        // FC3: dot product -> tanh -> centipawns
        let mut output = self.fc3_bias;
        let mut sum_vec = f32x8::ZERO;
        let mut i = 0;
        while i + 8 <= self.hidden2 {
            let input_vec = f32x8::new([
                hidden2[i], hidden2[i+1], hidden2[i+2], hidden2[i+3],
                hidden2[i+4], hidden2[i+5], hidden2[i+6], hidden2[i+7],
            ]);
            let wt_vec = f32x8::new([
                self.fc3_weight[i], self.fc3_weight[i+1],
                self.fc3_weight[i+2], self.fc3_weight[i+3],
                self.fc3_weight[i+4], self.fc3_weight[i+5],
                self.fc3_weight[i+6], self.fc3_weight[i+7],
            ]);
            sum_vec = sum_vec + input_vec * wt_vec;
            i += 8;
        }
        let arr = sum_vec.to_array();
        output += arr[0]+arr[1]+arr[2]+arr[3]+arr[4]+arr[5]+arr[6]+arr[7];

        (output.tanh() * 1000.0) as i32
    }
}

// ============================================================================
// EVALUATION CACHE
// ============================================================================

/// Eval cache size (power of 2 for fast modulo)
pub(crate) const EVAL_CACHE_SIZE: usize = 1 << 16; // 65536 entries

#[derive(Clone, Copy, Default)]
pub(crate) struct EvalCacheEntry {
    hash: u64,
    score: i32,
    generation: u8,
}

pub(crate) struct EvalCache {
    entries: Vec<EvalCacheEntry>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    generation: u8,
}

impl EvalCache {
    pub(crate) fn new() -> Self {
        EvalCache {
            entries: vec![EvalCacheEntry::default(); EVAL_CACHE_SIZE],
            hits: 0,
            misses: 0,
            generation: 0,
        }
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash as usize) & (EVAL_CACHE_SIZE - 1)
    }

    #[inline]
    pub(crate) fn probe(&mut self, hash: u64) -> Option<i32> {
        let idx = self.index(hash);
        let entry = &self.entries[idx];
        if entry.hash == hash && entry.generation == self.generation {
            self.hits += 1;
            Some(entry.score)
        } else {
            self.misses += 1;
            None
        }
    }

    #[inline]
    pub(crate) fn store(&mut self, hash: u64, score: i32) {
        let idx = self.index(hash);
        self.entries[idx] = EvalCacheEntry { hash, score, generation: self.generation };
    }

    /// Clear cache using generation counter (O(1))
    pub(crate) fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.hits = 0;
        self.misses = 0;
    }

    pub(crate) fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}
