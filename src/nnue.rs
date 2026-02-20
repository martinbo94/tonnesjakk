use serde::Deserialize;
use wide::{f32x8, i16x16, i32x8};

use crate::board::*;

// NNUE feature sizes
/// Base features: 6x6 board * 4 piece types = 144
pub const BASE_FEATURES: usize = NUM_SQUARES * 4;
/// Relational features (20 total)
pub const RELATIONAL_FEATURES: usize = 20;
/// Total input features to NNUE
pub const INPUT_SIZE: usize = BASE_FEATURES + RELATIONAL_FEATURES; // 164

/// NNUE vekter lastet fra JSON
#[derive(Deserialize)]
pub(crate) struct NNUEWeights {
    pub(crate) fc1_weight: Vec<Vec<f32>>,  // [hidden1][144]
    pub(crate) fc1_bias: Vec<f32>,          // [hidden1]
    pub(crate) fc2_weight: Vec<Vec<f32>>,  // [hidden2][hidden1]
    pub(crate) fc2_bias: Vec<f32>,          // [hidden2]
    pub(crate) fc3_weight: Vec<Vec<f32>>,  // [1][hidden2]
    pub(crate) fc3_bias: Vec<f32>,          // [1]
}

#[derive(Deserialize)]
pub(crate) struct NNUEModel {
    pub(crate) hidden1: usize,
    pub(crate) hidden2: usize,
    pub(crate) weights: NNUEWeights,
}

/// JSON format for quantized NNUE weights (int16/int32)
#[derive(Deserialize)]
pub(crate) struct QuantizedNNUEWeights {
    fc1_weight: Vec<Vec<i16>>,   // [hidden1][input_size]
    fc1_bias: Vec<i16>,          // [hidden1]
    fc2_weight: Vec<Vec<i16>>,   // [hidden2][hidden1]
    fc2_bias: Vec<i32>,          // [hidden2]
    fc3_weight: Vec<i16>,        // [hidden2]
    fc3_bias: i32,               // scalar
}

#[derive(Deserialize)]
pub(crate) struct QuantizedNNUEJson {
    hidden1: usize,
    hidden2: usize,
    input_size: usize,
    fc1_weight_scale: i32,
    fc2_weight_scale: i32,
    #[allow(dead_code)]
    fc3_weight_scale: i32,
    crelu_shift: u32,
    output_scale: f64,
    weights: QuantizedNNUEWeights,
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
// INKREMENTELL NNUE - Rask evaluering med delta-oppdateringer
// ============================================================================

/// Input feature indeks: sq * 4 + piece_type
/// piece_type: 0=WhiteBarrel, 1=BlackBarrel, 2=WhitePail, 3=BlackPail
#[inline]
pub const fn feature_index(sq: usize, piece_type: usize) -> usize {
    sq * 4 + piece_type
}

/// Representerer en endring i input-features
#[derive(Clone, Copy, Debug)]
pub struct FeatureDelta {
    pub index: u8,   // 0-143: hvilken input-feature
    pub delta: i8,   // +1 (brikke lagt til) eller -1 (fjernet)
}

/// Maximum supported hidden layer sizes (for fixed-size accumulator arrays)
pub const MAX_HIDDEN1: usize = 512;
pub const MAX_HIDDEN2: usize = 128;

/// Accumulator for inkrementell NNUE
/// Cacher layer 1 output for å unngå full reberegning
#[derive(Clone)]
pub struct Accumulator {
    /// Pre-activation verdier (før ReLU)
    pub pre_activation: [f32; MAX_HIDDEN1],
    /// Post-activation verdier (etter ReLU) - brukes av layer 2
    pub post_activation: [f32; MAX_HIDDEN1],
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            pre_activation: [0.0; MAX_HIDDEN1],
            post_activation: [0.0; MAX_HIDDEN1],
        }
    }
}

impl Accumulator {
    /// Kopier fra en annen accumulator (only pre_activation - post is recomputed)
    #[inline]
    pub fn copy_from(&mut self, other: &Accumulator) {
        // Only copy pre_activation - post_activation is recomputed in apply_relu()
        self.pre_activation = other.pre_activation;
        // Skip post_activation copy - saves 256 bytes per push!
    }

    /// Anvend ReLU på pre_activation og lagre i post_activation (SIMD-accelerert)
    #[inline]
    pub fn apply_relu(&mut self) {
        let zero = f32x8::ZERO;
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= 64 {
            let pre_vec = f32x8::new([
                self.pre_activation[i],
                self.pre_activation[i + 1],
                self.pre_activation[i + 2],
                self.pre_activation[i + 3],
                self.pre_activation[i + 4],
                self.pre_activation[i + 5],
                self.pre_activation[i + 6],
                self.pre_activation[i + 7],
            ]);
            let result = pre_vec.max(zero);
            let arr = result.to_array();
            self.post_activation[i] = arr[0];
            self.post_activation[i + 1] = arr[1];
            self.post_activation[i + 2] = arr[2];
            self.post_activation[i + 3] = arr[3];
            self.post_activation[i + 4] = arr[4];
            self.post_activation[i + 5] = arr[5];
            self.post_activation[i + 6] = arr[6];
            self.post_activation[i + 7] = arr[7];
            i += 8;
        }
    }
}

/// Stack med accumulators for søketreet
/// Bruker fast størrelse for å unngå heap-allokeringer i hot path
pub struct AccumulatorStack {
    accumulators: [Accumulator; MAX_DEPTH],
    depth: usize,
}

impl Default for AccumulatorStack {
    fn default() -> Self {
        AccumulatorStack {
            accumulators: std::array::from_fn(|_| Accumulator::default()),
            depth: 0,
        }
    }
}

impl AccumulatorStack {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hent current accumulator
    #[inline]
    pub fn current(&self) -> &Accumulator {
        &self.accumulators[self.depth]
    }

    /// Hent mutable current accumulator
    #[inline]
    pub fn current_mut(&mut self) -> &mut Accumulator {
        &mut self.accumulators[self.depth]
    }

    /// Push: kopier current til neste nivå og øk dybde
    #[inline]
    pub fn push(&mut self) {
        if self.depth + 1 < MAX_DEPTH {
            let (left, right) = self.accumulators.split_at_mut(self.depth + 1);
            right[0].copy_from(&left[self.depth]);
            self.depth += 1;
        }
    }

    /// Pop: gå tilbake til forrige nivå
    #[inline]
    pub fn pop(&mut self) {
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    /// Reset til root
    #[inline]
    pub fn reset(&mut self) {
        self.depth = 0;
    }

    /// Nåværende dybde
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }
}

// ============================================================================
// QUANTIZED NNUE - Int16 accumulator for ~2-4x faster evaluation
// ============================================================================

/// Quantized accumulator: i16 pre-activation values at scale fc1_weight_scale
#[derive(Clone)]
pub struct QAccumulator {
    pub values: [i16; MAX_HIDDEN1],
}

impl Default for QAccumulator {
    fn default() -> Self {
        QAccumulator { values: [0i16; MAX_HIDDEN1] }
    }
}

impl QAccumulator {
    #[inline]
    pub fn copy_from(&mut self, other: &QAccumulator) {
        self.values = other.values;
    }
}

/// Stack of quantized accumulators for the search tree
pub struct QAccumulatorStack {
    accumulators: [QAccumulator; MAX_DEPTH],
    depth: usize,
}

impl Default for QAccumulatorStack {
    fn default() -> Self {
        QAccumulatorStack {
            accumulators: std::array::from_fn(|_| QAccumulator::default()),
            depth: 0,
        }
    }
}

impl QAccumulatorStack {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn current(&self) -> &QAccumulator {
        &self.accumulators[self.depth]
    }

    #[inline]
    pub fn current_mut(&mut self) -> &mut QAccumulator {
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

/// Inkrementell NNUE evaluator
/// Cacher layer 1 og oppdaterer kun endrede features
///
/// Supports two architectures:
///   - Legacy (144 features): Base position features only
///   - Enhanced (157 features): Base + 13 relational features (distances, scored, pails, player)
pub struct IncrementalNNUE {
    // Vekter (delt med standard NNUE)
    #[allow(dead_code)]
    pub(crate) fc1_weight: Vec<f32>,  // [hidden1 * input_size] where input_size is 144 or 157
    pub(crate) fc1_weight_t: Vec<f32>,  // Transposed: [input_size * hidden1] for cache-friendly feature updates
    pub(crate) fc1_bias: Vec<f32>,
    pub(crate) fc2_weight: Vec<f32>,
    pub(crate) fc2_bias: Vec<f32>,
    pub(crate) fc3_weight: Vec<f32>,
    pub(crate) fc3_bias: f32,
    pub(crate) hidden1: usize,
    pub(crate) hidden2: usize,
    pub(crate) input_size: usize,  // 144 (legacy) or 157 (with relational features)
}

/// Compute NNUE feature deltas for a move (free function, used by both f32 and quantized paths)
pub fn compute_nnue_move_deltas(bb: &BitBoard, mv: &BitMove) -> Vec<FeatureDelta> {
    let mut deltas = Vec::with_capacity(4);
    let player = bb.current_player;

    // 1. Pail plassering
    if let Some(pail_sq) = mv.pail_pos() {
        let piece_type = match player {
            Player::White => 2,
            Player::Black => 3,
        };
        deltas.push(FeatureDelta {
            index: feature_index(pail_sq as usize, piece_type) as u8,
            delta: 1,
        });
    }

    // 2. Tønne-bevegelse
    let barrel_piece = match player {
        Player::White => 0,
        Player::Black => 1,
    };

    if mv.is_placement() {
        let to_sq = mv.barrel_to() as usize;
        let goal_row = bb.goal_row(player);
        let (to_row, _) = sq_to_coords(to_sq);
        if to_row != goal_row {
            deltas.push(FeatureDelta {
                index: feature_index(to_sq, barrel_piece) as u8,
                delta: 1,
            });
        }
    } else {
        let from_sq = mv.barrel_from().unwrap() as usize;
        let to_sq = mv.barrel_to() as usize;
        let goal_row = bb.goal_row(player);
        let (to_row, _) = sq_to_coords(to_sq);

        deltas.push(FeatureDelta {
            index: feature_index(from_sq, barrel_piece) as u8,
            delta: -1,
        });

        if to_row != goal_row {
            deltas.push(FeatureDelta {
                index: feature_index(to_sq, barrel_piece) as u8,
                delta: 1,
            });
        }
    }

    deltas
}

impl IncrementalNNUE {
    /// Last modell fra JSON
    pub fn load(json_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json_str = std::fs::read_to_string(json_path)?;
        let model: NNUEModel = serde_json::from_str(&json_str)?;

        let fc1_weight: Vec<f32> = model.weights.fc1_weight.into_iter().flatten().collect();
        let fc2_weight: Vec<f32> = model.weights.fc2_weight.into_iter().flatten().collect();
        let fc3_weight: Vec<f32> = model.weights.fc3_weight.into_iter().flatten().collect();

        // Detect input size from weight dimensions
        // fc1_weight has shape [hidden1, input_size], flattened to [hidden1 * input_size]
        let input_size = fc1_weight.len() / model.hidden1;
        let hidden1 = model.hidden1;

        // Create transposed weight matrix for cache-friendly feature updates
        // Original: fc1_weight[neuron * input_size + feature]
        // Transposed: fc1_weight_t[feature * hidden1 + neuron]
        let mut fc1_weight_t = vec![0.0f32; input_size * hidden1];
        for neuron in 0..hidden1 {
            for feature in 0..input_size {
                fc1_weight_t[feature * hidden1 + neuron] = fc1_weight[neuron * input_size + feature];
            }
        }

        Ok(Self {
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
        })
    }

    /// Compute 13 relational features from a BitBoard position.
    ///
    /// Features (13 total):
    ///   [0-3]  White barrel distances to goal (normalized 0-1, closest first)
    ///   [4-7]  Black barrel distances to goal (normalized 0-1, closest first)
    ///   [8]    White barrels scored (normalized 0-1)
    ///   [9]    Black barrels scored (normalized 0-1)
    ///   [10]   White pail placed (0 or 1)
    ///   [11]   Black pail placed (0 or 1)
    ///   [12]   Current player (+1 white, -1 black)
    #[inline]
    pub fn compute_relational_features(bb: &BitBoard) -> [f32; 20] {
        let mut features = [0.0f32; 20];

        // White barrel distances to goal (row 0 is goal, distance = row)
        // Sort ascending by distance, then feature = 1.0 - normalized_dist
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
            if row == 1 { white_threats += 1; } // 1 step from row 0 goal
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
            if row == 4 { black_threats += 1; } // 1 step from row 5 goal
            n_black += 1;
            barrels &= barrels - 1;
        }
        black_dists[..n_black].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..n_black {
            features[4 + i] = 1.0 - black_dists[i];
        }

        // We also need white barrel columns for blocking computation
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

    /// Add relational features to accumulator using SIMD-optimized transposed weights.
    #[inline]
    pub(crate) fn add_relational_features(&self, bb: &BitBoard, acc: &mut Accumulator) {
        if self.input_size <= BASE_FEATURES {
            return; // Legacy model without relational features
        }

        let hidden1 = self.hidden1;
        let base = BASE_FEATURES;
        let rel_features = Self::compute_relational_features(bb);

        // Add contribution of relational features to accumulator using transposed weights
        for (feat_idx, &feat_val) in rel_features.iter().enumerate() {
            if feat_val == 0.0 {
                continue; // Skip zero features
            }
            let weight_idx = base + feat_idx;
            let base_idx = weight_idx * hidden1;  // Contiguous in transposed layout

            // SIMD loop: process 8 elements at a time
            let scale_vec = f32x8::splat(feat_val);
            let mut i = 0;
            while i + 8 <= hidden1 {
                let acc_vec = f32x8::new([
                    acc.pre_activation[i],
                    acc.pre_activation[i + 1],
                    acc.pre_activation[i + 2],
                    acc.pre_activation[i + 3],
                    acc.pre_activation[i + 4],
                    acc.pre_activation[i + 5],
                    acc.pre_activation[i + 6],
                    acc.pre_activation[i + 7],
                ]);
                let weight_vec = f32x8::new([
                    self.fc1_weight_t[base_idx + i],
                    self.fc1_weight_t[base_idx + i + 1],
                    self.fc1_weight_t[base_idx + i + 2],
                    self.fc1_weight_t[base_idx + i + 3],
                    self.fc1_weight_t[base_idx + i + 4],
                    self.fc1_weight_t[base_idx + i + 5],
                    self.fc1_weight_t[base_idx + i + 6],
                    self.fc1_weight_t[base_idx + i + 7],
                ]);
                let result = acc_vec + scale_vec * weight_vec;
                let arr = result.to_array();
                acc.pre_activation[i] = arr[0];
                acc.pre_activation[i + 1] = arr[1];
                acc.pre_activation[i + 2] = arr[2];
                acc.pre_activation[i + 3] = arr[3];
                acc.pre_activation[i + 4] = arr[4];
                acc.pre_activation[i + 5] = arr[5];
                acc.pre_activation[i + 6] = arr[6];
                acc.pre_activation[i + 7] = arr[7];
                i += 8;
            }
        }
    }

    /// Full evaluering - brukes for root eller når vi ikke har en accumulator
    pub fn evaluate_full(&self, bb: &BitBoard, acc: &mut Accumulator) -> f32 {
        // Reset accumulator til bias
        for i in 0..self.hidden1 {
            acc.pre_activation[i] = self.fc1_bias[i];
        }

        // Legg til bidrag fra alle aktive base features (piece positions)
        self.add_features_from_bitboard(bb, acc);

        // Legg til relational features (if model supports them)
        self.add_relational_features(bb, acc);

        // Anvend ReLU
        acc.apply_relu();

        // Beregn resten av nettverket
        self.evaluate_from_accumulator(acc)
    }

    /// Legg til alle features fra et bitboard til accumulator
    pub(crate) fn add_features_from_bitboard(&self, bb: &BitBoard, acc: &mut Accumulator) {
        // White barrels (piece_type = 0)
        let mut barrels = bb.white_barrels;
        while barrels != 0 {
            let sq = barrels.trailing_zeros() as usize;
            let feat = feature_index(sq, 0);
            self.add_feature(acc, feat);
            barrels &= barrels - 1;
        }

        // Black barrels (piece_type = 1)
        barrels = bb.black_barrels;
        while barrels != 0 {
            let sq = barrels.trailing_zeros() as usize;
            let feat = feature_index(sq, 1);
            self.add_feature(acc, feat);
            barrels &= barrels - 1;
        }

        // White pail (piece_type = 2)
        if bb.white_pail != 0 {
            let sq = bb.white_pail.trailing_zeros() as usize;
            let feat = feature_index(sq, 2);
            self.add_feature(acc, feat);
        }

        // Black pail (piece_type = 3)
        if bb.black_pail != 0 {
            let sq = bb.black_pail.trailing_zeros() as usize;
            let feat = feature_index(sq, 3);
            self.add_feature(acc, feat);
        }
    }

    /// Legg til én feature til accumulator (SIMD-accelerert med transponerte vekter)
    #[inline]
    fn add_feature(&self, acc: &mut Accumulator, feat: usize) {
        let hidden1 = self.hidden1;
        let base_idx = feat * hidden1;  // Contiguous weights for this feature
        let mut i = 0;

        // SIMD loop: process 8 elements at a time with contiguous memory access
        while i + 8 <= hidden1 {
            let acc_vec = f32x8::new([
                acc.pre_activation[i],
                acc.pre_activation[i + 1],
                acc.pre_activation[i + 2],
                acc.pre_activation[i + 3],
                acc.pre_activation[i + 4],
                acc.pre_activation[i + 5],
                acc.pre_activation[i + 6],
                acc.pre_activation[i + 7],
            ]);
            // Now weights are contiguous in memory!
            let weight_vec = f32x8::new([
                self.fc1_weight_t[base_idx + i],
                self.fc1_weight_t[base_idx + i + 1],
                self.fc1_weight_t[base_idx + i + 2],
                self.fc1_weight_t[base_idx + i + 3],
                self.fc1_weight_t[base_idx + i + 4],
                self.fc1_weight_t[base_idx + i + 5],
                self.fc1_weight_t[base_idx + i + 6],
                self.fc1_weight_t[base_idx + i + 7],
            ]);
            let result = acc_vec + weight_vec;
            let arr = result.to_array();
            acc.pre_activation[i] = arr[0];
            acc.pre_activation[i + 1] = arr[1];
            acc.pre_activation[i + 2] = arr[2];
            acc.pre_activation[i + 3] = arr[3];
            acc.pre_activation[i + 4] = arr[4];
            acc.pre_activation[i + 5] = arr[5];
            acc.pre_activation[i + 6] = arr[6];
            acc.pre_activation[i + 7] = arr[7];
            i += 8;
        }
    }

    /// Fjern én feature fra accumulator (SIMD-accelerert med transponerte vekter)
    #[inline]
    fn remove_feature(&self, acc: &mut Accumulator, feat: usize) {
        let hidden1 = self.hidden1;
        let base_idx = feat * hidden1;  // Contiguous weights for this feature
        let mut i = 0;

        // SIMD loop: process 8 elements at a time with contiguous memory access
        while i + 8 <= hidden1 {
            let acc_vec = f32x8::new([
                acc.pre_activation[i],
                acc.pre_activation[i + 1],
                acc.pre_activation[i + 2],
                acc.pre_activation[i + 3],
                acc.pre_activation[i + 4],
                acc.pre_activation[i + 5],
                acc.pre_activation[i + 6],
                acc.pre_activation[i + 7],
            ]);
            // Now weights are contiguous in memory!
            let weight_vec = f32x8::new([
                self.fc1_weight_t[base_idx + i],
                self.fc1_weight_t[base_idx + i + 1],
                self.fc1_weight_t[base_idx + i + 2],
                self.fc1_weight_t[base_idx + i + 3],
                self.fc1_weight_t[base_idx + i + 4],
                self.fc1_weight_t[base_idx + i + 5],
                self.fc1_weight_t[base_idx + i + 6],
                self.fc1_weight_t[base_idx + i + 7],
            ]);
            let result = acc_vec - weight_vec;
            let arr = result.to_array();
            acc.pre_activation[i] = arr[0];
            acc.pre_activation[i + 1] = arr[1];
            acc.pre_activation[i + 2] = arr[2];
            acc.pre_activation[i + 3] = arr[3];
            acc.pre_activation[i + 4] = arr[4];
            acc.pre_activation[i + 5] = arr[5];
            acc.pre_activation[i + 6] = arr[6];
            acc.pre_activation[i + 7] = arr[7];
            i += 8;
        }
    }

    /// Oppdater accumulator med deltas
    #[inline]
    pub fn apply_deltas(&self, acc: &mut Accumulator, deltas: &[FeatureDelta]) {
        for d in deltas {
            let feat = d.index as usize;
            if d.delta > 0 {
                self.add_feature(acc, feat);
            } else {
                self.remove_feature(acc, feat);
            }
        }
        acc.apply_relu();
    }

    /// Beregn feature deltas for et trekk (delegates to free function)
    pub fn compute_move_deltas(&self, bb: &BitBoard, mv: &BitMove) -> Vec<FeatureDelta> {
        compute_nnue_move_deltas(bb, mv)
    }

    /// Evaluer fra ferdig accumulator (layer 2 og 3) med SIMD
    pub fn evaluate_from_accumulator(&self, acc: &Accumulator) -> f32 {
        // Layer 2: FC + ReLU (64 inputs -> 32 outputs)
        // Use SIMD to compute dot products 8 elements at a time
        let mut hidden2 = [0.0f32; MAX_HIDDEN2];

        for i in 0..self.hidden2 {
            let weight_offset = i * self.hidden1;

            // SIMD dot product: process 8 floats at a time
            let mut sum_vec = f32x8::ZERO;
            let mut j = 0;

            // Main SIMD loop (64 / 8 = 8 iterations)
            while j + 8 <= self.hidden1 {
                let input_vec = f32x8::new([
                    acc.post_activation[j],
                    acc.post_activation[j + 1],
                    acc.post_activation[j + 2],
                    acc.post_activation[j + 3],
                    acc.post_activation[j + 4],
                    acc.post_activation[j + 5],
                    acc.post_activation[j + 6],
                    acc.post_activation[j + 7],
                ]);
                let weight_vec = f32x8::new([
                    self.fc2_weight[weight_offset + j],
                    self.fc2_weight[weight_offset + j + 1],
                    self.fc2_weight[weight_offset + j + 2],
                    self.fc2_weight[weight_offset + j + 3],
                    self.fc2_weight[weight_offset + j + 4],
                    self.fc2_weight[weight_offset + j + 5],
                    self.fc2_weight[weight_offset + j + 6],
                    self.fc2_weight[weight_offset + j + 7],
                ]);
                sum_vec = sum_vec + input_vec * weight_vec;
                j += 8;
            }

            // Horizontal sum + bias
            let arr = sum_vec.to_array();
            let sum = self.fc2_bias[i]
                + arr[0] + arr[1] + arr[2] + arr[3]
                + arr[4] + arr[5] + arr[6] + arr[7];

            // ReLU
            hidden2[i] = sum.max(0.0);
        }

        // Layer 3: FC + Tanh (32 inputs -> 1 output)
        // SIMD dot product for 32 elements (4 iterations of 8)
        let mut sum_vec = f32x8::ZERO;
        let mut i = 0;

        while i + 8 <= self.hidden2 {
            let input_vec = f32x8::new([
                hidden2[i],
                hidden2[i + 1],
                hidden2[i + 2],
                hidden2[i + 3],
                hidden2[i + 4],
                hidden2[i + 5],
                hidden2[i + 6],
                hidden2[i + 7],
            ]);
            let weight_vec = f32x8::new([
                self.fc3_weight[i],
                self.fc3_weight[i + 1],
                self.fc3_weight[i + 2],
                self.fc3_weight[i + 3],
                self.fc3_weight[i + 4],
                self.fc3_weight[i + 5],
                self.fc3_weight[i + 6],
                self.fc3_weight[i + 7],
            ]);
            sum_vec = sum_vec + input_vec * weight_vec;
            i += 8;
        }

        // Horizontal sum + bias
        let arr = sum_vec.to_array();
        let output = self.fc3_bias
            + arr[0] + arr[1] + arr[2] + arr[3]
            + arr[4] + arr[5] + arr[6] + arr[7];

        output.tanh()
    }

    /// Evaluer og skaler til centipawn
    #[inline]
    pub fn evaluate_cp(&self, acc: &Accumulator) -> i32 {
        (self.evaluate_from_accumulator(acc) * 1000.0) as i32
    }

    /// Evaluate with relational features (for models with INPUT_SIZE > 144)
    ///
    /// This method adds relational features to the base accumulator and evaluates.
    /// Used when incremental updates only track base features.
    pub fn evaluate_with_relational(&self, bb: &BitBoard, base_acc: &Accumulator) -> f32 {
        if self.input_size <= BASE_FEATURES {
            // Legacy model - just use base accumulator
            let mut acc = Accumulator::default();
            acc.pre_activation = base_acc.pre_activation;
            acc.apply_relu();
            return self.evaluate_from_accumulator(&acc);
        }

        // Create working accumulator with base features
        let mut acc = Accumulator::default();
        acc.pre_activation = base_acc.pre_activation;

        // Add relational features
        self.add_relational_features(bb, &mut acc);

        // Apply ReLU and evaluate
        acc.apply_relu();
        self.evaluate_from_accumulator(&acc)
    }

    /// Evaluate with relational features and return centipawn score
    #[inline]
    pub fn evaluate_with_relational_cp(&self, bb: &BitBoard, base_acc: &Accumulator) -> i32 {
        (self.evaluate_with_relational(bb, base_acc) * 1000.0) as i32
    }

    /// Evaluate using a reusable working accumulator (avoids allocation per eval)
    /// This is the fast path - reuses eval_acc instead of creating new Accumulator
    #[inline]
    pub fn evaluate_with_reusable_acc(&self, bb: &BitBoard, base_acc: &Accumulator, eval_acc: &mut Accumulator) -> i32 {
        // Copy pre_activation from base accumulator
        eval_acc.pre_activation = base_acc.pre_activation;

        // Add relational features using SIMD-optimized path
        self.add_relational_features(bb, eval_acc);

        // Apply ReLU and evaluate
        eval_acc.apply_relu();
        (self.evaluate_from_accumulator(eval_acc) * 1000.0) as i32
    }

    /// Evaluate using only base features (skip relational features for benchmarking)
    #[inline]
    pub fn evaluate_base_only_cp(&self, base_acc: &Accumulator) -> i32 {
        let mut acc = Accumulator::default();
        acc.pre_activation = base_acc.pre_activation;
        acc.apply_relu();
        (self.evaluate_from_accumulator(&acc) * 1000.0) as i32
    }

    /// Inkrementell evaluering: oppdater accumulator og evaluer
    pub fn evaluate_incremental(
        &self,
        bb: &BitBoard,
        mv: &BitMove,
        acc: &mut Accumulator,
    ) -> f32 {
        let deltas = self.compute_move_deltas(bb, mv);
        self.apply_deltas(acc, &deltas);
        self.evaluate_from_accumulator(acc)
    }
}

// ============================================================================
// QUANTIZED NNUE EVALUATOR - Int16 weights with i16x16 SIMD
// ============================================================================

/// Quantized NNUE with int16 weights and int16 accumulator.
/// Uses i16x16 SIMD (16 lanes) for accumulator updates and i16x16.dot() for FC2.
/// FC3 is computed in f32 for simplicity (only 32 multiply-adds).
pub struct QuantizedNNUE {
    /// FC1 transposed weights: [input_size * hidden1], i16 at scale fc1_weight_scale
    fc1_weight_t: Vec<i16>,
    /// FC1 bias: [hidden1], i16 at scale fc1_weight_scale
    fc1_bias: Vec<i16>,
    /// FC2 weights: [hidden2 * hidden1], i16 at scale fc2_weight_scale
    fc2_weight: Vec<i16>,
    /// FC2 bias: [hidden2], i32 at scale (fc1_scale >> crelu_shift) * fc2_scale
    fc2_bias: Vec<i32>,
    /// FC3 weights: [hidden2], converted to f32 during load (pre-scaled by output_scale)
    fc3_weight: Vec<f32>,
    /// FC3 bias: f32 (pre-scaled by output_scale)
    fc3_bias: f32,
    /// Right-shift for ClippedReLU: clamp(acc >> crelu_shift, 0, 127)
    crelu_shift: u32,
    /// 1.0 / ((fc1_scale >> crelu_shift) * fc2_scale) — converts FC2 i32 output to f32
    #[allow(dead_code)]
    fc2_output_scale_inv: f32,
    hidden1: usize,
    hidden2: usize,
    input_size: usize,
}

impl QuantizedNNUE {
    /// Load quantized NNUE from parsed JSON
    pub(crate) fn from_json(json: QuantizedNNUEJson) -> Result<Self, Box<dyn std::error::Error>> {
        let hidden1 = json.hidden1;
        let hidden2 = json.hidden2;
        let input_size = json.input_size;

        // Flatten and transpose FC1 weights: [hidden1][input_size] -> [input_size * hidden1]
        let fc1_weight: Vec<i16> = json.weights.fc1_weight.into_iter().flatten().collect();
        let mut fc1_weight_t = vec![0i16; input_size * hidden1];
        for neuron in 0..hidden1 {
            for feature in 0..input_size {
                fc1_weight_t[feature * hidden1 + neuron] = fc1_weight[neuron * input_size + feature];
            }
        }

        // Flatten FC2 weights: [hidden2][hidden1] -> [hidden2 * hidden1]
        let fc2_weight: Vec<i16> = json.weights.fc2_weight.into_iter().flatten().collect();

        // Compute the scale of FC2's dot product output
        let fc2_input_scale = json.fc1_weight_scale >> json.crelu_shift;
        let fc2_output_scale = fc2_input_scale * json.fc2_weight_scale;
        let fc2_output_scale_inv = 1.0 / fc2_output_scale as f32;

        // Convert FC3 weights from i16 to f32, pre-scaled by output_scale
        // output_scale = 1.0 / (fc2_output_scale * fc3_weight_scale)
        let output_scale = json.output_scale as f32;
        let fc3_weight: Vec<f32> = json.weights.fc3_weight.iter()
            .map(|&w| w as f32 * output_scale)
            .collect();
        let fc3_bias = json.weights.fc3_bias as f32 * output_scale;

        Ok(Self {
            fc1_weight_t,
            fc1_bias: json.weights.fc1_bias,
            fc2_weight,
            fc2_bias: json.weights.fc2_bias,
            fc3_weight,
            fc3_bias,
            crelu_shift: json.crelu_shift,
            fc2_output_scale_inv,
            hidden1,
            hidden2,
            input_size,
        })
    }

    /// Initialize quantized accumulator with bias + all active base features
    pub fn init_accumulator(&self, bb: &BitBoard, acc: &mut QAccumulator) {
        for i in 0..self.hidden1 {
            acc.values[i] = self.fc1_bias[i];
        }
        self.add_features_from_bitboard_q(bb, acc);
    }

    /// Add all base features from a bitboard to quantized accumulator
    fn add_features_from_bitboard_q(&self, bb: &BitBoard, acc: &mut QAccumulator) {
        let mut barrels = bb.white_barrels;
        while barrels != 0 {
            let sq = barrels.trailing_zeros() as usize;
            self.add_feature_q(acc, feature_index(sq, 0));
            barrels &= barrels - 1;
        }

        barrels = bb.black_barrels;
        while barrels != 0 {
            let sq = barrels.trailing_zeros() as usize;
            self.add_feature_q(acc, feature_index(sq, 1));
            barrels &= barrels - 1;
        }

        if bb.white_pail != 0 {
            let sq = bb.white_pail.trailing_zeros() as usize;
            self.add_feature_q(acc, feature_index(sq, 2));
        }

        if bb.black_pail != 0 {
            let sq = bb.black_pail.trailing_zeros() as usize;
            self.add_feature_q(acc, feature_index(sq, 3));
        }
    }

    /// Add one binary feature to quantized accumulator (i16x16 SIMD, 16 lanes)
    #[inline]
    fn add_feature_q(&self, acc: &mut QAccumulator, feat: usize) {
        let hidden1 = self.hidden1;
        let base_idx = feat * hidden1;
        let mut j = 0;

        while j + 16 <= hidden1 {
            let mut acc_arr = [0i16; 16];
            acc_arr.copy_from_slice(&acc.values[j..j + 16]);
            let acc_vec = i16x16::new(acc_arr);

            let mut wt_arr = [0i16; 16];
            wt_arr.copy_from_slice(&self.fc1_weight_t[base_idx + j..base_idx + j + 16]);
            let wt_vec = i16x16::new(wt_arr);

            let result = acc_vec + wt_vec;
            acc.values[j..j + 16].copy_from_slice(&result.to_array());
            j += 16;
        }
    }

    /// Remove one binary feature from quantized accumulator (i16x16 SIMD)
    #[inline]
    fn remove_feature_q(&self, acc: &mut QAccumulator, feat: usize) {
        let hidden1 = self.hidden1;
        let base_idx = feat * hidden1;
        let mut j = 0;

        while j + 16 <= hidden1 {
            let mut acc_arr = [0i16; 16];
            acc_arr.copy_from_slice(&acc.values[j..j + 16]);
            let acc_vec = i16x16::new(acc_arr);

            let mut wt_arr = [0i16; 16];
            wt_arr.copy_from_slice(&self.fc1_weight_t[base_idx + j..base_idx + j + 16]);
            let wt_vec = i16x16::new(wt_arr);

            let result = acc_vec - wt_vec;
            acc.values[j..j + 16].copy_from_slice(&result.to_array());
            j += 16;
        }
    }

    /// Apply feature deltas to quantized accumulator (no activation applied here)
    #[inline]
    pub fn apply_deltas_q(&self, acc: &mut QAccumulator, deltas: &[FeatureDelta]) {
        for d in deltas {
            let feat = d.index as usize;
            if d.delta > 0 {
                self.add_feature_q(acc, feat);
            } else {
                self.remove_feature_q(acc, feat);
            }
        }
    }

    /// Add relational features to a working copy of accumulator values.
    /// Uses f32x8 SIMD since relational features are continuous [-1, 1] values
    /// that need float multiplication with i16 weights.
    #[inline]
    fn add_relational_features_q(&self, bb: &BitBoard, acc: &mut [i16; MAX_HIDDEN1]) {
        if self.input_size <= BASE_FEATURES {
            return;
        }

        let hidden1 = self.hidden1;
        let rel_feats = IncrementalNNUE::compute_relational_features(bb);

        for (feat_idx, &feat_val) in rel_feats.iter().enumerate() {
            if feat_val == 0.0 {
                continue;
            }
            let weight_base = (BASE_FEATURES + feat_idx) * hidden1;

            // Process 8 neurons at a time: load i16 weight → f32, multiply, round, add back
            let feat_splat = f32x8::splat(feat_val);
            let mut j = 0;
            while j + 8 <= hidden1 {
                let w = f32x8::new([
                    self.fc1_weight_t[weight_base + j] as f32,
                    self.fc1_weight_t[weight_base + j + 1] as f32,
                    self.fc1_weight_t[weight_base + j + 2] as f32,
                    self.fc1_weight_t[weight_base + j + 3] as f32,
                    self.fc1_weight_t[weight_base + j + 4] as f32,
                    self.fc1_weight_t[weight_base + j + 5] as f32,
                    self.fc1_weight_t[weight_base + j + 6] as f32,
                    self.fc1_weight_t[weight_base + j + 7] as f32,
                ]);
                let contrib = feat_splat * w;
                let arr = contrib.to_array();
                for k in 0..8 {
                    acc[j + k] = acc[j + k].saturating_add(arr[k].round() as i16);
                }
                j += 8;
            }
        }
    }

    /// Full evaluation from quantized accumulator: ClippedReLU → FC2 (dot) → FC3 → tanh
    /// Returns centipawn score.
    pub fn evaluate_from_qacc(&self, bb: &BitBoard, base_acc: &QAccumulator) -> i32 {
        // 1. Copy base accumulator and add relational features
        let mut acc = base_acc.values;
        self.add_relational_features_q(bb, &mut acc);

        // 2. ClippedReLU: clamp(acc >> shift, 0, 127)
        let mut clipped = [0i16; MAX_HIDDEN1];
        {
            let zero = i16x16::ZERO;
            let max_val = i16x16::new([127i16; 16]);
            let mut j = 0;
            while j + 16 <= self.hidden1 {
                let mut arr = [0i16; 16];
                arr.copy_from_slice(&acc[j..j + 16]);
                let v = i16x16::new(arr);
                // Right-shift, then clamp to [0, 127]
                let shifted = v >> self.crelu_shift;
                let clamped = shifted.max(zero).min(max_val);
                clipped[j..j + 16].copy_from_slice(&clamped.to_array());
                j += 16;
            }
        }

        // 3. FC2: dot product with i16x16.dot() → i32x8 (the main speedup)
        // 64 inputs → 32 outputs, processing 16 at a time
        let mut hidden2 = [0i32; MAX_HIDDEN2];
        for neuron in 0..self.hidden2 {
            let wt_base = neuron * self.hidden1;
            let mut sum = i32x8::ZERO;

            let mut j = 0;
            while j + 16 <= self.hidden1 {
                let mut in_arr = [0i16; 16];
                in_arr.copy_from_slice(&clipped[j..j + 16]);
                let input = i16x16::new(in_arr);

                let mut wt_arr = [0i16; 16];
                wt_arr.copy_from_slice(&self.fc2_weight[wt_base + j..wt_base + j + 16]);
                let weight = i16x16::new(wt_arr);

                // vpmaddwd: 16 multiplies → 8 pairwise sums as i32
                sum = sum + input.dot(weight);
                j += 16;
            }

            // Horizontal sum of i32x8 + bias
            hidden2[neuron] = sum.reduce_add() + self.fc2_bias[neuron];
        }

        // 4. FC2 ReLU (in i32) + convert to f32 for FC3
        // FC3 weights are pre-scaled by output_scale during loading
        let mut output = self.fc3_bias;
        for i in 0..self.hidden2 {
            let h2_relu = hidden2[i].max(0) as f32;
            output += h2_relu * self.fc3_weight[i];
        }

        // 5. tanh and scale to centipawns
        (output.tanh() * 1000.0) as i32
    }
}

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

/// HalfPail NNUE evaluator
///
/// Architecture:
///   White sparse → EmbeddingBag(3996, H1) + bias → ReLU → acc_white
///   Black sparse → EmbeddingBag(3996, H1) + bias → ReLU → acc_black
///                       (shared weights)
///   concat(acc_white, acc_black, dense_6) → FC2 → ReLU → FC3 → Tanh
pub struct HalfPailNNUE {
    /// FC1 transposed weights: [HALFPAIL_FEATURES * hidden1]
    /// Transposed for cache-friendly feature updates: fc1_weight_t[feature * hidden1 + neuron]
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
        // in column-major order: fc1_weight_t[feature * hidden1 + neuron]
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
                    // White placed their pail → white perspective bucket changes → full recompute
                    white_recompute = true;
                    // For black perspective: enemy pail appeared at pail_sq
                    black_deltas.push(HalfPailDelta {
                        index: halfpail_feature_index(b_bucket, pail_sq, 2),
                        delta: 1,
                    });
                }
                Player::Black => {
                    // Black placed their pail → black perspective bucket changes → full recompute
                    black_recompute = true;
                    // For white perspective: enemy pail appeared at pail_sq
                    white_deltas.push(HalfPailDelta {
                        index: halfpail_feature_index(w_bucket, pail_sq, 2),
                        delta: 1,
                    });
                }
            }
        }

        // After pail placement, use updated buckets for barrel deltas
        // But only if the perspective is NOT being recomputed (recomputed ones get fresh init)
        let w_bucket_for_deltas = if white_recompute {
            // Doesn't matter - will be recomputed anyway
            // But we still need a valid bucket for black's perspective of white's barrels
            w_bucket
        } else {
            w_bucket
        };
        let b_bucket_for_deltas = if black_recompute {
            b_bucket
        } else {
            b_bucket
        };

        // 2. Handle barrel movement
        if mv.is_placement() {
            let to_sq = mv.barrel_to() as usize;
            let goal_row = bb.goal_row(player);
            let (to_row, _) = sq_to_coords(to_sq);

            if to_row != goal_row {
                // Barrel placed on board (not scored immediately)
                match player {
                    Player::White => {
                        if !white_recompute {
                            white_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(w_bucket_for_deltas, to_sq, 0), // friendly
                                delta: 1,
                            });
                        }
                        if !black_recompute {
                            black_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(b_bucket_for_deltas, to_sq, 1), // enemy
                                delta: 1,
                            });
                        }
                    }
                    Player::Black => {
                        if !white_recompute {
                            white_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(w_bucket_for_deltas, to_sq, 1), // enemy
                                delta: 1,
                            });
                        }
                        if !black_recompute {
                            black_deltas.push(HalfPailDelta {
                                index: halfpail_feature_index(b_bucket_for_deltas, to_sq, 0), // friendly
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
                    // Remove from old position
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
                    // Add to new position (if not scored)
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

    /// Compute 20 dense features from a BitBoard position.
    /// Reuses the same relational features as the legacy NNUE, in the same order
    /// as the training data (features 144-163 of the 164-feature encoding).
    #[inline]
    fn compute_dense_features(bb: &BitBoard) -> [f32; HALFPAIL_DENSE] {
        IncrementalNNUE::compute_relational_features(bb)
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
        let dense = Self::compute_dense_features(bb);

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

        // FC3: dot product → tanh → centipawns
        let mut output = self.fc3_bias;
        // SIMD for FC3 if hidden2 >= 8
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
// EVALUATION CACHE - Unngå redundante NNUE-evalueringer
// ============================================================================

/// Størrelse på eval cache (power of 2 for rask modulo)
pub(crate) const EVAL_CACHE_SIZE: usize = 1 << 16; // 65536 entries

/// Entry i eval cache
#[derive(Clone, Copy, Default)]
pub(crate) struct EvalCacheEntry {
    hash: u64,
    score: i32,
    generation: u8,
}

/// Cache for statiske evalueringer
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

    /// Prøv å hente cached evaluering
    #[inline]
    pub(crate) fn probe(&mut self, hash: u64) -> Option<i32> {
        let idx = self.index(hash);
        let entry = &self.entries[idx];
        // Check hash AND generation
        if entry.hash == hash && entry.generation == self.generation {
            self.hits += 1;
            Some(entry.score)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Lagre evaluering i cache
    #[inline]
    pub(crate) fn store(&mut self, hash: u64, score: i32) {
        let idx = self.index(hash);
        self.entries[idx] = EvalCacheEntry { hash, score, generation: self.generation };
    }

    /// Tøm cache - O(1) using generation counter
    pub(crate) fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.hits = 0;
        self.misses = 0;
    }

    /// Hit ratio
    pub(crate) fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}
