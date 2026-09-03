use serde::Deserialize;
use wide::{i16x16, i32x8};

use crate::board::*;

/// Maximum supported hidden layer sizes (for fixed-size accumulator arrays)
pub const MAX_HIDDEN1: usize = 512;
pub const MAX_HIDDEN2: usize = 128;
/// Dense (relational) feature count when enabled
pub const HALFPAIL_DENSE: usize = 20;
/// Dense set v2: the 20 relational features + 8 (race distances, forward jump
/// availability, forward mobility, no-progress clock, mid-turn flag).
pub const DENSE_V2: usize = 28;
pub const MAX_DENSE: usize = 32;
/// Upper bound on simultaneously active sparse features per perspective:
/// 4 own barrels + 4 enemy barrels + 2 pails.
pub const MAX_ACTIVE_FEATURES: usize = 12;
/// Label normalization used by the trainer: label = tanh(score_cp / SCORE_SCALE).
/// Must match python/tonnesjakk/nnue.py SCORE_SCALING.
pub const SCORE_SCALE: f32 = 600.0;

// ─── Quantized inference ───
// First layer (embedding + bias) and the accumulators are i16 at scale Q_ACC;
// second-layer weights are i16 at scale Q_W2; the FC2 dot product runs as
// i16x16 widening multiply-adds into i32 (16 MACs per SIMD op vs 8 for f32).
// Bounds chosen so the i32 accumulation cannot overflow:
//   |input| <= IN_CLIP (=23.4 in float units), |w2| <= W2_CLIP (=3.0):
//   3000 * 3072 * 256 inputs ≈ 2.4e9 < i32::MAX (inputs are padded to ≤ 256).
pub const Q_ACC: f32 = 128.0;
pub const Q_W2: f32 = 1024.0;
const IN_CLIP: i16 = 3000;
const W2_CLIP: f32 = 3.0;
const FC2_PAD: usize = 16;

// ============================================================================
// FEATURE SETS
// ============================================================================
//
// One generic evaluator, many input encodings. A feature set maps a board
// (seen from one perspective: "own" vs "opponent" pieces) to a small set of
// active sparse indices. Everything else — accumulators, incremental updates,
// the dense stack, JSON loading — is shared, so a new architecture costs one
// indexing function here plus its mirror in the Python trainer.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FeatureSet {
    /// Pail-square buckets: own pail square (36 + "not placed") conditions
    /// every piece feature (king-bucket analog). 3 piece types per square:
    /// own barrel, enemy barrel, enemy pail. 37 * 36 * 3 = 3996 features.
    HalfPail,
    /// No buckets: 4 piece types per square (own barrel, enemy barrel,
    /// enemy pail, own pail). 36 * 4 = 144 features.
    Plain,
}

impl FeatureSet {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "halfpail" => Some(FeatureSet::HalfPail),
            "plain" => Some(FeatureSet::Plain),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            FeatureSet::HalfPail => "halfpail",
            FeatureSet::Plain => "plain",
        }
    }

    pub const fn num_features(self) -> usize {
        match self {
            FeatureSet::HalfPail => (NUM_SQUARES + 1) * NUM_SQUARES * 3,
            FeatureSet::Plain => NUM_SQUARES * 4,
        }
    }
}

/// Mirror a square vertically (row r -> BOARD_SIZE-1-r). Used so the black
/// perspective sees the board in white's orientation and shared weights mean
/// the same thing for both colors ("own barrel one row from goal").
#[inline(always)]
pub const fn mirror_sq(sq: usize) -> usize {
    (BOARD_SIZE - 1 - sq / BOARD_SIZE) * BOARD_SIZE + sq % BOARD_SIZE
}

/// HalfPail feature index (kept public: the legacy decoder uses it)
#[inline(always)]
pub const fn halfpail_feature_index(bucket: usize, sq: usize, piece_type: usize) -> u16 {
    (bucket * (NUM_SQUARES * 3) + sq * 3 + piece_type) as u16
}

/// Architecture description shared by the Rust evaluator, the training-data
/// decoder, and the Python trainer (via JSON).
#[derive(Clone, Copy, Debug)]
pub struct NnueConfig {
    pub feature_set: FeatureSet,
    /// Black perspective sees a vertically mirrored board.
    pub mirror_black: bool,
    /// 0 or HALFPAIL_DENSE relational features appended before FC2.
    pub dense_size: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    /// 1, or 25 = separate output head per (white_scored, black_scored).
    pub output_buckets: usize,
}

impl NnueConfig {
    /// Active sparse features for one perspective. Returns the count written
    /// into `out`.
    pub fn active_features(&self, bb: &BitBoard, persp: Player, out: &mut [u16; MAX_ACTIVE_FEATURES]) -> usize {
        let (own_b, opp_b, own_p, opp_p) = match persp {
            Player::White => (bb.white_barrels, bb.black_barrels, bb.white_pail, bb.black_pail),
            Player::Black => (bb.black_barrels, bb.white_barrels, bb.black_pail, bb.white_pail),
        };
        let flip = self.mirror_black && persp == Player::Black;
        let tf = |sq: usize| if flip { mirror_sq(sq) } else { sq };

        let mut n = 0usize;
        match self.feature_set {
            FeatureSet::HalfPail => {
                let bucket = if own_p != 0 { tf(own_p.trailing_zeros() as usize) } else { NUM_SQUARES };
                let mut m = own_b;
                while m != 0 {
                    out[n] = halfpail_feature_index(bucket, tf(m.trailing_zeros() as usize), 0);
                    n += 1;
                    m &= m - 1;
                }
                m = opp_b;
                while m != 0 {
                    out[n] = halfpail_feature_index(bucket, tf(m.trailing_zeros() as usize), 1);
                    n += 1;
                    m &= m - 1;
                }
                if opp_p != 0 {
                    out[n] = halfpail_feature_index(bucket, tf(opp_p.trailing_zeros() as usize), 2);
                    n += 1;
                }
            }
            FeatureSet::Plain => {
                let mut m = own_b;
                while m != 0 {
                    out[n] = (tf(m.trailing_zeros() as usize) * 4) as u16;
                    n += 1;
                    m &= m - 1;
                }
                m = opp_b;
                while m != 0 {
                    out[n] = (tf(m.trailing_zeros() as usize) * 4 + 1) as u16;
                    n += 1;
                    m &= m - 1;
                }
                if opp_p != 0 {
                    out[n] = (tf(opp_p.trailing_zeros() as usize) * 4 + 2) as u16;
                    n += 1;
                }
                if own_p != 0 {
                    out[n] = (tf(own_p.trailing_zeros() as usize) * 4 + 3) as u16;
                    n += 1;
                }
            }
        }
        n
    }

    /// Output head index for a position.
    #[inline]
    pub fn output_bucket(&self, bb: &BitBoard) -> usize {
        if self.output_buckets >= 25 {
            (bb.white_scored as usize) * 5 + bb.black_scored as usize
        } else {
            0
        }
    }

    /// Does this perspective's feature bucket change between two positions
    /// (HalfPail: own pail placed)? Then a full recompute beats a diff.
    #[inline]
    fn bucket_changed(&self, before: &BitBoard, after: &BitBoard, persp: Player) -> bool {
        if self.feature_set != FeatureSet::HalfPail {
            return false;
        }
        let (b, a) = match persp {
            Player::White => (before.white_pail, after.white_pail),
            Player::Black => (before.black_pail, after.black_pail),
        };
        b != a
    }
}


const ALL36: u64 = (1u64 << NUM_SQUARES) - 1;
const COL0: u64 = 0x041041041; // column 0 of a 6x6 board (bits 0,6,12,18,24,30)
const COL5: u64 = COL0 << 5;

/// Shift every set square by (dr, dc); squares leaving the board are dropped.
#[inline(always)]
fn shift_sq(mask: u64, dr: i32, dc: i32) -> u64 {
    let m = match dc { -1 => mask & !COL0, 1 => mask & !COL5, _ => mask };
    let s = dr * BOARD_SIZE as i32 + dc;
    (if s >= 0 { m << s } else { m >> (-s) }) & ALL36
}

/// Origin barrels (subset of `barrels`) that have (a) a forward jump — over any
/// piece except `enemy_pail`, landing on an empty on-board square — and (b) an
/// empty forward step square, for the side whose forward row delta is `dr`.
#[inline]
fn forward_jump_step_origins(barrels: u64, occ: u64, enemy_pail: u64, dr: i32) -> (u64, u64) {
    let (mut jump, mut step) = (0u64, 0u64);
    for dc in -1i32..=1 {
        let s1 = shift_sq(barrels, dr, dc);
        step |= shift_sq(s1 & !occ, -dr, -dc) & barrels;
        let over = s1 & occ & !enemy_pail;
        let land = shift_sq(over, dr, dc) & !occ;
        jump |= shift_sq(shift_sq(land, -dr, -dc), -dr, -dc) & barrels;
    }
    (jump, step)
}

/// Dense inputs for a given `dense_size` (0, 20 or 28). Sizes above 20 extend
/// the 20 relational features; all are computed from the board, so training
/// data (which stores only the first 20) never needs regenerating.
#[inline]
pub fn compute_dense(bb: &BitBoard, dense_size: usize) -> [f32; MAX_DENSE] {
    let mut out = [0.0f32; MAX_DENSE];
    if dense_size == 0 { return out; }
    out[..HALFPAIL_DENSE].copy_from_slice(&compute_relational_features(bb));
    if dense_size <= HALFPAIL_DENSE { return out; }
    // [20-21] race distance per side (min moves for a lone side to score
    // everything; the eval term worth +36/+63 Elo), /8 clamped
    let rt = &*crate::race::RACE_TABLE;
    out[20] = (rt.side_distance(bb, Player::White) as f32 / 8.0).min(1.0);
    out[21] = (rt.side_distance(bb, Player::Black) as f32 / 8.0).min(1.0);
    // [22-23] barrels with a forward jump available; [24-25] barrels with an
    // empty forward square (per side, /4). Forward = toward the goal row;
    // jumps go over any piece except the enemy pail and land on an empty
    // on-board square. Bit-parallel: shift the whole side's mask per
    // direction and map landing squares back to their origin barrels.
    let occ = bb.white_barrels | bb.black_barrels | bb.white_pail | bb.black_pail;
    let (wj, ws) = forward_jump_step_origins(bb.white_barrels, occ, bb.black_pail, -1);
    let (bj, bs) = forward_jump_step_origins(bb.black_barrels, occ, bb.white_pail, 1);
    out[22] = wj.count_ones() as f32 / 4.0;
    out[23] = bj.count_ones() as f32 / 4.0;
    out[24] = ws.count_ones() as f32 / 4.0;
    out[25] = bs.count_ones() as f32 / 4.0;
    // [26] no-progress clock (/60, clamped); [27] mid-turn (pail just placed)
    out[26] = (bb.halfmove_clock as f32 / 60.0).min(1.0);
    out[27] = if bb.awaiting_barrel { 1.0 } else { 0.0 };
    out
}

/// Rebuild a BitBoard from the 164-float training row (144 one-hot piece
/// planes + 20 relational features). Used by the training-data decoder so
/// feature indices come from the exact same code the engine evaluates with.
pub fn bitboard_from_dense164(row: &[f32]) -> BitBoard {
    let mut bb = BitBoard::new();
    bb.white_barrels = 0;
    bb.black_barrels = 0;
    for sq in 0..NUM_SQUARES {
        let base = sq * 4;
        if row[base] > 0.5 { bb.white_barrels |= 1u64 << sq; }
        if row[base + 1] > 0.5 { bb.black_barrels |= 1u64 << sq; }
        if row[base + 2] > 0.5 { bb.white_pail = 1u64 << sq; }
        if row[base + 3] > 0.5 { bb.black_pail = 1u64 << sq; }
    }
    bb.occupied = bb.white_barrels | bb.black_barrels | bb.white_pail | bb.black_pail;
    bb.white_pail_placed = bb.white_pail != 0;
    bb.black_pail_placed = bb.black_pail != 0;
    let rel = &row[144..];
    bb.white_scored = (rel[8] * 4.0).round() as u8;
    bb.black_scored = (rel[9] * 4.0).round() as u8;
    bb.current_player = if rel[12] < 0.0 { Player::Black } else { Player::White };
    let w_on = bb.white_barrels.count_ones() as u8;
    let b_on = bb.black_barrels.count_ones() as u8;
    bb.white_barrels_off_board = (BARRELS_PER_PLAYER as u8).saturating_sub(w_on + bb.white_scored);
    bb.black_barrels_off_board = (BARRELS_PER_PLAYER as u8).saturating_sub(b_on + bb.black_scored);
    bb
}

// ============================================================================
// DENSE FEATURES - 20 relational features
// ============================================================================

/// Compute 20 relational/dense features from a BitBoard position.
///
/// These are the same features as the training data (features 144-163 of the
/// 164-feature encoding):
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

    // Sort ≤4 small integers in place (insertion sort; branch-light).
    #[inline(always)]
    fn sort4(v: &mut [u8; 4], n: usize) {
        for i in 1..n {
            let mut j = i;
            while j > 0 && v[j - 1] > v[j] {
                v.swap(j - 1, j);
                j -= 1;
            }
        }
    }

    // White barrels: rows/cols, threats (row 1 = one step from goal row 0)
    let mut white_rows = [0u8; 4];
    let mut white_cols = [0u8; 4];
    let mut n_white = 0usize;
    let mut white_threats = 0u32;
    let mut barrels = bb.white_barrels;
    while barrels != 0 && n_white < 4 {
        let sq = barrels.trailing_zeros() as usize;
        white_rows[n_white] = (sq / 6) as u8;
        white_cols[n_white] = (sq % 6) as u8;
        if white_rows[n_white] == 1 { white_threats += 1; }
        n_white += 1;
        barrels &= barrels - 1;
    }
    // Distances (row/5) sorted ascending == rows sorted ascending; identical
    // arithmetic to the original float formulation.
    let mut wr = white_rows;
    sort4(&mut wr, n_white);
    for i in 0..n_white {
        features[i] = 1.0 - (wr[i] as f32 / 5.0);
    }

    // Black barrels: distance = 5 - row; sorted ascending == rows descending
    let mut black_rows = [0u8; 4];
    let mut black_cols = [0u8; 4];
    let mut n_black = 0usize;
    let mut black_threats = 0u32;
    barrels = bb.black_barrels;
    while barrels != 0 && n_black < 4 {
        let sq = barrels.trailing_zeros() as usize;
        black_rows[n_black] = (sq / 6) as u8;
        black_cols[n_black] = (sq % 6) as u8;
        if black_rows[n_black] == 4 { black_threats += 1; }
        n_black += 1;
        barrels &= barrels - 1;
    }
    let mut bd = [0u8; 4];
    for i in 0..n_black {
        bd[i] = 5 - black_rows[i];
    }
    sort4(&mut bd, n_black);
    for i in 0..n_black {
        features[4 + i] = 1.0 - (bd[i] as f32 / 5.0);
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
            if pail_col == black_cols[i] as usize && pail_row > black_rows[i] as usize {
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
            if pail_col == white_cols[i] as usize && pail_row < white_rows[i] as usize {
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

/// Dual-perspective accumulator: pre-activation first-layer sums, i16 at
/// scale Q_ACC (exactly reproducible: incremental == from-scratch bit for bit).
#[derive(Clone)]
pub struct DualAccumulator {
    pub white_pre: [i16; MAX_HIDDEN1],
    pub black_pre: [i16; MAX_HIDDEN1],
}

impl Default for DualAccumulator {
    fn default() -> Self {
        DualAccumulator {
            white_pre: [0; MAX_HIDDEN1],
            black_pre: [0; MAX_HIDDEN1],
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

    /// Push a copy of the current accumulator (child inherits parent's state).
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
// SPARSE NNUE EVALUATOR (generic over feature set)
// ============================================================================

/// Architecture:
///   own-perspective sparse -> Embedding(F, H1) + bias -> ReLU -> acc_white
///   opp-perspective sparse -> Embedding(F, H1) + bias -> ReLU -> acc_black
///                              (shared weights; black optionally mirrored)
///   concat(acc_white, acc_black[, dense]) -> FC2[bucket] -> ReLU -> FC3[bucket] -> tanh
pub struct SparseNNUE {
    pub config: NnueConfig,
    // ── f32 reference weights (loading, tests, parity) ──
    /// [num_features * hidden1]
    fc1_weight_t: Vec<f32>,
    /// [hidden1]
    fc1_bias: Vec<f32>,
    /// [buckets * hidden2 * fc2_input]
    fc2_weight: Vec<f32>,
    /// [buckets * hidden2]
    fc2_bias: Vec<f32>,
    /// [buckets * hidden2]
    fc3_weight: Vec<f32>,
    /// [buckets]
    fc3_bias: Vec<f32>,
    // ── quantized inference weights ──
    /// [num_features * hidden1], scale Q_ACC
    fc1_weight_q: Vec<i16>,
    /// [hidden1], scale Q_ACC
    fc1_bias_q: Vec<i16>,
    /// [buckets * hidden2 * fc2_in_pad], scale Q_W2, zero-padded rows
    fc2_weight_q: Vec<i16>,
    /// fc2 input length rounded up to a multiple of FC2_PAD
    fc2_in_pad: usize,
}

// ─── JSON formats ───

/// Legacy HalfPail export (single output head, 20 dense, no mirroring).
#[derive(Deserialize)]
struct LegacyWeights {
    fc1_weight: Vec<Vec<f32>>,
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<Vec<f32>>,
    fc2_bias: Vec<f32>,
    fc3_weight: Vec<Vec<f32>>,
    fc3_bias: Vec<f32>,
}

#[derive(Deserialize)]
struct LegacyHalfPailJson {
    #[allow(dead_code)]
    halfpail: bool,
    hidden1: usize,
    hidden2: usize,
    #[allow(dead_code)]
    num_perspective_features: usize,
    #[allow(dead_code)]
    dense_size: usize,
    weights: LegacyWeights,
}

/// Generic v2 export: `"format": "sparse_nnue_v2"`.
#[derive(Deserialize)]
struct V2Weights {
    fc1_weight: Vec<Vec<f32>>,       // [F][H1]
    fc1_bias: Vec<f32>,              // [H1]
    fc2_weight: Vec<Vec<Vec<f32>>>,  // [B][H2][2*H1 + dense]
    fc2_bias: Vec<Vec<f32>>,         // [B][H2]
    fc3_weight: Vec<Vec<f32>>,       // [B][H2]
    fc3_bias: Vec<f32>,              // [B]
}

#[derive(Deserialize)]
struct SparseJsonV2 {
    #[allow(dead_code)]
    format: String,
    feature_set: String,
    mirror_black: bool,
    dense_size: usize,
    hidden1: usize,
    hidden2: usize,
    output_buckets: usize,
    weights: V2Weights,
}

impl SparseNNUE {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json_str = std::fs::read_to_string(path)?;
        Self::from_json_str(&json_str)
    }

    pub fn from_json_str(json_str: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if let Ok(v2) = serde_json::from_str::<SparseJsonV2>(json_str) {
            return Self::from_v2(v2);
        }
        let legacy: LegacyHalfPailJson = serde_json::from_str(json_str)?;
        Self::from_legacy(legacy)
    }

    fn from_v2(j: SparseJsonV2) -> Result<Self, Box<dyn std::error::Error>> {
        let feature_set = FeatureSet::from_name(&j.feature_set)
            .ok_or_else(|| format!("unknown feature_set '{}'", j.feature_set))?;
        let config = NnueConfig {
            feature_set,
            mirror_black: j.mirror_black,
            dense_size: j.dense_size,
            hidden1: j.hidden1,
            hidden2: j.hidden2,
            output_buckets: j.output_buckets.max(1),
        };
        Self::validate(&config)?;
        if j.weights.fc1_weight.len() != feature_set.num_features() {
            return Err(format!(
                "fc1_weight has {} rows, feature set {} needs {}",
                j.weights.fc1_weight.len(), feature_set.name(), feature_set.num_features()
            ).into());
        }
        let fc1_weight_t = transpose_embedding(&j.weights.fc1_weight, config.hidden1);
        Ok(Self::finish(
            config,
            fc1_weight_t,
            j.weights.fc1_bias,
            j.weights.fc2_weight.into_iter().flatten().flatten().collect(),
            j.weights.fc2_bias.into_iter().flatten().collect(),
            j.weights.fc3_weight.into_iter().flatten().collect(),
            j.weights.fc3_bias,
        ))
    }

    fn from_legacy(j: LegacyHalfPailJson) -> Result<Self, Box<dyn std::error::Error>> {
        let config = NnueConfig {
            feature_set: FeatureSet::HalfPail,
            mirror_black: false,
            dense_size: HALFPAIL_DENSE,
            hidden1: j.hidden1,
            hidden2: j.hidden2,
            output_buckets: 1,
        };
        Self::validate(&config)?;
        let fc1_weight_t = transpose_embedding(&j.weights.fc1_weight, config.hidden1);
        Ok(Self::finish(
            config,
            fc1_weight_t,
            j.weights.fc1_bias,
            j.weights.fc2_weight.into_iter().flatten().collect(),
            j.weights.fc2_bias,
            j.weights.fc3_weight.into_iter().flatten().collect(),
            j.weights.fc3_bias,
        ))
    }

    /// Build the quantized inference weights from the f32 weights.
    fn finish(
        config: NnueConfig,
        fc1_weight_t: Vec<f32>,
        fc1_bias: Vec<f32>,
        fc2_weight: Vec<f32>,
        fc2_bias: Vec<f32>,
        fc3_weight: Vec<f32>,
        fc3_bias: Vec<f32>,
    ) -> Self {
        let q16 = |v: f32, scale: f32| -> i16 {
            (v * scale).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
        };
        let fc1_weight_q: Vec<i16> = fc1_weight_t.iter().map(|&w| q16(w, Q_ACC)).collect();
        let fc1_bias_q: Vec<i16> = fc1_bias.iter().map(|&b| q16(b, Q_ACC)).collect();

        let fc2_in = 2 * config.hidden1 + config.dense_size;
        let fc2_in_pad = (fc2_in + FC2_PAD - 1) / FC2_PAD * FC2_PAD;
        let rows = config.output_buckets * config.hidden2;
        let mut fc2_weight_q = vec![0i16; rows * fc2_in_pad];
        for r in 0..rows {
            for k in 0..fc2_in {
                let w = fc2_weight[r * fc2_in + k].clamp(-W2_CLIP, W2_CLIP);
                fc2_weight_q[r * fc2_in_pad + k] = q16(w, Q_W2);
            }
        }

        Self {
            config,
            fc1_weight_t,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
            fc1_weight_q,
            fc1_bias_q,
            fc2_weight_q,
            fc2_in_pad,
        }
    }

    fn validate(c: &NnueConfig) -> Result<(), Box<dyn std::error::Error>> {
        if c.hidden1 > MAX_HIDDEN1 || c.hidden1 % 16 != 0 {
            return Err(format!("hidden1 must be a multiple of 16 and <= {}", MAX_HIDDEN1).into());
        }
        if c.hidden2 > MAX_HIDDEN2 || c.hidden2 % 8 != 0 {
            return Err(format!("hidden2 must be a multiple of 8 and <= {}", MAX_HIDDEN2).into());
        }
        if c.dense_size != 0 && c.dense_size != HALFPAIL_DENSE && c.dense_size != DENSE_V2 {
            return Err(format!("dense_size must be 0, {} or {}", HALFPAIL_DENSE, DENSE_V2).into());
        }
        if c.output_buckets != 1 && c.output_buckets != 25 {
            return Err("output_buckets must be 1 or 25".into());
        }
        Ok(())
    }

    /// Test/benchmark helper: deterministic pseudo-random weights.
    pub fn random(config: NnueConfig, seed: u64) -> Self {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 0.2 - 0.1
        };
        let f = config.feature_set.num_features();
        let fc2_in = 2 * config.hidden1 + config.dense_size;
        let b = config.output_buckets;
        Self::finish(
            config,
            (0..f * config.hidden1).map(|_| next()).collect(),
            (0..config.hidden1).map(|_| next()).collect(),
            (0..b * config.hidden2 * fc2_in).map(|_| next()).collect(),
            (0..b * config.hidden2).map(|_| next()).collect(),
            (0..b * config.hidden2).map(|_| next()).collect(),
            (0..b).map(|_| next()).collect(),
        )
    }

    #[inline]
    fn add_feature(&self, acc: &mut [i16; MAX_HIDDEN1], feat: usize) {
        let h = self.config.hidden1;
        let w = &self.fc1_weight_q[feat * h..(feat + 1) * h];
        for (a, wc) in acc[..h].chunks_exact_mut(16).zip(w.chunks_exact(16)) {
            let r = i16x16::from(<[i16; 16]>::try_from(&*a).unwrap())
                .saturating_add(i16x16::from(<[i16; 16]>::try_from(wc).unwrap()));
            a.copy_from_slice(&r.to_array());
        }
    }

    #[inline]
    fn remove_feature(&self, acc: &mut [i16; MAX_HIDDEN1], feat: usize) {
        let h = self.config.hidden1;
        let w = &self.fc1_weight_q[feat * h..(feat + 1) * h];
        for (a, wc) in acc[..h].chunks_exact_mut(16).zip(w.chunks_exact(16)) {
            let r = i16x16::from(<[i16; 16]>::try_from(&*a).unwrap())
                .saturating_sub(i16x16::from(<[i16; 16]>::try_from(wc).unwrap()));
            a.copy_from_slice(&r.to_array());
        }
    }

    fn init_perspective(&self, bb: &BitBoard, persp: Player, acc: &mut [i16; MAX_HIDDEN1]) {
        let h = self.config.hidden1;
        acc[..h].copy_from_slice(&self.fc1_bias_q[..h]);
        let mut feats = [0u16; MAX_ACTIVE_FEATURES];
        let n = self.config.active_features(bb, persp, &mut feats);
        for &f in &feats[..n] {
            self.add_feature(acc, f as usize);
        }
    }

    /// Initialize both perspectives from scratch.
    pub fn init_accumulators(&self, bb: &BitBoard, acc: &mut DualAccumulator) {
        self.init_perspective(bb, Player::White, &mut acc.white_pre);
        self.init_perspective(bb, Player::Black, &mut acc.black_pre);
    }

    /// Incrementally update `acc` (holding `before`'s state) to `after`.
    /// Generic diff of active feature sets — correct for any feature set with
    /// no per-move special cases. Falls back to a full recompute of a
    /// perspective when its bucket changed.
    pub fn update(&self, before: &BitBoard, after: &BitBoard, acc: &mut DualAccumulator) {
        for persp in [Player::White, Player::Black] {
            let target = match persp {
                Player::White => &mut acc.white_pre,
                Player::Black => &mut acc.black_pre,
            };
            if self.config.bucket_changed(before, after, persp) {
                self.init_perspective(after, persp, target);
                continue;
            }
            let mut fb = [0u16; MAX_ACTIVE_FEATURES];
            let mut fa = [0u16; MAX_ACTIVE_FEATURES];
            let nb = self.config.active_features(before, persp, &mut fb);
            let na = self.config.active_features(after, persp, &mut fa);
            for &f in &fb[..nb] {
                if !fa[..na].contains(&f) {
                    self.remove_feature(target, f as usize);
                }
            }
            for &f in &fa[..na] {
                if !fb[..nb].contains(&f) {
                    self.add_feature(target, f as usize);
                }
            }
        }
    }

    /// Map the FC3 pre-activation to centipawns. Training labels are
    /// tanh(score_cp / SCORE_SCALE); invert so the NNUE lives on the same
    /// centipawn scale as the heuristic eval and every search margin keeps
    /// its meaning.
    #[inline]
    fn to_centipawns(out: f32) -> i32 {
        let v = out.tanh().clamp(-0.999, 0.999);
        (SCORE_SCALE * v.atanh()) as i32
    }

    #[inline]
    fn fc3(&self, bucket: usize, hidden: &[f32; MAX_HIDDEN2]) -> i32 {
        let h2 = self.config.hidden2;
        let fc3_w = &self.fc3_weight[bucket * h2..(bucket + 1) * h2];
        let mut out = self.fc3_bias[bucket];
        for i in 0..h2 {
            out += hidden[i] * fc3_w[i];
        }
        Self::to_centipawns(out)
    }

    /// Quantized evaluate: clipped-ReLU on the i16 accumulators, dense
    /// features quantized to Q_ACC, FC2 as i16x16 widening dot products
    /// accumulated in i32, then FC3 in f32 -> centipawns (White perspective).
    pub fn evaluate(&self, bb: &BitBoard, acc: &DualAccumulator) -> i32 {
        let h1 = self.config.hidden1;
        let h2 = self.config.hidden2;
        let dense_size = self.config.dense_size;
        let in_pad = self.fc2_in_pad;
        let bucket = self.config.output_bucket(bb);

        // Build the padded i16 input vector: relu(white), relu(black), dense
        let mut input = [0i16; 2 * MAX_HIDDEN1 + 32];
        for i in 0..h1 {
            input[i] = acc.white_pre[i].clamp(0, IN_CLIP);
            input[h1 + i] = acc.black_pre[i].clamp(0, IN_CLIP);
        }
        if dense_size > 0 {
            let dense = compute_dense(bb, dense_size);
            for k in 0..dense_size {
                input[2 * h1 + k] = (dense[k] * Q_ACC).round() as i16;
            }
        }
        let input = &input[..in_pad];

        let inv_scale = 1.0 / (Q_ACC * Q_W2);
        let mut hidden = [0.0f32; MAX_HIDDEN2];
        let fc2_w = &self.fc2_weight_q[bucket * h2 * in_pad..(bucket + 1) * h2 * in_pad];
        let fc2_b = &self.fc2_bias[bucket * h2..(bucket + 1) * h2];
        for neuron in 0..h2 {
            let w = &fc2_w[neuron * in_pad..(neuron + 1) * in_pad];
            let mut sum = i32x8::ZERO;
            for (a, wc) in input.chunks_exact(16).zip(w.chunks_exact(16)) {
                sum += i16x16::from(<[i16; 16]>::try_from(a).unwrap())
                    .dot(i16x16::from(<[i16; 16]>::try_from(wc).unwrap()));
            }
            let s = sum.reduce_add() as f32 * inv_scale + fc2_b[neuron];
            hidden[neuron] = s.max(0.0);
        }
        self.fc3(bucket, &hidden)
    }

    /// f32 reference evaluation from scratch (no quantization). Used by tests
    /// to bound quantization error; not used in search.
    pub fn evaluate_reference(&self, bb: &BitBoard) -> i32 {
        let h1 = self.config.hidden1;
        let h2 = self.config.hidden2;
        let dense_size = self.config.dense_size;
        let fc2_in = 2 * h1 + dense_size;
        let bucket = self.config.output_bucket(bb);

        let mut white = vec![0.0f32; h1];
        let mut black = vec![0.0f32; h1];
        white.copy_from_slice(&self.fc1_bias[..h1]);
        black.copy_from_slice(&self.fc1_bias[..h1]);
        let mut feats = [0u16; MAX_ACTIVE_FEATURES];
        for (persp, target) in [(Player::White, &mut white), (Player::Black, &mut black)] {
            let n = self.config.active_features(bb, persp, &mut feats);
            for &f in &feats[..n] {
                let w = &self.fc1_weight_t[f as usize * h1..(f as usize + 1) * h1];
                for i in 0..h1 {
                    target[i] += w[i];
                }
            }
        }
        let dense = compute_dense(bb, dense_size);
        let mut hidden = [0.0f32; MAX_HIDDEN2];
        let fc2_w = &self.fc2_weight[bucket * h2 * fc2_in..(bucket + 1) * h2 * fc2_in];
        for neuron in 0..h2 {
            let w = &fc2_w[neuron * fc2_in..(neuron + 1) * fc2_in];
            let mut s = self.fc2_bias[bucket * h2 + neuron];
            for i in 0..h1 {
                s += white[i].max(0.0) * w[i] + black[i].max(0.0) * w[h1 + i];
            }
            for k in 0..dense_size {
                s += dense[k] * w[2 * h1 + k];
            }
            hidden[neuron] = s.max(0.0);
        }
        self.fc3(bucket, &hidden)
    }

    /// Convenience: quantized evaluation from scratch (no incremental state).
    pub fn evaluate_from_scratch(&self, bb: &BitBoard) -> i32 {
        let mut acc = DualAccumulator::default();
        self.init_accumulators(bb, &mut acc);
        self.evaluate(bb, &acc)
    }
}

fn transpose_embedding(rows: &[Vec<f32>], hidden1: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows.len() * hidden1];
    for (feat, row) in rows.iter().enumerate() {
        for (neuron, &w) in row.iter().enumerate().take(hidden1) {
            out[feat * hidden1 + neuron] = w;
        }
    }
    out
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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn all_configs() -> Vec<NnueConfig> {
        let mut v = Vec::new();
        for fs in [FeatureSet::HalfPail, FeatureSet::Plain] {
            for mirror in [false, true] {
                for buckets in [1usize, 25] {
                    v.push(NnueConfig {
                        feature_set: fs, mirror_black: mirror, dense_size: 20,
                        hidden1: 64, hidden2: 16, output_buckets: buckets,
                    });
                }
            }
        }
        v
    }

    /// Incremental updates must match from-scratch initialization along a
    /// random game, for every feature set / mirroring / bucket combination.
    #[test]
    fn test_incremental_matches_scratch() {
        for config in all_configs() {
            let net = SparseNNUE::random(config, 7);
            let mut state = 12345u64;
            let mut next = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (state >> 33) as usize };

            let mut bb = BitBoard::new();
            let mut acc = DualAccumulator::default();
            net.init_accumulators(&bb, &mut acc);

            for ply in 0..60 {
                let moves = bb.generate_moves();
                if moves.is_empty() || bb.check_winner().is_some() { break; }
                let mv = moves[next() % moves.len()];
                let before = bb;
                bb.make_move(&mv);
                net.update(&before, &bb, &mut acc);

                let mut fresh = DualAccumulator::default();
                net.init_accumulators(&bb, &mut fresh);
                let h = config.hidden1;
                // Integer accumulators: incremental must equal scratch exactly
                assert_eq!(&acc.white_pre[..h], &fresh.white_pre[..h], "{:?} ply {} white", config, ply);
                assert_eq!(&acc.black_pre[..h], &fresh.black_pre[..h], "{:?} ply {} black", config, ply);
                assert_eq!(net.evaluate(&bb, &acc), net.evaluate(&bb, &fresh));
            }
        }
    }

    /// Quantized evaluation must track the f32 reference closely along
    /// random games (bounded quantization error, no overflow).
    #[test]
    fn test_quantized_matches_reference() {
        for config in all_configs() {
            let net = SparseNNUE::random(config, 11);
            let mut state = 777u64;
            let mut next = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (state >> 33) as usize };
            let mut bb = BitBoard::new();
            let (mut max_diff, mut sum_diff, mut n) = (0i32, 0i64, 0i64);
            for _ in 0..80 {
                let moves = bb.generate_moves();
                if moves.is_empty() || bb.check_winner().is_some() { break; }
                bb.make_move(&moves[next() % moves.len()]);
                let q = net.evaluate_from_scratch(&bb);
                let r = net.evaluate_reference(&bb);
                let d = (q - r).abs();
                max_diff = max_diff.max(d);
                sum_diff += d as i64;
                n += 1;
            }
            assert!(max_diff <= 25, "{:?}: quantized vs reference max diff {} cp", config, max_diff);
            assert!(sum_diff as f64 / n as f64 <= 6.0, "{:?}: mean diff {:.2} cp", config, sum_diff as f64 / n as f64);
        }
    }

    /// Mirroring: with mirror_black, a color-swapped + vertically flipped
    /// position must produce identical perspective features (own/opp swap).
    #[test]
    fn test_mirror_symmetry() {
        let config = NnueConfig {
            feature_set: FeatureSet::Plain, mirror_black: true, dense_size: 0,
            hidden1: 64, hidden2: 16, output_buckets: 1,
        };
        let mut bb = BitBoard::new();
        bb.white_barrels = (1u64 << 20) | (1u64 << 8);
        bb.black_barrels = 1u64 << 27;
        bb.white_pail = 1u64 << 14;
        bb.white_pail_placed = true;
        bb.occupied = bb.white_barrels | bb.black_barrels | bb.white_pail;

        // Color-swap + flip
        let mut sw = BitBoard::new();
        sw.black_barrels = crate::race::mirror_rows(bb.white_barrels);
        sw.white_barrels = crate::race::mirror_rows(bb.black_barrels);
        sw.black_pail = crate::race::mirror_rows(bb.white_pail);
        sw.black_pail_placed = true;
        sw.occupied = sw.white_barrels | sw.black_barrels | sw.black_pail;

        let mut a = [0u16; MAX_ACTIVE_FEATURES];
        let mut b = [0u16; MAX_ACTIVE_FEATURES];
        let na = config.active_features(&bb, Player::White, &mut a);
        let nb = config.active_features(&sw, Player::Black, &mut b);
        let (mut va, mut vb) = (a[..na].to_vec(), b[..nb].to_vec());
        va.sort(); vb.sort();
        assert_eq!(va, vb, "white view of position must equal black view of its mirror");
    }

    #[test]
    fn test_dense164_roundtrip() {
        // Build a row from a real board and decode it back
        let mut bb = BitBoard::new();
        let mut state = 99u64;
        let mut next = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1); (state >> 33) as usize };
        for _ in 0..14 {
            let moves = bb.generate_moves();
            if moves.is_empty() { break; }
            let mv = moves[next() % moves.len()];
            bb.make_move(&mv);
        }
        let mut row = vec![0.0f32; 164];
        for sq in 0..NUM_SQUARES {
            if bb.white_barrels & (1 << sq) != 0 { row[sq * 4] = 1.0; }
            if bb.black_barrels & (1 << sq) != 0 { row[sq * 4 + 1] = 1.0; }
            if bb.white_pail & (1 << sq) != 0 { row[sq * 4 + 2] = 1.0; }
            if bb.black_pail & (1 << sq) != 0 { row[sq * 4 + 3] = 1.0; }
        }
        let rel = compute_relational_features(&bb);
        row[144..].copy_from_slice(&rel);
        let d = bitboard_from_dense164(&row);
        assert_eq!(d.white_barrels, bb.white_barrels);
        assert_eq!(d.black_barrels, bb.black_barrels);
        assert_eq!(d.white_pail, bb.white_pail);
        assert_eq!(d.black_pail, bb.black_pail);
        assert_eq!(d.white_scored, bb.white_scored);
        assert_eq!(d.black_scored, bb.black_scored);
        assert_eq!(d.current_player, bb.current_player);
        assert_eq!(d.white_barrels_off_board, bb.white_barrels_off_board);
    }
    #[test]
    fn test_dense_v2_extends_v1() {
        let bb = BitBoard::new();
        let d = compute_dense(&bb, DENSE_V2);
        let v1 = compute_relational_features(&bb);
        assert_eq!(&d[..HALFPAIL_DENSE], &v1[..]);
        for k in 0..DENSE_V2 { assert!(d[k] >= -1.0 && d[k] <= 1.0, "feature {} = {}", k, d[k]); }
        assert_eq!(compute_dense(&bb, 0), [0.0f32; MAX_DENSE]);
        assert_eq!(&compute_dense(&bb, HALFPAIL_DENSE)[..HALFPAIL_DENSE], &v1[..]);
        // start position: no jumps (nothing to jump over yet), clock 0, not mid-turn
        assert_eq!(d[22], 0.0); assert_eq!(d[23], 0.0);
        assert_eq!(d[26], 0.0); assert_eq!(d[27], 0.0);
        // race distances are symmetric at the start
        assert_eq!(d[20], d[21]);
        // after a few moves the features stay in range and jumps can appear
        let mut b = bb;
        let mut rng = 12345u64;
        let mut saw_jump = false;
        for _ in 0..40 {
            let moves = b.generate_moves();
            if moves.is_empty() || b.check_winner().is_some() { break; }
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            b.make_move(&moves[(rng % moves.len() as u64) as usize]);
            let d = compute_dense(&b, DENSE_V2);
            for k in 0..DENSE_V2 { assert!(d[k] >= -1.0 && d[k] <= 1.0); }
            if d[22] > 0.0 || d[23] > 0.0 { saw_jump = true; }
        }
        assert!(saw_jump, "expected some forward jump availability in a random game");
    }

    /// Bit-parallel jump/step origins must equal a plain per-barrel scan.
    #[test]
    fn test_forward_jump_step_bitparallel_matches_loop() {
        fn loop_version(barrels: u64, occ: u64, enemy_pail: u64, dir: i32) -> (u32, u32) {
            let n = BOARD_SIZE as i32;
            let (mut jumps, mut steps) = (0u32, 0u32);
            let mut m = barrels;
            while m != 0 {
                let sq = m.trailing_zeros() as i32; m &= m - 1;
                let (r, c) = (sq / n, sq % n);
                let (mut cj, mut cs) = (false, false);
                for dc in -1i32..=1 {
                    let (r1, c1) = (r + dir, c + dc);
                    if r1 < 0 || r1 >= n || c1 < 0 || c1 >= n { continue; }
                    let s1 = 1u64 << (r1 * n + c1);
                    if occ & s1 == 0 { cs = true; continue; }
                    if enemy_pail & s1 != 0 { continue; }
                    let (r2, c2) = (r + 2 * dir, c + 2 * dc);
                    if r2 < 0 || r2 >= n || c2 < 0 || c2 >= n { continue; }
                    if occ & (1u64 << (r2 * n + c2)) == 0 { cj = true; }
                }
                jumps += cj as u32; steps += cs as u32;
            }
            (jumps, steps)
        }
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        for _ in 0..20000 {
            // random sparse occupancy: up to 4 barrels per side, pails
            let mut w = 0u64; let mut b = 0u64;
            for _ in 0..4 { w |= 1u64 << (next() % 36); }
            for _ in 0..4 { let s = 1u64 << (next() % 36); if w & s == 0 { b |= s; } }
            let mut wp = 1u64 << (next() % 36); if (w | b) & wp != 0 { wp = 0; }
            let mut bp = 1u64 << (next() % 36); if (w | b | wp) & bp != 0 { bp = 0; }
            let occ = w | b | wp | bp;
            for (side, dir, ep) in [(w, -1i32, bp), (b, 1i32, wp)] {
                let (j, s) = forward_jump_step_origins(side, occ, ep, dir);
                let (lj, ls) = loop_version(side, occ, ep, dir);
                assert_eq!((j.count_ones(), s.count_ones()), (lj, ls), "w={:x} b={:x} wp={:x} bp={:x} dir={}", w, b, wp, bp, dir);
                assert_eq!(j & !side, 0); assert_eq!(s & !side, 0);
            }
        }
    }

}
