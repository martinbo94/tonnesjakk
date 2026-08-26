//! Endgame tablebases, one phase at a time.
//!
//! A phase is (white barrels remaining, black barrels remaining) where
//! remaining = on board + in hand = 4 - scored. Scoring is irreversible, so a
//! phase depends only on phases with fewer barrels: solve small phases first
//! and look their values up when a move scores.
//!
//! Within a phase the game is loopy (barrels move back and forth), so values
//! come from retrograde iteration to a fixpoint, processed in distance order
//! so the stored distance-to-win is exact: pass p assigns "win in p" to
//! states with a child that is "loss in p-1" for the opponent, and "loss in
//! p" to states all of whose children are opponent wins with the longest
//! being p-1. States never assigned are draws (neither side can force a win;
//! under threefold repetition these are the shuffle-forever positions).
//!
//! State = (white cfg, black cfg, white pail, black pail, side to move,
//! awaiting_barrel). Configs enumerate subsets of size 0..=remaining over the
//! 30 squares a colour may occupy (never its own goal row); the in-hand count
//! is implied. Values are one byte per state, White's perspective.

use std::path::Path;

use crate::board::*;

pub const V_UNKNOWN: u8 = 0;
pub const V_DRAW: u8 = 128;
pub const V_INVALID: u8 = 255;
const MAX_DIST: u8 = 120;

#[inline]
pub fn v_white_win(d: u8) -> u8 { d.min(MAX_DIST) + 1 }          // 1..=121
#[inline]
pub fn v_black_win(d: u8) -> u8 { 129 + d.min(MAX_DIST) }        // 129..=249
#[inline]
pub fn is_white_win(v: u8) -> bool { (1..=121).contains(&v) }
#[inline]
pub fn is_black_win(v: u8) -> bool { (129..=249).contains(&v) }
#[inline]
pub fn win_dist(v: u8) -> u8 { if is_white_win(v) { v - 1 } else { v - 129 } }

const SQUARES_PER_COLOR: usize = 30;
const PAIL_STATES: usize = NUM_SQUARES + 1; // 36 squares + "not placed"

/// Binomial coefficients C[n][k] for n <= 30, k <= 4
fn binom(n: usize, k: usize) -> u64 {
    if k > n { return 0; }
    let mut r = 1u64;
    for i in 0..k {
        r = r * (n - i) as u64 / (i + 1) as u64;
    }
    r
}

/// Per-colour config space: subsets of size 0..=rem over 30 legal squares.
struct ConfigSpace {
    rem: usize,
    /// offset[k] = number of configs with fewer than k barrels
    offset: [u64; 5],
    count: u64,
    /// legal squares for this colour in ascending order
    squares: [u8; SQUARES_PER_COLOR],
    /// square -> position in `squares` (or 255)
    pos_of: [u8; NUM_SQUARES],
}

impl ConfigSpace {
    fn new(player: Player, rem: usize) -> Self {
        let goal_row = match player { Player::White => 0, Player::Black => BOARD_SIZE - 1 };
        let mut squares = [0u8; SQUARES_PER_COLOR];
        let mut pos_of = [255u8; NUM_SQUARES];
        let mut n = 0;
        for sq in 0..NUM_SQUARES {
            if sq / BOARD_SIZE != goal_row {
                squares[n] = sq as u8;
                pos_of[sq] = n as u8;
                n += 1;
            }
        }
        let mut offset = [0u64; 5];
        let mut acc = 0u64;
        for k in 0..=rem {
            offset[k] = acc;
            acc += binom(SQUARES_PER_COLOR, k);
        }
        ConfigSpace { rem, offset, count: acc, squares, pos_of }
    }

    /// Rank a barrel mask (combinatorial number system).
    fn rank(&self, mask: u64) -> Option<u64> {
        let k = mask.count_ones() as usize;
        if k > self.rem { return None; }
        let mut r = 0u64;
        let mut m = mask;
        let mut i = 0usize;
        while m != 0 {
            let sq = m.trailing_zeros() as usize;
            let p = self.pos_of[sq];
            if p == 255 { return None; }
            i += 1;
            r += binom(p as usize, i);
            m &= m - 1;
        }
        Some(self.offset[k] + r)
    }

    /// Inverse of rank.
    fn unrank(&self, idx: u64) -> u64 {
        let mut k = 0usize;
        while k + 1 <= self.rem && idx >= self.offset[k + 1] { k += 1; }
        let mut r = idx - self.offset[k];
        let mut mask = 0u64;
        let mut kk = k;
        while kk > 0 {
            // largest p with C(p, kk) <= r
            let mut p = kk - 1;
            while p + 1 < SQUARES_PER_COLOR && binom(p + 1, kk) <= r { p += 1; }
            r -= binom(p, kk);
            mask |= 1u64 << self.squares[p];
            kk -= 1;
        }
        mask
    }
}

/// One solved (or being solved) phase.
pub struct Phase {
    pub wr: usize,
    pub br: usize,
    wcfg: ConfigSpace,
    bcfg: ConfigSpace,
    pub values: Vec<u8>,
}

impl Phase {
    pub fn new(wr: usize, br: usize) -> Self {
        let wcfg = ConfigSpace::new(Player::White, wr);
        let bcfg = ConfigSpace::new(Player::Black, br);
        let n = (wcfg.count * bcfg.count) as usize * PAIL_STATES * PAIL_STATES * 4;
        Phase { wr, br, wcfg, bcfg, values: vec![V_UNKNOWN; n] }
    }

    pub fn num_states(&self) -> usize { self.values.len() }

    #[inline]
    fn pail_index(pail: u64) -> usize {
        if pail == 0 { 0 } else { pail.trailing_zeros() as usize + 1 }
    }

    /// Index of a position in this phase (None if it doesn't belong here or is
    /// not representable: wrong scored counts, overlapping pieces).
    pub fn index(&self, bb: &BitBoard) -> Option<usize> {
        if (4 - bb.white_scored as usize) != self.wr || (4 - bb.black_scored as usize) != self.br {
            return None;
        }
        let wi = self.wcfg.rank(bb.white_barrels)?;
        let bi = self.bcfg.rank(bb.black_barrels)?;
        let stm = if bb.current_player == Player::White { 0 } else { 1 };
        let aw = bb.awaiting_barrel as usize;
        let idx = ((((wi * self.bcfg.count + bi) as usize * PAIL_STATES + Self::pail_index(bb.white_pail))
            * PAIL_STATES + Self::pail_index(bb.black_pail)) * 2 + stm) * 2 + aw;
        Some(idx)
    }

    /// Reconstruct the position at an index (None if the index is an invalid
    /// combination: overlapping pieces, awaiting without a placed pail, ...).
    pub fn decode(&self, idx: usize) -> Option<BitBoard> {
        let aw = idx % 2;
        let stm = (idx / 2) % 2;
        let bp = (idx / 4) % PAIL_STATES;
        let wp = (idx / (4 * PAIL_STATES)) % PAIL_STATES;
        let cfg = (idx / (4 * PAIL_STATES * PAIL_STATES)) as u64;
        let wi = cfg / self.bcfg.count;
        let bi = cfg % self.bcfg.count;
        let wmask = self.wcfg.unrank(wi);
        let bmask = self.bcfg.unrank(bi);
        let wpail = if wp == 0 { 0 } else { 1u64 << (wp - 1) };
        let bpail = if bp == 0 { 0 } else { 1u64 << (bp - 1) };
        let occ = [wmask, bmask, wpail, bpail];
        // pieces must not overlap
        let mut all = 0u64;
        for o in occ {
            if all & o != 0 { return None; }
            all |= o;
        }
        let player = if stm == 0 { Player::White } else { Player::Black };
        // awaiting_barrel means the side to move has just placed its pail
        let own_pail = if stm == 0 { wpail } else { bpail };
        if aw == 1 && own_pail == 0 { return None; }

        let mut bb = BitBoard::new();
        bb.white_barrels = wmask;
        bb.black_barrels = bmask;
        bb.white_pail = wpail;
        bb.black_pail = bpail;
        bb.white_pail_placed = wpail != 0;
        bb.black_pail_placed = bpail != 0;
        bb.occupied = all;
        bb.white_scored = (4 - self.wr) as u8;
        bb.black_scored = (4 - self.br) as u8;
        bb.white_barrels_off_board = (self.wr - wmask.count_ones() as usize) as u8;
        bb.black_barrels_off_board = (self.br - bmask.count_ones() as usize) as u8;
        bb.current_player = player;
        bb.awaiting_barrel = aw == 1;
        Some(bb)
    }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(format!("tb_{}v{}.bin", self.wr, self.br)), &self.values)
    }

    pub fn load(dir: &Path, wr: usize, br: usize) -> std::io::Result<Self> {
        let mut p = Phase::new(wr, br);
        let bytes = std::fs::read(dir.join(format!("tb_{}v{}.bin", wr, br)))?;
        if bytes.len() != p.values.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "tablebase size mismatch"));
        }
        p.values = bytes;
        Ok(p)
    }
}

/// Collection of solved phases, indexed by (wr, br).
#[derive(Default)]
pub struct Tablebase {
    phases: Vec<Option<Phase>>, // index wr*5 + br
}

impl Tablebase {
    pub fn new() -> Self { Tablebase { phases: (0..25).map(|_| None).collect() } }

    pub fn insert(&mut self, phase: Phase) {
        let slot = phase.wr * 5 + phase.br;
        self.phases[slot] = Some(phase);
    }

    pub fn get(&self, wr: usize, br: usize) -> Option<&Phase> {
        self.phases.get(wr * 5 + br).and_then(|p| p.as_ref())
    }

    pub fn loaded_phases(&self) -> Vec<(usize, usize)> {
        self.phases.iter().flatten().map(|p| (p.wr, p.br)).collect()
    }

    /// Value of a position from White's perspective, if its phase is solved.
    /// Terminal positions (a side scored everything) are answered directly.
    pub fn value(&self, bb: &BitBoard) -> Option<u8> {
        if bb.white_scored == 4 { return Some(v_white_win(0)); }
        if bb.black_scored == 4 { return Some(v_black_win(0)); }
        let wr = 4 - bb.white_scored as usize;
        let br = 4 - bb.black_scored as usize;
        let phase = self.get(wr, br)?;
        let idx = phase.index(bb)?;
        let v = phase.values[idx];
        if v == V_INVALID || v == V_UNKNOWN { None } else { Some(v) }
    }

    /// Load every phase file present in `dir`.
    pub fn load_dir(dir: &Path) -> std::io::Result<Self> {
        let mut tb = Tablebase::new();
        for wr in 1..=4 {
            for br in 1..=4 {
                if dir.join(format!("tb_{}v{}.bin", wr, br)).exists() {
                    tb.insert(Phase::load(dir, wr, br)?);
                }
            }
        }
        Ok(tb)
    }

    /// Solve phase (wr, br). All phases it can score into ((wr-1, br) and
    /// (wr, br-1), unless terminal) must already be present.
    pub fn solve(&mut self, wr: usize, br: usize, verbose: bool) -> &Phase {
        let mut phase = Phase::new(wr, br);
        let n = phase.num_states();

        // Mark invalid states; count valid
        let mut valid = 0usize;
        for idx in 0..n {
            if phase.decode(idx).is_none() {
                phase.values[idx] = V_INVALID;
            } else {
                valid += 1;
            }
        }
        if verbose {
            eprintln!("phase {}v{}: {} states, {} valid", wr, br, n, valid);
        }

        // Child value lookup: same phase (current array) or lower phases (self)
        let child_value = |phase: &Phase, child: &BitBoard| -> u8 {
            if child.white_scored as usize > 4 - wr || child.black_scored as usize > 4 - br {
                self.value(child).unwrap_or(V_DRAW) // lower phase (must be loaded)
            } else {
                phase.values[phase.index(child).expect("child in phase")]
            }
        };

        let mut pass: u8 = 1;
        let mut unknown: Vec<usize> = (0..n).filter(|&i| phase.values[i] == V_UNKNOWN).collect();
        loop {
            let mut assigned: Vec<(usize, u8)> = Vec::new();
            let mut still_unknown = Vec::with_capacity(unknown.len());
            for &idx in &unknown {
                let bb = phase.decode(idx).unwrap();
                let stm_white = bb.current_player == Player::White;
                let moves = bb.generate_moves();
                let mut best_win_child: Option<u8> = None;   // min dist among children that are wins for stm
                let mut all_lose = !moves.is_empty();
                let mut max_lose_dist: u8 = 0;
                for mv in &moves {
                    let mut child = bb;
                    child.make_move(mv);
                    let v = child_value(&phase, &child);
                    let stm_wins = if stm_white { is_white_win(v) } else { is_black_win(v) };
                    let stm_loses = if stm_white { is_black_win(v) } else { is_white_win(v) };
                    if stm_wins {
                        let d = win_dist(v);
                        best_win_child = Some(best_win_child.map_or(d, |b| b.min(d)));
                    }
                    if stm_loses {
                        max_lose_dist = max_lose_dist.max(win_dist(v));
                    } else {
                        all_lose = false;
                    }
                }
                let mut value = V_UNKNOWN;
                if let Some(d) = best_win_child {
                    if d + 1 == pass {
                        value = if stm_white { v_white_win(pass) } else { v_black_win(pass) };
                    }
                }
                if value == V_UNKNOWN && all_lose && max_lose_dist + 1 == pass {
                    value = if stm_white { v_black_win(pass) } else { v_white_win(pass) };
                }
                if value == V_UNKNOWN && moves.is_empty() {
                    // No legal move: treat as draw (cannot happen with barrels in hand;
                    // fully boxed-in positions are neither side's win here).
                    value = V_DRAW;
                }
                if value != V_UNKNOWN {
                    assigned.push((idx, value));
                } else {
                    still_unknown.push(idx);
                }
            }
            for &(idx, v) in &assigned {
                phase.values[idx] = v;
            }
            if verbose {
                eprintln!("  pass {:3}: assigned {:9}, unknown {:10}", pass, assigned.len(), still_unknown.len());
            }
            unknown = still_unknown;
            if assigned.is_empty() || unknown.is_empty() || pass >= MAX_DIST {
                break;
            }
            pass += 1;
        }
        for &idx in &unknown {
            phase.values[idx] = V_DRAW;
        }
        self.insert(phase);
        self.get(wr, br).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_unrank_roundtrip() {
        for rem in 0..=4 {
            let cs = ConfigSpace::new(Player::White, rem);
            for idx in 0..cs.count {
                let mask = cs.unrank(idx);
                assert_eq!(cs.rank(mask), Some(idx), "rem {} idx {}", rem, idx);
                assert!(mask & ROW_MASK[0] == 0, "white never on goal row");
            }
        }
        let cs = ConfigSpace::new(Player::Black, 4);
        assert_eq!(cs.count, 1 + 30 + 435 + 4060 + 27405);
    }

    #[test]
    fn test_phase_index_roundtrip() {
        let phase = Phase::new(1, 1);
        let mut seen = 0;
        for idx in 0..phase.num_states() {
            if let Some(bb) = phase.decode(idx) {
                assert_eq!(phase.index(&bb), Some(idx));
                seen += 1;
            }
        }
        assert!(seen > 1_000_000);
    }

    /// 1v1: solve, then spot-check against exhaustive search on a few states.
    #[test]
    fn test_solve_1v1_matches_search() {
        let mut tb = Tablebase::new();
        let phase = tb.solve(1, 1, false);
        let n_draw = phase.values.iter().filter(|&&v| v == V_DRAW).count();
        let n_w = phase.values.iter().filter(|&&v| is_white_win(v)).count();
        let n_b = phase.values.iter().filter(|&&v| is_black_win(v)).count();
        assert!(n_w > 0 && n_b > 0, "both sides should have wins in 1v1 ({} / {} / {} draws)", n_w, n_b, n_draw);

        // Position: white barrel one step from goal, white to move, black barrel far: white wins in 1
        let mut bb = BitBoard::new();
        bb.white_barrels = 1u64 << sq(1, 2);
        bb.black_barrels = 1u64 << sq(1, 4);
        bb.occupied = bb.white_barrels | bb.black_barrels;
        bb.white_scored = 3; bb.black_scored = 3; // both pails still in hand
        bb.white_barrels_off_board = 0; bb.black_barrels_off_board = 0;
        let v = tb.value(&bb).unwrap();
        assert!(is_white_win(v) && win_dist(v) == 1, "expected white win in 1, got {}", v);
    }
}
