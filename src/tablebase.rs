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

/// One solved (or being solved) phase. Values live either in an owned Vec
/// (while solving) or in a read-only memory map of the phase file (when
/// loaded by engines): the OS page cache then shares one copy across all
/// worker processes, so ten data-gen workers loading 2v2 cost 1.2 GB total,
/// not 12.
pub struct Phase {
    pub wr: usize,
    pub br: usize,
    wcfg: ConfigSpace,
    bcfg: ConfigSpace,
    pub values: Vec<u8>,
    mapped: Option<memmap2::Mmap>,
    num_states: usize,
}

impl Phase {
    pub fn new(wr: usize, br: usize) -> Self {
        let wcfg = ConfigSpace::new(Player::White, wr);
        let bcfg = ConfigSpace::new(Player::Black, br);
        let n = (wcfg.count * bcfg.count) as usize * PAIL_STATES * PAIL_STATES * 4;
        Phase { wr, br, wcfg, bcfg, values: vec![V_UNKNOWN; n], mapped: None, num_states: n }
    }

    pub fn num_states(&self) -> usize { self.num_states }

    /// Read access to values regardless of storage.
    #[inline]
    pub fn vals(&self) -> &[u8] {
        match &self.mapped {
            Some(m) => &m[..],
            None => &self.values,
        }
    }

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
        std::fs::write(dir.join(format!("tb_{}v{}.bin", self.wr, self.br)), self.vals())
    }

    /// Memory-map a solved phase file (read-only, shared across processes).
    pub fn load(dir: &Path, wr: usize, br: usize) -> std::io::Result<Self> {
        let wcfg = ConfigSpace::new(Player::White, wr);
        let bcfg = ConfigSpace::new(Player::Black, br);
        let n = (wcfg.count * bcfg.count) as usize * PAIL_STATES * PAIL_STATES * 4;
        let file = std::fs::File::open(dir.join(format!("tb_{}v{}.bin", wr, br)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        if mmap.len() != n {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "tablebase size mismatch"));
        }
        Ok(Phase { wr, br, wcfg, bcfg, values: Vec::new(), mapped: Some(mmap), num_states: n })
    }
}

/// The colour-swapped, vertically flipped position: White's pieces become
/// Black's on mirrored rows and vice versa, side to move swaps. The value of
/// the mirrored position is the negation of the original's.
pub fn color_mirror(bb: &BitBoard) -> BitBoard {
    let m = crate::race::mirror_rows;
    let mut out = BitBoard::new();
    out.white_barrels = m(bb.black_barrels);
    out.black_barrels = m(bb.white_barrels);
    out.white_pail = m(bb.black_pail);
    out.black_pail = m(bb.white_pail);
    out.white_pail_placed = bb.black_pail_placed;
    out.black_pail_placed = bb.white_pail_placed;
    out.occupied = out.white_barrels | out.black_barrels | out.white_pail | out.black_pail;
    out.white_scored = bb.black_scored;
    out.black_scored = bb.white_scored;
    out.white_barrels_off_board = bb.black_barrels_off_board;
    out.black_barrels_off_board = bb.white_barrels_off_board;
    out.current_player = match bb.current_player {
        Player::White => Player::Black,
        Player::Black => Player::White,
    };
    out.awaiting_barrel = bb.awaiting_barrel;
    out
}

// ============================================================================
// PACKED SYMMETRIC PHASES (r v r): 2-bit WDL, white-to-move only
// ============================================================================
//
// 3v3 has 1.1e11 raw states — 112 GB at a byte each, more than RAM. Two
// reductions make it fit: (1) in a symmetric phase every black-to-move state
// is the colour mirror of a white-to-move state, so only white-to-move states
// are stored (÷2); (2) 2 bits per state — win / loss / draw-or-unknown /
// invalid — instead of a byte with distance (÷4). 3v3 → 14 GB. Without
// distances the solver is plain fixpoint iteration (win if any child is a
// loss for the opponent, loss if all children are wins for the opponent),
// which converges to the same WDL as the distance-ordered solver.

pub const P_UNKNOWN: u8 = 0; // draw once solving has finished
pub const P_WHITE: u8 = 1;
pub const P_BLACK: u8 = 2;
pub const P_INVALID: u8 = 3;

pub struct PackedPhase {
    pub r: usize,
    wcfg: ConfigSpace,
    bcfg: ConfigSpace,
    /// 2 bits per white-to-move state
    data: Vec<u8>,
    mapped: Option<memmap2::Mmap>,
    n_states: usize,
}

impl PackedPhase {
    pub fn new(r: usize) -> Self {
        let wcfg = ConfigSpace::new(Player::White, r);
        let bcfg = ConfigSpace::new(Player::Black, r);
        let n = (wcfg.count * bcfg.count) as usize * PAIL_STATES * PAIL_STATES * 2; // awaiting flag only
        PackedPhase { r, wcfg, bcfg, data: vec![0u8; (n + 3) / 4], mapped: None, n_states: n }
    }

    pub fn num_states(&self) -> usize { self.n_states }

    #[inline]
    fn bytes(&self) -> &[u8] {
        match &self.mapped { Some(m) => &m[..], None => &self.data }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> u8 {
        (self.bytes()[idx >> 2] >> ((idx & 3) * 2)) & 3
    }

    #[inline]
    fn set(data: &mut [u8], idx: usize, v: u8) {
        let shift = (idx & 3) * 2;
        let b = &mut data[idx >> 2];
        *b = (*b & !(3 << shift)) | (v << shift);
    }

    /// Index of a WHITE-TO-MOVE position in this phase.
    pub fn index(&self, bb: &BitBoard) -> Option<usize> {
        if bb.current_player != Player::White { return None; }
        if (4 - bb.white_scored as usize) != self.r || (4 - bb.black_scored as usize) != self.r {
            return None;
        }
        let wi = self.wcfg.rank(bb.white_barrels)?;
        let bi = self.bcfg.rank(bb.black_barrels)?;
        Some((((wi * self.bcfg.count + bi) as usize * PAIL_STATES + Phase::pail_index(bb.white_pail))
            * PAIL_STATES + Phase::pail_index(bb.black_pail)) * 2 + bb.awaiting_barrel as usize)
    }

    pub fn decode(&self, idx: usize) -> Option<BitBoard> {
        let aw = idx % 2;
        let bp = (idx / 2) % PAIL_STATES;
        let wp = (idx / (2 * PAIL_STATES)) % PAIL_STATES;
        let cfg = (idx / (2 * PAIL_STATES * PAIL_STATES)) as u64;
        let wmask = self.wcfg.unrank(cfg / self.bcfg.count);
        let bmask = self.bcfg.unrank(cfg % self.bcfg.count);
        let wpail = if wp == 0 { 0 } else { 1u64 << (wp - 1) };
        let bpail = if bp == 0 { 0 } else { 1u64 << (bp - 1) };
        let mut all = 0u64;
        for o in [wmask, bmask, wpail, bpail] {
            if all & o != 0 { return None; }
            all |= o;
        }
        if aw == 1 && wpail == 0 { return None; } // white awaiting ⇒ white pail placed
        let mut bb = BitBoard::new();
        bb.white_barrels = wmask;
        bb.black_barrels = bmask;
        bb.white_pail = wpail;
        bb.black_pail = bpail;
        bb.white_pail_placed = wpail != 0;
        bb.black_pail_placed = bpail != 0;
        bb.occupied = all;
        bb.white_scored = (4 - self.r) as u8;
        bb.black_scored = (4 - self.r) as u8;
        bb.white_barrels_off_board = (self.r - wmask.count_ones() as usize) as u8;
        bb.black_barrels_off_board = (self.r - bmask.count_ones() as usize) as u8;
        bb.current_player = Player::White;
        bb.awaiting_barrel = aw == 1;
        Some(bb)
    }

    /// WDL value (White perspective, as a Phase-style byte with distance 0)
    /// for any side to move: black-to-move goes through the colour mirror.
    pub fn value(&self, bb: &BitBoard) -> Option<u8> {
        let (pos, flip) = if bb.current_player == Player::White { (*bb, false) } else { (color_mirror(bb), true) };
        let v = self.get(self.index(&pos)?);
        match (v, flip) {
            (P_INVALID, _) => None,
            (P_UNKNOWN, _) => Some(V_DRAW),
            (P_WHITE, false) | (P_BLACK, true) => Some(v_white_win(0)),
            _ => Some(v_black_win(0)),
        }
    }

    fn file_name(r: usize) -> String { format!("tb_{}v{}.p2", r, r) }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(Self::file_name(self.r)), self.bytes())
    }

    pub fn load(dir: &Path, r: usize) -> std::io::Result<Self> {
        let mut p = PackedPhase::new(r);
        let expected = p.data.len();
        p.data = Vec::new();
        let file = std::fs::File::open(dir.join(Self::file_name(r)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        if mmap.len() != expected {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "packed tablebase size mismatch"));
        }
        p.mapped = Some(mmap);
        Ok(p)
    }
}

/// Plain fixpoint solve of a symmetric phase into the packed format.
/// `checkpoint`: write the partial array to `dir` every N passes and resume
/// from it if present (long solves survive a stop).
pub fn solve_packed(lower: &Tablebase, r: usize, dir: &Path, checkpoint_every: usize, verbose: bool) -> PackedPhase {
    let mut phase = PackedPhase::new(r);
    let n = phase.num_states();
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).max(1);
    let chunk_states = ((n + threads - 1) / threads + 3) / 4 * 4; // multiple of 4 → byte-aligned chunks
    let t_start = std::time::Instant::now();
    let ckpt_path = dir.join(format!("tb_{}v{}.p2.partial", r, r));

    let mut data = std::mem::take(&mut phase.data);
    let resumed = if ckpt_path.exists() {
        match std::fs::read(&ckpt_path) {
            Ok(bytes) if bytes.len() == data.len() => { data = bytes; true }
            _ => false,
        }
    } else { false };

    if !resumed {
        // Mark invalid states in parallel (byte-aligned chunks)
        std::thread::scope(|s| {
            let phase = &phase;
            for (ci, slice) in data.chunks_mut(chunk_states / 4).enumerate() {
                s.spawn(move || {
                    let base = ci * chunk_states;
                    for i in 0..slice.len() * 4 {
                        let idx = base + i;
                        if idx >= n { break; }
                        if phase.decode(idx).is_none() {
                            PackedPhase::set(slice, i, P_INVALID);
                        }
                    }
                });
            }
        });
    }
    let valid = data.iter().map(|b| (0..4).filter(|k| (b >> (k * 2)) & 3 != P_INVALID).count()).sum::<usize>().min(n);
    if verbose {
        eprintln!("packed phase {}v{}: {} states (white to move), {} valid, {} threads, resumed={} ({:.0}s)",
                  r, r, n, valid, threads, resumed, t_start.elapsed().as_secs_f64());
    }

    let mut pass = 0usize;
    loop {
        pass += 1;
        // Read-only snapshot for this pass; collect assignments per thread
        let results: Vec<Vec<(usize, u8)>> = std::thread::scope(|s| {
            let phase = &phase;
            let snap: &[u8] = &data;
            let handles: Vec<_> = (0..threads).map(|ci| {
                s.spawn(move || {
                    let lo = ci * chunk_states;
                    let hi = ((ci + 1) * chunk_states).min(n);
                    let mut out = Vec::new();
                    let get = |idx: usize| (snap[idx >> 2] >> ((idx & 3) * 2)) & 3;
                    for idx in lo..hi {
                        if get(idx) != P_UNKNOWN { continue; }
                        let bb = match phase.decode(idx) { Some(b) => b, None => continue };
                        let moves = bb.generate_moves();
                        if moves.is_empty() { continue; } // boxed in: stays unknown = draw
                        let mut any_win = false;
                        let mut all_lose = true;
                        for mv in &moves {
                            let mut child = bb;
                            child.make_move(mv);
                            // White perspective value of the child
                            let v: Option<u8> = if child.white_scored as usize > 4 - r || child.black_scored as usize > 4 - r {
                                lower.value(&child)
                            } else if child.current_player == Player::White {
                                // white just made a pail sub-move: still white to move
                                match get(phase.index(&child).unwrap()) { P_WHITE => Some(v_white_win(0)), P_BLACK => Some(v_black_win(0)), _ => None }
                            } else {
                                let m = color_mirror(&child);
                                match get(phase.index(&m).unwrap()) { P_WHITE => Some(v_black_win(0)), P_BLACK => Some(v_white_win(0)), _ => None }
                            };
                            match v {
                                Some(x) if is_white_win(x) => { any_win = true; break; }
                                Some(x) if is_black_win(x) => {}
                                _ => { all_lose = false; }
                            }
                        }
                        if any_win { out.push((idx, P_WHITE)); }
                        else if all_lose { out.push((idx, P_BLACK)); }
                    }
                    out
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let n_assigned: usize = results.iter().map(|a| a.len()).sum();
        for a in &results {
            for &(idx, v) in a {
                PackedPhase::set(&mut data, idx, v);
            }
        }
        if verbose {
            eprintln!("  pass {:3}: assigned {:10} ({:.0}s)", pass, n_assigned, t_start.elapsed().as_secs_f64());
        }
        if checkpoint_every > 0 && pass % checkpoint_every == 0 {
            // Write-then-rename so a reboot mid-write leaves the previous
            // checkpoint intact instead of a truncated file (which the
            // length check on resume would reject, restarting from scratch).
            let tmp = ckpt_path.with_extension("partial.tmp");
            if std::fs::write(&tmp, &data).is_ok() {
                let _ = std::fs::rename(&tmp, &ckpt_path);
            }
        }
        if n_assigned == 0 { break; }
    }
    let _ = std::fs::remove_file(&ckpt_path);
    phase.data = data;
    phase
}

/// Collection of solved phases, indexed by (wr, br).
#[derive(Default)]
pub struct Tablebase {
    phases: Vec<Option<Phase>>, // index wr*5 + br
    packed: Vec<Option<PackedPhase>>, // index r (symmetric phases)
}

impl Tablebase {
    pub fn new() -> Self {
        Tablebase { phases: (0..25).map(|_| None).collect(), packed: (0..5).map(|_| None).collect() }
    }

    pub fn insert_packed(&mut self, p: PackedPhase) {
        let r = p.r;
        self.packed[r] = Some(p);
    }

    pub fn insert(&mut self, phase: Phase) {
        let slot = phase.wr * 5 + phase.br;
        self.phases[slot] = Some(phase);
    }

    pub fn get(&self, wr: usize, br: usize) -> Option<&Phase> {
        self.phases.get(wr * 5 + br).and_then(|p| p.as_ref())
    }

    pub fn loaded_phases(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = self.phases.iter().flatten().map(|p| (p.wr, p.br)).collect();
        v.extend(self.packed.iter().flatten().map(|p| (p.r, p.r)));
        v.sort();
        v
    }

    /// Value of a position from White's perspective, if its phase is solved.
    /// Terminal positions (a side scored everything) are answered directly.
    /// Phase (a, b) can also be answered from (b, a) by the game's colour
    /// symmetry: swap colours, flip the board vertically, swap side to move,
    /// negate the value — so only one of each mirrored pair needs solving.
    pub fn value(&self, bb: &BitBoard) -> Option<u8> {
        if bb.white_scored == 4 { return Some(v_white_win(0)); }
        if bb.black_scored == 4 { return Some(v_black_win(0)); }
        let wr = 4 - bb.white_scored as usize;
        let br = 4 - bb.black_scored as usize;
        if let Some(phase) = self.get(wr, br) {
            let idx = phase.index(bb)?;
            let v = phase.vals()[idx];
            return if v == V_INVALID || v == V_UNKNOWN { None } else { Some(v) };
        }
        if wr == br {
            if let Some(p) = self.packed.get(wr).and_then(|p| p.as_ref()) {
                return p.value(bb);
            }
        }
        let mirror_phase = self.get(br, wr)?;
        let m = color_mirror(bb);
        let idx = mirror_phase.index(&m)?;
        let v = mirror_phase.vals()[idx];
        if v == V_INVALID || v == V_UNKNOWN {
            None
        } else if is_white_win(v) {
            Some(v_black_win(win_dist(v)))
        } else if is_black_win(v) {
            Some(v_white_win(win_dist(v)))
        } else {
            Some(v)
        }
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
        for r in 1..=4 {
            if dir.join(PackedPhase::file_name(r)).exists() && tb.get(r, r).is_none() {
                tb.insert_packed(PackedPhase::load(dir, r)?);
            }
        }
        Ok(tb)
    }

    /// Solve phase (wr, br). All phases it can score into ((wr-1, br) and
    /// (wr, br-1), unless terminal) must already be present.
    pub fn solve(&mut self, wr: usize, br: usize, verbose: bool) -> &Phase {
        let phase = solve_phase(self, wr, br, verbose);
        self.insert(phase);
        self.get(wr, br).unwrap()
    }
}

/// Result of examining one unknown state at a given pass.
enum Verdict {
    /// Determined now.
    Value(u8),
    /// Not due yet; would be determined at this (later) pass if nothing changes.
    Pending(u8),
    /// No determined children at all (a draw unless children change).
    Open,
}

/// Evaluate one unknown state at `pass` (distance being assigned).
#[inline]
fn determine(lower: &Tablebase, phase: &Phase, values: &[u8], bb: &BitBoard, pass: u8) -> Verdict {
    let wr = phase.wr;
    let br = phase.br;
    let stm_white = bb.current_player == Player::White;
    let moves = bb.generate_moves();
    if moves.is_empty() {
        // Fully boxed in with nothing in hand: neither side can force a win here.
        return Verdict::Value(V_DRAW);
    }
    let mut best_win_child: Option<u8> = None; // min dist among children that are wins for stm
    let mut all_lose = true;
    let mut max_lose_dist: u8 = 0;
    for mv in &moves {
        let mut child = *bb;
        child.make_move(mv);
        let v = if child.white_scored as usize > 4 - wr || child.black_scored as usize > 4 - br {
            lower.value(&child).unwrap_or(V_DRAW) // lower phase (must be loaded)
        } else {
            values[phase.index(&child).expect("child in phase")]
        };
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
    // Wins are determined at distance (closest winning child + 1); a state
    // with a winning child is never a loss, so check wins first.
    if let Some(d) = best_win_child {
        let due = d.saturating_add(1);
        return if due <= pass {
            Verdict::Value(if stm_white { v_white_win(due) } else { v_black_win(due) })
        } else {
            Verdict::Pending(due)
        };
    }
    if all_lose {
        let due = max_lose_dist.saturating_add(1);
        return if due <= pass {
            Verdict::Value(if stm_white { v_black_win(due) } else { v_white_win(due) })
        } else {
            Verdict::Pending(due)
        };
    }
    Verdict::Open
}

/// Parallel retrograde solve of one phase. Threads scan contiguous index
/// ranges (no per-state index list — 3v2 has 1.1e10 states), read the shared
/// value array, and buffer their assignments; assignments are applied
/// between passes so every pass sees a consistent snapshot.
pub fn solve_phase(lower: &Tablebase, wr: usize, br: usize, verbose: bool) -> Phase {
    let mut phase = Phase::new(wr, br);
    let n = phase.num_states();
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).max(1);
    let chunk = (n + threads - 1) / threads;
    let mut values = std::mem::take(&mut phase.values);
    let t_start = std::time::Instant::now();

    // 1) Mark invalid states (parallel over disjoint chunks of `values`)
    let valid: usize = std::thread::scope(|s| {
        let phase = &phase;
        let handles: Vec<_> = values
            .chunks_mut(chunk)
            .enumerate()
            .map(|(ci, slice)| {
                s.spawn(move || {
                    let base = ci * chunk;
                    let mut valid = 0usize;
                    for (i, v) in slice.iter_mut().enumerate() {
                        if phase.decode(base + i).is_none() {
                            *v = V_INVALID;
                        } else {
                            valid += 1;
                        }
                    }
                    valid
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    if verbose {
        eprintln!("phase {}v{}: {} states, {} valid, {} threads ({:.1}s)", wr, br, n, valid, threads,
                  t_start.elapsed().as_secs_f64());
    }

    // 2) Passes in distance order. A pass assigns every state whose distance
    //    is due (<= pass). Distances are not contiguous when wins run through
    //    lower phases (their distances start above 0), so instead of stopping
    //    on an empty pass we jump to the smallest pending distance; the loop
    //    ends when no unknown state has any determined child.
    let mut pass: u8 = 1;
    let mut unknown_left = valid;
    loop {
        let results: Vec<(Vec<(usize, u8)>, Option<u8>)> = std::thread::scope(|s| {
            let phase = &phase;
            let values_ref: &[u8] = &values;
            let handles: Vec<_> = (0..threads)
                .map(|ci| {
                    s.spawn(move || {
                        let lo = ci * chunk;
                        let hi = ((ci + 1) * chunk).min(n);
                        let mut out = Vec::new();
                        let mut min_pending: Option<u8> = None;
                        for idx in lo..hi {
                            if values_ref[idx] != V_UNKNOWN {
                                continue;
                            }
                            let bb = phase.decode(idx).unwrap();
                            match determine(lower, phase, values_ref, &bb, pass) {
                                Verdict::Value(v) => out.push((idx, v)),
                                Verdict::Pending(d) => {
                                    min_pending = Some(min_pending.map_or(d, |m| m.min(d)));
                                }
                                Verdict::Open => {}
                            }
                        }
                        (out, min_pending)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let n_assigned: usize = results.iter().map(|(a, _)| a.len()).sum();
        let min_pending = results.iter().filter_map(|(_, m)| *m).min();
        for (a, _) in &results {
            for &(idx, v) in a {
                values[idx] = v;
            }
        }
        unknown_left -= n_assigned;
        if verbose {
            eprintln!("  pass {:3}: assigned {:10}, unknown {:11}, next pending {:?} ({:.0}s)",
                      pass, n_assigned, unknown_left, min_pending, t_start.elapsed().as_secs_f64());
        }
        if unknown_left == 0 || pass >= MAX_DIST {
            break;
        }
        if n_assigned > 0 {
            // New values may create wins at distance pass+1: step, don't jump.
            pass = (pass + 1).min(MAX_DIST);
        } else {
            match min_pending {
                // Nothing changed this pass, so nothing can become due before
                // the smallest pending distance: jump straight to it.
                Some(d) => pass = d.max(pass + 1).min(MAX_DIST),
                None => break, // nothing determined and nothing pending: the rest are draws
            }
        }
    }
    for v in values.iter_mut() {
        if *v == V_UNKNOWN {
            *v = V_DRAW;
        }
    }
    phase.values = values;
    phase
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

    /// Packed WDL solve of a symmetric phase must agree (win/loss/draw) with
    /// the full distance solve. 1v1 keeps the test fast; the machinery is the
    /// same for 3v3.
    #[test]
    fn test_packed_matches_full_solve_1v1() {
        let tb = Tablebase::new();
        let full = solve_phase(&tb, 1, 1, false);
        let dir = std::env::temp_dir().join("tonnesjakk_tb_test");
        let packed = solve_packed(&tb, 1, &dir, 0, false);
        let mut checked = 0;
        for idx in (0..full.num_states()).step_by(97) {
            let Some(bb) = full.decode(idx) else { continue };
            let v = full.vals()[idx];
            if v == V_INVALID { continue; }
            let p = packed.value(&bb).expect("packed value");
            let same = (is_white_win(v) && is_white_win(p)) || (is_black_win(v) && is_black_win(p)) || (v == V_DRAW && p == V_DRAW);
            assert!(same, "idx {} full {} packed {}", idx, v, p);
            checked += 1;
        }
        assert!(checked > 10_000);
    }

    /// Colour symmetry: the value of 1v2 states read through the 2v1 table via
    /// the mirror must equal the directly solved 1v2 table.
    #[test]
    fn test_mirror_symmetry_matches_direct_solve() {
        let mut tb = Tablebase::new();
        tb.solve(1, 1, false);
        tb.solve(2, 1, false);
        let mut direct = Tablebase::new();
        direct.solve(1, 1, false);
        direct.solve(1, 2, false);
        let phase12 = direct.get(1, 2).unwrap();
        let mut checked = 0;
        for idx in (0..phase12.num_states()).step_by(9973) {
            if let Some(bb) = phase12.decode(idx) {
                let d = phase12.vals()[idx];
                if d == V_INVALID { continue; }
                let via_mirror = tb.value(&bb).expect("mirror lookup");
                assert_eq!(via_mirror, d, "mirror mismatch at idx {}", idx);
                checked += 1;
            }
        }
        assert!(checked > 1000);
    }

    /// 1v1: solve, then spot-check against exhaustive search on a few states.
    #[test]
    fn test_solve_1v1_matches_search() {
        let mut tb = Tablebase::new();
        let phase = tb.solve(1, 1, false);
        let n_draw = phase.vals().iter().filter(|&&v| v == V_DRAW).count();
        let n_w = phase.vals().iter().filter(|&&v| is_white_win(v)).count();
        let n_b = phase.vals().iter().filter(|&&v| is_black_win(v)).count();
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
