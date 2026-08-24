//! Single-agent race distance table.
//!
//! For one side alone on the board (no opponent pieces, no pails), the exact
//! minimum number of moves to score ALL remaining barrels, for every
//! configuration of (on-board barrel mask, barrels still in hand).
//!
//! This is the Chinese-checkers "single-agent distance" technique
//! (Roschke & Sturtevant 2013): the two-sided difference is the strongest
//! known race evaluation term in this game family. It is exact for the
//! lone-side race (jump chains over OWN barrels included) and optimistic
//! about interference (ignores enemy blocks, but also enemy ladders).
//!
//! Table is oriented for WHITE (start row 5, goal row 0); black positions
//! are mirrored before lookup. ~75k states, built once lazily (<1s).

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::board::{BitBoard, Player, BARRELS_PER_PLAYER, BOARD_SIZE, NUM_SQUARES};

pub struct RaceTable {
    index: HashMap<(u64, u8), u32>,
    dist: Vec<u16>,
}

const UNREACHED: u16 = u16::MAX;

impl RaceTable {
    fn build() -> Self {
        // ── Enumerate all states: masks with popcount <= 4 (rows 1..=5 only:
        // a barrel can never rest on the goal row), off in 0..=4-popcount ──
        let mut index: HashMap<(u64, u8), u32> = HashMap::new();
        let mut states: Vec<(u64, u8)> = Vec::new();

        // Squares below the goal row (rows 1..=5). Masks containing goal-row
        // bits are unreachable as rest states and are skipped entirely.
        let squares: Vec<u8> = (BOARD_SIZE as u8..NUM_SQUARES as u8).collect();
        let mut masks: Vec<u64> = vec![0];
        let mut current: Vec<u64> = vec![0];
        for _ in 0..BARRELS_PER_PLAYER {
            let mut next = Vec::new();
            for &m in &current {
                let highest = 63 - (m | 1).leading_zeros() as u8;
                for &sq in &squares {
                    if m == 0 || sq > highest {
                        next.push(m | 1u64 << sq);
                    }
                }
            }
            masks.extend_from_slice(&next);
            current = next;
        }

        for &mask in &masks {
            let on = mask.count_ones() as u8;
            for off in 0..=(BARRELS_PER_PLAYER as u8 - on) {
                let id = states.len() as u32;
                index.insert((mask, off), id);
                states.push((mask, off));
            }
        }

        // ── Successor lists via the real move generator (single-agent board:
        // only white barrels present, pails marked placed to suppress
        // pail sub-moves) ──
        let mut succs: Vec<Vec<u32>> = Vec::with_capacity(states.len());
        for &(mask, off) in &states {
            let mut bb = BitBoard::new();
            bb.white_barrels = mask;
            bb.occupied = mask;
            bb.white_pail_placed = true;
            bb.black_pail_placed = true;
            bb.white_barrels_off_board = off;
            bb.white_scored = BARRELS_PER_PLAYER as u8 - mask.count_ones() as u8 - off;
            bb.current_player = Player::White;

            let mut list = Vec::new();
            for mv in bb.generate_moves() {
                let mut child = bb;
                child.make_move(&mv);
                let key = (child.white_barrels, child.white_barrels_off_board);
                if let Some(&id) = index.get(&key) {
                    list.push(id);
                } else {
                    debug_assert!(false, "successor state not enumerated");
                }
            }
            list.sort_unstable();
            list.dedup();
            succs.push(list);
        }

        // ── Value iteration: dist = 1 + min(succ dists), terminal = 0 ──
        let mut dist = vec![UNREACHED; states.len()];
        let terminal = index[&(0u64, 0u8)];
        dist[terminal as usize] = 0;

        loop {
            let mut changed = false;
            for i in 0..states.len() {
                let mut best = UNREACHED;
                for &s in &succs[i] {
                    let d = dist[s as usize];
                    if d < best {
                        best = d;
                    }
                }
                if best != UNREACHED && best + 1 < dist[i] {
                    dist[i] = best + 1;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        RaceTable { index, dist }
    }

    /// Min moves for a lone side (white orientation) to score everything.
    #[inline]
    pub fn lookup(&self, mask: u64, off: u8) -> u16 {
        match self.index.get(&(mask, off)) {
            Some(&id) => self.dist[id as usize],
            None => UNREACHED, // shouldn't happen for legal states
        }
    }

    /// Distance for either side of a real position.
    #[inline]
    pub fn side_distance(&self, bb: &BitBoard, player: Player) -> u16 {
        match player {
            Player::White => self.lookup(bb.white_barrels, bb.white_barrels_off_board),
            Player::Black => self.lookup(mirror_rows(bb.black_barrels), bb.black_barrels_off_board),
        }
    }
}

/// Mirror a bitboard vertically (row r -> BOARD_SIZE-1-r), mapping black's
/// racing direction onto white's.
#[inline]
pub fn mirror_rows(mask: u64) -> u64 {
    let mut out = 0u64;
    let mut m = mask;
    while m != 0 {
        let sq = m.trailing_zeros() as usize;
        let (row, col) = (sq / BOARD_SIZE, sq % BOARD_SIZE);
        out |= 1u64 << ((BOARD_SIZE - 1 - row) * BOARD_SIZE + col);
        m &= m - 1;
    }
    out
}

pub static RACE_TABLE: LazyLock<RaceTable> = LazyLock::new(RaceTable::build);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::sq;

    #[test]
    fn test_race_distances() {
        let t = &RACE_TABLE;
        // All scored
        assert_eq!(t.lookup(0, 0), 0);
        // One barrel in hand: 1 placement (row 5) + 5 steps to row 0 = 6
        assert_eq!(t.lookup(0, 1), 6);
        // One barrel one step from goal: 1
        assert_eq!(t.lookup(1u64 << sq(1, 2), 0), 1);
        // Jump chain: barrel at (2,2) jumps over (1,2) to goal = 1 move,
        // then (1,2) scores next = total 2
        let mask = (1u64 << sq(2, 2)) | (1u64 << sq(1, 2));
        assert_eq!(t.lookup(mask, 0), 2);
        // Four in hand must be at least 4 placements + walks
        assert!(t.lookup(0, 4) >= 4 + 5);
        // Mirror sanity: black barrel at (4,3) is one step from its goal (row 5)
        let black = 1u64 << sq(4, 3);
        assert_eq!(t.lookup(mirror_rows(black), 0), 1);
    }
}
