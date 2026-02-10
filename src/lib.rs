use pyo3::prelude::*;
use std::fmt;
use wide::f32x8;

// ============================================================================
// KONSTANTER
// ============================================================================

/// Størrelse på brettet (6x6)
pub const BOARD_SIZE: usize = 6;

/// Antall ruter på brettet
pub const NUM_SQUARES: usize = BOARD_SIZE * BOARD_SIZE; // 36

/// Antall tønner per spiller
pub const BARRELS_PER_PLAYER: usize = 4;

/// Maks søkedybde (for killer moves array)
pub const MAX_DEPTH: usize = 32;

/// Størrelse på transposition table (antall entries)
pub const TT_SIZE: usize = 1 << 20; // ~1 million entries

// NNUE feature sizes
/// Base features: 6x6 board * 4 piece types = 144
pub const BASE_FEATURES: usize = NUM_SQUARES * 4;
/// Relational features (only cheap, non-inferable ones):
///   - White barrels scored (1) - can't infer from position alone
///   - Black barrels scored (1) - can't infer from position alone
///   - Current player (1) - not encoded in base features
pub const RELATIONAL_FEATURES: usize = 3;
/// Total input features to NNUE
pub const INPUT_SIZE: usize = BASE_FEATURES + RELATIONAL_FEATURES; // 147


// ============================================================================
// BITBOARD - Rask brettrepresentasjon med 64-bit integers
// ============================================================================

/// Konverter (rad, kolonne) til bitindex
/// Rad 0: bits 0-5, Rad 1: bits 6-11, ..., Rad 5: bits 30-35
#[inline(always)]
const fn sq(row: usize, col: usize) -> usize {
    row * BOARD_SIZE + col
}

/// Konverter bitindex til (rad, kolonne)
#[inline(always)]
const fn sq_to_coords(sq: usize) -> (usize, usize) {
    (sq / BOARD_SIZE, sq % BOARD_SIZE)
}

/// Bitmask for én rute
#[inline(always)]
const fn bit(sq: usize) -> u64 {
    1u64 << sq
}

// ─────────────────────────────────────────────────────────────────────────────
// Prekalkulerte oppslagstabeller (const, kompilert inn i binærfilen)
// ─────────────────────────────────────────────────────────────────────────────

/// Maske for hver rad (6 bits per rad)
const ROW_MASK: [u64; BOARD_SIZE] = {
    let mut masks = [0u64; BOARD_SIZE];
    let mut row = 0;
    while row < BOARD_SIZE {
        masks[row] = 0b111111u64 << (row * BOARD_SIZE);
        row += 1;
    }
    masks
};

/// Naboer for hvert felt (alle 8 retninger: ortogonalt + diagonalt)
const ADJACENT: [u64; NUM_SQUARES] = {
    let mut adj = [0u64; NUM_SQUARES];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        let (row, col) = sq_to_coords(sq);
        let mut mask = 0u64;

        // Orthogonal directions
        // Opp
        if row > 0 {
            mask |= bit(sq - BOARD_SIZE);
        }
        // Ned
        if row < BOARD_SIZE - 1 {
            mask |= bit(sq + BOARD_SIZE);
        }
        // Venstre
        if col > 0 {
            mask |= bit(sq - 1);
        }
        // Høyre
        if col < BOARD_SIZE - 1 {
            mask |= bit(sq + 1);
        }

        // Diagonal directions
        // Opp-venstre
        if row > 0 && col > 0 {
            mask |= bit(sq - BOARD_SIZE - 1);
        }
        // Opp-høyre
        if row > 0 && col < BOARD_SIZE - 1 {
            mask |= bit(sq - BOARD_SIZE + 1);
        }
        // Ned-venstre
        if row < BOARD_SIZE - 1 && col > 0 {
            mask |= bit(sq + BOARD_SIZE - 1);
        }
        // Ned-høyre
        if row < BOARD_SIZE - 1 && col < BOARD_SIZE - 1 {
            mask |= bit(sq + BOARD_SIZE + 1);
        }

        adj[sq] = mask;
        sq += 1;
    }
    adj
};

/// Number of jump directions (8: orthogonal + diagonal)
const NUM_JUMP_DIRS: usize = 8;

/// For hvert felt og retning: feltet som hoppes over (-1 hvis ugyldig)
/// Directions: 0=Up, 1=Down, 2=Left, 3=Right, 4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight
const JUMP_OVER: [[i8; NUM_JUMP_DIRS]; NUM_SQUARES] = {
    let mut table = [[-1i8; NUM_JUMP_DIRS]; NUM_SQUARES];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        let (row, col) = sq_to_coords(sq);

        // Orthogonal directions
        // Opp (dir=0)
        if row >= 1 {
            table[sq][0] = (sq - BOARD_SIZE) as i8;
        }
        // Ned (dir=1)
        if row < BOARD_SIZE - 1 {
            table[sq][1] = (sq + BOARD_SIZE) as i8;
        }
        // Venstre (dir=2)
        if col >= 1 {
            table[sq][2] = (sq - 1) as i8;
        }
        // Høyre (dir=3)
        if col < BOARD_SIZE - 1 {
            table[sq][3] = (sq + 1) as i8;
        }

        // Diagonal directions
        // Opp-venstre (dir=4)
        if row >= 1 && col >= 1 {
            table[sq][4] = (sq - BOARD_SIZE - 1) as i8;
        }
        // Opp-høyre (dir=5)
        if row >= 1 && col < BOARD_SIZE - 1 {
            table[sq][5] = (sq - BOARD_SIZE + 1) as i8;
        }
        // Ned-venstre (dir=6)
        if row < BOARD_SIZE - 1 && col >= 1 {
            table[sq][6] = (sq + BOARD_SIZE - 1) as i8;
        }
        // Ned-høyre (dir=7)
        if row < BOARD_SIZE - 1 && col < BOARD_SIZE - 1 {
            table[sq][7] = (sq + BOARD_SIZE + 1) as i8;
        }

        sq += 1;
    }
    table
};

/// For hvert felt og retning: landingsfeltet etter hopp (-1 hvis ugyldig)
/// Directions: 0=Up, 1=Down, 2=Left, 3=Right, 4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight
const JUMP_LANDING: [[i8; NUM_JUMP_DIRS]; NUM_SQUARES] = {
    let mut table = [[-1i8; NUM_JUMP_DIRS]; NUM_SQUARES];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        let (row, col) = sq_to_coords(sq);

        // Orthogonal directions
        // Opp (dir=0) - trenger 2 rader over
        if row >= 2 {
            table[sq][0] = (sq - 2 * BOARD_SIZE) as i8;
        }
        // Ned (dir=1) - trenger 2 rader under
        if row < BOARD_SIZE - 2 {
            table[sq][1] = (sq + 2 * BOARD_SIZE) as i8;
        }
        // Venstre (dir=2) - trenger 2 kolonner til venstre
        if col >= 2 {
            table[sq][2] = (sq - 2) as i8;
        }
        // Høyre (dir=3) - trenger 2 kolonner til høyre
        if col < BOARD_SIZE - 2 {
            table[sq][3] = (sq + 2) as i8;
        }

        // Diagonal directions
        // Opp-venstre (dir=4) - trenger 2 rader over og 2 kolonner til venstre
        if row >= 2 && col >= 2 {
            table[sq][4] = (sq - 2 * BOARD_SIZE - 2) as i8;
        }
        // Opp-høyre (dir=5)
        if row >= 2 && col < BOARD_SIZE - 2 {
            table[sq][5] = (sq - 2 * BOARD_SIZE + 2) as i8;
        }
        // Ned-venstre (dir=6)
        if row < BOARD_SIZE - 2 && col >= 2 {
            table[sq][6] = (sq + 2 * BOARD_SIZE - 2) as i8;
        }
        // Ned-høyre (dir=7)
        if row < BOARD_SIZE - 2 && col < BOARD_SIZE - 2 {
            table[sq][7] = (sq + 2 * BOARD_SIZE + 2) as i8;
        }

        sq += 1;
    }
    table
};

/// Bitboard-representasjon av spillbrettet
/// Bruker u64 hvor bit 0-35 representerer feltene (rad*6 + kolonne)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BitBoard {
    pub white_barrels: u64,    // Maks 4 bits satt
    pub black_barrels: u64,    // Maks 4 bits satt
    pub white_pail: u64,       // 0 eller 1 bit
    pub black_pail: u64,       // 0 eller 1 bit
    pub occupied: u64,         // Derivert: alle brikker

    // Game state
    pub current_player: Player,
    pub move_count: u32,
    pub hash: u64,
    pub white_pail_placed: bool,
    pub black_pail_placed: bool,
    pub white_barrels_off_board: u8,
    pub black_barrels_off_board: u8,
    pub white_scored: u8,
    pub black_scored: u8,
}

/// Informasjon for å angre et trekk
#[derive(Clone, Copy)]
pub struct UndoInfo {
    pub white_barrels: u64,
    pub black_barrels: u64,
    pub white_pail: u64,
    pub black_pail: u64,
    pub occupied: u64,
    pub hash: u64,
    pub white_pail_placed: bool,
    pub black_pail_placed: bool,
    pub white_barrels_off_board: u8,
    pub black_barrels_off_board: u8,
    pub white_scored: u8,
    pub black_scored: u8,
}

/// Kompakt representasjon av et trekk for bitboard (32 bits)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BitMove {
    /// Packed data:
    /// bits 0-5: barrel_to (0-35)
    /// bits 6-11: barrel_from (0-35, eller 63 for plassering)
    /// bits 12-17: pail_pos (0-35, eller 63 for ingen)
    /// bit 18: is_placement
    /// bits 19-23: path_len (0-31)
    pub packed: u32,
    /// Hopp-sti (opptil 8 hopp)
    pub path: [u8; 8],
}

impl BitMove {
    const NO_POS: u8 = 63;

    #[inline]
    pub fn new_placement(barrel_to: u8, pail_pos: Option<u8>) -> Self {
        let pail = pail_pos.unwrap_or(Self::NO_POS);
        let packed = (barrel_to as u32)
            | ((Self::NO_POS as u32) << 6)
            | ((pail as u32) << 12)
            | (1 << 18)
            | (1 << 19); // path_len = 1
        BitMove {
            packed,
            path: [barrel_to, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    #[inline]
    pub fn new_move(barrel_from: u8, barrel_to: u8, path: &[u8], pail_pos: Option<u8>) -> Self {
        let pail = pail_pos.unwrap_or(Self::NO_POS);
        let path_len = path.len() as u32;
        let packed = (barrel_to as u32)
            | ((barrel_from as u32) << 6)
            | ((pail as u32) << 12)
            | (0 << 18) // not placement
            | (path_len << 19);

        let mut path_arr = [0u8; 8];
        for (i, &p) in path.iter().enumerate().take(8) {
            path_arr[i] = p;
        }

        BitMove {
            packed,
            path: path_arr,
        }
    }

    #[inline]
    pub fn barrel_to(&self) -> u8 {
        (self.packed & 0x3F) as u8
    }

    #[inline]
    pub fn barrel_from(&self) -> Option<u8> {
        let from = ((self.packed >> 6) & 0x3F) as u8;
        if from == Self::NO_POS { None } else { Some(from) }
    }

    #[inline]
    pub fn pail_pos(&self) -> Option<u8> {
        let pail = ((self.packed >> 12) & 0x3F) as u8;
        if pail == Self::NO_POS { None } else { Some(pail) }
    }

    #[inline]
    pub fn is_placement(&self) -> bool {
        (self.packed >> 18) & 1 == 1
    }

    #[inline]
    pub fn path_len(&self) -> usize {
        ((self.packed >> 19) & 0x1F) as usize
    }

    /// Konverter til Move (for Python-kompatibilitet)
    pub fn to_move(&self) -> Move {
        let barrel_to = self.barrel_to();
        let (to_row, to_col) = sq_to_coords(barrel_to as usize);
        let to_pos = Position::new(to_row as i8, to_col as i8);

        let pail = self.pail_pos().map(|p| {
            let (r, c) = sq_to_coords(p as usize);
            Position::new(r as i8, c as i8)
        });

        if self.is_placement() {
            Move::place_barrel(pail, to_pos)
        } else {
            let from = self.barrel_from().unwrap();
            let (from_row, from_col) = sq_to_coords(from as usize);
            let from_pos = Position::new(from_row as i8, from_col as i8);

            let path: Vec<Position> = (0..self.path_len())
                .map(|i| {
                    let sq = self.path[i];
                    let (r, c) = sq_to_coords(sq as usize);
                    Position::new(r as i8, c as i8)
                })
                .collect();

            Move::move_barrel(pail, from_pos, path)
        }
    }
}

impl BitBoard {
    /// Opprett et nytt tomt BitBoard
    pub fn new() -> Self {
        // Beregn initial Zobrist hash (tomt brett)
        let mut hash = 0u64;
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let piece_idx = ZobristKeys::piece_index(Cell::Empty);
                hash ^= ZOBRIST.pieces[row][col][piece_idx];
            }
        }

        BitBoard {
            white_barrels: 0,
            black_barrels: 0,
            white_pail: 0,
            black_pail: 0,
            occupied: 0,
            current_player: Player::White,
            move_count: 0,
            hash,
            white_pail_placed: false,
            black_pail_placed: false,
            white_barrels_off_board: BARRELS_PER_PLAYER as u8,
            black_barrels_off_board: BARRELS_PER_PLAYER as u8,
            white_scored: 0,
            black_scored: 0,
        }
    }

    /// Konverter fra Board til BitBoard
    pub fn from_board(board: &Board) -> Self {
        let mut bb = BitBoard {
            white_barrels: 0,
            black_barrels: 0,
            white_pail: 0,
            black_pail: 0,
            occupied: 0,
            current_player: board.current_player,
            move_count: board.move_count,
            hash: board.hash,
            white_pail_placed: board.white_pail_placed,
            black_pail_placed: board.black_pail_placed,
            white_barrels_off_board: board.white_barrels_off_board,
            black_barrels_off_board: board.black_barrels_off_board,
            white_scored: board.white_scored,
            black_scored: board.black_scored,
        };

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let sq = sq(row, col);
                match board.cells[row][col] {
                    Cell::WhiteBarrel => bb.white_barrels |= bit(sq),
                    Cell::BlackBarrel => bb.black_barrels |= bit(sq),
                    Cell::WhitePail => bb.white_pail |= bit(sq),
                    Cell::BlackPail => bb.black_pail |= bit(sq),
                    Cell::Empty => {}
                }
            }
        }

        bb.occupied = bb.white_barrels | bb.black_barrels | bb.white_pail | bb.black_pail;
        bb
    }

    /// Konverter til Board
    pub fn to_board(&self) -> Board {
        let mut board = Board::new();
        board.current_player = self.current_player;
        board.move_count = self.move_count;
        board.hash = self.hash;
        board.white_pail_placed = self.white_pail_placed;
        board.black_pail_placed = self.black_pail_placed;
        board.white_barrels_off_board = self.white_barrels_off_board;
        board.black_barrels_off_board = self.black_barrels_off_board;
        board.white_scored = self.white_scored;
        board.black_scored = self.black_scored;

        // Sett celler fra bitboards
        let mut bb = self.white_barrels;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            let (row, col) = sq_to_coords(sq);
            board.cells[row][col] = Cell::WhiteBarrel;
            bb &= bb - 1;
        }

        bb = self.black_barrels;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            let (row, col) = sq_to_coords(sq);
            board.cells[row][col] = Cell::BlackBarrel;
            bb &= bb - 1;
        }

        bb = self.white_pail;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            let (row, col) = sq_to_coords(sq);
            board.cells[row][col] = Cell::WhitePail;
            bb &= bb - 1;
        }

        bb = self.black_pail;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            let (row, col) = sq_to_coords(sq);
            board.cells[row][col] = Cell::BlackPail;
            bb &= bb - 1;
        }

        board
    }

    /// Alle tønner (begge spillere)
    #[inline]
    pub fn all_barrels(&self) -> u64 {
        self.white_barrels | self.black_barrels
    }

    /// Tomme felt
    #[inline]
    pub fn empty(&self) -> u64 {
        !self.occupied & ((1u64 << NUM_SQUARES) - 1)
    }

    /// Spillerens startrad (maske)
    #[inline]
    pub fn starting_row_mask(&self, player: Player) -> u64 {
        match player {
            Player::White => ROW_MASK[BOARD_SIZE - 1], // Rad 5
            Player::Black => ROW_MASK[0],              // Rad 0
        }
    }

    /// Spillerens målrad
    #[inline]
    pub fn goal_row(&self, player: Player) -> usize {
        match player {
            Player::White => 0,
            Player::Black => BOARD_SIZE - 1,
        }
    }

    /// Sjekk vinner
    pub fn check_winner(&self) -> Option<Player> {
        if self.white_scored == BARRELS_PER_PLAYER as u8 {
            Some(Player::White)
        } else if self.black_scored == BARRELS_PER_PLAYER as u8 {
            Some(Player::Black)
        } else {
            None
        }
    }

    /// Lagre state for undo
    #[inline]
    pub fn save_undo(&self) -> UndoInfo {
        UndoInfo {
            white_barrels: self.white_barrels,
            black_barrels: self.black_barrels,
            white_pail: self.white_pail,
            black_pail: self.black_pail,
            occupied: self.occupied,
            hash: self.hash,
            white_pail_placed: self.white_pail_placed,
            black_pail_placed: self.black_pail_placed,
            white_barrels_off_board: self.white_barrels_off_board,
            black_barrels_off_board: self.black_barrels_off_board,
            white_scored: self.white_scored,
            black_scored: self.black_scored,
        }
    }

    /// Gjenopprett state fra undo
    #[inline]
    pub fn restore_undo(&mut self, undo: &UndoInfo) {
        self.white_barrels = undo.white_barrels;
        self.black_barrels = undo.black_barrels;
        self.white_pail = undo.white_pail;
        self.black_pail = undo.black_pail;
        self.occupied = undo.occupied;
        self.hash = undo.hash;
        self.white_pail_placed = undo.white_pail_placed;
        self.black_pail_placed = undo.black_pail_placed;
        self.white_barrels_off_board = undo.white_barrels_off_board;
        self.black_barrels_off_board = undo.black_barrels_off_board;
        self.white_scored = undo.white_scored;
        self.black_scored = undo.black_scored;

        // Bytt tilbake spiller
        self.current_player = self.current_player.opponent();
        self.move_count -= 1;
    }

    /// Generer alle lovlige trekk (bitboard-versjon)
    ///
    /// New rules: Pail must be moved every turn!
    /// - First turn: Place pail somewhere (required) + place/move barrel
    /// - Subsequent turns: Move pail to new location (required) + move barrel
    pub fn generate_moves(&self) -> Vec<BitMove> {
        let mut moves = Vec::with_capacity(64);
        let player = self.current_player;

        let pail_placed = match player {
            Player::White => self.white_pail_placed,
            Player::Black => self.black_pail_placed,
        };
        let barrels_off = match player {
            Player::White => self.white_barrels_off_board,
            Player::Black => self.black_barrels_off_board,
        };
        let my_barrels = match player {
            Player::White => self.white_barrels,
            Player::Black => self.black_barrels,
        };

        let start_row_mask = self.starting_row_mask(player);

        // Empty squares for moves
        let empty = self.empty();

        // Pail placement: only on first turn when pail not yet placed
        // Generate pail destinations (empty if already placed)
        let pail_destinations: Vec<Option<u8>> = if !pail_placed {
            // First placement: any empty square
            let mut dests = Vec::with_capacity(32);
            let mut e = empty;
            while e != 0 {
                let sq = e.trailing_zeros() as u8;
                dests.push(Some(sq));
                e &= e - 1;
            }
            dests
        } else {
            // Pail already placed - no pail move needed
            vec![None]
        };

        // For each pail destination (or None if already placed), generate barrel moves
        for pail_opt in &pail_destinations {
            // Calculate occupied squares with pail in new position (if placing)
            let temp_occupied = if let Some(pail_sq) = pail_opt {
                self.occupied | bit(*pail_sq as usize)
            } else {
                self.occupied
            };
            let temp_empty = !temp_occupied & ((1u64 << NUM_SQUARES) - 1);

            // A: Plasser ny tønne fra utenfor brettet
            if barrels_off > 0 {
                let placements = start_row_mask & temp_empty;
                let mut p = placements;
                while p != 0 {
                    let to_sq = p.trailing_zeros() as u8;
                    moves.push(BitMove::new_placement(to_sq, *pail_opt));
                    p &= p - 1;
                }
            }

            // B: Flytt eksisterende tønne
            let mut barrels = my_barrels;
            while barrels != 0 {
                let from_sq = barrels.trailing_zeros() as u8;

                // Enkle trekk (ett felt) - now 8 directions
                let adjacent = ADJACENT[from_sq as usize] & temp_empty;
                let mut adj = adjacent;
                while adj != 0 {
                    let to_sq = adj.trailing_zeros() as u8;
                    moves.push(BitMove::new_move(from_sq, to_sq, &[to_sq], *pail_opt));
                    adj &= adj - 1;
                }

                // Hopp-sekvenser (iterativ DFS)
                self.find_jumps_iterative(
                    from_sq,
                    temp_occupied,
                    *pail_opt,
                    &mut moves,
                );

                barrels &= barrels - 1;
            }
        }

        moves
    }

    /// Finn alle hopp-sekvenser fra et felt (iterativ versjon)
    /// Supports 8 directions (orthogonal + diagonal)
    /// Can jump over barrels AND your own pail, but NOT opponent's pail
    fn find_jumps_iterative(
        &self,
        from_sq: u8,
        temp_occupied: u64,
        pail_opt: Option<u8>,
        moves: &mut Vec<BitMove>,
    ) {
        // Stack: (current_sq, visited_mask, path)
        let mut stack: Vec<(u8, u64, Vec<u8>)> = Vec::with_capacity(16);

        // Start med alle retninger fra from_sq
        let all_barrels = self.all_barrels();
        // Opponent's pail blocks jumps, own pail does not
        let opponent_pail = match self.current_player {
            Player::White => self.black_pail,
            Player::Black => self.white_pail,
        };
        let own_pail = match self.current_player {
            Player::White => self.white_pail,
            Player::Black => self.black_pail,
        };
        // Can jump over: barrels OR own pail
        let jumpable = all_barrels | own_pail;
        let visited_start = bit(from_sq as usize);

        for dir in 0..NUM_JUMP_DIRS {
            let over = JUMP_OVER[from_sq as usize][dir];
            let landing = JUMP_LANDING[from_sq as usize][dir];

            if over < 0 || landing < 0 {
                continue;
            }

            let over_bit = bit(over as usize);
            let landing_bit = bit(landing as usize);

            // Can jump over barrels or own pail, but NOT opponent's pail
            if (jumpable & over_bit) != 0
                && (opponent_pail & over_bit) == 0  // Cannot jump over opponent's pail
                && (temp_occupied & landing_bit) == 0
            {
                let path = vec![landing as u8];
                let visited = visited_start | landing_bit;

                // Legg til dette trekket
                moves.push(BitMove::new_move(from_sq, landing as u8, &path, pail_opt));

                // Push til stack for å fortsette søket
                stack.push((landing as u8, visited, path));
            }
        }

        // DFS
        while let Some((current, visited, path)) = stack.pop() {
            for dir in 0..NUM_JUMP_DIRS {
                let over = JUMP_OVER[current as usize][dir];
                let landing = JUMP_LANDING[current as usize][dir];

                if over < 0 || landing < 0 {
                    continue;
                }

                let over_bit = bit(over as usize);
                let landing_bit = bit(landing as usize);

                // Allerede besøkt?
                if (visited & landing_bit) != 0 {
                    continue;
                }

                // Can jump over barrels or own pail, but NOT opponent's pail
                if (jumpable & over_bit) != 0
                    && (opponent_pail & over_bit) == 0  // Cannot jump over opponent's pail
                    && (temp_occupied & landing_bit) == 0
                {
                    let mut new_path = path.clone();
                    new_path.push(landing as u8);
                    let new_visited = visited | landing_bit;

                    // Legg til trekket
                    let to = landing as u8;
                    moves.push(BitMove::new_move(from_sq, to, &new_path, pail_opt));

                    // Fortsett søket
                    stack.push((to, new_visited, new_path));
                }
            }
        }
    }

    /// Utfør et trekk
    pub fn make_move(&mut self, mv: &BitMove) -> UndoInfo {
        let undo = self.save_undo();
        let player = self.current_player;
        let goal_row = self.goal_row(player);

        // 1. Place pail (one-time placement, only when not yet placed)
        if let Some(pail_sq) = mv.pail_pos() {
            let pail_bit = bit(pail_sq as usize);
            let (pail_row, pail_col) = sq_to_coords(pail_sq as usize);

            match player {
                Player::White => {
                    debug_assert!(!self.white_pail_placed, "White pail already placed!");
                    self.white_pail = pail_bit;
                    self.white_pail_placed = true;
                    self.hash ^= ZOBRIST.pieces[pail_row][pail_col][ZobristKeys::piece_index(Cell::Empty)];
                    self.hash ^= ZOBRIST.pieces[pail_row][pail_col][ZobristKeys::piece_index(Cell::WhitePail)];
                }
                Player::Black => {
                    debug_assert!(!self.black_pail_placed, "Black pail already placed!");
                    self.black_pail = pail_bit;
                    self.black_pail_placed = true;
                    self.hash ^= ZOBRIST.pieces[pail_row][pail_col][ZobristKeys::piece_index(Cell::Empty)];
                    self.hash ^= ZOBRIST.pieces[pail_row][pail_col][ZobristKeys::piece_index(Cell::BlackPail)];
                }
            }
            self.occupied |= pail_bit;
        }

        // 2. Håndter tønne
        let to_sq = mv.barrel_to();
        let to_bit = bit(to_sq as usize);
        let (to_row, to_col) = sq_to_coords(to_sq as usize);

        let barrel_cell = match player {
            Player::White => Cell::WhiteBarrel,
            Player::Black => Cell::BlackBarrel,
        };

        if mv.is_placement() {
            // Plasser ny tønne
            match player {
                Player::White => self.white_barrels_off_board -= 1,
                Player::Black => self.black_barrels_off_board -= 1,
            }

            // Sjekk om den scorer med en gang
            if to_row == goal_row {
                match player {
                    Player::White => self.white_scored += 1,
                    Player::Black => self.black_scored += 1,
                }
            } else {
                match player {
                    Player::White => self.white_barrels |= to_bit,
                    Player::Black => self.black_barrels |= to_bit,
                }
                self.occupied |= to_bit;
                self.hash ^= ZOBRIST.pieces[to_row][to_col][ZobristKeys::piece_index(Cell::Empty)];
                self.hash ^= ZOBRIST.pieces[to_row][to_col][ZobristKeys::piece_index(barrel_cell)];
            }
        } else {
            // Flytt eksisterende tønne
            let from_sq = mv.barrel_from().unwrap();
            let from_bit = bit(from_sq as usize);
            let (from_row, from_col) = sq_to_coords(from_sq as usize);

            // Fjern fra gammel posisjon
            match player {
                Player::White => self.white_barrels &= !from_bit,
                Player::Black => self.black_barrels &= !from_bit,
            }
            self.occupied &= !from_bit;
            self.hash ^= ZOBRIST.pieces[from_row][from_col][ZobristKeys::piece_index(barrel_cell)];
            self.hash ^= ZOBRIST.pieces[from_row][from_col][ZobristKeys::piece_index(Cell::Empty)];

            // Sjekk om scorer
            if to_row == goal_row {
                match player {
                    Player::White => self.white_scored += 1,
                    Player::Black => self.black_scored += 1,
                }
            } else {
                match player {
                    Player::White => self.white_barrels |= to_bit,
                    Player::Black => self.black_barrels |= to_bit,
                }
                self.occupied |= to_bit;
                self.hash ^= ZOBRIST.pieces[to_row][to_col][ZobristKeys::piece_index(Cell::Empty)];
                self.hash ^= ZOBRIST.pieces[to_row][to_col][ZobristKeys::piece_index(barrel_cell)];
            }
        }

        // 3. Bytt spiller
        self.hash ^= ZOBRIST.player_to_move;
        self.current_player = player.opponent();
        self.move_count += 1;

        undo
    }

    /// Angre et trekk
    #[inline]
    pub fn unmake_move(&mut self, undo: &UndoInfo) {
        self.restore_undo(undo);
    }

    /// Utfør et null-trekk (bare bytt spiller uten å flytte noe)
    /// Brukes for Null Move Pruning i søket
    #[inline]
    pub fn make_null_move(&mut self) {
        self.hash ^= ZOBRIST.player_to_move;
        self.current_player = self.current_player.opponent();
        self.move_count += 1;
    }

    /// Angre et null-trekk
    #[inline]
    pub fn unmake_null_move(&mut self) {
        self.hash ^= ZOBRIST.player_to_move;
        self.current_player = self.current_player.opponent();
        self.move_count -= 1;
    }

    /// Sjekk om en tønne er nær mål (0 eller 1 rader unna)
    /// Brukes for å unngå null move pruning i taktiske stillinger
    #[inline]
    pub fn has_barrel_near_goal(&self) -> bool {
        let player = self.current_player;
        let my_barrels = match player {
            Player::White => self.white_barrels,
            Player::Black => self.black_barrels,
        };

        // Sjekk om noen tønner er på rad 0 eller 1 for hvit, rad 4 eller 5 for svart
        let near_goal_mask = match player {
            Player::White => ROW_MASK[0] | ROW_MASK[1], // Hvit mål er rad 0
            Player::Black => ROW_MASK[4] | ROW_MASK[5], // Svart mål er rad 5
        };

        (my_barrels & near_goal_mask) != 0
    }
}

impl Default for BitBoard {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ZOBRIST HASHING
// ============================================================================

/// Zobrist-nøkler for hashing av brett-tilstander
/// Generert med fast seed for reproduserbarhet
struct ZobristKeys {
    pieces: [[[u64; 5]; BOARD_SIZE]; BOARD_SIZE], // [row][col][piece_type]
    player_to_move: u64,
}

impl ZobristKeys {
    fn new() -> Self {
        // Enkel PRNG med fast seed
        let mut state: u64 = 0x123456789ABCDEF0;
        let mut next_random = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545F4914F6CDD1D)
        };

        let mut pieces = [[[0u64; 5]; BOARD_SIZE]; BOARD_SIZE];
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                for piece in 0..5 {
                    pieces[row][col][piece] = next_random();
                }
            }
        }

        ZobristKeys {
            pieces,
            player_to_move: next_random(),
        }
    }

    fn piece_index(cell: Cell) -> usize {
        match cell {
            Cell::Empty => 0,
            Cell::WhiteBarrel => 1,
            Cell::BlackBarrel => 2,
            Cell::WhitePail => 3,
            Cell::BlackPail => 4,
        }
    }
}

// Global Zobrist keys (initialisert én gang)
static ZOBRIST: std::sync::LazyLock<ZobristKeys> = std::sync::LazyLock::new(ZobristKeys::new);

/// Representerer en spiller
#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    White = 0,
    Black = 1,
}

#[pymethods]
impl Player {
    fn opponent(&self) -> Player {
        match self {
            Player::White => Player::Black,
            Player::Black => Player::White,
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Player::White => "Player.White",
            Player::Black => "Player.Black",
        }
    }
}

/// Representerer innholdet i en rute
#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Empty,
    WhiteBarrel,
    BlackBarrel,
    WhitePail,  // Melkespann
    BlackPail,
}

impl Cell {
    fn is_barrel(&self) -> bool {
        matches!(self, Cell::WhiteBarrel | Cell::BlackBarrel)
    }

    fn is_pail(&self) -> bool {
        matches!(self, Cell::WhitePail | Cell::BlackPail)
    }

    fn is_barrel_of(&self, player: Player) -> bool {
        match (self, player) {
            (Cell::WhiteBarrel, Player::White) => true,
            (Cell::BlackBarrel, Player::Black) => true,
            _ => false,
        }
    }
}

/// En posisjon på brettet
#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Position {
    #[pyo3(get)]
    pub row: i8,
    #[pyo3(get)]
    pub col: i8,
}

#[pymethods]
impl Position {
    #[new]
    fn new(row: i8, col: i8) -> Self {
        Position { row, col }
    }

    fn is_valid(&self) -> bool {
        self.row >= 0
            && self.row < BOARD_SIZE as i8
            && self.col >= 0
            && self.col < BOARD_SIZE as i8
    }

    fn __repr__(&self) -> String {
        format!("Position({}, {})", self.row, self.col)
    }

    fn __eq__(&self, other: &Position) -> bool {
        self.row == other.row && self.col == other.col
    }

    fn __hash__(&self) -> u64 {
        (self.row as u64) << 8 | (self.col as u64)
    }
}

/// Et trekk i spillet
/// Består av:
/// - Valgfritt: plasser melkespann (kun hvis ikke allerede plassert)
/// - Påkrevd: ENTEN plasser ny tønne på startrad ELLER flytt eksisterende tønne
#[pyclass]
#[derive(Clone, Debug)]
pub struct Move {
    #[pyo3(get)]
    pub place_pail: Option<Position>,    // Hvor plassere melkespann (None = ikke plasser)
    #[pyo3(get)]
    pub is_barrel_placement: bool,       // true = plasser ny tønne, false = flytt eksisterende
    #[pyo3(get)]
    pub barrel_from: Option<Position>,   // None hvis plassering fra utenfor brettet
    #[pyo3(get)]
    pub barrel_to: Position,             // Hvor tønnen ender opp
    #[pyo3(get)]
    pub barrel_path: Vec<Position>,      // Hopp-sti (for flytt med flere hopp)
}

#[pymethods]
impl Move {
    #[new]
    fn new(
        place_pail: Option<Position>,
        is_barrel_placement: bool,
        barrel_from: Option<Position>,
        barrel_to: Position,
        barrel_path: Vec<Position>,
    ) -> Self {
        Move {
            place_pail,
            is_barrel_placement,
            barrel_from,
            barrel_to,
            barrel_path,
        }
    }

    /// Hjelpefunksjon for å lage et "plasser tønne"-trekk
    #[staticmethod]
    fn place_barrel(pail: Option<Position>, to: Position) -> Self {
        Move {
            place_pail: pail,
            is_barrel_placement: true,
            barrel_from: None,
            barrel_to: to,
            barrel_path: vec![to],
        }
    }

    /// Hjelpefunksjon for å lage et "flytt tønne"-trekk
    #[staticmethod]
    fn move_barrel(pail: Option<Position>, from: Position, path: Vec<Position>) -> Self {
        let to = *path.last().unwrap_or(&from);
        Move {
            place_pail: pail,
            is_barrel_placement: false,
            barrel_from: Some(from),
            barrel_to: to,
            barrel_path: path,
        }
    }

    fn __repr__(&self) -> String {
        let pail_str = match &self.place_pail {
            Some(p) => format!("+pail({},{})", p.row, p.col),
            None => String::new(),
        };
        let barrel_str = if self.is_barrel_placement {
            format!("place_barrel({},{})", self.barrel_to.row, self.barrel_to.col)
        } else {
            let from = self.barrel_from.unwrap();
            format!("move({},{})→({},{})", from.row, from.col, self.barrel_to.row, self.barrel_to.col)
        };
        format!("Move({}{})", pail_str, barrel_str)
    }
}

/// Spillbrettet og tilstanden
#[pyclass]
#[derive(Clone)]
pub struct Board {
    cells: [[Cell; BOARD_SIZE]; BOARD_SIZE],
    #[pyo3(get)]
    current_player: Player,
    #[pyo3(get)]
    move_count: u32,
    hash: u64,  // Zobrist hash for transposition table
    #[pyo3(get)]
    white_pail_placed: bool,   // Har hvit plassert melkespannet?
    #[pyo3(get)]
    black_pail_placed: bool,   // Har svart plassert melkespannet?
    #[pyo3(get)]
    white_barrels_off_board: u8,  // Antall hvite tønner som ikke er plassert ennå
    #[pyo3(get)]
    black_barrels_off_board: u8,  // Antall svarte tønner som ikke er plassert ennå
    #[pyo3(get)]
    white_scored: u8,  // Antall hvite tønner som har nådd mål (fjernet fra brettet)
    #[pyo3(get)]
    black_scored: u8,  // Antall svarte tønner som har nådd mål (fjernet fra brettet)
}

#[pymethods]
impl Board {
    /// Opprett et nytt brett - alle brikker starter UTENFOR brettet
    #[new]
    fn new() -> Self {
        let cells = [[Cell::Empty; BOARD_SIZE]; BOARD_SIZE];

        // Alle tønner starter utenfor brettet
        // Melkespann er heller ikke plassert

        // Beregn initial Zobrist hash (tomt brett)
        let mut hash = 0u64;
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let piece_idx = ZobristKeys::piece_index(Cell::Empty);
                hash ^= ZOBRIST.pieces[row][col][piece_idx];
            }
        }

        Board {
            cells,
            current_player: Player::White,
            move_count: 0,
            hash,
            white_pail_placed: false,
            black_pail_placed: false,
            white_barrels_off_board: BARRELS_PER_PLAYER as u8,
            black_barrels_off_board: BARRELS_PER_PLAYER as u8,
            white_scored: 0,
            black_scored: 0,
        }
    }

    /// Sjekk om nåværende spiller har plassert melkespannet
    fn current_player_has_pail(&self) -> bool {
        match self.current_player {
            Player::White => self.white_pail_placed,
            Player::Black => self.black_pail_placed,
        }
    }

    /// Antall tønner nåværende spiller har utenfor brettet
    fn current_player_barrels_off_board(&self) -> u8 {
        match self.current_player {
            Player::White => self.white_barrels_off_board,
            Player::Black => self.black_barrels_off_board,
        }
    }

    /// Spillerens startrad (der nye tønner kan plasseres)
    /// Hvit starter nederst (rad 5), svart starter øverst (rad 0)
    fn starting_row(&self, player: Player) -> usize {
        match player {
            Player::White => BOARD_SIZE - 1,  // Rad 5 (nederst)
            Player::Black => 0,                // Rad 0 (øverst)
        }
    }

    /// Spillerens målrad
    /// Hvit vil til øverst (rad 0), svart vil til nederst (rad 5)
    fn goal_row(&self, player: Player) -> usize {
        match player {
            Player::White => 0,                // Rad 0 (øverst)
            Player::Black => BOARD_SIZE - 1,  // Rad 5 (nederst)
        }
    }

    /// Hent Zobrist hash (for transposition table)
    fn get_hash(&self) -> u64 {
        self.hash
    }

    /// Hent innholdet i en celle
    fn get(&self, pos: Position) -> Option<Cell> {
        if pos.is_valid() {
            Some(self.cells[pos.row as usize][pos.col as usize])
        } else {
            None
        }
    }

    /// Sett innholdet i en celle (for testing/setup)
    fn set(&mut self, pos: Position, cell: Cell) {
        if pos.is_valid() {
            self.cells[pos.row as usize][pos.col as usize] = cell;
        }
    }

    /// Finn posisjonen til en spillers melkespann
    fn find_pail(&self, player: Player) -> Option<Position> {
        let target = match player {
            Player::White => Cell::WhitePail,
            Player::Black => Cell::BlackPail,
        };

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.cells[row][col] == target {
                    return Some(Position::new(row as i8, col as i8));
                }
            }
        }
        None
    }

    /// Finn alle posisjoner med tønner for en spiller
    fn find_barrels(&self, player: Player) -> Vec<Position> {
        let mut barrels = Vec::new();
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.cells[row][col].is_barrel_of(player) {
                    barrels.push(Position::new(row as i8, col as i8));
                }
            }
        }
        barrels
    }

    /// Sjekk om en spiller har vunnet
    /// En spiller vinner når alle 4 tønner har nådd mål (fjernet fra brettet)
    fn check_winner(&self) -> Option<Player> {
        if self.white_scored == BARRELS_PER_PLAYER as u8 {
            return Some(Player::White);
        }
        if self.black_scored == BARRELS_PER_PLAYER as u8 {
            return Some(Player::Black);
        }
        None
    }

    /// Generer alle lovlige trekk for nåværende spiller
    /// Rules:
    /// 1. (Påkrevd FØRSTE TUR) Plasser melkespann på ledig felt
    /// 2. (Påkrevd) ENTEN:
    ///    a) Plasser ny tønne fra utenfor brettet på ledig felt på startrad
    ///    b) Flytt eksisterende tønne på brettet (8 directions + jumps)
    fn generate_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        let player = self.current_player;

        let pail_placed = match player {
            Player::White => self.white_pail_placed,
            Player::Black => self.black_pail_placed,
        };
        let barrels_off = match player {
            Player::White => self.white_barrels_off_board,
            Player::Black => self.black_barrels_off_board,
        };

        let barrels_on_board = self.find_barrels(player);
        let start_row = self.starting_row(player);

        // Generate pail destinations (only when not yet placed - one-time!)
        let pail_destinations: Vec<Option<Position>> = if !pail_placed {
            // First placement: any empty square
            let mut dests = Vec::new();
            for row in 0..BOARD_SIZE {
                for col in 0..BOARD_SIZE {
                    if self.cells[row][col] == Cell::Empty {
                        dests.push(Some(Position::new(row as i8, col as i8)));
                    }
                }
            }
            dests
        } else {
            // Pail already placed - no pail move needed
            vec![None]
        };

        for pail_opt in &pail_destinations {
            // Create temp board with pail placed (if this is first pail placement)
            let temp_board = if let Some(pail_dest) = pail_opt {
                let mut temp = self.clone();
                let pail_cell = match player {
                    Player::White => Cell::WhitePail,
                    Player::Black => Cell::BlackPail,
                };
                temp.cells[pail_dest.row as usize][pail_dest.col as usize] = pail_cell;
                temp
            } else {
                self.clone()
            };

            // Alternativ A: Plasser ny tønne fra utenfor brettet
            if barrels_off > 0 {
                for col in 0..BOARD_SIZE {
                    if temp_board.cells[start_row][col] == Cell::Empty {
                        let to_pos = Position::new(start_row as i8, col as i8);
                        moves.push(Move::place_barrel(*pail_opt, to_pos));
                    }
                }
            }

            // Alternativ B: Flytt eksisterende tønne på brettet
            for &barrel_pos in &barrels_on_board {
                let barrel_moves = temp_board.get_barrel_moves(barrel_pos, player);
                for path in barrel_moves {
                    moves.push(Move::move_barrel(*pail_opt, barrel_pos, path));
                }
            }
        }

        moves
    }

    /// Utfør et trekk på brettet
    fn make_move(&mut self, mv: &Move) -> bool {
        let player = self.current_player;
        let barrel_cell = match player {
            Player::White => Cell::WhiteBarrel,
            Player::Black => Cell::BlackBarrel,
        };

        // 1. Place pail (one-time placement, only when not yet placed)
        if let Some(pail_pos) = mv.place_pail {
            let pail_cell = match player {
                Player::White => Cell::WhitePail,
                Player::Black => Cell::BlackPail,
            };

            // Debug assert: pail should not already be placed
            debug_assert!(
                match player {
                    Player::White => !self.white_pail_placed,
                    Player::Black => !self.black_pail_placed,
                },
                "Pail already placed!"
            );

            // Place pail at position
            let pos = (pail_pos.row as usize, pail_pos.col as usize);
            self.hash ^= ZOBRIST.pieces[pos.0][pos.1][ZobristKeys::piece_index(Cell::Empty)];
            self.hash ^= ZOBRIST.pieces[pos.0][pos.1][ZobristKeys::piece_index(pail_cell)];
            self.cells[pos.0][pos.1] = pail_cell;

            match player {
                Player::White => self.white_pail_placed = true,
                Player::Black => self.black_pail_placed = true,
            }
        }

        // 2. Håndter tønne: enten plassering eller flytting
        let barrel_to = (mv.barrel_to.row as usize, mv.barrel_to.col as usize);
        let goal_row = self.goal_row(player);

        if mv.is_barrel_placement {
            // Plasser ny tønne fra utenfor brettet
            // Reduser antall tønner utenfor brettet
            match player {
                Player::White => self.white_barrels_off_board -= 1,
                Player::Black => self.black_barrels_off_board -= 1,
            }

            // Sjekk om tønnen når mål med en gang (usannsynlig, men håndter det)
            if barrel_to.0 == goal_row {
                // Tønnen scorer! Ikke plasser den på brettet, bare tell poeng
                match player {
                    Player::White => self.white_scored += 1,
                    Player::Black => self.black_scored += 1,
                }
                // Oppdater hash for tom rute (den forblir tom)
            } else {
                // Plasser tønnen på brettet
                self.hash ^= ZOBRIST.pieces[barrel_to.0][barrel_to.1][ZobristKeys::piece_index(Cell::Empty)];
                self.hash ^= ZOBRIST.pieces[barrel_to.0][barrel_to.1][ZobristKeys::piece_index(barrel_cell)];
                self.cells[barrel_to.0][barrel_to.1] = barrel_cell;
            }
        } else {
            // Flytt eksisterende tønne
            let barrel_from = mv.barrel_from.unwrap();
            let from = (barrel_from.row as usize, barrel_from.col as usize);

            // Fjern tønnen fra gammel posisjon
            self.hash ^= ZOBRIST.pieces[from.0][from.1][ZobristKeys::piece_index(barrel_cell)];
            self.hash ^= ZOBRIST.pieces[from.0][from.1][ZobristKeys::piece_index(Cell::Empty)];
            self.cells[from.0][from.1] = Cell::Empty;

            // Sjekk om tønnen når mål
            if barrel_to.0 == goal_row {
                // Tønnen scorer! Fjernes fra spillet
                match player {
                    Player::White => self.white_scored += 1,
                    Player::Black => self.black_scored += 1,
                }
                // Ikke plasser tønnen på brettet
            } else {
                // Plasser tønnen på ny posisjon
                self.hash ^= ZOBRIST.pieces[barrel_to.0][barrel_to.1][ZobristKeys::piece_index(Cell::Empty)];
                self.hash ^= ZOBRIST.pieces[barrel_to.0][barrel_to.1][ZobristKeys::piece_index(barrel_cell)];
                self.cells[barrel_to.0][barrel_to.1] = barrel_cell;
            }
        }

        // 3. Bytt spiller og oppdater hash
        self.hash ^= ZOBRIST.player_to_move;
        self.current_player = player.opponent();
        self.move_count += 1;

        true
    }

    /// Konverter brettet til en numpy-vennlig liste
    fn to_array(&self) -> Vec<Vec<i8>> {
        self.cells
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| match cell {
                        Cell::Empty => 0,
                        Cell::WhiteBarrel => 1,
                        Cell::BlackBarrel => -1,
                        Cell::WhitePail => 2,
                        Cell::BlackPail => -2,
                    })
                    .collect()
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("{}", self)
    }

    /// Vis brettet som ASCII
    fn display(&self) -> String {
        format!("{}", self)
    }
}

// Hjelpefunksjoner (ikke eksponert til Python)
impl Board {
    /// Finn alle mulige trekk for en tønne (inkludert hopp-sekvenser)
    /// Supports 8 directions (orthogonal + diagonal)
    fn get_barrel_moves(&self, from: Position, player: Player) -> Vec<Vec<Position>> {
        let mut all_paths = Vec::new();
        // 8 directions: orthogonal + diagonal
        let directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  // orthogonal
            (1, 1), (1, -1), (-1, 1), (-1, -1) // diagonal
        ];

        // Enkle trekk (ett felt) - now 8 directions
        for (dr, dc) in directions {
            let to = Position::new(from.row + dr, from.col + dc);
            if to.is_valid() {
                if let Some(cell) = self.get(to) {
                    if cell == Cell::Empty {
                        all_paths.push(vec![to]);
                    }
                }
            }
        }

        // Hopp-sekvenser
        let mut visited = vec![vec![false; BOARD_SIZE]; BOARD_SIZE];
        visited[from.row as usize][from.col as usize] = true;
        self.find_jump_sequences(from, player, &mut visited, &mut Vec::new(), &mut all_paths);

        all_paths
    }

    /// Rekursivt finn alle hopp-sekvenser
    /// Supports 8 directions, cannot jump over pails
    fn find_jump_sequences(
        &self,
        current: Position,
        player: Player,
        visited: &mut Vec<Vec<bool>>,
        current_path: &mut Vec<Position>,
        all_paths: &mut Vec<Vec<Position>>,
    ) {
        // 8 directions: orthogonal + diagonal
        let directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  // orthogonal
            (1, 1), (1, -1), (-1, 1), (-1, -1) // diagonal
        ];

        // Opponent's pail blocks jumps
        let opponent_pail = match player {
            Player::White => Cell::BlackPail,
            Player::Black => Cell::WhitePail,
        };

        for (dr, dc) in directions {
            // Sjekk om det er en tønne å hoppe over
            let over = Position::new(current.row + dr, current.col + dc);
            let landing = Position::new(current.row + 2 * dr, current.col + 2 * dc);

            if !over.is_valid() || !landing.is_valid() {
                continue;
            }

            // Can jump over barrels or OWN pail, but NOT opponent's pail
            if let (Some(over_cell), Some(landing_cell)) = (self.get(over), self.get(landing)) {
                let can_jump_over = over_cell.is_barrel() || over_cell.is_pail();
                let is_opponent_pail = over_cell == opponent_pail;

                if can_jump_over
                    && !is_opponent_pail  // Cannot jump over opponent's pail
                    && landing_cell == Cell::Empty
                    && !visited[landing.row as usize][landing.col as usize]
                {
                    // Gyldig hopp!
                    visited[landing.row as usize][landing.col as usize] = true;
                    current_path.push(landing);

                    // Lagre denne stien
                    all_paths.push(current_path.clone());

                    // Fortsett å lete etter flere hopp
                    self.find_jump_sequences(landing, player, visited, current_path, all_paths);

                    current_path.pop();
                    visited[landing.row as usize][landing.col as usize] = false;
                }
            }
        }
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  0 1 2 3 4 5 6 7")?;
        writeln!(f, " +-----------------+")?;
        for row in 0..BOARD_SIZE {
            write!(f, "{}| ", row)?;
            for col in 0..BOARD_SIZE {
                let ch = match self.cells[row][col] {
                    Cell::Empty => '.',
                    Cell::WhiteBarrel => 'W',
                    Cell::BlackBarrel => 'B',
                    Cell::WhitePail => 'w',
                    Cell::BlackPail => 'b',
                };
                write!(f, "{} ", ch)?;
            }
            writeln!(f, "|")?;
        }
        writeln!(f, " +-----------------+")?;
        writeln!(
            f,
            "Turn: {:?} | Move: {}",
            self.current_player, self.move_count
        )
    }
}

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

/// Transposition Table - cache av evaluerte posisjoner
/// Bruker depth-preferred replacement med generasjons-aging
struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
    hits: u64,
    misses: u64,
    generation: u8,   // Økes hver gang søket starter
}

impl TranspositionTable {
    fn new(size: usize) -> Self {
        TranspositionTable {
            entries: vec![None; size],
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
        (hash as usize) % self.entries.len()
    }

    fn probe(&mut self, hash: u64) -> Option<&TTEntry> {
        let idx = self.index(hash);
        if let Some(ref entry) = self.entries[idx] {
            // Check hash match AND generation (stale entries are invalid)
            if entry.hash == hash && entry.generation == self.generation {
                self.hits += 1;
                return Some(entry);
            }
        }
        self.misses += 1;
        None
    }

    fn store(&mut self, hash: u64, depth: u8, score: i32, flag: TTFlag, best_move: Option<Move>) {
        let idx = self.index(hash);

        // Depth-preferred replacement strategy med generation tie-breaker
        let should_replace = match &self.entries[idx] {
            None => true,
            Some(existing) => {
                // Always replace if same position (update with potentially better info)
                if existing.hash == hash {
                    true
                }
                // Replace old generations (stale entries)
                else if existing.generation != self.generation {
                    true
                }
                // Replace if new entry has greater or equal depth
                else {
                    depth >= existing.depth
                }
            }
        };

        if should_replace {
            self.entries[idx] = Some(TTEntry {
                hash,
                depth,
                score,
                flag,
                generation: self.generation,
                best_move,
            });
        }
    }

    fn clear(&mut self) {
        // O(1) clear using generation counter - stale entries ignored by probe()
        // Wrap generation to invalidate all existing entries
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

    /// Skip relational features during NNUE evaluation (for benchmarking)
    fn set_skip_relational(&mut self, skip: bool) {
        self.inner.skip_relational = skip;
    }
}


// ============================================================================
// NNUE - NEURAL NETWORK EVALUATOR
// ============================================================================

use serde::Deserialize;

/// NNUE vekter lastet fra JSON
#[derive(Deserialize)]
struct NNUEWeights {
    fc1_weight: Vec<Vec<f32>>,  // [hidden1][144]
    fc1_bias: Vec<f32>,          // [hidden1]
    fc2_weight: Vec<Vec<f32>>,  // [hidden2][hidden1]
    fc2_bias: Vec<f32>,          // [hidden2]
    fc3_weight: Vec<Vec<f32>>,  // [1][hidden2]
    fc3_bias: Vec<f32>,          // [1]
}

#[derive(Deserialize)]
struct NNUEModel {
    hidden1: usize,
    hidden2: usize,
    weights: NNUEWeights,
}

/// Effektiv NNUE-evaluator i Rust
///
/// Arkitektur: Input(144) -> FC(64) -> ReLU -> FC(32) -> ReLU -> FC(1) -> Tanh
pub struct NNUE {
    // Flattede vekter for bedre cache-ytelse
    fc1_weight: Vec<f32>,  // [hidden1 * 144]
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<f32>,  // [hidden2 * hidden1]
    fc2_bias: Vec<f32>,
    fc3_weight: Vec<f32>,  // [1 * hidden2]
    fc3_bias: f32,
    hidden1: usize,
    hidden2: usize,
}

impl NNUE {
    /// Last NNUE-modell fra JSON-fil
    pub fn load(json_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json_str = std::fs::read_to_string(json_path)?;
        let model: NNUEModel = serde_json::from_str(&json_str)?;

        // Flatten vekter for bedre cache-ytelse
        let fc1_weight: Vec<f32> = model.weights.fc1_weight.into_iter().flatten().collect();
        let fc2_weight: Vec<f32> = model.weights.fc2_weight.into_iter().flatten().collect();
        let fc3_weight: Vec<f32> = model.weights.fc3_weight.into_iter().flatten().collect();

        Ok(Self {
            fc1_weight,
            fc1_bias: model.weights.fc1_bias,
            fc2_weight,
            fc2_bias: model.weights.fc2_bias,
            fc3_weight,
            fc3_bias: model.weights.fc3_bias[0],
            hidden1: model.hidden1,
            hidden2: model.hidden2,
        })
    }

    /// Konverter brett til input-vektor (one-hot encoding)
    /// Kanal 0: Hvit tonne, Kanal 1: Svart tonne, Kanal 2: Hvit spann, Kanal 3: Svart spann
    #[inline]
    fn board_to_input(board: &Board) -> [f32; 144] {
        let mut input = [0.0f32; 144];

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let idx = row * BOARD_SIZE + col;
                match board.cells[row][col] {
                    Cell::WhiteBarrel => input[idx * 4] = 1.0,
                    Cell::BlackBarrel => input[idx * 4 + 1] = 1.0,
                    Cell::WhitePail => input[idx * 4 + 2] = 1.0,
                    Cell::BlackPail => input[idx * 4 + 3] = 1.0,
                    Cell::Empty => {}
                }
            }
        }

        input
    }

    /// Evaluer en posisjon med NNUE
    /// Returnerer score mellom -1.0 (svart vinner) og +1.0 (hvit vinner)
    #[inline]
    pub fn evaluate(&self, board: &Board) -> f32 {
        let input = Self::board_to_input(board);

        // Layer 1: FC + ReLU
        let mut hidden1 = vec![0.0f32; self.hidden1];
        for i in 0..self.hidden1 {
            let mut sum = self.fc1_bias[i];
            let weight_offset = i * 144;
            for j in 0..144 {
                sum += input[j] * self.fc1_weight[weight_offset + j];
            }
            hidden1[i] = sum.max(0.0);  // ReLU
        }

        // Layer 2: FC + ReLU
        let mut hidden2 = vec![0.0f32; self.hidden2];
        for i in 0..self.hidden2 {
            let mut sum = self.fc2_bias[i];
            let weight_offset = i * self.hidden1;
            for j in 0..self.hidden1 {
                sum += hidden1[j] * self.fc2_weight[weight_offset + j];
            }
            hidden2[i] = sum.max(0.0);  // ReLU
        }

        // Layer 3: FC + Tanh
        let mut output = self.fc3_bias;
        for i in 0..self.hidden2 {
            output += hidden2[i] * self.fc3_weight[i];
        }

        // Tanh aktivering
        output.tanh()
    }

    /// Evaluer og skaler til centipawn (for kompatibilitet med eksisterende engine)
    #[inline]
    pub fn evaluate_cp(&self, board: &Board) -> i32 {
        // Skaler fra [-1, 1] til [-1000, 1000] centipawn
        (self.evaluate(board) * 1000.0) as i32
    }
}

// ============================================================================
// INKREMENTELL NNUE - Rask evaluering med delta-oppdateringer
// ============================================================================

/// Input feature indeks: sq * 4 + piece_type
/// piece_type: 0=WhiteBarrel, 1=BlackBarrel, 2=WhitePail, 3=BlackPail
#[inline]
const fn feature_index(sq: usize, piece_type: usize) -> usize {
    sq * 4 + piece_type
}

/// Representerer en endring i input-features
#[derive(Clone, Copy, Debug)]
pub struct FeatureDelta {
    pub index: u8,   // 0-143: hvilken input-feature
    pub delta: i8,   // +1 (brikke lagt til) eller -1 (fjernet)
}

/// Accumulator for inkrementell NNUE
/// Cacher layer 1 output for å unngå full reberegning
#[derive(Clone)]
pub struct Accumulator {
    /// Pre-activation verdier (før ReLU)
    pub pre_activation: [f32; 64],
    /// Post-activation verdier (etter ReLU) - brukes av layer 2
    pub post_activation: [f32; 64],
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            pre_activation: [0.0; 64],
            post_activation: [0.0; 64],
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

/// Inkrementell NNUE evaluator
/// Cacher layer 1 og oppdaterer kun endrede features
///
/// Supports two architectures:
///   - Legacy (144 features): Base position features only
///   - Enhanced (157 features): Base + relational features
pub struct IncrementalNNUE {
    // Vekter (delt med standard NNUE)
    fc1_weight: Vec<f32>,  // [hidden1 * input_size] where input_size is 144 or 157
    fc1_weight_t: Vec<f32>,  // Transposed: [input_size * hidden1] for cache-friendly feature updates
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<f32>,
    fc2_bias: Vec<f32>,
    fc3_weight: Vec<f32>,
    fc3_bias: f32,
    hidden1: usize,
    hidden2: usize,
    input_size: usize,  // 144 (legacy) or 157 (with relational features)
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

    /// Compute and add relational features (only cheap, non-inferable features)
    ///
    /// Relational features (3 total):
    ///   - White barrels scored (1): normalized by 4 - can't infer from position
    ///   - Black barrels scored (1): normalized by 4 - can't infer from position
    ///   - Current player (1): +1 white, -1 black - not in base features
    ///
    /// Removed (can be learned from base features):
    ///   - Barrel distances (NN can learn row 0/5 are goals)
    ///   - Pail placed (if pail is on board, it's placed)
    #[inline]
    fn add_relational_features(&self, bb: &BitBoard, acc: &mut Accumulator) {
        if self.input_size <= BASE_FEATURES {
            return; // Legacy model without relational features
        }

        let hidden1 = self.hidden1;
        let base = BASE_FEATURES;

        // Only 3 cheap features - no loops, no sorting, just memory reads
        let rel_features: [f32; 3] = [
            bb.white_scored as f32 / 4.0,  // [0] White scored (0.0 - 1.0)
            bb.black_scored as f32 / 4.0,  // [1] Black scored (0.0 - 1.0)
            match bb.current_player {       // [2] Current player
                Player::White => 1.0,
                Player::Black => -1.0,
            },
        ];

        // Add contribution of relational features to accumulator using transposed weights
        for (feat_idx, &feat_val) in rel_features.iter().enumerate() {
            if feat_val == 0.0 {
                continue; // Skip zero features
            }
            let weight_idx = base + feat_idx;
            let base_idx = weight_idx * hidden1;  // Contiguous weights

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
    fn add_features_from_bitboard(&self, bb: &BitBoard, acc: &mut Accumulator) {
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

    /// Beregn feature deltas for et trekk
    pub fn compute_move_deltas(&self, bb: &BitBoard, mv: &BitMove) -> Vec<FeatureDelta> {
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
            // Ny tønne plasseres
            let to_sq = mv.barrel_to() as usize;
            let goal_row = bb.goal_row(player);
            let (to_row, _) = sq_to_coords(to_sq);

            // Hvis den ikke scorer, legg til feature
            if to_row != goal_row {
                deltas.push(FeatureDelta {
                    index: feature_index(to_sq, barrel_piece) as u8,
                    delta: 1,
                });
            }
        } else {
            // Eksisterende tønne flyttes
            let from_sq = mv.barrel_from().unwrap() as usize;
            let to_sq = mv.barrel_to() as usize;
            let goal_row = bb.goal_row(player);
            let (to_row, _) = sq_to_coords(to_sq);

            // Fjern fra gammel posisjon
            deltas.push(FeatureDelta {
                index: feature_index(from_sq, barrel_piece) as u8,
                delta: -1,
            });

            // Legg til på ny posisjon (hvis ikke scoring)
            if to_row != goal_row {
                deltas.push(FeatureDelta {
                    index: feature_index(to_sq, barrel_piece) as u8,
                    delta: 1,
                });
            }
        }

        deltas
    }

    /// Evaluer fra ferdig accumulator (layer 2 og 3) med SIMD
    pub fn evaluate_from_accumulator(&self, acc: &Accumulator) -> f32 {
        // Layer 2: FC + ReLU (64 inputs -> 32 outputs)
        // Use SIMD to compute dot products 8 elements at a time
        let mut hidden2 = [0.0f32; 32];

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

        // Add relational features (cheap - only 3 features now)
        if self.input_size > BASE_FEATURES {
            let input_size = self.input_size;
            let base = BASE_FEATURES;
            let rel_features: [f32; 3] = [
                bb.white_scored as f32 / 4.0,
                bb.black_scored as f32 / 4.0,
                match bb.current_player {
                    Player::White => 1.0,
                    Player::Black => -1.0,
                },
            ];
            for (feat_idx, &feat_val) in rel_features.iter().enumerate() {
                if feat_val == 0.0 { continue; }
                let weight_idx = base + feat_idx;
                for i in 0..self.hidden1 {
                    eval_acc.pre_activation[i] += feat_val * self.fc1_weight[i * input_size + weight_idx];
                }
            }
        }

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
// EVALUATION CACHE - Unngå redundante NNUE-evalueringer
// ============================================================================

/// Størrelse på eval cache (power of 2 for rask modulo)
const EVAL_CACHE_SIZE: usize = 1 << 16; // 65536 entries

/// Entry i eval cache
#[derive(Clone, Copy, Default)]
struct EvalCacheEntry {
    hash: u64,
    score: i32,
    generation: u8,
}

/// Cache for statiske evalueringer
struct EvalCache {
    entries: Vec<EvalCacheEntry>,
    hits: u64,
    misses: u64,
    generation: u8,
}

impl EvalCache {
    fn new() -> Self {
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
    fn probe(&mut self, hash: u64) -> Option<i32> {
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
    fn store(&mut self, hash: u64, score: i32) {
        let idx = self.index(hash);
        self.entries[idx] = EvalCacheEntry { hash, score, generation: self.generation };
    }

    /// Tøm cache - O(1) using generation counter
    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.hits = 0;
        self.misses = 0;
    }

    /// Hit ratio
    fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
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

    // NNUE evaluator
    nnue: Option<IncrementalNNUE>,

    // Accumulator stack
    acc_stack: AccumulatorStack,

    // Working accumulator for evaluation (reused to avoid allocations)
    eval_acc: Accumulator,

    // Skip relational features (for benchmarking)
    skip_relational: bool,

    // LMR reduction table: lmr_table[depth][move_count] = reduction
    // Precomputed using ln(depth) * ln(move_count) / 2.5
    lmr_table: [[u8; 64]; 32],

    // Fallback heuristisk vekter
    pub weight_progress: i32,
    pub weight_center_pail: i32,
    pub weight_blocking: i32,
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
            skip_relational: false,
            lmr_table,
            weight_progress: 100,
            weight_center_pail: 10,
            weight_blocking: 15,
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

    /// Last NNUE-modell
    pub fn load_nnue(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let nnue = IncrementalNNUE::load(path)?;
        self.nnue = Some(nnue);
        Ok(())
    }

    /// Clear NNUE (revert to heuristic evaluation)
    pub fn clear_nnue(&mut self) {
        self.nnue = None;
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
    fn evaluate_heuristic(&self, bb: &BitBoard) -> i32 {
        if let Some(winner) = bb.check_winner() {
            return match winner {
                Player::White => 100_000,
                Player::Black => -100_000,
            };
        }

        let mut score = 0;

        // Poeng for scorede tønner (big bonus)
        score += (bb.white_scored as i32 - bb.black_scored as i32) * 500;

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
        score += (white_threats - black_threats) * 200; // Immediate threats are valuable

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

        // Float NNUE
        if self.nnue.is_some() {
            let base_acc = self.acc_stack.current();
            let pre_activation = base_acc.pre_activation;

            let nnue = self.nnue.as_ref().unwrap();
            let eval_acc = &mut self.eval_acc;

            eval_acc.pre_activation = pre_activation;

            if !self.skip_relational && nnue.input_size > BASE_FEATURES {
                let input_size = nnue.input_size;
                let base = BASE_FEATURES;
                let rel_features: [f32; 3] = [
                    bb.white_scored as f32 / 4.0,
                    bb.black_scored as f32 / 4.0,
                    match bb.current_player {
                        Player::White => 1.0,
                        Player::Black => -1.0,
                    },
                ];
                for (feat_idx, &feat_val) in rel_features.iter().enumerate() {
                    if feat_val == 0.0 { continue; }
                    let weight_idx = base + feat_idx;
                    for i in 0..nnue.hidden1 {
                        eval_acc.pre_activation[i] += feat_val * nnue.fc1_weight[i * input_size + weight_idx];
                    }
                }
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
        if let Some(ref nnue) = self.nnue {
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

        // Finn "taktiske" trekk: tønner som når mål eller er 1 rad unna
        let player = bb.current_player;

        let moves = bb.generate_moves();
        let tactical_moves: Vec<BitMove> = moves
            .into_iter()
            .filter(|mv| {
                let to_sq = mv.barrel_to() as usize;
                let (to_row, _) = sq_to_coords(to_sq);
                // Tønne når mål eller er én rad unna
                let dist_to_goal = if player == Player::White {
                    to_row // White's goal is row 0
                } else {
                    BOARD_SIZE - 1 - to_row // Black's goal is row 5
                };
                dist_to_goal <= 1
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

                // Oppdater accumulator
                if self.nnue.is_some() {
                    let nnue = self.nnue.as_ref().unwrap();
                    let deltas = nnue.compute_move_deltas(bb, &mv);
                    self.acc_stack.push();
                    let acc = self.acc_stack.current_mut();
                    nnue.apply_deltas(acc, &deltas);
                }

                let score = self.quiesce(&new_bb, alpha, beta, false, qsdepth + 1);

                if self.nnue.is_some() {
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

                // Oppdater accumulator
                if self.nnue.is_some() {
                    let nnue = self.nnue.as_ref().unwrap();
                    let deltas = nnue.compute_move_deltas(bb, &mv);
                    self.acc_stack.push();
                    let acc = self.acc_stack.current_mut();
                    nnue.apply_deltas(acc, &deltas);
                }

                let score = self.quiesce(&new_bb, alpha, beta, true, qsdepth + 1);

                if self.nnue.is_some() {
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

        if let Some((tt_depth, tt_score, tt_flag, tt_mv_opt)) = tt_result {
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
        if depth <= 3 {
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
        // NULL MOVE PRUNING
        // ═══════════════════════════════════════════════════════════════
        // If giving opponent a free move still results in a beta cutoff,
        // this position is so good we can prune.
        // Only use when position is already favorable (otherwise unlikely to cutoff)
        let nmp_margin = 50; // Only try NMP if we're at least this much better
        let nmp_allowed = depth >= 4
            && static_eval.abs() < 90_000
            && !bb.has_barrel_near_goal()
            && beta.abs() < 90_000
            && if maximizing {
                static_eval >= beta - nmp_margin
            } else {
                static_eval <= alpha + nmp_margin
            };

        if nmp_allowed {
            // Determine reduction: R=2 for shallow, R=3 for deeper
            let r = if depth >= 6 { 3 } else { 2 };
            let null_depth = depth.saturating_sub(r + 1);

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
        let futility_pruning = depth <= 8
            && static_eval.abs() < 90_000 // Not near mate
            && if maximizing {
                static_eval + FUTILITY_MARGINS[depth as usize] < alpha
            } else {
                static_eval - FUTILITY_MARGINS[depth as usize] > beta
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

            // Oppdater accumulator inkrementelt
            if self.nnue.is_some() {
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
            if self.nnue.is_some() {
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

    /// Iterative deepening search med aspiration windows
    pub fn search_iterative(&mut self, bb: &BitBoard, max_depth: u8) -> (i32, Option<BitMove>) {
        let mut best_score = 0;
        let mut best_move = None;
        let mut total_nodes = 0u64;
        let mut total_quiesce = 0u64;
        let mut total_cutoffs = 0u64;
        let mut total_tt_hits = 0u64;
        let mut prev_score: Option<i32> = None;

        for depth in 1..=max_depth {
            // Bruk forrige score for aspiration windows
            let (score, mv) = self.search_with_aspiration(bb, depth, prev_score);
            prev_score = Some(score);

            total_nodes += self.nodes_searched;
            total_quiesce += self.quiesce_nodes;
            total_cutoffs += self.cutoffs;
            total_tt_hits += self.tt_hits;

            best_score = score;
            if mv.is_some() {
                best_move = mv;
            }

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

// ============================================================================
// PYTHON MODUL
// ============================================================================

/// Python-modul
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Player>()?;
    m.add_class::<Cell>()?;
    m.add_class::<Position>()?;
    m.add_class::<Move>()?;
    m.add_class::<Board>()?;
    m.add_class::<Engine>()?;
    m.add_class::<SearchResult>()?;
    m.add("BOARD_SIZE", BOARD_SIZE)?;
    m.add("BARRELS_PER_PLAYER", BARRELS_PER_PLAYER)?;
    Ok(())
}

// ============================================================================
// TESTER
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Hjelpefunksjon: sett opp et brett med noen brikker
    fn setup_test_board() -> Board {
        let mut board = Board::new();
        // Plasser noen brikker for testing
        board.cells[5][2] = Cell::WhiteBarrel;
        board.cells[5][3] = Cell::WhiteBarrel;
        board.cells[0][2] = Cell::BlackBarrel;
        board.cells[0][3] = Cell::BlackBarrel;
        board.white_barrels_off_board = 2;
        board.black_barrels_off_board = 2;
        board
    }

    /// Test at BitBoard konvertering er korrekt
    #[test]
    fn test_bitboard_conversion() {
        let board = setup_test_board();
        let bb = BitBoard::from_board(&board);
        let board2 = bb.to_board();

        // Sjekk at alle celler er like
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                assert_eq!(
                    board.cells[row][col], board2.cells[row][col],
                    "Mismatch at ({}, {}): {:?} vs {:?}",
                    row, col, board.cells[row][col], board2.cells[row][col]
                );
            }
        }

        // Sjekk game state
        assert_eq!(board.current_player, board2.current_player);
        assert_eq!(board.white_barrels_off_board, board2.white_barrels_off_board);
        assert_eq!(board.black_barrels_off_board, board2.black_barrels_off_board);
    }

    /// Konverter Move til en sammenlignbar nøkkel
    fn move_key(mv: &Move) -> String {
        let pail = match mv.place_pail {
            Some(p) => format!("pail({},{})", p.row, p.col),
            None => "no_pail".to_string(),
        };
        let barrel = if mv.is_barrel_placement {
            format!("place({},{})", mv.barrel_to.row, mv.barrel_to.col)
        } else {
            let from = mv.barrel_from.unwrap();
            format!("move({},{})→({},{})", from.row, from.col, mv.barrel_to.row, mv.barrel_to.col)
        };
        format!("{}:{}", pail, barrel)
    }

    /// Test at BitBoard genererer samme trekk som Board
    #[test]
    fn test_move_generation_equivalence() {
        let board = setup_test_board();
        let bb = BitBoard::from_board(&board);

        // Generer trekk fra begge
        let board_moves = board.generate_moves();
        let bb_moves = bb.generate_moves();

        // Konverter BitMoves til Move for sammenligning
        let bb_moves_converted: Vec<Move> = bb_moves.iter().map(|m| m.to_move()).collect();

        // Samle unike trekk-nøkler
        let board_keys: HashSet<String> = board_moves.iter().map(|m| move_key(m)).collect();
        let bb_keys: HashSet<String> = bb_moves_converted.iter().map(|m| move_key(m)).collect();

        // Finn forskjeller
        let only_in_board: Vec<_> = board_keys.difference(&bb_keys).collect();
        let only_in_bb: Vec<_> = bb_keys.difference(&board_keys).collect();

        assert!(
            only_in_board.is_empty() && only_in_bb.is_empty(),
            "Move generation mismatch!\nOnly in Board ({}):\n{:?}\n\nOnly in BitBoard ({}):\n{:?}",
            only_in_board.len(), only_in_board,
            only_in_bb.len(), only_in_bb
        );

        println!("✓ Both generated {} unique moves", board_keys.len());
    }

    /// Test at make_move/unmake_move fungerer korrekt
    #[test]
    fn test_make_unmake_move() {
        let board = setup_test_board();
        let bb_original = BitBoard::from_board(&board);
        let mut bb = bb_original;

        let moves = bb.generate_moves();
        assert!(!moves.is_empty(), "No moves generated");

        for mv in moves.iter().take(10) {
            let undo = bb.make_move(mv);

            // Sjekk at noe har endret seg
            assert_ne!(bb.current_player, bb_original.current_player);

            // Angre trekket
            bb.unmake_move(&undo);

            // Sjekk at vi er tilbake til original
            assert_eq!(bb.white_barrels, bb_original.white_barrels);
            assert_eq!(bb.black_barrels, bb_original.black_barrels);
            assert_eq!(bb.white_pail, bb_original.white_pail);
            assert_eq!(bb.black_pail, bb_original.black_pail);
            assert_eq!(bb.occupied, bb_original.occupied);
            assert_eq!(bb.current_player, bb_original.current_player);
        }

        println!("✓ make_move/unmake_move works correctly");
    }

    /// Test at prekalkulerte tabeller er korrekte
    #[test]
    fn test_precomputed_tables() {
        // Test ADJACENT
        // Hjørne (0,0) har 2 naboer
        let adj_00 = ADJACENT[sq(0, 0)];
        assert_eq!(adj_00.count_ones(), 2);
        assert!(adj_00 & bit(sq(0, 1)) != 0); // høyre
        assert!(adj_00 & bit(sq(1, 0)) != 0); // ned

        // Senter (2,2) har 4 naboer
        let adj_22 = ADJACENT[sq(2, 2)];
        assert_eq!(adj_22.count_ones(), 4);

        // Test JUMP_LANDING
        // Fra (2,2) kan vi hoppe i alle 4 retninger
        for dir in 0..4 {
            assert!(JUMP_LANDING[sq(2, 2)][dir] >= 0);
        }

        // Fra (0,0) kan vi bare hoppe ned og høyre
        assert!(JUMP_LANDING[sq(0, 0)][0] < 0); // opp - ugyldig
        assert!(JUMP_LANDING[sq(0, 0)][1] >= 0); // ned - gyldig
        assert!(JUMP_LANDING[sq(0, 0)][2] < 0); // venstre - ugyldig
        assert!(JUMP_LANDING[sq(0, 0)][3] >= 0); // høyre - gyldig

        println!("✓ Precomputed tables are correct");
    }

    /// Benchmark: sammenlign ytelse mellom Board og BitBoard
    #[test]
    fn bench_move_generation() {
        let board = setup_test_board();
        let bb = BitBoard::from_board(&board);

        const ITERATIONS: u32 = 10_000;

        // Benchmark Board
        let start = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            let moves = board.generate_moves();
            std::hint::black_box(moves);
        }
        let board_time = start.elapsed();

        // Benchmark BitBoard
        let start = std::time::Instant::now();
        for _ in 0..ITERATIONS {
            let moves = bb.generate_moves();
            std::hint::black_box(moves);
        }
        let bb_time = start.elapsed();

        println!("Board move gen: {:?} ({} iterations)", board_time, ITERATIONS);
        println!("BitBoard move gen: {:?} ({} iterations)", bb_time, ITERATIONS);
        println!("Speedup: {:.2}x", board_time.as_nanos() as f64 / bb_time.as_nanos() as f64);
    }

    /// Perft test: tell antall noder på ulike dybder
    fn perft(bb: &BitBoard, depth: u8) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = bb.generate_moves();
        if depth == 1 {
            return moves.len() as u64;
        }

        let mut count = 0u64;
        for mv in moves {
            let mut new_bb = *bb;
            let _undo = new_bb.make_move(&mv);
            count += perft(&new_bb, depth - 1);
        }
        count
    }

    #[test]
    fn test_perft() {
        let board = setup_test_board();
        let bb = BitBoard::from_board(&board);

        // Kjør perft på lave dybder
        for depth in 1..=3 {
            let count = perft(&bb, depth);
            println!("Perft depth {}: {} nodes", depth, count);
        }
    }

    /// Test BitBoardEngine søk
    #[test]
    fn test_bitboard_engine_search() {
        let board = setup_test_board();
        let bb = BitBoard::from_board(&board);

        let mut engine = BitBoardEngine::new();
        let (score, best_move) = engine.search(&bb, 3);

        println!("Search depth 3: score={}, nodes={}", score, engine.nodes_searched);
        assert!(best_move.is_some(), "No move found");
    }

    /// Benchmark: BitBoardEngine søketid på ulike dybder
    #[test]
    fn bench_search_depths() {
        println!("\n{}", "=".repeat(70));
        println!("BENCHMARK: BitBoardEngine search time at different depths");
        println!("{}\n", "=".repeat(70));

        // Use the test board with some pieces
        let board = setup_test_board();
        let bb = BitBoard::from_board(&board);

        println!("{:>5} | {:>12} | {:>12} | {:>12} | {:>10}",
                 "Depth", "Time", "Nodes", "Cutoffs", "NPS");
        println!("{}", "-".repeat(70));

        for depth in 1..=8 {
            let mut engine = BitBoardEngine::new();
            engine.clear_tt();

            let start = std::time::Instant::now();
            let (_score, _) = engine.search(&bb, depth);
            let elapsed = start.elapsed();

            let nps = if elapsed.as_secs_f64() > 0.0 {
                (engine.nodes_searched as f64 / elapsed.as_secs_f64()) as u64
            } else {
                0
            };

            println!("{:>5} | {:>12.3?} | {:>12} | {:>12} | {:>10}",
                     depth, elapsed, engine.nodes_searched, engine.cutoffs, nps);

            // Stop if taking too long
            if elapsed.as_secs() > 30 {
                println!("\n(Stopped - depth {} took over 30 seconds)", depth);
                break;
            }
        }

        println!("\nScore at max depth: {}", {
            let mut engine = BitBoardEngine::new();
            let (score, _) = engine.search(&bb, 5);
            score
        });
    }
}
