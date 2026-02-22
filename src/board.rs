use pyo3::prelude::*;
use std::fmt;

// ============================================================================
// KONSTANTER
// ============================================================================

/// Board size (6x6)
pub const BOARD_SIZE: usize = 6;
pub const NUM_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
pub const BARRELS_PER_PLAYER: usize = 4;
pub const MAX_DEPTH: usize = 32;

// ============================================================================
// BITBOARD - Rask brettrepresentasjon med 64-bit integers
// ============================================================================

/// Konverter (rad, kolonne) til bitindex
/// Rad 0: bits 0-5, Rad 1: bits 6-11, ..., Rad 5: bits 30-35
#[inline(always)]
pub const fn sq(row: usize, col: usize) -> usize {
    row * BOARD_SIZE + col
}

/// Konverter bitindex til (rad, kolonne)
#[inline(always)]
pub const fn sq_to_coords(sq: usize) -> (usize, usize) {
    (sq / BOARD_SIZE, sq % BOARD_SIZE)
}

/// Bitmask for én rute
#[inline(always)]
pub const fn bit(sq: usize) -> u64 {
    1u64 << sq
}

// ─────────────────────────────────────────────────────────────────────────────
// Prekalkulerte oppslagstabeller (const, kompilert inn i binærfilen)
// ─────────────────────────────────────────────────────────────────────────────

/// Maske for hver rad (6 bits per rad)
pub const ROW_MASK: [u64; BOARD_SIZE] = {
    let mut masks = [0u64; BOARD_SIZE];
    let mut row = 0;
    while row < BOARD_SIZE {
        masks[row] = 0b111111u64 << (row * BOARD_SIZE);
        row += 1;
    }
    masks
};

/// Maske for hver kolonne (1 bit per rad i den kolonnen)
pub const COL_MASK: [u64; BOARD_SIZE] = {
    let mut masks = [0u64; BOARD_SIZE];
    let mut col = 0;
    while col < BOARD_SIZE {
        let mut mask = 0u64;
        let mut row = 0;
        while row < BOARD_SIZE {
            mask |= 1u64 << (row * BOARD_SIZE + col);
            row += 1;
        }
        masks[col] = mask;
        col += 1;
    }
    masks
};

/// Naboer for hvert felt (alle 8 retninger: ortogonalt + diagonalt)
pub const ADJACENT: [u64; NUM_SQUARES] = {
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
pub const NUM_JUMP_DIRS: usize = 8;

/// For hvert felt og retning: feltet som hoppes over (-1 hvis ugyldig)
/// Directions: 0=Up, 1=Down, 2=Left, 3=Right, 4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight
pub const JUMP_OVER: [[i8; NUM_JUMP_DIRS]; NUM_SQUARES] = {
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
pub const JUMP_LANDING: [[i8; NUM_JUMP_DIRS]; NUM_SQUARES] = {
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
        // Stack: (current_sq, visited_mask, path_buf, path_len)
        // Fixed-size path buffer eliminates per-jump heap allocations
        let mut stack: Vec<(u8, u64, [u8; 8], u8)> = Vec::with_capacity(16);

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
                let mut path = [0u8; 8];
                path[0] = landing as u8;
                let visited = visited_start | landing_bit;

                // Legg til dette trekket
                moves.push(BitMove::new_move(from_sq, landing as u8, &path[..1], pail_opt));

                // Push til stack for å fortsette søket
                stack.push((landing as u8, visited, path, 1));
            }
        }

        // DFS
        while let Some((current, visited, path, path_len)) = stack.pop() {
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
                    let mut new_path = path; // 8-byte stack copy (no heap alloc)
                    let new_len = (path_len as usize).min(7); // safety cap
                    new_path[new_len] = landing as u8;
                    let new_path_len = (path_len + 1).min(8);
                    let new_visited = visited | landing_bit;

                    // Legg til trekket
                    let to = landing as u8;
                    moves.push(BitMove::new_move(from_sq, to, &new_path[..new_path_len as usize], pail_opt));

                    // Fortsett søket
                    stack.push((to, new_visited, new_path, new_path_len));
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

    /// Policy index for AlphaZero training: encodes from/to as index in [0, 1331].
    fn policy_index(&self) -> u16 {
        let to_idx = (self.barrel_to.row * 6 + self.barrel_to.col) as u16;
        let from_idx = if self.is_barrel_placement {
            36u16
        } else if let Some(pos) = &self.barrel_from {
            (pos.row * 6 + pos.col) as u16
        } else {
            36u16
        };
        from_idx * 36 + to_idx
    }
}

/// Spillbrettet og tilstanden
#[pyclass]
#[derive(Clone)]
pub struct Board {
    pub(crate) cells: [[Cell; BOARD_SIZE]; BOARD_SIZE],
    #[pyo3(get)]
    pub(crate) current_player: Player,
    #[pyo3(get)]
    pub(crate) move_count: u32,
    pub(crate) hash: u64,  // Zobrist hash for transposition table
    #[pyo3(get)]
    pub(crate) white_pail_placed: bool,   // Har hvit plassert melkespannet?
    #[pyo3(get)]
    pub(crate) black_pail_placed: bool,   // Har svart plassert melkespannet?
    #[pyo3(get)]
    pub(crate) white_barrels_off_board: u8,  // Antall hvite tønner som ikke er plassert ennå
    #[pyo3(get)]
    pub(crate) black_barrels_off_board: u8,  // Antall svarte tønner som ikke er plassert ennå
    #[pyo3(get)]
    pub(crate) white_scored: u8,  // Antall hvite tønner som har nådd mål (fjernet fra brettet)
    #[pyo3(get)]
    pub(crate) black_scored: u8,  // Antall svarte tønner som har nådd mål (fjernet fra brettet)
}

#[pymethods]
impl Board {
    /// Opprett et nytt brett - alle brikker starter UTENFOR brettet
    #[new]
    pub(crate) fn new() -> Self {
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
    pub(crate) fn generate_moves(&self) -> Vec<Move> {
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

    /// Create a copy of the board (needed for MCTS tree search)
    fn copy(&self) -> Board {
        self.clone()
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
