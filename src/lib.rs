use pyo3::prelude::*;

pub mod board;
pub mod nnue;
pub mod search;
pub mod mcts;

// Re-export everything for backward compatibility
pub use board::*;
pub use nnue::*;
pub use search::{BitBoardEngine, Engine, SearchResult, TT_SIZE};

/// Decode a 164-feature dense row into HalfPail sparse indices + dense features.
///
/// This is the hot path for training data preparation — called ~63M times per epoch.
/// Moving it from Python to Rust gives ~20-50x speedup.
///
/// Accepts either a list of 164 floats or raw bytes (656 bytes = 164 × f32).
///
/// Returns: (white_indices: list[int], black_indices: list[int], dense_6: list[float])
#[pyfunction]
fn decode_halfpail(data: &Bound<'_, pyo3::types::PyAny>) -> PyResult<(Vec<u16>, Vec<u16>, Vec<f32>)> {
    let row: Vec<f32> = if let Ok(bytes) = data.extract::<Vec<u8>>() {
        // Fast path: raw bytes (656 bytes = 164 × f32 little-endian)
        if bytes.len() != 164 * 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected 656 bytes (164×f32), got {}", bytes.len())
            ));
        }
        bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else if let Ok(list) = data.extract::<Vec<f32>>() {
        // Fallback: list of floats
        if list.len() != 164 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected 164 features, got {}", list.len())
            ));
        }
        list
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected list of 164 floats or 656 bytes"
        ));
    };

    // Decode board from first 144 features (36 squares × 4 piece types)
    let mut white_barrels: Vec<usize> = Vec::with_capacity(5);
    let mut black_barrels: Vec<usize> = Vec::with_capacity(5);
    let mut white_pail: Option<usize> = None;
    let mut black_pail: Option<usize> = None;

    for sq in 0..NUM_SQUARES {
        let base = sq * 4;
        if row[base] > 0.5 { white_barrels.push(sq); }
        if row[base + 1] > 0.5 { black_barrels.push(sq); }
        if row[base + 2] > 0.5 { white_pail = Some(sq); }
        if row[base + 3] > 0.5 { black_pail = Some(sq); }
    }

    // Extract scored counts and current player from relational features
    let rel = &row[144..];
    let white_scored = (rel[8] * 4.0).round() as i32;
    let black_scored = (rel[9] * 4.0).round() as i32;
    let current_player: f32 = if rel[12] > 0.0 { 1.0 } else { -1.0 };

    // White perspective: bucket = white pail position
    let w_bucket = white_pail.unwrap_or(NUM_SQUARES);
    let mut white_indices: Vec<u16> = Vec::with_capacity(10);
    for &sq in &white_barrels {
        white_indices.push(halfpail_feature_index(w_bucket, sq, 0));
    }
    for &sq in &black_barrels {
        white_indices.push(halfpail_feature_index(w_bucket, sq, 1));
    }
    if let Some(bp) = black_pail {
        white_indices.push(halfpail_feature_index(w_bucket, bp, 2));
    }

    // Black perspective: bucket = black pail position
    let b_bucket = black_pail.unwrap_or(NUM_SQUARES);
    let mut black_indices: Vec<u16> = Vec::with_capacity(10);
    for &sq in &black_barrels {
        black_indices.push(halfpail_feature_index(b_bucket, sq, 0));
    }
    for &sq in &white_barrels {
        black_indices.push(halfpail_feature_index(b_bucket, sq, 1));
    }
    if let Some(wp) = white_pail {
        black_indices.push(halfpail_feature_index(b_bucket, wp, 2));
    }

    // Dense features (6 values)
    let wb_on_board = white_barrels.len() as f32;
    let bb_on_board = black_barrels.len() as f32;
    let dense = vec![
        white_scored as f32 / 4.0,
        black_scored as f32 / 4.0,
        (white_scored - black_scored) as f32 / 4.0,
        current_player,
        wb_on_board / 4.0,
        bb_on_board / 4.0,
    ];

    Ok((white_indices, black_indices, dense))
}

/// Batch-decode multiple 164-feature rows into packed HalfPail tensors.
///
/// Accepts raw bytes from numpy arrays (via .tobytes()) for zero-copy transfer,
/// or falls back to list-of-float for compatibility.
///
/// Returns everything the collate_fn would produce, skipping all Python-level overhead:
///   (white_indices, white_offsets, black_indices, black_offsets, dense_flat, labels)
/// All as flat Vec ready to be wrapped in torch tensors.
#[pyfunction]
fn decode_halfpail_batch(
    data: Vec<f32>,      // N * 164 floats, row-major (from numpy .ravel().tolist())
    labels: Vec<f32>,    // N floats
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<f32>, Vec<f32>)> {
    let n = labels.len();
    if data.len() != n * 164 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Expected {} floats ({}×164), got {}", n * 164, n, data.len())
        ));
    }
    let data_owned = data;

    let mut all_w_idx: Vec<i64> = Vec::with_capacity(n * 10);
    let mut all_b_idx: Vec<i64> = Vec::with_capacity(n * 10);
    let mut w_offsets: Vec<i64> = Vec::with_capacity(n);
    let mut b_offsets: Vec<i64> = Vec::with_capacity(n);
    let mut dense_flat: Vec<f32> = Vec::with_capacity(n * HALFPAIL_DENSE);
    let mut labels_out: Vec<f32> = Vec::with_capacity(n);

    for i in 0..n {
        let row = &data_owned[i * 164..(i + 1) * 164];

        // Record offsets before adding indices
        w_offsets.push(all_w_idx.len() as i64);
        b_offsets.push(all_b_idx.len() as i64);

        // Decode board (BARRELS_PER_PLAYER=4, but use extra space for safety)
        let mut white_barrels: [usize; 8] = [0; 8];
        let mut black_barrels: [usize; 8] = [0; 8];
        let mut n_wb: usize = 0;
        let mut n_bb: usize = 0;
        let mut white_pail: usize = NUM_SQUARES; // 36 = not placed
        let mut black_pail: usize = NUM_SQUARES;

        for sq in 0..NUM_SQUARES {
            let base = sq * 4;
            if row[base] > 0.5 && n_wb < 8 { white_barrels[n_wb] = sq; n_wb += 1; }
            if row[base + 1] > 0.5 && n_bb < 8 { black_barrels[n_bb] = sq; n_bb += 1; }
            if row[base + 2] > 0.5 { white_pail = sq; }
            if row[base + 3] > 0.5 { black_pail = sq; }
        }

        let rel = &row[144..];

        // White perspective
        for j in 0..n_wb {
            all_w_idx.push(halfpail_feature_index(white_pail, white_barrels[j], 0) as i64);
        }
        for j in 0..n_bb {
            all_w_idx.push(halfpail_feature_index(white_pail, black_barrels[j], 1) as i64);
        }
        if black_pail < NUM_SQUARES {
            all_w_idx.push(halfpail_feature_index(white_pail, black_pail, 2) as i64);
        }

        // Black perspective
        for j in 0..n_bb {
            all_b_idx.push(halfpail_feature_index(black_pail, black_barrels[j], 0) as i64);
        }
        for j in 0..n_wb {
            all_b_idx.push(halfpail_feature_index(black_pail, white_barrels[j], 1) as i64);
        }
        if white_pail < NUM_SQUARES {
            all_b_idx.push(halfpail_feature_index(black_pail, white_pail, 2) as i64);
        }

        // Dense features: all 20 relational features from training data (features 144-163)
        for k in 0..HALFPAIL_DENSE {
            dense_flat.push(rel[k]);
        }

        labels_out.push(labels[i]);
    }

    Ok((all_w_idx, w_offsets, all_b_idx, b_offsets, dense_flat, labels_out))
}

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
    m.add_function(wrap_pyfunction!(decode_halfpail, m)?)?;
    m.add_function(wrap_pyfunction!(decode_halfpail_batch, m)?)?;
    m.add("BOARD_SIZE", BOARD_SIZE)?;
    m.add("BARRELS_PER_PLAYER", BARRELS_PER_PLAYER)?;
    m.add("POLICY_SIZE", mcts::POLICY_SIZE)?;
    m.add_class::<mcts::MCTSEngine>()?;
    m.add_class::<mcts::MCTSSearchResult>()?;
    m.add_class::<mcts::TrainingExample>()?;
    m.add_class::<mcts::SelfPlayResult>()?;
    m.add_class::<mcts::EvalMatchResult>()?;
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
