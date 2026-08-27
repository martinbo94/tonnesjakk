use pyo3::prelude::*;

pub mod board;
pub mod nnue;
pub mod race;
pub mod search;
pub mod tablebase;
pub mod mcts;

// Re-export everything for backward compatibility
pub use board::*;
pub use nnue::*;
pub use search::{BitBoardEngine, Engine, SearchResult, TT_SIZE};

/// Generic batch decoder: 164-feature rows -> sparse indices for ANY
/// architecture the Rust evaluator supports. Feature indices come from the
/// same `NnueConfig::active_features` the engine uses, so training and
/// inference can never disagree on the encoding.
///
/// Returns (white_indices, white_offsets, black_indices, black_offsets,
///          dense_flat, output_bucket_per_sample, labels).
#[pyfunction]
#[pyo3(signature = (data, labels, feature_set, mirror_black, dense_size, output_buckets))]
fn decode_sparse_batch(
    data: Vec<f32>,
    labels: Vec<f32>,
    feature_set: &str,
    mirror_black: bool,
    dense_size: usize,
    output_buckets: usize,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<f32>, Vec<i64>, Vec<f32>)> {
    let n = labels.len();
    if data.len() != n * 164 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Expected {} floats ({}×164), got {}", n * 164, n, data.len())
        ));
    }
    let fs = FeatureSet::from_name(feature_set).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unknown feature_set '{}'", feature_set))
    })?;
    let config = NnueConfig {
        feature_set: fs, mirror_black, dense_size, hidden1: 0, hidden2: 0, output_buckets,
    };

    let mut w_idx: Vec<i64> = Vec::with_capacity(n * 10);
    let mut b_idx: Vec<i64> = Vec::with_capacity(n * 10);
    let mut w_off: Vec<i64> = Vec::with_capacity(n);
    let mut b_off: Vec<i64> = Vec::with_capacity(n);
    let mut dense_flat: Vec<f32> = Vec::with_capacity(n * dense_size);
    let mut buckets: Vec<i64> = Vec::with_capacity(n);
    let mut feats = [0u16; MAX_ACTIVE_FEATURES];

    for i in 0..n {
        let row = &data[i * 164..(i + 1) * 164];
        let bb = bitboard_from_dense164(row);

        w_off.push(w_idx.len() as i64);
        let nw = config.active_features(&bb, Player::White, &mut feats);
        w_idx.extend(feats[..nw].iter().map(|&f| f as i64));

        b_off.push(b_idx.len() as i64);
        let nb = config.active_features(&bb, Player::Black, &mut feats);
        b_idx.extend(feats[..nb].iter().map(|&f| f as i64));

        if dense_size > 0 {
            dense_flat.extend_from_slice(&row[144..144 + dense_size]);
        }
        buckets.push(config.output_bucket(&bb) as i64);
    }

    Ok((w_idx, w_off, b_idx, b_off, dense_flat, buckets, labels))
}

/// Solve tablebase phase (wr, br) — barrels remaining per side — and write
/// `dir/tb_{wr}v{br}.bin`. Lower phases must already exist in `dir`.
/// Returns (states, white_wins, black_wins, draws).
#[pyfunction]
#[pyo3(signature = (dir, wr, br, verbose=true))]
fn solve_tablebase(py: Python, dir: &str, wr: usize, br: usize, verbose: bool) -> PyResult<(usize, usize, usize, usize)> {
    let path = std::path::Path::new(dir);
    let mut tb = tablebase::Tablebase::load_dir(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("load tablebases: {}", e)))?;
    for (lw, lb) in [(wr.saturating_sub(1), br), (wr, br.saturating_sub(1))] {
        if lw >= 1 && lb >= 1 && tb.get(lw, lb).is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("phase {}v{} must be solved before {}v{}", lw, lb, wr, br)));
        }
    }
    let stats = py.allow_threads(|| {
        let phase = tb.solve(wr, br, verbose);
        let n = phase.num_states();
        let w = phase.vals().iter().filter(|&&v| tablebase::is_white_win(v)).count();
        let b = phase.vals().iter().filter(|&&v| tablebase::is_black_win(v)).count();
        let d = phase.vals().iter().filter(|&&v| v == tablebase::V_DRAW).count();
        phase.save(path).map(|_| (n, w, b, d))
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("save tablebase: {}", e)))?;
    Ok(stats)
}

/// Write the 2-bit WDL companion (`tb_{wr}v{br}.wdl`) of a solved full phase.
#[pyfunction]
fn repack_tablebase_wdl(py: Python, dir: &str, wr: usize, br: usize) -> PyResult<usize> {
    let path = std::path::Path::new(dir);
    let phase = tablebase::Phase::load(path, wr, br)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("load phase: {}", e)))?;
    py.allow_threads(|| phase.save_wdl(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("write wdl: {}", e)))?;
    Ok((phase.num_states() + 3) / 4)
}

/// Solve a SYMMETRIC phase (r v r) into the packed 2-bit WDL format
/// (`dir/tb_{r}v{r}.p2`): white-to-move states only, no distances — 3v3 fits
/// in ~14 GB. Checkpoints every `checkpoint_every` passes (resumable).
/// `lowmem` loads the lower phases from their `.wdl` companions (see
/// `repack_tablebase_wdl`) so the 11.5 GB 3v2 table is not resident too.
/// Returns (states, white_wins, black_wins, draws) over white-to-move states.
#[pyfunction]
#[pyo3(signature = (dir, r, checkpoint_every = 5, verbose = true, lowmem = false))]
fn solve_tablebase_packed(py: Python, dir: &str, r: usize, checkpoint_every: usize, verbose: bool, lowmem: bool) -> PyResult<(usize, usize, usize, usize)> {
    let path = std::path::Path::new(dir);
    let tb = if lowmem { tablebase::Tablebase::load_dir_lowmem(path) } else { tablebase::Tablebase::load_dir(path) }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("load tablebases: {}", e)))?;
    if r >= 2 && tb.get(r, r - 1).is_none() && tb.get(r - 1, r).is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!("phase {}v{} must be solved first", r, r - 1)));
    }
    py.allow_threads(|| {
        let p = tablebase::solve_packed(&tb, r, path, checkpoint_every, verbose);
        let n = p.num_states();
        let (mut w, mut b, mut d) = (0usize, 0usize, 0usize);
        for idx in 0..n {
            match p.get(idx) {
                tablebase::P_WHITE => w += 1,
                tablebase::P_BLACK => b += 1,
                tablebase::P_UNKNOWN => d += 1,
                _ => {}
            }
        }
        p.save(path).map(|_| (n, w, b, d))
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("save tablebase: {}", e)))
}

/// Python-modul
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_tablebase, m)?)?;
    m.add_function(wrap_pyfunction!(solve_tablebase_packed, m)?)?;
    m.add_function(wrap_pyfunction!(repack_tablebase_wdl, m)?)?;
    m.add_class::<Player>()?;
    m.add_class::<Cell>()?;
    m.add_class::<Position>()?;
    m.add_class::<Move>()?;
    m.add_class::<Board>()?;
    m.add_class::<Engine>()?;
    m.add_class::<SearchResult>()?;
    m.add_function(wrap_pyfunction!(decode_sparse_batch, m)?)?;
    m.add("BOARD_SIZE", BOARD_SIZE)?;
    m.add("BARRELS_PER_PLAYER", BARRELS_PER_PLAYER)?;
    m.add("POLICY_SIZE", mcts::POLICY_SIZE)?;
    m.add_class::<mcts::MCTSEngine>()?;
    m.add_class::<mcts::MCTSSearchResult>()?;
    m.add_class::<mcts::TrainingExample>()?;
    m.add_class::<mcts::SelfPlayResult>()?;
    m.add_class::<mcts::EvalMatchResult>()?;
    m.add_class::<mcts::OnnxSession>()?;
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
        if mv.is_pail_only {
            let pail = mv.place_pail.unwrap();
            return format!("pail_only({},{})", pail.row, pail.col);
        }
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

    /// Hjelpefunksjon: sett opp et brett med pails allerede plassert
    fn setup_test_board_with_pails() -> Board {
        let mut board = Board::new();
        board.cells[5][2] = Cell::WhiteBarrel;
        board.cells[5][3] = Cell::WhiteBarrel;
        board.cells[0][2] = Cell::BlackBarrel;
        board.cells[0][3] = Cell::BlackBarrel;
        board.white_barrels_off_board = 2;
        board.black_barrels_off_board = 2;
        board.cells[3][3] = Cell::WhitePail;
        board.white_pail_placed = true;
        board.cells[2][2] = Cell::BlackPail;
        board.black_pail_placed = true;
        board
    }

    /// Test at make_move/unmake_move fungerer korrekt for pail sub-moves
    #[test]
    fn test_make_unmake_pail_submove() {
        let board = setup_test_board(); // no pails placed
        let bb_original = BitBoard::from_board(&board);
        let mut bb = bb_original;

        let moves = bb.generate_moves();
        assert!(!moves.is_empty(), "No moves generated");
        assert!(moves[0].is_pail_placement(), "First moves should be pail placements");

        for mv in moves.iter().take(10) {
            let undo = bb.make_move(mv);

            // Pail sub-move: player stays the same, awaiting_barrel is set
            assert_eq!(bb.current_player, bb_original.current_player);
            assert!(bb.awaiting_barrel);

            // Angre trekket
            bb.unmake_move(&undo);

            // Sjekk at vi er tilbake til original
            assert_eq!(bb.white_barrels, bb_original.white_barrels);
            assert_eq!(bb.black_barrels, bb_original.black_barrels);
            assert_eq!(bb.white_pail, bb_original.white_pail);
            assert_eq!(bb.black_pail, bb_original.black_pail);
            assert_eq!(bb.occupied, bb_original.occupied);
            assert_eq!(bb.current_player, bb_original.current_player);
            assert_eq!(bb.awaiting_barrel, bb_original.awaiting_barrel);
        }

        println!("✓ make_move/unmake_move works for pail sub-moves");
    }

    /// Test at make_move/unmake_move fungerer korrekt for barrel moves
    #[test]
    fn test_make_unmake_move() {
        let board = setup_test_board_with_pails();
        let bb_original = BitBoard::from_board(&board);
        let mut bb = bb_original;

        let moves = bb.generate_moves();
        assert!(!moves.is_empty(), "No moves generated");

        for mv in moves.iter().take(10) {
            let undo = bb.make_move(mv);

            // Barrel move: player should switch
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
            assert_eq!(bb.awaiting_barrel, bb_original.awaiting_barrel);
        }

        println!("✓ make_move/unmake_move works correctly for barrel moves");
    }

    /// Test at prekalkulerte tabeller er korrekte
    #[test]
    fn test_precomputed_tables() {
        // Test ADJACENT (8 retninger)
        // Hjørne (0,0) har 3 naboer (høyre, ned, ned-høyre)
        let adj_00 = ADJACENT[sq(0, 0)];
        assert_eq!(adj_00.count_ones(), 3);
        assert!(adj_00 & bit(sq(0, 1)) != 0); // høyre
        assert!(adj_00 & bit(sq(1, 0)) != 0); // ned
        assert!(adj_00 & bit(sq(1, 1)) != 0); // ned-høyre

        // Senter (2,2) har 8 naboer
        let adj_22 = ADJACENT[sq(2, 2)];
        assert_eq!(adj_22.count_ones(), 8);

        // Test JUMP_LANDING
        // Fra (2,2) kan vi hoppe i alle 8 retninger
        for dir in 0..NUM_JUMP_DIRS {
            assert!(JUMP_LANDING[sq(2, 2)][dir] >= 0);
        }

        // Fra (0,0) kan vi bare hoppe ned, høyre og ned-høyre
        assert!(JUMP_LANDING[sq(0, 0)][0] < 0); // opp - ugyldig
        assert!(JUMP_LANDING[sq(0, 0)][1] >= 0); // ned - gyldig
        assert!(JUMP_LANDING[sq(0, 0)][2] < 0); // venstre - ugyldig
        assert!(JUMP_LANDING[sq(0, 0)][3] >= 0); // høyre - gyldig
        assert!(JUMP_LANDING[sq(0, 0)][7] >= 0); // ned-høyre - gyldig

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

    // ═══════════════════════════════════════════════════════════════
    // Draw-rule tests (halfmove clock, off-board Zobrist keys, repetition)
    // ═══════════════════════════════════════════════════════════════

    /// Play a pail placement + a barrel placement for the side to move.
    fn place_pail_and_barrel(bb: &mut BitBoard, pail_sq: u8) {
        bb.make_move(&BitMove::new_pail_placement(pail_sq));
        let placement = bb
            .generate_moves()
            .into_iter()
            .find(|m| m.is_placement())
            .expect("placement available");
        bb.make_move(&placement);
    }

    #[test]
    fn test_halfmove_clock() {
        let mut bb = BitBoard::new();
        assert_eq!(bb.halfmove_clock, 0);

        place_pail_and_barrel(&mut bb, sq(2, 0) as u8); // white
        assert_eq!(bb.halfmove_clock, 0, "placement resets clock");
        place_pail_and_barrel(&mut bb, sq(3, 5) as u8); // black
        assert_eq!(bb.halfmove_clock, 0);

        // Reversible barrel moves increment the clock
        for expected in 1..=4u16 {
            let mv = bb
                .generate_moves()
                .into_iter()
                .find(|m| !m.is_placement() && !m.is_pail_placement())
                .expect("barrel move available");
            bb.make_move(&mv);
            assert_eq!(bb.halfmove_clock, expected);
        }

        // Placement resets again
        let mv = bb
            .generate_moves()
            .into_iter()
            .find(|m| m.is_placement())
            .expect("placement available");
        bb.make_move(&mv);
        assert_eq!(bb.halfmove_clock, 0);
    }

    /// Off-board Zobrist keys: same board occupancy with different
    /// off-board/scored splits must hash differently.
    #[test]
    fn test_offboard_hash_keys() {
        let a = BitBoard::new();
        let mut b = BitBoard::new();
        // Simulate "one white barrel scored, none in hand difference":
        // manually alter off-board count and re-derive expected hash change.
        b.hash ^= ZOBRIST_TEST_KEYS(0, 4);
        b.hash ^= ZOBRIST_TEST_KEYS(0, 3);
        b.white_barrels_off_board = 3;
        b.white_scored = 1;
        assert_ne!(a.hash, b.hash, "off-board count must affect the hash");
    }

    /// Repetition: with the position already in game history, search must
    /// score the repeating line as a draw (0), not as the static eval.
    #[test]
    fn test_repetition_scored_as_draw() {
        let mut bb = BitBoard::new();
        place_pail_and_barrel(&mut bb, sq(2, 0) as u8);
        place_pail_and_barrel(&mut bb, sq(3, 5) as u8);

        let mut engine = BitBoardEngine::new();

        // Baseline: no history → normal score
        engine.game_history.clear();
        let (score_free, _) = engine.search(&bb, 4);

        // Poison history: every child position of every white move is
        // "already seen twice" → every line starts with a repetition draw.
        let mut all_children = Vec::new();
        for mv in bb.generate_moves() {
            let mut child = bb;
            child.make_move(&mv);
            all_children.push(child.hash);
        }
        engine.full_reset();
        engine.game_history = all_children;
        let (score_rep, _) = engine.search(&bb, 4);

        assert_eq!(score_rep, 0, "all-repetition position must score as draw");
        assert_ne!(score_free, score_rep, "baseline should differ from draw score");
    }

    /// No-progress rule: a position at the clock limit scores as a draw one
    /// ply into the search.
    #[test]
    fn test_no_progress_draw() {
        let mut bb = BitBoard::new();
        place_pail_and_barrel(&mut bb, sq(2, 0) as u8);
        place_pail_and_barrel(&mut bb, sq(3, 5) as u8);
        bb.halfmove_clock = 59;

        let mut engine = BitBoardEngine::new();
        engine.no_progress_limit = 60;
        // Only reversible moves exist here besides placements; a placement
        // resets the clock, so search should either place a barrel (progress)
        // or accept the draw — never return a large advantage from shuffling.
        let (score, mv) = engine.search(&bb, 6);
        assert!(mv.is_some());
        assert!(score.abs() < 90_000);
    }
}

/// Test-only accessor for the off-board Zobrist keys (keeps ZOBRIST private).
#[cfg(test)]
#[allow(non_snake_case)]
fn ZOBRIST_TEST_KEYS(player: usize, count: usize) -> u64 {
    crate::board::zobrist_off_board_key(player, count)
}
