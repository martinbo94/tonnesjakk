"""
Tonnesjakk Web Server
FastAPI backend for spillet
"""

import json
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

# Importer fra installert pakke (via pip install)
from tonnesjakk import Board, Engine, Player, Position, Move
from tonnesjakk._core import MCTSEngine

app = FastAPI(title="Tonnesjakk")

# Game state (enkel in-memory lagring)
games: dict[str, dict] = {}


def _load_alphazero_eval(model_path: str):
    """Load an AlphaZero model and return an eval function for MCTS."""
    from tonnesjakk.alphazero import make_network
    import torch
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    # Infer hidden channels from first conv layer weights
    hidden = checkpoint["model_state_dict"]["conv_init.weight"].shape[0]
    net = make_network(checkpoint.get("network_type", "resnet"), hidden=hidden)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    def eval_fn(batch_planes):
        tensor = torch.tensor(batch_planes, dtype=torch.float32)
        with torch.no_grad():
            policy, value = net(tensor)
        return policy.tolist(), value.tolist()

    return eval_fn


class MoveRequest(BaseModel):
    game_id: str
    # Pail-only sub-move (place pail, then barrel moves follow)
    is_pail_only: bool = False
    # Pail placement (optional, or the pail position for pail-only moves)
    place_pail: Optional[tuple[int, int]] = None
    # Barrel action (ignored for pail-only moves)
    is_barrel_placement: bool = False  # True = place new from off-board
    barrel_from: Optional[tuple[int, int]] = None  # None if placement
    barrel_to: Optional[tuple[int, int]] = None


class NewGameRequest(BaseModel):
    player_color: str = "white"  # "white" eller "black"
    ai_time_ms: int = 200        # time budget per move (iterative deepening)
    ai_depth: int = 6            # fallback depth (used by the analyze endpoint)
    ai_blunder: float = 0.0      # P(random legal move) — weakens the engine for easy tiers
    # "ml"         = NNUE evaluation + alpha-beta search + solved-endgame tablebases
    # "rule_based" = hand-crafted evaluation + alpha-beta search (no net, no TB)
    engine_type: str = "ml"


# ── ONE shared tablebase engine (NNUE + every solved phase), loaded ONCE.
# The packed big-phase index alone is ~14 GB of RAM (a dense wc×bc pair_offset
# per phase), so it must never be loaded more than once. The old code built a
# fresh tablebase engine PER GAME — and `games` is never fully torn down — which
# multiplied that to tens of GB and hard-crashed the machine. Now the explorer's
# proven-verdict probes AND "ml" play share this single engine. Access is
# serialized by _engine_lock because FastAPI runs sync endpoints in a threadpool,
# so a probe (/api/explore) and a search (/api/ai-move) can otherwise overlap on
# the same &mut engine.
_engine_lock = threading.Lock()
_TB_ENGINE = None
_TB_PHASES = []


def _get_tb_engine():
    global _TB_ENGINE, _TB_PHASES
    if _TB_ENGINE is None:
        _TB_ENGINE = Engine()
        nnue = Path(__file__).resolve().parent.parent / "models" / "net3_plain_m_d20_64x16_b25_l05.json"
        if nnue.exists():
            try:
                _TB_ENGINE.load_nnue(str(nnue))
                print(f"# TB engine NNUE: {nnue.name}")
            except Exception as e:
                print(f"# NNUE load failed ({e}); handcrafted eval")
        tb = Path(__file__).resolve().parent.parent / "tablebases"
        if tb.is_dir():
            try:
                _TB_PHASES = _TB_ENGINE.load_tablebases(str(tb))
                print(f"# Tablebases loaded once (~14 GB): {_TB_PHASES}")
            except Exception as e:
                print(f"# Tablebase load failed ({e}); continuing without")
    return _TB_ENGINE


# The explorer probes verdicts on the same shared engine (no second load).
def _get_explorer():
    return _get_tb_engine()


_RULE_ENGINE = None
_ANALYSIS_ENGINE = None


def _get_rule_engine():
    """Hand-crafted evaluation only (no net, no tablebases) — cheap."""
    global _RULE_ENGINE
    if _RULE_ENGINE is None:
        _RULE_ENGINE = Engine()
    return _RULE_ENGINE


def _engine_for(game):
    return _get_tb_engine() if game.get("engine_type") == "ml" else _get_rule_engine()


def _get_analysis_engine():
    """Dedicated engine for the analyze endpoint (separate TT from play)."""
    global _ANALYSIS_ENGINE
    if _ANALYSIS_ENGINE is None:
        _ANALYSIS_ENGINE = Engine()
    return _ANALYSIS_ENGINE


def board_to_dict(board: Board) -> dict:
    """Konverter brett til JSON-vennlig dict."""
    arr = board.to_array()

    # Finn brikkeposisjoner
    white_barrels = [(p.row, p.col) for p in board.find_barrels(Player.White)]
    black_barrels = [(p.row, p.col) for p in board.find_barrels(Player.Black)]
    white_pail = board.find_pail(Player.White)
    black_pail = board.find_pail(Player.Black)

    # PyO3 enum comparison needs string repr
    current_player_str = "white" if "White" in repr(board.current_player) else "black"

    return {
        "board": arr,
        "current_player": current_player_str,
        "move_count": board.move_count,
        "white_barrels": white_barrels,
        "black_barrels": black_barrels,
        "white_pail": (white_pail.row, white_pail.col) if white_pail else None,
        "black_pail": (black_pail.row, black_pail.col) if black_pail else None,
        "white_pail_placed": board.white_pail_placed,
        "black_pail_placed": board.black_pail_placed,
        "white_barrels_off_board": board.white_barrels_off_board,
        "black_barrels_off_board": board.black_barrels_off_board,
        "white_scored": board.white_scored,
        "black_scored": board.black_scored,
        "winner": None if board.check_winner() is None else (
            "white" if "White" in repr(board.check_winner()) else "black"
        ),
    }


def get_valid_moves(board: Board) -> list[dict]:
    """Hent alle gyldige trekk som JSON."""
    moves = board.generate_moves()
    result = []
    for m in moves:
        d = {
            "is_pail_only": m.is_pail_only,
            "place_pail": (m.place_pail.row, m.place_pail.col) if m.place_pail else None,
            "is_barrel_placement": m.is_barrel_placement,
            "barrel_from": (m.barrel_from.row, m.barrel_from.col) if m.barrel_from else None,
            "barrel_to": (m.barrel_to.row, m.barrel_to.col),
            "barrel_path": [(p.row, p.col) for p in m.barrel_path],
        }
        result.append(d)
    return result


@app.post("/api/new-game")
def new_game(req: NewGameRequest):
    """Start et nytt spill."""
    import uuid
    game_id = str(uuid.uuid4())[:8]

    board = Board()
    # No per-game engine: "ml"/"rule_based" games share module-level engines
    # (loaded once) chosen at move time via _engine_for(). This keeps tablebase
    # and NNUE memory constant regardless of how many games are created.
    game_data = {
        "board": board,
        "player_color": req.player_color,
        "ai_depth": req.ai_depth,
        "ai_time_ms": max(10, int(req.ai_time_ms)),
        "engine_type": req.engine_type,
        "blunder_rate": max(0.0, min(1.0, float(req.ai_blunder))),
        "board_history": [board.copy()],
    }

    # Bound memory: evict the oldest games so the dict can't grow without limit
    # (each game holds board history + a move tree). Dict preserves insertion order.
    MAX_GAMES = 24
    while len(games) >= MAX_GAMES:
        oldest = next(iter(games))
        games.pop(oldest, None)

    games[game_id] = game_data
    _init_tree(game_data)

    print(f"\n{'#'*50}")
    print(f"# NYTT SPILL: {game_id}")
    print(f"# Spiller: {req.player_color}, Engine: {req.engine_type}, Tid/trekk: {game_data['ai_time_ms']} ms")
    print(f"{'#'*50}\n")

    result = {
        "game_id": game_id,
        "state": board_to_dict(board),
        "valid_moves": get_valid_moves(board),
        "player_color": req.player_color,
    }

    # Hvis spilleren er svart, la AI gjore forste trekk
    if req.player_color == "black":
        ai_move_result = _do_ai_move(game_data)
        if ai_move_result:
            result["state"] = board_to_dict(board)
            result["valid_moves"] = get_valid_moves(board)
            result["ai_move"] = ai_move_result["ai_move"]

    return result


def move_to_dict(m) -> dict:
    """Konverter et Move-objekt til dict."""
    return {
        "place_pail": (m.place_pail.row, m.place_pail.col) if m.place_pail else None,
        "is_barrel_placement": m.is_barrel_placement,
        "barrel_from": (m.barrel_from.row, m.barrel_from.col) if m.barrel_from else None,
        "barrel_to": (m.barrel_to.row, m.barrel_to.col),
        "barrel_path": [(p.row, p.col) for p in m.barrel_path],
    }


@app.get("/api/game/{game_id}")
def get_game(game_id: str):
    """Hent spilltilstand."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")

    game = games[game_id]
    return {
        "game_id": game_id,
        "state": board_to_dict(game["board"]),
        "valid_moves": get_valid_moves(game["board"]),
        "player_color": game["player_color"],
    }


@app.get("/api/explore/{game_id}")
def explore(game_id: str):
    """Lichess-database-style panel data: the proven tablebase verdict of the
    current position and of every legal move (verdicts are absolute colors;
    dist is only meaningful once <= 5 barrels remain — the distance phases)."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")
    board = games[game_id]["board"]
    ex = _get_explorer()

    def probe_dict(b):
        w = b.check_winner()
        if w is not None:
            return {"verdict": "white" if "White" in repr(w) else "black", "dist": 0, "terminal": True}
        v = ex.tablebase_probe(b)
        if v is None:
            return {"verdict": "unknown", "dist": None, "terminal": False}
        remaining = (4 - b.white_scored) + (4 - b.black_scored)
        return {"verdict": v[0], "dist": (v[1] if remaining <= 5 else None), "terminal": False}

    mover = "white" if "White" in repr(board.current_player) else "black"

    def progress(child, cd):
        """Lower = closer to the mover's win, so winning moves sort shortest-
        path first. The big phases are WDL-only (no distance), so there the
        mover's single-agent race distance to the child is the progress proxy;
        exact distance-to-win is used wherever it is stored (<= 5 barrels)."""
        if cd["verdict"] != mover:
            return None
        if cd["terminal"]:
            return 0.0
        if cd["dist"] is not None:
            return float(cd["dist"])            # exact plies-to-win
        # Big WDL phase: no stored distance. Estimate plies from the single-agent
        # race lower bound (2*race - parity), on the SAME scale as exact "om N"
        # so a genuinely shorter proxy line is not buried under a longer exact
        # one. It is a lower-bound estimate, hence the "~" in the UI.
        rd = ex.race_distances(child)
        rw = rd[0] if mover == "white" else rd[1]
        wtm = ("white" if "White" in repr(child.current_player) else "black") == mover
        return float(2 * rw - (1 if wtm else 0))

    # Serialize probes: the shared tb engine is also used by /api/ai-move.
    with _engine_lock:
        rows = []
        for m in board.generate_moves():
            child = board.copy()
            child.make_move(m)
            d = {
                "is_pail_only": m.is_pail_only,
                "place_pail": (m.place_pail.row, m.place_pail.col) if m.place_pail else None,
                "is_barrel_placement": m.is_barrel_placement,
                "barrel_from": (m.barrel_from.row, m.barrel_from.col) if m.barrel_from else None,
                "barrel_to": (m.barrel_to.row, m.barrel_to.col),
                "barrel_path": [(p.row, p.col) for p in m.barrel_path],
            }
            d.update(probe_dict(child))
            d["progress"] = progress(child, d)
            rows.append(d)
        root = probe_dict(board)
    return {
        "root": root,
        "mover": "white" if "White" in repr(board.current_player) else "black",
        "moves": rows,
        "solved": len(_TB_PHASES) >= 12,
    }


@app.post("/api/move")
def make_move(req: MoveRequest):
    """Gjor et trekk og fa AI-svar."""
    if req.game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")

    game = games[req.game_id]
    board = game["board"]

    # Finn matchende trekk
    moves = board.generate_moves()
    matching_move = None

    if req.is_pail_only:
        # Pail-only sub-move: match by pail position
        for m in moves:
            if not m.is_pail_only:
                continue
            if m.place_pail.row == req.place_pail[0] and m.place_pail.col == req.place_pail[1]:
                matching_move = m
                break
    else:
        for m in moves:
            if m.is_pail_only:
                continue

            # Sjekk place_pail
            if req.place_pail is None:
                if m.place_pail is not None:
                    continue
            else:
                if m.place_pail is None:
                    continue
                if m.place_pail.row != req.place_pail[0] or m.place_pail.col != req.place_pail[1]:
                    continue

            # Sjekk is_barrel_placement
            if m.is_barrel_placement != req.is_barrel_placement:
                continue

            # Sjekk barrel_from (for flytting)
            if req.is_barrel_placement:
                if req.barrel_from is not None:
                    continue
            else:
                if req.barrel_from is None or m.barrel_from is None:
                    continue
                if m.barrel_from.row != req.barrel_from[0] or m.barrel_from.col != req.barrel_from[1]:
                    continue

            # Sjekk barrel_to
            if m.barrel_to.row != req.barrel_to[0] or m.barrel_to.col != req.barrel_to[1]:
                continue

            matching_move = m
            break

    if not matching_move:
        raise HTTPException(status_code=400, detail="Ugyldig trekk")

    # Utfor spillerens trekk
    move = matching_move
    if move.is_pail_only:
        move_str = f"pail_only({move.place_pail.row},{move.place_pail.col})"
    elif move.is_barrel_placement:
        move_str = f"place ({move.barrel_to.row},{move.barrel_to.col})"
    else:
        move_str = f"({move.barrel_from.row},{move.barrel_from.col})->({move.barrel_to.row},{move.barrel_to.col})"
    if move.place_pail and not move.is_pail_only:
        move_str = f"pail@({move.place_pail.row},{move.place_pail.col}) " + move_str

    print(f"\n>>> Spiller: {move_str}")

    board.make_move(matching_move)
    game["board_history"].append(board.copy())
    _sync_tree(game)

    # Returner state etter spillerens trekk (IKKE AI-trekk enna)
    draw = _draw_status(game)
    return {
        "state": board_to_dict(board),
        "valid_moves": get_valid_moves(board),
        "game_over": board.check_winner() is not None or draw is not None,
        "draw": draw,
    }


def _recent_hashes(game: dict) -> list[int]:
    """Hashes of positions since the last irreversible event (incl. current).

    Passed to the engine so search scores repetitions of actual game
    positions as draws. Only these can repeat: the Zobrist off-board keys
    guarantee no hash collisions across irreversible events.
    """
    history = game.get("board_history", [])
    recent: list[int] = []
    for b in reversed(history):
        recent.append(b.get_hash())
        if b.halfmove_clock == 0:
            break
    recent.reverse()
    return recent


def _draw_status(game: dict) -> Optional[str]:
    """Return 'threefold' / 'no_progress' if the game is drawn, else None."""
    board = game["board"]
    if board.check_winner() is not None:
        return None
    if board.halfmove_clock >= 60:
        return "no_progress"
    current = board.get_hash()
    count = sum(1 for b in game.get("board_history", []) if b.get_hash() == current)
    if count >= 3:
        return "threefold"
    return None


def _do_ai_move(game: dict) -> Optional[dict]:
    """Execute AI move for the given game. Returns move info dict or None."""
    import time
    import random
    board = game["board"]
    engine_type = game.get("engine_type", "heuristic")

    if board.check_winner() is not None or _draw_status(game) is not None:
        return None

    start_time = time.time()

    if engine_type == "alphazero" and "eval_fn" in game:
        # AlphaZero models were trained with pail placed on the first turn:
        # keep the center-biased random pail placement for them.
        moves = [m for m in board.generate_moves() if m.is_pail_only]
        if moves:
            weights = []
            for m in moves:
                pos = m.place_pail
                dist = abs(pos.row - 2.5) + abs(pos.col - 2.5)
                w = max(6.0 - dist, 0.5)
                weights.append(w * w)
            total = sum(weights)
            weights = [w / total for w in weights]
            chosen = random.choices(moves, weights=weights, k=1)[0]
            pail_str = f"pail_only({chosen.place_pail.row},{chosen.place_pail.col})"
            print(f"\n>>> AI: {pail_str}")
            board.make_move(chosen)
            if "board_history" in game:
                game["board_history"].append(board.copy())
        if board.check_winner() is not None:
            return None
        # AlphaZero: use MCTS with neural network — fresh engine each move to avoid stale tree
        mcts_engine = MCTSEngine(game.get("mcts_simulations", 300), 1.4)
        eval_fn = game["eval_fn"]
        print(f"  [alphazero] searching... board.current_player={board.current_player}, awaiting_barrel={board.awaiting_barrel}")
        try:
            mcts_result = mcts_engine.search_network_batched(board, eval_fn, 8)
        except Exception as e:
            print(f"  [alphazero] ERROR in search: {e}")
            import traceback; traceback.print_exc()
            return None
        elapsed = time.time() - start_time
        print(f"  [alphazero] search done in {int(elapsed*1000)}ms, best_move={mcts_result.best_move}")

        if mcts_result.best_move:
            move = mcts_result.best_move
            move_str = _format_move(move)
            score_cp = int(mcts_result.root_value * 1000)

            print(f"\n{'='*50}")
            print(f"info alphazero sims={game.get('mcts_simulations', 400)} value={mcts_result.root_value:.3f} time {int(elapsed*1000)}ms")
            print(f"bestmove {move_str}")
            print(f"{'='*50}\n")

            board.make_move(move)
            if "board_history" in game:
                game["board_history"].append(board.copy())
            return {
                "ai_move": move_to_dict(move),
                "ai_score": score_cp,
            }
    else:
        # Heuristic: alpha-beta decides the whole turn, including whether to
        # spend the pail (an optional sub-move). If search picks a pail
        # placement, the AI is still to move — search again for the barrel.
        engine = _engine_for(game)
        pail_pos = None
        barrel_move = None
        last_result = None
        total_nodes = 0
        time_ms = game.get("ai_time_ms", 200)
        blunder = game.get("blunder_rate", 0.0)

        for _ in range(2):  # at most: pail sub-move + barrel move
            # Serialize engine use: the shared tb engine is also used by /api/explore.
            with _engine_lock:
                engine.set_game_history(_recent_hashes(game))
                # Time-based iterative deepening (the difficulty knob). Both engines
                # share this loop; only their evaluators (NNUE+TB vs hand-crafted) differ.
                ai_result = engine.search_timed(board, time_ms)
            if ai_result.best_move is None:
                break
            move = ai_result.best_move
            # Weakening (easy tiers): sometimes discard the best move for a random
            # legal one. The reported score still reflects the engine's real view.
            if blunder > 0.0 and random.random() < blunder:
                legal = list(board.generate_moves())
                if legal:
                    move = random.choice(legal)
            move_str = _format_move(move) if not move.is_pail_only else \
                f"pail_only({move.place_pail.row},{move.place_pail.col})"
            last_result = ai_result
            total_nodes += ai_result.nodes_searched

            elapsed = time.time() - start_time
            score_str = f"+{ai_result.score}" if ai_result.score >= 0 else str(ai_result.score)
            nps = int(total_nodes / elapsed) if elapsed > 0 else 0
            print(f"\n{'='*50}")
            print(f"info depth {ai_result.depth} score cp {ai_result.score} nodes {ai_result.nodes_searched} nps {nps} time {int(elapsed*1000)}ms")
            print(f"info string eval: {score_str} ({'hvit' if ai_result.score > 0 else 'svart' if ai_result.score < 0 else 'likt'} leder)")
            print(f"bestmove {move_str}")
            print(f"{'='*50}\n")

            board.make_move(move)
            if "board_history" in game:
                game["board_history"].append(board.copy())

            if move.is_pail_only:
                pail_pos = move.place_pail
                continue  # turn not complete — search the barrel move
            barrel_move = move
            break

        if last_result is not None:
            move_dict = move_to_dict(barrel_move) if barrel_move is not None else {
                "place_pail": None, "is_barrel_placement": False,
                "barrel_from": None, "barrel_to": None,
            }
            if pail_pos is not None:
                move_dict["place_pail"] = (pail_pos.row, pail_pos.col)
            return {
                "ai_move": move_dict,
                "ai_score": last_result.score,
                "ai_nodes": total_nodes,
            }

    return None


def _sqrc(row: int, col: int) -> str:
    return f"{chr(97 + col)}{6 - row}"


def _derive_move(before, after) -> dict:
    """Notation for the make_move that turned `before` into `after`, by diffing
    the two board snapshots — so the move log rides on board_history and needs
    no bookkeeping at the (several) make_move call sites."""
    is_white = "White" in repr(before.current_player)
    pl = Player.White if is_white else Player.Black
    bb = {(p.row, p.col) for p in before.find_barrels(pl)}
    ab = {(p.row, p.col) for p in after.find_barrels(pl)}
    gone, new = bb - ab, ab - bb
    before_pail, after_pail = before.find_pail(pl), after.find_pail(pl)
    scored = (after.white_scored - before.white_scored) if is_white else (after.black_scored - before.black_scored)
    parts = []
    if before_pail is None and after_pail is not None:
        parts.append(f"🥛{_sqrc(after_pail.row, after_pail.col)}")
    if len(gone) == 1 and len(new) == 1:
        (fr, ft), (tr, tc) = next(iter(gone)), next(iter(new))
        parts.append(f"{_sqrc(fr, ft)}→{_sqrc(tr, tc)}")
    elif len(new) == 1 and not gone:
        tr, tc = next(iter(new)); parts.append(f"+{_sqrc(tr, tc)}")
    elif len(gone) == 1 and not new and scored > 0:
        fr, fc = next(iter(gone)); parts.append(f"{_sqrc(fr, fc)}✓")
    return {"side": "white" if is_white else "black", "notation": " ".join(parts) or "…"}


@app.get("/api/history/{game_id}")
def get_history(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")
    hist = games[game_id].get("board_history", [])
    moves = [dict(ply=i + 1, **_derive_move(hist[i], hist[i + 1])) for i in range(len(hist) - 1)]
    return {"moves": moves, "current_ply": len(hist) - 1}


class GotoRequest(BaseModel):
    ply: int


@app.post("/api/goto/{game_id}")
def goto(game_id: str, req: GotoRequest):
    """Jump the live game back to the position after `ply` moves (ply 0 = start),
    truncating history so you can explore a different line from there."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")
    game = games[game_id]
    hist = game.get("board_history", [])
    if req.ply < 0 or req.ply >= len(hist):
        raise HTTPException(status_code=400, detail="ply out of range")
    game["board"] = hist[req.ply].copy()
    game["board_history"] = hist[: req.ply + 1]
    return {"state": board_to_dict(game["board"]), "valid_moves": get_valid_moves(game["board"])}


def _format_move(move) -> str:
    """Format a Move object as a human-readable string."""
    if move.is_barrel_placement:
        move_str = f"place ({move.barrel_to.row},{move.barrel_to.col})"
    else:
        move_str = f"({move.barrel_from.row},{move.barrel_from.col})->({move.barrel_to.row},{move.barrel_to.col})"
    if move.place_pail:
        move_str = f"pail@({move.place_pail.row},{move.place_pail.col}) " + move_str
    return move_str


# ---------------------------------------------------------------------------
# Move tree (PGN-style: mainline + variations, navigate without truncating)
# The server owns the tree because the Board snapshots live here; the client
# renders it and navigates by node id. Nodes are matched by resulting-position
# hash, so replaying an existing move re-enters its node instead of branching.
# ---------------------------------------------------------------------------

def _init_tree(game: dict) -> None:
    b0 = game["board_history"][0]
    game["tree"] = {
        "nodes": {0: {"parent": None, "depth": 0, "notation": "", "side": None, "children": []}},
        "boards": {0: b0.copy()},
        "current": 0,
        "next_id": 1,
    }


def _sync_tree(game: dict) -> None:
    """Fold any snapshots appended to board_history since the current node into
    the tree (one node per make_move), advancing `current`. New moves branch;
    replayed moves re-enter the existing child."""
    tree = game.get("tree")
    if tree is None:
        return
    hist = game["board_history"]
    node = tree["current"]
    depth = tree["nodes"][node]["depth"]
    for i in range(depth + 1, len(hist)):
        h = hist[i].get_hash()
        parent = node
        child = next((c for c in tree["nodes"][parent]["children"]
                      if tree["boards"][c].get_hash() == h), None)
        if child is None:
            nid = tree["next_id"]; tree["next_id"] += 1
            nm = _derive_move(hist[i - 1], hist[i])
            tree["nodes"][nid] = {"parent": parent, "depth": i,
                                  "notation": nm["notation"], "side": nm["side"], "children": []}
            tree["boards"][nid] = hist[i].copy()
            tree["nodes"][parent]["children"].append(nid)
            child = nid
        node = child
    tree["current"] = node


@app.get("/api/tree/{game_id}")
def get_tree(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")
    tree = games[game_id].get("tree")
    if tree is None:
        return {"nodes": {}, "current": 0, "root": 0}
    nodes = {str(nid): {"parent": nd["parent"], "notation": nd["notation"],
                        "side": nd["side"], "children": nd["children"]}
             for nid, nd in tree["nodes"].items()}
    return {"nodes": nodes, "current": tree["current"], "root": 0}


class GotoNodeRequest(BaseModel):
    node_id: int


@app.post("/api/goto-node/{game_id}")
def goto_node(game_id: str, req: GotoNodeRequest):
    """Set the live board to a tree node's position (no truncation)."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")
    game = games[game_id]
    tree = game.get("tree")
    if tree is None or req.node_id not in tree["nodes"]:
        raise HTTPException(status_code=400, detail="node not found")
    # rebuild board_history along root -> node so future moves branch correctly
    path = []
    n = req.node_id
    while n is not None:
        path.append(tree["boards"][n])
        n = tree["nodes"][n]["parent"]
    path.reverse()
    game["board_history"] = [b.copy() for b in path]
    game["board"] = tree["boards"][req.node_id].copy()
    tree["current"] = req.node_id
    return {"state": board_to_dict(game["board"]), "valid_moves": get_valid_moves(game["board"])}


@app.post("/api/ai-move/{game_id}")
def ai_move(game_id: str):
    """La AI gjore sitt trekk."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")

    game = games[game_id]
    board = game["board"]

    # Sjekk om spillet er over
    if board.check_winner() is not None:
        return {
            "state": board_to_dict(board),
            "valid_moves": [],
            "ai_move": None,
            "ai_score": None,
        }

    result = {
        "state": board_to_dict(board),
        "valid_moves": get_valid_moves(board),
        "ai_move": None,
        "ai_score": None,
    }

    ai_result = _do_ai_move(game)
    if ai_result:
        result.update(ai_result)
        result["state"] = board_to_dict(board)
        result["valid_moves"] = get_valid_moves(board)
    result["draw"] = _draw_status(game)
    _sync_tree(game)

    return result


# ---------------------------------------------------------------------------
# Post-game engine analysis
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    position_index: int


def _get_analysis_state(game: dict) -> dict:
    """Get or create analysis state for a game."""
    if "analysis_state" not in game:
        game["analysis_state"] = {
            "analysis_id": 0,
            "position_index": -1,
            "current_result": None,
            "is_running": False,
            "lock": threading.Lock(),
        }
    return game["analysis_state"]


def _run_analysis(game_id: str, position_index: int, analysis_id: int):
    """Background worker: iterative deepening analysis for a position."""
    game = games.get(game_id)
    if not game:
        print(f"[analysis] game {game_id} not found")
        return

    astate = game["analysis_state"]
    board_history = game.get("board_history", [])
    if position_index < 0 or position_index >= len(board_history):
        print(f"[analysis] position {position_index} out of range (history has {len(board_history)})")
        with astate["lock"]:
            astate["is_running"] = False
        return

    board = board_history[position_index].copy()

    # Shared dedicated analysis engine (separate TT from the play engines, and
    # not stored per-game so it can't leak). Cleared before each analysis.
    engine = _get_analysis_engine()
    engine.clear_tt()

    print(f"[analysis] start pos={position_index} id={analysis_id}")

    try:
        for depth in range(1, 100):
            # Check cancellation before search
            with astate["lock"]:
                if astate["analysis_id"] != analysis_id:
                    print(f"[analysis] cancelled before depth {depth}")
                    return

            result = engine.search(board, depth)

            # Check cancellation after search and store result
            with astate["lock"]:
                if astate["analysis_id"] != analysis_id:
                    print(f"[analysis] cancelled after depth {depth}")
                    return

                best_move = None
                if result.best_move:
                    best_move = move_to_dict(result.best_move)

                astate["current_result"] = {
                    "best_move": best_move,
                    "score": result.score,
                    "depth": result.depth,
                    "nodes": result.nodes_searched,
                }

            print(f"[analysis] depth {depth} score {result.score} nodes {result.nodes_searched}")

            # Yield GIL so HTTP poll handlers can process between depths
            time.sleep(0.01)

            # Stop early if winning sequence found
            if abs(result.score) > 90000:
                print(f"[analysis] early stop: score {result.score}")
                break
    except Exception as e:
        print(f"[analysis] ERROR: {e}")
    finally:
        # Always mark as not running when thread exits
        with astate["lock"]:
            if astate["analysis_id"] == analysis_id:
                astate["is_running"] = False
        print(f"[analysis] done pos={position_index} id={analysis_id}")


def _analysis_response(astate: dict) -> dict:
    """Build the JSON response from analysis state."""
    with astate["lock"]:
        return {
            "analysis_id": astate["analysis_id"],
            "position_index": astate["position_index"],
            "result": astate["current_result"],
            "is_running": astate["is_running"],
        }


@app.post("/api/analyze/{game_id}")
def start_analysis(game_id: str, req: AnalyzeRequest):
    """Start or restart engine analysis for a position."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")

    game = games[game_id]
    astate = _get_analysis_state(game)

    start_new = False
    new_id = 0
    with astate["lock"]:
        # If already analyzing the same position, just return current results
        if astate["position_index"] == req.position_index and astate["is_running"]:
            pass
        else:
            # Cancel previous analysis and start new one
            astate["analysis_id"] += 1
            astate["position_index"] = req.position_index
            astate["current_result"] = None
            astate["is_running"] = True
            new_id = astate["analysis_id"]
            start_new = True

    if start_new:
        thread = threading.Thread(
            target=_run_analysis,
            args=(game_id, req.position_index, new_id),
            daemon=True,
        )
        thread.start()

    return _analysis_response(astate)


@app.get("/api/analyze/{game_id}")
def poll_analysis(game_id: str):
    """Poll current analysis results (no side effects)."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")

    game = games[game_id]
    astate = _get_analysis_state(game)
    return _analysis_response(astate)


@app.post("/api/analyze/{game_id}/stop")
def stop_analysis(game_id: str):
    """Cancel running analysis."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")

    game = games[game_id]
    astate = _get_analysis_state(game)
    with astate["lock"]:
        astate["analysis_id"] += 1
        astate["is_running"] = False

    return {"ok": True}


# ---------------------------------------------------------------------------
# Game record replay endpoints
# ---------------------------------------------------------------------------

GAME_RECORDS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "game_records"


@app.get("/api/game-records")
def list_game_records():
    """List available game record files."""
    if not GAME_RECORDS_DIR.exists():
        return {"records": []}

    records = []
    for f in sorted(GAME_RECORDS_DIR.glob("*.json"), reverse=True):
        # Read just the metadata to build the listing
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            meta = data.get("metadata", {})
            records.append({
                "filename": f.name,
                "model_a": meta.get("model_a", "?"),
                "model_b": meta.get("model_b", "?"),
                "date": meta.get("date", "?"),
                "total_games": meta.get("total_games", 0),
                "wins": meta.get("wins", 0),
                "losses": meta.get("losses", 0),
                "draws": meta.get("draws", 0),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return {"records": records}


@app.get("/api/game-records/{filename}")
def get_game_record(filename: str):
    """Load a specific game record file."""
    # Sanitise: only allow simple filenames (no path traversal)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    filepath = GAME_RECORDS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Record not found")

    with open(filepath, "r") as f:
        data = json.load(f)

    return data


# Serve static files
static_path = Path(__file__).parent
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
def index():
    return FileResponse(static_path / "index.html")


if __name__ == "__main__":
    import uvicorn
    print("\n=== Tonnesjakk server starter ===")
    print("    Apne http://localhost:8000 i nettleseren\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
