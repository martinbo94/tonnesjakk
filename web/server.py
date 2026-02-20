"""
Tonnesjakk Web Server
FastAPI backend for spillet
"""

import json

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

# Importer fra installert pakke (via pip install)
from tonnesjakk import Board, Engine, Player, Position, Move

app = FastAPI(title="Tonnesjakk")

# Game state (enkel in-memory lagring)
games: dict[str, dict] = {}


class MoveRequest(BaseModel):
    game_id: str
    # Pail placement (optional)
    place_pail: Optional[tuple[int, int]] = None
    # Barrel action
    is_barrel_placement: bool  # True = place new from off-board
    barrel_from: Optional[tuple[int, int]] = None  # None if placement
    barrel_to: tuple[int, int]


class NewGameRequest(BaseModel):
    player_color: str = "white"  # "white" eller "black"
    ai_depth: int = 6


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
    return [
        {
            "place_pail": (m.place_pail.row, m.place_pail.col) if m.place_pail else None,
            "is_barrel_placement": m.is_barrel_placement,
            "barrel_from": (m.barrel_from.row, m.barrel_from.col) if m.barrel_from else None,
            "barrel_to": (m.barrel_to.row, m.barrel_to.col),
            "barrel_path": [(p.row, p.col) for p in m.barrel_path],
        }
        for m in moves
    ]


@app.post("/api/new-game")
def new_game(req: NewGameRequest):
    """Start et nytt spill."""
    import uuid
    game_id = str(uuid.uuid4())[:8]

    board = Board()
    engine = Engine()

    games[game_id] = {
        "board": board,
        "engine": engine,
        "player_color": req.player_color,
        "ai_depth": req.ai_depth,
    }

    print(f"\n{'#'*50}")
    print(f"# NYTT SPILL: {game_id}")
    print(f"# Spiller: {req.player_color}, AI dybde: {req.ai_depth}")
    print(f"{'#'*50}\n")

    result = {
        "game_id": game_id,
        "state": board_to_dict(board),
        "valid_moves": get_valid_moves(board),
        "player_color": req.player_color,
    }

    # Hvis spilleren er svart, la AI gjore forste trekk
    if req.player_color == "black":
        ai_result = engine.search(board, req.ai_depth)
        if ai_result.best_move:
            board.make_move(ai_result.best_move)
            result["state"] = board_to_dict(board)
            result["valid_moves"] = get_valid_moves(board)
            result["ai_move"] = move_to_dict(ai_result.best_move)

    return result


def move_to_dict(m) -> dict:
    """Konverter et Move-objekt til dict."""
    return {
        "place_pail": (m.place_pail.row, m.place_pail.col) if m.place_pail else None,
        "is_barrel_placement": m.is_barrel_placement,
        "barrel_from": (m.barrel_from.row, m.barrel_from.col) if m.barrel_from else None,
        "barrel_to": (m.barrel_to.row, m.barrel_to.col),
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


@app.post("/api/move")
def make_move(req: MoveRequest):
    """Gjor et trekk og fa AI-svar."""
    if req.game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")

    game = games[req.game_id]
    board = game["board"]
    engine = game["engine"]

    # Finn matchende trekk
    moves = board.generate_moves()
    matching_move = None

    for m in moves:
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
            # Plassering - barrel_from skal vaere None
            if req.barrel_from is not None:
                continue
        else:
            # Flytting - sjekk barrel_from
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
    if move.is_barrel_placement:
        move_str = f"place ({move.barrel_to.row},{move.barrel_to.col})"
    else:
        move_str = f"({move.barrel_from.row},{move.barrel_from.col})->({move.barrel_to.row},{move.barrel_to.col})"
    if move.place_pail:
        move_str = f"pail@({move.place_pail.row},{move.place_pail.col}) " + move_str

    print(f"\n>>> Spiller: {move_str}")

    board.make_move(matching_move)

    # Returner state etter spillerens trekk (IKKE AI-trekk enna)
    return {
        "state": board_to_dict(board),
        "valid_moves": get_valid_moves(board),
        "game_over": board.check_winner() is not None,
    }


@app.post("/api/ai-move/{game_id}")
def ai_move(game_id: str):
    """La AI gjore sitt trekk."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Spill ikke funnet")

    game = games[game_id]
    board = game["board"]
    engine = game["engine"]

    # Sjekk om spillet er over
    if board.check_winner() is not None:
        return {
            "state": board_to_dict(board),
            "valid_moves": [],
            "ai_move": None,
            "ai_score": None,
        }

    # AI-trekk
    import time
    start_time = time.time()
    ai_result = engine.search(board, game["ai_depth"])
    elapsed = time.time() - start_time

    result = {
        "state": board_to_dict(board),
        "valid_moves": get_valid_moves(board),
        "ai_move": None,
        "ai_score": None,
    }

    if ai_result.best_move:
        result["ai_move"] = move_to_dict(ai_result.best_move)
        result["ai_score"] = ai_result.score
        result["ai_nodes"] = ai_result.nodes_searched

        # Stockfish-lignende output
        score_str = f"+{ai_result.score}" if ai_result.score >= 0 else str(ai_result.score)
        nps = int(ai_result.nodes_searched / elapsed) if elapsed > 0 else 0

        move = ai_result.best_move
        if move.is_barrel_placement:
            move_str = f"place ({move.barrel_to.row},{move.barrel_to.col})"
        else:
            move_str = f"({move.barrel_from.row},{move.barrel_from.col})->({move.barrel_to.row},{move.barrel_to.col})"
        if move.place_pail:
            move_str = f"pail@({move.place_pail.row},{move.place_pail.col}) " + move_str

        print(f"\n{'='*50}")
        print(f"info depth {ai_result.depth} score cp {ai_result.score} nodes {ai_result.nodes_searched} nps {nps} time {int(elapsed*1000)}ms")
        print(f"info string eval: {score_str} ({'hvit' if ai_result.score > 0 else 'svart' if ai_result.score < 0 else 'likt'} leder)")
        print(f"info string tt_hits: {ai_result.tt_hits} cutoffs: {ai_result.cutoffs}")
        print(f"bestmove {move_str}")
        print(f"{'='*50}\n")

        board.make_move(ai_result.best_move)
        result["state"] = board_to_dict(board)
        result["valid_moves"] = get_valid_moves(board)

    return result


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
