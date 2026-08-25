"""
Tønnesjakk - Strategispill med AI

Hybrid Python/Rust implementasjon for høy ytelse.
"""

from tonnesjakk._core import (
    Board,
    Move,
    Position,
    Player,
    Cell,
    Engine,
    SearchResult,
    BOARD_SIZE,
    BARRELS_PER_PLAYER,
    decode_sparse_batch,
)

__version__ = "0.1.0"
__all__ = [
    "Board",
    "Move",
    "Position",
    "Player",
    "Cell",
    "Engine",
    "SearchResult",
    "BOARD_SIZE",
    "BARRELS_PER_PLAYER",
    "decode_sparse_batch",
]
