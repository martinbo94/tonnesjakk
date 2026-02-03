"""
Høynivå spillogikk for Tønnesjakk.

Dette er Python-laget som wrapper Rust-kjernen og legger til
funksjonalitet for AI-trening og interaktivt spill.
"""

import random
from typing import Optional
from tonnesjakk import Board, Move, Player


class Game:
    """Wrapper rundt Board med ekstra funksjonalitet."""

    def __init__(self):
        self.board = Board()
        self.history: list[Move] = []

    @property
    def current_player(self) -> Player:
        return self.board.current_player

    @property
    def move_count(self) -> int:
        return self.board.move_count

    def get_moves(self) -> list[Move]:
        """Hent alle lovlige trekk."""
        return self.board.generate_moves()

    def make_move(self, move: Move) -> bool:
        """Utfør et trekk og lagre i historikk."""
        success = self.board.make_move(move)
        if success:
            self.history.append(move)
        return success

    def is_game_over(self) -> bool:
        """Sjekk om spillet er ferdig."""
        return self.board.check_winner() is not None

    def winner(self) -> Optional[Player]:
        """Returner vinneren, eller None hvis spillet pågår."""
        return self.board.check_winner()

    def display(self) -> str:
        """Vis brettet som ASCII."""
        return self.board.display()

    def reset(self):
        """Start et nytt spill."""
        self.board = Board()
        self.history = []


class RandomPlayer:
    """En spiller som velger tilfeldige trekk (for testing)."""

    def select_move(self, game: Game) -> Optional[Move]:
        moves = game.get_moves()
        if not moves:
            return None
        return random.choice(moves)


def play_random_game() -> Optional[Player]:
    """Spill et helt spill mellom to tilfeldige spillere."""
    game = Game()
    player = RandomPlayer()

    while not game.is_game_over():
        move = player.select_move(game)
        if move is None:
            break
        game.make_move(move)

        # Sikkerhet: maks 1000 trekk
        if game.move_count > 1000:
            print("Spillet tok for lang tid!")
            break

    return game.winner()


if __name__ == "__main__":
    # Kjør et testspill
    print("Starter testspill mellom to tilfeldige spillere...\n")

    game = Game()
    player = RandomPlayer()

    print(game.display())

    while not game.is_game_over() and game.move_count < 100:
        move = player.select_move(game)
        if move:
            game.make_move(move)

    print(game.display())

    winner = game.winner()
    if winner:
        print(f"Vinner: {winner}")
    else:
        print("Uavgjort eller ikke ferdig")
