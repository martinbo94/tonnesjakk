"""
Interaktivt CLI-spill: Menneske vs AI i Tønnesjakk
"""

from tonnesjakk import Board, Engine, Player, Position, Move


def print_board_with_coords(board: Board):
    """Vis brett med koordinater og forklaring."""
    print()
    print("    0   1   2   3   4   5   6   7")
    print("  +---+---+---+---+---+---+---+---+")

    arr = board.to_array()
    symbols = {0: " ", 1: "W", -1: "B", 2: "w", -2: "b"}

    for row in range(8):
        line = f"{row} |"
        for col in range(8):
            val = arr[row][col]
            line += f" {symbols[val]} |"
        print(line)
        print("  +---+---+---+---+---+---+---+---+")

    print()
    print("Symboler: W=hvit tønne, B=svart tønne, w=hvit melkespann, b=svart melkespann")
    print(f"Tur: {'Hvit' if board.current_player == Player.White else 'Svart'} | Trekk: {board.move_count}")
    print()


def get_position(prompt: str) -> Position:
    """Les en posisjon fra brukeren."""
    while True:
        try:
            inp = input(prompt).strip()
            if inp.lower() == 'q':
                raise KeyboardInterrupt
            parts = inp.replace(',', ' ').split()
            row, col = int(parts[0]), int(parts[1])
            if 0 <= row < 8 and 0 <= col < 8:
                return Position(row, col)
            print("Posisjon må være mellom 0-7!")
        except (ValueError, IndexError):
            print("Ugyldig format! Bruk: rad kolonne (f.eks. '2 3')")


def find_matching_move(board: Board, pail_to: Position, barrel_from: Position, barrel_to: Position) -> Move | None:
    """Finn et lovlig trekk som matcher brukerens valg."""
    moves = board.generate_moves()

    for mv in moves:
        if (mv.pail_to.row == pail_to.row and mv.pail_to.col == pail_to.col and
            mv.barrel_from.row == barrel_from.row and mv.barrel_from.col == barrel_from.col and
            mv.barrel_to().row == barrel_to.row and mv.barrel_to().col == barrel_to.col):
            return mv

    return None


def human_turn(board: Board) -> Move | None:
    """La mennesket gjøre et trekk."""
    print("=== DITT TREKK ===")
    print("(Skriv 'q' for å avslutte)")
    print()

    # Vis mulige trekk
    moves = board.generate_moves()
    print(f"Du har {len(moves)} mulige trekk.")

    # Finn spillerens brikker
    pail = board.find_pail(board.current_player)
    barrels = board.find_barrels(board.current_player)

    print(f"Ditt melkespann: ({pail.row}, {pail.col})")
    print(f"Dine tønner: {[(b.row, b.col) for b in barrels]}")
    print()

    while True:
        print("Steg 1: Flytt melkespannet (kan være samme posisjon)")
        pail_to = get_position(f"  Flytt melkespann til (nå på {pail.row},{pail.col}): ")

        print("\nSteg 2: Flytt en tønne")
        barrel_from = get_position("  Hvilken tønne vil du flytte? (rad kol): ")
        barrel_to = get_position("  Flytt tønnen til: ")

        move = find_matching_move(board, pail_to, barrel_from, barrel_to)

        if move:
            return move
        else:
            print("\n❌ Ugyldig trekk! Prøv igjen.\n")
            print("Tips: Tønner kan bare flytte 1 felt (eller hoppe over andre tønner)")
            print("      Melkespann kan ikke hoppes over")
            print()


def ai_turn(board: Board, engine: Engine, depth: int = 6) -> Move | None:
    """La AI-en gjøre et trekk."""
    print("=== AI TENKER ===")

    import time
    start = time.perf_counter()
    result = engine.search(board, depth)
    elapsed = time.perf_counter() - start

    print(f"Dybde: {depth}")
    print(f"Evaluering: {result.score:+d}")
    print(f"Noder søkt: {result.nodes_searched:,}")
    print(f"Tid: {elapsed:.2f}s")

    if result.best_move:
        mv = result.best_move
        print(f"AI spiller: melkespann til ({mv.pail_to.row},{mv.pail_to.col}), ")
        print(f"            tønne fra ({mv.barrel_from.row},{mv.barrel_from.col}) til ({mv.barrel_to().row},{mv.barrel_to().col})")

    print()
    return result.best_move


def play_game(human_color: Player = Player.White, ai_depth: int = 6):
    """Spill en kamp mot AI-en."""
    board = Board()
    engine = Engine()

    print("\n" + "="*50)
    print("       TØNNESJAKK - Menneske vs AI")
    print("="*50)
    print()
    print(f"Du spiller som: {'HVIT (W)' if human_color == Player.White else 'SVART (B)'}")
    print(f"AI dybde: {ai_depth}")
    print()
    print("MÅL: Få alle dine 4 tønner over til motstanderens side!")
    print("     Hvit skal til rad 7, Svart skal til rad 0")
    print()
    input("Trykk ENTER for å starte...")

    while board.check_winner() is None:
        print_board_with_coords(board)

        if board.current_player == human_color:
            move = human_turn(board)
        else:
            move = ai_turn(board, engine, ai_depth)

        if move is None:
            print("Ingen trekk mulig!")
            break

        board.make_move(move)

    # Vis sluttresultat
    print_board_with_coords(board)
    winner = board.check_winner()

    print("="*50)
    if winner == human_color:
        print("🎉 GRATULERER! Du vant!")
    elif winner is not None:
        print("🤖 AI vant! Bedre lykke neste gang.")
    else:
        print("Uavgjort!")
    print("="*50)


def main():
    print("\n=== TØNNESJAKK ===\n")
    print("1. Spill som Hvit (starter først)")
    print("2. Spill som Svart")
    print("3. Se AI vs AI")
    print()

    choice = input("Velg (1/2/3): ").strip()

    if choice == "1":
        depth = input("AI-dybde (standard 6): ").strip()
        depth = int(depth) if depth.isdigit() else 6
        play_game(Player.White, depth)
    elif choice == "2":
        depth = input("AI-dybde (standard 6): ").strip()
        depth = int(depth) if depth.isdigit() else 6
        play_game(Player.Black, depth)
    elif choice == "3":
        ai_vs_ai()
    else:
        print("Ugyldig valg!")


def ai_vs_ai(depth: int = 6, max_moves: int = 100):
    """Se to AI-er spille mot hverandre."""
    import time

    board = Board()
    engine = Engine()

    print(f"\n=== AI vs AI (dybde {depth}) ===\n")

    while board.check_winner() is None and board.move_count < max_moves:
        print_board_with_coords(board)

        start = time.perf_counter()
        result = engine.search(board, depth)
        elapsed = time.perf_counter() - start

        player = "Hvit" if board.current_player == Player.White else "Svart"
        print(f"{player}: score={result.score:+d}, tid={elapsed:.2f}s")

        if result.best_move:
            board.make_move(result.best_move)
        else:
            break

        input("Trykk ENTER for neste trekk...")

    print_board_with_coords(board)
    winner = board.check_winner()
    if winner:
        print(f"Vinner: {'Hvit' if winner == Player.White else 'Svart'}")
    else:
        print("Uavgjort / timeout")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSpillet avsluttet.")
