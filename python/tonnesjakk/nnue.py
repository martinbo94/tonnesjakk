"""
NNUE Training for Tonnesjakk

A neural network evaluator trained via self-play.
Based on best practices from Stockfish NNUE and chess engine research.

Key features:
- Random opening moves for diversity (prevents all games being identical)
- Outcome-based labeling with temporal discounting
- Support for multiple architectures
- Export to JSON for Rust inference
- Balanced dataset validation

Usage:
    python -m tonnesjakk.nnue                    # Train with default settings
    python -m tonnesjakk.nnue --games 10000     # Train with more games
    python -m tonnesjakk.nnue --test            # Quick test
    python -m tonnesjakk.nnue --benchmark       # Run benchmark
"""

import json
import random
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Constants
BOARD_SIZE = 6
NUM_PIECE_TYPES = 4  # WhiteBarrel, BlackBarrel, WhitePail, BlackPail

# Feature sizes
BASE_FEATURES = BOARD_SIZE * BOARD_SIZE * NUM_PIECE_TYPES  # 144 (piece positions)
# Relational features:
#   - White barrel distances to goal (4 barrels max) = 4
#   - Black barrel distances to goal (4 barrels max) = 4
#   - White barrels scored = 1
#   - Black barrels scored = 1
#   - White pail placed = 1
#   - Black pail placed = 1
#   - Current player (1 = white, -1 = black) = 1
RELATIONAL_FEATURES = 13
INPUT_SIZE = BASE_FEATURES + RELATIONAL_FEATURES  # 157


# =============================================================================
# Neural Network Architectures
# =============================================================================

class TonnesjakkNNUE(nn.Module):
    """
    NNUE-inspired network for Tonnesjakk.

    Architecture options:
        - 144 -> 64 -> 32 -> 1  (default, ~11K params)
        - 144 -> 128 -> 32 -> 1 (wider, ~21K params)
        - 144 -> 64 -> 16 -> 1  (narrower, ~10K params)

    The first layer is most important - it learns piece-square relationships.
    """

    def __init__(self, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Tanh()  # Output between -1 and +1
        )

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def board_to_tensor(
    board_array: List[List[int]],
    white_scored: int = 0,
    black_scored: int = 0,
    current_player: int = 1  # 1 = white, -1 = black
) -> torch.Tensor:
    """
    Convert board array to neural network input tensor with relational features.

    Base features (144): One-hot encoding per square
    - Channel 0: White barrel
    - Channel 1: Black barrel
    - Channel 2: White pail
    - Channel 3: Black pail

    Relational features (13):
    - White barrel distances to goal (4 values, normalized 0-1)
    - Black barrel distances to goal (4 values, normalized 0-1)
    - White/Black barrels scored (2 values, normalized 0-1)
    - White/Black pail placed (2 values, 0 or 1)
    - Current player (1 value, +1 white, -1 black)
    """
    # Base features: piece positions
    base = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_PIECE_TYPES), dtype=np.float32)

    # Track barrel positions for distance calculation
    white_barrel_rows = []
    black_barrel_rows = []
    white_pail_placed = 0.0
    black_pail_placed = 0.0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            val = board_array[row][col]
            if val == 1:    # WhiteBarrel
                base[row, col, 0] = 1.0
                white_barrel_rows.append(row)
            elif val == -1:  # BlackBarrel
                base[row, col, 1] = 1.0
                black_barrel_rows.append(row)
            elif val == 2:   # WhitePail
                base[row, col, 2] = 1.0
                white_pail_placed = 1.0
            elif val == -2:  # BlackPail
                base[row, col, 3] = 1.0
                black_pail_placed = 1.0

    # Relational features
    relational = np.zeros(RELATIONAL_FEATURES, dtype=np.float32)

    # White barrel distances to goal (row 0 is goal, so distance = row)
    # Normalize by max distance (5) and sort so closest first
    white_dists = sorted([r / 5.0 for r in white_barrel_rows])
    for i, d in enumerate(white_dists[:4]):
        relational[i] = 1.0 - d  # Closer to goal = higher value

    # Black barrel distances to goal (row 5 is goal, so distance = 5 - row)
    black_dists = sorted([(5 - r) / 5.0 for r in black_barrel_rows])
    for i, d in enumerate(black_dists[:4]):
        relational[4 + i] = 1.0 - d

    # Scored barrels (normalized by 4, which is max)
    relational[8] = white_scored / 4.0
    relational[9] = black_scored / 4.0

    # Pails placed
    relational[10] = white_pail_placed
    relational[11] = black_pail_placed

    # Current player
    relational[12] = current_player

    # Combine base and relational features
    features = np.concatenate([base.flatten(), relational])
    return torch.from_numpy(features)


def flip_board_horizontal(board_array: List[List[int]]) -> List[List[int]]:
    """Flip board horizontally for data augmentation (exploits symmetry)."""
    return [[board_array[r][5-c] for c in range(6)] for r in range(6)]


def board_to_tensor_simple(board_array: List[List[int]]) -> torch.Tensor:
    """Simple version for backward compatibility (base features only)."""
    x = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_PIECE_TYPES), dtype=np.float32)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            val = board_array[row][col]
            if val == 1:
                x[row, col, 0] = 1.0
            elif val == -1:
                x[row, col, 1] = 1.0
            elif val == 2:
                x[row, col, 2] = 1.0
            elif val == -2:
                x[row, col, 3] = 1.0
    # Pad with zeros for relational features
    features = np.concatenate([x.flatten(), np.zeros(RELATIONAL_FEATURES, dtype=np.float32)])
    return torch.from_numpy(features)


# =============================================================================
# Data Generation
# =============================================================================

@dataclass
class PositionData:
    """Training data for a single position."""
    board: List[List[int]]
    search_score: float  # From engine search (normalized to -1 to +1)
    white_scored: int    # Number of white barrels scored
    black_scored: int    # Number of black barrels scored
    current_player: int  # 1 = white, -1 = black


@dataclass
class GameResult:
    """Result of a single self-play game."""
    positions: List[PositionData]  # Rich position data with search scores
    outcome: float  # +1 white wins, -1 black wins, 0 draw
    num_moves: int


@dataclass
class TrainingStats:
    """Statistics from training data generation."""
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_positions: int = 0

    @property
    def total_games(self) -> int:
        return self.white_wins + self.black_wins + self.draws

    @property
    def balance_ratio(self) -> float:
        """Ratio of white wins to black wins (1.0 = perfectly balanced)."""
        if self.black_wins == 0:
            return float('inf')
        return self.white_wins / self.black_wins

    def __str__(self) -> str:
        total = self.total_games
        if total == 0:
            return "No games played"
        return (f"W:{self.white_wins} ({100*self.white_wins/total:.1f}%) "
                f"B:{self.black_wins} ({100*self.black_wins/total:.1f}%) "
                f"D:{self.draws} ({100*self.draws/total:.1f}%)")


class DataGenerator:
    """
    Generates training data via self-play.

    Key features:
    - Random opening moves for diversity
    - Can use NNUE or heuristic evaluation
    - Filters out near-terminal positions
    """

    def __init__(self, nnue_path: Optional[str] = None):
        """
        Initialize the data generator.

        Args:
            nnue_path: Path to NNUE weights JSON file. If provided, uses NNUE
                      for evaluation during self-play. Otherwise uses heuristics.
        """
        from tonnesjakk import Board, Engine
        self.Board = Board
        self.Engine = Engine
        # Single reusable engine - avoids memory issues with multiple engine instances
        self._engine = Engine()
        self._using_nnue = False

        if nnue_path is not None:
            try:
                self._engine.load_nnue(nnue_path)
                self._using_nnue = True
                print(f"  Loaded NNUE weights from: {nnue_path}")
            except Exception as e:
                print(f"  Warning: Failed to load NNUE ({e}), using heuristics")

    def play_game(
        self,
        depth: int = 6,
        random_opening_moves: int = 4,
        max_moves: int = 100
    ) -> GameResult:
        """
        Play a single self-play game.

        Args:
            depth: Search depth for the engine
            random_opening_moves: Number of random moves at start (2-6 recommended)
            max_moves: Maximum moves before declaring draw

        Returns:
            GameResult with positions and outcome
        """
        board = self.Board()
        engine = self._engine
        engine.full_reset()  # Full reset to prevent memory issues
        positions = []

        # Random opening moves for diversity
        # This is crucial to prevent all games being identical
        actual_random = random.randint(
            max(1, random_opening_moves - 2),
            random_opening_moves + 2
        )

        for _ in range(actual_random):
            moves = board.generate_moves()
            if not moves or board.check_winner():
                break
            board.make_move(random.choice(moves))

        # Engine plays the rest
        move_count = 0
        while board.check_winner() is None and move_count < max_moves:
            result = engine.search(board, depth)
            if result.best_move is None:
                break

            # Save position with search score (skip first few moves - too random)
            # Use search score instead of game outcome for more precise learning
            if move_count >= 2:
                # Determine current player
                is_white = "White" in repr(board.current_player)
                current_player = 1 if is_white else -1

                # Normalize search score to -1 to +1 range
                # Engine scores are in centipawns-like units, typically -10000 to +10000
                raw_score = result.score
                normalized_score = max(-1.0, min(1.0, raw_score / 10000.0))

                # Flip score perspective to always be from White's viewpoint
                if not is_white:
                    normalized_score = -normalized_score

                positions.append(PositionData(
                    board=board.to_array(),
                    search_score=normalized_score,
                    white_scored=board.white_scored,
                    black_scored=board.black_scored,
                    current_player=current_player
                ))

            board.make_move(result.best_move)
            move_count += 1

        # Determine outcome
        winner = board.check_winner()
        if winner is None:
            outcome = 0.0  # Draw / max moves reached
        elif "White" in repr(winner):
            outcome = 1.0
        else:
            outcome = -1.0

        return GameResult(positions, outcome, move_count)

    def generate_dataset(
        self,
        num_games: int = 10000,
        depth: int = 6,
        random_opening_moves: int = 4,
        use_search_scores: bool = True,
        augment: bool = True,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, TrainingStats]:
        """
        Generate training dataset from self-play games.

        Args:
            num_games: Number of games to play
            depth: Search depth (6-8 recommended for quality/speed tradeoff)
            random_opening_moves: Random moves at start (4-6 recommended)
            use_search_scores: Use engine search scores (better) vs game outcomes
            augment: Apply horizontal flip data augmentation (doubles data)
            verbose: Print progress

        Returns:
            (X, y, stats) - input tensors, labels, and statistics
        """
        all_X = []
        all_y = []
        stats = TrainingStats()

        start_time = time.time()

        for game_num in range(num_games):
            result = self.play_game(depth, random_opening_moves)

            # Update statistics
            if result.outcome > 0.5:
                stats.white_wins += 1
            elif result.outcome < -0.5:
                stats.black_wins += 1
            else:
                stats.draws += 1

            # Process each position with search scores and context
            for pos_data in result.positions:
                # Use search score (more precise) or game outcome (smoother)
                label = pos_data.search_score if use_search_scores else result.outcome

                # Original position
                tensor = board_to_tensor(
                    pos_data.board,
                    white_scored=pos_data.white_scored,
                    black_scored=pos_data.black_scored,
                    current_player=pos_data.current_player
                )
                all_X.append(tensor)
                all_y.append(label)

                # Augmented (horizontally flipped) position - exploits board symmetry
                if augment:
                    flipped_board = flip_board_horizontal(pos_data.board)
                    flipped_tensor = board_to_tensor(
                        flipped_board,
                        white_scored=pos_data.white_scored,
                        black_scored=pos_data.black_scored,
                        current_player=pos_data.current_player
                    )
                    all_X.append(flipped_tensor)
                    all_y.append(label)  # Same label (score doesn't change with flip)

            stats.total_positions = len(all_X)

            # Progress reporting - every 50 games for visibility
            if verbose and (game_num + 1) % 50 == 0:
                elapsed = time.time() - start_time
                gps = (game_num + 1) / elapsed
                eta = (num_games - game_num - 1) / gps
                aug_note = " (2x augmented)" if augment else ""
                print(f"  Game {game_num + 1:5d}/{num_games} "
                      f"({gps:.1f}/s, ETA {eta/60:.1f}m) | "
                      f"{stats} | {stats.total_positions:,} positions{aug_note}", flush=True)

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nGeneration complete in {elapsed:.1f}s")
            print(f"  {stats}")
            print(f"  {stats.total_positions:,} total positions" +
                  (" (includes augmentation)" if augment else ""))
            print(f"  Balance ratio: {stats.balance_ratio:.2f} (1.0 = perfect)")
            print(f"  Labels: {'search scores' if use_search_scores else 'game outcomes'}")

        # Convert to tensors
        X = torch.stack(all_X)
        y = torch.tensor(all_y, dtype=torch.float32).unsqueeze(1)

        return X, y, stats

    def save_dataset(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        stats: TrainingStats,
        path: str,
        config: Optional[Dict] = None
    ):
        """Save generated dataset to file for reuse."""
        np.savez_compressed(
            path,
            X=X.numpy(),
            y=y.numpy(),
            white_wins=stats.white_wins,
            black_wins=stats.black_wins,
            draws=stats.draws,
            total_positions=stats.total_positions,
            config=json.dumps(config) if config else ""
        )

    @staticmethod
    def load_dataset(path: str) -> Tuple[torch.Tensor, torch.Tensor, TrainingStats, Optional[Dict]]:
        """Load dataset from file."""
        data = np.load(path, allow_pickle=True)
        X = torch.tensor(data['X'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.float32)

        stats = TrainingStats()
        stats.white_wins = int(data['white_wins'])
        stats.black_wins = int(data['black_wins'])
        stats.draws = int(data['draws'])
        stats.total_positions = int(data['total_positions'])

        config = None
        config_str = str(data['config'])
        if config_str:
            try:
                config = json.loads(config_str)
            except:
                pass

        return X, y, stats, config


# =============================================================================
# Training
# =============================================================================

def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    hidden1: int = 64,
    hidden2: int = 32,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
    verbose: bool = True
) -> Tuple[TonnesjakkNNUE, Dict]:
    """
    Train the NNUE model.

    Args:
        X: Input tensor (N, 144)
        y: Labels (N, 1)
        hidden1: First hidden layer size
        hidden2: Second hidden layer size
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Adam learning rate
        validation_split: Fraction for validation
        verbose: Print progress

    Returns:
        (model, history) - trained model and training history
    """
    # Split data
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int((1 - validation_split) * n)

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if verbose:
        print(f"  Training set: {len(X_train):,} positions")
        print(f"  Validation set: {len(X_val):,} positions")

    # Create model
    model = TonnesjakkNNUE(hidden1, hidden2)
    if verbose:
        print(f"  Model: {INPUT_SIZE} -> {hidden1} -> {hidden2} -> 1 "
              f"({model.num_parameters:,} parameters)")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        # Mini-batch training
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            marker = " *"
        else:
            marker = ""

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f}{marker}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose:
        print(f"  Best validation loss: {best_val_loss:.4f}")

    return model, history


# =============================================================================
# Export
# =============================================================================

def export_to_json(model: TonnesjakkNNUE, output_path: str):
    """
    Export model weights to JSON for Rust inference.

    The JSON format matches what the Rust IncrementalNNUE expects.
    """
    state_dict = model.state_dict()
    weights = {}

    for name, tensor in state_dict.items():
        if "0.weight" in name:
            weights["fc1_weight"] = tensor.tolist()
        elif "0.bias" in name:
            weights["fc1_bias"] = tensor.tolist()
        elif "2.weight" in name:
            weights["fc2_weight"] = tensor.tolist()
        elif "2.bias" in name:
            weights["fc2_bias"] = tensor.tolist()
        elif "4.weight" in name:
            weights["fc3_weight"] = tensor.tolist()
        elif "4.bias" in name:
            weights["fc3_bias"] = tensor.tolist()

    output = {
        "hidden1": model.hidden1,
        "hidden2": model.hidden2,
        "weights": weights
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"Exported to {output_path}")


# =============================================================================
# Main Training Pipeline
# =============================================================================

def train_nnue(
    num_games: int = 10000,
    depth: int = 6,
    random_moves: int = 4,
    hidden1: int = 64,
    hidden2: int = 32,
    epochs: int = 50,
    output_dir: str = ".",
    use_nnue: Optional[str] = None,
    use_search_scores: bool = True,
    augment: bool = True,
    compare: bool = True,
    compare_games: int = 50,
    track_history: bool = True,
    save_data: Optional[str] = None,
    load_data: Optional[str] = None
) -> TonnesjakkNNUE:
    """
    Complete NNUE training pipeline.

    Recommended settings:
    - num_games: 10,000 - 20,000 for good results
    - depth: 6-8 (higher = better quality, slower generation)
    - random_moves: 4-6 (ensures diverse openings)
    - hidden1: 64-128 (first layer is most important)
    - hidden2: 32 (second layer can be smaller)
    - use_nnue: Path to existing NNUE weights for self-play (self-improvement loop)
    - use_search_scores: Use engine search scores (True) vs game outcomes (False)
    - augment: Apply horizontal flip augmentation (doubles training data)
    - save_data: Save generated positions to file for reuse
    - load_data: Load positions from file instead of generating
    """
    print("=" * 60)
    print("NNUE TRAINING FOR TONNESJAKK")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Architecture: {INPUT_SIZE} -> {hidden1} -> {hidden2} -> 1")
    print(f"  Features: {BASE_FEATURES} base + {RELATIONAL_FEATURES} relational")

    # Step 1: Generate or load data
    if load_data:
        print(f"\n[1/3] Loading training data from {load_data}...")
        X, y, stats, loaded_config = DataGenerator.load_dataset(load_data)
        print(f"  Loaded {len(X):,} positions")
        print(f"  {stats}")
        if loaded_config:
            print(f"  Original config: {loaded_config.get('games', '?')} games, depth {loaded_config.get('depth', '?')}")
    else:
        print(f"  Games: {num_games:,}")
        print(f"  Search depth: {depth}")
        print(f"  Random opening moves: {random_moves}")
        if use_nnue:
            print(f"  Self-play eval: NNUE ({use_nnue})")
        else:
            print(f"  Self-play eval: Heuristic")
        print(f"  Labels: {'search scores' if use_search_scores else 'game outcomes'}")
        print(f"  Augmentation: {'enabled (2x data)' if augment else 'disabled'}")

        print(f"\n[1/3] Generating training data...")
        generator = DataGenerator(nnue_path=use_nnue)
        X, y, stats = generator.generate_dataset(
            num_games=num_games,
            depth=depth,
            random_opening_moves=random_moves,
            use_search_scores=use_search_scores,
            augment=augment
        )

        # Save data if requested
        if save_data:
            config = {
                "games": num_games,
                "depth": depth,
                "random_moves": random_moves,
                "use_nnue": use_nnue,
                "use_search_scores": use_search_scores,
                "augment": augment,
                "input_size": INPUT_SIZE
            }
            generator.save_dataset(X, y, stats, save_data, config)
            print(f"  Saved dataset to: {save_data}")

    # Check balance
    if stats.balance_ratio < 0.5 or stats.balance_ratio > 2.0:
        print(f"\n  WARNING: Dataset is unbalanced (ratio: {stats.balance_ratio:.2f})")
        print("  Consider adjusting search depth or random moves")

    # Step 2: Train
    print(f"\n[2/3] Training model...")
    model, history = train_model(
        X, y,
        hidden1=hidden1,
        hidden2=hidden2,
        epochs=epochs
    )

    # Step 3: Export
    print(f"\n[3/3] Exporting...")
    output_path = Path(output_dir)
    json_path = output_path / "nnue_weights.json"
    old_weights_path = None

    # Backup old weights if they exist
    if json_path.exists():
        import shutil
        # Find next available backup number
        gen = 1
        while (output_path / f"nnue_weights_gen{gen}.json").exists():
            gen += 1
        old_weights_path = output_path / f"nnue_weights_gen{gen}.json"
        shutil.copy(json_path, old_weights_path)
        print(f"  Backed up old weights to: {old_weights_path}")

    # Save PyTorch model
    torch_path = output_path / "nnue_model.pt"
    torch.save(model.state_dict(), torch_path)
    print(f"  PyTorch model: {torch_path}")

    # Save JSON for Rust
    export_to_json(model, str(json_path))
    print(f"  JSON weights: {json_path}")

    # Step 4: Compare with previous version
    comparison = None
    if compare and old_weights_path:
        print(f"\n[4/4] Comparing with previous version ({compare_games} games)...")
        comparison = compare_nnue(
            str(json_path),
            str(old_weights_path),
            num_games=compare_games,
            depth=min(depth, 6),  # Use depth 6 max for faster comparison
            verbose=True
        )
        print(f"\n  Results: New={comparison['wins_a']} Old={comparison['wins_b']} Draws={comparison['draws']}")
        print(f"  Win rate: {comparison['win_rate_a']*100:.1f}%")
        print(f"  Estimated ELO diff: {comparison['elo_diff']:+d}")

        if comparison['elo_diff'] > 0:
            print("  [+] New model is STRONGER!")
        elif comparison['elo_diff'] < -20:
            print("  [!] New model appears WEAKER - consider reverting")
        else:
            print("  [=] Models are roughly equal strength")

    # Track history
    if track_history:
        history_path = output_path / "nnue_history.json"
        history = load_training_history(str(history_path))
        generation = len(history) + 1

        config = {
            "games": num_games,
            "depth": depth,
            "random_moves": random_moves,
            "arch": [hidden1, hidden2],
            "epochs": epochs,
            "used_nnue": use_nnue is not None,
            "search_scores": use_search_scores,
            "augment": augment,
            "input_size": INPUT_SIZE
        }
        add_training_result(generation, config, comparison, str(history_path))
        print(f"\n  Training history saved (generation {generation})")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Show history summary
    if track_history:
        print_training_history(str(output_path / "nnue_history.json"))

    return model


# =============================================================================
# NNUE Comparison and History
# =============================================================================

def compare_nnue(
    nnue_a: str,
    nnue_b: str,
    num_games: int = 100,
    depth: int = 6,
    verbose: bool = True
) -> Dict:
    """
    Play matches between two NNUE versions to compare strength.

    Args:
        nnue_a: Path to first NNUE weights (or "heuristic" for no NNUE)
        nnue_b: Path to second NNUE weights (or "heuristic" for no NNUE)
        num_games: Number of games to play
        depth: Search depth

    Returns:
        Dict with results: wins_a, wins_b, draws, win_rate_a, elo_diff
    """
    from tonnesjakk import Board, Engine

    engine_a = Engine()
    engine_b = Engine()

    if nnue_a != "heuristic":
        engine_a.load_nnue(nnue_a)
    if nnue_b != "heuristic":
        engine_b.load_nnue(nnue_b)

    wins_a = 0
    wins_b = 0
    draws = 0

    for game_idx in range(num_games):
        # Alternate colors
        white_is_a = (game_idx % 2 == 0)

        board = Board()
        engine_a.full_reset()
        engine_b.full_reset()

        # Random opening for variety
        for _ in range(random.randint(2, 4)):
            moves = board.generate_moves()
            if not moves or board.check_winner():
                break
            board.make_move(random.choice(moves))

        # Play game
        move_count = 0
        while board.check_winner() is None and move_count < 100:
            # Determine which engine plays
            is_white_turn = "White" in repr(board.current_player)
            current_engine = engine_a if (is_white_turn == white_is_a) else engine_b

            result = current_engine.search(board, depth)
            if result.best_move is None:
                break
            board.make_move(result.best_move)
            move_count += 1

        # Score result
        winner = board.check_winner()
        if winner is None:
            draws += 1
        elif "White" in repr(winner):
            if white_is_a:
                wins_a += 1
            else:
                wins_b += 1
        else:
            if white_is_a:
                wins_b += 1
            else:
                wins_a += 1

        if verbose and (game_idx + 1) % 10 == 0:
            print(f"  Game {game_idx + 1}/{num_games}: A={wins_a} B={wins_b} D={draws}")

    # Calculate stats
    total = wins_a + wins_b + draws
    win_rate_a = (wins_a + 0.5 * draws) / total if total > 0 else 0.5

    # Approximate ELO difference (using logistic model)
    # ELO diff = 400 * log10(win_rate / (1 - win_rate))
    import math
    if win_rate_a > 0.01 and win_rate_a < 0.99:
        elo_diff = 400 * math.log10(win_rate_a / (1 - win_rate_a))
    else:
        elo_diff = 400 if win_rate_a > 0.5 else -400

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": win_rate_a,
        "elo_diff": round(elo_diff),
        "games": total
    }


def load_training_history(path: str = "nnue_history.json") -> List[Dict]:
    """Load training history from JSON file."""
    if Path(path).exists():
        with open(path, 'r') as f:
            return json.load(f)
    return []


def save_training_history(history: List[Dict], path: str = "nnue_history.json"):
    """Save training history to JSON file."""
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def add_training_result(
    generation: int,
    config: Dict,
    comparison: Optional[Dict] = None,
    history_path: str = "nnue_history.json"
):
    """Add a training result to the history."""
    history = load_training_history(history_path)

    entry = {
        "generation": generation,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "comparison": comparison
    }
    history.append(entry)

    save_training_history(history, history_path)
    return entry


def print_training_history(history_path: str = "nnue_history.json"):
    """Print training history summary."""
    history = load_training_history(history_path)

    if not history:
        print("No training history found.")
        return

    print("\n" + "=" * 60)
    print("TRAINING HISTORY")
    print("=" * 60)
    print(f"{'Gen':>4} {'Date':>12} {'Games':>7} {'Depth':>5} {'vs Prev':>10} {'ELO Diff':>10}")
    print("-" * 60)

    for entry in history:
        gen = entry.get("generation", "?")
        date = entry.get("timestamp", "?")[:10]
        config = entry.get("config", {})
        games = config.get("games", "?")
        depth = config.get("depth", "?")

        comp = entry.get("comparison")
        if comp:
            win_rate = f"{comp['win_rate_a']*100:.0f}%"
            elo = f"{comp['elo_diff']:+d}"
        else:
            win_rate = "-"
            elo = "-"

        print(f"{gen:>4} {date:>12} {games:>7} {depth:>5} {win_rate:>10} {elo:>10}")

    print("=" * 60)


# =============================================================================
# Testing and Benchmarking
# =============================================================================

def quick_test():
    """Quick test of NNUE implementation."""
    print("Testing NNUE...")

    model = TonnesjakkNNUE()
    print(f"Model parameters: {model.num_parameters:,}")

    # Test with empty board
    empty_board = [[0]*6 for _ in range(6)]
    x = board_to_tensor(empty_board).unsqueeze(0)
    score = model(x).item()
    print(f"Empty board score: {score:.4f} (expected: ~0)")

    # Test with some pieces
    board = [[0]*6 for _ in range(6)]
    board[5][2] = 1   # White barrel
    board[0][3] = -1  # Black barrel
    x = board_to_tensor(board).unsqueeze(0)
    score = model(x).item()
    print(f"Board with pieces: {score:.4f}")

    print("Test OK!")


def benchmark():
    """Benchmark data generation and training speed."""
    print("Benchmarking...")

    generator = DataGenerator()

    # Benchmark game generation
    print("\n1. Game generation speed:")
    for depth in [4, 6, 8]:
        start = time.time()
        for _ in range(10):
            generator.play_game(depth=depth, random_opening_moves=4)
        elapsed = time.time() - start
        print(f"  Depth {depth}: {10/elapsed:.1f} games/sec")

    # Benchmark training
    print("\n2. Training speed (1000 positions):")
    X = torch.randn(1000, INPUT_SIZE)
    y = torch.randn(1000, 1)

    for arch in [(64, 32), (128, 32), (64, 16)]:
        model = TonnesjakkNNUE(arch[0], arch[1])
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        start = time.time()
        for _ in range(10):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        elapsed = time.time() - start

        print(f"  144->{arch[0]}->{arch[1]}->1: {10/elapsed:.1f} epochs/sec "
              f"({model.num_parameters:,} params)")

    print("\nBenchmark complete!")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train NNUE for Tonnesjakk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tonnesjakk.nnue                    # Default training (10K games)
  python -m tonnesjakk.nnue --games 10000 --depth 8 --save-data data.npz  # Save positions
  python -m tonnesjakk.nnue --load-data data.npz --epochs 100  # Reuse positions
  python -m tonnesjakk.nnue --use-nnue nnue_weights.json  # Self-improvement loop
  python -m tonnesjakk.nnue --history          # Show training history
  python -m tonnesjakk.nnue --compare nnue_weights.json heuristic  # Compare versions
        """
    )

    parser.add_argument("--games", type=int, default=10000,
                        help="Number of self-play games (default: 10000)")
    parser.add_argument("--depth", type=int, default=6,
                        help="Search depth for self-play (default: 6)")
    parser.add_argument("--random-moves", type=int, default=4,
                        help="Random opening moves (default: 4)")
    parser.add_argument("--arch", type=int, nargs=2, default=[64, 32],
                        metavar=("H1", "H2"),
                        help="Hidden layer sizes (default: 64 32)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory (default: current)")
    parser.add_argument("--use-nnue", type=str, default=None,
                        help="Use existing NNUE weights for self-play (self-improvement loop)")
    parser.add_argument("--save-data", type=str, default=None,
                        help="Save generated positions to .npz file for reuse")
    parser.add_argument("--load-data", type=str, default=None,
                        help="Load positions from .npz file instead of generating")
    parser.add_argument("--no-search-scores", action="store_true",
                        help="Use game outcomes instead of search scores for labels")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable horizontal flip data augmentation")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip comparison with previous version")
    parser.add_argument("--compare-games", type=int, default=50,
                        help="Number of games for comparison (default: 50)")
    parser.add_argument("--no-history", action="store_true",
                        help="Don't track training history")
    parser.add_argument("--history", action="store_true",
                        help="Show training history and exit")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("NNUE_A", "NNUE_B"),
                        help="Compare two NNUE versions (use 'heuristic' for no NNUE)")
    parser.add_argument("--test", action="store_true",
                        help="Run quick test")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark")

    args = parser.parse_args()

    if args.test:
        quick_test()
    elif args.benchmark:
        benchmark()
    elif args.history:
        print_training_history(str(Path(args.output) / "nnue_history.json"))
    elif args.compare:
        print(f"Comparing {args.compare[0]} vs {args.compare[1]}...")
        result = compare_nnue(
            args.compare[0],
            args.compare[1],
            num_games=args.compare_games,
            depth=6,
            verbose=True
        )
        print(f"\nFinal: A={result['wins_a']} B={result['wins_b']} D={result['draws']}")
        print(f"Win rate A: {result['win_rate_a']*100:.1f}%")
        print(f"ELO difference: {result['elo_diff']:+d} (A vs B)")
    else:
        train_nnue(
            num_games=args.games,
            depth=args.depth,
            random_moves=args.random_moves,
            hidden1=args.arch[0],
            hidden2=args.arch[1],
            epochs=args.epochs,
            output_dir=args.output,
            use_nnue=args.use_nnue,
            use_search_scores=not args.no_search_scores,
            augment=not args.no_augment,
            compare=not args.no_compare,
            compare_games=args.compare_games,
            track_history=not args.no_history,
            save_data=args.save_data,
            load_data=args.load_data
        )


if __name__ == "__main__":
    main()
