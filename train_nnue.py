#!/usr/bin/env python
"""
Convenience script to run NNUE training.

Usage:
    python train_nnue.py                  # Default (10K games)
    python train_nnue.py --games 20000    # More games
    python train_nnue.py --test           # Quick test
    python train_nnue.py --benchmark      # Speed benchmark
"""
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from tonnesjakk.nnue import main

if __name__ == "__main__":
    main()
