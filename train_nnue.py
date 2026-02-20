#!/usr/bin/env python
"""
Convenience script to run HalfPail NNUE training.

Usage:
    python train_nnue.py                                    # Default training
    python train_nnue.py --load-data data.bin --epochs 50   # Train on data
    python train_nnue.py --games 20000 --save-data data.bin # Generate data
"""
import sys
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from tonnesjakk.nnue import main

if __name__ == "__main__":
    main()
