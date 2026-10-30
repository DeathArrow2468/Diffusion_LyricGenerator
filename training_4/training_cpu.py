"""CPU entry point; shares the same loss and training loop as CUDA."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_4.training import train

if __name__ == '__main__':
    train(device='cpu')
