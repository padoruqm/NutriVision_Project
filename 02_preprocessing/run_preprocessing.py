import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.load_dataset import load_and_split
from src.filtering import process_dataset
def main():
    load_and_split()
    process_dataset()

if __name__ == "__main__":
    main()
