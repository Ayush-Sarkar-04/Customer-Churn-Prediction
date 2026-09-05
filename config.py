from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_DIR = BASE_DIR / "data" / "training"
UPLOAD_DATA_DIR = BASE_DIR / "data" / "uploads"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_ROWS = 500_000
MAX_FILES = 4