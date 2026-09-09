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

# =========================================================
# UI COLOR SYSTEM
# =========================================================

# --- Application ---
COLOR_BACKGROUND = "#0E1117"
COLOR_SURFACE = "#161A1F"
COLOR_SURFACE_LIGHT = "#1E2329"

# --- Typography ---
COLOR_TEXT = "#E8E8E8"
COLOR_TEXT_MUTED = "#92979D"
COLOR_TEXT_DARK = "#25282B"

# --- Borders / Structure ---
COLOR_BORDER = "#30343A"
COLOR_GRID = "#282C31"

# --- Primary Data Color ---
COLOR_PRIMARY = "#7F929C"       # Smoky Blue-Gray

# --- Supporting Data Colors ---
COLOR_SECONDARY = "#849187"     # Dusty Sage
COLOR_TERTIARY = "#8D8795"      # Smoky Lavender
COLOR_ALERT = "#987F82"         # Dusty Rose
COLOR_NEUTRAL = "#777C80"       # Charcoal Gray

# =========================================================
# RISK COLORS
# =========================================================

RISK_COLORS = {
    "Low": COLOR_SECONDARY,
    "Medium": COLOR_PRIMARY,
    "High": COLOR_TERTIARY,
    "Very High": COLOR_ALERT,
}


# =========================================================
# RETENTION PRIORITY COLORS
# =========================================================

RETENTION_PRIORITY_COLORS = {
    "Low": COLOR_SECONDARY,
    "Medium": COLOR_PRIMARY,
    "High": COLOR_TERTIARY,
    "Critical": COLOR_ALERT,
}


# =========================================================
# STANDARD CHART PALETTE
# =========================================================

CHART_PALETTE = [
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    COLOR_ALERT,
    COLOR_NEUTRAL,
]