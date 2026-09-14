from pathlib import Path


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_DIR = BASE_DIR / "data" / "training"
UPLOAD_DATA_DIR = BASE_DIR / "data" / "uploads"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


# =========================================================
# DATA / RESOURCE LIMITS
# =========================================================

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_ROWS = 500_000
MAX_FILES = 4


# =========================================================
# UI COLOR SYSTEM
# =========================================================
# Global theme: dark blue + warm cream + muted brown.
# This matches the intended application UI rather than the purple/pink palette.
# All frontend modules should import UI colors from this file.

COLOR_BACKGROUND = "#0D2F4A"
COLOR_SURFACE = "#1F4562"
COLOR_SURFACE_LIGHT = "#274F6D"
COLOR_SURFACE_ALT = "#234B68"

# Typography
COLOR_TEXT = "#FAF0E6"
COLOR_TEXT_MUTED = "#B8C3C8"
COLOR_TEXT_MUTED_DARK = "#91A4AF"
COLOR_TEXT_DARK = "#0D2F4A"
COLOR_WHITE = "#FFFFFF"

# Light analytical cards / customer profile cards.
COLOR_CARD = "#D3D4C0"
COLOR_CARD_TEXT = "#0D2F4A"
COLOR_CARD_MUTED = "#526A78"

# Borders / structure
COLOR_BORDER = "#46667E"
COLOR_GRID = "#527087"

# Primary / supporting data colors
COLOR_PRIMARY = "#9A6A3F"      # Muted brown / primary action
COLOR_SECONDARY = "#6F8492"    # Slate blue
COLOR_TERTIARY = "#B0A98E"     # Warm muted neutral
COLOR_ALERT = "#B8734E"         # Muted terracotta
COLOR_NEUTRAL = "#7E927D"       # Muted sage


# =========================================================
# RISK COLORS
# =========================================================

RISK_COLORS = {
    "Low": COLOR_NEUTRAL,
    "Medium": COLOR_PRIMARY,
    "High": COLOR_SECONDARY,
    "Very High": COLOR_ALERT,
}


# =========================================================
# RETENTION PRIORITY COLORS
# =========================================================

RETENTION_PRIORITY_COLORS = {
    "Low": COLOR_NEUTRAL,
    "Medium": COLOR_PRIMARY,
    "High": COLOR_SECONDARY,
    "Critical": COLOR_ALERT,
}


# =========================================================
# STANDARD CHART PALETTE
# =========================================================

CHART_PALETTE = [
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_NEUTRAL,
    COLOR_TERTIARY,
    COLOR_ALERT,
]
