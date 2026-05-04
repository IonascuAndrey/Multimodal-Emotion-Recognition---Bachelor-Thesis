import torch

# ── Data paths ────────────────────────────────────────────────────────────────
# Relative to the notebooks/ directory (same depth as the old Code/ folder).
TRAIN_CSV = "../Data/MELD.Raw/train/train_sent_emo.csv"
DEV_CSV   = "../Data/MELD.Raw/dev_sent_emo.csv"
TEST_CSV  = "../Data/MELD.Raw/test_sent_emo.csv"

# ── Column names ──────────────────────────────────────────────────────────────
TEXT_COL  = "Utterance"
LABEL_COL = "Emotion"

# ── Label remapping ───────────────────────────────────────────────────────────
# MELD ships with "joy"; we rename it to "happiness" for clarity.
LABEL_REMAP = {"joy": "happiness"}

# ── Experiment output ─────────────────────────────────────────────────────────
EXP_ROOT = "../experiments/text_unimodal"
VERSION  = "v1"   # bump when you want a fresh run directory

# ── Training hyperparameters ──────────────────────────────────────────────────
MAX_LEN        = 128
BATCH_SIZE     = 16
EPOCHS         = 4
LR             = 2e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.06
GRAD_CLIP_NORM = 1.0
SEED           = 42

# ── Models ────────────────────────────────────────────────────────────────────
MODELS_TO_RUN = [
    "bert-base-uncased",
    "distilbert-base-uncased",
]

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
