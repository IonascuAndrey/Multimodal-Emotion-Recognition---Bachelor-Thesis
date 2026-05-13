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
EPOCHS         = 200
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


# ════════════════════════════════════════════════════════════════════════════
# AUDIO
# ════════════════════════════════════════════════════════════════════════════

# ── Audio data paths ──────────────────────────────────────────────────────────
# Files are named dia{dialogue_id}_utt{utterance_id}.wav
AUDIO_TRAIN_DIR = "../Data/MELD.Audio/train"
AUDIO_DEV_DIR   = "../Data/MELD.Audio/dev"
AUDIO_TEST_DIR  = "../Data/MELD.Audio/test"

# ── Audio processing ──────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000          # Hz — wav2vec2 / DistilHuBERT both expect 16 kHz
MAX_AUDIO_S   = 10.0            # Truncate waveforms longer than this (seconds)

# Laughter trim: cut a fixed amount from each end before any other processing.
# Set to 0.0 to disable. Units: seconds.
TRIM_START_S  = 0.0             # e.g. 0.5 removes the first 500 ms
TRIM_END_S    = 0.0             # e.g. 0.5 removes the last  500 ms

# ── Audio experiment output ───────────────────────────────────────────────────
EXP_ROOT_AUDIO = "../experiments/audio_unimodal"
# VERSION is shared with text (bump once to refresh both)

# ── Audio batch size ──────────────────────────────────────────────────────────
# Raw waveforms are much larger than token sequences; reduce if you hit OOM.
AUDIO_BATCH_SIZE = 8

# ── Audio models ──────────────────────────────────────────────────────────────
AUDIO_MODELS_TO_RUN = [
    "facebook/wav2vec2-base",
    "ntu-spml/distilhubert",
]
# ════════════════════════════════════════════════════════════════════════════
# FACE / VIDEO PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

# ── Video source directories (one per split) ──────────────────────────────────
# Within each directory, files are named dia{Dialogue_ID}_utt{Utterance_ID}.mp4
FACE_VIDEO_DIRS = {
    "train": "../Data/MELD.Raw/train/train_splits",
    "dev":   "../Data/MELD.Raw/dev/dev_splits_complete",
    "test":  "../Data/MELD.Raw/test/output_repeated_splits_test",
}

# CSV paths (reuses the text CSVs — same rows, same labels)
FACE_CSV_PATHS = {
    "train": TRAIN_CSV,
    "dev":   DEV_CSV,
    "test":  TEST_CSV,
}

# ── Output ────────────────────────────────────────────────────────────────────
FACE_OUTPUT_ROOT = "../Data/processed_meld_faces"
# VERSION is shared — bump it to start a clean preprocessing run

# ── FER scoring model ─────────────────────────────────────────────────────────
# enet_b0_8_best_afew is trained on AFEW (movie/TV clips) — best domain match
# for MELD. It outputs 8 classes; Contempt is dropped and the rest renormalised.
FACE_FER_MODEL    = "enet_b0_8_best_afew"
FACE_FER_N_CLASSES = 8   # raw output size before remapping

# Class order for the 8-class AFEW model
FACE_IDX_TO_CLASS_8 = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]
# Indices to keep when remapping to the 7 MELD classes (drop Contempt = idx 1)
FACE_AFEW8_TO_MELD7 = [0, 2, 3, 4, 5, 6, 7]
FACE_MELD_CLASSES   = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

# ── Frame extraction ──────────────────────────────────────────────────────────
FACE_FRAME_STRIDE  = 3      # sample every Nth frame (1 = all, 3 = ~10 fps at 30 fps source)
FACE_K_FRAMES      = 5      # how many face crops to save per utterance
FACE_MIN_GAP_S     = 0.3    # minimum seconds between any two selected frames
FACE_EXPAND_SCALE  = 1.2    # expand MTCNN box by this factor before cropping
FACE_MIN_FACE_PX   = 48     # minimum face width AND height (pixels) after expansion

# ── Scoring ───────────────────────────────────────────────────────────────────
# score = gt_prob_renorm * max_prob_renorm
# Selects frames that are both GT-matching and high-confidence.
FACE_SMOOTH_SIGMA_S = 0.5   # Gaussian smoothing window in seconds before peak detection

# ── Speaker embedding tracking ────────────────────────────────────────────────
# Uses facenet_pytorch InceptionResnetV1 (pretrained on VGGFace2) to match
# faces to known speakers via cosine similarity.
FACE_EMBEDDING_EMA  = 0.9   # exponential moving average decay for speaker embeddings

# ── Checkpointing ─────────────────────────────────────────────────────────────
FACE_SAVE_EVERY = 100       # flush metadata CSV every N completed utterances

# ════════════════════════════════════════════════════════════════════════════
# FACE / VIDEO MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

# ── Models ────────────────────────────────────────────────────────────────────
# "vit_b_16"           — torchvision ViT-B/16 pretrained on ImageNet-21k
# "mobilenet_v3_small" — torchvision MobileNetV3-Small (lightweight baseline)
FACE_MODELS_TO_RUN = ["vit_b_16", "mobilenet_v3_small"]
# FACE_MODELS_TO_RUN = ["mobilenet_v3_small"]

# ── Experiment output ─────────────────────────────────────────────────────────
EXP_ROOT_FACE = "../experiments/face_unimodal"
# VERSION is shared — bump to start a clean training run

# ── Image preprocessing ───────────────────────────────────────────────────────
FACE_IMG_SIZE = 224         # resize both dimensions to this before normalisation

# ── Training hyperparameters ──────────────────────────────────────────────────
FACE_TRAIN_BATCH_SIZE  = 32
FACE_TRAIN_EPOCHS      = 20
FACE_TRAIN_LR          = 1e-4
FACE_WEIGHT_DECAY      = 1e-2
FACE_WARMUP_RATIO      = 0.1
FACE_GRAD_CLIP_NORM    = 1.0

# ── NPZ pixel cache ───────────────────────────────────────────────────────────
# Pre-decodes every PNG crop to uint8 (K, 224, 224, 3) arrays stored as .npz.
# Loading from cache skips repeated JPEG/PNG decoding and is ~3-5× faster.
# Set to False to always read PNGs from disk (useful when disk space is tight).
FACE_USE_CACHE  = True
FACE_CACHE_DIR  = "../Data/processed_meld_faces/cache"

# ── Score-weighted loss ───────────────────────────────────────────────────────
# Each frame's cross-entropy loss is multiplied by a weight derived from the
# FER score saved during preprocessing:
#   w_raw = clip(score_comb ^ FACE_SCORE_WEIGHT_POWER, FACE_SCORE_WEIGHT_MIN, 1.0)
# If FACE_SCORE_WEIGHT_NORMALIZE is True the weights in each mini-batch are
# divided by their mean so the effective learning rate stays constant.
# Set FACE_USE_SCORE_WEIGHTS = False to fall back to plain cross-entropy.
FACE_USE_SCORE_WEIGHTS       = True
FACE_SCORE_WEIGHT_POWER      = 1.0    # exponent applied to score_comb
FACE_SCORE_WEIGHT_MIN        = 0.1    # floor weight (prevents near-zero frames from being ignored entirely)
FACE_SCORE_WEIGHT_NORMALIZE  = True   # normalise batch weights to mean = 1

# ── Anti-overfitting ──────────────────────────────────────────────────────────

# Differential learning rates
# The pretrained backbone is updated at lr × FACE_BACKBONE_LR_MULT;
# the new classification head trains at lr (the value set in FACE_TRAIN_LR).
FACE_BACKBONE_LR_MULT   = 0.1   # backbone LR = head LR × 0.1  (i.e. 1e-5 vs 1e-4)

# Backbone freeze warm-up
# Train ONLY the classification head for the first N epochs so the head can
# orient itself before the backbone starts moving.  After N epochs the full
# model is fine-tuned with differential LRs.
FACE_FREEZE_EPOCHS      = 3

# Label smoothing — prevents overconfidence on training samples
FACE_LABEL_SMOOTHING    = 0.1

# Class-balanced sampling
# Draws training samples so every emotion class appears roughly equally often
# per epoch.  Helps with MELD's heavy neutral / happiness imbalance.
FACE_BALANCED_SAMPLER   = True

# Dropout in the ViT classification head
# MobileNet already has a Dropout in its classifier; ViT's head is bare Linear.
FACE_HEAD_DROPOUT       = 0.3

# RandomErasing augmentation
# Randomly blacks out a rectangular patch of the face image during training,
# forcing the model not to rely on specific facial regions.
FACE_RANDOM_ERASING_P   = 0.3


# ════════════════════════════════════════════════════════════════════════════
# VIDEO MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

# ── Models ────────────────────────────────────────────────────────────────────
# "mvit_v2_s" — Multiscale Vision Transformer V2-Small (torchvision, 35 M params)
#               Strong temporal modelling via multiscale attention.
# "s3d"       — Separable 3D convolutions (torchvision, ~8 M params, fast).
VIDEO_MODELS_TO_RUN = ["mvit_v2_s", "s3d"]

# ── Experiment output ─────────────────────────────────────────────────────────
EXP_ROOT_VIDEO = "../experiments/video_unimodal"

# ── Temporal clip length (frames per clip fed to the model) ───────────────────
# MViT-V2-S needs T=16; S3D is flexible — T=8 keeps it within 12 GB VRAM.
# VIDEO_T_CACHE is the number of frames stored per utterance in the frame
# cache (= the maximum T across models).  Smaller-T models subsample at
# load time.
VIDEO_T_FRAMES: dict = {"mvit_v2_s": 16, "s3d": 8}
VIDEO_T_CACHE        = 16    # frames stored per utterance (max across models)
VIDEO_IMG_SIZE       = 224

# ── Video frame cropping ──────────────────────────────────────────────────────
# "face"  : crop to the speaker's face region using face-preprocessing boxes.
# "full"  : feed the full video frame.
VIDEO_CROP_MODE         = "face"
VIDEO_FACE_BOX_MAX_GAP  = 10    # max frames from nearest detected box before
                                 # falling back to full-frame crop

# ── Frame cache ───────────────────────────────────────────────────────────────
# One .npy file per utterance: (VIDEO_T_CACHE, H, W, 3) uint8.
# Stored at: <VIDEO_CACHE_DIR>/npy/<version>/<split>/dia{D}_utt{U}.npy
VIDEO_CACHE_DIR = "../Data/processed_meld_video/cache"

# ── Batch sizes and gradient accumulation ────────────────────────────────────
# Effective batch = batch_size × grad_accum_steps = 16 for both models.
# Tuned to keep peak VRAM ≤ 10 GB on an RTX 3060/3070 (12 GB card).
VIDEO_TRAIN_BATCH_SIZE: dict = {"mvit_v2_s": 4, "s3d": 8}
VIDEO_GRAD_ACCUM_STEPS: dict = {"mvit_v2_s": 4, "s3d": 2}

# ── Training hyperparameters ──────────────────────────────────────────────────
VIDEO_TRAIN_EPOCHS   = 20
VIDEO_TRAIN_LR       = 1e-4
VIDEO_WEIGHT_DECAY   = 1e-2
VIDEO_WARMUP_RATIO   = 0.1
VIDEO_GRAD_CLIP_NORM = 1.0

# ── Anti-overfitting (same strategy as face modality) ─────────────────────────
VIDEO_BACKBONE_LR_MULT  = 0.1   # backbone LR = head LR × 0.1
VIDEO_FREEZE_EPOCHS     = 3     # train head-only for first N epochs
VIDEO_LABEL_SMOOTHING   = 0.1
VIDEO_BALANCED_SAMPLER  = True
VIDEO_HEAD_DROPOUT      = 0.3   # dropout before MViT classification Linear
VIDEO_RANDOM_ERASING_P  = 0.3   # per-frame RandomErasing probability
