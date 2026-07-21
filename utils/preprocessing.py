import numpy as np
import pandas as pd
import librosa

from pathlib import Path
from tqdm import tqdm

# =====================================================
# Paths
# =====================================================

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

AUDIO_DIR = DATA_DIR / "fma_small"
TRACKS_CSV = DATA_DIR / "tracks.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Audio Parameters
# =====================================================

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 30

EXPECTED_SAMPLES = SAMPLE_RATE * DURATION

# =====================================================
# Split Parameters
# =====================================================

VAL_SIZE = 700      # 700 songs for validation
TEST_SIZE = 700     # 700 songs for testing
RANDOM_SEED = 42    # For reproducibility

# =====================================================
# Load Metadata
# =====================================================

print("Loading metadata...")

tracks = pd.read_csv(
    TRACKS_CSV,
    header=[0, 1],
    index_col=0
)

# =====================================================
# Find Audio Files
# =====================================================

audio_files = sorted(AUDIO_DIR.rglob("*.mp3"))

print(f"Found {len(audio_files)} audio files.")

# =====================================================
# Genre Mapping
# =====================================================

genres = sorted(
    tracks[("track", "genre_top")]
    .dropna()
    .unique()
)

genre_to_index = {
    genre: idx
    for idx, genre in enumerate(genres)
}

print("\nGenres:")
for genre, idx in genre_to_index.items():
    print(f"{idx:2d} -> {genre}")

# =====================================================
# Helper Function
# =====================================================

def get_track_id(filepath):
    return int(filepath.stem)

# =====================================================
# Storage Lists
# =====================================================

X = []
y = []

# =====================================================
# Process Every Audio File
# =====================================================

print("\nProcessing audio files...\n")

for audio_path in tqdm(audio_files):

    track_id = get_track_id(audio_path)

    try:
        track_info = tracks.loc[track_id]
    except KeyError:
        continue

    genre = track_info[("track", "genre_top")]

    if pd.isna(genre):
        continue

    try:
        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE
        )

        # Pad Short Audio
        if len(audio) < EXPECTED_SAMPLES:
            padding = EXPECTED_SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding))

        # Trim Long Audio
        else:
            audio = audio[:EXPECTED_SAMPLES]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        X.append(mel_db.astype(np.float32))
        y.append(genre_to_index[genre])

    except Exception as e:
        print(f"Skipped {audio_path.name}: {e}")

# =====================================================
# Convert To NumPy
# =====================================================

X = np.array(X)
y = np.array(y)

print("\nFinished!")
print("Spectrogram Shape :", X.shape)
print("Labels Shape      :", y.shape)

# =====================================================
# Train / Validation / Test Split
# =====================================================

total_samples = len(X)

print(f"\nTotal samples: {total_samples}")
print(f"Validation size: {VAL_SIZE}")
print(f"Test size: {TEST_SIZE}")
print(f"Train size: {total_samples - VAL_SIZE - TEST_SIZE}")

assert total_samples >= VAL_SIZE + TEST_SIZE, (
    f"Not enough samples ({total_samples}) for "
    f"val ({VAL_SIZE}) + test ({TEST_SIZE}) = {VAL_SIZE + TEST_SIZE}"
)

# Shuffle indices randomly
rng = np.random.default_rng(RANDOM_SEED)
indices = np.arange(total_samples)
rng.shuffle(indices)

# Split: first 700 -> test, next 700 -> val, rest -> train
test_indices = indices[:TEST_SIZE]
val_indices = indices[TEST_SIZE:TEST_SIZE + VAL_SIZE]
train_indices = indices[TEST_SIZE + VAL_SIZE:]

X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]
X_test, y_test = X[test_indices], y[test_indices]

print(f"\nSplit complete:")
print(f"  Train      : {X_train.shape[0]} samples")
print(f"  Validation : {X_val.shape[0]} samples")
print(f"  Test       : {X_test.shape[0]} samples")

# =====================================================
# Save Dataset
# =====================================================

# Save train split
np.save(PROCESSED_DIR / "X_train.npy", X_train)
np.save(PROCESSED_DIR / "y_train.npy", y_train)

# Save validation split
np.save(PROCESSED_DIR / "X_val.npy", X_val)
np.save(PROCESSED_DIR / "y_val.npy", y_val)

# Save test split
np.save(PROCESSED_DIR / "X_test.npy", X_test)
np.save(PROCESSED_DIR / "y_test.npy", y_test)

# Save Genre Mapping

genre_df = pd.DataFrame({
    "genre": list(genre_to_index.keys()),
    "label": list(genre_to_index.values())
})

genre_df.to_csv(
    PROCESSED_DIR / "genre_mapping.csv",
    index=False
)

print("\nSaved:")
print("  data/processed/X_train.npy")
print("  data/processed/y_train.npy")
print("  data/processed/X_val.npy")
print("  data/processed/y_val.npy")
print("  data/processed/X_test.npy")
print("  data/processed/y_test.npy")
print("  data/processed/genre_mapping.csv")