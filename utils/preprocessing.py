import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data/raw")
AUDIO_DIR = DATA_DIR / "fma_small"
TRACKS_CSV = DATA_DIR / "tracks.csv"

# Read the metadata
tracks = pd.read_csv(
    TRACKS_CSV,
    header=[0, 1],
    index_col=0
)

# Find all MP3 files recursively
audio_files = sorted(AUDIO_DIR.rglob("*.mp3"))

# Verify everything
print("Type of audio_files:", type(audio_files))
print("Total audio files:", len(audio_files))
print("First audio file:", audio_files[0])

def get_track_id(filepath):
    return(int(filepath.stem))
track_id = get_track_id(audio_files[0])

track_info = tracks.loc[track_id]

genre = track_info[("track", "genre_top")]

print("Track ID:", track_id)
print("Genre:", genre)

genre = track_info[("track", "genre_top")]
print (genre)