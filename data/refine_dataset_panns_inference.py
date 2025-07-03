"""EXPECTED STRUCTURE

processed_data/
├── genres.csv                 # CSV mapping split, track ID, and assigned genre
├── train/                     # 700 training tracks
│   ├── Track00001/
│   │   ├── backing.wav        # mixed non-drum stems (mono, 16 kHz)
│   │   ├── drums.wav          # drum stem (mono, 16 kHz)
│   │   ├── drums.mid          # drum MIDI aligned to backing.wav
│   │   └── genre.txt          # (optional) file containing the top-1 genre tag
│   ├── Track00002/
│   │   ├── backing.wav
│   │   ├── drums.wav
│   │   ├── drums.mid
│   │   └── genre.txt
│   └── …                      # up to Track00700
├── validation/                # 200 validation tracks
│   ├── Track00701/
│   │   ├── backing.wav
│   │   ├── drums.wav
│   │   ├── drums.mid
│   │   └── genre.txt
│   └── …                      # up to Track00899
└── test/                      # 200 test tracks
    ├── Track00900/
    │   ├── backing.wav
    │   ├── drums.wav
    │   ├── drums.mid
    │   └── genre.txt
    └── …                      # up to Track01099

"""

# probably need to consider the way how to work with genres, because now the split is native,
# so it doesn't support split across genres...

import os
import json
import subprocess
import shutil

# kinda works
'''to install
# 1. (If you don’t already have PyTorch + torchaudio)
pip install torch torchaudio

# 2. The PANNs inference package itself
pip install panns-inference

# 3. Other Python deps used in the script
pip install librosa numpy
'''
from panns_inference import AudioTagging, labels as PANN_LABELS

import librosa
import numpy as np
import csv

SRC_ROOT = "slakh2100_16k"
DST_ROOT = "processed_data"
INDEX_PATH = os.path.join(SRC_ROOT, "slakh_index_2100-yourmt3-16k.json")
SPLITS = ["train", "validation", "test"]
GENRE_CSV = os.path.join(DST_ROOT, "genres.csv")

with open(INDEX_PATH) as f:
    index = json.load(f)

os.makedirs(DST_ROOT, exist_ok=True)

tagger = AudioTagging(checkpoint_path=None, device="cpu")

with open(GENRE_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["split", "track", "genre"])
    for split in SPLITS:
        for track, info in index[split].items():
            src = os.path.join(SRC_ROOT, split, track)
            dst = os.path.join(DST_ROOT, split, track)
            os.makedirs(dst, exist_ok=True)

            stems = os.path.join(src, "stems")
            midi = os.path.join(src, "midi")
            drum_ids = [sid for sid, m in info["stems"].items() if m["is_drum"]]
            non_ids  = [sid for sid, m in info["stems"].items() if not m["is_drum"]]

            drum = drum_ids[0]
            subprocess.run([
                "ffmpeg", "-y",
                "-i", os.path.join(stems, f"{drum}.wav"),
                "-ar", "16000", "-ac", "1",
                os.path.join(dst, "drums.wav")
            ], check=True)
            shutil.copy(
                os.path.join(midi, f"{drum}.mid"),
                os.path.join(dst, "drums.mid")
            )

            inputs = []
            for sid in non_ids:
                inputs += ["-i", os.path.join(stems, f"{sid}.wav")]
            filter_complex = "".join(f"[{i}:0]" for i in range(len(non_ids))) + f"amix=inputs={len(non_ids)}"
            subprocess.run(
                ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-ar", "16000", "-ac", "1",
                 os.path.join(dst, "backing.wav")],
                check=True
            )

            mix_path = os.path.join(src, "mix.wav")
            audio, _ = librosa.load(mix_path, sr=32000, mono=True)
            clipwise_output, _ = tagger.inference(audio[None, :])
            genre = PANN_LABELS[int(np.argmax(clipwise_output))]
            writer.writerow([split, track, genre])
