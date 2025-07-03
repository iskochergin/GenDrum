# the script is to refine the dataset + to add genres for each track

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

# need conda env (without idk)
from musicnn.tagger import top_tags

import csv

SRC_ROOT = "slakh2100_16k"
DST_ROOT = "processed_data"
INDEX_PATH = os.path.join(SRC_ROOT, "slakh_index_2100-yourmt3-16k.json")
SPLITS = ["train", "validation", "test"]
GENRE_CSV = os.path.join(DST_ROOT, "genres.csv")

with open(INDEX_PATH) as f:
    index = json.load(f)

os.makedirs(DST_ROOT, exist_ok=True)
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
            tags = top_tags(mix_path, model="MTT_musicnn", topN=1)
            writer.writerow([split, track, tags[0]])
