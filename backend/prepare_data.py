"""Build data/train/<class>/*.wav from the raw downloads in data/raw.

Sources and label mapping:
  p2016      PhysioNet/CinC 2016   normal -> normal, abnormal -> other_abnormal
  circor     CirCor 2022 (Kaggle)  murmur Present -> murmur, Absent -> normal (Unknown skipped)
  heartbeat  Pascal/Kaggle         normal, murmur, extrahls -> other_abnormal, extrastole -> extrasystole
  valve      AS/MR/MS/MVP/N clips  -> aortic_stenosis, mitral_regurgitation, mitral_stenosis,
                                      mitral_valve_prolapse, normal
Filenames keep a source prefix; CirCor names carry the patient id (circor_<pid>_<loc>.wav)
so train.py can split by patient.
"""
import csv, random, shutil
from pathlib import Path

RAW, OUT = Path("data/raw"), Path("data/train")
CIRCOR_NORMAL_CAP = 1200  # CirCor is ~80% murmur-absent; cap to keep classes workable
random.seed(0)


def put(src: Path, cls: str, name: str):
    (OUT / cls).mkdir(parents=True, exist_ok=True)
    shutil.copy(src, OUT / cls / name)


def p2016():
    for ref in RAW.glob("p2016/**/REFERENCE.csv"):
        for name, lab in csv.reader(ref.open()):
            src = ref.parent / f"{name}.wav"
            if src.exists():
                put(src, "normal" if int(lab) == -1 else "other_abnormal", f"p2016_{ref.parent.name}_{name}.wav")


def circor():
    d = RAW / "circor_kaggle"
    rows = list(csv.DictReader((d / "training_data.csv").open()))
    absent = []
    for r in rows:
        if r["Murmur"] == "Unknown":
            continue
        for wav in (d / "training_data").rglob(f"{r['Patient ID']}_*.wav"):
            if r["Murmur"] == "Present":
                put(wav, "murmur", f"circor_{wav.name}")
            else:
                absent.append(wav)
    for wav in random.sample(absent, min(CIRCOR_NORMAL_CAP, len(absent))):
        put(wav, "normal", f"circor_{wav.name}")


def heartbeat():
    """Label comes from the real filename prefix (the CSV fname column does not match the files)."""
    cmap = {"normal": "normal", "murmur": "murmur", "extrahls": "other_abnormal", "extrastole": "extrasystole"}
    for src in sorted((RAW / "heartbeat").glob("set_[ab]/*.wav")):
        key = src.name.split("__")[0].lower().replace("btraining_", "").replace("atraining_", "")
        if key in cmap:
            put(src, cmap[key], f"hb_{src.name}")


def valve():
    cmap = {"AS": "aortic_stenosis", "MR": "mitral_regurgitation", "MS": "mitral_stenosis",
            "MVP": "mitral_valve_prolapse", "N": "normal"}
    for k, cls in cmap.items():
        for wav in (RAW / "valve_x" / k).rglob("*.wav"):
            put(wav, cls, f"valve_{wav.name}")


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    for fn in (p2016, circor, heartbeat, valve):
        fn(); print("done", fn.__name__)
    for d in sorted(OUT.iterdir()):
        print(f"{d.name:24s}{len(list(d.glob('*.wav')))}")
