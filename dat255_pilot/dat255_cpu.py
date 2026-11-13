"""DAT255 CPU inference and a recording-level single-vessel robustness pilot.

Upstream: voibit/DAT255, DeepShip/rest.py (MIT). No training or model changes.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
from pathlib import Path, PurePath
import platform
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from fastai.vision.all import (
    Tensor, TensorImageBase, RandTransform, get_files, load_learner,
    show_images, store_attr,
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, hamming_loss, multilabel_confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 32000
IMAGE_SIZE = 460
NFFT_LF, HOP_LF = 32768, 32768 // 5
NFFT_HF, HOP_HF = 1024, 768
WINDOW_SAMPLES = HOP_LF * (IMAGE_SIZE - 1)
HF_SAMPLES = HOP_HF * IMAGE_SIZE + NFFT_HF
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
LIMITATION = (
    "Robustness experiment on the available recordings. DAT255 was trained on "
    "DeepShip, so these recordings may already have appeared in its training "
    "data. Training overlap has not been ruled out; a new split of these files "
    "cannot establish that the pretrained model has never encountered them."
)
stft_lf = torchaudio.transforms.Spectrogram(NFFT_LF, hop_length=HOP_LF, power=2).cpu().float()
stft_hf = torchaudio.transforms.Spectrogram(NFFT_HF, hop_length=HOP_HF, power=2).cpu().float()


def normSpec(spec):
    """Original log/mean/std/clamp normalization; reject undefined inputs."""
    ret = torch.log10(spec + 1e-20)
    std = torch.std(ret)
    if not torch.isfinite(std) or std == 0:
        raise ValueError("Constant/non-finite spectrogram: DAT255 normalization is undefined")
    return torch.clamp((ret - torch.mean(ret)) / (std * 4) + 0.5, 0, 1)


def wavToSpecs(wavs: torch.Tensor, hf_idx=0):
    """Exact upstream three channels, FFT defaults and frequency crops on CPU."""
    wavs = wavs.cpu().float()
    if wavs.ndim != 2 or wavs.shape[1] != WINDOW_SAMPLES:
        raise ValueError(f"Expected (channels, {WINDOW_SAMPLES}) audio")
    lf = stft_lf(wavs[:1])[0]  # Upstream uses channel zero, never a stereo mean.
    lf0 = normSpec(lf[:IMAGE_SIZE, :IMAGE_SIZE])
    lf1 = normSpec(lf[IMAGE_SIZE:IMAGE_SIZE * 2, :IMAGE_SIZE])
    hf = stft_hf(wavs[:1, hf_idx:hf_idx + HF_SAMPLES])[0]
    hf = normSpec(hf[12:IMAGE_SIZE + 12, :IMAGE_SIZE])
    result = torch.stack((lf0, lf1, hf), 0)
    if result.shape != (3, IMAGE_SIZE, IMAGE_SIZE) or not torch.isfinite(result).all():
        raise ValueError("Invalid spectrogram")
    return result


def read_window(path, crop="centre"):
    """One deterministic window; read only its samples at the native 32 kHz."""
    info = sf.info(path)
    if not info.frames:
        raise ValueError("Empty recording")
    if info.samplerate == SAMPLE_RATE and info.frames >= WINDOW_SAMPLES:
        start = (info.frames - WINDOW_SAMPLES) // 2 if crop == "centre" else 0
        with sf.SoundFile(path) as audio:
            audio.seek(start)
            samples = audio.read(WINDOW_SAMPLES, dtype="float32", always_2d=True)
        wav = torch.from_numpy(samples[:, 0].copy())[None]
        frames = info.frames
    else:
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(samples[:, 0].copy())[None]
        if rate != SAMPLE_RATE:
            # Resample before choosing a crop, as in predTensor; no partial-filter edges.
            wav = torchaudio.functional.resample(wav, rate, SAMPLE_RATE)
        frames = wav.shape[1]
        start = (frames - WINDOW_SAMPLES) // 2 if crop == "centre" else 0
        if frames < WINDOW_SAMPLES:
            repeats = math.ceil(WINDOW_SAMPLES / frames)
            wav = wav.repeat(1, repeats)
            start = (wav.shape[1] - WINDOW_SAMPLES) // 2 if crop == "centre" else 0
        wav = wav[:, start:start + WINDOW_SAMPLES]
    if not torch.isfinite(wav).all() or torch.count_nonzero(wav) == 0:
        raise ValueError("Silent or non-finite selected audio")
    return wav, {
        "window_start_sample_32k": start,
        "window_seconds": WINDOW_SECONDS,
        "resampled_recording_frames": frames,
        "requires_repetition": frames < WINDOW_SAMPLES,
        "repeat_count": math.ceil(WINDOW_SAMPLES / frames),
        "crop": crop,
    }


# Keep these original names available for the fastai pickle, including notebook use.
def readWav(p: Path, rand=True):
    if rand:
        raise ValueError("Random training crops are disabled in this evaluation pipeline")
    return read_window(p)[0]


class Spectrogram(TensorImageBase):
    @classmethod
    def create(cls, o: Tensor):
        return cls(o)

    def show(self, figsize=None, ctx=None, **kwargs):
        return show_images(self, nrows=1, ncols=self.shape[0])


class SpectrogramTransform(RandTransform):
    split_idx, order = None, 0

    def __init__(self):
        store_attr()

    def before_call(self, b, split_idx):
        self.idx = split_idx

    def encodes(self, p: Path):
        return Spectrogram.create(wavToSpecs(readWav(p, False), (WINDOW_SAMPLES - HF_SAMPLES) // 2))


def get_wavs(p: Path):
    return get_files(p, ".wav")


def label_func(p: Path):
    if PurePath(p).parent.name == "AmbientSE":
        return []
    return [PurePath(p).parent.parent.name]


def load_classifier(checkpoint):
    for name in ("Spectrogram", "SpectrogramTransform", "label_func", "get_wavs"):
        setattr(sys.modules["__main__"], name, globals()[name])
    # The export was made on Linux. Scope pathlib's Windows compatibility shim
    # to loading; do not change the pickle or globally retain the substitution.
    original = pathlib.PosixPath
    try:
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        learn = load_learner(checkpoint, cpu=True)
    finally:
        pathlib.PosixPath = original
    learn.model = learn.model.cpu().float().eval()
    class_names = [str(name) for name in learn.dls.vocab]
    if len(class_names) != 4:
        raise ValueError(f"Expected four vessel outputs, found {class_names}")
    return learn, class_names


def model_input(learn, spectrogram):
    # cnn_learner can add normalization even if DataBlock has no batch_tfms.
    # Execute the EXPORTED validation batch pipeline, rather than guessing stats.
    batch = (Spectrogram.create(spectrogram[None].cpu().float()),)
    pipeline = learn.dls.valid.after_batch
    original_split = pipeline.split_idx
    try:
        pipeline.split_idx = 1
        return pipeline(batch)[0].as_subclass(torch.Tensor).cpu().float()
    finally:
        pipeline.split_idx = original_split


def predict_window(learn, path, crop="centre", hf_crop="centre"):
    wav, metadata = read_window(path, crop)
    hf_idx = (WINDOW_SAMPLES - HF_SAMPLES) // 2 if hf_crop == "centre" else 0
    with torch.inference_mode():
        spec = wavToSpecs(wav, hf_idx)
        logits = learn.model(model_input(learn, spec))
        probabilities = logits.sigmoid()[0].cpu().numpy()
    if probabilities.shape != (4,) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid model output")
    metadata.update(hf_start_sample_32k=hf_idx, hf_crop=hf_crop)
    return probabilities, metadata


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vessel_metadata(path):
    """Join the headerless DeepShip metafile on class directory + recording ID.

    Names may be truncated and are not verified MMSI/IMO identifiers. Retain
    ambiguous matches instead of guessing a vessel from duplicate metadata IDs.
    """
    matches = []
    for metadata in sorted(path.parent.glob("*-metafile")):
        with metadata.open(encoding="utf-8-sig", newline="") as source:
            for fields in csv.reader(source):
                if len(fields) >= 5 and fields[0].strip() == path.stem:
                    matches.append({"vessel_name": fields[2].strip(),
                                    "metadata_date": fields[3].strip(),
                                    "metadata_time": fields[4].strip(),
                                    "metadata_source": metadata.name})
    result = {"vessel_id": "", "vessel_name": "", "metadata_date": "",
              "metadata_time": "", "metadata_source": "",
              "vessel_identity_status": "no metadata match", "metadata_candidates": ""}
    if len(matches) == 1:
        result.update(matches[0])
        result["vessel_identity_status"] = "name only; no verified MMSI/IMO"
    elif matches:
        result["vessel_identity_status"] = "ambiguous metadata recording ID"
        result["metadata_candidates"] = json.dumps(matches)
    return result


def inventory(data, output):
    rows = []
    paths = sorted(p for p in data.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not paths:
        raise ValueError(f"No WAV recordings in {data}")
    for path in paths:
        recording_id = path.relative_to(data).as_posix()
        row = {"recording_id": recording_id, "class": path.relative_to(data).parts[0],
               **vessel_metadata(path), "error": ""}
        try:
            info = sf.info(path)
            digest = hashlib.sha256(f"{info.samplerate}:{info.channels}:".encode())
            energy, peak, zeros, count, quiet, blocks = 0., 0., 0, 0, 0, 0
            # One second at a time: bounded memory, inspect the entire recording.
            with sf.SoundFile(path) as audio:
                for samples in audio.blocks(blocksize=info.samplerate, dtype="float32", always_2d=True):
                    if not np.isfinite(samples).all():
                        raise ValueError("Non-finite audio")
                    digest.update(samples.astype("<f4", copy=False).tobytes())
                    first = samples[:, 0].astype("float64")
                    power = np.dot(first, first)
                    energy += power
                    peak = max(peak, float(np.max(np.abs(first))))
                    zeros += int(np.count_nonzero(first == 0))
                    count += len(first)
                    quiet += int(math.sqrt(power / len(first)) <= 0.001)  # -60 dBFS
                    blocks += 1
            if not count:
                raise ValueError("Empty recording")
            row.update(duration_seconds=info.duration, sample_rate=info.samplerate,
                       frames=info.frames, channels=info.channels, subtype=info.subtype,
                       file_sha256=sha256(path), pcm_sha256=digest.hexdigest(),
                       rms=math.sqrt(energy / count), peak=peak,
                       zero_sample_fraction=zeros / count,
                       quiet_1s_block_fraction=quiet / blocks, silent=peak == 0,
                       requires_repetition=math.ceil(info.frames * SAMPLE_RATE / info.samplerate) < WINDOW_SAMPLES)
        except (RuntimeError, ValueError) as error:
            row["error"] = str(error)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for key in ("file_sha256", "pcm_sha256"):
        if key not in frame:
            frame[key] = None
        frame[f"duplicate_{key}"] = frame[key].notna() & frame.duplicated(key, keep=False)
    frame.to_csv(output / "inventory.csv", index=False)
    return frame


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(predictions, names, output, threshold, manifest=None):
    metrics = {"limitation": LIMITATION, "class_names": names, "threshold": threshold, "groups": {}}
    groups = {"all": predictions,
              "native_length": predictions[~predictions.requires_repetition],
              "repeated_short": predictions[predictions.requires_repetition]}
    for group, frame in groups.items():
        if frame.empty:
            metrics["groups"][group] = {"n_recordings": 0}
            continue
        true = np.array([names.index(label) for label in frame["class"]])
        scores = frame[[f"score_{name}" for name in names]].to_numpy()
        predicted = scores.argmax(axis=1)
        binary_true = np.eye(len(names), dtype=int)[true]
        binary_pred = (scores >= threshold).astype(int)
        report = classification_report(true, predicted, labels=list(range(len(names))),
                                       target_names=names, zero_division=0, output_dict=True)
        independent = classification_report(binary_true, binary_pred, target_names=names,
                                            zero_division=0, output_dict=True)
        cm = confusion_matrix(true, predicted, labels=list(range(len(names))))
        metrics["groups"][group] = {
            "n_recordings": len(frame), "accuracy": accuracy_score(true, predicted),
            "macro_f1": report["macro avg"]["f1-score"], "argmax_report": report,
            "independent_labels": {"report": independent,
                "exact_match_accuracy": accuracy_score(binary_true, binary_pred),
                "hamming_loss": hamming_loss(binary_true, binary_pred),
                "confusion_matrices_tn_fp_fn_tp": multilabel_confusion_matrix(binary_true, binary_pred).tolist()},
        }
        pd.DataFrame(report).T.to_csv(output / f"classification_{group}.csv")
        pd.DataFrame(independent).T.to_csv(output / f"independent_labels_{group}.csv")
        pd.DataFrame(cm, index=names, columns=names).to_csv(output / f"confusion_{group}.csv")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Available-recording robustness baseline: {group}\nn={len(frame)}; training overlap unresolved")
        fig.tight_layout()
        fig.savefig(output / f"confusion_{group}.png", dpi=160)
        plt.close(fig)
    if manifest is not None:
        metrics["inventory"] = {
            "n_source_recordings": len(manifest),
            "class_counts": {str(k): int(v) for k, v in manifest.groupby("class").size().items()},
            "duplicate_file_flags": int(manifest.duplicate_file_sha256.sum()),
            "duplicate_pcm_flags": int(manifest.duplicate_pcm_sha256.sum()),
            "exact_silent_recordings": int(manifest.silent.fillna(False).sum()),
            "recordings_with_quiet_blocks": int((manifest.quiet_1s_block_fraction > 0).sum()),
            "vessel_metadata_status_counts": {str(k): int(v) for k, v in manifest.vessel_identity_status.value_counts().items()},
        }
    save_json(output / "metrics.json", metrics)
    lines = ["# DAT255 single-vessel baseline", "", LIMITATION, "",
             f"Class order: {', '.join(names)}. Independent-label threshold: {threshold}.", "",
             "| Group | Recordings | Argmax accuracy | Macro-F1 | Independent exact match |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for group, result in metrics["groups"].items():
        if result["n_recordings"]:
            lines.append(f"| {group} | {result['n_recordings']} | {result['accuracy']:.3f} | "
                         f"{result['macro_f1']:.3f} | {result['independent_labels']['exact_match_accuracy']:.3f} |")
    if manifest is not None:
        info = metrics["inventory"]
        lines += ["", f"Inventory: {len(manifest)} source recordings; "
                  f"{info['duplicate_file_flags']} exact-file duplicate flags; "
                  f"{info['duplicate_pcm_flags']} decoded-audio duplicate flags; "
                  f"{info['exact_silent_recordings']} silent recordings; "
                  f"{info['recordings_with_quiet_blocks']} recordings with quiet one-second blocks.",
                  "", "Vessel metadata: " + "; ".join(f"{count} {status}" for status, count in
                                                         info["vessel_metadata_status_counts"].items()) + "."]
    lines += ["", "## Per-class single-vessel baseline", "",
              "| Class | Source recordings | Precision | Recall | F1 | Recall at threshold |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        a = metrics["groups"]["all"]["argmax_report"][name]
        b = metrics["groups"]["all"]["independent_labels"]["report"][name]
        lines.append(f"| {name} | {a['support']:.0f} | {a['precision']:.3f} | {a['recall']:.3f} | "
                     f"{a['f1-score']:.3f} | {b['recall']:.3f} |")
    tug_count = int((predictions["class"].str.casefold() == "tug").sum())
    lines += ["", "Keep this baseline beside any later mixture results. Poor detection of a class alone "
              f"cannot be attributed to overlap. This run contains {tug_count} tug source recordings; "
              "extra windows do not create extra independent vessels. The inventory preserves metafile vessel names, "
              "which can be truncated and are not verified MMSI/IMO identifiers.",
              "", "Subgroup macro-F1 averages over all four classes, including classes with zero support. "
              "Consult per-class support before comparing subgroup F1.",
              "", "## Independent labels at the fixed threshold", "",
              "| Class | Precision | Recall | F1 | Positive-label support |",
              "| --- | ---: | ---: | ---: | ---: |"]
    independent = metrics["groups"]["all"]["independent_labels"]
    for name in names:
        item = independent["report"][name]
        lines.append(f"| {name} | {item['precision']:.3f} | {item['recall']:.3f} | "
                     f"{item['f1-score']:.3f} | {item['support']:.0f} |")
    lines += ["", f"Independent-label macro-F1: {independent['report']['macro avg']['f1-score']:.3f}; "
              f"Hamming loss: {independent['hamming_loss']:.4f}.",
              "", "One deterministic window per recording. No threshold tuning, training, new split, or mixture experiment.",
              "Duplicate and silence flags are in inventory.csv. Failed recordings are listed in failures.csv."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def provenance(args, learn, names):
    packages = ["torch", "torchvision", "torchaudio", "fastai", "fastcore", "numpy",
                "soundfile", "pandas", "scikit-learn", "matplotlib"]
    normalizers = []
    for transform in learn.dls.after_batch.fs:
        if type(transform).__name__ == "Normalize":
            normalizers.append({"mean": transform.mean.cpu().tolist(), "std": transform.std.cpu().tolist()})
    return {"limitation": LIMITATION, "class_names": names, "device": "cpu", "dtype": "float32",
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
            "data_root": str(args.data.resolve()), "script_sha256": sha256(Path(__file__)),
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in packages},
            "batch_size": 1, "threads": args.threads, "threshold": args.threshold,
            "crop": args.crop, "hf_crop": args.hf_crop,
            "window_samples_32k": WINDOW_SAMPLES, "window_seconds": WINDOW_SECONDS,
            "hf_samples_32k": HF_SAMPLES, "sample_rate": SAMPLE_RATE,
            "fft_lf": NFFT_LF, "hop_lf": HOP_LF, "fft_hf": NFFT_HF, "hop_hf": HOP_HF,
            "after_batch": repr(learn.dls.after_batch), "saved_normalizers": normalizers,
            "upstream_repository": "https://github.com/voibit/DAT255",
            "upstream_local_commit": (ROOT / "DAT255/.git/refs/heads/main").read_text().strip()
                if (ROOT / "DAT255/.git/refs/heads/main").exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "predict", "baseline"])
    parser.add_argument("--data", type=Path, default=ROOT / "DeepShip")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "DAT255/models/resnet50-93")
    parser.add_argument("--output", type=Path, help="Defaults to results (or results/smoke for predict)")
    parser.add_argument("--file", type=Path, help="One audio recording for predict")
    parser.add_argument("--crop", choices=["centre", "start"], default="centre")
    parser.add_argument("--hf-crop", choices=["centre", "start"], default="centre")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.threads < 1:
        parser.error("threshold must be in [0,1] and threads must be positive")
    if args.command == "predict" and args.file is None:
        parser.error("predict requires --file")
    if args.output is None:
        args.output = Path(__file__).parent / "results"
        if args.command == "predict":
            args.output = args.output / "smoke"
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if args.command == "inventory":
        frame = inventory(args.data, args.output)
        print(frame.groupby("class").size().to_string())
        return
    learn, names = load_classifier(args.checkpoint)
    save_json(args.output / "run.json", provenance(args, learn, names))
    if args.command == "predict":
        scores, metadata = predict_window(learn, args.file, args.crop, args.hf_crop)
        result = {"file": str(args.file), "class_names": names, "scores": dict(zip(names, scores.tolist())),
                  "argmax": names[int(scores.argmax())], "window": metadata, "limitation": LIMITATION}
        save_json(args.output / "prediction.json", result)
        print(json.dumps(result, indent=2))
        return
    manifest = inventory(args.data, args.output)
    predictions, failures = [], []
    for i, row in enumerate(manifest.to_dict("records"), 1):
        recording_id = row["recording_id"]
        began = time.perf_counter()
        try:
            if row["error"]:
                raise ValueError(row["error"])
            label = next((name for name in names if name.casefold() == row["class"].casefold()), None)
            if label is None:
                raise ValueError(f"Unknown class: {row['class']}")
            scores, metadata = predict_window(learn, args.data / recording_id, args.crop, args.hf_crop)
            predictions.append({"recording_id": recording_id, "window_id": recording_id + "#0",
                                "class": label, "vessel_id": row["vessel_id"],
                                "vessel_name": row["vessel_name"],
                                "vessel_identity_status": row["vessel_identity_status"],
                                "metadata_date": row["metadata_date"],
                                "metadata_time": row["metadata_time"],
                                "metadata_source": row["metadata_source"],
                                "metadata_candidates": row["metadata_candidates"], **metadata,
                                "file_sha256": row["file_sha256"],
                                "pcm_sha256": row["pcm_sha256"],
                                **{f"score_{n}": float(s) for n, s in zip(names, scores)},
                                "argmax": names[int(scores.argmax())],
                                **{f"positive_{n}": bool(s >= args.threshold) for n, s in zip(names, scores)},
                                "elapsed_seconds": time.perf_counter() - began})
            print(f"[{i}/{len(manifest)}] {recording_id}: {predictions[-1]['argmax']}", flush=True)
        except (ValueError, RuntimeError) as error:
            failures.append({"recording_id": recording_id, "error": str(error)})
            print(f"[{i}/{len(manifest)}] FAILED {recording_id}: {error}", flush=True)
        # Save each completed row, so a later interrupted run still leaves raw scores.
        pd.DataFrame(predictions).to_csv(args.output / "predictions.csv", index=False)
        pd.DataFrame(failures, columns=["recording_id", "error"]).to_csv(args.output / "failures.csv", index=False)
    if not predictions:
        raise RuntimeError("No predictions; see failures.csv")
    metrics = evaluate(pd.DataFrame(predictions), names, args.output, args.threshold, manifest)
    print(json.dumps({"n_success": len(predictions), "n_failed": len(failures),
                      "accuracy": metrics["groups"]["all"]["accuracy"],
                      "macro_f1": metrics["groups"]["all"]["macro_f1"]}, indent=2))
    if failures:
        raise SystemExit("Incomplete baseline: failed recordings are recorded in failures.csv")


if __name__ == "__main__":
    main()
