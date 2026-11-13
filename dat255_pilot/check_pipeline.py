"""Run offline checks, including equivalence to DAT255's original transforms."""
import ast
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch

import dat255_cpu as pilot


def main():
    torch.set_num_threads(2)
    # Execute only these pure functions from upstream, never its Flask server.
    source = pilot.ROOT / "DAT255/DeepShip/rest.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"normSpec", "wavToSpecs"}]
    assert len(functions) == 2
    scope = {"torch": torch, "imgsize": pilot.IMAGE_SIZE, "rng_hf": pilot.HF_SAMPLES,
             "stft_lf": pilot.stft_lf, "stft_hf": pilot.stft_hf}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    wav = torch.randn((1, pilot.WINDOW_SAMPLES), generator=torch.Generator().manual_seed(4))
    for hf_idx in (0, (pilot.WINDOW_SAMPLES - pilot.HF_SAMPLES) // 2):
        expected = scope["wavToSpecs"](wav, hf_idx)
        actual = pilot.wavToSpecs(wav, hf_idx)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        assert actual.shape == (3, 460, 460)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "Cargo").mkdir()
        path = root / "Cargo/one.wav"
        (root / "Cargo/cargo-metafile").write_text(
            "one,70,EXAMPLE VESSEL,20171104,203623,99,1-2,\n"
            "ambiguous,70,FIRST,20171104,203623,99,1-2,\n"
            "ambiguous,70,SECOND,20171104,203623,99,1-2,\n", encoding="utf-8")
        assert pilot.vessel_metadata(path)["vessel_name"] == "EXAMPLE VESSEL"
        assert pilot.vessel_metadata(path)["vessel_id"] == ""
        assert pilot.vessel_metadata(root / "Cargo/ambiguous.wav")["vessel_name"] == ""
        assert "ambiguous" in pilot.vessel_metadata(root / "Cargo/ambiguous.wav")["vessel_identity_status"]
        samples = np.column_stack((np.linspace(-0.2, 0.2, pilot.WINDOW_SAMPLES + 1000, dtype="float32"),
                                   np.ones(pilot.WINDOW_SAMPLES + 1000, dtype="float32")))
        sf.write(path, samples, pilot.SAMPLE_RATE, subtype="FLOAT")
        selected, metadata = pilot.read_window(path)
        np.testing.assert_array_equal(selected[0].numpy(), samples[500:500 + pilot.WINDOW_SAMPLES, 0])
        assert metadata["window_start_sample_32k"] == 500
        assert not metadata["requires_repetition"]
        short = root / "Cargo/short.wav"
        sf.write(short, samples[:3200], 16000, subtype="FLOAT")
        selected, metadata = pilot.read_window(short)
        assert metadata["resampled_recording_frames"] == 6400
        assert metadata["requires_repetition"] and selected.shape == (1, pilot.WINDOW_SAMPLES)
        torch.testing.assert_close(selected, pilot.read_window(short)[0], rtol=0, atol=0)
        duplicate = root / "Cargo/duplicate.wav"
        duplicate.write_bytes(path.read_bytes())
        silent = root / "Cargo/silent.wav"
        sf.write(silent, np.zeros(32000, dtype="float32"), 32000)
        try:
            pilot.read_window(silent)
        except ValueError:
            pass
        else:
            raise AssertionError("Silence must be rejected before undefined normalization")
        try:
            pilot.wavToSpecs(torch.zeros((1, pilot.WINDOW_SAMPLES)))
        except ValueError:
            pass
        else:
            raise AssertionError("Constant spectrogram must be rejected")
        manifest = pilot.inventory(root, root)
        assert manifest.duplicate_file_sha256.sum() == 2
        assert manifest.duplicate_pcm_sha256.sum() == 2
        assert manifest.loc[manifest.recording_id == "Cargo/silent.wav", "silent"].item()

    # Correct argmax predictions can still yield a multi-positive and an empty
    # independent-label prediction at the fixed threshold.
    rows = []
    names = ["Cargo", "Passengership", "Tanker", "Tug"]
    for index, scores in enumerate(([0.8, 0.6, 0.1, 0.1], [0.1, 0.4, 0.3, 0.2])):
        rows.append({"class": names[index], "requires_repetition": bool(index),
                     **{f"score_{name}": score for name, score in zip(names, scores)}})
    with tempfile.TemporaryDirectory() as directory:
        result = pilot.evaluate(pilot.pd.DataFrame(rows), names, Path(directory), 0.5)
        assert result["groups"]["all"]["accuracy"] == 1.0
        assert result["groups"]["all"]["independent_labels"]["exact_match_accuracy"] == 0.0
        assert result["groups"]["native_length"]["n_recordings"] == 1
        assert result["groups"]["repeated_short"]["n_recordings"] == 1

    # Real export integration: CPU direct inference must match fastai's public
    # path on exactly the same preprocessed tensor, including saved after_batch.
    learn, names = pilot.load_classifier(pilot.ROOT / "DAT255/models/resnet50-93")
    learn.to_fp32()
    learn.dls.num_workers = 0
    with torch.inference_mode():
        spec = pilot.Spectrogram.create(pilot.wavToSpecs(wav))
        direct = learn.model(pilot.model_input(learn, spec)).sigmoid()[0]
        dl = learn.dls.test_dl([spec], bs=1, num_workers=0)
        via_fastai = learn.get_preds(dl=dl)[0][0].as_subclass(torch.Tensor)
    torch.testing.assert_close(direct, via_fastai, rtol=1e-5, atol=1e-6)
    assert len(names) == 4
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
               for parameter in learn.model.parameters())
    print("PASS: exact upstream spectrogram equivalence; deterministic crop/repeat/resample; "
          "channel zero; vessel metadata/ambiguity; duplicate/silence inventory; grouped metrics; "
          "fastai/direct CPU score parity")


if __name__ == "__main__":
    main()
