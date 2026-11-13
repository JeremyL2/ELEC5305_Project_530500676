"""Numerical checks of overlap construction and multilabel outcome accounting."""
import numpy as np
import pandas as pd
import torch

import overlap_experiments as experiment


def main():
    torch.set_num_threads(2)
    n = 32000
    wave = torch.linspace(-1, 1, n)[None]
    torch.testing.assert_close(experiment.place_fragment(wave, 1, "centre"), wave)
    assert torch.count_nonzero(experiment.place_fragment(wave, 0, "centre")) == 0
    fragments = []
    for position in ("onset", "centre", "end"):
        start, length, weights = experiment.layout(n, .5, position)
        result = experiment.place_fragment(wave, .5, position)
        assert torch.count_nonzero(result[:, :start]) == 0
        assert torch.count_nonzero(result[:, start + length:]) == 0
        fragments.append(result[:, start:start + length])
        assert weights[0, 0] == 0 and weights[0, -1] == 0
    torch.testing.assert_close(fragments[0], fragments[1], rtol=0, atol=0)
    torch.testing.assert_close(fragments[1], fragments[2], rtol=0, atol=0)
    names = ["Cargo", "Passengership", "Tanker", "Tug"]
    source_table = pd.DataFrame({"class": ["Cargo"] * 12 + ["Passengership"] * 5 + ["Tanker"] * 7 + ["Tug"] * 3})
    scenes = experiment.conditions(source_table)
    assert sum(s["family"] == "level" for s in scenes) == 1255
    assert sum(s["family"] == "duration" for s in scenes) == 1255
    assert sum(s["family"] == "placement" for s in scenes) == 108
    for pair_class in {s["pair_class"] for s in scenes}:
        subset = [s for s in scenes if s["family"] == "duration" and s["fraction"] == 1 and s["pair_class"] == pair_class]
        first_class = pair_class.split(" + ")[0]
        first_references = sum(source_table.iloc[s["a"]]["class"] == first_class for s in subset)
        assert abs(first_references - (len(subset) - first_references)) <= 1
    # A is detected alone; adding B loses A and falsely activates an absent class.
    row = {"class_a": "Cargo", "class_b": "Tug", "fraction": .5}
    vectors = {"mixture": [.2, .8, .1, .9], "solo_a": [.9, .1, .1, .1],
               "solo_b": [.1, .1, .1, .8], "dc_a": [.7, .1, .1, .1], "dc_b": [.1, .1, .1, .7]}
    for kind, vector in vectors.items():
        row.update({f"{kind}_{name}": value for name, value in zip(names, vector)})
    row.update({f"full_solo_b_{name}": value for name, value in zip(names, vectors["solo_b"])})
    frame = experiment.add_outcomes(pd.DataFrame([row]), names)
    result = experiment.summarize_group(frame, names)
    assert result["retention_all"] == .5
    assert result["all_present_recall"] == 0 and result["exact_match"] == 0
    assert result["mean_absent_false_positives"] == 1
    assert np.isclose(result["score_change_a"], -.7)
    assert np.isclose(result["dc_adjusted_change_a"], -.5)
    print("PASS: overlap masks, fixed-fragment placement, fades, no repetition, "
          "2618 planned scenarios, balanced roles, paired retention, false positives and DC contrasts")


if __name__ == "__main__":
    main()
