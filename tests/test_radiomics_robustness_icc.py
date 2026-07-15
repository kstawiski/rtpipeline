import pandas as pd

from rtpipeline.radiomics_robustness import ICCConfig, compute_icc_pingouin


def test_compute_icc_accepts_current_pingouin_ci_column_name():
    frame = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "rater": ["original", "perturbed"] * 4,
            "value": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1],
        }
    )
    result = compute_icc_pingouin(frame, ICCConfig(icc_type="ICC3", ci=True))
    assert result["icc"] > 0.9
    assert result["icc_ci95_low"] <= result["icc_ci95_high"]
