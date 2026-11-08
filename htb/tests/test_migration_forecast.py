import json

import numpy as np
import pandas as pd
import torch

from wildcast.migration_forecast import forecast
from wildcast.migration_model import MigrationModel


def test_forecast_cannot_consult_or_export_observations_after_issue_time(tmp_path):
    torch.manual_seed(7)
    model = MigrationModel(10, 14, 2, [1, 7, 30], history_length=25,
                           width=8, layers=1, components=2, dropout=0)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_kind": "wildcast_migration_joint", "model": model.state_dict(),
                "config": model.config, "manifest": {"species": {"bird": 1}},
                "manifest_sha256": "test-fixture"}, checkpoint)
    dates = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    f = pd.DataFrame({"timestamp": dates, "lat": np.linspace(30, 35, 30), "lon": np.linspace(10, 20, 30)})
    first = tmp_path / "first.csv"; second = tmp_path / "second.csv"
    f.to_csv(first, index=False)
    f.loc[f.index > 14, ["lat", "lon"]] = [-75, 165]
    f.to_csv(second, index=False)
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    for source, output in [(first, one), (second, two)]:
        forecast(checkpoint, source, output, "bird", issue_time="2020-01-15T00:00:00Z", samples=4)
    a, b = json.loads(one.read_text()), json.loads(two.read_text())
    assert a["sampled_latlon_paths"] == b["sampled_latlon_paths"]
    assert a["observed_history"] == b["observed_history"]
    assert len(a["observed_history"]) == 15
    assert "observed_future" not in a and "target_latlon" not in a
