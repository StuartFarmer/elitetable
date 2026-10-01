"""Offline evidence, missingness and pooling regression checks."""

import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

import analyze


class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = analyze.load_evidence()

    def test_archived_statistics_and_missing_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, results, totals = analyze.analyze(self.evidence, Path(directory))
            analyze.check_reference(Path(directory))
        self.assertEqual(totals["heldout_panels"], 599)
        self.assertEqual(totals["paired_searches"], 60)
        self.assertEqual(totals["cost"], "unknown")
        self.assertEqual(totals["unknown_usage_calls"], 2)
        np.testing.assert_allclose([r["final_median"] for r in results],
                                   [500, -107.065, -101.525, 214.0443698591905,
                                    -7.648553676605225, 691.8104695275915], rtol=0, atol=1e-12)
        run = next(r for r in self.evidence["runs"] if r["run"].endswith("MountainCar-v0-8"))
        panel = analyze.trajectory(run)[5]
        self.assertIsNone(panel["heldout_mean"])
        self.assertTrue(run["heldout"][panel["policy_id"]]["failure"])
        cached = run["policies"][panel["policy_id"]]["cached_scores"]
        # Even cached successful episodes cannot repair a failed held-out panel.
        valid_cached = [v for k, v in cached.items() if int(k) >= 1000 and analyze.finite(v)]
        self.assertEqual(len(valid_cached), 99)
        row = next(r for r in rows if r["run"] == run["run"])
        self.assertTrue(row["eligible"])
        self.assertFalse(row["complete_trajectory"])
        self.assertIsNotNone(row["improvement"])

    def test_population_pooling_and_completed_filter(self):
        run = dict(generations=[dict(number=1, status="completed"), dict(number=2, status="running")],
                   population=[[1, 1, "evaluated", 0, 0], [2, 1, "evaluated", 0, 0],
                               [3, 1, "evaluated", 0, 0], [4, 1, "discarded", 999, 0],
                               [5, 2, "evaluated", 999, 0], [6, 1, "evaluated", None, 0]])
        population = analyze.population(run)
        self.assertEqual(population, [(1, 0), (1, 0), (1, 0)])
        pooled = [score for _, score in population] + [100, 200]
        actual = analyze.quartiles({1: pooled + [np.nan, np.inf], 2: [7]}, [1, 2, 3])
        np.testing.assert_allclose(actual[:, 0], [0, 0, 100])
        np.testing.assert_allclose(actual[:, 1], [7, 7, 7])
        self.assertTrue(np.isnan(actual[:, 2]).all())

    def test_policy_and_source_checks_distinguish_custom_runs(self):
        record = copy.deepcopy(self.evidence["runs"][0])
        self.assertEqual(analyze.protocol_reasons(record, self.evidence["study"]), [])
        record["frozen_source_matches"] = False
        self.assertIn("protocol:frozen_source_snapshot", analyze.protocol_reasons(record, self.evidence["study"]))
        self.assertEqual(analyze.protocol_reasons(record), [])
        record["source_snapshots"]["resume-source.zip"] = {"different.py": "different"}
        self.assertIn("protocol:source_snapshot", analyze.protocol_reasons(record))
        next(iter(record["policies"].values()))["implementation"] += "\n# changed\n"
        self.assertIn("protocol:policy_identity", analyze.protocol_reasons(record))

    def test_custom_runs_reject_mixed_worker_counts(self):
        records = [copy.deepcopy(self.evidence["runs"][0]) for _ in range(2)]
        for seed, record in enumerate(records):
            record["experiment"].update(search_seed=seed, concurrency=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(2):
                (root / str(seed)).mkdir()
                (root / str(seed) / "experiment.json").touch()
            with patch.object(analyze, "extract_run", side_effect=records):
                self.assertEqual(len(analyze.load_custom(root)["runs"]), 2)
            records[1]["experiment"]["concurrency"] = 1
            with patch.object(analyze, "extract_run", side_effect=records):
                with self.assertRaisesRegex(ValueError, "Mixed protocols"):
                    analyze.load_custom(root)


if __name__ == "__main__":
    unittest.main()
