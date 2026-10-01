"""Exercise saved evidence and held-out separation without API calls or Docker."""

import contextlib
import csv
import io
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run
from rsikit import Executor


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_worker_fails_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "offline-test"}),
                patch.object(run, "command_output", return_value=None),
                patch.object(run, "LoggedOpenRouter") as provider,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                await run.main(["--output", str(output)])
            provider.assert_not_called()
            self.assertFalse(output.exists())

    async def test_batch_dry_run_resolves_equals_flags_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                await run.main(["--all", "--dry-run", f"--output={output}", "--concurrency=3"])
            commands = [shlex.split(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(len(commands), 60)
            for command in commands:
                self.assertEqual(command.count("--output"), 1)
                self.assertFalse(any(word.startswith("--output=") for word in command))
                self.assertEqual(command[command.index("--concurrency") + 1], "3")
            self.assertFalse(output.exists())

    async def test_paper_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                await run.main(["--dry-run", "--output", str(output)])
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan["config"]["generations"], 10)
            self.assertEqual(plan["config"]["population_size"], 50)
            self.assertIsNone(plan["config"]["target_score"])
            self.assertEqual(plan["episode_timeout"], 45)
            self.assertEqual(plan["concurrency"], 8)
            self.assertEqual(plan["heldout_seeds"], list(range(1000, 1100)))
            self.assertFalse(output.exists())

    async def test_search_selection_is_not_changed_by_heldout_or_resume(self):
        class Provider:
            next_value = 0

            async def acall(self, context, **kwargs):
                value = self.next_value
                self.next_value += 1
                if "Return exact search/replacement edits" in context:
                    previous = context.split("return ")[-1].split()[0]
                    response = dict(name=str(value), description="edit", edits=[
                        dict(search=f"return {previous}", replacement=f"return {value}")])
                else:
                    response = dict(name=str(value), description="fresh", implementation=
                        "from rsikit import Policy\nclass Solution(Policy):\n"
                        f"    async def act(self, observation):\n        return {value}\n")
                return json.dumps(response), []

        class Sandbox:
            async def start(self, workers):
                pass

            async def close(self):
                pass

            async def evaluate(self, source, environment, seed, call_timeout):
                if "action_space.sample" in source:
                    return -2.0, {}
                value = int(source.split("return ")[-1].strip())
                return float(value if seed < 1000 else 10 - value), {}

        provider = Provider()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "offline-test"}),
                patch.object(run, "LoggedOpenRouter", return_value=provider),
                patch.object(run, "Executor", side_effect=lambda **kw: Executor(sandbox=Sandbox())),
                patch.object(run, "command_output", return_value="offline-test-image"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                await run.main(["--output", str(output), "--population", "2", "--elites", "2",
                                "--generations", "2", "--seeds", "0", "--heldout-seeds", "1000", "1001"])
                with (output / "curves.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([float(r["search_mean"]) for r in rows], [1, 3])
                self.assertEqual([float(r["heldout_mean"]) for r in rows], [9, 7])
                self.assertEqual(provider.next_value, 4)
                await run.main(["--resume", str(output)])
                self.assertEqual(provider.next_value, 4)
                self.assertEqual(json.loads((output / "status.json").read_text())["status"], "completed")
                self.assertEqual(len(json.loads((output / "attempts.json").read_text())), 2)
                with patch.object(run, "command_output", return_value="different-image"), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        await run.main(["--resume", str(output)])
                self.assertEqual(len(json.loads((output / "attempts.json").read_text())), 2)


if __name__ == "__main__":
    unittest.main()
