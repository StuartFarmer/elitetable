"""Elite retention, reproduction, repair, and overlapping isolated evaluations."""

import asyncio
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, patch

from slick import prompts
from slick.providers import ProviderError

from elitetable import Config, EliteTable, Generation, Measurement, Organism
from tests.providers import ScriptedProvider


def program(value):
    return json.dumps(
        dict(
            name=f"Policy {value}",
            description=f"Idea {value}",
            implementation="from rsikit import Policy\nclass Solution(Policy):\n"
            f"    async def act(self, observation):\n        return {value}\n",
        )
    )


class EliteTableTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch.object(
            prompts, "TEMPLATE_ROOT", Path("/nonexistent/unrelated-application-prompts")
        )
        root.start()
        self.addCleanup(root.stop)

    async def test_model_deadline_retries_preserve_repair_budget_and_evidence(self):
        provider = ScriptedProvider([program(0), program(1)])
        original = provider.acall
        repair_calls = 0

        async def respond(context, **kwargs):
            nonlocal repair_calls
            if provider.calls:
                repair_calls += 1
                if repair_calls <= 2:
                    await asyncio.Event().wait()  # Exercise the actual per-attempt deadline.
            return await original(context, **kwargs)

        async def evaluate(policies):
            return {p.id: Measurement({}, "invalid action") if p.name == "Policy 0"
                    else Measurement({0: 12}) for p in policies}

        agent = EliteTable("Score", provider, evaluate,
                            config=Config(population_size=1, generations=1,
                                          max_repairs=1, generation_timeout=0.02))
        with patch.object(provider, "acall", side_effect=respond), \
                patch("elitetable.asyncio.sleep", new_callable=AsyncMock):
            await agent.run()
        row = agent.organisms[0]
        self.assertEqual(row.score, 12)
        self.assertEqual(row.repairs, 1)
        self.assertEqual([c["operation"] for c in row.calls], ["invent", "repair", "repair", "repair"])
        self.assertTrue(all("TimeoutError" in c["error"] for c in row.calls[1:3]))
        self.assertNotIn("error", row.calls[-1])

    async def test_model_retries_are_bounded_and_do_not_retry_cancellation_or_bad_requests(self):
        for failure, attempts in ((TimeoutError(), 3), (asyncio.CancelledError(), 1),
                                  (ProviderError("invalid credentials"), 1)):
            with self.subTest(failure=type(failure).__name__):
                agent = EliteTable("Score", ScriptedProvider([]), None)
                row = Organism(id=1, generation=1, kind="new")

                async def request(*args, **kwargs):
                    raise failure

                with patch("elitetable.asyncio.sleep", new_callable=AsyncMock):
                    with self.assertRaises(type(failure)):
                        await agent._call(row, request)
                self.assertEqual(len(row.calls), attempts)
                self.assertTrue(all("error" in call for call in row.calls))

    async def test_fixed_population_snapshot_three_operators_and_top_elites(self):
        provider = ScriptedProvider([])
        next_value = 0

        async def respond(context, **kwargs):
            nonlocal next_value
            value = next_value
            next_value += 1
            if "Return exact search/replacement edits" in context:
                parent_value = context.split("return ")[-1].split()[0]
                return json.dumps(
                    dict(
                        name=f"Policy {value}",
                        description="Edited",
                        edits=[
                            dict(search=f"return {parent_value}", replacement=f"return {value}")
                        ],
                    )
                ), []
            return program(value), []

        async def evaluate(policies):
            return {p.id: Measurement({0: int(p.name.split()[-1])}) for p in policies}

        agent = EliteTable(
            "Score policies",
            provider,
            evaluate,
            config=Config(elite_size=2, population_size=5, generations=2),
        )
        with patch.object(provider, "acall", side_effect=respond):
            await agent.run()
        self.assertEqual([row.score for row in agent.elites], [9, 8])
        self.assertEqual(len(agent.organisms), 10)
        founders, children = agent.organisms[:5], agent.organisms[5:]
        self.assertTrue(all(row.kind == "new" for row in founders))
        self.assertEqual(Counter(row.kind for row in children), {"new": 1, "edit": 2, "remix": 2})
        old_elites = set(agent.generations[0].elite_ids)
        for row in children:
            self.assertTrue(set(row.parent_ids) <= old_elites)
            self.assertEqual(len(row.parent_ids), {"new": 0, "edit": 1, "remix": 2}[row.kind])
        self.assertEqual(agent.reason, "completed")

    async def test_ties_and_worse_candidates_keep_incumbents(self):
        provider = ScriptedProvider([program(i) for i in range(6)])
        scores = iter([10, 9, 8, 9, -10, 8])

        async def evaluate(policies):
            return {p.id: Measurement({0: next(scores)}) for p in policies}

        agent = EliteTable(
            "Score",
            provider,
            evaluate,
            config=Config(
                elite_size=2,
                population_size=3,
                generations=2,
                new_fraction=1,
                remix_fraction=0,
                generation_concurrency=1,
            ),
        )
        await agent.run()
        self.assertEqual([row.id for row in agent.elites], [1, 2])
        self.assertEqual(agent.generations[0].elite_ids, agent.generations[1].elite_ids)

    async def test_repair_budget_covers_invalid_output_and_execution(self):
        provider = ScriptedProvider(["bad JSON", program(0), program(1)])
        evaluated = []

        async def evaluate(policies):
            evaluated.extend(p.name for p in policies)
            return {
                p.id: Measurement({}, "invalid action")
                if p.name == "Policy 0"
                else Measurement({0: 12})
                for p in policies
            }

        agent = EliteTable(
            "Score",
            provider,
            evaluate,
            config=Config(
                population_size=1,
                generations=1,
                max_repairs=2,
            ),
        )
        await agent.run()
        self.assertEqual(evaluated, ["Policy 0", "Policy 1"])
        self.assertEqual(agent.elites[0].score, 12)
        self.assertEqual(agent.organisms[0].repairs, 2)
        self.assertEqual(len(agent.organisms[0].revisions), 2)
        self.assertIn("invalid action", provider.calls[-1])

    async def test_generation_overlaps_evaluation_and_cancels_on_failure(self):
        provider = ScriptedProvider([program(0), program(1)])
        evaluating, generating, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = provider.acall

        async def respond(context, **kwargs):
            result = await original(context, **kwargs)
            if result[0] == program(1):
                await evaluating.wait()
                generating.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return result

        async def evaluate(policies):
            evaluating.set()
            await generating.wait()
            raise RuntimeError("worker offline")

        agent = EliteTable(
            "Score", provider, evaluate, config=Config(population_size=2, generations=1)
        )
        with patch.object(provider, "acall", side_effect=respond):
            with self.assertRaisesRegex(RuntimeError, "worker offline"):
                await asyncio.wait_for(agent.run(), 2)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(agent.reason, "error")
        self.assertEqual(agent.elites, [])

    async def test_invalid_scores_never_enter_leaderboard(self):
        for scores in ({}, {0: float("nan")}, {0: float("inf")}):
            with self.subTest(scores=scores):

                async def evaluate(policies):
                    return {p.id: Measurement(scores) for p in policies}

                agent = EliteTable(
                    "Score",
                    ScriptedProvider([program(0)]),
                    evaluate,
                    config=Config(population_size=1, generations=1),
                )
                with self.assertRaisesRegex(ValueError, "finite per-seed"):
                    await agent.run()
                self.assertEqual(agent.elites, [])
                self.assertEqual(agent.reason, "error")

    async def test_arriving_candidate_evaluates_while_previous_candidate_is_running(self):
        provider = ScriptedProvider([program(0), program(1)])
        first_started, second_started = asyncio.Event(), asyncio.Event()
        original = provider.acall

        async def respond(context, **kwargs):
            result = await original(context, **kwargs)
            if result[0] == program(1):
                await first_started.wait()
            return result

        async def evaluate(policies):
            if policies[0].name == "Policy 0":
                first_started.set()
                await asyncio.wait_for(second_started.wait(), 1)
            else:
                second_started.set()
            return {p.id: Measurement({0: 7}) for p in policies}

        agent = EliteTable(
            "Score", provider, evaluate, config=Config(population_size=2, generations=1)
        )
        with patch.object(provider, "acall", side_effect=respond):
            await agent.run()
        self.assertTrue(all(row.status == "evaluated" for row in agent.organisms))

    async def test_duplicate_and_failed_populations_preserve_existing_elites(self):
        async def evaluate(policies):
            return {
                p.id: Measurement({0: 7})
                if p.name == "Policy 0"
                else Measurement({}, "broken policy")
                for p in policies
            }

        agent = EliteTable(
            "Score",
            ScriptedProvider([program(0), program(0), program(1)]),
            evaluate,
            config=Config(
                population_size=1, generations=3, new_fraction=1, remix_fraction=0, max_repairs=0
            ),
        )
        await agent.run()
        self.assertEqual([row.id for row in agent.elites], [1])
        self.assertEqual(
            [row.status for row in agent.organisms], ["evaluated", "discarded", "discarded"]
        )
        self.assertEqual([g.elite_ids for g in agent.generations], [[1], [1], [1]])
        self.assertEqual(agent.reason, "completed")

    async def test_provider_failure_cancels_inflight_evaluation(self):
        provider = ScriptedProvider([program(0), program(1)])
        evaluating, cancelled = asyncio.Event(), asyncio.Event()
        original = provider.acall

        async def respond(context, **kwargs):
            result = await original(context, **kwargs)
            if result[0] == program(1):
                await evaluating.wait()
                raise ProviderError("offline")
            return result

        async def evaluate(policies):
            evaluating.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        agent = EliteTable(
            "Score", provider, evaluate, config=Config(population_size=2, generations=1)
        )
        with patch.object(provider, "acall", side_effect=respond):
            with self.assertRaisesRegex(ProviderError, "offline"):
                await asyncio.wait_for(agent.run(), 0.5)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(agent.reason, "error")

    async def test_five_repairs_allow_six_total_proposals(self):
        provider = ScriptedProvider(["bad JSON"] * 6)
        agent = EliteTable(
            "Score", provider, None,
            config=Config(population_size=1, generations=1, max_repairs=5),
        )
        await agent.run()
        self.assertEqual(agent.organisms[0].repairs, 5)
        self.assertEqual(agent.organisms[0].status, "discarded")
        self.assertEqual(len(provider.calls), 6)

    async def test_tie_priority_is_edit_remix_incumbent_new_then_id(self):
        agent = EliteTable("Score", None, None, config=Config(elite_size=6))
        agent.elites = [Organism(id=1, generation=1, kind="new", score=10)]
        rows = [
            Organism(id=i, generation=2, kind=kind, score=10)
            for i, kind in [(2, "new"), (3, "remix"), (4, "edit"), (5, "edit")]
        ]
        self.assertEqual([r.id for r in agent._rank(rows)], [4, 5, 3, 1, 2])

    async def test_restore_reuses_generated_candidate_without_provider_call(self):
        async def interrupted(policies):
            raise RuntimeError("worker offline")

        async def evaluate(policies):
            return {p.id: Measurement({0: 5, 1: 7}) for p in policies}

        config = Config(population_size=1, generations=1)
        original = EliteTable("Score", ScriptedProvider([program(0)]), interrupted, config=config)
        with self.assertRaisesRegex(RuntimeError, "worker offline"):
            await original.run()
        provider = ScriptedProvider([])
        restored = EliteTable("Score", provider, evaluate, config=config)
        restored.restore(original.organisms, original.generations)
        await restored.run()
        self.assertEqual(restored.reason, "completed")
        self.assertEqual(restored.elites[0].score, 6)
        self.assertEqual(provider.calls, [])
        self.assertEqual(restored.best.id, original.organisms[0].policy_id)

    async def test_restore_replays_parent_rng_and_rejects_changed_seed(self):
        async def evaluate(policies):
            return {p.id: Measurement({0: int(p.name.split()[-1])}) for p in policies}

        config = Config(elite_size=2, population_size=5, generations=1)
        original = EliteTable(
            "Score", ScriptedProvider([program(i) for i in range(5)]), evaluate,
            config=config, seed=17,
        )
        await original.run()
        restored = EliteTable("Score", None, evaluate, config=config, seed=17)
        restored.restore(original.organisms, original.generations)
        self.assertEqual(restored.rng.getstate(), original.rng.getstate())
        self.assertEqual(restored.best.id, original.best.id)
        self.assertEqual(
            [(r.kind, r.parent_ids) for r in restored._population(Generation(number=2))],
            [(r.kind, r.parent_ids) for r in original._population(Generation(number=2))],
        )
        with self.assertRaisesRegex(ValueError, "fresh agent"):
            restored.restore([], [])
        # A changed seed becomes observable once parent sampling starts.
        generation = Generation(number=2, status="running")
        original.generations.append(generation)
        changed = EliteTable("Score", None, evaluate, config=config, seed=18)
        with self.assertRaisesRegex(ValueError, "parents"):
            changed.restore(original.organisms, original.generations)

    async def test_restore_preserves_legacy_incumbent_tie_ranking(self):
        async def evaluate(policies):
            return {p.id: Measurement({0: 7}) for p in policies}

        mutation = json.dumps(dict(
            name="Edited", description="Changed source",
            edits=[dict(search="return 0", replacement="return 1")],
        ))
        config = Config(
            elite_size=1, population_size=1, generations=2,
            new_fraction=0, remix_fraction=0,
        )
        original = EliteTable("Score", ScriptedProvider([program(0), mutation]), evaluate, config=config)
        await original.run()
        self.assertEqual(original.generations[-1].elite_ids, [2])
        original.generations[-1].elite_ids = [1]
        restored = EliteTable("Score", None, evaluate, config=config)
        restored.restore(original.organisms, original.generations)
        self.assertEqual([r.id for r in restored.elites], [1])

    async def test_inconsistent_seed_panels_abort_search(self):
        async def evaluate(policies):
            return {p.id: Measurement({int(p.name.split()[-1]): 1}) for p in policies}

        agent = EliteTable(
            "Score", ScriptedProvider([program(0), program(1)]), evaluate,
            config=Config(population_size=2, generations=1),
        )
        with self.assertRaisesRegex(ValueError, "same seed panel"):
            await agent.run()
        self.assertEqual(agent.reason, "error")

    async def test_marked_source_keeps_archived_edit_boundaries(self):
        from _elitetable.edits import Edit, InvalidCandidate, apply_edits

        source = "# immutable\n# EVOLVE-BLOCK-START\nvalue = 1\n# EVOLVE-BLOCK-END\n"
        self.assertIn("value = 2", apply_edits(source, [Edit(search="value = 1", replacement="value = 2")]))
        with self.assertRaisesRegex(InvalidCandidate, "immutable boundary"):
            apply_edits(source, [Edit(search="# immutable", replacement="# changed")])

    async def test_requests_match_archived_slick_rendering_outside_cwd(self):
        import hashlib
        import os

        parent = Organism(
            id=1, generation=1, kind="new", name="Policy 0", description="Idea 0",
            score=3, implementation=json.loads(program(0))["implementation"],
        )
        agent = EliteTable("Score policies", None, None, context="Finite actions; seeds 0 and 1.")
        # SHA256 of requests rendered by the release's four decorated methods using Slick 0.3.0.
        cases = [
            (agent.invent, (7, [parent]), "4a172fae78b71f67bbcebdd5fea11eba3fdc3cb0b5f404e61848d481d9e2dcd1"),
            (agent.edit, (parent,), "0ead64acd40cfc4b8de98ee4dd6839aad03c0ecf35320e483fbf98b895fc7ae9"),
            (agent.remix, ([parent, parent],), "4f795a27ce2bffb6dc65e38ca70ceefa7441daf6173a3da359fbc2b1b34ca30f"),
            (agent.repair, (parent.implementation, "broken source", "invalid action"), "d5b2ca39de916c07e97e227e23e8b4698ed34541ef8320cbcc09a0df477043b5"),
        ]
        previous = Path.cwd()
        original_template_root = prompts.TEMPLATE_ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                for operation, args, expected in cases:
                    response = program(2)
                    if operation == agent.edit:
                        response = json.dumps(dict(name="Edit", description="Change", edits=[dict(search="0", replacement="2")]))
                    provider = ScriptedProvider([response])
                    await operation(*args, provider=provider)
                    self.assertEqual(hashlib.sha256(provider.calls[0].encode()).hexdigest(), expected)
                    self.assertEqual(prompts.TEMPLATE_ROOT, original_template_root)
        finally:
            os.chdir(previous)
