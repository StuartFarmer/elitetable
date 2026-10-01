"""Breed a fixed population from measured elites and retain the best across generations."""

from __future__ import annotations

import ast
import asyncio
import logging
import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from statistics import fmean

from pydantic import ValidationError
from slick.providers import Provider

from _elitetable.edits import (
    InvalidCandidate,
    Mutation,
    Program,
    apply_edits,
    check_program,
    check_rewrite,
)
from rsikit import Policy
from rsikit.generation import WORKER_LIBRARIES
from rsikit.policy import _policy_class

from _elitetable.records import Generation, Organism
from _elitetable.rendering import generate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    elite_size: int = 10
    population_size: int = 50
    generations: int = 20
    new_fraction: float = 0.2
    remix_fraction: float = 0.4
    remix_parents: int = 3
    generation_concurrency: int = 100
    generation_timeout: float = 120
    max_repairs: int = 2
    target_score: float | None = None


@dataclass(frozen=True)
class Measurement:
    """Per-seed rewards or a candidate failure; infrastructure failures must raise."""

    scores: dict[int, float]
    failure: str | None = None


class _RecordedProvider:
    def __init__(self, provider, record):
        self.provider, self.record = provider, record

    async def acall(self, *args, **kwargs):
        text, calls = await self.provider.acall(*args, **kwargs)
        self.record["raw"] = text
        return text, calls


class EliteTable:
    """One run per instance. Evaluation is injected and must isolate generated code.

    Each population reads the prior generation's elites. Mean reward ranks first;
    ties prefer current edits, remixes, incumbents, then new ideas. Prompt resources
    resolve within the installed package independently of the working directory.
    Evaluation callbacks run concurrently as candidates arrive; use Executor to
    bound episode workers. Each generation still ranks only after all candidates finish.
    """

    libraries = WORKER_LIBRARIES

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Sequence[type[Policy]]], Awaitable[dict[str, Measurement]]],
        *,
        context: str = "",
        config: Config = Config(),
        seed: int = 0,
        on_checkpoint: Callable[[EliteTable], None] | None = None,
    ):
        if config.target_score is not None and not math.isfinite(config.target_score):
            raise ValueError("target_score must be finite")
        self.task, self.context, self.provider = task, context, provider
        self.evaluate, self.config, self.on_checkpoint = evaluate, config, on_checkpoint
        self.rng = random.Random(seed)
        self.elites: list[Organism] = []
        self.organisms: list[Organism] = []
        self.generations: list[Generation] = []
        self.reason = "ready"
        self._policies: dict[int, type[Policy]] = {}
        self._sources: set[str] = set()
        self._seed_panel: set[int] | None = None
        self._call_slots = asyncio.Semaphore(config.generation_concurrency)

    @property
    def best(self) -> type[Policy] | None:
        return self._policies[self.elites[0].id] if self.elites else None

    def records(self):
        return [*self.generations, *self.organisms]

    def restore(self, organisms: list[Organism], generations: list[Generation]) -> None:
        """Restore trusted records into a fresh agent; replay only parent allocation.

        Replaying allocation recovers RNG state for legacy runs without an RNG
        checkpoint. No model calls, policy execution, or score updates occur here.
        """
        if self.organisms or self.generations:
            raise ValueError("Restore requires a fresh agent")
        organisms = sorted(organisms, key=lambda row: row.id)
        generations = sorted(generations, key=lambda row: row.number)
        if [row.id for row in organisms] != list(range(1, len(organisms) + 1)):
            raise ValueError("Checkpoint organism IDs must be contiguous")
        if [g.number for g in generations] != list(range(1, len(generations) + 1)):
            raise ValueError("Checkpoint generation numbers must be contiguous")
        if any(g.status != "completed" for g in generations[:-1]):
            raise ValueError("Only the last checkpoint generation may be unfinished")
        if len(organisms) != len(generations) * self.config.population_size:
            raise ValueError("Checkpoint population does not match the search config")
        for generation in generations:
            expected = self._population(generation)
            rows = [row for row in organisms if row.generation == generation.number]
            if [(r.id, r.kind, r.parent_ids) for r in rows] != [
                (r.id, r.kind, r.parent_ids) for r in expected
            ]:
                raise ValueError("Checkpoint parents do not match the search seed/config")
            self.organisms[-len(rows) :] = rows
            self.generations.append(generation)
            for row in rows:
                if row.score is not None:
                    panel = set(map(int, row.seed_scores))
                    if (
                        row.status != "evaluated"
                        or not row.policy_id
                        or not panel
                        or not math.isfinite(row.score)
                        or any(not math.isfinite(v) for v in row.seed_scores.values())
                        or (self._seed_panel is not None and panel != self._seed_panel)
                    ):
                        raise ValueError("Invalid checkpoint scores")
                    self._seed_panel = panel
                if row.policy_id is not None:
                    policy = _policy_class(row.name, row.implementation, row.description)
                    if policy.id != row.policy_id:
                        raise ValueError("Checkpoint policy ID does not match its source")
                    self._policies[row.id] = policy
                implementations = [row.implementation] if row.policy_id else []
                implementations.extend(
                    revision["implementation"]
                    for revision in row.revisions
                    if revision.get("policy_id") is not None
                )
                for implementation in implementations:
                    check_program(implementation)
                    self._sources.add(ast.dump(ast.parse(implementation), include_attributes=False))
            if generation.status == "completed":
                ranked = self._rank(rows)
                if [row.id for row in ranked] != generation.elite_ids:
                    # Preserve pre-priority checkpoints and the parent order they used.
                    ranked = sorted(
                        self.elites + [row for row in rows if row.score is not None],
                        key=lambda row: (-row.score, row.id),
                    )[: self.config.elite_size]
                    if [row.id for row in ranked] != generation.elite_ids:
                        raise ValueError("Checkpoint elites do not match recorded scores")
                self.elites = ranked

    def _checkpoint(self):
        # ponytail: full snapshots; use incremental writes when history persistence dominates runtime.
        if self.on_checkpoint is not None:
            self.on_checkpoint(self)

    async def invent(self, proposal: int, elites: list[Organism], *, provider) -> Program:
        return await generate("new.j2", Program, provider, instance=self, proposal=proposal, elites=elites)

    async def edit(self, parent: Organism, *, provider) -> Mutation:
        return await generate("edit.j2", Mutation, provider, instance=self, parent=parent)

    async def remix(self, parents: list[Organism], *, provider) -> Program:
        return await generate("remix.j2", Program, provider, instance=self, parents=parents)

    async def repair(self, reference: str, failed: str, diagnostic: str, *, provider) -> Program:
        return await generate(
            "repair.j2", Program, provider, instance=self,
            reference=reference, failed=failed, diagnostic=diagnostic,
        )

    async def _call(self, row, operation, *args):
        for attempt in range(3):
            call = dict(operation=operation.__name__)
            row.calls.append(call)
            self._checkpoint()  # Transport retries do not consume candidate repairs.
            try:
                async with self._call_slots:
                    return await asyncio.wait_for(
                        operation(*args, provider=_RecordedProvider(self.provider, call)),
                        self.config.generation_timeout,
                    )
            except asyncio.TimeoutError as exc:
                call["error"] = f"TimeoutError: model call exceeded {self.config.generation_timeout:g}s"
                self._checkpoint()
                if attempt == 2:
                    raise TimeoutError(
                        f"Model {operation.__name__} for organism {row.id} timed out "
                        f"after 3 attempts ({self.config.generation_timeout:g}s each)"
                    ) from exc
                delay = 2**attempt
                logger.warning(
                    "Model %s for organism %s timed out after %gs; retry %s/2 in %ss",
                    operation.__name__, row.id, self.config.generation_timeout, attempt + 1, delay,
                )
                await asyncio.sleep(delay)
            except ValidationError as exc:
                call["error"] = str(exc)
                if "raw" not in call:
                    raise
                raise InvalidCandidate(f"Invalid generated output: {exc}") from exc
            except BaseException as exc:
                call["error"] = f"{type(exc).__name__}: {exc}"
                raise

    def _population(self, generation):
        n = self.config.population_size
        new = int(n * self.config.new_fraction) if self.elites else n
        remix = int(n * self.config.remix_fraction) if len(self.elites) >= 2 else 0
        kinds = ["new"] * new + ["remix"] * remix + ["edit"] * (n - new - remix)
        self.rng.shuffle(kinds)
        rows = []
        for kind in kinds:
            count = (
                0
                if kind == "new"
                else 1
                if kind == "edit"
                else min(len(self.elites), self.config.remix_parents)
            )
            parents = self.rng.sample(self.elites, count)
            rows.append(
                Organism(
                    id=len(self.organisms) + len(rows) + 1,
                    generation=generation.number,
                    kind=kind,
                    parent_ids=[p.id for p in parents],
                )
            )
        self.organisms.extend(rows)
        return rows

    async def _generate(self, row, *, repairing=False):
        parents = [self.organisms[i - 1] for i in row.parent_ids]
        reference = parents[0].implementation if row.kind == "edit" else ""
        while True:
            if repairing:
                if row.repairs >= self.config.max_repairs:
                    row.status = "discarded"
                    logger.warning(
                        "Discarded organism %s: %s",
                        row.id,
                        row.error,
                        extra={"event": "proposal_discarded"},
                    )
                    return row
                failed = row.implementation or row.calls[-1].get("raw", "")
                row.revisions.append(
                    dict(
                        implementation=failed,
                        policy_id=row.policy_id,
                        error=row.error,
                        status=row.status,
                    )
                )
                row.repairs += 1
                operation, args = self.repair, (reference, failed, row.error)
                logger.warning(
                    "Repairing organism %s (%s/%s): %s",
                    row.id,
                    row.repairs,
                    self.config.max_repairs,
                    row.error,
                )
            elif row.kind == "new":
                operation, args = self.invent, (row.id, list(self.elites))
            elif row.kind == "edit":
                operation, args = self.edit, (parents[0],)
            else:
                operation, args = self.remix, (parents,)
            row.status, row.policy_id, row.implementation = "generating", None, ""
            self._policies.pop(row.id, None)
            try:
                proposal = await self._call(row, operation, *args)
                row.name, row.description = proposal.name, proposal.description
                row.implementation = (
                    apply_edits(reference, proposal.edits)
                    if operation == self.edit
                    else proposal.implementation
                )
                check_program(row.implementation)
                if reference:
                    check_rewrite(reference, row.implementation)
                key = ast.dump(ast.parse(row.implementation), include_attributes=False)
                if key in self._sources:
                    raise InvalidCandidate("Duplicate program; make a substantive change")
                self._sources.add(key)
                policy = _policy_class(row.name, row.implementation, row.description)
                self._policies[row.id] = policy
                row.policy_id, row.status, row.error = policy.id, "generated", None
                logger.info(
                    "Generated %s (%s) — %s",
                    row.name,
                    row.kind,
                    row.description,
                    extra={"event": "policy_generated"},
                )
                self._checkpoint()
                return row
            except InvalidCandidate as exc:
                row.status, row.error = "rejected", str(exc)
                repairing = True
            except BaseException as exc:
                row.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                row.error = f"{type(exc).__name__}: {exc}"
                raise

    async def _measure(self, rows):
        while rows:
            for row in rows:
                row.status = "evaluating"
            self._checkpoint()
            results = await self.evaluate([self._policies[row.id] for row in rows])
            if set(results) != {row.policy_id for row in rows}:
                raise ValueError("Evaluator must return exactly the requested policy IDs")
            panel = self._seed_panel
            for result in results.values():
                if result.failure is not None:
                    continue
                if not result.scores or any(not math.isfinite(v) for v in result.scores.values()):
                    raise ValueError("Measurements must contain finite per-seed scores")
                if panel is not None and set(result.scores) != panel:
                    raise ValueError("All candidates must use the same seed panel")
                panel = set(result.scores)
            self._seed_panel = panel
            failed = []
            for row in rows:
                result = results[row.policy_id]
                if result.failure is not None:
                    row.status, row.error = "execution_failed", result.failure
                    failed.append(row)
                else:
                    row.score = fmean(result.scores.values())
                    row.seed_scores = {str(seed): value for seed, value in result.scores.items()}
                    row.status = "evaluated"
            self._checkpoint()
            if not failed:
                return
            logger.info(
                "Repairing %s failed organisms",
                len(failed),
                extra={"event": "generation_started", "total": len(failed)},
            )
            tasks = [asyncio.create_task(self._generate(row, repairing=True)) for row in failed]
            try:
                repaired = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            rows = [row for row in repaired if row.status == "generated"]

    async def _experiment(self, rows):
        logger.info(
            "Generating population of %s",
            len(rows),
            extra={"event": "generation_started", "total": len(rows)},
        )

        async def produce(row):
            if row.status in ("evaluated", "discarded"):
                return
            repairing = row.status in ("rejected", "execution_failed")
            if not repairing and row.id in self._policies:
                row.status = "generated"  # Reuse cached episodes; evaluate only missing seeds.
            else:
                if row.calls and row.calls[-1]["operation"] == "repair" and row.revisions:
                    repairing = True
                    if not row.implementation:
                        row.implementation = row.revisions[-1]["implementation"]
                        row.error = row.revisions[-1]["error"]
                await self._generate(row, repairing=repairing)
            if row.status == "generated":
                await self._measure([row])

        tasks = [asyncio.create_task(produce(row)) for row in rows]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _rank(self, rows):
        incumbents = {row.id for row in self.elites}
        return sorted(
            self.elites + [row for row in rows if row.score is not None],
            key=lambda row: (
                -row.score,
                2 if row.id in incumbents else {"edit": 0, "remix": 1, "new": 3}[row.kind],
                row.id,
            ),
        )[: self.config.elite_size]

    def _promote(self, generation, rows):
        previous = {row.id for row in self.elites}
        self.elites = self._rank(rows)
        generation.elite_ids = [row.id for row in self.elites]
        generation.promoted_ids = [row.id for row in self.elites if row.id not in previous]
        generation.status = "completed"
        logger.info(
            "Generation %s: %s promotions; %s/%s elites; best=%s",
            generation.number,
            len(generation.promoted_ids),
            len(self.elites),
            self.config.elite_size,
            self.elites[0].score if self.elites else "unfilled",
        )

    async def run(self) -> list[Organism]:
        self.reason = "running"
        try:
            pending = self.generations and self.generations[-1].status != "completed"
            if (
                not pending
                and self.config.target_score is not None
                and self.elites
                and self.elites[0].score >= self.config.target_score
            ):
                self.reason = "target_reached"
                return self.elites
            start = len(self.generations) if pending else len(self.generations) + 1
            for number in range(start, self.config.generations + 1):
                if pending and number == start:
                    generation = self.generations[-1]
                    generation.status, generation.error = "running", None
                    rows = [row for row in self.organisms if row.generation == number]
                else:
                    generation = Generation(
                        number=number, elite_ids=[row.id for row in self.elites]
                    )
                    self.generations.append(generation)
                    rows = self._population(generation)
                self._checkpoint()
                await self._experiment(rows)
                self._promote(generation, rows)
                self._checkpoint()
                if (
                    self.config.target_score is not None
                    and self.elites
                    and self.elites[0].score >= self.config.target_score
                ):
                    self.reason = "target_reached"
                    logger.info(
                        "Stopping after generation %s: best=%g reached target=%g",
                        number,
                        self.elites[0].score,
                        self.config.target_score,
                    )
                    return self.elites
            self.reason = "completed"
            return self.elites
        except BaseException as exc:
            self.reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            if self.generations:
                self.generations[-1].status = self.reason
                self.generations[-1].error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._checkpoint()
