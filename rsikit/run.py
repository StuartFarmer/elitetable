"""Store generated policies and their scores in one local SQLite database."""

import asyncio
import fcntl
import logging
import re
from collections.abc import Sequence
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from uuid import uuid4

import gymnasium as gym
from sqlalchemy import JSON, Column, inspect
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .execution import Executor
from .policy import Policy, _policy_class

logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:64] or "policy"


class _Settings(SQLModel, table=True):
    __tablename__ = "settings"
    name: str = Field(primary_key=True)
    export: bool = True


class _StoredPolicy(SQLModel, table=True):
    __tablename__ = "policy"
    id: str = Field(primary_key=True)
    name: str
    description: str = ""
    implementation: str
    scores: dict[str, float | None] = Field(default_factory=dict, sa_column=Column(JSON))


class Run:
    """A directory containing policies and scores. Unfinished scores are None."""

    @classmethod
    def create(
        cls,
        *,
        name: str,
        environment: gym.Env,
        path: str | Path | None = None,
        export: bool = True,
        executor: Executor | None = None,
    ) -> "Run":
        settings = _Settings(name=name, export=export)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = (
            Path(path)
            if path is not None
            else Path("runs") / f"{_slug(name)}-{stamp}-{uuid4().hex[:8]}"
        )
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory, environment, settings, executor=executor)

    @classmethod
    def open(
        cls, path: str | Path, *, environment: gym.Env, executor: Executor | None = None
    ) -> "Run":
        directory = Path(path)
        if not (directory / "run.sqlite").is_file():
            raise FileNotFoundError(directory / "run.sqlite")
        return cls(directory, environment, executor=executor)

    def __init__(
        self,
        path: Path,
        environment: gym.Env,
        settings: _Settings | None = None,
        *,
        executor: Executor | None = None,
    ):
        self.environment = environment
        self.executor = Executor() if executor is None else executor
        self.path = path.resolve()
        self._lock = (self.path / ".lock").open("a")
        self._engine = None
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._engine = create_engine(f"sqlite:///{self.path / 'run.sqlite'}")
            if settings is not None:
                SQLModel.metadata.create_all(
                    self._engine, tables=[_Settings.__table__, _StoredPolicy.__table__]
                )
            if "description" not in {
                column["name"] for column in inspect(self._engine).get_columns("policy")
            }:
                with self._engine.begin() as connection:
                    connection.exec_driver_sql(
                        "ALTER TABLE policy ADD COLUMN description TEXT NOT NULL DEFAULT ''"
                    )
            with Session(self._engine) as session:
                if settings is not None:
                    session.add(settings)
                    session.commit()
                self._settings = session.exec(select(_Settings)).one()
                for stored in session.exec(select(_StoredPolicy)):
                    self._export(stored)
            self._policy_locks = {}
        except BaseException:
            self.close()
            raise

    @property
    def name(self) -> str:
        return self._settings.name

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc):
        self.close()

    async def __aenter__(self) -> "Run":
        await self.executor.__aenter__()
        return self

    async def __aexit__(self, *exc):
        try:
            await self.executor.__aexit__(*exc)
        finally:
            self.close()

    async def aclose(self) -> None:
        """Close the persistent evaluation sandbox and release the run database."""
        try:
            await self.executor.aclose()
        finally:
            self.close()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
        self._lock.close()

    def save(self, *records: SQLModel) -> None:
        """Save typed records together, inferring and creating their tables.

        Primary keys identify records to update. Database-generated primary keys
        are copied back onto the supplied objects after a successful commit.
        """
        if self._lock.closed:
            raise RuntimeError("Run is closed")
        tables = list(dict.fromkeys(record.__table__ for record in records))
        SQLModel.metadata.create_all(self._engine, tables=tables)
        with Session(self._engine, expire_on_commit=False) as db:
            saved = [db.merge(record) for record in records]
            db.commit()
            for record, stored in zip(records, saved):
                mapper = inspect(record).mapper
                for column in mapper.primary_key:
                    key = mapper.get_property_by_column(column).key
                    setattr(record, key, getattr(stored, key))

    def database(self) -> Session:
        """Open a native SQLModel session for queries against this Run's SQLite database."""
        if self._lock.closed:
            raise RuntimeError("Run is closed")
        return Session(self._engine)

    def _export(self, policy: _StoredPolicy) -> None:
        if self._settings.export:
            destination = self.path / "exports" / f"{policy.id}_{_slug(policy.name)}.py"
            if not destination.exists():
                destination.parent.mkdir(exist_ok=True)
                temporary = destination.with_suffix(".py.tmp")
                temporary.write_text(policy.implementation, encoding="utf-8")
                temporary.replace(destination)

    def policies(self) -> list[type[Policy]]:
        with Session(self._engine) as session:
            return [
                _policy_class(row.name, row.implementation, row.description)
                for row in session.exec(select(_StoredPolicy))
            ]

    def scores(self, policy: type[Policy]) -> dict[int, float | None]:
        with Session(self._engine) as session:
            row = session.get(_StoredPolicy, policy.id)
            return {} if row is None else {int(seed): score for seed, score in row.scores.items()}

    @asynccontextmanager
    async def _evaluating(self, policy_ids):
        # Serialize overlapping policies to retain score reuse; different candidates
        # can feed the same executor while a previous candidate's last seed finishes.
        async with AsyncExitStack() as stack:
            for policy_id in sorted(set(policy_ids)):
                lock = self._policy_locks.setdefault(policy_id, asyncio.Lock())
                await stack.enter_async_context(lock)
            yield

    async def evaluate(self, policies: Sequence[type[Policy]], *, seeds=(0,)) -> dict[str, float]:
        """Save and evaluate a batch, returning mean scores keyed by policy ID.

        Each policy uses the same seeds, defaulting to one episode with seed 0.
        Completed scores are reused. Individual episode scores remain in scores().
        """
        seeds = tuple(dict.fromkeys(seeds))
        if not seeds:
            raise ValueError("Evaluation requires at least one seed")
        policies = tuple({policy.id: policy for policy in policies}.values())
        async with self._evaluating(policy.id for policy in policies):
            if self._lock.closed:
                raise RuntimeError("Run is closed")
            with Session(self._engine) as session:
                for policy in policies:
                    stored = session.get(_StoredPolicy, policy.id)
                    if stored is None:
                        stored = _StoredPolicy(
                            id=policy.id,
                            name=policy.name,
                            description=policy.description,
                            implementation=policy._implementation,
                        )
                    elif (stored.name, stored.implementation) != (
                        policy.name,
                        policy._implementation,
                    ):
                        raise ValueError("A stored policy cannot change under the same ID")
                    stored.scores = {**dict.fromkeys(map(str, seeds)), **stored.scores}
                    session.add(stored)
                # Persist the whole group before dispatching any evaluation.
                session.commit()
                jobs = []
                for policy in policies:
                    stored = session.get(_StoredPolicy, policy.id)
                    self._export(stored)
                    jobs.extend(
                        (stored.id, stored.implementation, seed)
                        for seed in seeds
                        if stored.scores[str(seed)] is None
                    )
            try:
                await self._execute(jobs)
            except Exception:
                unfinished = [
                    p.name for p in policies if any(self.scores(p)[seed] is None for seed in seeds)
                ]
                if unfinished:
                    logger.error("Unfinished policies: %s", ", ".join(unfinished))
                raise
            result = {}
            for policy in policies:
                scores = self.scores(policy)
                result[policy.id] = fmean(scores[seed] for seed in seeds)
            return result

    async def _execute(self, jobs):
        requested = {(policy_id, seed) for policy_id, _, seed in jobs}
        logger.info(
            "Evaluating %s episodes (%s workers)",
            len(jobs),
            self.executor.concurrency,
            extra={"event": "evaluation_started", "total": len(jobs)},
        )
        async with aclosing(self.executor.evaluate(jobs, self.environment)) as results:
            async for policy_id, seed, result in results:
                if (policy_id, seed) not in requested:
                    raise ValueError("Executor returned an unexpected or duplicate result")
                root = (self.path / "artifacts" / policy_id / str(seed)).resolve()
                if not root.is_relative_to(self.path):
                    raise ValueError("Artifact directory must stay inside the run")
                for name, data in result.artifacts.items():
                    destination = (root / name).resolve()
                    if not destination.is_relative_to(root):
                        raise ValueError("Artifact path must stay inside the evaluation directory")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".tmp")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                with Session(self._engine) as session:
                    stored = session.get(_StoredPolicy, policy_id)
                    policy_name = stored.name
                    stored.scores = {**stored.scores, str(seed): result.score}
                    session.add(stored)
                    session.commit()
                logger.info(
                    "%s: score=%g (seed=%s)",
                    policy_name,
                    result.score,
                    seed,
                    extra={"event": "policy_evaluated", "policy_id": policy_id, "seed": seed},
                )
                requested.remove((policy_id, seed))
        if requested:
            raise RuntimeError("Executor finished without returning all requested results")

    async def resume(self) -> None:
        """Evaluate only missing scores with this run's executor."""
        if self._lock.closed:
            raise RuntimeError("Run is closed")
        policy_ids = {policy.id for policy in self.policies()}
        async with self._evaluating(policy_ids):
            if self._lock.closed:
                raise RuntimeError("Run is closed")
            with Session(self._engine) as session:
                jobs = [
                    (row.id, row.implementation, int(seed))
                    for row in session.exec(select(_StoredPolicy))
                    if row.id in policy_ids
                    for seed, score in row.scores.items()
                    if score is None
                ]
            await self._execute(jobs)
