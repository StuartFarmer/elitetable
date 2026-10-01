"""Evaluate groups inside one sandbox; Run only persists returned results."""

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Protocol

import cloudpickle
import gymnasium as gym
from pydantic import BaseModel, Field, FiniteFloat

from .episode import InfrastructureError, PolicyError
from .sandbox.docker import InProcessDockerSandbox


class Sandbox(Protocol):
    async def start(self, workers: int) -> None: ...
    async def evaluate(
        self, implementation: str, environment: bytes, seed: int, call_timeout: float
    ) -> tuple[float, dict[str, bytes]]: ...
    async def close(self) -> None: ...


class Result(BaseModel):
    score: FiniteFloat
    artifacts: dict[str, bytes] = Field(default_factory=dict)


class Executor:
    """Own worker concurrency, policy call timeouts, and sandbox lifecycle."""

    def __init__(
        self, *, sandbox: Sandbox | None = None, concurrency: int = 1, call_timeout: float = 10.0
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.sandbox = InProcessDockerSandbox() if sandbox is None else sandbox
        self.concurrency = concurrency
        self.call_timeout = call_timeout
        self._busy = asyncio.Condition()
        self._slots = asyncio.Semaphore(concurrency)
        self._users = 0
        self._broken = False
        self._queued = self._running = 0
        self._keep_alive = False
        self._started = False
        self._needs_cleanup = False

    async def __aenter__(self):
        self._keep_alive = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            await self.aclose()
        except Exception:
            if exc is None:
                raise
            logging.getLogger(__name__).exception("Sandbox cleanup failed during shutdown")
        finally:
            self._keep_alive = False

    async def aclose(self):
        """Release the sandbox after all active evaluation work has finished."""
        async with self._busy:
            await self._busy.wait_for(lambda: self._users == 0)
            await self._close()

    async def _close(self):
        self._started = False
        if self._needs_cleanup:
            await self.sandbox.close()
            self._needs_cleanup = False
            logging.getLogger(__name__).info("Sandbox closed")

    @asynccontextmanager
    async def _session(self):
        async with self._busy:
            if self._broken and self._users:
                raise InfrastructureError(
                    "Sandbox interrupted; wait for active evaluations to finish"
                )
            if not self._started:
                try:
                    await self._close()
                    self._needs_cleanup = True
                    starting = asyncio.create_task(self.sandbox.start(self.concurrency))
                    try:
                        await asyncio.shield(starting)
                    except asyncio.CancelledError:
                        await starting
                        raise
                    self._started = True
                    self._broken = False
                except BaseException:
                    try:
                        await self._close()
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Sandbox cleanup failed during startup"
                        )
                    raise
            self._users += 1
        try:
            yield
        finally:
            primary = sys.exc_info()[1]
            async with self._busy:
                self._users -= 1
                try:
                    if not self._users and (not self._keep_alive or self._broken):
                        await self._close()
                except Exception:
                    if primary is None:
                        raise
                    logging.getLogger(__name__).exception(
                        "Sandbox cleanup failed during evaluation"
                    )
                finally:
                    self._busy.notify_all()

    async def evaluate(
        self,
        jobs: Iterable[tuple[str, str, int]],
        environment: gym.Env,
    ) -> AsyncIterator[tuple[str, int, Result]]:
        """Yield episode results; an async context retains the sandbox between batches."""
        jobs = list(jobs)
        if not jobs:
            return
        # Only caller-provided environments are serialized, never returned worker objects.
        definition = cloudpickle.dumps(environment)
        async with self._session():

            async def evaluate(policy_id, implementation, seed):
                queued_at = perf_counter()
                self._queued += 1
                try:
                    await self._slots.acquire()
                finally:
                    self._queued -= 1
                started_at = perf_counter()
                self._running += 1
                try:
                    if self._broken:
                        raise InfrastructureError("Sandbox interrupted during another evaluation")
                    try:
                        score, artifacts = await self.sandbox.evaluate(
                            implementation,
                            definition,
                            seed,
                            self.call_timeout,
                        )
                        return policy_id, seed, Result(score=score, artifacts=artifacts)
                    except asyncio.CancelledError:
                        self._broken = True
                        raise
                    except Exception as exc:
                        logging.getLogger(__name__).error(
                            "Policy %s failed (seed=%s): %s",
                            policy_id[:12],
                            seed,
                            exc,
                            extra={
                                "event": "evaluation_failed",
                                "policy_id": policy_id,
                                "seed": seed,
                            },
                        )
                        if isinstance(exc, PolicyError):
                            return policy_id, seed, exc
                        self._broken = True
                        raise
                finally:
                    self._running -= 1
                    self._slots.release()
                    logging.getLogger(__name__).debug(
                        "Episode %s seed=%s: queued %.4fs, evaluation %.4fs; %s active, %s queued",
                        policy_id[:12],
                        seed,
                        started_at - queued_at,
                        perf_counter() - started_at,
                        self._running,
                        self._queued,
                        extra={
                            "event": "episode_timing",
                            "policy_id": policy_id,
                            "seed": seed,
                            "queue_seconds": started_at - queued_at,
                            "evaluation_seconds": perf_counter() - started_at,
                            "active": self._running,
                            "queued": self._queued,
                        },
                    )

            tasks = []
            error = None
            policy_error = None
            failures = {}
            try:
                tasks = [asyncio.create_task(evaluate(*job)) for job in jobs]
                for task in asyncio.as_completed(tasks):
                    try:
                        completed = await task
                    except Exception as exc:
                        if error is None:
                            error = exc
                    else:
                        policy_id, seed, result = completed
                        if isinstance(result, PolicyError):
                            policy_error = policy_error or result
                            diagnostic = f"seed={seed}: {type(result).__name__}: {result}"
                            failures[policy_id] = "\n".join(
                                filter(None, [failures.get(policy_id), diagnostic])
                            )
                        else:
                            yield completed
                if error is not None:
                    raise error
                if policy_error is not None:
                    policy_error.failures = failures
                    raise policy_error
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
