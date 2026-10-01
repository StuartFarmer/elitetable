"""Run a policy using Gymnasium's environment loop and episode statistics."""

import logging
import sys
from collections.abc import Callable
from copy import deepcopy

import gymnasium as gym
import numpy as np

from .policy import Policy


class InfrastructureError(RuntimeError):
    """An isolated execution backend failed."""


class PolicyError(RuntimeError):
    """A generated policy failed; batch failures map policy IDs to diagnostics."""

    def __init__(self, message: str):
        super().__init__(message)
        self.failures: dict[str, str] = {}


class PolicyTimeout(PolicyError):
    """An isolated policy exceeded its execution deadline."""


async def run_episode(
    make_env: str | Callable[[], gym.Env],
    make_policy: Callable[..., Policy],
    *,
    env_seed: int | None = None,
    policy_seed: int | None = None,
    max_steps: int | None = None,
    instructions: str | None = None,
) -> tuple:
    """Return the final Gymnasium step tuple; totals are in info['episode'].

    Accept an environment ID or a factory returning a fresh environment. Existing
    Gymnasium time limits apply unless max_steps supplies an additional cap.
    Instructions are optional. Exceptions propagate after resources are closed.
    Use run_program for generated source; this path executes trusted classes.
    """
    env = gym.make(make_env) if isinstance(make_env, str) else make_env()
    policy = None
    try:
        if max_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)
        if instructions is None:
            instructions = (
                env.get_wrapper_attr("instructions") if env.has_wrapper_attr("instructions") else ""
            )
        policy = make_policy(
            deepcopy(env.observation_space),
            deepcopy(env.action_space),
            instructions=instructions,
        )
        observation, _ = env.reset(seed=env_seed)
        await policy.reset(seed=policy_seed)
        while True:
            action = await policy.act(deepcopy(observation))
            try:
                valid = env.action_space.contains(action)
            except (ValueError, TypeError, OverflowError):
                valid = False
            if not valid:
                raise PolicyError("Action outside action_space")
            # Discrete.contains accepts scalar arrays, but toy-text uses dict keys.
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(action)
            elif isinstance(env.action_space, gym.spaces.Box):
                # Box.contains validates lists via a temporary array; pass that representation.
                action = np.asarray(action, dtype=env.action_space.dtype)
            try:
                result = env.step(action)
            except gym.error.InvalidAction as exc:
                raise PolicyError(str(exc)) from exc
            observation, _, terminated, truncated, _ = result
            if terminated or truncated:
                return result
    finally:
        primary = sys.exc_info()[1]
        try:
            try:
                if policy is not None:
                    await policy.close()
            finally:
                env.close()
        except BaseException:
            if primary is None:
                raise
            # Keep the original failure/cancellation; expose secondary cleanup errors.
            logging.getLogger(__name__).exception("Cleanup failed while handling an episode error")
