"""The executable policy contract; training is not part of this lifecycle."""

import ast
import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from gymnasium import Space

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class Policy(ABC, Generic[Observation, Action]):
    """Choose actions using public task instructions and episode-local state."""

    def __init__(
        self,
        observation_space: Space[Observation],
        action_space: Space[Action],
        *,
        instructions: str = "",
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.instructions = instructions

    async def reset(self, *, seed: int | None = None) -> None:
        """Reset randomness; subclasses also clear their own episode memory."""
        self.rng = np.random.default_rng(seed)
        self.action_space.seed(seed)

    @abstractmethod
    async def act(self, observation: Observation) -> Action:
        """Return an action without access to the environment."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release policy resources."""


def _policy_class(name: str, implementation: str, description: str = "") -> type[Policy]:
    """Declare a generated policy without loading its implementation or choosing an executor."""
    tree = ast.parse(implementation)
    if not name.strip():
        raise ValueError("The generated policy needs a name")
    if not any(isinstance(node, ast.ClassDef) and node.name == "Solution" for node in tree.body):
        raise ValueError("The generated policy must define a Solution class")
    return type(
        "Solution",
        (Policy,),
        {
            "name": name,
            "description": description,
            "id": hashlib.sha256((name + "\0" + implementation).encode()).hexdigest(),
            "_implementation": implementation,
        },
    )
