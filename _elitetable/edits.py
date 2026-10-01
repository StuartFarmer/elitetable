"""Apply exact text edits while preserving marked evolution boundaries."""

import ast
import re
from inspect import Parameter, Signature
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

MARKER = re.compile(r"^.*?EVOLVE-BLOCK-(START|END)[^\r\n]*(?:\r?\n|$)", re.MULTILINE)


class Edit(BaseModel, extra="forbid"):
    search: str = Field(min_length=1)
    replacement: str


class Mutation(BaseModel, extra="forbid"):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    edits: list[Edit] = Field(min_length=1)


class Program(BaseModel, extra="forbid"):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    implementation: str = Field(min_length=1)


class InvalidCandidate(ValueError):
    """A generated candidate failed an edit or evaluation requirement."""


def evolution_regions(content: str) -> tuple[list[tuple[int, int]], tuple[str, ...]]:
    """Return editable spans and immutable pieces, including marker lines.

    Marker tokens are reserved, language-independent lexical delimiters. A file
    without markers is entirely editable. Nested or unbalanced markers fail.
    """
    regions, skeleton = [], []
    start, previous = None, 0
    for marker in MARKER.finditer(content):
        if marker[1] == "START" and start is None:
            skeleton.append(content[previous : marker.end()])
            start = marker.end()
        elif marker[1] == "END" and start is not None:
            regions.append((start, marker.start()))
            previous, start = marker.start(), None
        else:
            raise InvalidCandidate("Evolution markers must be balanced and non-nested")
    if start is not None:
        raise InvalidCandidate("Unclosed evolution block")
    if not regions:
        return [(0, len(content))], ()
    skeleton.append(content[previous:])
    return regions, tuple(skeleton)


def check_rewrite(parent: str, child: str) -> str:
    """Reject empty, unchanged, or out-of-bounds replacements without normalizing code."""
    if not child.strip() or child == parent:
        raise InvalidCandidate("Candidate is blank or unchanged")
    if evolution_regions(parent)[1] != evolution_regions(child)[1]:
        raise InvalidCandidate("Candidate changed the immutable skeleton or markers")
    return child


def apply_edits(parent: str, edits: list[Edit]) -> str:
    """Apply exact, sequential edits atomically within the evolution boundaries."""
    child = parent
    for edit in edits:
        search, replacement = edit.search, edit.replacement
        start = child.find(search)
        if not search or start < 0 or child.find(search, start + 1) >= 0:
            raise InvalidCandidate("Search must match exactly one nonempty source segment")
        end = start + len(search)
        if not any(left <= start and end <= right for left, right in evolution_regions(child)[0]):
            raise InvalidCandidate("Search crosses an immutable boundary")
        child = child[:start] + replacement + child[end:]
    return check_rewrite(parent, child)


def check_program(source: str) -> None:
    """Check syntax, edit boundaries, and the exported class without importing code."""
    evolution_regions(source)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise InvalidCandidate(f"Invalid Python: {exc}") from exc
    solutions = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solution"
    ]
    if len(solutions) != 1:
        raise InvalidCandidate("Program must define exactly one top-level Solution class")
    for method in reversed(solutions[0].body):
        if (
            not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            or method.name != "__init__"
        ):
            continue
        if isinstance(method, ast.AsyncFunctionDef):
            raise InvalidCandidate(
                "Solution.__init__ must be synchronous; initialize state in async reset"
            )
        args = method.args
        positional = [*args.posonlyargs, *args.args]
        required = len(positional) - len(args.defaults)
        parameters = [
            Parameter(
                arg.arg,
                Parameter.POSITIONAL_ONLY
                if i < len(args.posonlyargs)
                else Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty if i < required else None,
            )
            for i, arg in enumerate(positional)
        ]
        if args.vararg is not None:
            parameters.append(Parameter(args.vararg.arg, Parameter.VAR_POSITIONAL))
        parameters.extend(
            Parameter(
                arg.arg,
                Parameter.KEYWORD_ONLY,
                default=Parameter.empty if default is None else None,
            )
            for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        )
        if args.kwarg is not None:
            parameters.append(Parameter(args.kwarg.arg, Parameter.VAR_KEYWORD))
        try:
            # Bind the worker's actual call without evaluating any generated code or defaults.
            Signature(parameters).bind(None, None, None, instructions="")
        except (TypeError, ValueError) as exc:
            raise InvalidCandidate(
                "Solution.__init__ must accept (self, observation_space, action_space, *, "
                "instructions=''). Prefer removing __init__ and initializing state in "
                f"async reset after await super().reset(seed=seed). Signature mismatch: {exc}"
            ) from exc
        break  # Python uses the last definition of a method in the class body.
