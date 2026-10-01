"""Execute generated policies in a fresh restricted Docker container per episode."""

import asyncio
import contextlib
import math
from pathlib import Path
from uuid import uuid4

from rsikit.episode import InfrastructureError, PolicyError, PolicyTimeout, run_episode
from rsikit.policy import Policy

from .codec import MAX_MESSAGE, MAX_SOURCE, decode, dumps, encode, encode_space, loads, pack, unpack


class SandboxPolicy(Policy):
    """Persistent remote instance; candidate source is never executed on the host."""

    def __init__(
        self,
        observation_space,
        action_space,
        *,
        instructions="",
        source,
        image="rsikit-sandbox:local",
        call_timeout=10.0,
    ):
        super().__init__(observation_space, action_space, instructions=instructions)
        if not math.isfinite(call_timeout) or call_timeout <= 0:
            raise InfrastructureError("call_timeout must be positive and finite")
        if len(source.encode()) > MAX_SOURCE:
            raise PolicyError("Source exceeds 64 KiB")
        try:
            self.space_definitions = (encode_space(observation_space), encode_space(action_space))
            dumps(self.space_definitions)
        except (TypeError, ValueError, RecursionError) as exc:
            raise InfrastructureError(f"Cannot encode policy spaces: {exc}") from exc
        self._implementation = source
        self.image, self.call_timeout = image, call_timeout
        self.container_name = f"rsikit-{uuid4().hex}"
        self.process = None
        self.ready = False
        self._binary = False

    async def reset(self, *, seed=None):
        if self.process is not None:
            raise InfrastructureError("A SandboxPolicy serves only one episode")
        try:
            self.process = await asyncio.create_subprocess_exec(
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "-i",
                "--name",
                self.container_name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                "512m",
                "--memory-swap",
                "512m",
                "--cpus",
                "1",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=16m",
                self.image,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=MAX_MESSAGE + 1,
            )
            line = await asyncio.wait_for(self.process.stdout.readline(), 60)
            if loads(line) != {"ready": True}:
                raise ValueError("Missing ready message")
            self.ready = True
        except asyncio.CancelledError as exc:
            await self._destroy_after(exc)
            raise
        except (OSError, ValueError, UnicodeError, asyncio.TimeoutError) as exc:
            failure = InfrastructureError(
                "Sandbox failed to start. Start Docker and build with: "
                "docker build -t rsikit-sandbox:local -f rsikit/sandbox/Dockerfile ."
            )
            await self._destroy_after(failure)
            raise failure from exc
        await self._request(
            {
                "command": "start",
                "source": self._implementation,
                "observation_space": self.space_definitions[0],
                "action_space": self.space_definitions[1],
                "instructions": self.instructions,
                "seed": seed,
            }
        )

    async def _request(self, request):
        if not self.ready:
            raise InfrastructureError("Sandbox is not running")
        try:
            payload = pack(request) if self._binary else dumps(request)
        except (ValueError, TypeError, RecursionError) as exc:
            raise InfrastructureError(f"Cannot encode sandbox request: {exc}") from exc

        try:
            data = await self._exchange_with_timeout(payload)
            response = unpack(data) if self._binary else loads(data)
            if not isinstance(response, dict):
                raise ValueError("Malformed supervisor response")
            if set(response) == {"error"} and isinstance(response["error"], str):
                raise PolicyError(response["error"])
            expected = "action" if request["command"] == "act" else "ok"
            if set(response) != {expected} or (expected == "ok" and response["ok"] is not True):
                raise ValueError("Malformed supervisor response")
            return response
        except asyncio.TimeoutError as exc:
            self.ready = False
            failure = PolicyTimeout(f"Policy {request['command']} exceeded {self.call_timeout}s")
            await self._destroy_after(failure)
            raise failure from exc
        except asyncio.CancelledError as exc:
            self.ready = False
            await self._destroy_after(exc)
            raise
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            UnicodeError,
            RecursionError,
        ) as exc:
            self.ready = False
            raise InfrastructureError(f"Sandbox protocol failed: {exc}") from exc

    async def _exchange(self, payload):
        self.process.stdin.write(payload)
        await self.process.stdin.drain()
        return await self.process.stdout.readline()

    async def _exchange_with_timeout(self, payload):
        return await asyncio.wait_for(self._exchange(payload), self.call_timeout)

    async def act(self, observation):
        try:
            encoded = observation if self._binary else encode(observation)
        except (ValueError, TypeError, RecursionError) as exc:
            raise InfrastructureError(f"Cannot encode observation: {exc}") from exc
        response = await self._request({"command": "act", "observation": encoded})
        try:
            return response["action"] if self._binary else decode(response["action"])
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as exc:
            raise PolicyError(f"Malformed candidate action: {exc}") from exc

    async def close(self):
        try:
            if self.ready:
                await self._request({"command": "close"})
        except BaseException as exc:
            await self._destroy_after(exc)
            raise
        else:
            await self._destroy()

    async def _destroy_after(self, primary):
        try:
            await self._destroy()
        except Exception as exc:
            primary.cleanup_error = exc

    async def _destroy(self):
        self.ready = False
        if self.process is None:
            return
        process, self.process = self.process, None
        process.stdin.close()
        cleanup = None
        try:
            cleanup = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                self.container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(cleanup.wait(), 10)
            await asyncio.wait_for(process.wait(), 10)
        except (OSError, asyncio.TimeoutError) as exc:
            raise InfrastructureError(
                f"Cannot remove sandbox {self.container_name}: {exc}"
            ) from exc
        finally:
            for child in (cleanup, process):
                if child is not None and child.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        child.kill()
                    await child.wait()


async def run_program(
    program: Path,
    make_env,
    *,
    env_seed: int | None = None,
    policy_seed: int | None = None,
    max_steps: int | None = None,
    instructions: str | None = None,
    call_timeout: float = 10.0,
):
    """Read source as data, then delegate the rollout to the shared episode runner."""
    try:
        with Path(program).open(encoding="utf-8") as stream:
            source = stream.read(MAX_SOURCE + 1)
    except (OSError, UnicodeError) as exc:
        raise InfrastructureError(f"Cannot read policy source: {exc}") from exc

    def make_policy(observation_space, action_space, *, instructions):
        return SandboxPolicy(
            observation_space,
            action_space,
            instructions=instructions,
            source=source,
            call_timeout=call_timeout,
        )

    return await run_episode(
        make_env,
        make_policy,
        env_seed=env_seed,
        policy_seed=policy_seed,
        max_steps=max_steps,
        instructions=instructions,
    )
