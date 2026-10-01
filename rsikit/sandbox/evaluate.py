"""Container-only environment evaluation; the generated agent runs in a separate process."""

import asyncio
import base64
import json
import os
import signal
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import cloudpickle
import gymnasium as gym

from rsikit.episode import InfrastructureError, PolicyError, PolicyTimeout, run_episode
from rsikit.sandbox import SandboxPolicy
from rsikit.sandbox.codec import MAX_MESSAGE, frame_size, loads


class ProcessPolicy(SandboxPolicy):
    def __init__(self, *args, channel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel = channel
        self._binary = channel is not None

    async def reset(self, *, seed=None):
        if self.channel is None:
            await self._start_process()
        self.ready = True
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

    async def _start_process(self):
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-u",
            "-c",
            "import sys; sys.path.insert(0, '/opt/worker'); from rsikit.sandbox.worker import main; main()",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            limit=MAX_MESSAGE + 1,
        )
        if loads(await asyncio.wait_for(self.process.stdout.readline(), 60)) != {"ready": True}:
            raise InfrastructureError("Policy worker did not start")

    async def _exchange_with_timeout(self, payload):
        if self.channel is None:
            return await super()._exchange_with_timeout(payload)
        # This process serves exactly one episode. Blocking here avoids a task and
        # event-loop round trip per action; the supervisor still owns cancellation.
        deadline = monotonic() + self.call_timeout
        self.channel.settimeout(self.call_timeout)
        self.channel.sendall(payload)

        def receive(size):
            data = bytearray()
            while len(data) < size:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("Policy response deadline exceeded")
                self.channel.settimeout(remaining)
                chunk = self.channel.recv(size - len(data))
                if not chunk:
                    raise PolicyError("Candidate exited without a complete response")
                data.extend(chunk)
            return bytes(data)

        return receive(frame_size(receive(4)))

    async def _destroy(self):
        self.ready = False
        if self.channel is not None:
            self.channel.close()
            return
        if self.process is None:
            return
        process, self.process = self.process, None
        process.stdin.close()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def evaluate(request, *, channel=None, directory=None, in_process=False):
    if request["python"] != list(sys.version_info[:2]):
        raise InfrastructureError(
            "Host and sandbox Python minor versions must match; rebuild with --build-arg PYTHON_VERSION=X.Y"
        )
    with TemporaryDirectory() if directory is None else nullcontext(directory) as directory:
        os.chdir(directory)
        env = cloudpickle.loads(base64.b64decode(request["environment"], validate=True))
        # Only relocate output paths into this worker's filesystem; recording options stay on Gymnasium.
        wrapper, index = env, 0
        while isinstance(wrapper, gym.Wrapper):
            if isinstance(wrapper, gym.wrappers.RecordVideo):
                wrapper.video_folder = str(Path(directory) / "videos" / str(index))
                Path(wrapper.video_folder).mkdir(parents=True)
                index += 1
            wrapper = wrapper.env

        def make_policy(observation_space, action_space, *, instructions):
            if in_process:
                from rsikit.sandbox.in_process import DirectPolicy

                return DirectPolicy(
                    observation_space,
                    action_space,
                    instructions=instructions,
                    source=request["implementation"],
                )
            return ProcessPolicy(
                observation_space,
                action_space,
                instructions=instructions,
                source=request["implementation"],
                call_timeout=request["call_timeout"],
                channel=channel,
            )

        *_, info = await run_episode(
            lambda: env, make_policy, env_seed=request["seed"], policy_seed=request["seed"]
        )
        artifacts = dict(info.get("artifacts", {}))
        for path in Path(directory).rglob("*"):
            if path.is_file() and not path.is_symlink():
                artifacts[str(path.relative_to(directory))] = path.read_bytes()
        return {
            "score": float(info["episode"]["r"]),
            "artifacts": {
                name: base64.b64encode(data).decode() for name, data in artifacts.items()
            },
        }


def error_result(exc):
    kind = (
        "timeout"
        if isinstance(exc, PolicyTimeout)
        else "policy"
        if isinstance(exc, PolicyError)
        else "infrastructure"
    )
    return {"kind": kind, "error": f"{type(exc).__name__}: {str(exc)[:2000]}"}


def result_bytes(request, **kwargs):
    try:
        result = asyncio.run(evaluate(request, **kwargs))
        data = json.dumps(result, allow_nan=False).encode()
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("Evaluation artifacts exceed 64 MiB")
        return data + b"\n"
    except Exception as exc:
        return json.dumps(error_result(exc)).encode() + b"\n"


def run_evaluation(request, channel, output, directory):
    with channel, output:
        output.sendall(result_bytes(request, channel=channel, directory=directory))


def main():
    # Environment prints and native library output must not corrupt the result channel.
    with os.fdopen(os.dup(1), "w") as output:
        os.dup2(2, 1)
        try:
            raw = sys.stdin.buffer.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("Environment request exceeds 64 MiB")
            data = result_bytes(json.loads(raw)).decode()
        except Exception as exc:
            data = json.dumps(error_result(exc)) + "\n"
        output.write(data)
        output.flush()
