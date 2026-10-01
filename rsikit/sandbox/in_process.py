"""Container-only whole-episode execution without per-action IPC."""

import os
import resource

from rsikit.episode import PolicyError
from rsikit.policy import Policy
from rsikit.sandbox.worker import block_connections, load_policy


class DirectPolicy(Policy):
    def __init__(self, observation_space, action_space, *, instructions, source):
        super().__init__(observation_space, action_space, instructions=instructions)
        try:
            self.policy = load_policy(source, observation_space, action_space, instructions)
        except BaseException as exc:
            raise PolicyError(f"{type(exc).__name__}: {str(exc)[:2000]}") from exc

    async def _call(self, method, *args, **kwargs):
        try:
            return await getattr(self.policy, method)(*args, **kwargs)
        except BaseException as exc:
            raise PolicyError(f"Policy {method}: {type(exc).__name__}: {str(exc)[:2000]}") from exc

    async def reset(self, *, seed=None):
        await self._call("reset", seed=seed)

    async def act(self, observation):
        return await self._call("act", observation)

    async def close(self):
        await self._call("close")


def run_in_process(request, output, directory):
    from rsikit.sandbox.evaluate import result_bytes

    # Give the supervisor a process group to reap, including video encoders or
    # other subprocesses. Docker still bounds the whole container's PIDs/CPU/RAM.
    os.setsid()
    block_connections(keep_process_group=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with output:
        output.sendall(result_bytes(request, directory=directory, in_process=True))
