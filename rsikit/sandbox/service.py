"""Container-only service: fresh episode processes from one preloaded forkserver."""

import asyncio
import json
import math
import multiprocessing as mp
import os
import signal
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from rsikit.episode import InfrastructureError, PolicyError, PolicyTimeout
from rsikit.sandbox.evaluate import error_result, run_evaluation
from rsikit.sandbox.in_process import run_in_process
from rsikit.sandbox.worker import run_candidate

MAX_FRAME = 64 * 1024 * 1024
PRELOAD_PID = os.getpid()
PRELOAD_THREADS = len(list(Path("/proc/self/task").iterdir())) if sys.platform == "linux" else None


def probe(channel):
    with channel:
        channel.sendall(
            json.dumps(
                {
                    "inherited": PRELOAD_PID == os.getppid() and PRELOAD_THREADS == 1,
                    "forkserver_pid": os.getppid(),
                }
            ).encode()
            + b"\n"
        )


def frame(value):
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_FRAME:
        raise InfrastructureError("Evaluation frame exceeds 64 MiB")
    return data + b"\n"


async def read_frame(reader):
    line = await reader.readline()
    if not line.endswith(b"\n") or len(line) > MAX_FRAME + 1:
        raise InfrastructureError("Missing or oversized evaluation frame")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise InfrastructureError("Evaluation frame must be an object")
    return value


def reap(process, *, group=False):
    if process.pid is not None:
        if group:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.is_alive():
            process.kill()
        process.join()
    process.close()


async def serve(workers, output):
    context = mp.get_context("forkserver")
    mp.set_forkserver_preload(["rsikit.sandbox.service"])
    parent, child = socket.socketpair()
    starting = context.Process(target=probe, args=(child,))
    writer = None
    try:
        starting.start()
        child.close()
        reader, writer = await asyncio.open_connection(sock=parent, limit=MAX_FRAME + 1)
        ready = await asyncio.wait_for(read_frame(reader), 60)
        if ready.get("inherited") is not True:
            raise InfrastructureError("Forkserver preload missing or started native threads")
    finally:
        child.close()
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        else:
            parent.close()
        reap(starting)

    write_lock = asyncio.Lock()

    def write(data):
        output.write(data)
        output.flush()

    async def send(value):
        data = frame(value)
        async with write_lock:
            await asyncio.to_thread(write, data)

    await send(
        {
            "ready": True,
            "protocol": 2,
            "supervisor_pid": os.getpid(),
            "forkserver_pid": ready["forkserver_pid"],
            "in_process": True,
        }
    )
    active = set()
    tasks = set()

    async def job(job_id, request):
        try:
            episode_timeout = request.get("episode_timeout", 60.0)
            if not math.isfinite(episode_timeout) or episode_timeout <= 0:
                raise InfrastructureError("episode_timeout must be positive and finite")
            in_process = request.get("in_process", False)
            if type(in_process) is not bool:
                raise InfrastructureError("in_process must be a boolean")
            with TemporaryDirectory() as directory:
                result_reader, result_writer = socket.socketpair()
                channels = [result_writer]
                if in_process:
                    processes = [
                        context.Process(
                            target=run_in_process, args=(request, result_writer, directory)
                        )
                    ]
                else:
                    policy_side, environment_side = socket.socketpair()
                    channels.extend((policy_side, environment_side))
                    processes = [
                        context.Process(target=run_candidate, args=(policy_side, directory)),
                        context.Process(
                            target=run_evaluation,
                            args=(request, environment_side, result_writer, directory),
                        ),
                    ]
                transport = None
                try:
                    for process in processes:
                        process.start()
                    for channel in channels:
                        channel.close()
                    reader, transport = await asyncio.open_connection(
                        sock=result_reader, limit=MAX_FRAME + 1
                    )
                    try:
                        result = await asyncio.wait_for(read_frame(reader), episode_timeout)
                    except asyncio.TimeoutError as exc:
                        raise PolicyTimeout(f"Episode exceeded {episode_timeout:g}s") from exc
                    except (InfrastructureError, ValueError, ConnectionError) as exc:
                        if in_process:
                            raise PolicyError(
                                f"Episode process exited without a valid result: {exc}"
                            ) from exc
                        raise
                finally:
                    for process in reversed(processes):
                        reap(process, group=in_process)
                    for channel in channels:
                        channel.close()
                    if transport is not None:
                        transport.close()
                        await transport.wait_closed()
                    else:
                        result_reader.close()
        except Exception as exc:
            result = error_result(exc)
        finally:
            active.discard(job_id)
        response = {"id": job_id, "result": result}
        try:
            frame(response)
        except InfrastructureError as exc:
            response["result"] = error_result(exc)
        await send(response)

    reader = asyncio.StreamReader(limit=MAX_FRAME + 1)
    protocol = asyncio.StreamReaderProtocol(reader)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: protocol, sys.stdin.buffer
    )
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            if not line.endswith(b"\n") or len(line) > MAX_FRAME + 1:
                raise InfrastructureError("Missing or oversized evaluation request")
            envelope = json.loads(line)
            if (
                not isinstance(envelope, dict)
                or set(envelope) != {"id", "request"}
                or not isinstance(envelope["id"], str)
                or not envelope["id"]
                or len(envelope["id"]) > 128
                or not isinstance(envelope["request"], dict)
            ):
                raise InfrastructureError("Malformed evaluation request")
            job_id = envelope["id"]
            if job_id in active:
                raise InfrastructureError("Duplicate evaluation ID")
            if len(active) >= workers:
                await send(
                    {
                        "id": job_id,
                        "result": error_result(
                            InfrastructureError("Sandbox concurrency limit exceeded")
                        ),
                    }
                )
                continue
            active.add(job_id)
            task = asyncio.create_task(job(job_id, envelope["request"]))
            tasks.add(task)
            # Keep tasks until shutdown so any output failure is observed.
            done = {task for task in tasks if task.done()}
            for completed in done:
                completed.result()
            tasks.difference_update(done)
    finally:
        transport.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    with os.fdopen(os.dup(1), "wb") as output:
        os.dup2(2, 1)
        asyncio.run(serve(workers, output))
