import asyncio
import os
import re
import sys
from collections import deque
from pathlib import Path

LOG_FILE = sys.argv[1]  # Path to the log file
MAX_CONCURRENT_ZSTD = 10  # Adjust as needed
ZSTD_COMMAND = [
    "zstd",
    "-f",
    "--rm",
    "--ultra",
    "-22",
    "--verbose",
]

# Regex to extract the file path from the log lines
LOG_PATTERN = re.compile(r"closing file=\"(.+)\"")
WRITTEN_PATTERN = re.compile(r"written=(\d+)")


def extract_file_path(line: str) -> str | None:
    """Extracts the file path from a log line if it matches the pattern."""
    match = LOG_PATTERN.search(line)
    return match.group(1) if match else None


def extract_written_size(line: str) -> int | None:
    """Extracts the written size from a log line if it matches the pattern."""
    match = WRITTEN_PATTERN.search(line)
    return int(match.group(1)) if match else None


async def compress_file(file_path: str):
    """Compress a file using zstd"""
    if Path(file_path).exists():
        process = await asyncio.create_subprocess_exec(
            *ZSTD_COMMAND,
            file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async for line in process.stderr:
            print(f"[ZSTD {file_path}]: {line.decode().strip()}")

        await process.wait()
    else:
        print(f"File not found: {file_path}")


# From https://stackoverflow.com/a/70858070
class AsyncTaskScheduler:
    def __init__(self):
        self.running = set()
        self.waiting = deque()

    @property
    def total_task_count(self):
        return len(self.running) + len(self.waiting)

    def add_task(self, coroutine):
        if len(self.running) >= MAX_CONCURRENT_ZSTD:
            self.waiting.append(coroutine)
        else:
            self._start_task(coroutine)

    def _start_task(self, coroutine):
        self.running.add(coroutine)
        asyncio.create_task(self._task(coroutine))

    async def _task(self, coroutine):
        try:
            return await coroutine
        finally:
            self.running.remove(coroutine)
            if self.waiting:
                coroutine2 = self.waiting.popleft()
                self._start_task(coroutine2)


async def main():
    runner = AsyncTaskScheduler()

    async def compress_file_wrapper(file_path):
        await compress_file(file_path)
        print(
            f"Finished compressing {file_path}, {runner.total_task_count - 1} tasks left"
        )

    fd = os.open(LOG_FILE, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "r", buffering=1) as log:
        log.seek(0, 2)  # Move to the end of the file
        while True:
            line = await asyncio.to_thread(log.readline)
            if not line:
                await asyncio.sleep(0.1)
                continue
            file_path = extract_file_path(line)
            size = extract_written_size(line)
            if file_path and size >= 2**10:
                print(
                    f"Detected file closure: {file_path}, total tasks: {runner.total_task_count}"
                )
                runner.add_task(compress_file_wrapper(file_path))

    # keep the program alive until all the tasks are done
    while runner.running_task_count > 0:
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
