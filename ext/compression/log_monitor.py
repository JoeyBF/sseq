import asyncio
import datetime
import os
import re
import sys
from compressor import compress_file_task  # Import Celery task

LOG_FILE = sys.argv[1]  # Path to the log file

LOG_PATTERN = re.compile(r"closing file=\"(.+)\"")


async def main():
    fd = os.open(LOG_FILE, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "r", buffering=1) as log:
        print(f"Opened {LOG_FILE}, waiting for filenames")
        while True:
            line = await asyncio.to_thread(log.readline)
            if not line:
                await asyncio.sleep(0.1)
                continue
            file_paths = LOG_PATTERN.findall(line)
            for file_path in file_paths:
                print(f"Dispatching {file_path} to Celery")
                compress_file_task.delay(file_path)  # Send task to Celery


if __name__ == "__main__":
    asyncio.run(main())
