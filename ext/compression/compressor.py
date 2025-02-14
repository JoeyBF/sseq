import datetime
import json
import os
import redis
import socket
import subprocess
import threading
import time
from celery import Celery

# Celery configuration (Redis as broker & backend)
app = Celery(
    "compressor",
    broker="redis://localhost:6379/0",  # Connect to Redis
    backend="redis://localhost:6379/0",
    task_acks_late=True,  # Ensures a task is only marked done *after* completion
    task_reject_on_worker_lost=True,  # Ensures crashed workers don’t leave tasks in limbo
    worker_prefetch_multiplier=1,  # Prevents one worker from hoarding multiple tasks at once
)

redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
log_channel = "compression_logs"
progress_channel = "compression_progress"
file_path = None


def refresh_lock(lock_key, interval=60):
    """Periodically refreshes the Redis lock while the task is running."""
    while True:
        time.sleep(interval)
        lock_info = redis_client.get(lock_key)
        if not lock_info:  # If lock is gone, stop refreshing
            break
        redis_client.expire(lock_key, 600)  # Extend the lock for another 10 minutes


def send_log(msg):
    redis_client.publish(log_channel, f"[{current_time()} {file_path}]: {msg}")


def send_progress(progress):
    redis_client.lpush(
        f"{progress_channel}:{file_path}", f"[{current_time()}]: {progress}"
    )

    # Limit list length to last 10 progress updates
    redis_client.ltrim(f"{progress_channel}:{file_path}", 0, 9)


def log_progress():
    """Retrieve last few progress logs from Redis."""
    last_lines = redis_client.lrange(
        f"{progress_channel}:{file_path}", 0, 1
    )  # Get last 2 logs

    for line in reversed(last_lines):
        send_log(line.decode("utf-8"))


def current_time():
    return datetime.datetime.now().isoformat()


def check_returncode(popen):
    return_code = popen.wait()
    args = " ".join(popen.args)
    send_log(f"{args} returned code {return_code}")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)


def zstd_process(lock_key):
    # Start the zstd process
    node_name = socket.gethostname()

    process = subprocess.Popen(
        ["zstd", "-f", "--ultra", "-22", "--progress", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    redis_client.set(lock_key, json.dumps({"node": node_name, "pid": process.pid}))

    # Read stderr line-by-line and stream logs while process is running
    first = True
    for line in process.stderr:
        stripped = line.strip()
        if first:
            send_log(stripped)
            first = False
        elif stripped:
            send_progress(line.strip())

    # Ensure process fully exits
    check_returncode(process)


def check_decompressed_hash():
    """Checks the hash of the decompressed file and deletes if it matches."""
    compressed_filename = f"{file_path}.zst"
    node_name = socket.gethostname()

    send_log(f"Decompressing")

    # Decompress into sha1sum
    process_zstd = subprocess.Popen(
        ["zstd", "-c", "-d", compressed_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=8192,
    )
    process_sha_compressed = subprocess.Popen(
        ["sha1sum"], stdin=process_zstd.stdout, stdout=subprocess.PIPE, text=True
    )
    process_sha_original = subprocess.Popen(
        ["sha1sum", file_path], text=True, stdout=subprocess.PIPE
    )

    check_returncode(process_zstd)
    process_zstd.stdout.close()

    check_returncode(process_sha_compressed)

    check_returncode(process_sha_original)

    # Read the hash result from sha1sum
    compressed_sha = process_sha_compressed.stdout.read().split()[0]
    original_sha = process_sha_original.stdout.read().split()[0]

    if original_sha != compressed_sha:
        send_log(f"Hashes do not match: {original_sha} != {compressed_sha}")
        raise ValueError(f"[{current_time()} {file_path}]: Hashes do not match!")
    else:
        send_log(f"Hashes match, deleting")
        os.remove(file_path)


@app.task(
    bind=True,
    autoretry_for=(
        BlockingIOError,
        RuntimeError,
        AssertionError,
        subprocess.CalledProcessError,
    ),
    retry_backoff=True,  # Exponential backoff
    retry_backoff_max=3600,  # Max delay of 1 hour
    max_retries=None,
)
def compress_file_task(self, _file_path):
    """Ensures only one worker compresses a file at a time while streaming logs to Redis."""
    global file_path
    file_path = _file_path
    lock_key = f"compressing:{file_path}"
    node_name = socket.gethostname()

    if not os.path.exists(file_path):
        assert os.path.exists(f"{file_path}.zst")
        send_log(f"Already processed and removed")
        return

    # Try to acquire the lock atomically
    lock_acquired = redis_client.set(
        lock_key, json.dumps({"node": node_name, "pid": None}), nx=True, ex=600
    )  # Lock expires in 10 min

    if not lock_acquired:
        send_log(f"Already being processed, checking progress...")
        log_progress()
        raise RuntimeError(
            f"File {file_path} is already being processed"
        )  # Raise so Celery retries later

    try:
        # Start lock refresher thread
        lock_refresher = threading.Thread(
            target=refresh_lock, args=(lock_key,), daemon=True
        )
        lock_refresher.start()

        zstd_process(lock_key)
        check_decompressed_hash()
    finally:
        # Remove the lock after successful execution
        redis_client.delete(lock_key)
