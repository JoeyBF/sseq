import datetime
import redis
import subprocess
import time
import threading
from celery import Celery

# Celery configuration (Redis as broker & backend)
app = Celery(
    "compressor",
    broker="redis://mdt6:6379/0",  # Connect to Redis
    backend="redis://mdt6:6379/0",
    task_acks_late=True,  # Ensures a task is only marked done *after* completion
    task_reject_on_worker_lost=True,  # Ensures crashed workers don’t leave tasks in limbo
    worker_prefetch_multiplier=1,  # Prevents one worker from hoarding multiple tasks at once
)

redis_client = redis.StrictRedis(host="mdt6", port=6379, db=0)


def refresh_lock(lock_key, interval=60):
    """Periodically refreshes the Redis lock while the task is running."""
    while True:
        time.sleep(interval)
        if not redis_client.exists(lock_key):  # If lock is gone, stop refreshing
            break
        redis_client.expire(lock_key, 600)  # Extend the lock for another 10 minutes


def zstd_process(file_path):
    log_channel = "compression_logs"

    # Start the zstd process
    process = subprocess.Popen(
        ["zstd", "-f", "--ultra", "-22", "--verbose", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Read stderr line-by-line and stream logs while process is running
    for line in process.stderr:
        current_time = datetime.datetime.now().isoformat()
        redis_client.publish(
            log_channel, f"[{current_time} {file_path}]: {line.strip()}"
        )

    # Ensure process fully exits
    return_code = process.wait()
    redis_client.publish(
        log_channel,
        f"[{current_time} {file_path}] Return code: {return_code}",
    )


def check_decompressed_hash(file_path):
    """Checks the hash of the decompressed file."""
    decompressed_filename = f"{file_path}.zst.un"
    subprocess.call(["zstd", "-d", file_path, "-o {decompressed_filename}"])

    original_sha = subprocess.check_output(["sha1sum", file_path])
    decompressed_sha = subprocess.check_output(["sha1sum", decompressed_filename])
    if original_sha != decompressed_sha:
        raise ValueError("Hashes do not match")
    else:
        subprocess.call(["rm", decompressed_filename])
        subprocess.call(["rm", file_path])


@app.task(bind=True)
def compress_file_task(self, file_path):
    """Ensures only one worker compresses a file at a time while streaming logs to Redis."""
    lock_key = f"compressing:{file_path}"

    # Try to acquire the lock atomically
    lock_acquired = redis_client.set(
        lock_key, "1", nx=True, ex=600
    )  # Lock expires in 10 min

    if not lock_acquired:
        print(f"Skipping {file_path}, already being processed")
        return  # Another worker already locked the task

    # Start lock refresher thread
    lock_refresher = threading.Thread(
        target=refresh_lock, args=(lock_key,), daemon=True
    )
    lock_refresher.start()

    try:
        zstd_process(file_path)
        check_decompressed_hash(file_path)
    finally:
        # Remove the lock after successful execution
        redis_client.delete(lock_key)
