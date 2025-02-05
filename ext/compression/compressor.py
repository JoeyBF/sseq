import subprocess
import redis
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

@app.task
def compress_file_task(file_path):
    """Compresses a file and streams logs via Redis."""
    process = subprocess.Popen(
        [
            "zstd",
            "--rm",
            "-f",
            "--ultra",
            "-22",
            "--verbose",
            file_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for line in process.stderr:
        redis_client.publish("compression_logs", f"[ZSTD {file_path}]: {line.strip()}")

    process.wait()

