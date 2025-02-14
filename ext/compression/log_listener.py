import redis

redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
pubsub = redis_client.pubsub()
pubsub.subscribe("compression_logs")

print("Listening for logs...")
for message in pubsub.listen():
    if message["type"] == "message":
        print(message["data"].decode("utf-8"))
