import json
import os
import aioredis

from fastapi_service.transaction import Transaction


class DBConnection:
    def __init__(self):
        self._redis = aioredis.Redis.from_url(os.environ.get("REDIS_DB_URL", "redis://localhost:16379"))

    async def get(self, trn_id):
        transaction = await self._redis.get(trn_id)
        if transaction is not None:
            transaction = Transaction.parse_obj(json.loads(transaction))
        return transaction

    async def put(self, transaction):
        json_transaction = json.dumps(transaction.dict())
        await self._redis.set(transaction.TransactionID, json_transaction)

    async def get_all(self):
        items_list = []
        keys = await self._redis.keys()
        for key in keys:
            items_list.append(await self.get(key))
        return items_list
