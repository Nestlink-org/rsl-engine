"""Kafka producer and consumer services."""

import json
import logging
from typing import Any, Dict, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.core.config import settings

logger = logging.getLogger(__name__)


class KafkaProducerService:
    def __init__(self):
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await self._producer.start()
        logger.info("Kafka producer started")

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")

    async def publish(self, topic: str, message_dict: Dict[str, Any]) -> None:
        if not self._producer:
            logger.warning("Kafka producer not started — skipping publish")
            return
        try:
            await self._producer.send_and_wait(topic, message_dict)
        except Exception as e:
            logger.error(f"Kafka publish error on topic {topic}: {e}")


class KafkaConsumerService:
    def __init__(self):
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running = False

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            "rsl.claims.upload",
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id=settings.KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
        )
        await self._consumer.start()
        self._running = True
        logger.info("Kafka consumer started on topic rsl.claims.upload")
        import asyncio
        asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            logger.info("Kafka consumer stopped")

    async def _consume_loop(self) -> None:
        from app.workers.job_worker import process_job
        async for msg in self._consumer:
            if not self._running:
                break
            try:
                data = msg.value
                await process_job(
                    job_id=data["job_id"],
                    file_path=data["file_path"],
                    file_type=data["file_type"],
                    user_id=data["user_id"],
                )
            except Exception as e:
                logger.error(f"Kafka consumer processing error: {e}")


# Singletons
_producer: Optional[KafkaProducerService] = None
_consumer: Optional[KafkaConsumerService] = None


def get_kafka_producer() -> KafkaProducerService:
    global _producer
    if _producer is None:
        _producer = KafkaProducerService()
    return _producer


def get_kafka_consumer() -> KafkaConsumerService:
    global _consumer
    if _consumer is None:
        _consumer = KafkaConsumerService()
    return _consumer
