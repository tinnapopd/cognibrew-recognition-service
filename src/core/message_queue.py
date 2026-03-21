from typing import Callable

import pika
from pika.adapters.blocking_connection import BlockingChannel

from core.config import settings
from core.logger import Logger

logger = Logger().get_logger()


class MessageQueue:
    """Thin wrapper around pika for a single exchange / queue pair.

    Each instance manages its own connection, exchange, and queue.
    Pass ``exchange_name`` / ``queue_name`` to override the defaults
    from ``settings.rabbitmq``.
    """

    def __init__(
        self,
        exchange_name: str | None = None,
        queue_name: str | None = None,
    ) -> None:
        self._config = settings.rabbitmq
        self._exchange = exchange_name or self._config.INFERENCE_EXCHANGE_NAME
        self._queue = queue_name or self._config.INFERENCE_QUEUE_NAME
        self._connection: pika.BlockingConnection | None = None
        self._channel: BlockingChannel | None = None

    def connect(self, binding_keys: list[str] | None = None) -> None:
        credentials = pika.PlainCredentials(
            username=self._config.USERNAME,
            password=self._config.PASSWORD,
        )
        parameters = pika.ConnectionParameters(
            host=self._config.HOST,
            port=self._config.PORT,
            credentials=credentials,
        )

        self._connection = pika.BlockingConnection(parameters)
        self._channel = self._connection.channel()

        # Declare a topic exchange (idempotent – safe for both sides)
        self._channel.exchange_declare(
            exchange=self._exchange,
            exchange_type="topic",
            durable=True,
        )

        # Only set up queue + bindings when the caller needs to consume
        if binding_keys is not None:
            self._channel.queue_declare(
                queue=self._queue,
                durable=True,
            )
            for key in binding_keys:
                self._channel.queue_bind(
                    queue=self._queue,
                    exchange=self._exchange,
                    routing_key=key,
                )

        logger.info(
            "Connected to RabbitMQ at %s:%s (exchange=%s, queue=%s)",
            self._config.HOST,
            self._config.PORT,
            self._exchange,
            self._queue,
        )

    @property
    def channel(self) -> BlockingChannel:
        if self._channel is None or self._channel.is_closed:
            self.connect()
        assert self._channel is not None
        return self._channel

    def publish(self, body: bytes, routing_key: str = "") -> None:
        """Publish a raw protobuf-serialised message."""
        self.channel.basic_publish(
            exchange=self._exchange,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/x-protobuf",
            ),
        )
        logger.debug("Published message with routing_key='%s'", routing_key)

    def consume(self, callback: Callable[[bytes], None]) -> None:
        """Start consuming.  ``callback`` receives the raw body bytes."""

        def _on_message(
            ch: BlockingChannel,
            method: pika.spec.Basic.Deliver,
            properties: pika.BasicProperties,
            body: bytes,
        ) -> None:
            logger.debug(
                "Received message with routing_key='%s'",
                method.routing_key,
            )
            try:
                callback(body)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception:
                logger.exception("Error processing message")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self._queue,
            on_message_callback=_on_message,
        )

        logger.info("Waiting for messages on queue '%s'...", self._queue)
        self.channel.start_consuming()

    def close(self) -> None:
        if self._connection and not self._connection.is_closed:
            self._connection.close()
            logger.info("RabbitMQ connection closed")
