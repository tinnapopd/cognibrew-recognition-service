"""Tests for core.message_queue - MessageQueue wrapper."""

from unittest.mock import MagicMock, patch

from core.message_queue import MessageQueue


class TestMessageQueueInit:
    def test_default_exchange_and_queue(self):
        mq = MessageQueue()
        assert mq._exchange == "cognibrew.inference"
        assert mq._queue == "cognibrew.inference.face_embedded"

    def test_custom_exchange_and_queue(self):
        mq = MessageQueue(
            exchange_name="my.exchange",
            queue_name="my.queue",
        )
        assert mq._exchange == "my.exchange"
        assert mq._queue == "my.queue"


class TestMessageQueueConnect:
    @patch("core.message_queue.pika.BlockingConnection")
    def test_connect_declares_exchange(self, mock_conn_cls):
        mock_channel = MagicMock()
        mock_conn_cls.return_value.channel.return_value = mock_channel

        mq = MessageQueue()
        mq.connect()

        mock_channel.exchange_declare.assert_called_once_with(
            exchange="cognibrew.inference",
            exchange_type="topic",
            durable=True,
        )
        # No queue declared when binding_keys is None
        mock_channel.queue_declare.assert_not_called()

    @patch("core.message_queue.pika.BlockingConnection")
    def test_connect_with_binding_keys(self, mock_conn_cls):
        mock_channel = MagicMock()
        mock_conn_cls.return_value.channel.return_value = mock_channel

        mq = MessageQueue()
        mq.connect(binding_keys=["face.embedded"])

        mock_channel.queue_declare.assert_called_once_with(
            queue="cognibrew.inference.face_embedded",
            durable=True,
        )
        mock_channel.queue_bind.assert_called_once_with(
            queue="cognibrew.inference.face_embedded",
            exchange="cognibrew.inference",
            routing_key="face.embedded",
        )


class TestMessageQueuePublish:
    @patch("core.message_queue.pika.BlockingConnection")
    def test_publish_sends_protobuf(self, mock_conn_cls):
        mock_channel = MagicMock()
        mock_channel.is_closed = False
        mock_conn_cls.return_value.channel.return_value = mock_channel

        mq = MessageQueue()
        mq.connect()
        mq.publish(body=b"\x08\x01", routing_key="face.recognized")

        mock_channel.basic_publish.assert_called_once()
        call_kwargs = mock_channel.basic_publish.call_args
        assert call_kwargs.kwargs["exchange"] == "cognibrew.inference"
        assert call_kwargs.kwargs["routing_key"] == "face.recognized"
        assert call_kwargs.kwargs["body"] == b"\x08\x01"


class TestMessageQueueClose:
    @patch("core.message_queue.pika.BlockingConnection")
    def test_close_connection(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn.is_closed = False
        mock_conn_cls.return_value = mock_conn
        mock_conn.channel.return_value = MagicMock()

        mq = MessageQueue()
        mq.connect()
        mq.close()

        mock_conn.close.assert_called_once()
