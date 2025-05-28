# test/test_client.py
from unittest.mock import patch, Mock
from client import GuessingGameClient

def test_client_access_denied():
    fake_socket = Mock()
    fake_socket.recv.return_value = b"Incorrect password. Access denied."

    with patch("socket.socket") as mock_socket_class:
        mock_socket_class.return_value.__enter__.return_value = fake_socket

        inputs = iter(["wrongpw"])
        outputs = []

        client = GuessingGameClient(input_fn=lambda _: next(inputs), output_fn=outputs.append)
        client.start()

        assert any("Access denied" in o for o in outputs)
