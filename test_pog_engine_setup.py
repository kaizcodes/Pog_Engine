import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


SETUP_PATH = Path(__file__).with_name("pog_engine_setup.py")


def load_setup_module():
    spec = importlib.util.spec_from_file_location("pog_engine_setup_under_test", SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status_code, headers):
        self.status_code = status_code
        self.headers = headers

    @property
    def ok(self):
        return self.status_code < 400

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class RemoteFileSizeTests(unittest.TestCase):
    def test_uses_target_size_after_http_redirect(self):
        setup = load_setup_module()
        url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"

        def fake_head(request_url, **kwargs):
            self.assertEqual(request_url, url)
            if kwargs.get("allow_redirects"):
                return FakeResponse(200, {"content-length": "3095033483"})
            return FakeResponse(302, {"content-length": "1099"})

        with patch.object(setup.requests, "head", side_effect=fake_head):
            self.assertEqual(setup.remote_file_size(url), 3095033483)


if __name__ == "__main__":
    unittest.main()
