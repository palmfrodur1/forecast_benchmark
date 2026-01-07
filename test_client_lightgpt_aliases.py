import unittest
from unittest.mock import patch

from client_lightgpt import call_lightgpt_batch


class _FakeResp:
    def __init__(self):
        self.status_code = 200
        self.url = "https://example.test/api/v1/lightgpt/batch"
        self.text = "{}"

    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True}


class TestClientLightGPTAliases(unittest.TestCase):
    @patch("client_lightgpt.requests.post")
    def test_api_key_aliases_are_normalized_and_header_set(self, post_mock):
        post_mock.return_value = _FakeResp()

        payload = {
            "apiKey": "  nixtla-key  ",
            "freq": "MS",
        }

        out = call_lightgpt_batch(
            base_url="https://example.test",
            payload=payload,
            api_key="nostradamus-key",
            timeout_s=1.0,
        )
        self.assertEqual(out, {"ok": True})

        self.assertTrue(post_mock.called)
        _, kwargs = post_mock.call_args

        sent_json = kwargs.get("json")
        sent_headers = kwargs.get("headers")

        self.assertIsInstance(sent_json, dict)
        self.assertEqual(sent_json.get("api_key"), "nixtla-key")
        self.assertEqual(sent_json.get("freq"), "MS")

        self.assertIsInstance(sent_headers, dict)
        self.assertEqual(sent_headers.get("X-API-Key"), "nostradamus-key")
        self.assertEqual(sent_headers.get("X-Nixtla-Api-Key"), "nixtla-key")


if __name__ == "__main__":
    unittest.main()
