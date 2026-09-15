import hashlib

from a_lmi.core.light_token import LightToken
from a_lmi.memory.object_storage_client import ObjectStorageClient


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self):
        return self.payload

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeObjectStore:
    def __init__(self):
        self.objects = {}

    def put_object(self, bucket, object_name, data, length, content_type=None, metadata=None):
        self.objects[(bucket, object_name)] = data.read(length)

    def get_object(self, bucket, object_name):
        return _Response(self.objects[(bucket, object_name)])


def _config():
    return {
        "infrastructure": {
            "minio": {
                "endpoint": "localhost:9000",
                "access_key": "local",
                "secret_key": "local",
                "bucket": "test-artifacts",
                "secure": False,
            }
        }
    }


def test_raw_artifact_bytes_get_uri_hash_and_token_provenance():
    fake = FakeObjectStore()
    storage = ObjectStorageClient(_config(), client=fake, ensure_bucket=False)
    token = LightToken("https://example.test", "text", "pending", "hello")
    payload = b"<html><body>cosmos</body></html>"

    receipt = storage.store_bytes_for_token(token, payload, content_type="text/html")

    expected_sha = hashlib.sha256(payload).hexdigest()
    assert receipt["sha256"] == expected_sha
    assert receipt["uri"].startswith("minio://test-artifacts/tokens/")
    assert token.raw_data_ref == receipt["uri"]
    assert token.metadata["raw_sha256"] == expected_sha
    assert token.metadata["raw_size_bytes"] == len(payload)
    assert storage.retrieve_uri(receipt["uri"]) == payload
