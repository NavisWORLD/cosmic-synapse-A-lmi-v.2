import hashlib
import io

from a_lmi.memory.object_storage_client import ObjectStorageClient


class FakeResponse:
    def __init__(self, data):
        self._data = data
    def read(self):
        return self._data
    def close(self):
        pass
    def release_conn(self):
        pass


class FakeObjectClient:
    def __init__(self):
        self.objects = {}
    def bucket_exists(self, bucket):
        return True
    def put_object(self, bucket, name, stream, length, content_type="application/octet-stream"):
        data = stream.read(length)
        self.objects[(bucket, name)] = (data, content_type)
    def get_object(self, bucket, name):
        return FakeResponse(self.objects[(bucket, name)][0])


def config():
    return {"infrastructure": {"minio": {"endpoint": "localhost:9000", "access_key": "x", "secret_key": "y", "bucket": "test", "secure": False}}}


def test_store_bytes_persists_exact_artifact_with_uri_and_sha256():
    fake = FakeObjectClient()
    store = ObjectStorageClient(config(), client=fake)
    payload = b"COSMOS raw artifact\x00\x01"
    artifact = store.store_bytes(payload, "raw/example.bin", "application/octet-stream")
    assert artifact.uri == "minio://test/raw/example.bin"
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.size == len(payload)
    assert store.retrieve_uri(artifact.uri) == payload
