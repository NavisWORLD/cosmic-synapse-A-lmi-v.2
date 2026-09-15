"""Raw artifact storage with explicit provenance.

The active path persists bytes to an S3-compatible object store and returns an
inspectable URI + SHA-256 receipt. The MinIO SDK is optional at import time;
tests and alternate deployments can inject any compatible client.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ..core.light_token import LightToken


@dataclass(frozen=True)
class ArtifactRecord:
    uri: str
    sha256: str
    size: int
    content_type: str
    object_name: str


class ObjectStorageClient:
    """S3/MinIO-backed raw artifact store with injectable transport."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        client: Any = None,
        ensure_bucket: bool = True,
    ):
        self.config = config["infrastructure"]["minio"]
        self.logger = logging.getLogger(__name__)
        self.bucket_name = self.config["bucket"]

        if client is None:
            try:
                from minio import Minio
            except ImportError as exc:
                raise RuntimeError(
                    "MinIO support is optional; install the storage extra or inject a compatible client"
                ) from exc
            client = Minio(
                endpoint=self.config["endpoint"],
                access_key=self.config["access_key"],
                secret_key=self.config["secret_key"],
                secure=bool(self.config.get("secure", False)),
            )

        self.client = client
        if ensure_bucket:
            self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                self.logger.info("Created bucket: %s", self.bucket_name)
        except Exception as exc:
            raise RuntimeError(f"Unable to prepare object bucket {self.bucket_name!r}") from exc

    def store_bytes(
        self,
        payload: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> ArtifactRecord:
        """Persist exact bytes and return a provenance receipt."""

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        data = bytes(payload)
        if not object_name or object_name.startswith("/"):
            raise ValueError("object_name must be a non-empty relative object key")

        digest = hashlib.sha256(data).hexdigest()
        stream = io.BytesIO(data)
        self.client.put_object(
            self.bucket_name,
            object_name,
            stream,
            len(data),
            content_type=content_type,
        )
        return ArtifactRecord(
            uri=f"minio://{self.bucket_name}/{object_name}",
            sha256=digest,
            size=len(data),
            content_type=content_type,
            object_name=object_name,
        )

    def store_bytes_for_token(
        self,
        token: LightToken,
        payload: bytes,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        """Persist bytes for a token and attach the resulting provenance."""

        object_name = f"tokens/{token.token_id}/{token.modality}.data"
        artifact = self.store_bytes(payload, object_name, content_type)
        token.raw_data_ref = artifact.uri
        token.metadata["raw_sha256"] = artifact.sha256
        token.metadata["raw_size_bytes"] = artifact.size
        token.metadata["raw_content_type"] = artifact.content_type
        return {
            "uri": artifact.uri,
            "sha256": artifact.sha256,
            "size": artifact.size,
            "content_type": artifact.content_type,
            "object_name": artifact.object_name,
        }

    def store_raw_data(
        self,
        token: LightToken,
        raw_bytes: Optional[bytes] = None,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """Persist the token's raw artifact; never manufacture a fake object key.

        ``raw_bytes`` is preferred. If omitted, local/file references are read.
        Existing ``minio://`` references are verified by retrieval and their
        hash metadata is refreshed without duplicating the object.
        """

        if raw_bytes is not None:
            return self.store_bytes_for_token(token, raw_bytes, content_type)["uri"]

        ref = token.raw_data_ref
        if not ref:
            self.logger.warning("Token %s has no raw artifact reference", token.token_id)
            return None

        parsed = urlparse(ref)
        if parsed.scheme == "minio":
            payload = self.retrieve_uri(ref)
            token.metadata["raw_sha256"] = hashlib.sha256(payload).hexdigest()
            token.metadata["raw_size_bytes"] = len(payload)
            return ref

        if parsed.scheme in ("", "file"):
            path = Path(parsed.path if parsed.scheme == "file" else ref)
            if not path.is_file():
                raise FileNotFoundError(path)
            return self.store_bytes_for_token(token, path.read_bytes(), content_type)["uri"]

        raise ValueError(f"Unsupported raw_data_ref scheme: {parsed.scheme!r}")

    def retrieve_uri(self, uri: str) -> bytes:
        parsed = urlparse(uri)
        if parsed.scheme != "minio":
            raise ValueError(f"Unsupported object URI: {uri!r}")
        bucket = parsed.netloc
        object_name = parsed.path.lstrip("/")
        if not bucket or not object_name:
            raise ValueError(f"Malformed object URI: {uri!r}")
        return self._retrieve(bucket, object_name)

    def retrieve_raw_data(self, object_name: str) -> bytes:
        """Backward-compatible retrieval by object key in the configured bucket."""

        return self._retrieve(self.bucket_name, object_name)

    def _retrieve(self, bucket: str, object_name: str) -> bytes:
        response = self.client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            close = getattr(response, "close", None)
            if close:
                close()
            release = getattr(response, "release_conn", None)
            if release:
                release()

    def upload_file(
        self,
        file_path: str,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> bool:
        try:
            self.client.fput_object(
                self.bucket_name, object_name, file_path, content_type=content_type
            )
            return True
        except Exception as exc:
            self.logger.error("Object upload failed: %s", exc)
            return False

    def download_file(self, object_name: str, file_path: str) -> bool:
        try:
            self.client.fget_object(self.bucket_name, object_name, file_path)
            return True
        except Exception as exc:
            self.logger.error("Object download failed: %s", exc)
            return False
