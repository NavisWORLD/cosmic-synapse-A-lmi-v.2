# LightToken Native Workstation — Interface Lock Addendum

This addendum is part of the implementation plan in `2026-09-17-lighttoken-native-workstation.md`. It freezes the cross-task interfaces discovered during plan self-review so implementation cannot drift between Rust, C++, Java, JNI, and Windows packaging.

## Rust core types

```rust
pub const LIGHTTOKEN_SCHEMA_VERSION: u32 = 1;
pub const EMBEDDING_DIMENSION: usize = 1536;
pub const SPECTRAL_DIMENSION: usize = 769;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityMethod {
    PowerCorrelation,
    Cosine,
    Euclidean,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComplexBin {
    pub real: f32,
    pub imag: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightTokenRecord {
    pub token_id: String,
    pub timestamp: String,
    pub source_uri: String,
    pub modality: String,
    pub raw_data_ref: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_text: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perceptual_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_embedding: Option<Vec<f32>>,
    #[serde(skip)]
    pub spectral_signature: Option<Vec<ComplexBin>>,
}
```

Wire serialization of an active spectrum remains the Python-compatible pair:

```json
{
  "spectral_signature_real": [0.0],
  "spectral_signature_imag": [0.0]
}
```

The Rust deserializer also accepts the explicitly supported historical pair `spectral_signature_magnitude` + `spectral_signature_phase`; writers emit only active real/imag form.

## Rust search types

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    pub method: SimilarityMethod,
    pub top_k: Option<usize>,
    pub threshold: Option<f32>,
    pub modality: Option<String>,
    pub source_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub token_id: String,
    pub score: f32,
    pub rank: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LightTokenSource {
    ReferenceOnly {
        token_id: String,
        source_path: String,
        raw_data_ref: String,
    },
    Resolved {
        token_id: String,
        source_path: String,
        raw_path: String,
        sha256: String,
    },
}
```

Ranking is score descending, then `token_id` ascending for ties.

## Native engine ABI

`LIGHTTOKEN_ABI_VERSION = 1`.

High-level C ABI calls operate on UTF-8 JSON and opaque context handles. Returned C strings are owned by the native library and released only with `lighttoken_string_free`.

JNI-facing Java interface:

```java
public interface NativeEngine extends AutoCloseable {
    int abiVersion();
    long contextHandle();
    ValidationResult validateJson(String json);
    TokenDetail inspectJson(String json);
    float[] embeddingValues(String json);
    float[] spectralPower(String json);
    ComparisonResult compareJson(String leftJson, String rightJson, SimilarityMethod method);
    SearchResult searchJson(String queryJson, String collectionJson, SearchRequest request);
    BackendInfo backendInfo();
    @Override void close();
}
```

`JniNativeEngine` is the sole production implementation in this project. Java controllers/services do not implement scoring math.

## Java application types

```java
public record BackendInfo(String activeBackend, boolean cppAvailable, String detail) {}
public record TokenSummary(String tokenId, String timestamp, String sourceUri, String modality, String rawDataRef) {}
public record SearchHit(String tokenId, float score, int rank) {}
public record SearchResult(String queryTokenId, SimilarityMethod method, String backend, List<SearchHit> hits) {}
```

SQLite schema version 1:

```sql
CREATE TABLE schema_version(version INTEGER NOT NULL);
CREATE TABLE sources(
  source_id INTEGER PRIMARY KEY,
  source_path TEXT NOT NULL UNIQUE,
  source_kind TEXT NOT NULL,
  content_sha256 TEXT,
  indexed_at TEXT NOT NULL
);
CREATE TABLE tokens(
  token_id TEXT PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
  source_uri TEXT NOT NULL,
  modality TEXT NOT NULL,
  raw_data_ref TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  cache_revision INTEGER NOT NULL
);
CREATE TABLE queries(
  query_id INTEGER PRIMARY KEY,
  query_token_id TEXT NOT NULL,
  method TEXT NOT NULL,
  backend TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE query_hits(
  query_id INTEGER NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
  rank INTEGER NOT NULL,
  token_id TEXT NOT NULL,
  score REAL NOT NULL,
  PRIMARY KEY(query_id, rank)
);
```

All user/source-derived SQL values use prepared statements.

## Packaged smoke contract

The Java application supports an automation mode that does not require a visible graphical desktop:

```text
LightTokenWorkstation --smoke --fixtures <fixture-dir> --json
```

The mode must:

1. load the packaged JNI library using the same application-owned lookup logic as normal startup;
2. open a temporary SQLite database;
3. ingest the committed synthetic fixture collection;
4. run one cosine and one power-correlation query through JNI/Rust;
5. exercise the optional C++ backend if its self-test succeeds, otherwise report Rust fallback without failing;
6. assert returned embedding length 1536 and spectral-power length 769;
7. emit one compact JSON result with `ok`, `abi_version`, `backend`, `token_count`, `embedding_dimension`, and `spectral_dimension`;
8. exit 0 on success and nonzero on any contract failure.

Normal JavaFX startup remains the real interactive workstation. CI uses `--smoke` for packaged-run verification so success does not depend on an interactive display server.

## Native library lookup

Java may load the Rust JNI library only from:

1. `-Dlighttoken.native.dir=<trusted CI/install directory>`, or
2. the packaged application-owned `native/` directory resolved relative to the launcher.

Rust may load the optional C++ acceleration library only from:

1. `LIGHTTOKEN_CPP_LIB` set by the trusted build/install process, or
2. the native application directory beside the Rust library/CLI.

No token metadata, source URI, raw data reference, imported workspace field, or SQLite row may choose a native library path.
