use crate::{similarity, Result, SpectrumError};
use libloading::{Library, Symbol};
use lighttoken_core::SimilarityMethod;
use std::env;
use std::path::{Path, PathBuf};

const CPP_ABI_VERSION: u32 = 1;
const CPP_OK: i32 = 0;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type SelfTestFn = unsafe extern "C" fn() -> i32;
type ManyFn = unsafe extern "C" fn(*const f32, *const f32, usize, usize, *mut f32) -> i32;
type SpectralPowerFn = unsafe extern "C" fn(*const f32, *const f32, usize, *mut f32) -> i32;
type TopKFn = unsafe extern "C" fn(*const f32, usize, usize, *mut usize) -> i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Rust,
    Cpp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDiagnostics {
    pub active: BackendKind,
    pub cpp_available: bool,
    pub detail: String,
}

struct CppBackend {
    _library: Library,
    cosine: ManyFn,
    correlation: ManyFn,
    euclidean: ManyFn,
}

fn disabled() -> bool {
    env::var("LIGHTTOKEN_DISABLE_CPP")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn library_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "lighttoken_accel.dll"
    } else if cfg!(target_os = "macos") {
        "liblighttoken_accel.dylib"
    } else {
        "liblighttoken_accel.so"
    }
}

fn configured_path() -> std::result::Result<PathBuf, String> {
    if let Ok(path) = env::var("LIGHTTOKEN_CPP_LIB") {
        if path.trim().is_empty() {
            return Err("LIGHTTOKEN_CPP_LIB is empty".into());
        }
        return Ok(PathBuf::from(path));
    }
    let executable =
        env::current_exe().map_err(|error| format!("cannot locate executable: {error}"))?;
    let directory = executable
        .parent()
        .ok_or_else(|| "native executable has no parent directory".to_string())?;
    Ok(directory.join(library_filename()))
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> std::result::Result<T, String> {
    let symbol: Symbol<'_, T> = library
        .get(name)
        .map_err(|error| format!("missing C++ accelerator symbol: {error}"))?;
    Ok(*symbol)
}

impl CppBackend {
    fn load() -> std::result::Result<Self, String> {
        if disabled() {
            return Err("disabled by LIGHTTOKEN_DISABLE_CPP".into());
        }
        let path = configured_path()?;
        Self::load_from_path(&path)
    }

    fn load_from_path(path: &Path) -> std::result::Result<Self, String> {
        if !path.is_file() {
            return Err(format!("C++ accelerator not found at {}", path.display()));
        }

        let library = unsafe { Library::new(path) }
            .map_err(|error| format!("cannot load C++ accelerator {}: {error}", path.display()))?;

        unsafe {
            let abi: AbiVersionFn = symbol(&library, b"lt_accel_abi_version\0")?;
            let self_test: SelfTestFn = symbol(&library, b"lt_accel_self_test\0")?;
            let cosine: ManyFn = symbol(&library, b"lt_accel_cosine_many\0")?;
            let correlation: ManyFn = symbol(&library, b"lt_accel_correlation_many\0")?;
            let euclidean: ManyFn = symbol(&library, b"lt_accel_euclidean_many\0")?;
            let _spectral_power: SpectralPowerFn = symbol(&library, b"lt_accel_spectral_power\0")?;
            let _top_k: TopKFn = symbol(&library, b"lt_accel_top_k\0")?;

            let reported = abi();
            if reported != CPP_ABI_VERSION {
                return Err(format!(
                    "C++ accelerator ABI mismatch: expected {CPP_ABI_VERSION}, got {reported}"
                ));
            }
            let self_test_status = self_test();
            if self_test_status != CPP_OK {
                return Err(format!(
                    "C++ accelerator self-test failed with status {self_test_status}"
                ));
            }

            Ok(Self {
                _library: library,
                cosine,
                correlation,
                euclidean,
            })
        }
    }

    fn similarity_many(
        &self,
        query: &[f32],
        candidates: &[Vec<f32>],
        method: SimilarityMethod,
    ) -> Result<Vec<f32>> {
        validate_batch(query, candidates)?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut flattened = Vec::with_capacity(candidates.len() * query.len());
        for candidate in candidates {
            flattened.extend_from_slice(candidate);
        }
        let mut output = vec![0.0f32; candidates.len()];
        let function = match method {
            SimilarityMethod::PowerCorrelation => self.correlation,
            SimilarityMethod::Cosine => self.cosine,
            SimilarityMethod::Euclidean => self.euclidean,
        };
        let status = unsafe {
            function(
                query.as_ptr(),
                flattened.as_ptr(),
                candidates.len(),
                query.len(),
                output.as_mut_ptr(),
            )
        };
        if status != CPP_OK {
            return Err(SpectrumError::Backend(format!(
                "C++ accelerator returned status {status}"
            )));
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(SpectrumError::Backend(
                "C++ accelerator returned a non-finite score".into(),
            ));
        }
        Ok(output)
    }
}

fn validate_batch(query: &[f32], candidates: &[Vec<f32>]) -> Result<()> {
    if query.is_empty() {
        return Err(SpectrumError::InvalidInput(
            "batch query must not be empty".into(),
        ));
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err(SpectrumError::InvalidInput(
            "batch query contains a non-finite value".into(),
        ));
    }
    for candidate in candidates {
        if candidate.len() != query.len() {
            return Err(SpectrumError::InvalidInput(format!(
                "batch candidate length {} does not match query length {}",
                candidate.len(),
                query.len()
            )));
        }
        if candidate.iter().any(|value| !value.is_finite()) {
            return Err(SpectrumError::InvalidInput(
                "batch candidate contains a non-finite value".into(),
            ));
        }
    }
    Ok(())
}

pub fn backend_diagnostics() -> BackendDiagnostics {
    match CppBackend::load() {
        Ok(_) => BackendDiagnostics {
            active: BackendKind::Cpp,
            cpp_available: true,
            detail: configured_path()
                .map(|path| format!("self-tested ABI v{CPP_ABI_VERSION}: {}", path.display()))
                .unwrap_or_else(|error| error),
        },
        Err(detail) => BackendDiagnostics {
            active: BackendKind::Rust,
            cpp_available: false,
            detail,
        },
    }
}

pub fn similarity_many(
    query: &[f32],
    candidates: &[Vec<f32>],
    method: SimilarityMethod,
) -> Result<Vec<f32>> {
    validate_batch(query, candidates)?;
    if let Ok(backend) = CppBackend::load() {
        return backend.similarity_many(query, candidates, method);
    }
    candidates
        .iter()
        .map(|candidate| similarity(query, candidate, method))
        .collect()
}
