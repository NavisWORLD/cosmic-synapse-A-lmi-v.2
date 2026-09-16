//! Deterministic, integrity-addressed `.cosmos` ZIP format compatible with the Python reference.

use almi_continuity::{inspect_workspace, require_workspace};
use almi_core::{canonical_json_bytes, AlmiError, ContinuityManifest, CosmosBundleMetadata, ManifestEntry, Result, SystemIdentity, BUNDLE_FORMAT_VERSION, BUNDLE_MANIFEST, MAX_BUNDLE_BYTES, MAX_BUNDLE_FILES, REQUIRED_WORKSPACE_FILES, WORKSPACE_FORMAT_VERSION};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, DateTime, ZipArchive, ZipWriter};

pub fn validate_archive_name(name: &str) -> Result<()> {
    if name.is_empty() || name.starts_with('/') || name.contains('\\') {
        return Err(AlmiError::Security(format!("unsafe archive path: {name:?}")));
    }
    let path = Path::new(name);
    if path.is_absolute() {
        return Err(AlmiError::Security(format!("unsafe archive path: {name:?}")));
    }
    if name.len() >= 3 && name.as_bytes()[1] == b':' && name.as_bytes()[2] == b'/' {
        return Err(AlmiError::Security(format!("unsafe archive path: {name:?}")));
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir | Component::CurDir | Component::RootDir | Component::Prefix(_)) {
            return Err(AlmiError::Security(format!("unsafe archive path: {name:?}")));
        }
    }
    if name.split('/').any(|part| part.is_empty() || part == "." || part == "..") {
        return Err(AlmiError::Security(format!("unsafe archive path: {name:?}")));
    }
    Ok(())
}

pub fn export_bundle(workspace: impl AsRef<Path>, destination: impl AsRef<Path>) -> Result<CosmosBundleMetadata> {
    let root = workspace.as_ref();
    require_workspace(root)?;
    let summary = inspect_workspace(root)?;
    let payloads = collect_payloads(root)?;
    let files: Vec<ManifestEntry> = payloads.iter().map(|(path, bytes)| ManifestEntry { path: path.clone(), sha256: digest(bytes), size: bytes.len() as u64 }).collect();
    let manifest = ContinuityManifest { bundle_format_version: BUNDLE_FORMAT_VERSION, workspace_format_version: WORKSPACE_FORMAT_VERSION, workspace_name: summary.system.name.clone(), files };
    let destination = destination.as_ref();
    if let Some(parent) = destination.parent() { fs::create_dir_all(parent)?; }
    let file = File::create(destination)?;
    let mut writer = ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .last_modified_time(DateTime::default())
        .unix_permissions(0o644);
    writer.start_file(BUNDLE_MANIFEST, options).map_err(zip_err)?;
    writer.write_all(&canonical_json_bytes(&manifest)?)?;
    for (path, bytes) in &payloads {
        writer.start_file(path, options).map_err(zip_err)?;
        writer.write_all(bytes)?;
    }
    writer.finish().map_err(zip_err)?;
    let bundle = fs::read(destination)?;
    Ok(CosmosBundleMetadata { valid: true, path: destination.display().to_string(), name: Some(summary.system.name), file_count: payloads.len(), total_bytes: payloads.iter().map(|(_,v)| v.len() as u64).sum(), sha256: digest(&bundle) })
}

pub fn inspect_bundle(path: impl AsRef<Path>) -> Result<CosmosBundleMetadata> { verify_bundle(path) }

pub fn verify_bundle(path: impl AsRef<Path>) -> Result<CosmosBundleMetadata> {
    let source = path.as_ref();
    let meta = fs::metadata(source).map_err(|_| AlmiError::Integrity("bundle is not a readable ZIP container".into()))?;
    if meta.len() > MAX_BUNDLE_BYTES.saturating_mul(2) {
        return Err(AlmiError::Security("bundle compressed size is excessive".into()));
    }
    let file = File::open(source)?;
    let mut archive = ZipArchive::new(file).map_err(zip_err)?;
    if archive.len() > MAX_BUNDLE_FILES + 1 {
        return Err(AlmiError::Security("bundle exceeds portable file-count limit".into()));
    }
    let mut seen = BTreeSet::new();
    let mut actual = BTreeSet::new();
    let mut total_declared_member_size = 0u64;
    for index in 0..archive.len() {
        let member = archive.by_index(index).map_err(zip_err)?;
        let name = member.name().to_owned();
        validate_archive_name(&name)?;
        if !seen.insert(name.clone()) {
            return Err(AlmiError::Security("bundle contains duplicate archive member names".into()));
        }
        if member.is_dir() {
            return Err(AlmiError::Security(format!("directory archive member is not allowed: {name}")));
        }
        if member.unix_mode().is_some_and(|mode| mode & 0o170000 == 0o120000) {
            return Err(AlmiError::Security(format!("symlink archive member is not allowed: {name}")));
        }
        if name != BUNDLE_MANIFEST && is_secret_path(&name) {
            return Err(AlmiError::Security(format!("secret-bearing archive member is not allowed: {name}")));
        }
        if member.size() > MAX_BUNDLE_BYTES {
            return Err(AlmiError::Security(format!("archive member is oversized: {name}")));
        }
        total_declared_member_size = total_declared_member_size.checked_add(member.size()).ok_or_else(|| AlmiError::Security("bundle size overflow".into()))?;
        if total_declared_member_size > MAX_BUNDLE_BYTES {
            return Err(AlmiError::Security("bundle exceeds portable uncompressed-size limit".into()));
        }
        if name != BUNDLE_MANIFEST { actual.insert(name); }
    }
    if !seen.contains(BUNDLE_MANIFEST) {
        return Err(AlmiError::Integrity("bundle manifest is missing".into()));
    }
    let manifest_bytes = read_member(&mut archive, BUNDLE_MANIFEST)?;
    let manifest: ContinuityManifest = serde_json::from_slice(&manifest_bytes).map_err(|_| AlmiError::Integrity("bundle manifest is invalid".into()))?;
    manifest.validate_versions()?;
    let mut declared = BTreeMap::new();
    for entry in &manifest.files {
        validate_archive_name(&entry.path)?;
        if entry.path == BUNDLE_MANIFEST {
            return Err(AlmiError::Integrity("bundle manifest cannot declare itself as payload".into()));
        }
        if declared.insert(entry.path.clone(), entry).is_some() {
            return Err(AlmiError::Integrity(format!("duplicate manifest declaration: {}", entry.path)));
        }
    }
    let declared_names: BTreeSet<String> = declared.keys().cloned().collect();
    if let Some(extra) = actual.difference(&declared_names).next() {
        return Err(AlmiError::Integrity(format!("undeclared archive member: {extra}")));
    }
    if let Some(missing) = declared_names.difference(&actual).next() {
        return Err(AlmiError::Integrity(format!("declared payload is missing: {missing}")));
    }
    for required in REQUIRED_WORKSPACE_FILES {
        if !declared.contains_key(*required) {
            return Err(AlmiError::Integrity(format!("bundle is missing required workspace payload: {required}")));
        }
    }
    let mut total = 0u64;
    for (name, entry) in &declared {
        let bytes = read_member(&mut archive, name)?;
        if bytes.len() as u64 != entry.size {
            return Err(AlmiError::Integrity(format!("size mismatch for {name}")));
        }
        if digest(&bytes) != entry.sha256 {
            return Err(AlmiError::Integrity(format!("hash mismatch for {name}")));
        }
        total += bytes.len() as u64;
    }
    let system_bytes = read_member(&mut archive, "system.json")?;
    let system: SystemIdentity = serde_json::from_slice(&system_bytes).map_err(|_| AlmiError::Integrity("bundle system.json is invalid".into()))?;
    system.validate()?;
    let bundle_bytes = fs::read(source)?;
    Ok(CosmosBundleMetadata { valid: true, path: source.display().to_string(), name: Some(system.name), file_count: declared.len(), total_bytes: total, sha256: digest(&bundle_bytes) })
}

pub fn import_bundle(source: impl AsRef<Path>, destination: impl AsRef<Path>) -> Result<CosmosBundleMetadata> {
    let source = source.as_ref();
    let verified = verify_bundle(source)?;
    let target = destination.as_ref();
    if target.exists() {
        let meta = fs::symlink_metadata(target)?;
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(AlmiError::Security(format!("unsafe extraction destination: {}", target.display())));
        }
        if fs::read_dir(target)?.next().transpose()?.is_some() {
            return Err(AlmiError::Security(format!("destination is not empty: {}", target.display())));
        }
    }
    fs::create_dir_all(target)?;
    let file = File::open(source)?;
    let mut archive = ZipArchive::new(file).map_err(zip_err)?;
    let manifest: ContinuityManifest = serde_json::from_slice(&read_member(&mut archive, BUNDLE_MANIFEST)?)?;
    for entry in &manifest.files {
        validate_archive_name(&entry.path)?;
        let bytes = read_member(&mut archive, &entry.path)?;
        let out = target.join(&entry.path);
        if let Some(parent) = out.parent() { fs::create_dir_all(parent)?; }
        fs::write(out, bytes)?;
    }
    inspect_workspace(target)?;
    Ok(CosmosBundleMetadata { path: target.display().to_string(), ..verified })
}

fn collect_payloads(root: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let mut files = Vec::new();
    visit(root, root, &mut files)?;
    files.sort_by(|a,b| a.0.cmp(&b.0));
    if files.len() > MAX_BUNDLE_FILES { return Err(AlmiError::Security("workspace exceeds portable bundle file-count limit".into())); }
    let total: u64 = files.iter().map(|(_,b)| b.len() as u64).sum();
    if total > MAX_BUNDLE_BYTES { return Err(AlmiError::Security("workspace exceeds portable bundle size limit".into())); }
    Ok(files)
}

fn visit(root: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?.collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        let meta = fs::symlink_metadata(&path)?;
        let relative = path.strip_prefix(root).map_err(|_| AlmiError::Security("workspace path escaped root".into()))?.to_string_lossy().replace('\\', "/");
        if meta.file_type().is_symlink() { return Err(AlmiError::Security(format!("workspace symlink is not portable: {relative}"))); }
        if meta.is_dir() { visit(root, &path, out)?; continue; }
        if !meta.is_file() { continue; }
        validate_archive_name(&relative)?;
        if is_secret_path(&relative) { return Err(AlmiError::Security(format!("secret-bearing file is excluded from bundles: {relative}"))); }
        let bytes = fs::read(&path)?;
        if bytes.len() as u64 > MAX_BUNDLE_BYTES { return Err(AlmiError::Security(format!("workspace file is oversized: {relative}"))); }
        out.push((relative, bytes));
    }
    Ok(())
}

fn is_secret_path(relative: &str) -> bool {
    relative.split('/').any(|part| {
        let lower = part.to_ascii_lowercase();
        matches!(lower.as_str(), ".env" | "credentials.json" | "secrets.json" | "id_rsa" | "id_ed25519")
            || lower.starts_with(".env")
            || [".pem", ".key", ".p12", ".pfx"].iter().any(|suffix| lower.ends_with(suffix))
    })
}

fn read_member<R: Read + std::io::Seek>(archive: &mut ZipArchive<R>, name: &str) -> Result<Vec<u8>> {
    let mut member = archive.by_name(name).map_err(zip_err)?;
    if member.size() > MAX_BUNDLE_BYTES { return Err(AlmiError::Security(format!("archive member is oversized: {name}"))); }
    let mut bytes = Vec::with_capacity(member.size() as usize);
    member.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn zip_err(error: zip::result::ZipError) -> AlmiError { AlmiError::Integrity(format!("ZIP error: {error}")) }
