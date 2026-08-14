//! Optional weight-file integrity (SHA-256).

use crate::error::HostError;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Computes lowercase hex SHA-256 of a file.
///
/// # Errors
///
/// I/O failures.
pub fn file_sha256_hex(path: &Path) -> Result<String, HostError> {
    let mut file = File::open(path).map_err(|e| HostError::Io(e.to_string()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 8 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| HostError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

/// Verifies `path` against an expected lowercase (or mixed-case) hex SHA-256.
///
/// # Errors
///
/// I/O failures or digest mismatch.
pub fn verify_file_sha256(path: &Path, expected_hex: &str) -> Result<(), HostError> {
    let expected = expected_hex.trim().to_ascii_lowercase();
    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(HostError::Integrity(format!(
            "invalid sha256 hex (need 64 hex chars): {expected_hex}"
        )));
    }
    let got = file_sha256_hex(path)?;
    if got != expected {
        return Err(HostError::Integrity(format!(
            "sha256 mismatch for {}: expected {expected}, got {got}",
            path.display()
        )));
    }
    Ok(())
}

/// When `expected` is `Some`, verifies; otherwise no-op.
///
/// # Errors
///
/// Integrity / I/O failures.
pub fn maybe_verify_sha256(path: &Path, expected: Option<&str>) -> Result<(), HostError> {
    if let Some(hex) = expected {
        verify_file_sha256(path, hex)?;
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use core::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn known_sha256_empty() {
        // SHA-256 of empty string.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("empty.bin");
        File::create(&p).unwrap();
        let got = file_sha256_hex(&p).unwrap();
        assert_eq!(
            got,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn mismatch_errors() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.bin");
        {
            let mut f = File::create(&p).unwrap();
            f.write_all(b"hello").unwrap();
        }
        let err = verify_file_sha256(
            &p,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        .unwrap_err();
        assert!(matches!(err, HostError::Integrity(_)));
    }
}
