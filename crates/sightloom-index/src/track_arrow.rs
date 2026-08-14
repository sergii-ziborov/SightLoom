//! Arrow-shaped columnar track stream (no Apache Arrow crate dependency).
//!
//! Logical schema matches what hosts would put in an Arrow `RecordBatch` so
//! external tools can convert. On-disk codec is the compact **`SLARROW1`**
//! format (little-endian columns + validity bitmaps). Package default remains
//! CBOR (`tracks.cbor`); use this for analytics / interop sidecars.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::track_stream::TrackSample;
use crate::MemoryError;
use sightloom_core::{ClassId, MediaTime, SourceId, SubjectId, TrackId, TrackUid};

/// Codec magic + version for [`encode_track_arrow`] / [`decode_track_arrow`].
pub const TRACK_ARROW_MAGIC: &[u8; 8] = b"SLARROW1";
/// Current schema version.
pub const TRACK_ARROW_VERSION: u32 = 1;

/// Columnar batch of track samples (Arrow-shaped logical schema).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackArrowBatch {
    /// Row count.
    pub len: usize,
    /// `sample_id`
    pub sample_id: Vec<u64>,
    /// `supersedes` (optional)
    pub supersedes: Vec<Option<u64>>,
    /// `revision`
    pub revision: Vec<u32>,
    /// `idempotency_key`
    pub idempotency_key: Vec<u64>,
    /// `source_id`
    pub source_id: Vec<u32>,
    /// `frame_index`
    pub frame_index: Vec<u64>,
    /// Presentation timestamp value (ns or host ticks).
    pub pts_value: Vec<i64>,
    /// Presentation timescale.
    pub pts_timescale: Vec<u32>,
    /// Local `track_id`
    pub track_id: Vec<u32>,
    /// Global `track_uid` (optional)
    pub track_uid: Vec<Option<u64>>,
    /// `subject_id` (optional)
    pub subject_id: Vec<Option<u64>>,
    /// `class_id` (optional)
    pub class_id: Vec<Option<u16>>,
    /// Bbox left
    pub left: Vec<f32>,
    /// Bbox top
    pub top: Vec<f32>,
    /// Bbox right
    pub right: Vec<f32>,
    /// Bbox bottom
    pub bottom: Vec<f32>,
    /// Confidence
    pub confidence: Vec<f32>,
    /// Mask store handle
    pub mask_ref: Vec<u64>,
}

impl TrackArrowBatch {
    /// Builds a columnar batch from row-oriented samples.
    #[must_use]
    pub fn from_samples(samples: &[TrackSample]) -> Self {
        let n = samples.len();
        let mut batch = Self {
            len: n,
            sample_id: Vec::with_capacity(n),
            supersedes: Vec::with_capacity(n),
            revision: Vec::with_capacity(n),
            idempotency_key: Vec::with_capacity(n),
            source_id: Vec::with_capacity(n),
            frame_index: Vec::with_capacity(n),
            pts_value: Vec::with_capacity(n),
            pts_timescale: Vec::with_capacity(n),
            track_id: Vec::with_capacity(n),
            track_uid: Vec::with_capacity(n),
            subject_id: Vec::with_capacity(n),
            class_id: Vec::with_capacity(n),
            left: Vec::with_capacity(n),
            top: Vec::with_capacity(n),
            right: Vec::with_capacity(n),
            bottom: Vec::with_capacity(n),
            confidence: Vec::with_capacity(n),
            mask_ref: Vec::with_capacity(n),
        };
        for s in samples {
            batch.sample_id.push(s.sample_id);
            batch.supersedes.push(s.supersedes);
            batch.revision.push(s.revision);
            batch.idempotency_key.push(s.idempotency_key);
            batch.source_id.push(s.source_id.0);
            batch.frame_index.push(s.frame_index);
            batch.pts_value.push(s.pts.ticks());
            batch.pts_timescale.push(s.pts.timescale());
            batch.track_id.push(s.track_id.0);
            batch.track_uid.push(s.track_uid.map(|u| u.0));
            batch.subject_id.push(s.subject_id.map(|u| u.0));
            batch.class_id.push(s.class_id.map(|c| c.0)); // u16
            batch.left.push(s.left);
            batch.top.push(s.top);
            batch.right.push(s.right);
            batch.bottom.push(s.bottom);
            batch.confidence.push(s.confidence);
            batch.mask_ref.push(s.mask_ref);
        }
        batch
    }

    /// Converts back to row-oriented samples.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::Invalid`] when column lengths disagree or media
    /// time construction fails.
    pub fn to_samples(&self) -> Result<Vec<TrackSample>, MemoryError> {
        if !self.column_lens_ok() {
            return Err(MemoryError::Invalid);
        }
        let mut out = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let pts = MediaTime::new(self.pts_value[i], self.pts_timescale[i])
                .map_err(|_| MemoryError::Invalid)?;
            out.push(TrackSample {
                sample_id: self.sample_id[i],
                supersedes: self.supersedes[i],
                revision: self.revision[i],
                idempotency_key: self.idempotency_key[i],
                source_id: SourceId(self.source_id[i]),
                frame_index: self.frame_index[i],
                pts,
                track_id: TrackId(self.track_id[i]),
                track_uid: self.track_uid[i].map(TrackUid),
                subject_id: self.subject_id[i].map(SubjectId),
                class_id: self.class_id[i].map(ClassId),
                left: self.left[i],
                top: self.top[i],
                right: self.right[i],
                bottom: self.bottom[i],
                confidence: self.confidence[i],
                mask_ref: self.mask_ref[i],
            });
        }
        Ok(out)
    }

    fn column_lens_ok(&self) -> bool {
        let n = self.len;
        self.sample_id.len() == n
            && self.supersedes.len() == n
            && self.revision.len() == n
            && self.idempotency_key.len() == n
            && self.source_id.len() == n
            && self.frame_index.len() == n
            && self.pts_value.len() == n
            && self.pts_timescale.len() == n
            && self.track_id.len() == n
            && self.track_uid.len() == n
            && self.subject_id.len() == n
            && self.class_id.len() == n
            && self.left.len() == n
            && self.top.len() == n
            && self.right.len() == n
            && self.bottom.len() == n
            && self.confidence.len() == n
            && self.mask_ref.len() == n
    }
}

/// Encodes samples as `SLARROW1` bytes.
///
/// # Errors
///
/// Propagates conversion failures (should not fail for valid samples).
pub fn encode_track_arrow(samples: &[TrackSample]) -> Result<Vec<u8>, MemoryError> {
    let batch = TrackArrowBatch::from_samples(samples);
    encode_track_arrow_batch(&batch)
}

/// Encodes a columnar batch.
///
/// # Errors
///
/// Length mismatch.
pub fn encode_track_arrow_batch(batch: &TrackArrowBatch) -> Result<Vec<u8>, MemoryError> {
    if !batch.column_lens_ok() {
        return Err(MemoryError::Invalid);
    }
    let n = batch.len;
    let mut out = Vec::with_capacity(64 + n * 80);
    out.extend_from_slice(TRACK_ARROW_MAGIC);
    out.extend_from_slice(&TRACK_ARROW_VERSION.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());

    write_u64_col(&mut out, &batch.sample_id);
    write_opt_u64_col(&mut out, &batch.supersedes);
    write_u32_col(&mut out, &batch.revision);
    write_u64_col(&mut out, &batch.idempotency_key);
    write_u32_col(&mut out, &batch.source_id);
    write_u64_col(&mut out, &batch.frame_index);
    write_i64_col(&mut out, &batch.pts_value);
    write_u32_col(&mut out, &batch.pts_timescale);
    write_u32_col(&mut out, &batch.track_id);
    write_opt_u64_col(&mut out, &batch.track_uid);
    write_opt_u64_col(&mut out, &batch.subject_id);
    write_opt_u16_col(&mut out, &batch.class_id);
    write_f32_col(&mut out, &batch.left);
    write_f32_col(&mut out, &batch.top);
    write_f32_col(&mut out, &batch.right);
    write_f32_col(&mut out, &batch.bottom);
    write_f32_col(&mut out, &batch.confidence);
    write_u64_col(&mut out, &batch.mask_ref);
    Ok(out)
}

/// Decodes `SLARROW1` bytes into track samples.
///
/// # Errors
///
/// Bad magic, truncated buffer, or invalid media times.
pub fn decode_track_arrow(bytes: &[u8]) -> Result<Vec<TrackSample>, MemoryError> {
    decode_track_arrow_batch(bytes)?.to_samples()
}

/// Decodes into a columnar batch.
///
/// # Errors
///
/// Bad magic / truncated buffer.
pub fn decode_track_arrow_batch(bytes: &[u8]) -> Result<TrackArrowBatch, MemoryError> {
    let mut cur = bytes;
    if cur.len() < 16 {
        return Err(MemoryError::Invalid);
    }
    if &cur[..8] != TRACK_ARROW_MAGIC.as_slice() {
        return Err(MemoryError::Invalid);
    }
    cur = &cur[8..];
    let version = read_u32(&mut cur)?;
    if version != TRACK_ARROW_VERSION {
        return Err(MemoryError::Invalid);
    }
    let n = read_u32(&mut cur)? as usize;
    let sample_id = read_u64_col(&mut cur, n)?;
    let supersedes = read_opt_u64_col(&mut cur, n)?;
    let revision = read_u32_col(&mut cur, n)?;
    let idempotency_key = read_u64_col(&mut cur, n)?;
    let source_id = read_u32_col(&mut cur, n)?;
    let frame_index = read_u64_col(&mut cur, n)?;
    let pts_value = read_i64_col(&mut cur, n)?;
    let pts_timescale = read_u32_col(&mut cur, n)?;
    let local_track = read_u32_col(&mut cur, n)?;
    let global_uid = read_opt_u64_col(&mut cur, n)?;
    let subject = read_opt_u64_col(&mut cur, n)?;
    let class = read_opt_u16_col(&mut cur, n)?;
    let left = read_f32_col(&mut cur, n)?;
    let top = read_f32_col(&mut cur, n)?;
    let right = read_f32_col(&mut cur, n)?;
    let bottom = read_f32_col(&mut cur, n)?;
    let conf = read_f32_col(&mut cur, n)?;
    let mask = read_u64_col(&mut cur, n)?;
    Ok(TrackArrowBatch {
        len: n,
        sample_id,
        supersedes,
        revision,
        idempotency_key,
        source_id,
        frame_index,
        pts_value,
        pts_timescale,
        track_id: local_track,
        track_uid: global_uid,
        subject_id: subject,
        class_id: class,
        left,
        top,
        right,
        bottom,
        confidence: conf,
        mask_ref: mask,
    })
}

fn write_u32_col(out: &mut Vec<u8>, col: &[u32]) {
    for v in col {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_u64_col(out: &mut Vec<u8>, col: &[u64]) {
    for v in col {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_i64_col(out: &mut Vec<u8>, col: &[i64]) {
    for v in col {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_f32_col(out: &mut Vec<u8>, col: &[f32]) {
    for v in col {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn write_validity(out: &mut Vec<u8>, present: impl Iterator<Item = bool>, n: usize) {
    let nbytes = n.div_ceil(8);
    let mut bits = vec![0_u8; nbytes];
    for (i, p) in present.enumerate() {
        if p {
            bits[i / 8] |= 1 << (i % 8);
        }
    }
    out.extend_from_slice(&bits);
}

fn write_opt_u64_col(out: &mut Vec<u8>, col: &[Option<u64>]) {
    write_validity(out, col.iter().map(Option::is_some), col.len());
    for v in col {
        out.extend_from_slice(&v.unwrap_or(0).to_le_bytes());
    }
}

fn write_opt_u16_col(out: &mut Vec<u8>, col: &[Option<u16>]) {
    write_validity(out, col.iter().map(Option::is_some), col.len());
    for v in col {
        out.extend_from_slice(&v.unwrap_or(0).to_le_bytes());
    }
}

fn read_u32(cur: &mut &[u8]) -> Result<u32, MemoryError> {
    if cur.len() < 4 {
        return Err(MemoryError::Invalid);
    }
    let v = u32::from_le_bytes(cur[..4].try_into().unwrap());
    *cur = &cur[4..];
    Ok(v)
}

fn read_u32_col(cur: &mut &[u8], n: usize) -> Result<Vec<u32>, MemoryError> {
    let need = n.saturating_mul(4);
    if cur.len() < need {
        return Err(MemoryError::Invalid);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        out.push(u32::from_le_bytes(cur[off..off + 4].try_into().unwrap()));
    }
    *cur = &cur[need..];
    Ok(out)
}

fn read_u64_col(cur: &mut &[u8], n: usize) -> Result<Vec<u64>, MemoryError> {
    let need = n.saturating_mul(8);
    if cur.len() < need {
        return Err(MemoryError::Invalid);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 8;
        out.push(u64::from_le_bytes(cur[off..off + 8].try_into().unwrap()));
    }
    *cur = &cur[need..];
    Ok(out)
}

fn read_i64_col(cur: &mut &[u8], n: usize) -> Result<Vec<i64>, MemoryError> {
    let need = n.saturating_mul(8);
    if cur.len() < need {
        return Err(MemoryError::Invalid);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 8;
        out.push(i64::from_le_bytes(cur[off..off + 8].try_into().unwrap()));
    }
    *cur = &cur[need..];
    Ok(out)
}

fn read_f32_col(cur: &mut &[u8], n: usize) -> Result<Vec<f32>, MemoryError> {
    let need = n.saturating_mul(4);
    if cur.len() < need {
        return Err(MemoryError::Invalid);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        out.push(f32::from_le_bytes(cur[off..off + 4].try_into().unwrap()));
    }
    *cur = &cur[need..];
    Ok(out)
}

fn read_validity(cur: &mut &[u8], n: usize) -> Result<Vec<bool>, MemoryError> {
    let nbytes = n.div_ceil(8);
    if cur.len() < nbytes {
        return Err(MemoryError::Invalid);
    }
    let bits = &cur[..nbytes];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((bits[i / 8] & (1 << (i % 8))) != 0);
    }
    *cur = &cur[nbytes..];
    Ok(out)
}

fn read_opt_u64_col(cur: &mut &[u8], n: usize) -> Result<Vec<Option<u64>>, MemoryError> {
    let valid = read_validity(cur, n)?;
    let vals = read_u64_col(cur, n)?;
    Ok(valid
        .into_iter()
        .zip(vals)
        .map(|(p, v)| if p { Some(v) } else { None })
        .collect())
}

fn read_u16_col(cur: &mut &[u8], n: usize) -> Result<Vec<u16>, MemoryError> {
    let need = n.saturating_mul(2);
    if cur.len() < need {
        return Err(MemoryError::Invalid);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 2;
        out.push(u16::from_le_bytes(cur[off..off + 2].try_into().unwrap()));
    }
    *cur = &cur[need..];
    Ok(out)
}

fn read_opt_u16_col(cur: &mut &[u8], n: usize) -> Result<Vec<Option<u16>>, MemoryError> {
    let valid = read_validity(cur, n)?;
    let vals = read_u16_col(cur, n)?;
    Ok(valid
        .into_iter()
        .zip(vals)
        .map(|(p, v)| if p { Some(v) } else { None })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sightloom_core::MediaTime;

    fn sample(id: u64, subject: Option<u64>) -> TrackSample {
        TrackSample {
            sample_id: id,
            supersedes: if id > 1 { Some(id - 1) } else { None },
            revision: 1,
            idempotency_key: 42,
            source_id: SourceId(3),
            frame_index: 10 + id,
            pts: MediaTime::new(1_000_000_000, 1_000_000_000).unwrap(),
            track_id: TrackId(7),
            track_uid: Some(TrackUid(99)),
            subject_id: subject.map(SubjectId),
            class_id: Some(ClassId(1)),
            left: 1.0,
            top: 2.0,
            right: 3.0,
            bottom: 4.0,
            confidence: 0.9,
            mask_ref: 0,
        }
    }

    #[test]
    fn arrow_roundtrip() {
        let samples = vec![sample(1, Some(5)), sample(2, None)];
        let bytes = encode_track_arrow(&samples).unwrap();
        assert!(bytes.starts_with(TRACK_ARROW_MAGIC));
        let back = decode_track_arrow(&bytes).unwrap();
        assert_eq!(back, samples);
    }

    #[test]
    fn empty_roundtrip() {
        let bytes = encode_track_arrow(&[]).unwrap();
        let back = decode_track_arrow(&bytes).unwrap();
        assert!(back.is_empty());
    }
}
