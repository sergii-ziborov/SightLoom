#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
//! On-disk `VisionIndex` package layout with transactional generations.
//!
//! Production layout:
//! ```text
//! package/
//!   CURRENT                 # points at active generation name
//!   gen-00000001/
//!     manifest.json
//!     checksums.json
//!     tracks.cbor
//!     masks.bin
//!     events.cbor
//!     entities.json
//!     [events.sqlite]
//! ```
//!
//! Write protocol:
//! 1. Write payload into `gen-N.tmp/`
//! 2. Compute per-file FNV-1a checksums + sizes
//! 3. Re-read verify
//! 4. fsync files and directory
//! 5. Atomic rename `gen-N.tmp` → `gen-N`
//! 6. Write `CURRENT` (temp + rename)
//! 7. Delete older generations after successful open of the new one
//!
//! Legacy flat layouts (manifest.json at package root) still load.

use crate::snapshot::{
    AnomalyEventDto, AppearanceDto, CoOccurrenceDto, EventEnvelopeDto, MediaTimeDto,
    PatternRecordDto, RouteDto, SourceTransitionDto, SubjectProfileDto, TrackSampleDto, VisitDto,
    ZoneStayDto,
};
use crate::{MemoryError, TrackSample, VisionIndex, VisionIndexHeader, VisionIndexSnapshot};
use sightloom_core::{ClassId, MaskRef, MediaTime, SourceId, SubjectId, TrackId};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Filename used for the package header JSON.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Pointer file naming the active generation directory.
pub const CURRENT_FILE: &str = "CURRENT";
/// Per-generation checksum manifest.
pub const CHECKSUMS_FILE: &str = "checksums.json";
/// Optional identity gallery sidecar written by the facade session.
pub const GALLERY_FILE: &str = "gallery.json";

/// Saves and loads a directory-based `VisionIndex` package.
#[derive(Clone, Debug, Default)]
pub struct VisionIndexPackage;

impl VisionIndexPackage {
    /// Writes `index` into `dir` using a transactional generation.
    ///
    /// # Errors
    ///
    /// Returns I/O or serialization failures.
    pub fn save(index: &VisionIndex, dir: impl AsRef<Path>) -> Result<(), MemoryError> {
        index.validate_fast()?;
        let root = dir.as_ref();
        fs::create_dir_all(root).map_err(|error| MemoryError::Io(error.to_string()))?;

        let next_gen = next_generation_id(root)?;
        let gen_name = format!("gen-{next_gen:08}");
        let tmp_name = format!("{gen_name}.tmp");
        let tmp_dir = root.join(&tmp_name);
        let final_dir = root.join(&gen_name);

        if tmp_dir.exists() {
            fs::remove_dir_all(&tmp_dir).map_err(|error| MemoryError::Io(error.to_string()))?;
        }
        fs::create_dir_all(&tmp_dir).map_err(|error| MemoryError::Io(error.to_string()))?;

        write_index_payload(index, &tmp_dir)?;
        let checksums = compute_checksums(&tmp_dir, index)?;
        write_json_atomic(&tmp_dir.join(CHECKSUMS_FILE), &checksums)?;
        verify_checksums(&tmp_dir, &checksums)?;
        fsync_dir_tree(&tmp_dir)?;

        if final_dir.exists() {
            fs::remove_dir_all(&final_dir).map_err(|error| MemoryError::Io(error.to_string()))?;
        }
        fs::rename(&tmp_dir, &final_dir).map_err(|error| MemoryError::Io(error.to_string()))?;
        fsync_path(&final_dir)?;

        write_current_pointer(root, &gen_name)?;
        // Open the new generation before deleting older ones.
        let _ = load_from_generation(&final_dir)?;
        prune_old_generations(root, &gen_name)?;
        Ok(())
    }

    /// Loads a package directory into an in-memory [`VisionIndex`].
    ///
    /// Supports generation layout (`CURRENT` + `gen-*`) and legacy flat layout.
    ///
    /// # Errors
    ///
    /// Returns I/O, validation, or deserialization failures.
    pub fn load(dir: impl AsRef<Path>) -> Result<VisionIndex, MemoryError> {
        let root = dir.as_ref();
        if let Some(generation) = read_current_pointer(root)? {
            let generation_dir = root.join(generation.trim());
            if generation_dir.is_dir() {
                let index = load_from_generation(&generation_dir)?;
                // Optional checksum verification when present.
                let checksums_path = generation_dir.join(CHECKSUMS_FILE);
                if checksums_path.exists() {
                    let text = fs::read_to_string(&checksums_path)
                        .map_err(|error| MemoryError::Io(error.to_string()))?;
                    let checksums: ChecksumsFile = serde_json::from_str(&text)
                        .map_err(|error| MemoryError::Serde(error.to_string()))?;
                    verify_checksums(&generation_dir, &checksums)?;
                }
                return Ok(index);
            }
        }
        // Legacy flat layout (manifest at root).
        if root.join(MANIFEST_FILE).exists() {
            return load_from_generation(root);
        }
        Err(MemoryError::Io(format!(
            "no VisionIndex package found at {}",
            root.display()
        )))
    }

    /// Returns the active generation directory name when present.
    #[must_use]
    pub fn current_generation(dir: impl AsRef<Path>) -> Option<String> {
        read_current_pointer(dir.as_ref()).ok().flatten()
    }

    /// Absolute path of the active generation directory (or package root for legacy).
    #[must_use]
    pub fn active_payload_dir(dir: impl AsRef<Path>) -> PathBuf {
        let root = dir.as_ref();
        if let Some(generation) = Self::current_generation(root) {
            let path = root.join(generation);
            if path.is_dir() {
                return path;
            }
        }
        root.to_path_buf()
    }
}

fn write_index_payload(index: &VisionIndex, dir: &Path) -> Result<(), MemoryError> {
    fs::write(dir.join(MANIFEST_FILE), index.header.to_json()?)
        .map_err(|error| MemoryError::Io(error.to_string()))?;

    write_tracks_cbor(
        dir.join(&index.header.track_stream_path),
        index.tracks.samples(),
    )?;
    write_masks_bin(
        dir.join(&index.header.mask_store_path),
        index.masks.entries(),
    )?;

    let snapshot = VisionIndexSnapshot::from_index(index);
    write_events_cbor(
        dir.join(events_cbor_name(&index.header.event_index_path)),
        &snapshot.events,
    )?;
    fs::write(
        dir.join(&index.header.entity_store_path),
        serde_json::to_string_pretty(&EntityFile {
            appearances: snapshot.appearances,
            visits: snapshot.visits,
            routes: snapshot.routes,
            zone_stays: snapshot.zone_stays,
            co_occurrences: snapshot.co_occurrences,
            source_transitions: snapshot.source_transitions,
            subjects: snapshot.subjects,
            patterns: snapshot.patterns,
            anomalies: snapshot.anomalies,
        })
        .map_err(|error| MemoryError::Serde(error.to_string()))?,
    )
    .map_err(|error| MemoryError::Io(error.to_string()))?;

    #[cfg(feature = "sqlite")]
    write_events_sqlite(&dir.join(&index.header.event_index_path), &snapshot.events)?;
    Ok(())
}

fn load_from_generation(dir: &Path) -> Result<VisionIndex, MemoryError> {
    let header = VisionIndexHeader::from_json(
        &fs::read_to_string(dir.join(MANIFEST_FILE))
            .map_err(|error| MemoryError::Io(error.to_string()))?,
    )?;

    let track_dtos: Vec<TrackSampleDto> = read_cbor(dir.join(&header.track_stream_path))?;
    let tracks = crate::TrackStream::from_samples(
        track_dtos
            .into_iter()
            .map(track_sample_from_dto)
            .collect::<Result<Vec<_>, _>>()?,
    );

    let mask_entries = read_masks_bin(dir.join(&header.mask_store_path))?;
    let masks = crate::MaskStore::from_entries(mask_entries);

    let event_dtos: Vec<EventEnvelopeDto> =
        read_cbor(dir.join(events_cbor_name(&header.event_index_path)))?;
    let entities: EntityFile = serde_json::from_str(
        &fs::read_to_string(dir.join(&header.entity_store_path))
            .map_err(|error| MemoryError::Io(error.to_string()))?,
    )
    .map_err(|error| MemoryError::Serde(error.to_string()))?;

    let mut index = VisionIndex::new(header.name.clone());
    index.header = header;
    index.tracks = tracks;
    index.masks = masks;
    let rebuild = VisionIndexSnapshot {
        header: index.header.clone(),
        tracks: index
            .tracks
            .samples()
            .iter()
            .copied()
            .map(Into::into)
            .collect(),
        events: event_dtos,
        appearances: entities.appearances,
        visits: entities.visits,
        routes: entities.routes,
        zone_stays: entities.zone_stays,
        co_occurrences: entities.co_occurrences,
        source_transitions: entities.source_transitions,
        subjects: entities.subjects,
        patterns: entities.patterns,
        anomalies: entities.anomalies,
    };
    apply_entities_from_snapshot(&mut index, &rebuild)?;
    apply_events_from_dtos(&mut index, &rebuild.events)?;
    index.validate_fast()?;
    Ok(index)
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct ChecksumsFile {
    algorithm: String,
    files: Vec<ChecksumEntry>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct ChecksumEntry {
    path: String,
    size: u64,
    fnv1a64: String,
}

fn compute_checksums(dir: &Path, index: &VisionIndex) -> Result<ChecksumsFile, MemoryError> {
    let mut files = Vec::new();
    #[allow(unused_mut)] // mutated when feature `sqlite` is enabled
    let mut relative = vec![
        MANIFEST_FILE.to_string(),
        index.header.track_stream_path.clone(),
        index.header.mask_store_path.clone(),
        events_cbor_name(&index.header.event_index_path)
            .to_string_lossy()
            .into_owned(),
        index.header.entity_store_path.clone(),
    ];
    #[cfg(feature = "sqlite")]
    relative.push(index.header.event_index_path.clone());
    for rel in relative {
        let path = dir.join(&rel);
        if !path.exists() {
            continue;
        }
        let bytes = fs::read(&path).map_err(|error| MemoryError::Io(error.to_string()))?;
        files.push(ChecksumEntry {
            path: rel,
            size: bytes.len() as u64,
            fnv1a64: format!("{:016x}", fnv1a64(&bytes)),
        });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(ChecksumsFile {
        algorithm: "fnv1a64".into(),
        files,
    })
}

fn verify_checksums(dir: &Path, checksums: &ChecksumsFile) -> Result<(), MemoryError> {
    for entry in &checksums.files {
        let path = dir.join(&entry.path);
        let bytes = fs::read(&path).map_err(|error| MemoryError::Io(error.to_string()))?;
        if bytes.len() as u64 != entry.size {
            return Err(MemoryError::Validation(format!(
                "checksum size mismatch for {}",
                entry.path
            )));
        }
        let digest = format!("{:016x}", fnv1a64(&bytes));
        if digest != entry.fnv1a64 {
            return Err(MemoryError::Validation(format!(
                "checksum mismatch for {}",
                entry.path
            )));
        }
    }
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn next_generation_id(root: &Path) -> Result<u64, MemoryError> {
    let mut max = 0_u64;
    if let Some(cur) = read_current_pointer(root)?
        && let Some(id) = parse_gen_id(cur.trim())
    {
        max = max.max(id);
    }
    let entries = fs::read_dir(root).map_err(|error| MemoryError::Io(error.to_string()))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(id) = parse_gen_id(&name) {
            max = max.max(id);
        }
    }
    Ok(max.saturating_add(1).max(1))
}

fn parse_gen_id(name: &str) -> Option<u64> {
    let name = name.strip_suffix(".tmp").unwrap_or(name);
    let rest = name.strip_prefix("gen-")?;
    rest.parse().ok()
}

fn read_current_pointer(root: &Path) -> Result<Option<String>, MemoryError> {
    let path = root.join(CURRENT_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|error| MemoryError::Io(error.to_string()))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}

fn write_current_pointer(root: &Path, gen_name: &str) -> Result<(), MemoryError> {
    let tmp = root.join("CURRENT.tmp");
    let final_path = root.join(CURRENT_FILE);
    fs::write(&tmp, format!("{gen_name}\n")).map_err(|error| MemoryError::Io(error.to_string()))?;
    fsync_path(&tmp)?;
    fs::rename(&tmp, &final_path).map_err(|error| MemoryError::Io(error.to_string()))?;
    fsync_path(root)?;
    Ok(())
}

fn prune_old_generations(root: &Path, keep: &str) -> Result<(), MemoryError> {
    let entries = fs::read_dir(root).map_err(|error| MemoryError::Io(error.to_string()))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == keep {
            continue;
        }
        if name.starts_with("gen-") {
            let path = entry.path();
            if path.is_dir() {
                let _ = fs::remove_dir_all(path);
            }
        }
    }
    Ok(())
}

fn write_json_atomic<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), MemoryError> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| MemoryError::Serde(error.to_string()))?;
    fs::write(path, text).map_err(|error| MemoryError::Io(error.to_string()))
}

fn fsync_path(path: &Path) -> Result<(), MemoryError> {
    // Directory fsync is best-effort: some platforms (notably Windows) deny
    // opening a directory handle for flush.
    if path.is_dir() {
        if let Ok(file) = fs::File::open(path) {
            let _ = file.sync_all();
        }
        return Ok(());
    }
    if path.exists() {
        let file = fs::File::open(path).map_err(|error| MemoryError::Io(error.to_string()))?;
        // On Windows, reopening read-only may still fail for some files; treat as best-effort.
        if let Err(error) = file.sync_all() {
            // Still require file durability when the OS allows it.
            #[cfg(not(target_os = "windows"))]
            return Err(MemoryError::Io(error.to_string()));
            #[cfg(target_os = "windows")]
            {
                let _ = error;
            }
        }
    }
    Ok(())
}

fn fsync_dir_tree(dir: &Path) -> Result<(), MemoryError> {
    let entries = fs::read_dir(dir).map_err(|error| MemoryError::Io(error.to_string()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            // Prefer syncing via a write-opened handle when possible.
            if let Ok(file) = fs::OpenOptions::new().write(true).open(&path) {
                let _ = file.sync_all();
            } else {
                fsync_path(&path)?;
            }
        }
    }
    fsync_path(dir)
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct EntityFile {
    appearances: Vec<AppearanceDto>,
    visits: Vec<VisitDto>,
    routes: Vec<RouteDto>,
    zone_stays: Vec<ZoneStayDto>,
    co_occurrences: Vec<CoOccurrenceDto>,
    source_transitions: Vec<SourceTransitionDto>,
    subjects: Vec<SubjectProfileDto>,
    patterns: Vec<PatternRecordDto>,
    anomalies: Vec<AnomalyEventDto>,
}

fn events_cbor_name(event_index_path: &str) -> PathBuf {
    // Prefer a dedicated CBOR stream name; SQLite keeps the configured path.
    let path = Path::new(event_index_path);
    let use_default = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("sqlite") || ext.eq_ignore_ascii_case("idx"));
    if use_default {
        PathBuf::from("events.cbor")
    } else {
        PathBuf::from(event_index_path)
    }
}

fn write_tracks_cbor(path: PathBuf, samples: &[TrackSample]) -> Result<(), MemoryError> {
    let dtos: Vec<TrackSampleDto> = samples.iter().copied().map(Into::into).collect();
    write_cbor(path, &dtos)
}

fn write_events_cbor(path: PathBuf, events: &[EventEnvelopeDto]) -> Result<(), MemoryError> {
    write_cbor(path, &events)
}

fn write_cbor<T: serde::Serialize>(path: PathBuf, value: &T) -> Result<(), MemoryError> {
    let mut file = fs::File::create(path).map_err(|error| MemoryError::Io(error.to_string()))?;
    ciborium::into_writer(value, &mut file).map_err(|error| MemoryError::Serde(error.to_string()))
}

fn read_cbor<T: serde::de::DeserializeOwned>(path: PathBuf) -> Result<T, MemoryError> {
    let mut file = fs::File::open(path).map_err(|error| MemoryError::Io(error.to_string()))?;
    ciborium::from_reader(&mut file).map_err(|error| MemoryError::Serde(error.to_string()))
}

/// Masks binary format: repeated records of
/// `u64 handle | u64 len | len bytes`.
fn write_masks_bin(path: PathBuf, entries: &[(MaskRef, Vec<u8>)]) -> Result<(), MemoryError> {
    let mut file = fs::File::create(path).map_err(|error| MemoryError::Io(error.to_string()))?;
    for (handle, bytes) in entries {
        file.write_all(&handle.0.to_le_bytes())
            .map_err(|error| MemoryError::Io(error.to_string()))?;
        file.write_all(&(bytes.len() as u64).to_le_bytes())
            .map_err(|error| MemoryError::Io(error.to_string()))?;
        file.write_all(bytes)
            .map_err(|error| MemoryError::Io(error.to_string()))?;
    }
    Ok(())
}

fn read_masks_bin(path: PathBuf) -> Result<Vec<(MaskRef, Vec<u8>)>, MemoryError> {
    let mut file = fs::File::open(path).map_err(|error| MemoryError::Io(error.to_string()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|error| MemoryError::Io(error.to_string()))?;
    let mut out = Vec::new();
    let mut offset = 0_usize;
    while offset + 16 <= buf.len() {
        let handle = u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let len = u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        if offset + len > buf.len() {
            return Err(MemoryError::Invalid);
        }
        out.push((MaskRef(handle), buf[offset..offset + len].to_vec()));
        offset += len;
    }
    if offset != buf.len() {
        return Err(MemoryError::Invalid);
    }
    Ok(out)
}

fn track_sample_from_dto(dto: TrackSampleDto) -> Result<TrackSample, MemoryError> {
    let pts = MediaTime::new(dto.pts.ticks, dto.pts.timescale).map_err(|_| MemoryError::Invalid)?;
    Ok(TrackSample {
        sample_id: dto.sample_id,
        supersedes: dto.supersedes,
        revision: dto.revision,
        idempotency_key: dto.idempotency_key,
        source_id: SourceId(dto.source_id),
        frame_index: dto.frame_index,
        pts,
        track_id: TrackId(dto.track_id),
        track_uid: dto.track_uid.map(sightloom_core::TrackUid),
        subject_id: dto.subject_id.map(SubjectId),
        class_id: dto.class_id.map(ClassId),
        left: dto.left,
        top: dto.top,
        right: dto.right,
        bottom: dto.bottom,
        confidence: dto.confidence,
        mask_ref: dto.mask_ref,
    })
}

fn apply_entities_from_snapshot(
    index: &mut VisionIndex,
    snapshot: &VisionIndexSnapshot,
) -> Result<(), MemoryError> {
    // Entities are stored as DTOs in snapshot; convert via JSON for maintainability.
    let json =
        serde_json::to_string(snapshot).map_err(|error| MemoryError::Serde(error.to_string()))?;
    let restored: VisionIndexSnapshot =
        serde_json::from_str(&json).map_err(|error| MemoryError::Serde(error.to_string()))?;
    // DTO-only fields: map the subset we can reconstruct without reverse From impls
    // by keeping typed vectors empty unless we add reverse mappers later.
    // For package round-trip fidelity of entity *counts* and queryable data,
    // rebuild from DTO using dedicated helpers below.
    index.appearances = restored
        .appearances
        .iter()
        .copied()
        .map(appearance_from_dto)
        .collect::<Result<Vec<_>, _>>()?;
    index.visits = restored
        .visits
        .iter()
        .copied()
        .map(visit_from_dto)
        .collect::<Result<Vec<_>, _>>()?;
    index.subjects = restored
        .subjects
        .iter()
        .cloned()
        .map(subject_from_dto)
        .collect::<Result<Vec<_>, _>>()?;
    // Routes/zone stays/co-occurrence/patterns/anomalies keep DTO fidelity via empty-safe mapping.
    index.routes = restored
        .routes
        .iter()
        .cloned()
        .map(route_from_dto)
        .collect::<Result<Vec<_>, _>>()?;
    index.zone_stays = restored
        .zone_stays
        .iter()
        .copied()
        .map(zone_stay_from_dto)
        .collect::<Result<Vec<_>, _>>()?;
    index.co_occurrences = restored
        .co_occurrences
        .iter()
        .copied()
        .map(co_occurrence_from_dto)
        .collect::<Result<Vec<_>, _>>()?;
    index.source_transitions = restored
        .source_transitions
        .iter()
        .copied()
        .map(source_transition_from_dto)
        .collect::<Result<Vec<_>, _>>()?;
    index.patterns = restored
        .patterns
        .iter()
        .cloned()
        .map(pattern_from_dto)
        .collect();
    index.anomalies = restored
        .anomalies
        .iter()
        .cloned()
        .map(anomaly_from_dto)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

fn apply_events_from_dtos(
    index: &mut VisionIndex,
    events: &[EventEnvelopeDto],
) -> Result<(), MemoryError> {
    use sightloom_core::{
        Direction, EventEnvelope, EventId, EventKind, EventPayload, EvidenceRef, FrameStamp, ZoneId,
    };
    index.events.clear();
    index.event_index = crate::EventIndex::new();
    for dto in events {
        let pts = MediaTime::new(dto.stamp.pts.ticks, dto.stamp.pts.timescale)
            .map_err(|_| MemoryError::Invalid)?;
        let stamp = FrameStamp::new(
            SourceId(dto.stamp.source_id),
            dto.stamp.frame_index,
            pts,
            dto.stamp.wall_clock_ns,
        );
        let kind = match dto.kind.as_str() {
            "zone" => EventKind::Zone,
            "dwell" => EventKind::Dwell,
            "occupancy" => EventKind::Occupancy,
            "identity" => EventKind::Identity,
            "pattern" => EventKind::Pattern,
            "anomaly" => EventKind::Anomaly,
            _ => EventKind::Custom,
        };
        let payload = match dto.payload_kind.as_str() {
            "entered" => EventPayload::Entered {
                zone_id: ZoneId(dto.payload_zone_id.unwrap_or(0)),
                class_id: dto.payload_class_id.map(ClassId),
            },
            "exited" => EventPayload::Exited {
                zone_id: ZoneId(dto.payload_zone_id.unwrap_or(0)),
                class_id: dto.payload_class_id.map(ClassId),
            },
            "crossed" => EventPayload::Crossed {
                zone_id: ZoneId(dto.payload_zone_id.unwrap_or(0)),
                direction: match dto.payload_direction.as_deref() {
                    Some("rtl") => Direction::RightToLeft,
                    _ => Direction::LeftToRight,
                },
            },
            "dwell_started" => EventPayload::DwellStarted {
                zone_id: ZoneId(dto.payload_zone_id.unwrap_or(0)),
            },
            "dwell_ended" => EventPayload::DwellEnded {
                zone_id: ZoneId(dto.payload_zone_id.unwrap_or(0)),
                duration_ns: dto.payload_duration_ns.unwrap_or(0),
                visit_count: dto.payload_visit_count.unwrap_or(0),
            },
            "occupancy" => EventPayload::Occupancy {
                zone_id: ZoneId(dto.payload_zone_id.unwrap_or(0)),
                occupancy: dto.payload_occupancy.unwrap_or(0),
            },
            "metrics" => EventPayload::Metrics {
                score: dto.payload_score.unwrap_or(0.0),
                aux: dto.payload_aux.unwrap_or(0.0),
                tag: dto.payload_tag.unwrap_or(0),
            },
            _ => EventPayload::Empty,
        };
        let mut envelope =
            EventEnvelope::new(EventId(dto.event_id), stamp, kind).with_payload(payload);
        if let Some(track_id) = dto.track_id {
            envelope = envelope.with_track(TrackId(track_id));
        }
        if let Some(subject_id) = dto.subject_id {
            envelope = envelope.with_subject(SubjectId(subject_id));
        }
        if let Some(zone_id) = dto.zone_id {
            envelope = envelope.with_zone(ZoneId(zone_id));
        }
        if let Some(evidence) = dto.evidence {
            envelope = envelope.with_evidence(EvidenceRef(evidence));
        }
        index.push_event(envelope);
    }
    Ok(())
}

fn media_from_dto(dto: MediaTimeDto) -> Result<MediaTime, MemoryError> {
    MediaTime::new(dto.ticks, dto.timescale).map_err(|_| MemoryError::Invalid)
}

fn appearance_from_dto(dto: AppearanceDto) -> Result<crate::Appearance, MemoryError> {
    Ok(crate::Appearance {
        appearance_id: sightloom_core::AppearanceId(dto.appearance_id),
        subject_id: dto.subject_id.map(SubjectId),
        track_id: dto.track_id.map(TrackId),
        source_id: SourceId(dto.source_id),
        start: media_from_dto(dto.start)?,
        end: media_from_dto(dto.end)?,
        class_id: dto.class_id.map(ClassId),
        peak_confidence: dto.peak_confidence,
        evidence: dto.evidence.map(sightloom_core::EvidenceRef),
    })
}

fn visit_from_dto(dto: VisitDto) -> Result<crate::Visit, MemoryError> {
    Ok(crate::Visit {
        visit_id: sightloom_core::VisitId(dto.visit_id),
        subject_id: dto.subject_id.map(SubjectId),
        start: media_from_dto(dto.start)?,
        end: media_from_dto(dto.end)?,
        source_count: dto.source_count,
        duration_ns: dto.duration_ns,
    })
}

fn subject_from_dto(dto: SubjectProfileDto) -> Result<crate::SubjectProfile, MemoryError> {
    Ok(crate::SubjectProfile {
        subject_id: SubjectId(dto.subject_id),
        label: dto.label,
        appearance_count: dto.appearance_count,
        source_count: dto.source_count,
        total_duration_ns: dto.total_duration_ns,
        first_seen: dto.first_seen.map(media_from_dto).transpose()?,
        last_seen: dto.last_seen.map(media_from_dto).transpose()?,
        embedding: dto.embedding.map(sightloom_core::EmbeddingRef),
    })
}

fn route_from_dto(dto: RouteDto) -> Result<crate::Route, MemoryError> {
    Ok(crate::Route {
        subject_id: SubjectId(dto.subject_id),
        zones: dto.zones.into_iter().map(sightloom_core::ZoneId).collect(),
        sources: dto.sources.into_iter().map(SourceId).collect(),
        start: media_from_dto(dto.start)?,
        end: media_from_dto(dto.end)?,
    })
}

fn zone_stay_from_dto(dto: ZoneStayDto) -> Result<crate::ZoneStay, MemoryError> {
    Ok(crate::ZoneStay {
        zone_id: sightloom_core::ZoneId(dto.zone_id),
        subject_id: dto.subject_id.map(SubjectId),
        track_id: dto.track_id.map(TrackId),
        start: media_from_dto(dto.start)?,
        end: media_from_dto(dto.end)?,
        duration_ns: dto.duration_ns,
    })
}

fn co_occurrence_from_dto(dto: CoOccurrenceDto) -> Result<crate::CoOccurrence, MemoryError> {
    Ok(crate::CoOccurrence {
        subject_a: SubjectId(dto.subject_a),
        subject_b: SubjectId(dto.subject_b),
        source_id: dto.source_id.map(SourceId),
        start: media_from_dto(dto.start)?,
        end: media_from_dto(dto.end)?,
        overlap_ns: dto.overlap_ns,
    })
}

fn source_transition_from_dto(
    dto: SourceTransitionDto,
) -> Result<crate::SourceTransition, MemoryError> {
    Ok(crate::SourceTransition {
        subject_id: SubjectId(dto.subject_id),
        from_source: SourceId(dto.from_source),
        to_source: SourceId(dto.to_source),
        at: media_from_dto(dto.at)?,
        evidence: dto.evidence.map(sightloom_core::EvidenceRef),
    })
}

fn pattern_from_dto(dto: PatternRecordDto) -> sightloom_analysis::PatternRecord {
    use sightloom_analysis::PatternKind;
    let kind = match dto.kind.as_str() {
        "time_of_day" => PatternKind::TimeOfDay,
        "day_of_week" => PatternKind::DayOfWeek,
        "visit_periodicity" => PatternKind::VisitPeriodicity,
        "dwell_distribution" => PatternKind::DwellDistribution,
        "route_sequence" => PatternKind::RouteSequence,
        "co_occurrence" => PatternKind::CoOccurrence,
        "event_before_event" => PatternKind::EventBeforeEvent,
        "expected_absence" => PatternKind::ExpectedAbsence,
        "group_formation" => PatternKind::GroupFormation,
        _ => PatternKind::Custom,
    };
    sightloom_analysis::PatternRecord {
        pattern_id: sightloom_core::PatternId(dto.pattern_id),
        kind,
        subject_id: dto.subject_id.map(SubjectId),
        confidence: dto.confidence,
        evidence_events: dto
            .evidence_events
            .into_iter()
            .map(sightloom_core::EventId)
            .collect(),
        tag: dto.tag,
    }
}

fn anomaly_from_dto(dto: AnomalyEventDto) -> Result<sightloom_analysis::AnomalyEvent, MemoryError> {
    use sightloom_analysis::{AnomalyReason, Severity};
    let severity = match dto.severity.as_str() {
        "medium" => Severity::Medium,
        "high" => Severity::High,
        "critical" => Severity::Critical,
        _ => Severity::Low,
    };
    let reasons = dto
        .reasons
        .iter()
        .map(|reason| match reason.as_str() {
            "unusual_appearance_time" => AnomalyReason::UnusualAppearanceTime,
            "unusual_frequency" => AnomalyReason::UnusualFrequency,
            "unusual_dwell" => AnomalyReason::UnusualDwell,
            "unusual_route" => AnomalyReason::UnusualRoute,
            "unusual_co_occurrence" => AnomalyReason::UnusualCoOccurrence,
            "missing_expected_appearance" => AnomalyReason::MissingExpectedAppearance,
            "sudden_behaviour_change" => AnomalyReason::SuddenBehaviourChange,
            other if other.starts_with("custom:") => {
                let code = other.trim_start_matches("custom:").parse().unwrap_or(0);
                AnomalyReason::Custom(code)
            }
            _ => AnomalyReason::Custom(0),
        })
        .collect();
    Ok(sightloom_analysis::AnomalyEvent {
        anomaly_id: sightloom_core::AnomalyId(dto.anomaly_id),
        score: dto.score,
        severity,
        reasons,
        evidence: dto
            .evidence
            .into_iter()
            .map(sightloom_core::EventId)
            .collect(),
        subject_id: dto.subject_id.map(SubjectId),
        source_id: dto.source_id.map(SourceId),
        at: media_from_dto(dto.at)?,
    })
}

#[cfg(feature = "sqlite")]
fn write_events_sqlite(path: &Path, events: &[EventEnvelopeDto]) -> Result<(), MemoryError> {
    use rusqlite::{Connection, params};
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| MemoryError::Io(error.to_string()))?;
    }
    let _ = fs::remove_file(path);
    let conn = Connection::open(path).map_err(|error| MemoryError::Io(error.to_string()))?;
    conn.execute_batch(
        "CREATE TABLE events (
            event_id INTEGER PRIMARY KEY,
            kind TEXT NOT NULL,
            track_id INTEGER,
            subject_id INTEGER,
            zone_id INTEGER,
            source_id INTEGER,
            frame_index INTEGER,
            pts_ticks INTEGER,
            pts_timescale INTEGER,
            payload_kind TEXT NOT NULL
        );
        CREATE INDEX idx_events_subject ON events(subject_id);
        CREATE INDEX idx_events_track ON events(track_id);
        CREATE INDEX idx_events_kind ON events(kind);",
    )
    .map_err(|error| MemoryError::Io(error.to_string()))?;
    {
        let mut stmt = conn
            .prepare(
                "INSERT INTO events (
                    event_id, kind, track_id, subject_id, zone_id, source_id,
                    frame_index, pts_ticks, pts_timescale, payload_kind
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )
            .map_err(|error| MemoryError::Io(error.to_string()))?;
        for event in events {
            stmt.execute(params![
                i64::try_from(event.event_id).unwrap_or(i64::MAX),
                event.kind.as_str(),
                event.track_id.map(i64::from),
                event
                    .subject_id
                    .map(|v| i64::try_from(v).unwrap_or(i64::MAX)),
                event.zone_id.map(i64::from),
                i64::from(event.stamp.source_id),
                i64::try_from(event.stamp.frame_index).unwrap_or(i64::MAX),
                event.stamp.pts.ticks,
                event.stamp.pts.timescale,
                event.payload_kind.as_str(),
            ])
            .map_err(|error| MemoryError::Io(error.to_string()))?;
        }
    }
    Ok(())
}

/// Query helpers over a saved `SQLite` event index.
#[cfg(feature = "sqlite")]
pub mod sqlite_query {
    use super::MemoryError;
    use rusqlite::{Connection, OptionalExtension};
    use std::path::Path;

    /// Counts events for a subject id in an `events.sqlite` database.
    ///
    /// # Errors
    ///
    /// Returns I/O errors when the database cannot be opened or queried.
    pub fn count_events_for_subject(
        db_path: impl AsRef<Path>,
        subject_id: u64,
    ) -> Result<u64, MemoryError> {
        let conn = Connection::open(db_path.as_ref())
            .map_err(|error| MemoryError::Io(error.to_string()))?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM events WHERE subject_id = ?1",
                [i64::try_from(subject_id).unwrap_or(i64::MAX)],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| MemoryError::Io(error.to_string()))?
            .unwrap_or(0);
        Ok(u64::try_from(count).unwrap_or(0))
    }
}
