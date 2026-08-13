//! Streaming / incremental subject query cursor over a live [`VisionIndex`].
//!
//! Hosts poll after ingest batches to page results and to discover subjects
//! that became matching since the last poll. Not a continuous push subscription
//! or multi-node query mesh.

use crate::{Page, SubjectHit, SubjectQuery, VisionIndex, execute_subject_query};
use sightloom_core::SubjectId;

/// Cursor state for paginated + incremental subject queries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueryCursor {
    /// Next page offset into the full match list.
    pub offset: usize,
    /// Page size (`0` = all remaining).
    pub limit: usize,
    /// Subjects already returned by this stream (for `poll_new`).
    pub delivered: Vec<SubjectId>,
    /// Track sample count watermark when the cursor was last advanced.
    pub sample_watermark: usize,
}

impl QueryCursor {
    /// Starts at offset 0 with the given page size.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            offset: 0,
            limit,
            delivered: Vec::new(),
            sample_watermark: 0,
        }
    }
}

/// Streaming subject query: holds a filter + cursor.
#[derive(Clone, Debug)]
pub struct StreamingSubjectQuery {
    /// Underlying filter (page fields are overwritten by the cursor).
    pub filter: SubjectQuery,
    /// Pagination / incremental state.
    pub cursor: QueryCursor,
}

impl StreamingSubjectQuery {
    /// Creates a stream from a filter and page size.
    #[must_use]
    pub fn new(filter: SubjectQuery, page_size: usize) -> Self {
        Self {
            filter,
            cursor: QueryCursor::new(page_size),
        }
    }

    /// Fetches the next page of matches and advances the cursor.
    pub fn next_page(&mut self, index: &VisionIndex) -> Vec<SubjectHit> {
        let mut q = self.filter.clone();
        q.page = Page {
            offset: self.cursor.offset,
            limit: self.cursor.limit,
        };
        let hits = execute_subject_query(index, &q);
        for h in &hits {
            if !self.cursor.delivered.contains(&h.subject_id) {
                self.cursor.delivered.push(h.subject_id);
            }
        }
        let n = hits.len();
        if self.cursor.limit == 0 {
            self.cursor.offset = self.cursor.offset.saturating_add(n);
        } else {
            self.cursor.offset = self.cursor.offset.saturating_add(self.cursor.limit);
        }
        self.cursor.sample_watermark = index.tracks.samples().len();
        hits
    }

    /// Returns matching subjects **not yet delivered**, if the index grew
    /// (or always re-eval when `force`).
    ///
    /// Useful after ingest: poll for newly matching subjects without replaying
    /// the entire result set.
    pub fn poll_new(&mut self, index: &VisionIndex, force: bool) -> Vec<SubjectHit> {
        let samples_now = index.tracks.samples().len();
        if !force
            && samples_now <= self.cursor.sample_watermark
            && !self.cursor.delivered.is_empty()
        {
            // No growth and we already have a baseline — skip.
            // Still allow first poll when delivered is empty.
            if self.cursor.sample_watermark > 0 {
                return Vec::new();
            }
        }
        let mut q = self.filter.clone();
        q.page = Page::default(); // full filter match
        let all = execute_subject_query(index, &q);
        let mut fresh = Vec::new();
        for h in all {
            if !self.cursor.delivered.contains(&h.subject_id) {
                self.cursor.delivered.push(h.subject_id);
                fresh.push(h);
            }
        }
        self.cursor.sample_watermark = samples_now;
        fresh
    }

    /// Resets pagination and delivered set (restarts the stream).
    pub fn reset(&mut self) {
        self.cursor.offset = 0;
        self.cursor.delivered.clear();
        self.cursor.sample_watermark = 0;
    }

    /// Number of distinct subjects delivered so far.
    #[must_use]
    pub fn delivered_count(&self) -> usize {
        self.cursor.delivered.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TrackSample, VisionIndex};
    use sightloom_core::{MediaTime, SourceId, SubjectId, TrackId};

    fn sample(subject: u64, frame: u64) -> TrackSample {
        TrackSample {
            sample_id: 0,
            supersedes: None,
            revision: 0,
            idempotency_key: 0,
            source_id: SourceId(1),
            frame_index: frame,
            pts: MediaTime::new(frame as i64, 1).unwrap(),
            track_id: TrackId(1),
            track_uid: None,
            subject_id: Some(SubjectId(subject)),
            class_id: None,
            left: 0.0,
            top: 0.0,
            right: 1.0,
            bottom: 1.0,
            confidence: 0.9,
            mask_ref: 0,
        }
    }

    #[test]
    fn pages_and_polls_new_subjects() {
        let mut index = VisionIndex::new("stream");
        index.push_track(sample(1, 0));
        index.push_track(sample(2, 1));
        index.push_track(sample(3, 2));

        let mut stream = StreamingSubjectQuery::new(SubjectQuery::new().seen_on(SourceId(1)), 2);
        let p1 = stream.next_page(&index);
        assert_eq!(p1.len(), 2);
        let p2 = stream.next_page(&index);
        assert_eq!(p2.len(), 1);
        assert_eq!(stream.delivered_count(), 3);

        // No new samples → empty poll.
        let none = stream.poll_new(&index, false);
        assert!(none.is_empty());

        index.push_track(sample(4, 3));
        let fresh = stream.poll_new(&index, false);
        assert_eq!(fresh.len(), 1);
        assert_eq!(fresh[0].subject_id, SubjectId(4));
    }
}
