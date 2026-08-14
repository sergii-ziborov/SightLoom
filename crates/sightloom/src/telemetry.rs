//! Host telemetry adapters (Prometheus text + OTLP-shaped JSON).
//!
//! `SightLoom` does **not** depend on the OpenTelemetry SDK. Hosts implement
//! [`MetricsExporter`] / [`SpanExporter`] or scrape Prometheus /
//! OTLP-shaped JSON helpers.
#![allow(clippy::cast_precision_loss, clippy::format_push_string)]

use crate::ingest::{IngestMetrics, prometheus_text};

/// One counter/gauge sample in a backend-neutral form.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricPoint {
    /// Metric name (e.g. `sightloom.ingest.accepted`).
    pub name: String,
    /// `counter` or `gauge`.
    pub kind: MetricKind,
    /// Value.
    pub value: f64,
    /// Optional unit (e.g. `1`, `ns`).
    pub unit: &'static str,
    /// Labels as `key=value` pairs.
    pub attributes: Vec<(String, String)>,
}

/// Metric kind for exporters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetricKind {
    /// Monotonic counter.
    Counter,
    /// Instantaneous gauge.
    Gauge,
}

/// Lightweight span-like event for host tracing bridges.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanEvent {
    /// Span name.
    pub name: String,
    /// Trace id hex (host-assigned; empty if none).
    pub trace_id: String,
    /// Span id hex.
    pub span_id: String,
    /// Start time unix nanoseconds (0 if unknown).
    pub start_unix_ns: u64,
    /// End time unix nanoseconds (0 if unknown).
    pub end_unix_ns: u64,
    /// Status: `ok` / `error`.
    pub status: SpanStatus,
    /// Attributes.
    pub attributes: Vec<(String, String)>,
}

/// Span completion status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpanStatus {
    /// Success.
    Ok,
    /// Error.
    Error,
}

/// Host metrics sink (Prometheus remote-write, OpenTelemetry SDK, log, …).
pub trait MetricsExporter {
    /// Export error type.
    type Error: core::fmt::Debug;

    /// Pushes a batch of metric points.
    ///
    /// # Errors
    ///
    /// Backend-defined export failures.
    fn export_metrics(&mut self, points: &[MetricPoint]) -> Result<(), Self::Error>;
}

/// Host span/trace sink.
pub trait SpanExporter {
    /// Export error type.
    type Error: core::fmt::Debug;

    /// Pushes span events.
    ///
    /// # Errors
    ///
    /// Backend-defined export failures.
    fn export_spans(&mut self, spans: &[SpanEvent]) -> Result<(), Self::Error>;
}

/// Converts [`IngestMetrics`] into neutral metric points.
#[must_use]
pub fn ingest_metric_points(session: &str, metrics: &IngestMetrics) -> Vec<MetricPoint> {
    let sess = session.to_string();
    let attr = |k: &str, v: &str| (k.to_string(), v.to_string());
    vec![
        MetricPoint {
            name: "sightloom.ingest.accepted".into(),
            kind: MetricKind::Counter,
            value: metrics.accepted as f64,
            unit: "1",
            attributes: vec![attr("session", &sess)],
        },
        MetricPoint {
            name: "sightloom.ingest.dropped".into(),
            kind: MetricKind::Counter,
            value: metrics.dropped as f64,
            unit: "1",
            attributes: vec![attr("session", &sess)],
        },
        MetricPoint {
            name: "sightloom.ingest.rejected_late".into(),
            kind: MetricKind::Counter,
            value: metrics.rejected_late as f64,
            unit: "1",
            attributes: vec![attr("session", &sess)],
        },
        MetricPoint {
            name: "sightloom.ingest.rejected_ooo".into(),
            kind: MetricKind::Counter,
            value: metrics.rejected_ooo as f64,
            unit: "1",
            attributes: vec![attr("session", &sess)],
        },
        MetricPoint {
            name: "sightloom.ingest.queue_hwm".into(),
            kind: MetricKind::Gauge,
            value: metrics.queue_hwm as f64,
            unit: "1",
            attributes: vec![attr("session", &sess)],
        },
        MetricPoint {
            name: "sightloom.ingest.source_resets".into(),
            kind: MetricKind::Counter,
            value: metrics.source_resets as f64,
            unit: "1",
            attributes: vec![attr("session", &sess)],
        },
        MetricPoint {
            name: "sightloom.ingest.checkpoints".into(),
            kind: MetricKind::Counter,
            value: metrics.checkpoints as f64,
            unit: "1",
            attributes: vec![attr("session", &sess)],
        },
    ]
}

/// Prometheus text (delegates to ingest helper).
#[must_use]
pub fn export_prometheus(namespace: &str, session: &str, metrics: &IngestMetrics) -> String {
    prometheus_text(namespace, session, metrics)
}

/// OTLP-shaped metrics JSON (resource + scopeMetrics) for host collectors.
///
/// This is **not** a full OTLP protobuf encoder — hosts can POST this JSON to
/// a collector or map into a real OpenTelemetry SDK.
#[must_use]
pub fn otlp_metrics_json(session: &str, metrics: &IngestMetrics) -> String {
    let points = ingest_metric_points(session, metrics);
    let mut metrics_json = String::from("[");
    for (i, p) in points.iter().enumerate() {
        if i > 0 {
            metrics_json.push(',');
        }
        let kind = match p.kind {
            MetricKind::Counter => "sum",
            MetricKind::Gauge => "gauge",
        };
        let attrs: String = p
            .attributes
            .iter()
            .map(|(k, v)| {
                format!(
                    r#"{{"key":"{}","value":{{"stringValue":"{}"}}}}"#,
                    escape_json(k),
                    escape_json(v)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        metrics_json.push_str(&format!(
            r#"{{"name":"{}","unit":"{}","{kind}":{{"dataPoints":[{{"asDouble":{},"attributes":[{attrs}]}}]}}}}"#,
            escape_json(&p.name),
            p.unit,
            p.value
        ));
    }
    metrics_json.push(']');
    format!(
        r#"{{"resourceMetrics":[{{"resource":{{"attributes":[{{"key":"service.name","value":{{"stringValue":"sightloom"}}}}]}},"scopeMetrics":[{{"scope":{{"name":"sightloom"}},"metrics":{metrics_json}}}]}}]}}"#
    )
}

/// Builds a span for one ingest frame (host fills timestamps / trace ids).
#[must_use]
pub fn ingest_frame_span(
    source_id: u32,
    frame_index: u64,
    accepted: bool,
    start_unix_ns: u64,
    end_unix_ns: u64,
) -> SpanEvent {
    SpanEvent {
        name: "sightloom.ingest.frame".into(),
        trace_id: String::new(),
        span_id: String::new(),
        start_unix_ns,
        end_unix_ns,
        status: if accepted {
            SpanStatus::Ok
        } else {
            SpanStatus::Error
        },
        attributes: vec![
            ("source_id".into(), source_id.to_string()),
            ("frame_index".into(), frame_index.to_string()),
            ("accepted".into(), accepted.to_string()),
        ],
    }
}

/// Serializes spans to a JSON array (host OpenTelemetry bridge).
#[must_use]
pub fn spans_to_json(spans: &[SpanEvent]) -> String {
    let mut out = String::from("[");
    for (i, s) in spans.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let status = match s.status {
            SpanStatus::Ok => "STATUS_CODE_OK",
            SpanStatus::Error => "STATUS_CODE_ERROR",
        };
        let attrs: String = s
            .attributes
            .iter()
            .map(|(k, v)| {
                format!(
                    r#"{{"key":"{}","value":{{"stringValue":"{}"}}}}"#,
                    escape_json(k),
                    escape_json(v)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        out.push_str(&format!(
            r#"{{"name":"{}","traceId":"{}","spanId":"{}","startTimeUnixNano":"{}","endTimeUnixNano":"{}","status":{{"code":"{status}"}},"attributes":[{attrs}]}}"#,
            escape_json(&s.name),
            escape_json(&s.trace_id),
            escape_json(&s.span_id),
            s.start_unix_ns,
            s.end_unix_ns
        ));
    }
    out.push(']');
    out
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// No-op exporter for tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullMetricsExporter;

impl MetricsExporter for NullMetricsExporter {
    type Error = ();

    fn export_metrics(&mut self, _points: &[MetricPoint]) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Collects metrics into an in-memory buffer (tests / host capture).
#[derive(Clone, Debug, Default)]
pub struct BufferMetricsExporter {
    /// Captured points.
    pub points: Vec<MetricPoint>,
}

impl MetricsExporter for BufferMetricsExporter {
    type Error = ();

    fn export_metrics(&mut self, points: &[MetricPoint]) -> Result<(), Self::Error> {
        self.points.extend_from_slice(points);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::IngestMetrics;

    #[test]
    fn otlp_json_contains_counters() {
        let m = IngestMetrics {
            accepted: 3,
            dropped: 1,
            ..IngestMetrics::default()
        };
        let j = otlp_metrics_json("demo", &m);
        assert!(j.contains("sightloom.ingest.accepted"));
        assert!(j.contains("resourceMetrics"));
        assert!(j.contains("3"));
    }

    #[test]
    fn buffer_exporter_records() {
        let m = IngestMetrics {
            accepted: 2,
            ..IngestMetrics::default()
        };
        let mut buf = BufferMetricsExporter::default();
        let points = ingest_metric_points("s", &m);
        buf.export_metrics(&points).unwrap();
        assert_eq!(buf.points.len(), 7);
    }
}
