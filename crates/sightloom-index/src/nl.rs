//! Deterministic natural-language → query bridge (no LLM).
//!
//! Parses a small English keyword grammar into [`crate::QueryNode`] trees.
//! Unknown phrases become warnings; hard failures return [`NlParseError`].
//!
//! Supported fragments (combinable with `and` / `or`):
//! - `seen on source N` / `on camera N` / `source N`
//! - `in zone N` / `zone N`
//! - `subject N` / `subject N,M`
//! - `min confidence F` / `confidence >= F`
//! - `min dwell Ns` / `dwell >= Ns` (nanoseconds integer)
//! - `during START END` (media times as integer nanoseconds)
//! - `then zone A then zone B` / `then A then B within Ns`
//! - `route zone A B C` (ordered subsequence)

use crate::{QueryNode, SpatialQuery, SubjectPredicate, ThenSeenIn};
use sightloom_core::{MediaTime, SourceId, SubjectId, ZoneId};

/// NL parse failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NlParseError {
    /// Empty or whitespace-only input.
    Empty,
    /// Could not extract any known predicate.
    NoPredicates,
}

/// Successful parse with optional non-fatal warnings.
#[derive(Clone, Debug, PartialEq)]
pub struct NlParseResult {
    /// Query tree.
    pub node: QueryNode,
    /// Soft issues (unknown tokens, ignored words).
    pub warnings: Vec<String>,
}

/// Parses a restricted English query string into a [`QueryNode`].
///
/// Multiple clauses joined by `and` become `QueryNode::And`; by `or` become
/// `QueryNode::Or`. Parentheses are not supported in this foundation.
///
/// # Errors
///
/// Returns [`NlParseError`] when nothing usable can be parsed.
pub fn parse_nl_query(text: &str) -> Result<NlParseResult, NlParseError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(NlParseError::Empty);
    }
    let lower = trimmed.to_ascii_lowercase();

    // Split top-level and/or (prefer or, then and — simple left-to-right).
    if let Some(parts) = split_top_level(&lower, " or ") {
        let mut children = Vec::new();
        let mut warnings = Vec::new();
        for part in parts {
            let r = parse_nl_query(part)?;
            children.push(r.node);
            warnings.extend(r.warnings);
        }
        return Ok(NlParseResult {
            node: QueryNode::or(children),
            warnings,
        });
    }
    if let Some(parts) = split_top_level(&lower, " and ") {
        let mut children = Vec::new();
        let mut warnings = Vec::new();
        for part in parts {
            let r = parse_nl_query(part)?;
            children.push(r.node);
            warnings.extend(r.warnings);
        }
        return Ok(NlParseResult {
            node: QueryNode::and(children),
            warnings,
        });
    }

    parse_clause(&lower)
}

fn split_top_level<'a>(text: &'a str, sep: &str) -> Option<Vec<&'a str>> {
    if !text.contains(sep) {
        return None;
    }
    let parts: Vec<&str> = text
        .split(sep)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() < 2 { None } else { Some(parts) }
}

#[allow(clippy::too_many_lines)]
fn parse_clause(lower: &str) -> Result<NlParseResult, NlParseError> {
    let mut preds: Vec<SubjectPredicate> = Vec::new();
    let mut warnings = Vec::new();
    let tokens: Vec<&str> = lower.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(NlParseError::Empty);
    }

    let mut i = 0;
    while i < tokens.len() {
        // seen on source N | on camera N | source N
        if (tokens[i] == "seen"
            && i + 3 < tokens.len()
            && tokens[i + 1] == "on"
            && (tokens[i + 2] == "source" || tokens[i + 2] == "camera"))
            || (tokens[i] == "on"
                && i + 2 < tokens.len()
                && (tokens[i + 1] == "source" || tokens[i + 1] == "camera"))
            || (tokens[i] == "source" && i + 1 < tokens.len())
        {
            let (n, next) = if tokens[i] == "seen" {
                (parse_u32(tokens[i + 3]), i + 4)
            } else if tokens[i] == "on" {
                (parse_u32(tokens[i + 2]), i + 3)
            } else {
                (parse_u32(tokens[i + 1]), i + 2)
            };
            if let Some(id) = n {
                preds.push(SubjectPredicate::SeenOn(SourceId(id)));
                i = next;
                continue;
            }
        }

        // in zone N | zone N
        if (tokens[i] == "in" && i + 2 < tokens.len() && tokens[i + 1] == "zone")
            || (tokens[i] == "zone"
                && i + 1 < tokens.len()
                && tokens
                    .get(i + 1)
                    .is_some_and(|t| t.chars().all(|c| c.is_ascii_digit())))
        {
            let (n, next) = if tokens[i] == "in" {
                (parse_u16(tokens[i + 2]), i + 3)
            } else {
                (parse_u16(tokens[i + 1]), i + 2)
            };
            if let Some(id) = n {
                preds.push(SubjectPredicate::SeenInZone(ZoneId(id)));
                i = next;
                continue;
            }
        }

        // then zone A then zone B [within Ns]
        if tokens[i] == "then"
            && let Some((first, second, within, next)) = parse_then_chain(&tokens, i)
        {
            preds.push(SubjectPredicate::ThenSeenIn(ThenSeenIn {
                first: ZoneId(first),
                then: ZoneId(second),
                within_ns: within,
            }));
            i = next;
            continue;
        }

        // route zone A B C
        if tokens[i] == "route" {
            let start = if tokens.get(i + 1) == Some(&"zone") {
                i + 2
            } else {
                i + 1
            };
            let mut zones = Vec::new();
            let mut j = start;
            while j < tokens.len() {
                if let Some(z) = parse_u16(tokens[j]) {
                    zones.push(ZoneId(z));
                    j += 1;
                } else {
                    break;
                }
            }
            if !zones.is_empty() {
                preds.push(SubjectPredicate::RouteContains(zones));
                i = j;
                continue;
            }
        }

        // subject N[,M]
        if tokens[i] == "subject" && i + 1 < tokens.len() {
            let mut ids = Vec::new();
            let mut j = i + 1;
            while j < tokens.len() {
                let raw = tokens[j].trim_end_matches(',');
                if let Some(id) = parse_u64(raw) {
                    ids.push(SubjectId(id));
                    j += 1;
                    if !tokens[j - 1].ends_with(',') {
                        // allow "subject 1 2" as well
                        if j < tokens.len() && parse_u64(tokens[j]).is_some() {
                            continue;
                        }
                        break;
                    }
                } else {
                    break;
                }
            }
            if !ids.is_empty() {
                preds.push(SubjectPredicate::Subjects(ids));
                i = j;
                continue;
            }
        }

        // min confidence F | confidence >= F
        if ((tokens[i] == "min" && i + 2 < tokens.len() && tokens[i + 1] == "confidence")
            || (tokens[i] == "confidence" && i + 2 < tokens.len() && tokens[i + 1] == ">="))
            && let Some(c) = parse_f32(tokens[i + 2])
        {
            preds.push(SubjectPredicate::MinConfidence(c));
            i += 3;
            continue;
        }

        // min dwell Ns | dwell >= Ns
        if ((tokens[i] == "min" && i + 2 < tokens.len() && tokens[i + 1] == "dwell")
            || (tokens[i] == "dwell" && i + 2 < tokens.len() && tokens[i + 1] == ">="))
            && let Some(ns) = parse_i64(tokens[i + 2])
        {
            preds.push(SubjectPredicate::MinDwellNs(ns));
            i += 3;
            continue;
        }

        // during START END (nanoseconds)
        if tokens[i] == "during"
            && i + 2 < tokens.len()
            && let (Some(a), Some(b)) = (parse_i64(tokens[i + 1]), parse_i64(tokens[i + 2]))
            && let (Ok(start), Ok(end)) = (
                MediaTime::new(a, 1_000_000_000),
                MediaTime::new(b, 1_000_000_000),
            )
        {
            preds.push(SubjectPredicate::During { start, end });
            i += 3;
            continue;
        }

        // near box L T R B (spatial)
        if tokens[i] == "near"
            && tokens.get(i + 1) == Some(&"box")
            && i + 5 < tokens.len()
            && let (Some(l), Some(t), Some(r), Some(b)) = (
                parse_f32(tokens[i + 2]),
                parse_f32(tokens[i + 3]),
                parse_f32(tokens[i + 4]),
                parse_f32(tokens[i + 5]),
            )
        {
            preds.push(SubjectPredicate::Spatial(SpatialQuery::new(l, t, r, b)));
            i += 6;
            continue;
        }

        // noise words
        if matches!(
            tokens[i],
            "the"
                | "a"
                | "an"
                | "who"
                | "that"
                | "were"
                | "was"
                | "persons"
                | "people"
                | "find"
                | "show"
                | "all"
        ) {
            i += 1;
            continue;
        }

        warnings.push(format!("unknown token '{}'", tokens[i]));
        i += 1;
    }

    if preds.is_empty() {
        return Err(NlParseError::NoPredicates);
    }
    let node = if preds.len() == 1 {
        QueryNode::pred(preds.pop().unwrap())
    } else {
        QueryNode::and(preds.into_iter().map(QueryNode::pred).collect::<Vec<_>>())
    };
    Ok(NlParseResult { node, warnings })
}

fn parse_then_chain(tokens: &[&str], start: usize) -> Option<(u16, u16, i64, usize)> {
    // then [zone] A then [zone] B [within Ns]
    let mut i = start;
    if tokens.get(i)? != &"then" {
        return None;
    }
    i += 1;
    if tokens.get(i) == Some(&"zone") {
        i += 1;
    }
    let first = parse_u16(tokens.get(i)?)?;
    i += 1;
    if tokens.get(i)? != &"then" {
        return None;
    }
    i += 1;
    if tokens.get(i) == Some(&"zone") {
        i += 1;
    }
    let second = parse_u16(tokens.get(i)?)?;
    i += 1;
    let mut within = 0_i64;
    if tokens.get(i) == Some(&"within") {
        i += 1;
        within = parse_i64(tokens.get(i)?)?;
        i += 1;
    }
    Some((first, second, within, i))
}

fn parse_u32(s: &str) -> Option<u32> {
    s.parse().ok()
}
fn parse_u16(s: &str) -> Option<u16> {
    s.parse().ok()
}
fn parse_u64(s: &str) -> Option<u64> {
    s.parse().ok()
}
fn parse_i64(s: &str) -> Option<i64> {
    s.parse().ok()
}
fn parse_f32(s: &str) -> Option<f32> {
    let v: f32 = s.parse().ok()?;
    v.is_finite().then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seen_on_and_zone() {
        let r = parse_nl_query("find people seen on source 1 and in zone 2").unwrap();
        assert!(
            r.warnings.is_empty()
                || r.warnings
                    .iter()
                    .all(|w| w.contains("people") || w.contains("find"))
        );
        match r.node {
            QueryNode::And(children) => assert!(children.len() >= 2),
            QueryNode::Pred(_) => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_or_and_subject() {
        let r = parse_nl_query("subject 5 or subject 7").unwrap();
        match r.node {
            QueryNode::Or(parts) => assert_eq!(parts.len(), 2),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_then_zone_chain() {
        let r = parse_nl_query("then zone 1 then zone 3 within 5000000000").unwrap();
        match r.node {
            QueryNode::Pred(SubjectPredicate::ThenSeenIn(t)) => {
                assert_eq!(t.first, ZoneId(1));
                assert_eq!(t.then, ZoneId(3));
                assert_eq!(t.within_ns, 5_000_000_000);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn empty_and_garbage() {
        assert_eq!(parse_nl_query("   ").unwrap_err(), NlParseError::Empty);
        assert_eq!(
            parse_nl_query("hello world").unwrap_err(),
            NlParseError::NoPredicates
        );
    }
}
