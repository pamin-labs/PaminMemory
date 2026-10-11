//! Experimental selection of the accurate reranker shortlist.

use pamin_core::{Channel, Why};

/// Reserve each effective non-graph channel's first candidate, then fill from
/// the original fused head. Graph-only additions retain their existing budget.
/// Input traces are live, unique topics in fused order; ties use that order.
pub(crate) fn pooled_head(traces: &[&[Why]], baseline: &[usize], head: usize) -> Vec<usize> {
    let head = head.min(traces.len());
    let mut selected = Vec::new();
    for channel in [
        Channel::LexicalSegmented,
        Channel::LexicalNgram,
        Channel::Vector,
    ] {
        if let Some(at) = traces.iter().position(|why| {
            why.iter().any(|entry| {
                matches!(entry, Why::Channel {
                    channel: found, rank: 1, weight, contribution, ..
                } if *found == channel && *weight > 0.0 && *contribution > 0.0)
            })
        }) {
            if !selected.contains(&at) && selected.len() < head {
                selected.push(at);
            }
        }
    }
    for at in baseline.iter().copied().filter(|at| *at < head) {
        if selected.len() == head {
            break;
        }
        if !selected.contains(&at) {
            selected.push(at);
        }
    }
    // These are graph-only, so cannot duplicate a reserved non-graph topic.
    selected.extend(baseline.iter().copied().filter(|at| *at >= head));
    selected.sort_unstable();
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(channel: Channel, rank: u32, weight: f32, contribution: f32) -> Why {
        Why::Channel {
            channel,
            rank,
            score: Some(1.0),
            weight,
            contribution,
        }
    }

    fn pool(traces: &[Vec<Why>], baseline: &[usize], head: usize) -> Vec<usize> {
        pooled_head(
            &traces.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            baseline,
            head,
        )
    }

    #[test]
    fn a_channel_winner_beyond_thirty_replaces_a_pair_instead_of_adding_one() {
        for rank in [34, 35, 37] {
            let mut traces = vec![Vec::new(); 50];
            traces[rank - 1] = vec![channel(Channel::LexicalSegmented, 1, 0.125, 0.01)];
            let baseline: Vec<_> = (0..30).collect();
            assert!(
                !baseline.contains(&(rank - 1)),
                "reproduction must miss the winner"
            );
            let selected = pool(&traces, &baseline, 30);
            assert_eq!(selected.len(), 30);
            assert!(selected.contains(&(rank - 1)));
            assert_eq!(&selected[..29], &(0..29).collect::<Vec<_>>());
            assert!(!selected.contains(&29));
        }
    }

    #[test]
    fn three_distinct_channel_winners_keep_the_same_pair_budget() {
        let mut traces = vec![Vec::new(); 50];
        for (at, kind) in [
            (33, Channel::LexicalSegmented),
            (34, Channel::LexicalNgram),
            (36, Channel::Vector),
        ] {
            traces[at] = vec![channel(kind, 1, 1.0, 0.1)];
        }
        let selected = pool(&traces, &(0..30).collect::<Vec<_>>(), 30);
        assert_eq!(selected, (0..27).chain([33, 34, 36]).collect::<Vec<_>>());
    }

    #[test]
    fn shared_channel_winners_are_one_pair_and_head_winners_cost_no_slot() {
        let mut traces = vec![Vec::new(); 50];
        traces[0] = vec![channel(Channel::Vector, 1, 1.0, 0.1)];
        traces[33] = vec![
            channel(Channel::LexicalSegmented, 1, 0.125, 0.01),
            channel(Channel::LexicalNgram, 1, 0.125, 0.01),
        ];
        assert_eq!(
            pool(&traces, &(0..30).collect::<Vec<_>>(), 30),
            (0..29).chain([33]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn disabled_and_ineffective_channels_do_not_reserve_slots() {
        let mut traces = vec![Vec::new(); 50];
        traces[33] = vec![channel(Channel::LexicalSegmented, 1, 0.0, 0.0)];
        traces[34] = vec![channel(Channel::LexicalNgram, 1, 0.125, 0.0)];
        traces[36] = vec![channel(Channel::Vector, 1, -1.0, -0.1)];
        let baseline: Vec<_> = (0..30).collect();
        assert_eq!(pool(&traces, &baseline, 30), baseline);
    }

    #[test]
    fn equal_scores_follow_channel_rank_and_duplicate_rank_one_uses_fused_order() {
        let mut traces = vec![Vec::new(); 50];
        traces[33] = vec![channel(Channel::LexicalNgram, 2, 0.125, 0.01)];
        traces[34] = vec![channel(Channel::LexicalNgram, 1, 0.125, 0.01)];
        traces[36] = vec![channel(Channel::LexicalNgram, 1, 0.125, 0.01)];
        let selected = pool(&traces, &(0..30).collect::<Vec<_>>(), 30);
        assert!(selected.contains(&34));
        assert!(!selected.contains(&33));
        assert!(!selected.contains(&36));
    }

    #[test]
    fn graph_additions_are_preserved_without_new_graph_candidates() {
        let mut traces = vec![Vec::new(); 70];
        traces[33] = vec![channel(Channel::LexicalSegmented, 1, 0.125, 0.01)];
        traces[60] = vec![channel(Channel::Graph, 1, 0.3, 0.1)];
        let baseline: Vec<_> = (0..30).chain([61, 62]).collect();
        let selected = pool(&traces, &baseline, 30);
        assert_eq!(selected.len(), baseline.len());
        assert!(selected.ends_with(&[61, 62]));
        assert!(!selected.contains(&60));
    }

    #[test]
    fn empty_and_short_lists_keep_their_count_and_order() {
        assert!(pool(&[], &[], 30).is_empty());
        let traces = vec![vec![channel(Channel::Vector, 1, 1.0, 0.1)], Vec::new()];
        assert_eq!(pool(&traces, &[0, 1], 30), vec![0, 1]);
        assert!(pool(&traces, &[], 0).is_empty());
    }
}
