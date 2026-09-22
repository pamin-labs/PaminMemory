//! A second pass over the shortlist.
//!
//! Fusion orders candidates by the ranks four channels gave them, and it never
//! reads a candidate and the query together. A cross-encoder does, which is why
//! it can fix an ordering fusion got wrong and why it costs what it costs: the
//! query and the document go through one model jointly, so nothing about a
//! document can be computed before the query arrives. That is a property of the
//! architecture rather than of this implementation, and it is why the
//! literature's answers to cross-encoder latency all change the architecture --
//! precomputing part of the document's representation at index time, or moving
//! to late interaction. Both trade the cost for storage proportional to
//! documents times tokens times width, which for a store meant to hold millions
//! of memories is tens of gigabytes. So the cost is paid at query time or not
//! at all.
//!
//! That arithmetic is right at millions and was quoted here as though it
//! settled the question at any size, which it does not: 128-dimensional int8
//! token vectors are about 67 MB over XQuAD-R's 13,014 sentences and 1.35 GB
//! over MIRACL's 131,924 passages, against indexes of 119 MB and 1.1 GB. Two
//! to three times the index is an argument, not a foreclosure. Nor is the
//! licence one any more: `lightonai/mLateOn` is Apache-2.0, multilingual and
//! carries its own int8 ONNX, where `jina-colbert-v2` is CC-BY-NC-4.0,
//! `answerai-colbert-small-v1` is English, and `colbert-xm` selects a
//! per-language adapter at runtime that does not fit one static graph.
//!
//! An earlier version of this note gave the remaining obstacle as language
//! coverage -- nine training languages against this product's eleven, with
//! Swahili absent. **That was backwards.** The model's own paper
//! (`arXiv:2607.27178`, 2026) is largely about generalising to languages
//! absent from retrieval training, and reports its unseen-language MIRACL
//! average about ten points above the dense model it is paired with. MIRACL
//! Swahili is not the case that rules mLateOn out; it is the case mLateOn
//! claims, and it is where this stack is weakest.
//!
//! What is actually left is three costs and no measurement of any of them: the
//! int8 export is the backbone alone, with three `*_Dense` projection modules
//! shipped separately as safetensors and applied after it, so the projection
//! is this project's to reimplement; the write path gains a forward pass per
//! memory; and the token vectors have to be stored.
//! `docs/adr/0001-tech-selection.md` carries the trigger.
//!
//! ## What it is worth, measured
//!
//! On 13,014 sentences in eleven languages, reranking the fused shortlist:
//!
//! | | cross-lingual | same-language | per query | of which reranking |
//! |---|---|---|---|---|
//! | off | — | — | 53 ms | — |
//! | `fast` | **+0.0381** | **-0.0053** | 264 ms | 211 ms |
//! | `accurate` | **+0.0448** | **+0.0017** | 1001 ms | 948 ms |
//!
//! Those score columns are differences of means, which is all this project
//! could report until `tests/statistics` existed. Re-taken query by query at
//! the lexical weight that ships now, the `fast` tier reads:
//!
//! | group | mean | wins / losses / ties | p |
//! |---|---|---|---|
//! | cross-lingual | **+0.0403** | 547 / 206 / 437 | 0.0001 |
//! | same-language | **-0.0061** | 3 / 19 / 1,168 | 0.0007 |
//!
//! Both are real, and the second one is more interesting than its mean. The
//! reranker touches twenty-two same-language queries out of 1,190 and makes
//! nineteen of them worse. The damage is not diffuse and small; it is
//! concentrated and consistent, and it looks small only because the confinement
//! to unlexical candidates keeps it away from almost every query.
//!
//! Measured through `Engine::search_reranked` -- the entry point `pamin search`
//! calls -- with `TIERS=1` on the cross-lingual harness, over all 1,190
//! queries. An earlier version of this table came from a scratch program that
//! reordered a dumped shortlist with its own copy of the pipeline, and it
//! overstated both gains by about half.
//!
//! The latency here replaces 204 ms and 508 ms, which replaced an earlier
//! 1795 ms for `accurate`. Re-running the same harness on the same machine and
//! workspace returns twice the recorded cost for `accurate`, so the first
//! figure was too high and its correction too low. The quality columns do
//! reproduce, to a thousandth rather than exactly -- the claim that these
//! tiers "return the same four decimals on every run" was a claim about a
//! fixed corpus and index, and the index is maintained between runs.
//!
//! Three things in that table need saying.
//!
//! **The same-language column is nearly, but not exactly, zero.** Every
//! reranker tried -- three, across two orders of magnitude in size -- improves
//! cross-lingual ranking and damages same-language ranking, because the fused
//! list is already good at same-language and a cross-encoder reorders it worse.
//! So only the candidates no lexical channel found are reranked, and they are
//! placed back into the positions they already held. That confines the damage
//! but does not eliminate it, and the earlier claim that the group "cannot
//! move" was wrong: a same-language answer the lexical channels *missed* is an
//! unlexical candidate like any other, and reordering can carry it down.
//! Sixty-one same-language queries leave something below rank ten before
//! reranking and seventy-one after. The cost is small and it is real.
//! (Unrestricted, two of those models scored +0.1698 / -0.2109 and +0.1678 /
//! -0.0806 on the scratch harness. Not re-measured here.)
//!
//! **`accurate` is better on both groups, not just the first.** It is the only
//! tier that does not cost same-language ranking.
//!
//! **`fast` is the default on latency, not on quality.** It is a twelve-layer
//! distilled MiniLM with 21M encoder parameters against XLM-RoBERTa-large's
//! 303M -- fourteen times smaller, and its pass costs 211 ms against 948, so
//! 4.5x. For that it gives up 0.0067 cross-lingual and the 0.0070
//! same-language that `accurate` gains. Whether six sevenths of the gain is
//! worth a quarter of the latency is a workspace's call and `--rerank
//! accurate` is how to make it.
//! That is the finding of [Shallow Cross-Encoders for Low-Latency
//! Retrieval](https://arxiv.org/abs/2403.20222) arrived at independently: under
//! a latency budget a shallow model beats a full-scale one, because the budget
//! buys more candidates.
//!
//! ## And on a corpus that is not parallel text, the gap doubles
//!
//! Everything above is XQuAD-R, where the eleven versions of a sentence are
//! translations of each other. MIRACL's Swahili dev split is not: 131,924 real
//! passages averaging 229 characters, 482 queries, human relevance judgements,
//! one language throughout.
//!
//! | | nDCG@10 | gain | a search | of which reranking |
//! |---|---|---|---|---|
//! | off | 0.7158 | — | 142 ms | — |
//! | `fast` | 0.7359 | **+0.0201** | 474 ms | 332 ms |
//! | `accurate` | 0.7654 | **+0.0496** | 1867 ms | 1725 ms |
//!
//! `fast` is worth about half what it is worth on parallel sentences. That much
//! was expected: the pass only reorders what the lexical channels missed, and
//! across a language boundary that is nearly the whole shortlist while within
//! one language it is a fraction -- 84 of 482 queries leave a relevant passage
//! below rank ten here against 1,042 of 1,190 there.
//!
//! **`accurate` was expected to shrink with it and does the opposite.** It
//! gains more on this corpus than on the other, +0.0496 against +0.0458, so the
//! ratio between the two tiers goes from 1.2 to 2.5. The passages are long and
//! genuinely varied, which is where twenty-one million parameters start to tell
//! against three hundred million. The shallow-cross-encoder argument the
//! default leans on was validated on parallel single sentences, and this is the
//! shape of corpus it does not describe.
//!
//! What it costs is the other half. Reranking is 332 ms and 1725 ms here
//! against 165 and 469 on sentences, because a cross-encoder reads the
//! candidate and `MAX_TOKENS` actually binds on a passage. Two seconds a search
//! is not an interactive budget, so `fast` stays the default -- but on real
//! passages the choice is giving up three fifths of the available gain rather
//! than a fifth, and a workspace of long documents should know that before
//! accepting it.
//!
//! recall@50 is 0.9494 for all three tiers, to four decimals, as on the other
//! corpus: the pass reorders a shortlist and never changes it.
//!
//! **And on this corpus the pass is worth less than nothing, which survives
//! being tested.** On the `speed` profile at the shipped lexical weight, the
//! `fast` tier scores 0.6730 against fusion's 0.6882: **-0.0152, 37 wins
//! against 57 losses over 482 queries, p = 0.0129**. It reaches ninety-four
//! queries and loses on more of them than it wins. That is the same shape as
//! the same-language group above, on a corpus where every query is
//! same-language -- so the two corpora agree about what this tier does when a
//! query and its answer share a language, and disagree only about how many such
//! queries there are.
//!
//!
//! ## What is not paid twice
//!
//! A cross-encoder cannot precompute anything about a memory before the query
//! arrives -- that is what joint encoding means, and it is why the literature's
//! answers to this latency all change the architecture: precomputing part of a
//! document's representation at index time, or moving to late interaction.
//! Both trade the cost for storage proportional to documents times tokens times
//! width, and both would mean shipping and maintaining a re-export of somebody
//! else's weights split in two. Neither is ruled out; neither is here, and the
//! storage is the smaller of the two obstacles -- see the module notes above
//! for the sizes and for the licence wall that is the larger one.
//!
//! [`Reranker::counted`] is what would decide the first of them. The score
//! cache's hit rate says whether the hot set is small enough for precomputing
//! part of each document to pay for itself, and until it was exposed nothing
//! in this project could read it.
//!
//! What is here is the one thing that can be kept: the score itself. A query
//! and a memory score the same every time, so a resident server remembers them,
//! and a repeated search costs nothing -- 69.6 ms the first time, 0.0 ms the
//! second, for the same ordering. It does nothing for a query never asked
//! before, which is most of them; it is worth its quarter of a megabyte because
//! agents retry.
//!
//! ## What the numbers do not say
//!
//! They were measured on four cores. Published figures for a MiniLM
//! cross-encoder on CPU are 0.5 to 3 ms per pair; this measures 9.75, and the
//! gap is two layers' worth of depth and a quarter of the cores. On an ordinary
//! server the `fast` tier should be tens of milliseconds rather than 165.
//!
//! The latency column is also the soft one. The same configuration measured in
//! two separate runs of the harness differs by as much as a fifth, so only
//! figures taken inside one run are comparable with each other. The nDCG
//! columns have no such problem: they repeat to four decimals.
//!
//! What was a caveat here -- that all of this was measured on parallel text,
//! and a corpus whose candidates are not translations of each other might
//! divide the gain differently -- is now the MIRACL section above. It does
//! divide it differently, and not in the direction the caveat guessed.

use std::path::{Path, PathBuf};

use fastembed::{
    OnnxSource, RerankInitOptionsUserDefined, TextRerank, TokenizerFiles, UserDefinedRerankingModel,
};
use serde::{Deserialize, Serialize};

use crate::error::{IndexError, Result};

/// What a tier's weights may be used for.
///
/// Carried in the type rather than looked up in a document, because the one
/// consequence that matters is a refusal: a tier whose weights are not free for
/// commercial use has to be asked for on purpose, and a caller cannot be
/// expected to have read `NOTICE` first.
///
/// This project redistributes no weights -- every model is fetched from the hub
/// by the user's own machine on first use -- so what is described here is what
/// the user acquires, not what we ship. That is also why a missing licence tag
/// is not one of the variants: an export with no tag of its own is usable when
/// the chain to a licensed source is readable, and two of the shipped models
/// are in exactly that position, with their chains written down in `NOTICE`.
/// What cannot be left to a document is a term that restricts what the user may
/// do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Licence {
    /// Free for any use, commercial included. Apache-2.0 or MIT, directly or
    /// through a readable chain.
    Permissive,
    /// Free for research and personal use, not for commercial use. Asking for a
    /// tier under this is an explicit act; see [`Rerank::licence`].
    NonCommercial,
}

/// How much to spend reordering the shortlist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rerank {
    /// Return what fusion ordered.
    ///
    /// The right setting for a workspace whose memories are all in one
    /// language, where there is very little for the pass to find: what it
    /// reorders is the candidates the lexical channels missed, and across a
    /// language boundary that is most of them.
    ///
    /// Not *nothing*, though, and the difference matters. The rule is "no
    /// lexical channel found it", not "it is in another language" -- the two
    /// agree on 93% of a shortlist but not on all of it -- so a one-language
    /// workspace still has vector-only candidates and the pass still moves
    /// them. On this project's own corpus the monolingual group does not
    /// budge, but that group sits at 0.9940 where nothing could move it.
    Off,
    /// A twelve-layer distilled multilingual MiniLM, 113 MB. The default.
    ///
    /// Chosen over the larger model on latency rather than on the score: it
    /// reaches four fifths of the cross-lingual gain for two fifths of the
    /// latency and a fifth of the download. It is the one tier that costs
    /// same-language ranking, by 0.0062.
    #[default]
    Fast,
    /// XLM-RoBERTa-large, 570 MB. Worth more the less the corpus looks like
    /// a parallel sentence benchmark.
    ///
    /// On XQuAD-R it is 2.8x the latency for 22% more gain, and better than
    /// `fast` on *both* groups -- the only tier that costs nothing on either.
    /// On MIRACL's Swahili passages it is worth **two and a half times** what
    /// `fast` is, +0.0496 against +0.0201, because a twenty-one-million
    /// parameter model runs out of capacity on long real text where a
    /// three-hundred-million one does not.
    ///
    /// It is also five times the cost there rather than three: reranking is
    /// 1725 ms a search against `fast`'s 332. Two seconds is not interactive,
    /// which is why this is not the default -- but a workspace of long
    /// documents that can afford it is giving up rather more by not asking.
    Accurate,
    /// GTE multilingual reranker base, twelve layers of 768, 341 MB.
    /// Seventy-plus languages.
    ///
    /// Between the two above by every structural measure and here to find out
    /// whether it is between them by score. Non-embedding parameters are the
    /// number that predicts compute -- a 250,000-row embedding table is a
    /// lookup and not a matrix multiply, so a card's total is misleading --
    /// and by that count this is 84.9M against `fast`'s 21.2M and
    /// `accurate`'s 302M: four times the one and a third of the other. The
    /// int8 export is 341 MB against 119 and 571, which is the same ordering
    /// and is what a user actually waits for on first use.
    ///
    /// The reason to measure it is that `accurate` is where the gain is and
    /// the latency is why nobody can have it. If four times `fast`'s compute
    /// buys most of fourteen times' worth, the tier that ships as the quality
    /// option should be this one.
    ///
    /// Apache-2.0 through the same shape of chain as `accurate`: the export
    /// carries no tag, `Alibaba-NLP/gte-multilingual-reranker-base` under it
    /// is Apache-2.0. Read from the hub API rather than from card prose, and
    /// written down in `NOTICE`.
    Balanced,
}

/// How many of the fused results a tier looks at.
///
/// Twenty for both, which is where the gain stops. Swept through
/// `search_reranked` on the `fast` model, 1,190 XQuAD-R queries, against a
/// baseline of 0.5722 cross-lingual with reranking off:
///
/// | depth | cross-lingual | gain | same-language |
/// |---|---|---|---|
/// | 10 | 0.5831 | +0.0110 | 0.8005 |
/// | 15 | 0.6047 | +0.0325 | 0.7987 |
/// | **20** | **0.6091** | **+0.0369** | **0.7974** |
/// | 30 | 0.6099 | +0.0378 | 0.7967 |
/// | 50 | 0.6055 | +0.0333 | 0.7957 |
///
/// The constant is unchanged and the reason for it is not. An earlier sweep,
/// on the scratch harness whose figures ran about half again high, put twenty
/// at +0.0572 and thirty at +0.0667 and recorded thirty as the better score
/// given up for latency. Measured through the engine, thirty buys +0.0009 --
/// a tenth of what was recorded, for sixteen per cent more latency. There is
/// no trade to make; twenty is simply where it stops.
///
/// Taken when the lexical weight was a quarter. It is now an eighth, and the
/// baseline this sweep measured against moved with it, 0.5722 to 0.6077: the
/// reranker has less dilution to undo, so where the gain stops could have
/// moved too. Re-running the five depths is five passes of the corpus, about
/// three quarters of an hour, and it has not been done.
///
/// Same-language ranking falls monotonically with depth, which is the same
/// effect the tier table describes: more candidates reranked means more of the
/// ones the lexical channels missed being carried down.
const DEPTH: usize = 20;

/// The tuning constants above, overridable for a sweep.
///
/// `DEPTH`, `BATCH` and `MAX_TOKENS` were each settled by measurement, and two
/// of the three were settled on the scratch harness whose figures turned out to
/// be about half again too high -- so they have to be re-settleable, and by the
/// harness rather than by editing a constant and rebuilding. Same shape as the
/// evaluation's own `SWEEP` and `TIERS`: unset means the constant, so nothing a
/// user runs is affected.
fn tuned(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(fallback)
}

/// How many candidates go through the model at once.
///
/// Eight, and the honest statement is that the corpus cannot separate it from
/// the alternatives. A batch is padded to its longest member, so in principle a
/// large batch pays for its longest candidate on every member, and an earlier
/// sweep recorded eight at 151 ms against sixteen at 191. Swept again through
/// the engine, at a depth of twenty:
///
/// | batch | cross-lingual | a search |
/// |---|---|---|
/// | 4 | 0.6102 | 226 ms |
/// | **8** | **0.6091** | **217 ms** |
/// | 16 | 0.6095 | 230 ms |
/// | 20 | 0.6098 | 242 ms |
///
/// Eight is fastest here, but the spread across all four is eleven per cent and
/// the same configuration measured in two separate runs differs by nineteen --
/// so this says only that none of them is clearly better, not that eight wins.
/// Separating them would need repeats inside one process, and nothing here
/// turns on the answer.
///
/// The score column is not noise, though: it moves by up to 0.0011 across batch
/// sizes because a quantized model scores a pair slightly differently depending
/// on what it was padded alongside. Anything that changes how candidates are
/// grouped moves the fourth decimal, which is worth knowing before attributing
/// such a change to something else.
const BATCH: usize = 8;

/// How long a candidate the model reads, and how many at once. See [`tuned`].
fn batch() -> usize {
    tuned("PAMIN_RERANK_BATCH", BATCH)
}

fn max_tokens() -> usize {
    tuned("PAMIN_RERANK_MAX_TOKENS", MAX_TOKENS)
}

/// The longest candidate the model reads.
///
/// Two hundred and fifty-six, and halving it is not the free saving this used
/// to record. The claim here was that 128 "changed latency by a fifth and
/// nothing else". Measured through the engine, at a depth of twenty:
///
/// | tokens | cross-lingual | same-language | a search |
/// |---|---|---|---|
/// | 128 | 0.6077 | 0.7974 | 207 ms |
/// | **256** | **0.6091** | **0.7974** | **217 ms** |
///
/// Both halves were wrong. The saving is five per cent rather than twenty, and
/// it costs 0.0014 of cross-lingual ranking rather than nothing. `fastembed`
/// pads a batch to its longest member and not to this limit, so the limit only
/// truncates the candidates that genuinely exceed it -- on a corpus of
/// sentences, few of them. It earns its place by bounding the worst case rather
/// than by shaping the ordinary one: one long memory cannot make one query
/// slow.
///
/// **On a corpus of passages it binds, and now there are numbers for both.**
/// The harnesses count the length of every candidate that reaches the model:
///
/// | corpus | mean | longest |
/// |---|---|---|
/// | XQuAD-R, sentences | 165 characters | 1,341 |
/// | MIRACL Swahili, Wikipedia passages | 311 characters | 5,567 |
///
/// Characters rather than tokens, because that is what can be counted without
/// asking the tokenizer -- see the sort in [`Reranker::rank`] for the small
/// factor between them. A 5,567-character passage is far past this limit
/// whatever the script, so on MIRACL the truncation is doing real work, and the
/// sentence-corpus measurement above says nothing about what it costs there.
/// The 128-against-256 sweep has only ever been run on the corpus where the
/// limit does not bind, which is the wrong one to run it on.
const MAX_TOKENS: usize = 256;

impl Rerank {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "off" => Some(Self::Off),
            "fast" => Some(Self::Fast),
            "accurate" => Some(Self::Accurate),
            "balanced" => Some(Self::Balanced),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fast => "fast",
            Self::Accurate => "accurate",
            Self::Balanced => "balanced",
        }
    }

    /// How deep into the fused list this tier reaches.
    pub fn depth(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Fast | Self::Accurate | Self::Balanced => tuned("PAMIN_RERANK_DEPTH", DEPTH),
        }
    }

    /// What this tier's weights may be used for.
    ///
    /// `Off` has none, which is a real answer rather than a missing one, so it
    /// is `None` and every tier that loads a model has a `Some`.
    ///
    /// The distinction this draws is narrow on purpose: whether the licence
    /// restricts what the user may do with the results. A permissive tier and a
    /// tier whose export carries no tag but descends from a permissive model
    /// are the same answer to that question, and `NOTICE` is where the chains
    /// are written down.
    pub fn licence(self) -> Option<Licence> {
        match self {
            Self::Off => None,
            // Apache-2.0. Distilled from a model with no tag of its own, whose
            // source is Microsoft's MIT MiniLMv2 recipe; see `NOTICE`.
            Self::Fast => Some(Licence::Permissive),
            // The export carries no tag; `BAAI/bge-reranker-v2-m3` under it is
            // Apache-2.0. See `NOTICE`.
            Self::Accurate => Some(Licence::Permissive),
            // Same shape of chain: no tag on the export,
            // `Alibaba-NLP/gte-multilingual-reranker-base` under it is
            // Apache-2.0. See `NOTICE`.
            Self::Balanced => Some(Licence::Permissive),
        }
    }

    fn repository(self) -> &'static str {
        match self {
            Self::Off => unreachable!("nothing is loaded for the off tier"),
            Self::Fast => "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            Self::Accurate => "onnx-community/bge-reranker-v2-m3-ONNX",
            Self::Balanced => "onnx-community/gte-multilingual-reranker-base",
        }
    }

    /// Which export of the model to fetch.
    ///
    /// The fast model publishes one quantized export per instruction set, and
    /// they are not interchangeable in speed: on a machine with AVX-512 VNNI
    /// the VNNI build is 1.5x the AVX2 one for the same scores. Picking at
    /// runtime rather than at build time, because a binary is built once and
    /// run on whatever is there.
    fn onnx(self) -> &'static str {
        match self {
            Self::Off => unreachable!("nothing is loaded for the off tier"),
            // One int8 export, not one per instruction set, so there is
            // nothing to detect at runtime the way `fast` has to.
            Self::Accurate | Self::Balanced => "onnx/model_int8.onnx",
            Self::Fast => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::arch::is_x86_feature_detected!("avx512vnni") {
                        return "onnx/model_qint8_avx512_vnni.onnx";
                    }
                    "onnx/model_quint8_avx2.onnx"
                }
                #[cfg(target_arch = "aarch64")]
                {
                    "onnx/model_qint8_arm64.onnx"
                }
                // No quantized build for this architecture; full precision runs
                // everywhere and is what the quantized ones were made from.
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    "onnx/model.onnx"
                }
            }
        }
    }
}

/// How many scores are remembered.
///
/// A score is a query *and* a document, so this helps when a query comes round
/// again -- which is what an agent does: it retries, it widens a limit, it asks
/// the same thing again after writing something. It cannot help a query never
/// asked before, and nothing about a document alone can be remembered, because
/// a cross-encoder reads the document with the query and that is the whole of
/// why it is worth running.
///
/// Four thousand entries is about a quarter of a megabyte, and the cache is per
/// process, so it is `pamin serve` that makes it worth anything: without a
/// resident process every command starts with an empty one.
const REMEMBERED_SCORES: usize = 4096;

/// Scores already computed, oldest first.
///
/// Keyed by a 64-bit hash of the query and the document rather than by either:
/// holding the text would cost more than the model saves, and the pair is what
/// identifies a score. A collision returns one candidate's score for another,
/// which misorders a result rather than breaking one, and at this size the
/// chance of one is around a trillion to one per lookup.
#[derive(Default)]
struct Scores {
    known: std::collections::HashMap<u64, f32>,
    order: std::collections::VecDeque<u64>,
    /// Lookups that found a score, and lookups that did not.
    ///
    /// Counted because the ADR makes them the evidence for a decision it has
    /// deferred: whether precomputing part of each document's representation
    /// at index time is worth its storage depends on the hot set being small,
    /// and this ratio is what says whether it is. Before this the only
    /// accessor reported occupancy, which says how much has been stored and
    /// nothing about how often it is read.
    hits: u64,
    misses: u64,
}

impl Scores {
    fn key(query: &str, document: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut hasher);
        // Separated, so that a query ending where a document begins cannot
        // collide with the other split of the same characters.
        0u8.hash(&mut hasher);
        document.hash(&mut hasher);
        hasher.finish()
    }

    fn get(&mut self, key: u64) -> Option<f32> {
        let found = self.known.get(&key).copied();
        match found {
            Some(_) => self.hits += 1,
            None => self.misses += 1,
        }
        found
    }

    /// Remembers a score, forgetting the oldest once full.
    ///
    /// Insertion order rather than use order. Keeping a true LRU means writing
    /// to the queue on every hit, and what this protects is milliseconds of
    /// inference; a query asked twice is asked twice close together.
    fn put(&mut self, key: u64, score: f32) {
        if self.known.insert(key, score).is_some() {
            return;
        }
        self.order.push_back(key);
        while self.order.len() > REMEMBERED_SCORES {
            if let Some(oldest) = self.order.pop_front() {
                self.known.remove(&oldest);
            }
        }
    }
}

/// A loaded cross-encoder, and what it has already scored.
pub struct Reranker {
    model: TextRerank,
    tier: Rerank,
    scores: Scores,
    lengths: Lengths,
}

/// How long the candidates that reached the model were.
///
/// Counted in characters because that is what the batching sort already uses
/// and what can be counted without asking the tokenizer -- see the sort in
/// [`Reranker::rank`] for why characters rather than bytes, and for the small
/// factor that separates them from tokens.
///
/// Here because the cost of a cross-encoder rises with sequence length and
/// nothing in this project had ever recorded the lengths it sees. `MAX_TOKENS`
/// says the limit binds on passage-shaped corpora and not on sentence-shaped
/// ones; that was an inference from what the corpora are, and this is what
/// turns it into a measurement.
#[derive(Default)]
struct Lengths {
    total: u64,
    longest: usize,
}

/// What a reranker has been asked to do since it was loaded.
///
/// Per process and per tier, like the reranker itself, and never reset -- so
/// on a resident server these are lifetime totals and in a harness they are
/// the run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reranked {
    /// Scores held in the cache.
    pub remembered: usize,
    /// Candidates handed to [`Reranker::rank`], cached or not.
    pub offered: u64,
    /// Candidates that reached the model, which is `offered` less cache hits.
    pub scored: u64,
    /// Characters across every candidate that reached the model.
    pub characters: u64,
    /// The longest single candidate that reached the model, in characters.
    pub longest: usize,
}

impl Reranker {
    /// Loads the tier's model, downloading it on first use.
    ///
    /// Through the same cache as the embedding model, so a workspace holds one
    /// directory of weights rather than two.
    pub fn load(tier: Rerank, cache_dir: &Path) -> Result<Self> {
        debug_assert!(tier != Rerank::Off, "the off tier loads nothing");
        std::fs::create_dir_all(cache_dir)?;

        let repository = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|error| IndexError::Engine(format!("reaching the model hub: {error}")))?
            .model(tier.repository().to_string());

        let fetch = |name: &str| -> Result<PathBuf> {
            repository.get(name).map_err(|error| {
                IndexError::Engine(format!(
                    "fetching {name} for the {} reranker: {error}",
                    tier.name()
                ))
            })
        };
        let read = |name: &str| -> Result<Vec<u8>> { Ok(std::fs::read(fetch(name)?)?) };

        let model = TextRerank::try_new_from_user_defined(
            UserDefinedRerankingModel::new(
                // By path rather than by bytes: the session maps the file, and
                // handing it a copy of half a gigabyte first serves no purpose.
                OnnxSource::File(fetch(tier.onnx())?),
                TokenizerFiles {
                    tokenizer_file: read("tokenizer.json")?,
                    config_file: read("config.json")?,
                    special_tokens_map_file: read("special_tokens_map.json")?,
                    tokenizer_config_file: read("tokenizer_config.json")?,
                },
            ),
            {
                let mut options = RerankInitOptionsUserDefined::new().with_max_length(max_tokens());
                // Same setting as the embedder's, and for the same reason:
                // see `crate::inference`.
                if let Some(threads) = crate::inference::threads() {
                    options = options.with_intra_threads(threads);
                }
                options
            },
        )
        .map_err(|error| IndexError::Engine(format!("loading the reranker: {error}")))?;

        Ok(Self {
            model,
            tier,
            scores: Scores::default(),
            lengths: Lengths::default(),
        })
    }

    pub fn tier(&self) -> Rerank {
        self.tier
    }

    /// Orders `documents` best first, returning their original positions.
    ///
    /// Batched by length rather than in the order given, because a batch is
    /// padded to its longest member and the caller's order is by relevance,
    /// which says nothing about length. The permutation is undone before the
    /// result is returned, so a caller sees positions into what it passed.
    pub fn rank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<usize>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let keys: Vec<u64> = documents
            .iter()
            .map(|document| Scores::key(query, document))
            .collect();
        let mut scores: Vec<Option<f32>> = keys
            .iter()
            .map(|key| self.scores.get(*key))
            .collect::<Vec<_>>();

        // Only what has not been scored before goes through the model, and
        // sorted by length, so that a batch is not padded to a length most of
        // its members do not have.
        //
        // By characters rather than by bytes. What the padding is measured in
        // is tokens, and `str::len` is UTF-8 bytes -- three per character for
        // the Chinese and Thai in this corpus against one for the Latin, so a
        // byte sort puts a short Thai candidate after a long English one and
        // the batches it forms are not the ones the saving assumes. Characters
        // are not tokens either, but they are within a small factor across
        // scripts where bytes are within three.
        //
        // Not score-neutral, and it was first recorded as though it were:
        // grouping candidates differently pads them differently, and a
        // quantized model scores a pair slightly differently depending on what
        // it shared a tensor with. Cross-lingual nDCG@10 moved from 0.6097 to
        // 0.6091 when this changed -- the fourth decimal, and in the direction
        // nobody would choose, but it is a real signed change rather than
        // noise. Both figures are at the lexical weight of the day, a quarter;
        // the same path scores 0.6480 at the eighth that ships now. See `BATCH`
        // for the same effect across batch sizes.
        let mut unscored: Vec<usize> = (0..documents.len())
            .filter(|position| scores[*position].is_none())
            .collect();
        unscored.sort_by_key(|position| documents[*position].chars().count());

        if !unscored.is_empty() {
            let batch: Vec<&str> = unscored
                .iter()
                .map(|position| documents[*position])
                .collect();
            for document in &batch {
                let characters = document.chars().count();
                self.lengths.total += characters as u64;
                self.lengths.longest = self.lengths.longest.max(characters);
            }
            let scored = self
                .model
                .rerank(query, &batch, false, Some(self::batch()))
                .map_err(|error| IndexError::Engine(format!("reranking: {error}")))?;

            for result in scored {
                let position = unscored[result.index];
                scores[position] = Some(result.score);
                self.scores.put(keys[position], result.score);
            }
        }

        let mut ordered: Vec<usize> = (0..documents.len()).collect();
        ordered.sort_by(|left, right| {
            scores[*right]
                .unwrap_or(f32::MIN)
                .total_cmp(&scores[*left].unwrap_or(f32::MIN))
                // A stable order when two candidates score alike, so one
                // shortlist ranks the same way twice.
                .then_with(|| left.cmp(right))
        });
        Ok(ordered)
    }

    /// What this reranker has been asked to do, and what it did.
    ///
    /// Exposed because three of the decisions this project has deferred turn
    /// on these five numbers and none of them had a value. The cache's hit
    /// rate is what says whether precomputing part of each document's
    /// representation at index time would pay for its storage. `scored`
    /// against `offered` is the size of the only lever proportional to the
    /// whole of the reranker's cost -- how many pairs reach the model at all,
    /// which is not the same as `DEPTH` and was never counted. And the lengths
    /// say whether `MAX_TOKENS` binds, which decides whether truncation is a
    /// lever or a rounding error on a given corpus.
    ///
    /// An accessor reporting occupancy alone came before this and answered
    /// none of them: it says how much has been stored and nothing about how
    /// often it is read.
    pub fn counted(&self) -> Reranked {
        Reranked {
            remembered: self.scores.known.len(),
            offered: self.scores.hits + self.scores.misses,
            scored: self.scores.misses,
            characters: self.lengths.total,
            longest: self.lengths.longest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_score_is_remembered_for_its_own_query_and_document() {
        let mut scores = Scores::default();
        let key = Scores::key("how does deployment work", "the pipeline signs artifacts");
        scores.put(key, 0.75);

        assert_eq!(scores.get(key), Some(0.75));
        assert_eq!(
            scores.get(Scores::key(
                "how does rollback work",
                "the pipeline signs artifacts"
            )),
            None,
            "a different query reused another query's score"
        );
        assert_eq!(
            scores.get(Scores::key(
                "how does deployment work",
                "backups run nightly"
            )),
            None,
            "a different document reused another document's score"
        );
    }

    /// The query and the document are hashed as two fields, not one string.
    ///
    /// Concatenated, "ab" + "c" and "a" + "bc" are the same bytes and would be
    /// the same score. Both splits are plausible: a query is a phrase and a
    /// memory begins with one.
    #[test]
    fn where_the_query_ends_and_the_document_begins_is_part_of_the_key() {
        assert_ne!(Scores::key("ab", "c"), Scores::key("a", "bc"));
    }

    #[test]
    fn the_oldest_score_is_forgotten_once_the_cache_is_full() {
        let mut scores = Scores::default();
        for n in 0..REMEMBERED_SCORES + 10 {
            scores.put(Scores::key("query", &n.to_string()), n as f32);
        }

        assert_eq!(scores.known.len(), REMEMBERED_SCORES);
        assert_eq!(
            scores.get(Scores::key("query", "0")),
            None,
            "the first score written was still there after the cache filled"
        );
        assert_eq!(
            scores.get(Scores::key("query", &(REMEMBERED_SCORES + 9).to_string())),
            Some((REMEMBERED_SCORES + 9) as f32),
            "the last score written was evicted"
        );
    }

    /// Rewriting a score must not queue its key a second time.
    ///
    /// It would evict an entry per rewrite while leaving the rewritten one in
    /// the map, so the cache would hold fewer and fewer live scores while
    /// reporting itself full.
    #[test]
    fn rewriting_a_score_does_not_shorten_the_cache() {
        let mut scores = Scores::default();
        let key = Scores::key("query", "document");
        for n in 0..100 {
            scores.put(key, n as f32);
        }

        assert_eq!(scores.order.len(), 1);
        assert_eq!(scores.get(key), Some(99.0));
    }

    /// Every tier that loads a model says what its weights may be used for.
    ///
    /// The point of asserting it rather than trusting the match is that adding
    /// a tier is a six-arm edit and the compiler catches five of them. This
    /// catches the sixth if it is ever written as a permissive default by
    /// reflex: a tier that loads weights must have an answer, and `off` must
    /// not, because "no weights" is a different statement from "weights you may
    /// use freely".
    #[test]
    fn every_tier_that_loads_weights_declares_what_they_may_be_used_for() {
        assert_eq!(Rerank::Off.licence(), None, "the off tier loads nothing");
        for tier in [Rerank::Fast, Rerank::Accurate, Rerank::Balanced] {
            assert!(
                tier.licence().is_some(),
                "the {} tier downloads weights and does not say under what terms",
                tier.name()
            );
        }
    }

    /// Whatever a tier parses from, it round-trips through its own name.
    ///
    /// Guards the pair of matches that a new tier has to touch together. The
    /// wire protocol is this string, so a name that parses to a different tier
    /// than it prints would route a caller to a model they did not ask for --
    /// and with a non-commercial tier in the list that is a licence question
    /// rather than a ranking one.
    #[test]
    fn a_tier_parses_from_the_name_it_prints() {
        for tier in [
            Rerank::Off,
            Rerank::Fast,
            Rerank::Accurate,
            Rerank::Balanced,
        ] {
            assert_eq!(
                Rerank::parse(tier.name()),
                Some(tier),
                "{} does not round-trip",
                tier.name()
            );
        }
    }
}
