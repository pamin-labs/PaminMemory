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
//! It was then built and measured, which ended it: over the exact candidates
//! the `accurate` tier is offered, reordering by MaxSim scored 0.5857 on
//! XQuAD-R's cross-lingual group against 0.6114 for not reranking at all and
//! 0.6597 for `accurate` -- worse than every tier, including `off`. The storage
//! it would have needed was never built, and the implementation was deleted.
//! `docs/adr/0001-tech-selection.md` has the table.
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
//! **`fast` was the default on latency, not on quality.** It is a twelve-layer
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
//! is not an interactive budget, which is why `fast` stayed the default -- and
//! on real passages that choice gave up three fifths of the available gain
//! rather than a fifth. The section below is why it no longer does.
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
//! ## Why `accurate` is the default
//!
//! The ranking this project works to is accuracy, then latency, then memory,
//! then disk. `fast` was chosen on the second; put to the first, query by query
//! against `accurate` through `search_reranked`:
//!
//! | corpus, group | `accurate` against `fast` | wins / losses / ties | p |
//! |---|---|---|---|
//! | XQuAD-R, cross-lingual | **+0.0086** | 1,190 queries | 0.0015 |
//! | XQuAD-R, same-language | **+0.0066** | 1,190 queries | 0.0001 |
//! | MIRACL Swahili, `speed` profile | **+0.0411** | 83 / 12 / 387 | 0.0001 |
//! | own corpus, cross-lingual | +0.0151 | 16 / 11 / 16 | 0.37 |
//! | own corpus, relational | +0.0096 | 3 / 1 / 16 | 0.63 |
//!
//! It is never worse, significantly better wherever there are enough queries
//! to say, and on the one non-parallel corpus the gap is four hundredths.
//! `fast` meanwhile loses to no reranking at all on both same-language
//! measurements: -0.0060 on XQuAD-R (p = 0.0002) and -0.0154 on MIRACL
//! (35 wins, 58 losses, p = 0.014) -- so the tier that shipped was, for any
//! query whose answer shares its language, worse than asking for nothing.
//!
//! What it costs is written down rather than argued away: 1522 ms a search on
//! XQuAD-R against 359, about a quarter of `fast`'s throughput, and 571 MB
//! loaded against 119. `--rerank fast` and `--rerank off` are how a workspace
//! that cannot afford it buys the time back, knowingly.
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
//! [`Reranker::counted`] measures reuse of complete batch contexts. It does
//! not measure document recurrence: changing neighbours can make a repeated
//! document a cache miss. Evaluating precomputed document layers requires a
//! separate document-recurrence measurement and a storage/cost comparison.
//!
//! What is kept is a complete ordered batch's logits, within one loaded model
//! and tokenizer. With this INT8 export, a pair's score can depend on its batch
//! neighbours and padding, so reusing it in a different context is incorrect.
//! The cache includes every pair identity and the logical and physical shapes;
//! changed contexts are scored again. An identical batch avoids a model forward
//! pass, but a repeated search still pays for tokenization, hashing, planning
//! and cache lookup, as well as the rest of retrieval. It is not a zero-cost
//! search. Loading another model or tokenizer creates a new cache.
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

use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Encoding;

use crate::encoder::Encoder;
use crate::error::{IndexError, Result};
use crate::hub::Repository;
use crate::inference::Device;

/// How much to spend reordering the shortlist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rerank {
    /// Return what fusion ordered.
    ///
    /// Keeps fusion's order without loading a cross-encoder.
    /// `fast` reorders candidates no lexical channel found; `accurate` can
    /// reorder the whole head.
    Off,
    /// A twelve-layer distilled multilingual MiniLM, 113 MB.
    ///
    /// The default until it was measured against the order it was chosen on:
    /// it reaches four fifths of `accurate`'s cross-lingual gain for a quarter
    /// of the latency, but it is the one tier that costs same-language
    /// ranking -- below no reranking at all, on both corpora that can say.
    /// The tier to ask for when a search must stay under half a second.
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
    /// 1725 ms a search against `fast`'s 332. That is the price of the default
    /// being the most accurate tier; the module note has the paired numbers
    /// that decided it.
    #[default]
    Accurate,
}

/// How many of the fused results a tier looks at.
///
/// Thirty, and it was twenty until the default tier was measured at it. The
/// twenty was settled on the `fast` model, where thirty bought +0.0009 of
/// XQuAD-R cross-lingual -- "twenty is simply where it stops". The default is
/// now `accurate`, a model fourteen times the size, and it keeps finding
/// answers further down. Paired against twenty through `search_reranked`,
/// nDCG@10 (the `DEPTH_VARIANTS` arm of each harness):
///
/// | depth | XQuAD-R cross (1,190) | own cross (43) | MIRACL (482) | MuSiQue (1,000) |
/// |---|---|---|---|---|
/// | 10 | **−0.0365**, p = 0.0001 | −0.0144 | −0.0021 | −0.0031, p = 0.069 |
/// | 15 | **−0.0096**, p = 0.0001 | +0.0027 | +0.0015 | −0.0004 |
/// | **30** | **+0.0063**, 192W/134L, p = 0.0001 | +0.0117 | −0.0008 | +0.0011 |
/// | 40 | **+0.0076**, p = 0.0001 | +0.0235, p = 0.062 | +0.0002 | +0.0007 |
///
/// Unmarked cells are not significant, and XQuAD-R's same-language group
/// moves by under 0.0006 at every depth. So shallower is refused -- fifteen
/// looked free on MIRACL and MuSiQue and costs a hundredth where the
/// reranker matters most -- and deeper is a significant gain on the
/// cross-lingual group and a loss nowhere. Thirty takes five sixths of forty's
/// gain for half its extra pairs: accuracy decides the direction and latency
/// the distance, and forty's last 0.0013 is not worth another third of the
/// reranker's time. The pass costs half again what it did at twenty in model
/// pairs. Its wall time at thirty is under [`BATCH_TOKENS`], taken on a
/// machine shared with other measurements rather than a quiet one, so it is
/// the ratios there that carry.
const DEPTH: usize = 30;

/// The tuning constants above, overridable for a sweep.
///
/// `DEPTH`, `BATCH` and `MAX_TOKENS` were each settled by measurement, and two
/// of the three were settled on the scratch harness whose figures turned out to
/// be about half again too high -- so they have to be re-settleable, and by the
/// harness rather than by editing a constant and rebuilding. Same shape as the
/// evaluation's own `SWEEP` and `TIERS`: unset means the constant, so nothing a
/// user runs is affected.
fn tuned(name: &str, fallback: usize) -> usize {
    pamin_core::env::positive(name).unwrap_or(fallback)
}

/// How many padded tokens go through the model at once.
///
/// Five hundred and twelve, with at most [`BATCH`] pairs, and the control is
/// the tokens rather than the count. A batch is padded to its longest member,
/// and on four cores a pair also costs more the more padded tokens share its
/// pass, so chunks of eight lost twice: to padding where short and long pairs
/// met, and to size where every pair was long -- MuSiQue's graph candidates,
/// which carry their seed's text, filled eight-pair passes with 256-token
/// rows. So every offered pair is tokenized once, sorted by its real length
/// and then its position, and each pass takes the next pair while the pass
/// stays within this many padded tokens.
///
/// Settled by a rule written down before anything was timed. A pilot of 24
/// queries a corpus ran seven candidates beside the batching this replaced --
/// sorted by characters, consecutive chunks of eight -- through
/// `Engine::search_reranked` at the shipped depth, all in one process, each
/// query through every candidate in rotated order. A whole search, as the
/// median of each query's ratio to chunks of eight:
///
/// | candidate | XQuAD-R | MIRACL | MuSiQue | pooled |
/// |---|---|---|---|---|
/// | chunks of four | 0.814 | 0.710 | 0.708 | 0.742 |
/// | 512 tokens | 0.766 | 0.576 | 0.556 | 0.626 |
/// | 1,024 tokens | 0.922 | 0.840 | 0.690 | 0.811 |
/// | 2,048 tokens | 1.292 | 1.109 | 1.057 | 1.149 |
/// | **512 tokens, four pairs** | **0.746** | **0.583** | **0.523** | **0.610** |
/// | 1,024 tokens, four pairs | 0.806 | 0.616 | 0.633 | 0.680 |
/// | 2,048 tokens, four pairs | 0.827 | 0.695 | 0.704 | 0.739 |
///
/// A budget of 2,048 with no cap is *slower* than chunks of eight: it packs a
/// dozen and more pairs into a pass, which pads more than chunks of eight did
/// and costs more a token besides.
///
/// The winner then went through every query, because grouping is not
/// score-neutral: the int8 export quantizes activations per tensor, so rows
/// padded together share one scale and a pair's score depends on its
/// neighbours. Paired against chunks of eight by sign-flip, 10,000 draws:
///
/// | group | queries | nDCG@10 | wins / losses | p | a search | faster on |
/// |---|---|---|---|---|---|---|
/// | XQuAD-R cross-lingual | 1,190 | −0.0004 | 166 / 192 | 0.60 | 0.833 | 1,053 of 1,187 |
/// | XQuAD-R same-language | 1,190 | −0.0002 | 5 / 6 | 0.62 | (the same searches) | |
/// | MIRACL Swahili | 482 | −0.0011 | 13 / 11 | 0.42 | 0.800 | 461 of 482 |
/// | MuSiQue, 2-hop | 1,000 | +0.0005 | 27 / 17 | 0.63 | 0.810 | 944 of 997 |
///
/// Not significantly worse in any group, and faster on 2,458 of 2,666 pooled
/// searches with a pooled ratio of 0.814, so it ships. The pilot's larger
/// saving was taken at a load average of 11 to 12 on four cores and this at 6
/// to 8; the saving grows with contention, and it is the paired ratio rather
/// than either set of milliseconds that carries. Measured on the `accurate`
/// tier; `fast` batches the same way and its scores were not re-measured.
const BATCH_TOKENS: usize = 512;

/// How many pairs go through the model at once, at most. See [`BATCH_TOKENS`].
const BATCH: usize = 4;

/// How much the model reads at once, and how long a candidate. See [`tuned`].
fn batch_tokens() -> usize {
    tuned("PAMIN_RERANK_BATCH_TOKENS", BATCH_TOKENS)
}

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
/// it costs 0.0014 of cross-lingual ranking rather than nothing. A batch is
/// padded to its longest member and not to this limit, so the limit only
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
/// Characters rather than tokens, as [`Reranker::counted`] reports them. A
/// 5,567-character passage is far past this limit
/// whatever the script, so on MIRACL the truncation is doing real work, and the
/// sentence-corpus measurement above says nothing about what it costs there.
///
/// **So 192 and 384 were run where it binds, and neither ships.** On MuSiQue
/// half the pairs reach this limit, because a graph candidate carries its
/// seed's text. Every query of each corpus, at a depth of thirty and batched
/// by [`BATCH_TOKENS`], each limit paired against 256 in the same process,
/// sign-flip over 10,000 draws; a search is the median of each query's ratio:
///
/// | tokens | XQuAD-R cross-lingual | MIRACL | MuSiQue | a search: XQuAD-R, MIRACL, MuSiQue |
/// |---|---|---|---|---|
/// | 192 | −0.0003 (p 0.07) | −0.0005 (p 0.44) | **−0.0045 (p 0.010)** | 0.993, 0.946, 0.801 |
/// | **256** | **0.6676** | **0.7883** | **0.6842** | **1** |
/// | 384 | +0.0000 (p 0.50) | +0.0006 (p 0.25) | −0.0016 (p 0.31) | 1.002, 1.029, 1.321 |
///
/// Same-language moved by nothing either way. The rule, written down before
/// the runs, shipped 384 only if it was significantly better somewhere and
/// worse nowhere, and 192 only if it was worse nowhere: 384 is better
/// nowhere, and costs a third more a search on MuSiQue for a reading of the
/// seed's text that did not help; 192 is a fifth cheaper there and loses
/// significantly on exactly the corpus where it truncates.
const MAX_TOKENS: usize = 256;

impl Rerank {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "off" => Some(Self::Off),
            "fast" => Some(Self::Fast),
            "accurate" => Some(Self::Accurate),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fast => "fast",
            Self::Accurate => "accurate",
        }
    }

    /// How deep into the fused list this tier reaches.
    pub fn depth(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Fast | Self::Accurate => tuned("PAMIN_RERANK_DEPTH", DEPTH),
        }
    }

    fn repository(self) -> &'static str {
        match self {
            Self::Off => unreachable!("nothing is loaded for the off tier"),
            Self::Fast => "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            Self::Accurate => "onnx-community/bge-reranker-v2-m3-ONNX",
        }
    }

    /// Which export of the model to fetch, for the device it will run on.
    ///
    /// On an accelerator, the half-precision export where there is one: an
    /// int8 graph of `MatMulInteger` and `DynamicQuantizeLinear` is a CPU
    /// format that GPU providers run poorly or partly on the CPU anyway. The
    /// fast model has no half-precision export and is small enough that full
    /// precision costs a GPU nothing worth counting.
    ///
    /// On the CPU, the fast model publishes one quantized export per instruction set, and
    /// they are not interchangeable in speed: on a machine with AVX-512 VNNI
    /// the VNNI build is 1.5x the AVX2 one for the same scores. Picking at
    /// runtime rather than at build time, because a binary is built once and
    /// run on whatever is there.
    fn onnx(self, device: Device) -> &'static str {
        if device != Device::Cpu {
            return match self {
                Self::Off => unreachable!("nothing is loaded for the off tier"),
                Self::Fast => "onnx/model.onnx",
                Self::Accurate => "onnx/model_fp16.onnx",
            };
        }
        match self {
            Self::Off => unreachable!("nothing is loaded for the off tier"),
            // One int8 export, not one per instruction set, so there is
            // nothing to detect at runtime the way `fast` has to.
            Self::Accurate => "onnx/model_int8.onnx",
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

/// How many logical pair scores are remembered across complete batches.
///
/// The cache belongs to one loaded model. A repeated list can reuse its
/// batches, while a changed neighbour or padded shape must be scored again:
/// this model's dynamic int8 quantization is not pair-independent.
const REMEMBERED_SCORES: usize = 4096;

type PairKey = [u8; 32];

fn pair_key(query: &str, document: &str) -> PairKey {
    let mut hash = Sha256::new();
    for text in [query, document] {
        hash.update((text.len() as u64).to_le_bytes());
        hash.update(text.as_bytes());
    }
    hash.finalize().into()
}

/// Complete ordered batch identity, within the immutable loaded model.
/// No document text or token encodings are kept after a rank call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BatchKey([u8; 32]);

impl BatchKey {
    fn of(pairs: &[PairKey], logical: (usize, usize), physical: (usize, usize)) -> Self {
        let mut hash = Sha256::new();
        for value in [logical.0, logical.1, physical.0, physical.1] {
            hash.update((value as u64).to_le_bytes());
        }
        for pair in pairs {
            hash.update(pair);
        }
        Self(hash.finalize().into())
    }
}

/// Scores already computed, oldest complete batch first. Capacity counts
/// logical score slots rather than batch entries. Hashes include duplicates
/// and row order, because both can change the model's activation range.
#[derive(Default)]
struct Scores {
    known: std::collections::HashMap<BatchKey, Vec<f32>>,
    order: std::collections::VecDeque<BatchKey>,
    remembered: usize,
    hits: u64,
    misses: u64,
}

impl Scores {
    fn get(&self, key: BatchKey) -> Option<&[f32]> {
        self.known.get(&key).map(Vec::as_slice)
    }

    /// Whole-batch FIFO. An oversized batch neither enters the cache nor
    /// purges entries it could not coexist with even in an empty cache.
    fn put(&mut self, key: BatchKey, values: Vec<f32>) {
        if values.len() > REMEMBERED_SCORES {
            return;
        }
        if let Some(previous) = self.known.get_mut(&key) {
            debug_assert_eq!(previous.len(), values.len());
            *previous = values;
            return;
        }
        while self.remembered + values.len() > REMEMBERED_SCORES {
            let oldest = self.order.pop_front().expect("nonempty score cache");
            self.remembered -= self.known.remove(&oldest).expect("queued batch").len();
        }
        self.remembered += values.len();
        self.known.insert(key, values);
        self.order.push_back(key);
    }

    /// Consult only fully planned batches. Nothing is committed until every
    /// missed batch succeeds, so a later model failure cannot leave partial
    /// scores, cache entries or lifetime counters behind.
    fn score(
        &self,
        planned: Vec<Batch>,
        keys: &[PairKey],
        characters: impl Fn(usize) -> usize,
        mut forward: impl FnMut(Vec<Encoding>) -> Result<(Vec<f32>, u64)>,
    ) -> Result<Attempt> {
        let mut attempt = Attempt {
            values: vec![f32::MIN; keys.len()],
            ..Attempt::default()
        };
        for batch in planned {
            let count = batch.positions.len();
            let identities: Vec<_> = batch.positions.iter().map(|at| keys[*at]).collect();
            let key = BatchKey::of(&identities, (count, batch.longest()), batch.shape);
            if let Some(values) = self.get(key) {
                attempt.hits += count as u64;
                for (position, value) in batch.positions.iter().zip(values) {
                    attempt.values[*position] = *value;
                }
                continue;
            }
            attempt.misses += count as u64;
            attempt.work.pairs += count as u64;
            attempt.work.tokens += batch.encodings.iter().map(|e| e.len() as u64).sum::<u64>();
            attempt.work.padded_tokens += (batch.shape.0 * batch.shape.1) as u64;
            attempt.work.batches += 1;
            for position in &batch.positions {
                let length = characters(*position);
                attempt.lengths.total += length as u64;
                attempt.lengths.longest = attempt.lengths.longest.max(length);
            }
            let (values, forward_us) = forward(batch.encodings)?;
            attempt.work.forward_us += forward_us;
            if values.len() != count {
                return Err(IndexError::Engine(
                    "reranker returned the wrong number of scores".into(),
                ));
            }
            for (position, value) in batch.positions.iter().zip(&values) {
                attempt.values[*position] = *value;
            }
            attempt.pending.push((key, values));
        }
        Ok(attempt)
    }
}

#[derive(Default)]
struct Attempt {
    values: Vec<f32>,
    pending: Vec<(BatchKey, Vec<f32>)>,
    hits: u64,
    misses: u64,
    work: Work,
    lengths: Lengths,
}

impl Attempt {
    fn commit(
        self,
        scores: &mut Scores,
        work: &mut Work,
        lengths: &mut Lengths,
        encode_us: u64,
    ) -> Vec<f32> {
        for (key, values) in self.pending {
            scores.put(key, values);
        }
        scores.hits += self.hits;
        scores.misses += self.misses;
        work.add(self.work);
        work.encode_us += encode_us;
        lengths.total += self.lengths.total;
        lengths.longest = lengths.longest.max(self.lengths.longest);
        self.values
    }
}

/// One candidate's place in the reranked order, and what the model scored it.
///
/// The score is here because the position alone cannot answer the question
/// that decides whether to keep a candidate at all: a shortlist where the
/// model put daylight between the second and the third is a different
/// shortlist from one where it could barely separate them, and both look
/// identical as an order. That is what the 2025 threshold result turns on --
/// dropping candidates below a cut rather than reordering all of them -- and
/// this project could not have measured it, because `rank` computed these
/// scores and discarded them.
///
/// **Never comparable across queries.** A cross-encoder's logit is calibrated
/// against nothing; it separates candidates within one shortlist and says
/// nothing absolute. A cut expressed against the other candidates of the same
/// query is expressible from this; a fixed threshold is not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ranked {
    /// Index into the slice handed to [`Reranker::rank`].
    pub position: usize,
    /// The model's own score, larger being more relevant.
    pub score: f32,
}

/// A loaded reranker, and what it has already scored.
pub struct Reranker {
    model: Encoder,
    tier: Rerank,
    device: Device,
    scores: Scores,
    lengths: Lengths,
    work: Work,
}

#[derive(Default)]
struct Work {
    pairs: u64,
    tokens: u64,
    padded_tokens: u64,
    batches: u64,
    encode_us: u64,
    forward_us: u64,
}

impl Work {
    fn add(&mut self, completed: Self) {
        self.pairs += completed.pairs;
        self.tokens += completed.tokens;
        self.padded_tokens += completed.padded_tokens;
        self.batches += completed.batches;
        self.encode_us += completed.encode_us;
        self.forward_us += completed.forward_us;
    }
}

/// How long the candidates that reached the model were.
///
/// Counted in characters rather than bytes, because UTF-8 is three bytes a
/// character for the Chinese and Thai in these corpora against one for the
/// Latin, and in characters rather than tokens so that these totals stay
/// comparable with the ones recorded before the batching counted tokens.
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
    /// Actual truncation limit of this loaded model, before batch padding.
    pub maximum_tokens: usize,
    /// Scores held in the cache.
    pub remembered: usize,
    /// Candidates in successfully completed [`Reranker::rank`] calls, cached or not.
    pub offered: u64,
    /// Candidates whose uncached model score completed successfully.
    pub scored: u64,
    /// Characters across every candidate that reached the model.
    pub characters: u64,
    /// The longest single candidate that reached the model, in characters.
    pub longest: usize,
    /// Tokens in scored pairs before batch padding.
    pub tokens: u64,
    /// Tokens actually passed to the model, including padding.
    pub padded_tokens: u64,
    /// Model batches run for uncached pairs.
    pub batches: u64,
    /// Time spent encoding all offered pairs in successful calls, in microseconds.
    /// Cached batches still need tokenization to establish their exact context.
    pub encode_us: u64,
    /// Time spent padding and running those batches, in microseconds.
    pub forward_us: u64,
}

impl Reranker {
    /// Loads the tier's model, downloading it on first use.
    ///
    /// Through the same cache as the embedding model, so a workspace holds one
    /// directory of weights rather than two.
    pub fn load(tier: Rerank, cache_dir: &Path) -> Result<Self> {
        debug_assert!(tier != Rerank::Off, "the off tier loads nothing");
        std::fs::create_dir_all(cache_dir)?;

        let repository = Repository::open(cache_dir, tier.repository())?;

        let session = |device: Device, providers| -> Result<Encoder> {
            let weights = repository.file(cache_dir, tier.onnx(device));
            let model = || match device {
                // The file the hub serves is copied onto the heap whole; on
                // the CPU, the prepared copy is mapped instead -- see
                // `crate::prepared` for what that saves -- and the download
                // removed once the copy has loaded.
                Device::Cpu => crate::prepared::load_path(&weights, cache_dir),
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                Device::CoreMl if tier == Rerank::Accurate => {
                    let source = repository.get(tier.onnx(device))?;
                    crate::native::prepare(&source, cache_dir)
                }
                _ => repository.get(tier.onnx(device)),
            };
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let loaded = if device == Device::CoreMl && tier == Rerank::Accurate {
                Encoder::load_fixed_coreml(model, &repository, max_tokens())
            } else {
                Encoder::load(model, &repository, max_tokens(), providers)
            };
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let loaded = Encoder::load(model, &repository, max_tokens(), providers);
            let encoder = loaded
                .map_err(|error| IndexError::Engine(format!("loading the reranker: {error}")))?;
            #[cfg(target_os = "windows")]
            let mut encoder = encoder;
            #[cfg(target_os = "windows")]
            if device == Device::DirectMl {
                // Compare the same accelerator export, not CPU int8 versus
                // accelerator FP16: quantization is a separate source of drift.
                let path = repository.get(tier.onnx(device))?;
                check_accelerator(&mut encoder, || {
                    Encoder::load(
                        || Ok(path.clone()),
                        &repository,
                        max_tokens(),
                        vec![crate::inference::cpu()],
                    )
                })?;
            }
            if device == Device::Cpu {
                crate::prepared::release(&weights, cache_dir);
            }
            Ok(encoder)
        };

        // On Apple Silicon, the Fast model's ARM INT8 CPU export preserved
        // XQuAD-R quality and beat its CoreML FP32 export on every paired
        // search. Accurate still uses the shared CoreML-first policy.
        let (model, device) =
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) && tier == Rerank::Fast {
                (
                    session(Device::Cpu, vec![crate::inference::cpu()])?,
                    Device::Cpu,
                )
            } else {
                crate::inference::preferred(session)?
            };
        tracing::info!(
            tier = tier.name(),
            device = device.name(),
            maximum_tokens = model.maximum_tokens(),
            "reranker loaded"
        );

        Ok(Self {
            model,
            tier,
            device,
            scores: Scores::default(),
            lengths: Lengths::default(),
            work: Work::default(),
        })
    }

    /// Whether the tier's weights for the CPU are on disk already, so that
    /// loading it reads a file rather than fetching half a gigabyte: the
    /// download, or the mapped copy that replaced it (see `crate::prepared`).
    ///
    /// For a caller loading a tier nobody has asked for yet -- a resident
    /// server warming a project -- which should not be what downloads it.
    pub fn is_downloaded(tier: Rerank, cache_dir: &Path) -> bool {
        tier != Rerank::Off
            && Repository::open(cache_dir, tier.repository()).is_ok_and(|repository| {
                crate::prepared::is_ready(
                    &repository.file(cache_dir, tier.onnx(Device::Cpu)),
                    cache_dir,
                )
            })
    }

    /// Where this reranker's passes run. See [`Device`].
    pub fn device(&self) -> Device {
        self.device
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
    pub fn rank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<Ranked>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let keys: Vec<_> = documents
            .iter()
            .map(|document| pair_key(query, document))
            .collect();
        // Tokenize every offered pair before lookup. Filtering hits first
        // changes the missed pairs' batch context and therefore their logits.
        let reranking = |error: IndexError| IndexError::Engine(format!("reranking: {error}"));
        let start = Instant::now();
        let encodings = self
            .model
            .encode(
                documents
                    .iter()
                    .map(|document| (query, *document))
                    .collect::<Vec<_>>(),
            )
            .map_err(reranking)?;
        let encode_us = start.elapsed().as_micros() as u64;
        let planned = model_batches(&self.model, encodings, batch_tokens(), batch());
        let attempt = self
            .scores
            .score(
                planned,
                &keys,
                |at| documents[at].chars().count(),
                |batch| forward(&mut self.model, batch),
            )
            .map_err(reranking)?;
        let scores = attempt.commit(
            &mut self.scores,
            &mut self.work,
            &mut self.lengths,
            encode_us,
        );

        let mut ordered: Vec<usize> = (0..documents.len()).collect();
        ordered.sort_by(|left, right| {
            scores[*right]
                .total_cmp(&scores[*left])
                // A stable order when two candidates score alike, so one
                // shortlist ranks the same way twice.
                .then_with(|| left.cmp(right))
        });
        Ok(ordered
            .into_iter()
            .map(|position| Ranked {
                position,
                score: scores[position],
            })
            .collect())
    }

    /// What this reranker has been asked to do, and what it did.
    ///
    /// `offered - scored` counts logical pairs reused in complete cached
    /// batch contexts. It does not measure a document hot set: changing batch
    /// neighbours can require scoring recurring documents again. Document
    /// precomputation needs a separate recurrence metric and storage/cost
    /// measurements. `scored` records actual model work, while token lengths
    /// show whether `MAX_TOKENS` binds on the offered corpus.
    ///
    /// An accessor reporting occupancy alone came before this and answered
    /// none of them: it says how much has been stored and nothing about how
    /// often it is read.
    pub fn counted(&self) -> Reranked {
        assert!(
            self.work.tokens <= self.work.padded_tokens,
            "reranker work counters disagree"
        );
        Reranked {
            maximum_tokens: self.model.maximum_tokens(),
            remembered: self.scores.remembered,
            offered: self.scores.hits + self.scores.misses,
            scored: self.work.pairs,
            characters: self.lengths.total,
            longest: self.lengths.longest,
            tokens: self.work.tokens,
            padded_tokens: self.work.padded_tokens,
            batches: self.work.batches,
            encode_us: self.work.encode_us,
            forward_us: self.work.forward_us,
        }
    }
}

/// A startup ordering guard for the actual model/export. A failed attempt
/// returns to `preferred`, which tries the next viable provider. No persistent
/// CPU-only setting is written; a later load can retry a repaired accelerator.
#[cfg(target_os = "windows")]
fn check_accelerator(
    accelerator: &mut Encoder,
    reference: impl FnOnce() -> Result<Encoder>,
) -> Result<()> {
    const PAIRS: [(&str, &str); 4] = [
        (
            "Where does the harbour pilot board ships?",
            "The harbour pilot boards ships at the outer buoy.",
        ),
        (
            "Where does the harbour pilot board ships?",
            "Chocolate cake is baked with flour and cocoa.",
        ),
        (
            "部署流水线在哪里运行？",
            "部署流水线运行在持续集成服务器上。",
        ),
        ("部署流水线在哪里运行？", "巧克力蛋糕使用面粉和可可粉烘焙。"),
    ];
    let encodings = accelerator.encode(PAIRS.to_vec())?;
    let observed = score(accelerator, encodings, batch_tokens(), batch())?.0;
    let mut reference = reference()?;
    let encodings = reference.encode(PAIRS.to_vec())?;
    let expected = score(&mut reference, encodings, batch_tokens(), batch())?.0;
    check_accelerator_ordering(&expected, &observed)
}

#[cfg(any(target_os = "windows", test))]
fn check_accelerator_ordering(expected: &[f32], observed: &[f32]) -> Result<()> {
    let failed =
        || IndexError::Engine("accelerator failed the startup reranker ordering fixture".into());
    if expected.len() != 4
        || observed.len() != 4
        || expected
            .iter()
            .chain(observed)
            .any(|score| !score.is_finite())
    {
        return Err(failed());
    }
    for (expected, observed) in expected
        .as_chunks::<2>()
        .0
        .iter()
        .zip(observed.as_chunks::<2>().0)
    {
        let order = expected[0].total_cmp(&expected[1]);
        if order.is_eq() || order != observed[0].total_cmp(&observed[1]) {
            return Err(failed());
        }
    }
    Ok(())
}

/// The model's score for each of `encodings` -- (query, document) pairs from
/// [`Encoder::encode`] -- in their order.
///
/// Shortest first, in the batches [`batches`] makes of their lengths, each
/// batch one forward pass and a pair's score the first column of its row of
/// `logits`. Unsorted, since [`Reranker::rank`] orders by score itself.
#[cfg(target_os = "windows")]
fn score(
    model: &mut Encoder,
    encodings: Vec<Encoding>,
    budget: usize,
    most: usize,
) -> Result<(Vec<f32>, Work)> {
    let count = encodings.len();
    let planned = model_batches(model, encodings, budget, most);
    let mut values = vec![f32::MIN; count];
    let mut work = Work::default();
    for batch in planned {
        work.batches += 1;
        work.padded_tokens += (batch.shape.0 * batch.shape.1) as u64;
        let (scores, forward_us) = forward(model, batch.encodings)?;
        work.forward_us += forward_us;
        for (position, score) in batch.positions.iter().zip(scores) {
            values[*position] = score;
        }
    }
    Ok((values, work))
}

/// Ephemeral plan: encodings are moved into batches and dropped after this
/// call. Cache entries retain only complete identities and logits.
struct Batch {
    positions: Vec<usize>,
    encodings: Vec<Encoding>,
    shape: (usize, usize),
}

impl Batch {
    fn longest(&self) -> usize {
        self.encodings.last().expect("nonempty batch").len()
    }
}

fn model_batches(
    model: &Encoder,
    encodings: Vec<Encoding>,
    budget: usize,
    most: usize,
) -> Vec<Batch> {
    let (budget, most) = model.batch_limits(budget, most);
    plan_batches(
        encodings,
        budget,
        most,
        |length| model.batching_length(length),
        |rows, length| model.execution_shape(rows, length),
    )
}

/// The shared fresh/cached planner. Sort by real length then original
/// position; group under effective padded-token and pair caps before lookup.
fn plan_batches(
    encodings: Vec<Encoding>,
    budget: usize,
    most: usize,
    batching_length: impl Fn(usize) -> usize,
    execution_shape: impl Fn(usize, usize) -> (usize, usize),
) -> Vec<Batch> {
    let mut sorted: Vec<_> = encodings.into_iter().enumerate().collect();
    sorted.sort_by_key(|(position, encoding)| (encoding.len(), *position));
    let lengths: Vec<_> = sorted
        .iter()
        .map(|(_, encoding)| batching_length(encoding.len()))
        .collect();
    let mut pending = sorted.into_iter();
    batches(&lengths, budget, most)
        .into_iter()
        .map(|size| {
            let (positions, encodings): (Vec<_>, Vec<_>) = pending.by_ref().take(size).unzip();
            let shape = execution_shape(size, encodings.last().expect("nonempty batch").len());
            Batch {
                positions,
                encodings,
                shape,
            }
        })
        .collect()
}

/// Actual Encoder kernel and unchanged first-column logit extraction,
/// shared by fresh startup checks and production cached inference.
fn forward(model: &mut Encoder, batch: Vec<Encoding>) -> Result<(Vec<f32>, u64)> {
    let count = batch.len();
    let start = Instant::now();
    let outputs = model.run_encoded(batch)?;
    let forward_us = start.elapsed().as_micros() as u64;
    let logits = outputs
        .get("logits")
        .ok_or_else(|| IndexError::Engine("the reranker returned no logits".into()))?;
    let (shape, values) = logits
        .try_extract_tensor::<f32>()
        .map_err(|error| IndexError::Engine(format!("reading the logits: {error}")))?;
    let labels = match **shape {
        [rows, labels] if rows as usize == count && labels > 0 => labels as usize,
        _ => {
            return Err(IndexError::Engine(format!(
                "logits of shape {shape:?} for {count} pairs"
            )));
        }
    };
    Ok((
        values.chunks(labels).map(|row| row[0]).collect(),
        forward_us,
    ))
}

/// How many pairs go in each batch, in order, for pairs of these `lengths`
/// in tokens, shortest first.
///
/// A batch is padded to its longest member, which in ascending order is the
/// last one it took, so a batch of `n` costs `n` times that length. Each takes
/// the next pair while that stays within `budget` padded tokens and under
/// `most` pairs. A pair longer than the budget on its own goes alone rather
/// than not at all.
fn batches(lengths: &[usize], budget: usize, most: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut size = 0;
    for length in lengths {
        if size > 0 && (size == most || (size + 1) * length > budget) {
            sizes.push(size);
            size = 0;
        }
        size += 1;
    }
    if size > 0 {
        sizes.push(size);
    }
    sizes
}

#[cfg(test)]
mod tests {

    #[test]
    fn accelerator_startup_checks_ordering_without_rejecting_score_drift() {
        let reference = [3.0, -2.0, 4.0, -1.0];
        assert!(super::check_accelerator_ordering(&reference, &[3.1, -1.9, 4.2, -0.8]).is_ok());
        assert!(super::check_accelerator_ordering(&reference, &[-2.0, 3.0, 4.0, -1.0]).is_err());
        assert!(
            super::check_accelerator_ordering(&reference, &[f32::NAN, -2.0, 4.0, -1.0]).is_err()
        );
        assert!(super::check_accelerator_ordering(&reference, &[3.0]).is_err());
        assert!(super::check_accelerator_ordering(&[0.0; 4], &[0.0; 4]).is_err());
    }
    use super::*;

    fn encoding(id: u32, tokens: usize) -> Encoding {
        Encoding::from_tokens(
            (0..tokens)
                .map(|_| tokenizers::Token {
                    id,
                    value: id.to_string(),
                    offsets: (0, 1),
                })
                .collect(),
            0,
        )
    }

    fn planned(rows: &[(u32, usize)], budget: usize, most: usize) -> Vec<Batch> {
        plan_batches(
            rows.iter().map(|(id, len)| encoding(*id, *len)).collect(),
            budget,
            most,
            |len| len,
            |rows, len| (rows, len),
        )
    }

    fn identities(rows: &[(u32, usize)]) -> Vec<PairKey> {
        rows.iter()
            .map(|(id, _)| pair_key("query", &id.to_string()))
            .collect()
    }

    // Deliberately context-sensitive: neighbours, order and longest row
    // affect every logit. A pair-only cache fails these history assertions.
    fn context(rows: Vec<Encoding>) -> Result<(Vec<f32>, u64)> {
        let offset: u32 = rows
            .iter()
            .enumerate()
            .map(|(at, e)| e.get_ids()[0] * (at as u32 + 1))
            .sum();
        let width = rows.iter().map(Encoding::len).max().unwrap();
        Ok((
            rows.iter()
                .map(|e| e.get_ids()[0] as f32 + offset as f32 * 10.0 + width as f32)
                .collect(),
            7,
        ))
    }

    fn cached(
        cache: &mut Scores,
        work: &mut Work,
        lengths: &mut Lengths,
        rows: &[(u32, usize)],
        budget: usize,
        most: usize,
    ) -> Vec<f32> {
        let attempt = cache
            .score(
                planned(rows, budget, most),
                &identities(rows),
                |at| at + 3,
                context,
            )
            .unwrap();
        attempt.commit(cache, work, lengths, 7)
    }

    fn bits(scores: &[f32]) -> Vec<u32> {
        scores.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn changed_neighbours_match_the_same_fresh_list_in_both_cache_histories() {
        let a = [(1, 3), (2, 3), (3, 3), (4, 3), (5, 12), (6, 12)];
        let b = [(1, 3), (2, 3), (3, 3), (9, 3), (5, 12), (6, 12)];
        let fresh = |rows: &[(u32, usize)]| {
            cached(
                &mut Scores::default(),
                &mut Work::default(),
                &mut Lengths::default(),
                rows,
                48,
                4,
            )
        };
        let fa = fresh(&a);
        let fb = fresh(&b);
        assert_ne!(
            fa[0].to_bits(),
            fb[0].to_bits(),
            "fixture must expose neighbour-sensitive logits"
        );
        for (first, second, expected) in [(&a, &b, &fb), (&b, &a, &fa)] {
            let mut cache = Scores::default();
            let mut work = Work::default();
            let mut lengths = Lengths::default();
            cached(&mut cache, &mut work, &mut lengths, first, 48, 4);
            let before = (work.pairs, work.tokens, work.padded_tokens, work.batches);
            let observed = cached(&mut cache, &mut work, &mut lengths, second, 48, 4);
            assert_eq!(bits(&observed), bits(expected));
            assert_eq!(
                work.pairs - before.0,
                4,
                "entire changed four-pair group must be rescored"
            );
            assert_eq!(cache.hits, 2, "unchanged two-pair group can hit");
            assert_eq!(cache.misses, 10);
            assert_eq!(
                (
                    work.tokens - before.1,
                    work.padded_tokens - before.2,
                    work.batches - before.3
                ),
                (12, 12, 1)
            );
        }
    }

    #[test]
    fn identical_hot_lists_retokenize_without_any_new_forward_work() {
        let rows = [(3, 6), (1, 2), (2, 2), (4, 6)];
        let mut cache = Scores::default();
        let mut work = Work::default();
        let mut lengths = Lengths::default();
        let cold = cached(&mut cache, &mut work, &mut lengths, &rows, 16, 2);
        let before = (
            work.pairs,
            work.tokens,
            work.padded_tokens,
            work.batches,
            work.forward_us,
            lengths.total,
            lengths.longest,
        );
        let hot = cache
            .score(
                planned(&rows, 16, 2),
                &identities(&rows),
                |_| panic!("hit counted as model character work"),
                |_| panic!("hot batch went through the model"),
            )
            .unwrap()
            .commit(&mut cache, &mut work, &mut lengths, 11);
        assert_eq!(bits(&cold), bits(&hot));
        assert_eq!(cache.hits, 4);
        assert_eq!(cache.misses, 4);
        assert_eq!(cache.remembered, 4);
        assert_eq!(
            (
                work.pairs,
                work.tokens,
                work.padded_tokens,
                work.batches,
                work.forward_us,
                lengths.total,
                lengths.longest
            ),
            before
        );
        assert_eq!(
            work.encode_us, 18,
            "all-pair tokenization is recorded even on cache hits"
        );
    }

    #[test]
    fn query_fields_duplicates_order_and_both_shapes_are_cache_identity() {
        assert_ne!(pair_key("ab", "c"), pair_key("a", "bc"));
        assert_ne!(pair_key("", "a"), pair_key("a", ""));
        assert_ne!(pair_key("", ""), pair_key("", "a"));
        assert_eq!(pair_key("", ""), pair_key("", ""));
        assert_ne!(pair_key("q", "a"), pair_key("r", "a"));
        let a = pair_key("q", "a");
        let b = pair_key("q", "b");
        let key = BatchKey::of(&[a, b], (2, 5), (4, 64));
        for other in [
            BatchKey::of(&[b, a], (2, 5), (4, 64)),
            BatchKey::of(&[a, a], (2, 5), (4, 64)),
            BatchKey::of(&[a], (1, 5), (4, 64)),
            BatchKey::of(&[a, b], (2, 6), (4, 64)),
            BatchKey::of(&[a, b], (2, 5), (4, 128)),
            BatchKey::of(&[a, b], (2, 5), (2, 64)),
        ] {
            assert_ne!(key, other);
        }
        let rows = [(1, 4), (1, 4), (2, 4)];
        let mut cache = Scores::default();
        let mut work = Work::default();
        let mut lengths = Lengths::default();
        let cold = cached(&mut cache, &mut work, &mut lengths, &rows, 64, 4);
        let hot = cached(&mut cache, &mut work, &mut lengths, &rows, 64, 4);
        assert_eq!(bits(&cold), bits(&hot));
        assert_eq!(cache.remembered, 3);
        assert_eq!(cache.hits, 3);
        assert_eq!(work.pairs, 3);
    }

    #[test]
    fn changed_order_and_caps_recompute_complete_groups_not_just_new_pairs() {
        let a = [(1, 4), (2, 4), (3, 4), (4, 4)];
        let b = [(2, 4), (1, 4), (3, 4), (4, 4)];
        let mut cache = Scores::default();
        let mut work = Work::default();
        let mut lengths = Lengths::default();
        cached(&mut cache, &mut work, &mut lengths, &a, 64, 4);
        let warmed = cached(&mut cache, &mut work, &mut lengths, &b, 64, 4);
        let fresh = cached(
            &mut Scores::default(),
            &mut Work::default(),
            &mut Lengths::default(),
            &b,
            64,
            4,
        );
        assert_eq!(bits(&warmed), bits(&fresh));
        assert_eq!(work.pairs, 8);
        assert_eq!(cache.hits, 0);
        cached(&mut cache, &mut work, &mut lengths, &a, 64, 2);
        assert_eq!(work.pairs, 12);
        assert_eq!(work.batches, 4, "new cap creates two new complete batches");
    }

    #[test]
    fn widening_reuses_only_complete_unchanged_groups() {
        let a = [(1, 4), (2, 4), (3, 4)];
        let b = [(1, 4), (2, 4), (3, 4), (4, 4)];
        let mut cache = Scores::default();
        let mut work = Work::default();
        let mut lengths = Lengths::default();
        cached(&mut cache, &mut work, &mut lengths, &a, 64, 2);
        let result = cached(&mut cache, &mut work, &mut lengths, &b, 64, 2);
        let fresh = cached(
            &mut Scores::default(),
            &mut Work::default(),
            &mut Lengths::default(),
            &b,
            64,
            2,
        );
        assert_eq!(bits(&result), bits(&fresh));
        assert_eq!(cache.hits, 2);
        assert_eq!(work.pairs, 5);
    }

    #[test]
    fn a_late_failure_commits_no_partial_cache_or_counters() {
        let a = [(1, 3), (2, 3)];
        let b = [(1, 3), (2, 3), (3, 4), (4, 4), (5, 5), (6, 5)];
        let mut cache = Scores::default();
        let mut work = Work::default();
        let mut lengths = Lengths::default();
        cached(&mut cache, &mut work, &mut lengths, &a, 64, 2);
        let before = (
            (
                cache.hits,
                cache.misses,
                cache.remembered,
                cache.order.clone(),
                cache.known.clone(),
            ),
            (
                work.pairs,
                work.tokens,
                work.padded_tokens,
                work.batches,
                work.encode_us,
                work.forward_us,
            ),
            (lengths.total, lengths.longest),
        );
        let mut calls = 0;
        let failed = cache.score(
            planned(&b, 64, 2),
            &identities(&b),
            |_| 10,
            |rows| {
                calls += 1;
                if calls == 2 {
                    Err(IndexError::Engine("late batch failure".into()))
                } else {
                    context(rows)
                }
            },
        );
        assert!(failed.is_err());
        assert_eq!(calls, 2, "must fail after a hit and one successful miss");
        assert_eq!(
            (
                (
                    cache.hits,
                    cache.misses,
                    cache.remembered,
                    cache.order.clone(),
                    cache.known.clone()
                ),
                (
                    work.pairs,
                    work.tokens,
                    work.padded_tokens,
                    work.batches,
                    work.encode_us,
                    work.forward_us
                ),
                (lengths.total, lengths.longest)
            ),
            before
        );
        let retry = cached(&mut cache, &mut work, &mut lengths, &b, 64, 2);
        let fresh = cached(
            &mut Scores::default(),
            &mut Work::default(),
            &mut Lengths::default(),
            &b,
            64,
            2,
        );
        assert_eq!(bits(&retry), bits(&fresh));
        assert_eq!(work.pairs, 6, "failed partial misses must execute again");
    }

    #[test]
    fn planner_keeps_ties_caps_and_native_physical_padding() {
        let rows = vec![
            encoding(1, 65),
            encoding(2, 4),
            encoding(3, 4),
            encoding(4, 256),
        ];
        let plans = plan_batches(
            rows,
            512,
            4,
            |len| {
                if len <= 64 {
                    64
                } else if len <= 128 {
                    128
                } else {
                    256
                }
            },
            |_, len| {
                if len <= 64 {
                    (4, 64)
                } else if len <= 128 {
                    (4, 128)
                } else {
                    (2, 256)
                }
            },
        );
        assert_eq!(
            plans
                .iter()
                .map(|b| b.positions.clone())
                .collect::<Vec<_>>(),
            vec![vec![1, 2, 0], vec![3]]
        );
        assert_eq!(
            plans.iter().map(|b| b.shape).collect::<Vec<_>>(),
            vec![(4, 128), (2, 256)]
        );
        let ordinary = planned(&[(1, 9), (2, 9), (3, 600)], 512, 4);
        assert_eq!(
            ordinary
                .iter()
                .map(|b| b.positions.clone())
                .collect::<Vec<_>>(),
            vec![vec![0, 1], vec![2]],
            "overbudget long pair remains a singleton"
        );
        assert!(planned(&[], 512, 4).is_empty());
    }

    #[test]
    fn weighted_fifo_evicts_whole_batches_without_duplicate_queue_entries() {
        let mut cache = Scores::default();
        let key = |n: u32, size: usize| {
            BatchKey::of(
                &vec![pair_key("q", &n.to_string()); size],
                (size, 4),
                (size, 4),
            )
        };
        let first = key(0, 2048);
        let second = key(1, 2048);
        let third = key(2, 3);
        cache.put(first, vec![0.0; 2048]);
        cache.put(second, vec![1.0; 2048]);
        assert_eq!(cache.remembered, 4096);
        for _ in 0..10 {
            cache.put(first, vec![2.0; 2048]);
        }
        assert_eq!(cache.order.len(), 2);
        cache.put(third, vec![3.0; 3]);
        assert!(cache.get(first).is_none());
        assert!(cache.get(second).is_some());
        assert!(cache.get(third).is_some());
        assert_eq!(cache.remembered, 2051);
        assert_eq!(cache.order.len(), 2);
        cache.put(
            key(3, REMEMBERED_SCORES + 1),
            vec![4.0; REMEMBERED_SCORES + 1],
        );
        assert_eq!(cache.remembered, 2051);
        assert_eq!(
            cache.order.len(),
            2,
            "oversized batch must not purge the cache"
        );
    }

    #[test]
    fn malformed_batch_output_does_not_enter_the_cache() {
        let cache = Scores::default();
        let rows = [(1, 3), (2, 3)];
        let failed = cache.score(
            planned(&rows, 64, 4),
            &identities(&rows),
            |_| 3,
            |_| Ok((vec![1.0], 7)),
        );
        assert!(failed.is_err());
        assert_eq!(cache.remembered, 0);
        assert_eq!((cache.hits, cache.misses), (0, 0));
    }

    #[test]
    fn the_shared_planner_preserves_sorted_rows_and_original_positions() {
        let groups = planned(&[(3, 20), (1, 4), (4, 6), (2, 4), (5, 512)], 16, 2);
        assert_eq!(
            groups
                .iter()
                .map(|b| b.positions.clone())
                .collect::<Vec<_>>(),
            vec![vec![1, 3], vec![2], vec![0], vec![4]]
        );
        let mut restored = vec![0; 5];
        for batch in groups {
            for (at, e) in batch.positions.iter().zip(batch.encodings) {
                restored[*at] = e.get_ids()[0];
            }
        }
        assert_eq!(restored, vec![3, 1, 4, 2, 5]);
    }

    /// Whatever a tier parses from, it round-trips through its own name.
    ///
    /// Guards the pair of matches that a new tier has to touch together. The
    /// wire protocol is this string, so a name that parses to a different tier
    /// than it prints would route a caller to a model they did not ask for.
    #[test]
    fn a_tier_parses_from_the_name_it_prints() {
        for tier in [Rerank::Off, Rerank::Fast, Rerank::Accurate] {
            assert_eq!(
                Rerank::parse(tier.name()),
                Some(tier),
                "{} does not round-trip",
                tier.name()
            );
        }
    }

    /// Every pair lands in exactly one batch, in order, and no batch pads
    /// past the budget unless it is one pair that is longer on its own.
    #[test]
    fn a_batch_stays_within_its_budget_and_every_pair_is_in_one() {
        let lengths = [9, 20, 20, 31, 60, 64, 64, 64, 200, 300, 301];
        for budget in [64, 128, 256, 512, 1024] {
            for most in [1, 2, 4, 8, usize::MAX] {
                let sizes = batches(&lengths, budget, most);
                assert_eq!(sizes.iter().sum::<usize>(), lengths.len());
                let mut start = 0;
                for size in sizes {
                    assert!(size > 0, "an empty batch");
                    assert!(size <= most, "{size} pairs against a cap of {most}");
                    let longest = lengths[start + size - 1];
                    assert!(
                        size * longest <= budget || size == 1,
                        "{size} pairs padded to {longest} against a budget of {budget}"
                    );
                    start += size;
                }
            }
        }
    }

    /// Greedy, not merely within budget: a batch closes only when the next
    /// pair would not fit, so short pairs share a pass and a long one does not
    /// drag them up to its length.
    #[test]
    fn short_pairs_share_a_pass_and_a_long_one_does_not_pad_them() {
        assert_eq!(batches(&[30, 30, 30, 30, 250], 256, usize::MAX), vec![4, 1]);
        assert_eq!(
            batches(&[30, 30, 30, 30, 250], 1024, usize::MAX),
            vec![4, 1]
        );
        assert_eq!(batches(&[30, 30, 30, 30, 250], 1024, 2), vec![2, 2, 1]);
        assert_eq!(batches(&[100, 100, 100], 300, usize::MAX), vec![3]);
        assert_eq!(batches(&[], 300, usize::MAX), Vec::<usize>::new());
    }
}
