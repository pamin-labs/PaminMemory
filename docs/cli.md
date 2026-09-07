# CLI reference

Every command takes `--json`. The usual caller is an agent parsing output rather
than a person reading it, so the text form is a convenience and the JSON form is
the contract.

The examples below are real output from a workspace built by the writes in
[Getting started](#getting-started), captured rather than composed.

## Global options

| Option | Environment | Default | Meaning |
| --- | --- | --- | --- |
| `--home <path>` | `PAMIN_HOME` | `~/.pamin` | Where the database, index, and downloaded models live |
| `--project <name>` | `PAMIN_PROJECT` | `default` | The memory namespace to operate on |
| `--profile <name>` | `PAMIN_PROFILE` | `balanced` | Embedding profile: `speed`, `balanced`, or `accuracy` |
| `--json` | | off | Emit JSON instead of text |

`PAMIN_LOG` sets the log filter (`PAMIN_LOG=debug`). Logs go to stderr, so they
never contaminate the JSON on stdout.

Changing `--profile` changes the vector space. The index records the profile it
was built with and refuses to open under a different one, naming `reindex` in
the error rather than silently mixing two spaces.

Every command exits non-zero on failure with the reason on stderr:

```console
$ pamin link nope deployment_pipeline --kind depends_on
Error: no topic named nope
```

## Getting started

```console
$ pamin init
Initialized project default in /home/you/.pamin
```

`init` provisions a local PostgreSQL and applies migrations. No Docker, no
configuration. The server is left running between commands so an agent invoking
the CLI repeatedly does not pay startup each time; `pamin stop` shuts it down.

```console
$ pamin write --topic deployment_pipeline "the deployment pipeline runs on continuous integration and publishes artifacts"
Wrote deployment_pipeline v1
$ pamin write --topic rollback_plan "a rollback reverts the deployment pipeline to the previous tag"
Wrote rollback_plan v1
$ pamin write --topic oncall_rota "the oncall rota rotates every monday morning"
Wrote oncall_rota v1
$ pamin write --topic deployment_pipeline "the deployment pipeline now runs on argo cd"
Wrote deployment_pipeline v2
```

## `pamin write`

Records a memory. Content comes from the argument, or from standard input when
it is omitted:

```console
$ git log -1 --format=%B | pamin write --topic release_notes
```

Evidence is stored before anything judges it, so a write that is not promoted to
a topic state is still recorded and still recoverable:

```console
$ pamin write --topic oncall_rota "ok"
Held in evidence only: content was too short to carry a durable claim
Stored as oncall_rota source version 2
```

```console
$ pamin write --topic oncall_rota "ok" --json
{
  "topic": "oncall_rota",
  "version": null,
  "promoted": false,
  "reason": "content was too short to carry a durable claim",
  "source_version": 3
}
```

A `null` version means the filter held it. `source_version` is always set: the
filter decides what reaches the retrieval surface, never what is kept.

Writing also derives relationships. See [Relationships](#relationships).

## `pamin read`

Reads a topic at a version. `--version-offset` counts back from the current one.

```console
$ pamin read deployment_pipeline
deployment_pipeline v2 (current, 0 of 2 versions)

the deployment pipeline now runs on argo cd
```

```console
$ pamin read deployment_pipeline --version-offset 1 --json
{
  "topic": "deployment_pipeline",
  "version": 1,
  "content": "the deployment pipeline runs on continuous integration and publishes artifacts",
  "is_current": false,
  "actual_version_offset": 1,
  "oldest_version": 1,
  "latest_version": 2,
  "available_versions": 2
}
```

An offset past the oldest surviving version clamps rather than failing, and
`actual_version_offset` reports how far the read actually reached. A caller
walking backwards can therefore stop when the two stop agreeing instead of
guessing at the depth first.

## `pamin search`

Retrieves across every recall channel and explains the result.

```console
$ pamin search "how do we deploy" --limit 3
0.0489  deployment_pipeline v2 (current)  the deployment pipeline now runs on argo cd
        lexical_ngram#1 vector#2 graph#1 via depends_on@1hop Importancex1.00 Worthx1.00
0.0479  rollback_plan v1 (current)  a rollback reverts the deployment pipeline to the previous tag
        lexical_ngram#2 vector#3 graph#3 via mentions@1hop Importancex1.00 Worthx1.00
0.0474  oncall_rota v1 (current)  the oncall rota rotates every monday morning
        lexical_ngram#4 vector#4 graph#2 via depends_on@1hop Importancex1.00 Worthx1.00
```

The JSON carries the same trace in full:

```console
$ pamin search "how do we deploy" --limit 1 --json
{
  "query": "how do we deploy",
  "hits": [
    {
      "topic": "deployment_pipeline",
      "topic_state": "6112147e-71bf-4c16-ae34-f812021ac10f",
      "version": 2,
      "is_current": true,
      "content": "the deployment pipeline now runs on argo cd",
      "score": 0.048915915,
      "why": [
        { "kind": "channel", "channel": "lexical_ngram", "rank": 1, "weight": 1.0, "contribution": 0.016393442 },
        { "kind": "channel", "channel": "vector", "rank": 2, "weight": 1.0, "contribution": 0.016129032 },
        { "kind": "channel", "channel": "graph", "rank": 1, "weight": 1.0, "contribution": 0.016393442 },
        { "kind": "path", "via": "oncall_rota", "hops": 1, "edge": "depends_on", "derivation": "explicit" },
        { "kind": "modifier", "modifier": "importance", "factor": 1.0 },
        { "kind": "modifier", "modifier": "worth", "factor": 1.0 }
      ],
      "source_span": "af72f5a0-37b0-42b0-ac08-7d499428fc63"
    }
  ]
}
```

### Reading the `why` trace

Three kinds of entry, and they answer different questions.

**`channel`** — this result appeared in that channel at that rank, and
contributed `weight / (60 + rank)` to the score. There are four channels:

| Channel | What it matches |
| --- | --- |
| `lexical_segmented` | Words, after segmentation. Works in languages written without spaces |
| `lexical_ngram` | Substrings: file paths, error codes, function names, configuration keys |
| `vector` | Meaning, across languages |
| `graph` | Topics connected to what the other channels found |

Ranks travel between channels; scores do not. A BM25 score and a cosine distance
are not comparable quantities, so fusion combines the ranks rather than
pretending the scores share a scale.

Fusion happens here rather than inside the retrieval engine. The engine offers to
fuse its own channels and that offer is declined: the graph lives in PostgreSQL
where the engine cannot see it, so an engine-fused list would have to be fused
again and its members counted twice, and the per-channel ranks would already be
gone.

**`path`** — accompanies a `graph` entry and says how the graph reached this
result: the topic on the other end of the final edge, how many edges were
crossed, which relationship, and whether it was asserted by a caller
(`explicit`) or derived by the engine (`deterministic`). Nobody can verify a
reciprocal rank; anyone can verify that two topics are related the way the path
claims.

**`modifier`** — a post-fusion adjustment, applied at most once each.
`importance` and `worth` lift a result; `superseded` down-weights a historical
state rather than removing it, because a question about how something changed
needs it.

## Relationships

The graph connects topics, and each endpoint resolves to whichever version is
current when a query runs. Edges arrive two ways.

### Derived automatically

Writing a memory that names another topic derives an edge to it. No command is
involved:

```console
$ pamin neighbors rollback_plan --depth 1
deployment_pipeline  1 hop  via rollback_plan --mentions--> (deterministic, 0.50)
```

`rollback_plan` says "reverts the deployment pipeline", which names
`deployment_pipeline`, so the edge exists. Matching compares segmented tokens
rather than substrings, so it works in any language and does not find `db`
inside `debt`. Creating a topic also links it to memories written earlier that
already named it.

Derived edges carry lower confidence than asserted ones, which orders neighbours
at equal distance.

### Asserted explicitly

Derivation only finds relationships the text states. For anything else — a
dependency, a contradiction, a supersession that nobody wrote down — assert it:

```console
$ pamin link oncall_rota deployment_pipeline --kind depends_on
oncall_rota --depends_on--> deployment_pipeline (v1)
$ pamin link oncall_rota deployment_pipeline --kind depends_on
Already linked: oncall_rota --depends_on--> deployment_pipeline (v1)
```

Asserting is idempotent, so re-running it changes nothing.

Kinds: `mentions`, `supports`, `contradicts`, `supersedes`, `related_to`,
`part_of`, `derived_from`, `same_as`, `depends_on`. Both topics must already
exist; linking a name that does not is a typo far more often than it is intent.

`--valid-from` and `--valid-to` bound when the relationship is asserted to hold,
as RFC 3339. Both are open by default, which is how most relationships are
stated. This is separate from when we recorded the claim.

### `pamin unlink`

Retracts a claim. Every row stays and the truth interval is untouched: this says
we no longer assert the relationship, not that it ended at this instant.

```console
$ pamin unlink oncall_rota deployment_pipeline --kind depends_on
Retracted oncall_rota --depends_on--> deployment_pipeline
$ pamin unlink oncall_rota deployment_pipeline --kind depends_on --json
{
  "from": "oncall_rota",
  "to": "deployment_pipeline",
  "kind": "depends_on",
  "closed": false
}
```

`closed: false` means nothing was open to retract, which is different from having
retracted something.

### `pamin neighbors`

Walks the graph with no ranking anywhere in the path.

```console
$ pamin neighbors rollback_plan --depth 1 --json
{
  "topic": "rollback_plan",
  "depth": 1,
  "neighbors": [
    {
      "topic": "deployment_pipeline",
      "hops": 1,
      "via": "rollback_plan",
      "edge": "mentions",
      "derivation": "deterministic",
      "confidence": 0.5
    }
  ]
}
```

Search returns what it judges relevant; this returns what is connected. It is
the question to ask when the ranking itself is what you doubt, and the only way
to see a derived edge that never placed high enough to surface in a search.

Traversal ignores edge direction, since both ends of a `depends_on` are relevant
to recall. `--kind` restricts it, repeatably. `--at <rfc3339>` follows only edges
asserted to hold at that instant, which is how a question about the past avoids
relationships that were only claimed later.

## `pamin grep`

Finds an exact string in the evidence. No pattern matching, no tokenizer, no
ranking model anywhere in the path.

```console
$ pamin write --topic incident_log "the checkout service returned E5521 during the tuesday outage"
Wrote incident_log v1
$ pamin write --topic incident_log "E5521"
Held in evidence only: content was too short to carry a durable claim
Stored as incident_log source version 2
```

```console
$ pamin grep E5521
manual:incident_log v2 (filtered)
        E5521
manual:incident_log v1 (promoted)
        the checkout service returned E5521 during the tuesday outage
```

The second write never became a memory, so `pamin search` cannot see it — which
is the filter working correctly. It is still evidence, and this is the route to
it:

```console
$ pamin grep E5521 --json
{
  "literal": "E5521",
  "matches": [
    {
      "source": "manual:incident_log",
      "source_version": "54cf4923-c45b-4cfe-a788-6588fb7eae6d",
      "version": 2,
      "filter_decision": "filtered",
      "filter_reason": "content was too short to carry a durable claim",
      "excerpt": "E5521"
    },
    {
      "source": "manual:incident_log",
      "source_version": "7f4b4500-9472-436e-b537-2d1cab87f268",
      "version": 1,
      "filter_decision": "promoted",
      "filter_reason": "promoted to the retrieval surface",
      "excerpt": "the checkout service returned E5521 during the tuesday outage"
    }
  ]
}
```

Every match reports whether it reached the retrieval surface and why. A
mandatory filter is only safe if its mistakes can be found, and this is how they
are found.

`-i` folds case; matching is case sensitive otherwise. `--limit` bounds the
result count.

It reaches superseded versions too, so it answers "what did that memory say
before it was rewritten" without walking the version list. Nothing tokenizes, so
it works on any language, on partial identifiers, and on strings a segmenter
would split.

Use `search` when you want relevance, and `grep` when you want certainty.

## `pamin reindex`

Discards the projection index and rebuilds it from PostgreSQL.

```console
$ pamin reindex
Rebuilt the index from postgres: 4 states
```

The index holds nothing PostgreSQL cannot reproduce, which is what makes the
retrieval engine replaceable and makes a breaking engine upgrade a rebuild
rather than a migration. Relationships are unaffected: they live in the
authority store, not the index.

Run it after changing `--profile`, or after deleting the index directory.

## `pamin stop`

```console
$ pamin stop
Stopped the local database server
```

Stops the local PostgreSQL. It is not run automatically, because the common case
is an agent issuing many commands in a row and paying startup once.

## Notes for agents

- Every command accepts `--json`, and stdout carries only that JSON. Logging
  goes to stderr.
- Failures exit non-zero with the reason on stderr.
- `search` gives ranked context; `neighbors` gives structure; `read` gives a
  specific version; `grep` gives the verbatim evidence including what the filter
  held. Reach for the last three when the ranking is what you doubt.
- Results are stable for stable inputs. Ties break on identifier, so an
  assembled context can be reused rather than rebuilt.
- Evidence is never translated and never rewritten. Anything a memory lost in
  summarizing is still in the source it came from, and `pamin grep` reaches it.
