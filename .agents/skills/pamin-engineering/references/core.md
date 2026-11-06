# Core engineering rules

Every repository that imports this file holds its code to these rules, whatever the language. The bar is high performance, low resource use, fast builds, and clean, DRY, decoupled code that is easy to maintain and extend. Each repository's `AGENTS.md` names its own layers, how it meets the rules below and the tests that check them; a language's reference (`rust.md`, `go.md`, `web.md`, `workers.md`) adds only that language's mechanisms. This file is the same in every repository that carries it; change it in all of them together.

## Before writing anything

- **Does it need to exist?** Build what a real caller needs now. No speculative options, layers, traits, interfaces or config; add them when a second real use appears.
- **One implementation per concern.** Search the repository, and the shared libraries it depends on, for the same concern before adding logic, and extend its owner through its interface; never add a parallel version with its own rules.
- **Do not reinvent what exists.** Reach, in order, for the repository's own implementation of the concern, the official SDK of the service, the standard library, the platform, a dependency already in the tree, then a maintained, widely used dependency; write your own only when none of them meets the need. A PR's "Reused" section names what it searched for, what it reused, and why anything was written by hand; hand-written code where one of those already covers the need is a defect, and review sends it back.
- **Dependency versions.** Lock files pin exact versions. A published library requires caret minimums so its consumers can unify, and pins exactly only with a stated reason.
- **Before adding a dependency,** check its maintenance, its licence and what it adds to build time and to bundle or binary size, and record that in the PR.

## Shape of the code

- **One rule, one place.** A change to a rule touches one function or module; if you must edit the same logic in two places, the logic is in the wrong place.
- **Business logic lives in one layer per area,** apart from transport, views, wiring and storage; each area's `AGENTS.md` says where.
- **A directory says what its code is.** New code goes into the directory its area's `AGENTS.md` names for its layer or concern, never beside them at the top, and that file names each of its parts by path. A new directory at that level changes the layout itself, so it updates that `AGENTS.md`, and the list of any test that checks the layout, in the same PR; where no test reads the list yet, review checks it.
- **Small functions with one job,** names that say what they hold, comments that say why rather than what, and no dead code.
- **Comments claim only what the code guarantees.** A guarantee that waits on later work is named as not yet given.
- **Errors carry context** and are handled once, where something can be done about them: returned or logged, never both, and never swallowed.
- **Types over strings and integers.** Ids, units, amounts and times are named types or standard types, never a bare string or number; input is parsed into them at the boundary, so inner code only sees valid values.
- **Trust boundaries are explicit.** Identity, tenant, end user and project come from authentication, never from a request body, a tool argument or a model.
- **Model external data with types.** Parse external JSON (requests, MCP results, other services' answers, webhooks, files another program writes) into typed structs and enums that ignore unknown fields, and serialize requests from typed structs. No hand-built JSON strings, no chains of untyped lookups, no searching payloads for substrings. A value stays opaque only when a protocol requires it to be echoed back unchanged (for example a model's thinking blocks), and the code says so.
- **No hard-coded value in more than one place.** "One definition per concern" below states it for code; this extends it to config, scripts, docs and projects. Every port, address, URL, id, version, limit, price, locale list, file name and default is defined once, where it is owned (a constant in the owning module, a variable at the top of a Makefile, an `ARG` in a Dockerfile, a Worker's Wrangler config, `.env.example`), and everything else reads it from there or is checked against it by a test or a generator. A literal that names a value owned elsewhere is a bug, in code, config, scripts, tests and docs alike. A value two projects use (a port, a document version, a field limit) has one source, and a test or a generated file keeps the other in step.
- **Deployments have no defaults.** A setting a deployed binary or service must be given (a database, keys, other services' addresses) has no fallback value in code: a missing or empty one stops the program at start, naming itself. Only local use names local values (a cache directory, a local socket, a local tool such as a Makefile, Vite or a benchmark), each once.
- **Shared state has one owner,** a type or a task, which keeps each lock next to the fields it guards.
- **Built files are never committed.** Build output (`target/`, `dist/`, binaries, bundles), benchmark and evaluation results, model caches, indexes and generated caches are git-ignored; only generated files a gate checks are committed, such as snapshots, generated queries and API documents.
- **Extensibility check before every PR.** Adding one more of something the code already has (an operation, a setting, a route, a module, a model, a locale, a transport) touches its one definition plus its tests, messages and docs. If the change needed more places, fold them into one source in the same PR.
- **Tests meet the same bar as code.** A test builds requests from typed values, decodes an answer into the API's own types (never a store row) and compares JSON by decoding it, never as strings, and names a value owned elsewhere from its owner, like any code.
- **No deferral inside a PR's own code.** A written rule that code the PR adds or touches breaks is fixed in that PR, in its own commit; a follow-up is only work outside the PR's concern.

## Layers, domains and decoupling

- **Layers point one way.** Code is layered by concern: entry, wiring, transport, domain, then store and integrations, then values. Dependencies point only from left to right, and nothing points back up. A test fails when one does: in a Rust workspace, a test that reads `cargo metadata`; in a single crate, a test over its modules' `use` paths; in Go, the import test. A new module or crate goes in its layer's directory, never beside the layers.
- **Transport only translates.** A handler, middleware or CLI command parses the request into types, calls one domain function, and shapes that function's result as the API's own types. A business rule in transport is a bug, and that includes refusals, quotas, limits, metering decisions and checks of any kind. A middleware may read the request, but it calls the function that decides, and that function declares the error.
- **Transport never reaches the store.** It uses no call, type or error from it. Even a domain function that only passes a call through is worth having, so that a rule added later has one place to go.
- **Each layer converts at its edge.** Row types never leave the store. The store returns its own plain read types, the domain converts them to its own types, and the API converts the domain's types to its own wire types, so a schema change fails to compile at a conversion instead of silently changing the API. A domain's public functions and fields take and return the domain's own types; the only store type a domain shows is the store handle it is built with. A type several domains and the store share is a value, not a store type.
- **Errors are declared once, with their kind and code, in the domain that owns them.** Transport answers any declared error by its kind, with no table that lists every domain's errors. A store error that transport must act on is mapped by the domain to its own error, so transport never names the store's errors.
- **A domain owns its data.** A domain calls another domain's service, never that domain's queries or tables, and the caller's domain orchestrates cross-domain work, not the store. Each store function is called from one domain only, and a test fails otherwise.
- **A domain module holds its service, its types, its errors and its rules.** It depends only on the store, on integrations through traits or interfaces it declares, and on other domains' services. It never imports an HTTP framework, MCP transport code or the API layer.
- **Wiring is in one place.** The entry point or app module builds each service once and hands each transport the services it uses. Every background task is domain code run on a schedule, under the same rules, owned by the wiring that awaits it at shutdown.
- **Each handler group sees only its own services.** Each business area registers its own operations in its own file and receives only its own service and the caller's identity; an API is never one handler holding every dependency.
- **Traits and interfaces are declared by their consumer,** with only the methods it calls. Functions return concrete types.
- **One definition per concern.** Each rule, constant, limit, price, page size, setting name, header, path or schema has one definition, in the module that owns it, and everything else imports it. Before adding a helper, search for one. A second copy of a loop over an SDK is a bug: put one helper per repeated SDK flow in its integration module.
- **A rule is computed in one function,** and another domain calls that function rather than restating it. What a client must do first is that function's answer, not a set of flags the client has to order.
- **A counter and its refusal come from one function.** The "left" figure a caller is told asks the same function that refuses, and a test checks that the two agree.
- **A closed set of values is one named type with one list.** Wire enums and schemas are built from that list, never written again in a tag or a string, and a test checks the list against the database's check constraint. A store value is never a wire name.
- **Modules are named for what they hold.** `utils`, `common`, `models` and `helpers` are not allowed. Each module covers one concern with a small public surface and is private by default. A file covers one business area, not one type, in this order: imports; constants and statics; types; constructors; methods grouped by type, public first; plain helpers, in rough call order. A type is declared before the code that uses it.
- **Writes are safe to repeat.** A write that a client or a retry may send twice is idempotent: a repeat returns what the first one stored and changes nothing more. A write that also reaches another service calls it outside the transaction and is safe to try again, with that service's "not found" counting as done. Steps are ordered so that a failure part way leaves no gap.
- **Per-transaction state only.** A deployed PostgreSQL may sit behind a transaction-pooling proxy, so a connection belongs to the code for one transaction only. Use `set local` or `set_config(name, value, true)`, and protocol-level prepared statements. Never use a session `set` or `set_config(..., false)` meant to outlast its transaction, session advisory locks (use `pg_advisory_xact_lock`), `LISTEN`/`NOTIFY`, temporary tables, SQL `PREPARE`, or cursors held across transactions. Test fixtures may.
- **State kept per user, tenant or key beyond a request is bounded,** like a rate limiter's buckets or a cache: a bounded structure with eviction, never a map pruned by hand. Per-user work that must be serialised is serialised in the database (a lease or row lock), not in one instance's memory, so instances stay stateless.
- **Hot paths keep their database round trips flat.** A request's round trips do not grow with the size of a user's data: no `count(*)`, `offset` or `max()` over a growing set per request. Keep counters, use keyset pagination with indexes that match each query, and batch independent reads.
- **Parse once.** Large values such as manifests, schemas, prompts and models are parsed once and cached by version in a bounded cache, never copied per request. Answers are streamed rather than buffered when they can be.

## Errors, data, settings and calls to other services

### Errors and API documents

- **Values are leaves.** A type that two layers share (a risk level, a usage count, a place, an identity) is declared once, in a value module that imports no service. The store reads and writes these types and never imports a domain, a model harness or transport.
- **One code, one error.** Every error a client is told about is declared once, where it is owned, with a kind (invalid, forbidden, conflict, too many, unavailable, ...) and a stable code; a code declared twice fails at start or in a test. A limit says when it lifts.
  - A business error belongs to the domain whose rule raises it; the store keeps only data errors, such as not found or busy.
  - A layer with nothing more particular to say reuses the error of its kind and adds context; it never declares the same code again.
- **What a client decides from an error is the server's to say.** An error carries whether it is retryable from its kind, and a client reads that, never a list of codes of its own.
- **An API document lists the codes its operations send, and no other.** Each operation declares the errors it answers with, and each middleware declares its own where it is installed; the document's enum of codes is built from those declarations, so a client never maps a code it cannot receive.
  - A document builds without warnings.
  - A file committed from a generator (an API document, a catalogue, a list another project reads) is checked to be current by a test or a CI step.
- **APIs are isolated.** A public API, an operator API and an internal API each have their own module, port or listener, and document. A module of one API never imports another's, and the operator API is reachable only from its own host.

### Data and transactions

- **Rules inside a transaction are the domain's.** When a rule must be judged atomically with a write, the store reads the facts inside its transaction and hands them to a function the domain passes in; the store reports counts and flags and never decides.
  - A query takes the domain's numbers as parameters (a window's start, a limit, a reason); it never names a period, an age or a reason itself.
- **Calls to other services stay out of transactions.** Commit, then call, and make the change idempotent so a failed call can be retried. What must still happen elsewhere is recorded in the same transaction as the change, and a worker retries it until it is done, with the other service's "not found" counting as done.
- **A write that charges, and that a client may send twice, takes a client id,** generated by the client, the same on every attempt and unique per caller.
  - A repeat with the same id and the same request returns what was stored and charges nothing.
  - A repeat with the same id and a different request is refused.
- **Schema changes expand, then contract.** A migration adds what the new code needs and removes only what no deployed version reads, so the previous release runs on the new schema and can be rolled back to. A service starts on a schema at or above its own version.
- **Queries and migrations.**
  - One query per concern, reused rather than copied.
  - No SQL is assembled in code except test fixtures. A column that holds one of a type's values decodes straight into that type, never through a conversion written by hand.
  - Migrations are written by hand, numbered and append-only: a merged migration is never edited, renamed or removed, and a change is a new migration. They are checked for locking and destructive changes before they merge.

### Settings and switches

- **Settings have one source each.** A deployment setting (address, key, connection cap) is read in one place at start into a typed config. The repository's `.env.example`, or the service's own list of required names, names it once, and a test keeps the two in step. It has no default in code.
- **Product settings live in the database.** Models, prices, plans, limits, retention and similar settings are named with their defaults in one settings module, stored only where they differ from those defaults, changed through the operator API with an audited reason, and read live, never as a copy taken at start.
- **No development-only behaviour in production code.** Local runs use a stub of each service, never a back door.
- **A kill switch is a product setting, judged in one domain function** that every entry point of its kind already calls, never as a check in a handler. Operators are warned while it is on, and its refusal is a declared error that is not retryable.
- **Operator writes are audited and safe to repeat.** Every operator change carries a reason, and the change and its audit record are written in one transaction.

### Calls to other services and models

- **Every call a request waits on has its own deadline and fallback.** The deadline is named once beside the call, as a constant, or as a setting when operators tune it, and set on the call's context or future, never left to an HTTP client's default timeouts. When it passes, it is logged and the request goes on without the result, or fails with a declared error.
- **One integration module per external service.** It wraps the service's official SDK, or a small typed client where there is none, and holds the service's URLs and ids as its own constants. A domain that needs it declares the trait or interface it calls, and the wiring hands it the implementation; only the integration module imports the SDK.
- **Model and inference calls have one owner.** Every call to a model goes through one layer (a harness, an inference module or a pool), which names the call's task, its budget and its settings in one list, and prices or meters each call's own usage. A new model call gets a row in the project's list of model calls, with its cost and latency measured, and an entry on the roadmap.
- **Cost is metered and watched.** Every metered unit is recorded once, by the layer that owns billing: for a module served by a hosting platform, that is the platform, never the module's own code. Spend no single user causes (background work, batch jobs, re-checks) is metered too, attributed to no user, and a total past an operator-set threshold raises an alert.
- **Stubs exist only for tests and local runs.** A stub of a service is reachable only from tests, the benchmark and the local runner, and a test fails when production code imports it.
- **Logs carry ids, counts and durations only,** never message text, memory content, tokens, secrets, request bodies or other personal data. Each request or call writes one line, joined to its trace.

## Resources, build speed and measurement

- **Resources.** Idle processes hold little memory, caches are bounded and evict, and work that can wait runs in the background with a budget.
- **Build speed.** Keep dependency trees shallow, prefer one well-chosen dependency over several small ones, and turn off default features you do not use. Split code so an edit rebuilds as little as possible, with heavy dependencies (inference, TLS) behind the modules or features that need them.
- **Footprint.** Release builds are as small as the language allows without losing safety checks the product relies on.
- **Measure before and after** any change that claims to be faster, smaller or cheaper, on the entry point the product actually calls, and put both numbers and how they were taken in the PR. A claim without a measurement is removed.

## Working and review

- Everything committed is English: code, comments, logs, commit messages, PR titles and bodies, documents.
- Never write tokens, keys or credentials to files, commits or logs.
- **Nothing lives only in a conversation.** Work found and not done (a review's follow-ups, an audit's findings, a flaky test) goes into the project's roadmap in the same session, and a decision the owner makes goes into a decision record before or with the change.
- **Before a PR opens,** rebase on the latest `main`; if `main` gained something overlapping the change, rework it to build on what landed and say what was reused and removed. Run the gates of every `AGENTS.md` the change touches, locally.
- **Review before every PR.** Go through the whole diff, tests included, against each rule of every `AGENTS.md` the change touches and the references it imports, and fix what misses in new commits; then run the peer-review skill and put its verdict and findings under the PR's verification. Work an agent delegated is reviewed the same way: the agent that opens the PR reads its diff and tests itself and never accepts them on the delegate's report alone.
- **A PR merges only when its final commit is proven:** GitHub reports it clean, every check has passed (a failed or pending check blocks it, however small the change), its tests pass, and, when it changes a service or binary that is deployed or shipped, an end-to-end check has passed on that same final commit: a local acceptance run or staging, never a run from before the last review fix.

## Commits and PRs

The coordination tool's commit skill holds the details, and its hooks and CI check them.

- Coordinate through the coordination tool before editing: declare an intent, work on its branch `<type>/i-<number>-<slug>` (no person or tool names), and put `Intent: I-<number>` in the PR description; CI rejects a PR without a live intent for its branch. Intents of the products this one uses appear in the session briefing; check them before changing code that depends on them.
- Conventional Commits with a required scope (`fix(engine): ...`) for commits and PR titles; the first line and the title are at most 72 characters, scope included, so count before committing, since a pushed commit is not rewritten without the owner.
- One logical change per commit, each building and passing on its own; refactors, behaviour changes and formatting go in separate commits.
- PRs are focused on one concern and reviewable in one sitting; large work is a stack of PRs, each mergeable on its own. A description has a summary, the changes, what it reused, how it was verified (with numbers when performance is involved), and any risk or follow-up.
- No agent attribution: no `Co-Authored-By` trailers, no session URLs or ids, no "Generated with" footers. Some tools append such a footer when they create or edit a PR, so read the description back after every such call and remove it.
- The commit author is the engineer's own git identity, never invented; if it is missing, stop and ask, then save it as `GIT_AUTHOR_NAME` and `GIT_AUTHOR_EMAIL` in the environment settings.
- Never squash-merge; keep every commit.
