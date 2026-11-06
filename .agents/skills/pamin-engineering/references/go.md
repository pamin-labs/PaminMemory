# Go

How Go code meets `core.md` and `performance.md`; read those first.

## Toolchain and build

- **Version and tools.** The Go version is pinned in `go.mod`, and upgrading Go or an SDK is its own change. Tools (staticcheck, sqlc, goose and the like) are pinned as `go tool`s with `go get -tool`.
- **Clean and modern.** Code is `gofmt`, `go vet`, staticcheck and `modernize` clean, and written in the idioms of the Go version in `go.mod` (`errors.AsType`, `any`, `min`/`max`, range over integers and iterators such as `strings.FieldsSeq`, `new(x)` for a pointer to a value); `go tool modernize -fix` applies them.
- **The standard library first:** `slices`, `maps`, `cmp`, `errors`, `context`, `log/slog`, `net/http` and its pattern routing, `sync`, and `errgroup` from `golang.org/x/sync`.
- **Builds** are static, `CGO_ENABLED=0`, with `-trimpath -ldflags="-s -w"`.

## Files and packages

- **File layout** follows the Uber Go Style Guide's "Function Grouping and Ordering", which is core's file order in Go terms: imports, `const`, `var` (sentinel errors included), types, constructors, methods grouped by receiver (exported first), plain helpers. A type is declared before the functions that use it, never between them.
- **Shared types.** Types several files of a package share go in its `types.go` (an API package's request and response types, a domain's errors and value types); a type with methods stays with its methods.
- **Packages** follow core's module rules: named for what they hold (`account`, never `utils`), one per concern, unexported by default, with interfaces declared by the consumer.

## Errors, context and concurrency

- **Errors** wrap with `%w` and context (`fmt.Errorf("load the profile: %w", err)`), are checked with `errors.Is` and `errors.As`, and are lowercase without trailing punctuation.
- **Context** is the first parameter of anything that does I/O and is never stored in a struct. Every goroutine ends when its context does, and the code that starts it waits for it (`errgroup`, `sync.WaitGroup.Go`).
- **No data races:** shared state is owned by one goroutine or guarded by one mutex, and `go test -race` runs in the gates.
- **Caches and per-user state** are bounded (`golang-lru`) and safe for concurrent use, never a map pruned by hand; a test can find such a map by its key type (a struct field keyed by a user id).

## Types

- **Types over strings and maps:** typed ids and enums, structs with explicit tags for JSON, `time.Time` and `time.Duration` for times, and zero values that are useful.
- **A closed set** is a named type with its constants and one list (`Status` and `Statuses`); an API's enum is built from the list by one helper, never written in a struct tag.
- **A stream's event** is named by a constant beside its payload, and a payload whose field is a closed set is converted to the API's own type in the stream, so its enum reaches the API document.

## Logs and tests

- **Logs** go through `log/slog` with stable keys.
- **Tests** are table-driven where cases repeat, use `t.Parallel()`, `t.Context()` and `t.Cleanup`, and run against the local database and the stubs, never a real paid API; parsers are fuzzed with `go test -fuzz`.

## Performance mechanisms

- **Round trips:** keyset pagination (`id < $1 order by id desc limit n`) with indexes that match each query; independent reads batched (`pgx.Batch`) or folded into one statement.
- **Memory:** `GOGC` and `GOMEMLIMIT` are set for the instance size.
- **Connections:** one pool per process, sized explicitly.
