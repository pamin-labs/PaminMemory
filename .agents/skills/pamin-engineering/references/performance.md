# Performance

Rules for code on a request's path (a service's hot path, which its `AGENTS.md` names), in any language; read `core.md` first, whose "Parse once", "Hot paths keep their database round trips flat", bounded state and "Measure before and after" rules apply too. Each language's reference (`rust.md`, `go.md`) names how its code meets these.

- **Hot paths allocate once.** No allocation or cloning inside loops: reuse buffers sized from what is known, never build an intermediate collection only to iterate it once, and never format a value that is only compared or written out.
- **No repeated parsing** on a hot path, and reuse the optimised kernels that already exist rather than writing a loop of your own.
- **No blocking I/O on async paths;** blocking work runs on a thread meant for it.
- **Static dispatch on hot paths;** dynamic dispatch only at boundaries crossed once per request or less.
- **Connections:** a database pool's size is set explicitly, and a transaction holds its connection only for SQL, never across a call to another service.
- **Start-up** does not wait on other services: connect, fetch keys, list tools and load what can wait on first use or in the background, unless the program can answer nothing without it.
- **Benchmarks** run on the entry point the product calls, at the deployed instance's size when the claim is about production, before and after the change; both numbers go in the PR.
