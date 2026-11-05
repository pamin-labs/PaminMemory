# Web clients

How a React, Vite and TypeScript web client meets `core.md`; read it first. It binds every web app that imports it, a user client or an operator console alike. A web client meets the same bar as a server.

## Stack and dependencies

- **TypeScript in strict mode,** with Bun for installs and scripts, Biome for formatting and lint, and Vitest for tests; Biome warnings fail the check, and after upgrading Biome its `biome migrate` runs so the config uses no deprecated keys.
- **Do not reinvent.** Reach for the platform (`Intl`, CSS, native elements, `fetch`, `URL`, `AbortController`; CSS over JavaScript, native elements over widgets), then the app's stack (its component library, its query library), then a maintained, widely used library pinned to an exact version, before writing your own.
- **Dependencies:** none without an import; the check fails on an unused one (knip).

## Modules

- **Modules by concern.** `src/` holds `api/` (the generated client, any stream, one query hook per resource), `i18n/` (messages, formatters, errors as text), `ui/` (shared components and tokens), one folder per screen under `screens/`, and any other folder the app's `AGENTS.md` names. That `AGENTS.md` lists every folder of `src/` in backticks with a trailing slash, and a test reads it and fails on a folder it does not name or one it names that is gone, so a new folder is described there first.
- **Module direction.** `api/`, `i18n/` and `ui/` never import from `screens/`, and a lint override fails one that does; a screen never imports another screen's internals.
- **Screens reach the server only through `api/`'s hooks:** a lint override fails a screen that imports the client, `QueryClient` or `useQueryClient`, or calls `fetch`.
- **Size.** A component file over about 250 lines, or with more than one data hook, is split by concern and its rules move to plain functions with tests; a test fails on a `.tsx` file over 250 lines.

## Logic and state

- **Logic apart from views.** Components render and handle events; rules, parsing and shaping live in plain functions next to them or in `api/`, tested without rendering. Date and calendar maths are plain functions in `i18n/` with tests.
- **No server rule is copied into the client.** The client reads the server's answer instead of computing it or ordering several answers itself; what the account must do first is the server's answer, mapped to a screen by one `switch`.
- **Server state only through the query library** (TanStack Query), never copied into `useState`; local UI state lives in the component that owns it.
- **Query keys and invalidation.** Keys come only from the generated client's query options (`$api.queryOptions(...).queryKey`) or a helper in `api/` built on them, never string tuples. Every mutation's invalidation lives in its hook in `api/`, so screens never take a `QueryClient`, and work done ahead of the screens (a preload) is a function in `api/` too.
- **An event stream** is parsed with a maintained parser (`eventsource-parser`) that honours `id:` for resuming, and its URLs and headers come from the same base path and auth as the generated client.
- **Storage.** `localStorage` and `sessionStorage` are reached only through one module that tolerates their being unavailable; every key is a constant there, and a value read back is checked with a guard, never cast.
- **One source for environment-derived values:** the API base path, the user's time zone and the locale each come from one helper.
- **No hard-coded values owned elsewhere.** Limits, enums and locale lists come from the generated API types or one module, never literals in a form; the API base path, the server's port and storage keys are each written once; a fact the server owns (a document's version) is read from it or generated, never copied.

## Types

- **Typed end to end** from the server's OpenAPI document: no `any`, no untyped JSON, no hand-written request or response types. Errors are the server's problems, read by their `code` and typed as `Problem["code"]`, never bare strings.
- **Guards, not casts.** No `as` cast except on generated types; a value from storage, the URL or a stream is checked with a type predicate, never cast after a partial check, and each guard has a test with malformed input. A stream event's required fields are listed once, typed from the document.
- **Exhaustive switches.** A `switch` over a kind the document enumerates handles every kind and ends with `default: x satisfies never`, so a kind the server adds fails the type check; Biome's `useExhaustiveSwitchCases` does not see generated unions (checked on 2.6.4), so TypeScript does it.
- **One route over a list.** A family of pages is one route whose segment takes values from one list, and a name the list lacks is a 404.
- **Every promise** is awaited, returned, or `void`ed with a `.catch`.

## Text and locales

- **Text** is only messages of the i18n library, reached through `i18n/`; dates, numbers and language names come from `Intl`.
- **Locales** are the i18n library's list, never written out; a value per locale is a `Record<Locale, …>` checked with `satisfies`, never a `locale === "…"` branch, a per-language name or a third-party locale switch that fails silently for a new locale.

## Interface

- **One component per pattern, composed, never rebuilt.** What appears on more than one screen in the same role is one component in `ui/`, and screens compose it; a screen never builds one again from classes nor restyles one with classes of its own, and a variant a second screen needs is a prop of the component. The design document names each component with its file, and a new pattern adds its component and its line there in the same PR. A shared piece moves to `ui/` the second time it appears.
- **One implementation per concern:** one query hook per resource, one formatter per kind of value, one component per pattern.
- **Tokens.** Colours and other design tokens are CSS variables from the design document, never literals in components; a token's value lives in one CSS file. Tailwind classes, no inline styles except computed values.
- **Accessibility:** native elements first; every control has an accessible name; focus is visible; motion respects `prefers-reduced-motion`, Tailwind's `animate-*` utilities included.

## Security and performance

- **Content Security Policy.** The build writes the page's CSP into `index.html`: its own files, each inline script by its hash, and each third-party origin by name; no `'unsafe-inline'` script. A new third-party origin is added there with a test; headers a meta tag cannot carry are the edge's.
- **Budget.** The shipped JavaScript has a budget that the check enforces; screens are split with `lazy`, and heavy libraries load on first use. Measure the bundle and the first render before and after a change that claims to be faster or smaller.

## Operator consoles

- **An operator console is its own app,** not screens of a user client: its types are generated from the operator API's own document, it shares no code with a user client but the design tokens, and a user client never reaches the operator API.
