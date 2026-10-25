# Internal Documentation

Private planning, strategy, and research notes for Påmin Labs maintainers live in `Pamin-Labs/InternalDocs`. It is not required to build, test, or use the public Påmin Memory project, and this repository does not embed it.

## Team Members

Clone InternalDocs next to this repository:

```bash
git clone git@github.com:Pamin-Labs/InternalDocs.git ../InternalDocs
```

Tools and agents read it from `$PAMIN_LABS_DEV_INTERNAL_DOCS`, which defaults to `../InternalDocs`. Pull it before reading; the latest `main` is the current truth. The PaminMemory notes live under `pamin-memory/`.

Change InternalDocs through its own pull requests. Nothing in this repository needs to change when it does.

## Citing Evidence

Cite an InternalDocs record from a PR or a public document with a SHA permalink, so the citation keeps pointing at the version it was written against:

```text
https://github.com/Pamin-Labs/InternalDocs/blob/<sha>/pamin-memory/research/<date>/<record>.md
```

Do not copy private content into public files.
