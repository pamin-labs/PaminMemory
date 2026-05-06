# Internal Documentation Submodule

This repository includes an optional private submodule at `internal-docs`.

The submodule points to `Pamin-Labs/InternalDocs`, which contains private planning, strategy, and research notes for Pamin Labs maintainers. It is not required to build, test, or use the public PaminMemory project.

## Public Users

Clone the repository normally:

```bash
git clone https://github.com/Pamin-Labs/PaminMemory.git
cd PaminMemory
```

Do not initialize submodules. If a tool tries to fetch `internal-docs` and reports an authentication or repository access error, that is expected for users who are not members of the private organization repository.

## Team Members

Team members with access to `Pamin-Labs/InternalDocs` can clone with submodules:

```bash
git clone --recurse-submodules git@github.com:Pamin-Labs/PaminMemory.git
cd PaminMemory
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

The PaminMemory internal notes live under:

```text
internal-docs/pamin-memory/
```

## Updating The Submodule

From inside `internal-docs`, pull or commit changes in the private repository first. Then return to the public repo and commit the updated submodule pointer:

```bash
cd internal-docs
git pull
cd ..
git add internal-docs
git commit -m "chore: update internal docs pointer"
```

