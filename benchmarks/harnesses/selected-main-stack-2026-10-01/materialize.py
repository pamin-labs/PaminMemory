#!/usr/bin/env python3
"""Export immutable public sources and inert templates to a new scratch tree.

No build, model, PostgreSQL, download or experiment is executed here.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export(repo, revision, destination):
    archive = subprocess.check_output(
        ['git', '--no-replace-objects', '-C', str(repo), 'archive', revision])
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(destination, filter='data')
    # git archive omits submodule contents; never fetch private InternalDocs.


def export_stack(destination, identity):
    archive = ROOT / identity['archive']
    manifest_path = ROOT / identity['manifest']
    require(sha(archive.read_bytes()) == identity['archive_sha256'], 'frozen source archive changed')
    require(sha(manifest_path.read_bytes()) == identity['manifest_sha256'], 'source subset manifest changed')
    manifest = json.loads(manifest_path.read_text())
    files = manifest['files']
    destination.mkdir()
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        regular = [member for member in members if member.isfile()]
        require(len(regular) == len(files) and {member.name for member in regular} == set(files),
                'frozen source subset membership differs')
        require(all(member.isfile() or member.isdir() for member in members), 'source archive links forbidden')
        for member in members:
            relative = Path(member.name)
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe source archive member')
        for member in regular:
            pin = files[member.name]
            data = stream.extractfile(member).read()
            require(len(data) == pin['bytes'] and sha(data) == pin['sha256'], 'frozen source bytes differ')
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            require(blob == pin['git_blob'], 'original compilation-input Git blob differs')
        stream.extractall(destination, filter='data')


def inventory(root):
    return {str(path.relative_to(root)): ({'link': str(path.readlink())} if path.is_symlink()
            else {'sha256': sha(path.read_bytes())}) for path in sorted(root.rglob('*'))
            if path.is_file() or path.is_symlink()}


def materialize(repo, work, paths):
    repo, work = repo.resolve(strict=True), work.resolve()
    require(not work.exists(), 'preserve existing output; choose a new scratch root')
    require(not work.is_relative_to(repo), 'scratch output must be outside the source repository')
    manifest = json.loads((ROOT / 'sources.json').read_text())
    probe = (ROOT / 'probe.rs.in').read_bytes()
    require(sha(probe) == manifest['probe_sha256'], 'measured probe template changed')
    work.mkdir(mode=0o700, parents=True)
    receipts = {}
    for arm, revision in manifest['revisions'].items():
        source = work / arm
        if arm == 'stack':
            export_stack(source, manifest['stack_source_subset'])
        else:
            tree = subprocess.check_output([
                'git', '--no-replace-objects', '-C', str(repo), 'rev-parse',
                '--verify', revision + '^{tree}'], text=True).strip()
            require(tree == manifest['trees'][arm], 'missing or mismatched immutable main source tree')
            export(repo, revision, source)
        for name, identity in manifest['fixture'].items():
            data = (source / 'crates/pamin-engine/tests/corpus' / name).read_bytes()
            require(sha(data) == identity['sha256'] and len(json.loads(data)) == identity['count'],
                    'immutable public fixture differs')
        scaffold = (source / 'crates/pamin-engine/tests/retrieval.rs').read_bytes()
        helper = scaffold + b'\n' + probe
        require(sha(helper) == manifest['helper_sha256'], 'actual measured helper/scaffold differs')
        (source / 'crates/pamin-engine/tests/scratch_cache_product_limits.rs').write_bytes(helper)
        if arm == 'main':
            (source / 'crates/pamin-engine/tests/scratch_synthetic_seed.rs').write_bytes(
                scaffold + b'\n' + (ROOT / 'seed.rs.in').read_bytes())
        receipts[arm] = inventory(source)
    for path in ROOT.glob('*.py.in'):
        shutil.copyfile(path, work / path.name.removesuffix('.in'))
    shutil.copyfile(ROOT / 'sources.json', work / 'sources.json')
    config = {name: str(Path(path).resolve(strict=True)) for name, path in paths.items()}
    config['work'] = str(work)
    (work / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (work / 'source-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    for name in ['models', 'native', 'ort', 'postgres', 'cargo', 'rustc', 'rustdoc']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.repo, args.work, {name: getattr(args, name) for name in
          ['models', 'native', 'ort', 'postgres', 'cargo', 'rustc', 'rustdoc']}))


if __name__ == '__main__':
    main()
