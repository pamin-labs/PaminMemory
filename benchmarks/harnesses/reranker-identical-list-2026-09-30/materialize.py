#!/usr/bin/env python3
"""Export source and inert controllers only; never build, fetch or run native code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
DEPENDENCY_SHA256 = '38dd5499147ab4731a76bf21d2db72a2a61cacccf44f6825a19af915686b4c7e'
BASELINE_MANIFEST_SHA256 = '1ddac91dcd4d4675b136ba0824deb2cf7af4e119d9130f5016fecdc635c299e6'
BASELINE_ARCHIVE_SHA256 = 'c72ae63e1fb498eba3d4a0a910214c4f75e001f11cc4f71abeb3a282ce956b50'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dependency(directory):
    require(sha((ROOT / 'shared-dependency.json').read_bytes()) == DEPENDENCY_SHA256, 'shared dependency receipt differs')
    manifest = json.loads((ROOT / 'shared-dependency.json').read_text())
    for name, pin in manifest['files'].items():
        data = (directory / name).read_bytes()
        require(len(data) == pin['bytes'] and sha(data) == pin['sha256'],
                'offline shared dependency differs: ' + name)
    return manifest


def export_subset(directory, identity, destination):
    archive = directory / identity['archive']
    manifest_path = directory / identity['manifest']
    require(sha(archive.read_bytes()) == identity['archive_sha256'], 'source archive binding')
    require(sha(manifest_path.read_bytes()) == identity['manifest_sha256'], 'source manifest binding')
    files = json.loads(manifest_path.read_text())['files']
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        require(all(member.isfile() or member.isdir() for member in members), 'source links refused')
        regular = [m for m in members if m.isfile()]
        require(len(regular) == len(files) and {m.name for m in regular} == set(files),
                'selected source membership')
        for member in members:
            path = Path(member.name)
            require(not path.is_absolute() and '..' not in path.parts, 'source traversal refused')
        for member in regular:
            data = stream.extractfile(member).read()
            pin = files[member.name]
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            require(len(data) == pin['bytes'] and sha(data) == pin['sha256']
                    and blob == pin['git_blob'], 'original selected Git blob differs')
        destination.mkdir()
        stream.extractall(destination, filter='data')


def inventory(root):
    return {str(p.relative_to(root)): {'sha256': sha(p.read_bytes())}
            for p in sorted(root.rglob('*')) if p.is_file()}


def materialize(shared, work, paths):
    shared, work = shared.resolve(strict=True), work.resolve()
    dependency(shared)
    require(not work.exists(), 'preserve existing output')
    standard_package = ('benchmarks', 'harnesses', 'selected-main-stack-2026-10-01')
    dependency_boundary = shared.parents[2] if shared.parts[-3:] == standard_package else shared
    require(not work.is_relative_to(ROOT.parents[2])
            and not work.is_relative_to(dependency_boundary),
            'scratch output must be outside source repository and dependency boundary')
    config = {name: str(Path(path).resolve(strict=True)) for name, path in paths.items()}
    manifest = json.loads((shared / 'sources.json').read_text())
    reconstruction = json.loads((ROOT / 'source-reconstruction.json').read_text())
    manifest['revisions']['main'] = reconstruction['baseline_commit']
    base_manifest = ROOT / 'baseline-source-manifest.json'
    base_archive = ROOT / 'baseline-source.tar.gz'
    require(sha(base_manifest.read_bytes()) == BASELINE_MANIFEST_SHA256
            and sha(base_archive.read_bytes()) == BASELINE_ARCHIVE_SHA256, 'baseline source receipt differs')
    manifest['baseline_source_subset'] = {
        'archive': base_archive.name, 'manifest': base_manifest.name,
        'archive_sha256': sha(base_archive.read_bytes()),
        'manifest_sha256': sha(base_manifest.read_bytes())}
    probe = (ROOT / 'probe.rs.in').read_bytes()
    require(sha(probe) == reconstruction['native_probe_sha256'] == manifest['probe_sha256'],
            'historical native probe differs')
    work.mkdir(parents=True, mode=0o700)
    receipts = {}
    for arm, directory, identity in [
        ('main', ROOT, manifest['baseline_source_subset']),
        ('stack', shared, manifest['stack_source_subset'])]:
        source = work / arm
        export_subset(directory, identity, source)
        for name, pin in manifest['fixture'].items():
            data = (source / 'crates/pamin-engine/tests/corpus' / name).read_bytes()
            require(sha(data) == pin['sha256'] and len(json.loads(data)) == pin['count'],
                    'public fixture differs')
        scaffold = (source / 'crates/pamin-engine/tests/retrieval.rs').read_bytes()
        helper = scaffold + b'\n' + probe
        require(sha(helper) == reconstruction['helper_sha256'], 'actual historical helper differs')
        (source / 'crates/pamin-engine/tests/scratch_cache_product_limits.rs').write_bytes(helper)
        if arm == 'main':
            (source / 'crates/pamin-engine/tests/scratch_synthetic_seed.rs').write_bytes(
                scaffold + b'\n' + (shared / 'seed.rs.in').read_bytes())
        receipts[arm] = inventory(source)
    for name in ['build', 'seed', 'owned_postgres']:
        shutil.copyfile(shared / (name + '.py.in'), work / (name + '.py'))
    shutil.copyfile(shared / 'common.py.in', work / 'shared_common.py')
    shutil.copyfile(shared / 'run.py.in', work / 'shared_run.py')
    for path in ROOT.glob('*.py.in'):
        shutil.copyfile(path, work / path.name.removesuffix('.in'))
    config['work'] = str(work)
    for name, value in [('config', config), ('sources', manifest), ('source-receipts', receipts),
                        ('shared-dependency', dependency(shared))]:
        (work / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-package', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    for name in ['models', 'native', 'ort', 'postgres', 'cargo', 'rustc', 'rustdoc']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.shared_package, args.work, {name: getattr(args, name) for name in
          ['models', 'native', 'ort', 'postgres', 'cargo', 'rustc', 'rustdoc']}))


if __name__ == '__main__':
    main()
