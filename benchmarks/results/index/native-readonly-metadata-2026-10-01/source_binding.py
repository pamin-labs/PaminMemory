"""Inert public production-input checks; never builds or runs the private helper."""
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_COMMIT = '47edbae70bbc72b108e4b6b09b938d32edab918c'
HISTORICAL_LOCAL_COMMIT = 'af917642a1fba637b9a8fc02bffb2c966c50f7ec'
PUBLIC_TAG = 'evidence/native-readonly-production-source-20261001'
MANIFEST_SHA256 = 'c20ae032459241b3dd117da69c73ed982b8db3d848f4eab3741e63b7b9ac8845'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    """Reject duplicate members at every depth before schema/digest checks."""
    def unique(pairs):
        value = {}
        for key, member in pairs:
            require(key not in value, 'duplicate JSON member: ' + key)
            value[key] = member
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def validate(binding):
    require(set(binding) == {'format', 'historical_local_source_commit', 'historical_source_public', 'public_equivalent_commit', 'public_reference', 'scope', 'files', 'tracked_cargo_config_paths', 'private_build_receipt', 'reconstructs_historical_helper_binary', 'public_tag', 'public_tree_url', 'public_tag_url', 'public_commit_url'}, 'source binding schema/scope')
    require(binding['format'] == 'public-equivalent-production-inputs-v1', 'source binding format')
    require(binding['historical_local_source_commit'] == HISTORICAL_LOCAL_COMMIT and binding['historical_source_public'] is False, 'historical source must remain local unpublished identity')
    require(binding['public_equivalent_commit'] == PUBLIC_COMMIT and binding['public_tag'] == PUBLIC_TAG, 'exact public source revision; aliases refused')
    require(binding['reconstructs_historical_helper_binary'] is False, 'private helper reconstruction unavailable')
    require(binding['tracked_cargo_config_paths'] == [], 'tracked configuration scope')
    require(len(binding['files']) == 35, 'public production subset cardinality')
    names = set()
    for row in binding['files']:
        require(set(row) == {'path', 'mode', 'git_blob', 'bytes', 'sha256'}, 'source file schema/scope')
        path = row['path']
        require(isinstance(path, str) and path not in names and not Path(path).is_absolute() and '..' not in Path(path).parts, 'source path')
        require(path.startswith(('crates/pamin-core/src/', 'crates/pamin-index/src/')) or path in {'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'} or (path.startswith('crates/') and path.endswith('/Cargo.toml')), 'production subset excludes private helpers')
        names.add(path)
        require(row['mode'] == '100644' and type(row['bytes']) is int and row['bytes'] > 0, 'source mode/size')
        require(isinstance(row['git_blob'], str) and re.fullmatch('[0-9a-f]{40}', row['git_blob']) is not None, 'exact Git blob; aliases refused')
        require(isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None, 'source SHA256')
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == MANIFEST_SHA256, 'public source manifest receipt binding')
    return True


def check_git(binding, repository=None):
    """Return35 when public objects match, None if Git/objects are unavailable.

    The public commit is addressed by its exact object ID, never by a tag alias.
    Replacement refs and inherited Git overrides cannot redirect these reads.
    Lazy fetching is disabled: missing promisor objects never start a fetch.
    The historical local commit is deliberately never consulted.
    """
    validate(binding)
    git = shutil.which('git')
    if git is None:
        return None
    repository = ROOT.parents[3] if repository is None else Path(repository)
    env = {key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_NO_REPLACE_OBJECTS='1', GIT_NO_LAZY_FETCH='1', GIT_CONFIG_NOSYSTEM='1',
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_COUNT='0')

    def read(*args):
        return subprocess.run([git, '--no-replace-objects', '--literal-pathspecs',
                               '-c', 'core.fsmonitor=false', '-C', str(repository), *args],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    commit = read('cat-file', '-t', PUBLIC_COMMIT)
    if commit.returncode != 0:
        return None
    require(commit.stdout == b'commit\n', 'public source object is not a commit')
    for row in binding['files']:
        entry = read('ls-tree', '-z', PUBLIC_COMMIT, '--', row['path'])
        require(entry.returncode == 0, 'public Git source tree read')
        expected = f"{row['mode']} blob {row['git_blob']}\t{row['path']}\0".encode()
        require(entry.stdout == expected, 'public Git mode/path/blob mismatch')
        blob = read('cat-file', 'blob', row['git_blob'])
        require(blob.returncode == 0 and len(blob.stdout) == row['bytes'] and hashlib.sha256(blob.stdout).hexdigest() == row['sha256'], 'public Git source bytes/SHA mismatch')
    return len(binding['files'])
