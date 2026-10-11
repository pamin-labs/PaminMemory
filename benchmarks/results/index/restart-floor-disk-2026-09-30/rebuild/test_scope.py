#!/usr/bin/env python3
"""Read-only partial-rebuild qualification and semantic negatives."""
import copy,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rebuild_binding',ROOT/'verify.py')
verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)

def full_attestation(review):review['full_source_to_artifact_attestation']=True
def certified_dependencies(review):review['third_party_source_to_artifact_provenance']='certified'
def missing_cached_artifact(review):review['cached_third_party_artifacts_per_arm']['main']-=1

if __name__=='__main__':
    verify.verify()
    original=json.loads((ROOT/'scope-review.json').read_text())
    for mutate in [full_attestation,certified_dependencies,missing_cached_artifact]:
        review=copy.deepcopy(original);mutate(review)
        try:verify.verify(scope_review=review)
        except AssertionError as error:
            expected='cached third-party scope differs' if mutate==missing_cached_artifact else 'falsely certifies cached third-party provenance'
            assert expected in str(error),str(error)
        else:raise AssertionError(mutate.__name__+' accepted')
    print('PASS: partial-rebuild qualification and 3 semantic negatives; no Cargo/models/DB')
