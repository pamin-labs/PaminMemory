from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
src=a.source/'crates/pamin-engine/tests/retrieval.rs'
dst=src.with_name('scratch_cache_product_limits.rs')
assert not dst.exists(),'preserve any prior source/harness'
dst.write_text(src.read_text()+'\n'+Path(__file__).with_name('probe.rs.append').read_text())
print(dst)
