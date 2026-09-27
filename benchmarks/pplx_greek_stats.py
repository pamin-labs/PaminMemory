"""Paired bootstrap and sign-flip check for the ignored Greek replay."""

import argparse
import json

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('paired_json')
    rows = json.load(open(parser.parse_args().paired_json))
    assert len(rows) == 108

    for group in ('cross', 'same'):
        for at, metric in enumerate(('nDCG@10', 'recall@50')):
            before = np.array([row[group]['bge'][at] for row in rows])
            after = np.array([row[group]['pplx'][at] for row in rows])
            delta = after - before
            rng = np.random.default_rng(20260927 + at)
            bootstrap = []
            permutation = []
            for _ in range(10):
                picks = rng.integers(0, len(delta), size=(1000, len(delta)))
                bootstrap.extend(delta[picks].mean(axis=1))
                signs = rng.choice([-1, 1], size=(1000, len(delta)))
                permutation.extend((delta * signs).mean(axis=1))
            low, high = np.percentile(bootstrap, [2.5, 97.5])
            p = (np.count_nonzero(np.abs(permutation) >= abs(delta.mean())) + 1) / 10001
            print(
                f'{group} {metric}: BGE={before.mean():.4f} pplx={after.mean():.4f} '
                f'delta={delta.mean():+.4f} CI95=[{low:+.4f},{high:+.4f}] '
                f'sign-flip p={p:.4f} W/L/T='
                f'{np.count_nonzero(delta > 1e-9)}/'
                f'{np.count_nonzero(delta < -1e-9)}/'
                f'{np.count_nonzero(abs(delta) <= 1e-9)}'
            )


if __name__ == '__main__':
    main()
