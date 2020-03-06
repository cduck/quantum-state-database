from __future__ import annotations

import pytest

import numpy as np
import cirq

import quad.lsh, quad.benchmark
from quad.benchmark.benchmark_generate import BenchmarakGenerate


def distance_metric(v1, v2):
    return max(0, 1-np.abs(np.vdot(v1, v2)) ** 2) ** 0.5

def find_closest_pairs_exhaustive(store, limit_top):
    ordered = sorted(
        ((distance_metric(v1, v2), vid1, vid2)
         for vid1, v1 in store.items()
         for vid2, v2 in [(9, np.array(store[9]))]################store.items()
         if (True########vid1 < vid2
            and store.is_available(vid1) and store.is_available(vid2)
            and (print(vid1, vid2) or True))
        )
    )
    print('Exhaustive pair count:', len(ordered))
    return ordered[:limit_top]

def find_closest_pairs_approximate_index(store, limit_top):
    d = store[0].shape[0]  # Dimension
    u = 0.83  # Max norm after preprocessing
    hash_seed = 23

    # State vectors are already unit length
    preproc_scale = u #/ np.max(np.linalg.norm(vectors, axis=0))

    # Create locality sensitive collection of vectors
    h = quad.lsh.L2DistanceHash.from_random(
        d=d,
        r=2.5,
        #m=3,
        #preproc_scale=preproc_scale,
        is_complex=True,
    )
    collection = quad.AsymmetricLocalCollection(
        vector_store=store,
        base_lsh=h,
        meta_hash_size=10,
        number_of_maps=10,
        prng=np.random.RandomState(hash_seed),
    )
    for vid in store:
        print(vid)
        collection.add(vid)

    # Find closest pairs using the collection
    dist_pairs = {}
    for vid, v in store.items():
        for vid_near in collection.iter_local_buckets(vid):
            print(vid, vid_near)
            if vid_near == vid:
                continue  # Skip self-comparison
            vid_pair = tuple(sorted((vid, vid_near)))
            if vid_pair in dist_pairs:
                continue  # Skip already computed pairs
            v_near = store[vid_near]
            dist = distance_metric(v, v_near)
            dist_pairs[vid_pair] = dist

    ordered = sorted((dist, vid1, vid2)
                     for (vid1, vid2), dist in dist_pairs.items())
    print('Indexed pair count:', len(ordered))
    return ordered[:limit_top]


LIMIT_TOP = 10

@pytest.mark.benchmark(
    group='perf-comparison',
)
def test_naive_performance(benchmark):
    '''Benchmark performance without using an index.'''
    store = BenchmarakGenerate.get_store()
    closest_exhaustive = benchmark(find_closest_pairs_exhaustive,
                                   store, LIMIT_TOP)
    print('Naive:', closest_exhaustive)

@pytest.mark.benchmark(
    group='perf-comparison',
)
def test_index_performance(benchmark):
    '''Benchmark performance when using an approximate index.'''
    store = BenchmarakGenerate.get_store()
    closest_approx = benchmark(find_closest_pairs_approximate_index,
                               store, LIMIT_TOP)
    print('Approx index:', closest_approx)
