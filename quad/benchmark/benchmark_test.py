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
    hash_seed = 23

    # State vectors are already unit length
    preproc_scale = 1 #/ np.max(np.linalg.norm(vectors, axis=0))

    # Create locality sensitive collection of vectors
    h = quad.lsh.StateVectorDistanceHash.from_random(
        d=d,
        r=2.5,
        preproc_scale=preproc_scale,
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
def test_naive_performance_div1100(benchmark):
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


def find_possible_nearby_pairs(store, subslice=slice(None, None), vids=None,
                               query_vid=0,
                               hash_seed=23, r=2.5, output_collection=None):
    d = store[0][subslice].shape[0]  # Dimension

    slice_len = len(range(d)[subslice])

    # State vectors are already unit length
    preproc_scale = 1 / max(np.linalg.norm(v[subslice])
                                           for vid, v in store.items())

    # Create locality sensitive collection of vectors
    h = quad.lsh.StateVectorDistanceHash.from_random(
        d=d,
        r=r,
        preproc_scale=preproc_scale,
    )
    if output_collection is None:
        output_collection = quad.AsymmetricLocalCollection(
            vector_store=store,
            base_lsh=h,
            meta_hash_size=10,
            number_of_maps=10,
            vector_slice=subslice,
            prng=np.random.RandomState(hash_seed),
        )
    print('store')
    for vid in (store if vids is None else vids):
        print(vid, end=', ')
        output_collection.add(vid)

    norm = np.linalg.norm(store[query_vid][subslice])
    close_vids = set(output_collection.iter_local_buckets(query_vid,
                                                          scale=1/norm))

    return close_vids, output_collection

def find_nearby_pairs_flat(store, query_vid):
    d = store[0].shape[0]  # Dimension
    hash_seed = 23

    # State vectors are already unit length
    preproc_scale = 1 #/ np.max(np.linalg.norm(vectors, axis=0))

    # Create locality sensitive collection of vectors
    h = quad.lsh.StateVectorDistanceHash.from_random(
        d=d,
        r=2.5,
        preproc_scale=preproc_scale,
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

def find_nearby_pairs_hierarchical(store, query_vid):
    close_vids = None
    d = store[0].shape[0]  # Dimension
    for n in [4, 8, 12, 16, 23]:
        subslice = slice(None, 2**n)
        close_vids, collection = find_possible_nearby_pairs(
            store, subslice, close_vids, query_vid,
            hash_seed=n, r=2.5)
        print()
        print(f'n={n}, num: {len(close_vids)}')
        if n >= d:
            break

@pytest.mark.benchmark(
    group='hierarchical-perf',
)
def test_index_performance_flat(benchmark):
    '''Benchmark flat (not hierarchical) index.'''
    store = BenchmarakGenerate.get_store()
    benchmark(find_nearby_pairs_flat, store, 19)

@pytest.mark.benchmark(
    group='hierarchical-perf',
)
def test_index_performance_hierarchical(benchmark):
    '''Benchmark hierarchical index.'''
    store = BenchmarakGenerate.get_store()
    benchmark(find_nearby_pairs_hierarchical, store, 19)
