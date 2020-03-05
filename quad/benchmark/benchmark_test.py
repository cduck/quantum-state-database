from __future__ import annotations

import pytest

import numpy as np
import cirq

import quad.lsh, quad.benchmark


BENCHMARK_STORE_PATH = 'benchmark-store'


def default_circuit_factory(*, num_qubits, depth, noise_p1, noise_p2
                           ) -> quad.benchmark.RandomCircuitFactory:
    qubits = cirq.GridQubit.rect(*(int(np.ceil(num_qubits**0.5)),)*2
                                )[:num_qubits]
    return quad.benchmark.RandomCircuitFactory(
        qubits=qubits, depth=depth,
        noise_model=quad.benchmark.RotationNoise(noise_p1, noise_p2, None))


def generate_vectors(seed, factory, num_base, num_noise_per
                    ) -> quad.VectorStore:
    prng = np.random.RandomState(seed)
    store = quad.VectorStore(BENCHMARK_STORE_PATH)
    circuit_seeds = []

    def add_circuit(circuit_seed, noise_seed):
        store._load(only_new=False)

        info = dict(
            type='random-layered',
            circuit_seed=circuit_seed,
            noise_seed=noise_seed,
            num_qubits=len(factory.qubits),
            depth=factory.depth,
        )
        if noise_seed is not None:
            info.update(dict(
                noise_type=type(factory.noise_model).__name__,
                noise_p1=factory.noise_model.p1,
                noise_p2=factory.noise_model.p2,
            ))
        # Check that not already calculated
        for vid in store:
            other_info = dict(store.get_info(vid, {}))
            other_info.pop('status', None)
            if info == other_info:
                print(f'Already calculated\n    {info}\n    {other_info}')
                return

        # Add placeholder
        info['status'] = 'placeholder'
        vid = store.add(np.array(0), info=info)

        print(f'Calculating\n{info}')
        circuit = factory.make(circuit_seed, noise_seed)
        state_vector = cirq.final_wavefunction(circuit)
        store[vid] = state_vector
        store.update_info(vid, status='done')
        return

    for base_i in range(num_base):
        sub_prng = np.random.RandomState(prng.randint(2**32))
        circuit_seed = sub_prng.randint(2**32)
        circuit_seeds.append(circuit_seed)
        add_circuit(circuit_seed, None)
        for noise_i in range(num_noise_per):
            noise_seed = sub_prng.randint(2**32)
            add_circuit(circuit_seed, noise_seed)

    #shuffled_store = quad.VectorStore()
    #for i_new, i in enumerate(prng.permutation(np.arange(len(store)))):
    #    shuffled_store[i_new] = store[i]
    #    shuffled_store.set_info(i_new, store.get_info(i))

    return store, circuit_seeds


def distance_metric(v1, v2):
    return max(0, 1-np.abs(np.vdot(v1, v2)) ** 2) ** 0.5

def find_closest_pairs_exhaustive(store, limit_top):
    ordered = sorted(
        ((distance_metric(v1, v2), vid1, vid2)
         for vid1, v1 in store.items()
         for vid2, v2 in store.items()
         if vid1 < vid2
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
        collection.add(vid)

    # Find closest pairs using the collection
    dist_pairs = {}
    for vid, v in store.items():
        for vid_near in collection.iter_local_buckets(vid):
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


class TestBenchmark:
    vectors_seed = 0

    num_base = 100
    num_noise_per = 10
    limit_top = 10

    num_qubits = 20
    depth = 20
    noise_p1 = 0.005
    noise_p2 = 0.005

    def setup_class(cls):
        factory = default_circuit_factory(
            num_qubits=cls.num_qubits, depth=cls.depth,
            noise_p1=cls.noise_p2, noise_p2=cls.noise_p2)
        cls.store, _ = generate_vectors(
            seed=cls.vectors_seed,
            factory=factory,
            num_base=cls.num_base,
            num_noise_per=cls.num_noise_per,
        )

    @pytest.mark.benchmark(
        group='perf-comparison',
    )
    def test_naive_performance(self, benchmark):
        '''Benchmark performance without using an index.'''
        closest_exhaustive = benchmark(find_closest_pairs_exhaustive,
                                       self.store, self.limit_top)
        print('Naive:', closest_exhaustive)

    @pytest.mark.benchmark(
        group='perf-comparison',
    )
    def test_index_performance(self, benchmark):
        '''Benchmark performance when using an approximate index.'''
        closest_approx = benchmark(find_closest_pairs_approximate_index,
                                   self.store, self.limit_top)
        print('Approx index:', closest_approx)
