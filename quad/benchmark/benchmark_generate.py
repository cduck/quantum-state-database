from __future__ import annotations

import numpy as np
import cirq

import quad.benchmark


NUM_QUBITS = 23
BENCHMARK_STORE_PATH = f'benchmark{NUM_QUBITS}-store'


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
        store._load(only_new=True)

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

    return store, circuit_seeds


class BenchmarakGenerate:
    vectors_seed = 0

    num_base = 100
    num_noise_per = 10

    num_qubits = NUM_QUBITS
    depth = 20
    noise_p1 = 0.005
    noise_p2 = 0.005

    @classmethod
    def generate(cls):
        factory = default_circuit_factory(
            num_qubits=cls.num_qubits, depth=cls.depth,
            noise_p1=cls.noise_p2, noise_p2=cls.noise_p2)
        _, _ = generate_vectors(
            seed=cls.vectors_seed,
            factory=factory,
            num_base=cls.num_base,
            num_noise_per=cls.num_noise_per,
        )

    @classmethod
    def get_store(cls):
        def load_filter(info):
            if info.get('type') != 'random-layered':
                return False
            if info.get('num_qubits') != cls.num_qubits:
                return False
            if info.get('depth') != cls.depth:
                return False
            if info.get('noise_seed') != None:
                if info.get('noise_type') != 'RotationNoise':
                    return False
                if info.get('noise_p1') != cls.noise_p1:
                    return False
                if info.get('noise_p2') != cls.noise_p2:
                    return False
            return True
        return quad.VectorStore(BENCHMARK_STORE_PATH, load_filter=load_filter)


if __name__ == '__main__':
    BenchmarakGenerate.generate()
