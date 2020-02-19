import numpy as np

import quad.lsh


def _prepare_collection():
    d = 20
    n = 20
    u = 0.83
    hash_seed = 5
    data_seed = 10

    data_prng = np.random.RandomState(data_seed)
    vectors = data_prng.normal(size=(d, n))
    preproc_scale = u / np.max(np.linalg.norm(vectors, axis=0))

    h = quad.lsh.MipsHash.from_random(
        d=d, r=2.5, m=3, preproc_scale=preproc_scale)
    store = quad.VectorStore.from_list(vectors.T)
    local_collection = quad.AsymmetricLocalCollection(
        store, h, meta_hash_size=10, number_of_maps=20,
        prng=np.random.RandomState(hash_seed))

    for vid in store:
        local_collection.add(vid)
    return store, local_collection

def test_locality():
    store, local_collection = _prepare_collection()

    found_list = []
    for vid in store:
        # Manually search
        order = sorted((vid2 for vid2 in store),
                       key=lambda vid2: np.dot(store[vid], store[vid2]),
                       reverse=True)

        # Fast approximate search
        local = set(local_collection.iter_local_buckets(vid, scale=1/np.linalg.norm(store[vid])))

        found = order[0] in local
        found_list.append(found)
    assert np.average(found_list) >= 0.8
