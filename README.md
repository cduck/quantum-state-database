# QUAD: Quantum State Database

Graduate research project for CMSC 33550 (Introduction to Databases) at University of Chicago.

## Installation

```bash
$ pip3 uninstall cirq  # Fix possibly conflicting packages
$ pip3 install quad  # Install
```


## Usage

```python
import quad

dimension = 100
store = quad.VectorStore('path/to/vector/database')  # Load or create vector database

# First time only: Add vectors to database
for i in range(10):
    prng = np.random.RandomState(i)
    base_vector = prng.normal(size=dimension)
    for j in range(10):
        # Generate any vectors
        vector = base_vector + np.random.normal(scale=0.05, size=dimension)
        info = {'any-data': ...}
        vid = store.add(vector, info)

# Several hashes available: L2DistanceHash, MipsHash, StateVectorDistanceHash
h = quad.lsh.L2DistanceHash.from_random(
    d=dimension,
    r=2.5,
    preproc_scale=1,
)

# Create locality sensitive collection of vectors
collection = quad.AsymmetricLocalCollection(
    vector_store=store,
    base_lsh=h,
    meta_hash_size=10,
    number_of_maps=10,
    prng=np.random.RandomState(seed=5),  # Ensure consistent across runs
)
for vid in store:
    collection.add(vid)

# Query similar vectors:
prng = np.random.RandomState(4)
query_vector = prng.normal(size=dimension)  # Some query vector
query_vid = store.add(query_vector, {'type': 'query'})
norm = 1#np.linalg.norm(query_vid)
close_vids = set(collection.iter_local_buckets(query_vid,
                                               scale=1/norm))
print('Possibly close vids:', close_vids)
assert close_vids == set(range(40, 50))
```


## Benchmarks

```bash
$ git clone https://github.com/cduck/quantum-state-database  # Clone repo
$ cd quantum-state-database
$ pip install -e .[dev]  # Install dev requirements
$ python quad/benchmark/benchmark_generate.py  # Generate test state vector data
$ pytest quad/benchmark/  # Run all benchmarks
```
