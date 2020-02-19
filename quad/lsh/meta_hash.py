from __future__ import annotations

import dataclasses

import numpy as np

from .raw_types import LocalitySensitiveHash


@dataclasses.dataclass(frozen=True)
class MetaHash(LocalitySensitiveHash):
    '''A locality sensitive hash build by concatenating the output of multiple
    other hashes.
    '''
    hashes: Tuple[LocalitySensitiveHash, ...]

    def __post_init__(self):
        if len(self.hashes) <= 0:
            raise ValueError('requires a non-empty tuple of hashes.')

    def random_copy(self, prng: np.random.RandomState=np.random) -> MetaHash:
        return MetaHash(tuple(h.random_copy(prng)
                              for h in self.hashes))

    @staticmethod
    def from_copies(k: int, lsh: LocalitySensitiveHash,
                    prng: np.random.RandomState=np.random
                   ) -> MetaHash:
        '''Returns a random meta-hash containing `k` random copies of the given
        hash, `lsh`.

        Arguments:
            k: The number of independent copies.
            lsh: The hash to copy (with randomized parameters).
            prng: The random number generator to use.
        '''
        return MetaHash(tuple(lsh.random_copy(prng=prng)
                              for _ in range(k)))

    @property
    def d(self) -> int:
        return self.hashes[0].d

    def hash_function(self, q_transformed: np.ndarray) -> Hashable:
        return tuple(h.hash_function(q_transformed)
                     for h in self.hashes)

    def preproc_transform(self, q: np.ndarray) -> np.ndarray:
        raise TypeError('preprocess transform is not supported by MetaHash')

    def query_transform(self, q: np.ndarray) -> np.ndarray:
        raise TypeError('query transform is not supported by MetaHash')

    def preproc_hash_raw(self, q: np.ndarray, **kwargs) -> Hashable:
        return tuple(h.preproc_hash_raw(q, **kwargs)
                     for h in self.hashes)

    def query_hash_raw(self, q: np.ndarray, **kwargs) -> Hashable:
        return tuple(h.query_hash_raw(q, **kwargs)
                     for h in self.hashes)
