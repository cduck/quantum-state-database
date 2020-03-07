from __future__ import annotations

from typing import Hashable

import dataclasses

import numpy as np

from .raw_types import LocalitySensitiveHash


@dataclasses.dataclass(frozen=True)
class StateVectorDistanceHash(LocalitySensitiveHash):
    '''A locality sensitive hash for distance between quantum state vectors.

    Inspired by arXiv:1405.5869, "Asymmetric LSH (ALSH) for Sublinear Time
    Maximum Inner Product Search (MIPS)".

    Attributes:
        r: A positive number affecting the probability of collision.
        rand_vector: A random vector with elements generated from i.i.d.
            Gaussian N(0, 1).
        rand_scalar: A float generated uniformly in range (0, r].
        preproc_scale: A fixed scaling factor for all preprocessed vectors.
            Defaults to 1.
    '''
    r: float
    rand_vector: np.ndarray
    rand_scalar: float
    preproc_scale: float = 1

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError('r must be a positive integer.')

    @staticmethod
    def from_random(d: int, r: float, preproc_scale: float=1,
                    prng: np.random.RandomState=np.random
                   ) -> StateVectorDistanceHash:
        '''Returns a random state vector distance hash with the given
        configuration.

        Arguments:
            d: The expected dimension (length) of the preprocess/query vectors.
            r: A positive number affecting the probability of collision.
            prng: The random number generator to use.
        '''
        a = np.empty(d, dtype=np.complex128)
        a.real = prng.normal(0, 1, size=d)
        a.imag = prng.normal(0, 1, size=d)
        b = -prng.uniform(-r, 0)  # Excludes 0
        return StateVectorDistanceHash(r=r, rand_vector=a, rand_scalar=b,
                                       preproc_scale=preproc_scale)

    def random_copy(self, prng: np.random.RandomState=np.random
                   ) -> StateVectorDistanceHash:
        return self.from_random(
            d=self.d, r=self.r, preproc_scale=self.preproc_scale, prng=prng)

    @property
    def d(self) -> int:
        return len(self.rand_vector)

    def hash_function(self, x: np.ndarray, scale: float=1) -> Hashable:
        dot = np.vdot(self.rand_vector, x) * scale
        val = (np.abs(dot) + self.rand_scalar) / self.r  # Ignore global phase
        return int(np.floor(np.abs(val)))

    def preproc_hash_raw(self, q: np.ndarray) -> Hashable:
        '''Returns the raw hash object of the vector used during preprocessing.
        '''
        return self.hash_function(self.preproc_transform(q),
                                  scale=self.preproc_scale)

    def query_hash_raw(self, q: np.ndarray, scale: float=1) -> Hashable:
        '''Returns the raw hash object of the vector used at query time.'''
        return self.hash_function(self.query_transform(q), scale=scale)
