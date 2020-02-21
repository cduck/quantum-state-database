from __future__ import annotations

from typing import Hashable

import dataclasses

import numpy as np

from .raw_types import LocalitySensitiveHash


@dataclasses.dataclass(frozen=True)
class L2DistanceHash(LocalitySensitiveHash):
    '''A locality sensitive hash for L2 distance between vectors.

    Based on arXiv:1405.5869, "Asymmetric LSH (ALSH) for Sublinear Time Maximum
    Inner Product Search (MIPS)".

    Attributes:
        r: A positive number affecting the probability of collision.
        rand_vector: A random vector with elements generated from i.i.d.
            Gaussian N(0, 1).
        rand_scalar: A float generated uniformly in range (0, r].
    '''
    r: float
    rand_vector: np.ndarray
    rand_scalar: float
    is_complex: bool = False

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError('r must be a positive integer.')

    @staticmethod
    def from_random(d: int, r: float, is_complex: bool=False,
                    prng: np.random.RandomState=np.random) -> L2DistanceHash:
        '''Returns a random L2 distance hash with the given configuration.

        Arguments:
            d: The expected dimension (length) of the preprocess/query vectors.
            r: A positive number affecting the probability of collision.
            prng: The random number generator to use.
        '''
        if is_complex:
            a = np.empty(d, dtype=np.complex128)
            a.real = prng.normal(0, 1, size=d)
            a.imag = prng.normal(0, 1, size=d)
            b = -prng.uniform(-r, 0) - prng.uniform(-r, 0) * 1j # Excludes 0
        else:
            a = prng.normal(0, 1, size=d)
            b = -prng.uniform(-r, 0)  # Excludes 0
        return L2DistanceHash(r=r, rand_vector=a, rand_scalar=b,
                              is_complex=is_complex)

    def random_copy(self, prng: np.random.RandomState=np.random
                   ) -> L2DistanceHash:
        return self.from_random(
            d=self.d, r=self.r, is_complex=self.is_complex, prng=prng)

    @property
    def d(self) -> int:
        return len(self.rand_vector)

    def hash_function(self, x: np.ndarray) -> Hashable:
        dot = np.vdot(self.rand_vector, x)
        val = (dot + self.rand_scalar) / self.r
        if self.is_complex:
            #return int(np.floor(val.real)), int(np.floor(val.imag))
            return int(np.floor(np.abs(val)))  # Ignore complex phase
        return int(np.floor(val))
