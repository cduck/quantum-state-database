from __future__ import annotations

from typing import Hashable

import dataclasses

import numpy as np

from .raw_types import LocalitySensitiveHash


@dataclasses.dataclass(frozen=True)
class MipsHash(LocalitySensitiveHash):
    '''A locality sensitive hash for maximum inner product between vectors.

    Based on arXiv:1405.5869, "Asymmetric LSH (ALSH) for Sublinear Time Maximum
    Inner Product Search (MIPS)".

    Before preprocessing, all stored vectors must be scaled so the norm of the
    largest is less than 1.  When querying, the norm of the query must be 1.
    According to the paper, near optimal parameters are m=3, U (max preprocess
    norm)=0.83, and r=2.5.

    Attributes:
        r: A positive number affecting the probability of collision.
        m: The number of new vector entries added by the asymmetric
            transformations.
        rand_vector: A random vector with elements generated from i.i.d.
            Gaussian N(0, 1).
        rand_scalar: A float generated uniformly in range (0, r].
        preproc_scale: A fixed scaling factor for all preprocessed vectors.
            Defaults to 1.
    '''
    r: float
    m: int
    rand_vector: np.ndarray
    rand_scalar: float
    preproc_scale: float = 1.0
    is_complex: bool = False

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError('r must be a positive integer.')
        if self.m < 0:
            raise ValueError('m must be a non-negative integer.')

    @staticmethod
    def from_random(d: int, r: float=2.5, m: int=3, preproc_scale: float=1.0,
                    is_complex: bool=False,
                    prng: np.random.RandomState=np.random):
        '''Returns a random L2 distance hash with the given configuration.

        Arguments:
            d: The expected dimension (length) of the preprocess/query vectors.
            r: A positive number affecting the probability of collision.
            m: The number of new vector entries added by the asymmetric
                transformations.
            preproc_scale: A fixed scaling factor for all preprocessed vectors.
                Defaults to 1.
            prng: The random number generator to use.
        '''
        if is_complex:
            a = np.empty(d+m, dtype=np.complex128)
            a.real = prng.normal(0, 1, size=d+m)
            a.imag = prng.normal(0, 1, size=d+m)
            b = -prng.uniform(-r, 0) - prng.uniform(-r, 0) * 1j # Excludes 0
        else:
            a = prng.normal(0, 1, size=d+m)
            b = -prng.uniform(-r, 0)  # Excludes 0
        return MipsHash(r=r, rand_vector=a, rand_scalar=b, m=m,
                        preproc_scale=preproc_scale, is_complex=is_complex)

    def random_copy(self, prng: np.random.RandomState=np.random
                   ) -> MipsHash:
        return self.from_random(d=self.d, r=self.r, m=self.m,
                                preproc_scale=self.preproc_scale,
                                is_complex=self.is_complex,
                                prng=prng)

    @property
    def d(self):
        return len(self.rand_vector) - self.m

    def _complex_floor(self, val) -> Hashable:
        #return int(np.floor(val.real)), int(np.floor(val.imag))
        return int(np.floor(np.abs(val)))  # Ignore complex phase

    def hash_function(self, x: np.ndarray) -> Hashable:
        dot = np.vdot(self.rand_vector, x)
        val = (dot + self.rand_scalar) / self.r
        if self.is_complex:
            return self._complex_floor(val)
        return int(np.floor(val))

    def preproc_transform(self, q: np.ndarray) -> np.ndarray:
        '''Returns the transformed vector.

        For MIPS, returns [q; ||q||_2^2; ||q||_2^4; ...; ||x||_2^{2^m}]
        '''
        raise NotImplementedError
        norm = np.linalg.norm(q, ord=2)
        if norm >= 1 + 1e-8:
            raise ValueError(
                'Preprocess vectors must already be scaled to norm <= U < 1.')
        return np.concatenate([q, extra])

    def query_transform(self, q: np.ndarray, scale: float=1) -> np.ndarray:
        '''Returns the transformed query vector.

        For MIPS, returns [q; 0.5; 0.5; ...; 0.5]
        '''
        raise NotImplementedError
        extra = np.full(self.m, 0.5, dtype=q.dtype)
        return np.concatenate([q, extra])

    # Override methods of LocalitySensitiveHash with optimized versions
    def preproc_hash_raw(self, q: np.ndarray) -> Hashable:
        '''Returns the raw hash object of the vector used during preprocessing.

        This method avoids the extra copy of the vector created by
        `preproc_transform`.
        '''
        norm = np.linalg.norm(q, ord=2) * self.preproc_scale
        if norm >= 1 + 1e-8:
            raise ValueError(
                'Preprocess vectors must already be scaled to norm <= U < 1.')
        extra = norm ** (2 ** np.arange(1, self.m+1))
        dot = np.vdot(self.rand_vector[:len(q)], q) * self.preproc_scale
        dot += np.vdot(self.rand_vector[len(q):], extra)
        val = (dot + self.rand_scalar) / self.r
        if self.is_complex:
            return self._complex_floor(val)
        return int(np.floor(val))

    def query_hash_raw(self, q: np.ndarray, scale: float=1) -> Hashable:
        '''Returns the raw hash object of the vector used at query time.

        This method avoids the extra copy of the vector created by
        `query_transform`.
        '''
        dot = np.vdot(self.rand_vector[:len(q)], q) * self.preproc_scale
        dot += np.sum(self.rand_vector[len(q):]) / 2
        val = (dot + self.rand_scalar) / self.r
        if self.is_complex:
            return self._complex_floor(val)
        return int(np.floor(val))

    def query_hash(self, q: np.ndarray, scale: float=1) -> int:
        '''Returns the hash of the vector used at query time.'''
        return hash(self.query_hash_raw(q, scale=scale))
