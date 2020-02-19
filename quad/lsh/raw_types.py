from __future__ import annotations

from typing import Hashable, TypeVar

import abc

import numpy as np


TSelf = TypeVar('TSelf', bound='LocalitySensitiveHash')


class LocalitySensitiveHash(metaclass=abc.ABCMeta):
    '''An interface for (asymmetric) locality sensitive hashing algorithms to
    implement.

    Subclasses should override the property `d` and the methods `hash_function`
    and `random_copy`.
    Asymmetric algorithms should also override `preproc_transform` and
    `query_transform`.  Other methods may be overridden to improve performance.
    '''

    @abc.abstractmethod
    def random_copy(self: TSelf, prng: np.random.RandomState=np.random
                   ) -> TSelf:
        '''Returns a new hash object with the same configuration but newly
        selected random parameters.

        Override this method in a subclass.

        Arguments:
            prng: The random number generator to use.
        '''

    @property
    @abc.abstractmethod
    def d(self) -> int:
        '''The vector dimension that this hash applies to.  I.e. the vector
        length.

        Override this property in a subclass.
        '''

    @abc.abstractmethod
    def hash_function(self, q_transformed: np.ndarray) -> Hashable:
        '''Returns an object (usually an int or tuple) that abstractly
        represents the hash of the vector.  The input vector should already be
        transformed by the preprocessing or query transformation.

        Override this method in a subclass.
        '''

    def preproc_transform(self, q: np.ndarray) -> np.ndarray:
        '''Returns the transformed vector.

        Override this method in a subclass for an asymmetric hash.
        '''
        return q

    def query_transform(self, q: np.ndarray) -> np.ndarray:
        '''Returns the transformed query vector.

        Override this method in a subclass for an asymmetric hash.
        '''
        return q

    def preproc_hash_raw(self, q: np.ndarray) -> Hashable:
        '''Returns the raw hash object of the vector used during preprocessing.
        '''
        return self.hash_function(self.preproc_transform(q))

    def preproc_hash(self, q: np.ndarray, **kwargs) -> int:
        '''Returns the hash of the vector used during preprocessing.

        Extra arguments are passed to `preproc_hash_raw`.
        '''
        return hash(self.preproc_hash_raw(q, **kwargs))

    def query_hash_raw(self, q: np.ndarray) -> Hashable:
        '''Returns the raw hash object of the vector used at query time.'''
        return self.hash_function(self.query_transform(q))

    def query_hash(self, q: np.ndarray, **kwargs) -> int:
        '''Returns the hash of the vector used at query time.

        Extra arguments are passed to `query_hash_raw`.
        '''
        return hash(self.query_hash_raw(q, **kwargs))
