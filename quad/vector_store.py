from __future__ import annotations

import collections.abc

import numpy as np


class VectorStore(collections.abc.MutableMapping):
    '''A class that manages disk-backed numpy arrays.

    TODO: Add disk backing.
    '''
    def __init__(self):
        self._map = {}

    @staticmethod
    def from_list(vectors: Sequence[np.ndarray]) -> VectorStore:
        '''Returns a new `VectorStore` with vids assigned sequentially.'''
        store = VectorStore()
        for i, v in enumerate(vectors):
            store[i] = v
        return store

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, vid: int) -> np.ndarray:
        # TODO: Disk backed
        return self._map[vid]

    def __setitem__(self, vid: int, vector: np.ndarray) -> None:
        # TODO: Disk backed
        self._map[vid] = vector

    def add(self, vector: np.ndarray) -> int:
        '''Stores the given vector under a unique vid and returns that vid.'''
        vid = -len(self._map)
        while vid in self._map:
            vid *= 2
        self[vid] = vector
        return vid

    def __delitem__(self, vid: int) -> None:
        # TODO: Disk backed
        del self._map[vid]

    def __iter__(self) -> Iterator[int]:
        return iter(self._map)

    def __contains__(self, vid: int) -> bool:
        return vid in self._map

    def keys(self) -> Iterable[int]:
        return self._map.keys()

    def __repr__(self) -> str:
        return f'VectorStore({self._map!r})'
