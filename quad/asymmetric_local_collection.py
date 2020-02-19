from __future__ import annotations

import collections.abc

import numpy as np

from .lsh import LocalitySensitiveHash, MetaHash
from .vector_store import VectorStore


class AsymmetricLocalCollection:
    '''A collection of vectors where it is efficient to query that have a high
    score relative to a query vector.  This score is determined by the given
    locality sensitive hash.
    '''

    def __init__(self,
                 vector_store: VectorStore,
                 base_lsh: LocalitySensitiveHash,
                 meta_hash_size: int=1,
                 number_of_maps: int=1,
                 prng: np.random.RandomState=np.random,
                 prng_state: Any=None):
        self.vector_store = vector_store
        self.base_lsh = base_lsh
        self.meta_hash_size = meta_hash_size
        self.number_of_maps = number_of_maps
        if prng_state is None:
            prng_state = prng.get_state()
        else:
            prng = np.random.RandomState(seed=0)
            prng.set_state(prng_state)
        self.prng_state = prng_state
        lsh = self.base_lsh
        if self.meta_hash_size > 1:
            lsh = MetaHash.from_copies(self.meta_hash_size, lsh, prng=prng)
        self._lsh_map_list: Tuple[Tuple[LocalitySensitiveHash,
                                        MutableMapping[Hashable, Set[int]]
                                       ], ...] = (
            (lsh, collections.defaultdict(set)),
             *((lsh.random_copy(prng), collections.defaultdict(set))
               for _ in range(self.number_of_maps-1)))
        self._len = 0

    def __len__(self):
        return self._len

    def add(self, vid: int, **preproc_args: Any) -> None:
        vector = self.vector_store[vid]  # Load vector
        num_added = 0
        for lsh, m in self._lsh_map_list:
            raw_hash = lsh.preproc_hash_raw(vector, **preproc_args)  # Hash
            bucket = m[raw_hash]
            len1 = len(bucket)
            bucket.add(vid)
            len2 = len(bucket)
            num_added += len2 - len1
        if num_added == 0:
            # Already added
            pass
        elif num_added == len(self._lsh_map_list):
            self._len += 1
        else:
            raise RuntimeError(
                f'Inconsistent state.  '
                f'Vector with vid={vid} must have changed.')

    def discard(self, vid: int) -> Any:
        '''Linear search for an exact vid match.

        This should be much faster than the hash lookup when vectors are large.
        '''
        return self.remove(vid, _raise=False)

    def remove(self, vid: int, *, _raise=True) -> Any:
        '''Linear search for an exact vid match.

        This should be much faster than the hash lookup when vectors are large.
        '''
        for i, (_, m) in enumerate(self._lsh_map_list):
            for bucket in m.values():
                try:
                    bucket.remove(vid)
                    break
                except KeyError:
                    pass
            else:
                if i == 0:
                    if _raise:
                        raise KeyError(vid)
                    else:
                        return
                else:
                    raise RuntimeError(
                        f'Inconsistent map state.  '
                        f'vid {vid!r} was found in some maps but not in map '
                        f'{i}.')
        self._len -= 1

    def items(self) -> Iterable[Tuple[int, Any]]:
        lsh, m = self._lsh_map_list[0]
        for bucket in m.values():
            yield from bucket  # vid

    def __iter__(self) -> Iterable[int]:
        lsh, m = self._lsh_map_list[0]
        for bucket in m.values():
            yield from bucket.keys()  # vid

    def iter_local_buckets(self, vid: int, **query_args: Any) -> Iterable[int]:
        '''Returns an iterator over vids that are likely to be close to the
        query.

        The closeness metric used is determined by the LSH attribute of the
        collection.
        '''
        vector = self.vector_store[vid]  # Load vector
        vid_set = set()
        for lsh, m in self._lsh_map_list:
            raw_hash = lsh.query_hash_raw(vector, **query_args)  # Hash
            bucket = m[raw_hash]
            yield from bucket - vid_set
            vid_set |= bucket

    def __repr__(self):
        return (f'<AsymmetricLocalCollection: '
                f'map0={dict(self._lsh_map_list[0][1])}>')
