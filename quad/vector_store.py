from __future__ import annotations

from typing import Any, Dict

import collections.abc
import pathlib
import pickle
import os
import shutil

import numpy as np
import zarr
import numcodecs


class VectorStore(collections.abc.MutableMapping):
    '''A class that manages disk-backed numpy arrays.
    '''
    compressor = numcodecs.Blosc(cname='lz4', clevel=3,
                                 shuffle=numcodecs.Blosc.BITSHUFFLE)

    def __init__(self, path: pathlib.Path, load_filter=None):
        self._path = pathlib.Path(path)
        self.load_filter = load_filter
        self._map = {}
        self._info = {}
        self._load(only_new=False, log=True)

    def _load(self, only_new=False, log=True):
        try:
            path_iter = self._path.iterdir()
            i = -1
            for i, path in enumerate(path_iter):
                if i == 0 and log:
                    print('Loading existing vector store')
                    print('    ', end='')
                prefix = path.name[:-len(path.suffix)]
                try:
                    vid = int(prefix)
                except ValueError:
                    continue
                if only_new and vid in self:
                    continue
                if path.suffix == '.zarr':
                    # Load info
                    info = {}
                    try:
                        with open(path.parent / (prefix+'.info'), 'rb') as f:
                            info = pickle.load(f, fix_imports=False)
                    except Exception as e:
                        print(f'Invalid vector info: {path.parent / prefix}'
                              f'.info\n'
                              f'    Exception: {e}')
                    else:
                        self._info[vid] = info
                        if log:
                            print(path.name, end=', ')
                        if self.load_filter is not None:
                            if not self.load_filter(info):
                                self._map[vid] = False
                                continue  # Don't load if filter returns False
                    # Load array
                    try:
                        self._map[vid] = zarr.open(str(path), mode='r+')
                    except Exception as e:
                        print(f'Invalid vector data: {path}\n'
                              f'Exception: {e}')
                    else:
                        if log:
                            print(path.name, end=', ')
                #elif path.suffix == '.info':
                #    with open(path, 'rb') as f:
                #        try:
                #            info = pickle.load(f, fix_imports=False)
                #        except Exception as e:
                #            print(f'Invalid vector info: {path}\n'
                #                  f'    Exception: {e}')
                #        else:
                #            self._info[vid] = info
                #            if log:
                #                print(path.name, end=', ')
            if log:
                print()
                print(f'    All vids: {sorted(self.keys())}')
            if i == -1 and log:
                print('Loaded empty vector store')
        except FileNotFoundError:
            try:
                os.mkdir(self._path)
            except FileExistsError:
                pass
            if log:
                print('Creating new vector store')

    def _save(self, vid: int=None, is_new: bool=False, log: bool=True,
              choose_new_vid: bool=False):
        if choose_new_vid:
            assert vid is not None, 'Invalid arguments'
            while True:
                try:
                    with open(self._path / f'{int(vid)}.info', 'x') as f:
                        pass
                except FileExistsError:
                    vid *= 2
                else:
                    break
            is_new = False
        try:
            os.mkdir(self._path)
        except FileExistsError:
            pass
        if vid is None:
            vids = self._map.keys()
        else:
            vids = (vid,)
        for vid in vids:
            path = self._path / f'{int(vid)}.info'
            path_tmp = self._path / f'{int(vid)}.info.tmp'
            if is_new:
                try:
                    with open(path, 'xb') as f:
                        with open(path_tmp, 'xb') as f_tmp:
                            pickle.dump(self._info[vid], f_tmp,
                                        fix_imports=False)
                            f_tmp.flush()
                            os.rename(path_tmp, path)
                except FileExistsError:
                    raise
            else:
                with open(path_tmp, 'wb') as f_tmp:
                    pickle.dump(self._info.get(vid, {}), f_tmp,
                                fix_imports=False)
                    f_tmp.flush()
                    os.rename(path_tmp, path)
        if log:
            if vid is None:
                print('Saved vector store')
            else:
                print(f'Saved vector vid {vid}')
        if choose_new_vid:
            return vid

    @staticmethod
    def from_list(vectors: Sequence[np.ndarray]) -> VectorStore:
        '''Returns a new `VectorStore` with vids assigned sequentially.'''
        store = VectorStore()
        for i, v in enumerate(vectors):
            store[i] = v
        return store

    def __len__(self) -> int:
        return len(self._map)

    def is_available(self, vid: int) -> bool:
        return self._map.get(vid, False) is not False

    def __getitem__(self, vid: int) -> np.ndarray:
        if self._map.get(vid, None) is False:
            raise RuntimeError(f'Vector with vid {vid} not loaded.')
        return self._map[vid]

    def __setitem__(self, vid: int, vector: np.ndarray) -> None:
        if self._map.get(vid, None) is False:
            raise RuntimeError(f'Vector with vid {vid} not loaded.')
        self._info.setdefault(vid, {})
        self._save(vid, log=False)
        backed = zarr.open(str(self._path / f'{int(vid)}.zarr'),
                           mode='w',
                           compressor=self.compressor,
                           shape=vector.shape,
                           dtype=vector.dtype)
        backed[...] = vector
        self._map[vid] = backed

    def get_info(self, vid: int, default: Any=None) -> Any:
        if vid not in self._map:
            raise KeyError(f'Vector with vid {vid} not found.')
        if self._map[vid] is False:
            raise RuntimeError(f'Vector with vid {vid} not loaded.')
        return self._info.get(vid, default)

    def set_info(self, vid: int, info: Any) -> None:
        if vid not in self._map:
            raise KeyError(f'Vector with vid {vid} not found.')
        if self._map[vid] is False:
            raise RuntimeError(f'Vector with vid {vid} not loaded.')
        self._info[vid] = info
        self._save(vid, log=True)

    def update_info(self, vid: int, *args, **kwargs) -> None:
        if vid not in self._map:
            raise KeyError(f'Vector with vid {vid} not found.')
        if self._map[vid] is False:
            raise RuntimeError(f'Vector with vid {vid} not loaded.')
        self._info[vid].update(*args, **kwargs)
        self._save(vid, log=True)

    def new(self, shape, dtype, info: Any=None) -> int:
        '''Creates a new zero-filled vector under a unique vid and returns that
        vid and vector.
        '''
        vid = len(self._map)
        while vid in self._map:
            vid *= 2
        if info is None:
            info = {}
            self._info.setdefault(vid, info)
        else:
            self._info[vid] = info
        vid = self._save(vid, choose_new_vid=True)
        backed = zarr.open(str(self._path / f'{int(vid)}.zarr'),
                           mode='w',
                           compressor=self.compressor,
                           shape=shape,
                           dtype=dtype)
        self._map[vid] = backed
        return vid, backed

    def add(self, vector: np.ndarray, info: Any=None) -> int:
        '''Stores the given vector under a unique vid and returns that vid.'''
        vid, backed = self.new(vector.shape, vector.dtype, info=info)
        backed[...] = vector
        return vid

    def __delitem__(self, vid: int) -> None:
        if self._map.get(vid, None) is False:
            raise RuntimeError(f'Vector with vid {vid} not loaded.')
        shutil.rmtree(self._path / f'{int(vid)}.zarr', ignore_errors=False)
        del self._map[vid]
        os.remove(self._path / f'{int(vid)}.info')
        del self._info[vid]

    def __iter__(self) -> Iterator[int]:
        return iter(self._map)

    def __contains__(self, vid: int) -> bool:
        return vid in self._map

    def keys(self) -> Iterable[int]:
        return self._map.keys()

    def __repr__(self) -> str:
        return f'VectorStore({self._path}, {self._map!r}, {self._info!r})'
