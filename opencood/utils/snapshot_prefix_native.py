"""Opt-in native IF8 search; wire format and Python final encoder remain unchanged."""
import ctypes
import functools
import os
import zlib
import numpy as np


@functools.lru_cache(None)
def library(path):
    lib = ctypes.CDLL(path)
    lib.snapshot_zlib_version.restype = ctypes.c_char_p
    if lib.snapshot_zlib_version().decode() != zlib.ZLIB_RUNTIME_VERSION:
        raise RuntimeError('Native/Python zlib runtime mismatch')
    lib.snapshot_first_overflow.argtypes = [ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    lib.snapshot_first_overflow.restype = ctypes.c_int64
    return lib


def search(coords, slots, channels, overhead, budget):
    path = os.environ.get('OPENCOOD_SNAPSHOT_NATIVE_LIB')
    if not path:
        return None
    threads = int(os.environ.get('OPENCOOD_SNAPSHOT_THREADS', '8'))
    chunk = int(os.environ.get('OPENCOOD_SNAPSHOT_CHUNK', '256'))
    if not 1 <= threads <= 32 or not threads <= chunk <= 4096:
        raise ValueError('Invalid native prefix concurrency')
    coords = np.ascontiguousarray(coords, dtype=np.uint32)
    slots = np.ascontiguousarray(slots, dtype=np.int64)
    channels = np.ascontiguousarray(channels, dtype=np.int64)
    rejected = ctypes.c_int64(-1)
    accepted = library(path).snapshot_first_overflow(coords.ctypes.data, len(coords),
        slots.ctypes.data, channels.ctypes.data, len(slots), overhead, budget,
        threads, chunk, ctypes.byref(rejected))
    if accepted < 0:
        raise RuntimeError('Native prefix search failed')
    return accepted, None if rejected.value < 0 else rejected.value
