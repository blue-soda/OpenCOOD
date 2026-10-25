"""Explicit host path relocation; never rewrite source artifacts or contracts."""
import json
import os
from functools import lru_cache
from pathlib import PurePosixPath


@lru_cache(maxsize=4)
def read_mapping(filename):
    if not filename:return ()
    with open(filename) as stream:mapping=json.load(stream)
    if not isinstance(mapping,dict):raise ValueError('Path map must be an object')
    for source,target in mapping.items():
        for value in (source,target):
            if not isinstance(value,str) or not value.startswith('/') or '..' in PurePosixPath(value).parts or value=='/':
                raise ValueError('Path map needs absolute non-root paths without traversal')
    return tuple(sorted(((k.rstrip('/'),v.rstrip('/')) for k,v in mapping.items()),key=lambda item:-len(item[0])))


def artifact_path(value):
    value=str(value)
    for source,target in read_mapping(os.environ.get('OPENCOOD_INT_PATH_MAP','')):
        if value==source or value.startswith(source+'/'):return target+value[len(source):]
    return value


def relocate_tree(value):
    if isinstance(value,dict):return {k:relocate_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [relocate_tree(v) for v in value]
    return artifact_path(value) if isinstance(value,str) else value
