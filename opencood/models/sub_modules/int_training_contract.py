"""Compare persisted preflight and live training settings in JSON form."""
import json


def require_matching_preflight(preflight,current):
    if preflight.get('preflight_only') is not True or current.get('preflight_only') is not False:
        raise ValueError('Expected completed preflight and training modes')
    normalized=json.loads(json.dumps(current))
    left={k:v for k,v in preflight.items() if k!='preflight_only'}
    right={k:v for k,v in normalized.items() if k!='preflight_only'}
    missing=object()
    mismatch=sorted(k for k in set(left)|set(right) if left.get(k,missing)!=right.get(k,missing))
    if mismatch:raise ValueError('Preflight contract mismatch: '+', '.join(mismatch))
