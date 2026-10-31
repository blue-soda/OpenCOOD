"""Summarize the stages a wrapper actually reports, without a fixed count."""


def stage_means(records):
    if not records:
        raise ValueError('No evaluation records')
    schema = [set(stage) for stage in records[0]['cache_stages']]
    for record in records:
        stages = record['cache_stages']
        if len(stages) != len(schema) or any(set(s) != keys for s, keys in zip(stages, schema)):
            raise ValueError('Cache stage schema changed within evaluation')
    return [{key: sum(float(r['cache_stages'][i][key]) for r in records) / len(records)
             for key in records[0]['cache_stages'][i]} for i in range(len(schema))]
