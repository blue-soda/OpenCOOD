"""Launch two frozen-weight diagnostic interventions, never retrained results."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from opencood.tools.launch_int_memory import digest


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--gpus',type=int,nargs=2,required=True)
    a=p.parse_args();root=Path(a.root).resolve();train=root/'full_delta_gru'
    if len(set(a.gpus))!=2:raise ValueError('Choose distinct GPUs')
    if read(train/'result.json')['status']!='PLATEAU_REACHED':raise ValueError('Training incomplete')
    runtime=read(train/'effective_runtime.json')['arguments'];contract=read(train/'manifest.json')['contract']
    best=read(train/'best.json')
    if digest(best['checkpoint'])!=best['sha256']:raise ValueError('Checkpoint changed')
    for key,field in [('config','config_sha256'),('manifest','manifest_sha256')]:
        if digest(runtime[key])!=contract[field]:raise ValueError('Training input changed')
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode()
    available={int(row[0]):(int(row[1]),int(row[2])) for row in (line.split(',') for line in snapshot.strip().splitlines())}
    if any(g not in available or available[g][0]<12000 or available[g][1]>5 for g in a.gpus):raise ValueError('GPU busy')
    target=root/'mechanism_probes';target.mkdir(exist_ok=False)
    config=yaml.safe_load(Path(runtime['config']).read_text())
    jobs=[];report=dict(scope='Frozen best weights, aligned full stream; inference interventions, not trained variants',
        parent_contract=contract,checkpoint=best,created_unix=time.time(),gpu_snapshot=snapshot,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),jobs=jobs)
    for gpu,(name,overrides) in zip(a.gpus,[('instant_residual',dict(residual_policy='instant')),
                                         ('dense_readout',dict(readout_policy='dense'))]):
        directory=target/name;directory.mkdir();derived=copy.deepcopy(config)
        derived['model']['args'].setdefault('int_delta_gru',{}).update(overrides)
        cfg=directory/'config.yaml';cfg.write_text(yaml.safe_dump(derived,sort_keys=False))
        output=directory/'eval'
        command=[sys.executable,'-m','opencood.tools.run_int_ego','--eval-only','--mode','delta_gru',
            '--config',str(cfg),'--data',config['data_dir'],'--manifest',runtime['manifest'],
            '--checkpoint',best['checkpoint'],'--seed',str(runtime['seed']),'--workers',str(runtime['workers']),
            '--history-policy','aligned','--output',str(output)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(Path.cwd()),
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMPY_MADVISE_HUGEPAGE='0')
        with (directory/'run.log').open('x') as log:
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(dict(name=name,overrides=overrides,gpu=gpu,pid=process.pid,config_sha256=digest(cfg),
                         output=str(output),command=command))
        (target/'launch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
