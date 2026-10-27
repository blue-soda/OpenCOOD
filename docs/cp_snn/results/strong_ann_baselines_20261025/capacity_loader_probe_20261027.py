import faulthandler,sys,time
faulthandler.dump_traceback_later(45,repeat=False)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from torch.utils.data import DataLoader
from opencood.tools.train_e3dsnn_single import seed_worker
if __name__=='__main__':
 c=load_yaml(sys.argv[1]);d=build_dataset(c,train=False)
 print('dataset ready',flush=True)
 l=DataLoader(d,batch_size=1,num_workers=2,collate_fn=d.collate_batch_test,worker_init_fn=seed_worker,multiprocessing_context='spawn',timeout=70)
 t=time.time();x=next(iter(l));print('batch ready',time.time()-t,flush=True)
