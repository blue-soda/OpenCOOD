import copy,unittest
import torch
from torch import nn
from opencood.models.sub_modules.resblock import BasicBlock
from opencood.models.sub_modules.int_fine_integer import FineIntegerEncoder
from opencood.models.sub_modules.int_fine_support_audit import output_support,FineSupportAudit


class FineSupportTests(unittest.TestCase):
    def test_support_by_independent_coordinate_enumeration(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for kernel in (1,3):
                for stride in (1,2):
                    for h,w in ((1,1),(5,6),(6,5)):
                        conv=nn.Conv2d(2,3,kernel,stride=stride,padding=kernel//2).to(device)
                        delta=torch.zeros(1,2,h,w,dtype=torch.int64,device=device)
                        delta[0,0,0,0]=1;delta[0,1,-1,-1]=-2
                        mask=output_support(delta,conv);expected=torch.zeros_like(mask)
                        for oy in range(mask.shape[-2]):
                            for ox in range(mask.shape[-1]):
                                for iy,ix in ((0,0),(h-1,w-1)):
                                    if 0<=iy-oy*stride+kernel//2<kernel and 0<=ix-ox*stride+kernel//2<kernel:expected[0,0,oy,ox]=True
                        self.assertTrue(torch.equal(mask,expected));delta.zero_();self.assertFalse(bool(output_support(delta,conv).any()))

    def test_observer_preserves_stream_and_inplace_values(self):
        torch.manual_seed(303)
        source=nn.Sequential(BasicBlock(2,4,2,nn.Sequential(nn.Conv2d(2,4,1,stride=2,bias=False),nn.BatchNorm2d(4))),BasicBlock(4,4),BasicBlock(4,4)).eval()
        calibration={n:dict(quanta=[.125]*c.in_channels) for n,c in source.named_modules() if isinstance(c,nn.Conv2d)}
        model=FineIntegerEncoder(source,calibration,'hybrid').eval();control=copy.deepcopy(model);audit=FineSupportAudit(model);states=[{},{}]
        with torch.no_grad():
            x=torch.randn(1,2,7,8)
            for step in range(4):
                reset=step in (0,3);audit.begin(reset);outputs=[]
                for i,m in enumerate((model,control)):
                    for name,leaf in m.leaves().items():leaf.previous=None if reset else states[i].get(name);leaf.capture_audit=True
                    outputs.append(m(x));states[i]={n:l.next_state for n,l in m.leaves().items() if l.next_state is not None}
                self.assertTrue(torch.equal(*outputs));rows=audit.end();self.assertEqual(len(rows['modules']),20)
                for row in rows['modules'].values():
                    self.assertEqual(row['cold'],reset)
                    if step==1:self.assertEqual(row['temporal_changed_elements'],0)
                    if row['kind']=='FineIntegerConv':self.assertLessEqual(row['nonzero_update_elements'],row['candidate_elements'])
                self.assertEqual(rows['current_shadow_bytes'],sum(r['shadow_bytes'] for r in rows['modules'].values()))
                if step==1:x=x+1
        audit.close();self.assertEqual(audit.handles,[])

if __name__=='__main__':unittest.main()
