"""Apply signed unit events directly to cached three-scale fusion convolutions."""
import ctypes as ct

import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.int_fusion_lattice import neighborhoods
from opencood.models.sub_modules.int_integer_event_reference import integer_input
from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionKernel, check


_SOURCE = r'''
extern "C" __global__ void fusion_nonzero(
    const int *row, const int *channel, const float *codes, const float *weight,
    const long long *sites, int height, int width, int incremental,
    float *output, int *visits, int *ones) {
    int job = blockIdx.x, site = sites[job], oc = threadIdx.x;
    int cy = site / width, cx = site % width;
    int count = 0, units = 0;
    float value = incremental ? output[site*128 + oc] : 0.f;
    for (int dy = -1; dy <= 1; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= height) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= width) continue;
            int pixel = y * width + x, tap = (dy+1)*3 + dx+1;
            int end = row[pixel+1];
            for (int i = row[pixel]; i < end; ++i) {
                float code = codes[i], w = weight[(tap*128+channel[i])*128+oc];
#ifdef WEIGHTED_CONTROL
                value = __fmaf_rn(code,w,value);
                units += (code == 1.f || code == -1.f);
#else
                if (code == 1.f) { value = __fadd_rn(value,w); ++units; }
                else if (code == -1.f) { value = __fsub_rn(value,w); ++units; }
                else { value = __fmaf_rn(code,w,value); }
#endif
                ++count;
            }
        }
    }
    output[site*128+oc] = value;
    if (oc == 0) { visits[job] = count; ones[job] = units; }
}
'''


class SignedKernel(NonzeroFusionKernel):
    def __init__(self, weighted=False):
        super().__init__(('#define WEIGHTED_CONTROL\n' if weighted else '') + _SOURCE)

    def update(self, codes, weight, sites, output, incremental):
        if not len(sites):
            return 0, 0
        channels, height, width = codes.shape
        csr = codes.permute(1, 2, 0).reshape(-1, channels).float().to_sparse_csr()
        row, column = csr.crow_indices().int(), csr.col_indices().int()
        values = csr.values()
        visits = torch.empty(len(sites), device=codes.device, dtype=torch.int32)
        units = torch.empty_like(visits)
        args = [ct.c_void_p(t.data_ptr()) for t in (row, column, values, weight, sites)]
        args += [ct.c_int(height), ct.c_int(width), ct.c_int(incremental)]
        args += [ct.c_void_p(t.data_ptr()) for t in (output, visits, units)]
        pointers = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
        stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        check(self.driver.cuLaunchKernel(self.function, len(sites), 1, 1, 128, 1, 1,
                                         0, stream, pointers, None), 'signed fusion')
        return int(visits.sum()), int(units.sum())


class DirectFusion(nn.Module):
    def __init__(self, conv, quanta, verify, weighted_check=False):
        super().__init__()
        if (conv.in_channels, conv.out_channels, conv.kernel_size, conv.padding,
                conv.stride, conv.dilation, conv.groups) != (384,128,(3,3),(1,1),(1,1),(1,1),1):
            raise ValueError('Requires the first linear three-scale fusion layer')
        self.weight = conv.weight
        self.bias = conv.bias
        self.verify = verify
        q = conv.weight.new_tensor(quanta).reshape(1,384,1,1)
        self.register_buffer('quanta', q)
        folded = conv.weight.detach() * q
        self.register_buffer('kernels', torch.stack([
            w.permute(2,3,1,0).contiguous() for w in folded.split(128, dim=1)]))
        self.kernel = SignedKernel()
        self.weighted = SignedKernel(weighted=True) if weighted_check else None

    def begin(self, prior, reason):
        self.previous = None if reason or prior is None else prior['branch_fusion']
        self.new = None

    def forward(self, value):
        if torch.is_grad_enabled() or value.dtype != torch.float32 or value.shape[:2] != (1,384):
            raise ValueError('Frozen single-stream FP32 input required')
        _, _, h, w = value.shape
        n = integer_input(value, self.quanta)
        if int(n.abs().max()) >= 2**22:
            raise ValueError('Fusion codes must be exactly representable')
        levels = n.int().reshape(3,128,h,w)
        static = neighborhoods(levels, 'nonzero')
        if self.previous is None:
            change = None
            use_delta = torch.zeros_like(static, dtype=torch.bool)
        else:
            old_levels, old_outputs = self.previous
            change = levels-old_levels
            delta = neighborhoods(change, 'nonzero')
            use_delta = delta < static

        outputs, branches = [], []
        for i in range(3):
            out = (value.new_empty(h*w,128) if self.previous is None else old_outputs[i].clone())
            if self.weighted:
                control = torch.empty_like(out) if self.previous is None else old_outputs[i].clone()
            static_ids = (~use_delta[i]).flatten().nonzero().flatten()
            st, su = self.kernel.update(levels[i], self.kernels[i], static_ids, out, False)
            if self.weighted:
                assert self.weighted.update(levels[i], self.kernels[i], static_ids, control, False) == (st,su)
            dt = du = 0
            if change is not None:
                active = (use_delta[i] & (delta[i] > 0)).flatten().nonzero().flatten()
                dt, du = self.kernel.update(change[i], self.kernels[i], active, out, True)
                if self.weighted:
                    assert self.weighted.update(change[i], self.kernels[i], active, control, True) == (dt,du)
            if self.weighted:
                assert torch.equal(out,control)
            branches.append(dict(static_chosen=st, delta_chosen=dt, one_static=su, one_delta=du))
            outputs.append(out)
        result = (outputs[0]+outputs[1]+outputs[2]).T.reshape(1,128,h,w)
        if self.bias is not None:
            result = result + self.bias.view(1,-1,1,1)
        self.new = (levels, outputs)
        self.previous = None
        self.ledger = dict(branches=branches, cache_bytes=levels.numel()*4+sum(o.numel()*4 for o in outputs),
                           merge_add=(2+int(self.bias is not None))*outputs[0].numel())
        if self.weighted:
            self.ledger['weighted_equal_branches'] = len(outputs)
        if self.verify:
            reference = F.conv2d(n.float()*self.quanta, self.weight, self.bias, padding=1)
            self.ledger['max_dense_error'] = float((result-reference).abs().max())
            assert torch.allclose(result, reference, atol=5e-4, rtol=3e-5), self.ledger
        return result
