"""CPU sparse signed-digit accumulation for sampled tiles of a real scored convolution."""
import numpy as np
import torch

from opencood.tools.audit_int_spike_pyramid_ops import signed_powers


def digit_planes():
    table = np.zeros((255, 8), dtype=np.int8)
    for code in range(-127, 128):
        for term in signed_powers(abs(code)):
            table[code + 127, abs(term).bit_length()-1] = np.sign(code)*np.sign(term)
    assert np.array_equal(table @ (2**np.arange(8)), np.arange(-127, 128))
    return table


def check_tiles(conv, code, weights, reference, first, table):
    if conv.groups != 1 or code.shape[0] != 1:
        raise ValueError('This audit expects the scored single-scan, ungrouped residual convolutions.')
    codes = code[0].to(dtype=torch.int8).cpu().numpy()
    base = weights.cpu().numpy().transpose(2, 3, 1, 0)
    kh, kw, _, outputs = base.shape
    bank = [base * np.float32(2**bit) for bit in range(table.shape[1])]
    bias = conv.bias.detach().cpu().numpy() if first and conv.bias is not None else None
    _, _, height, width = reference.shape
    side = min(4, height, width)
    tiles = []
    for top, left in ((0, 0), ((height-side)//2, (width-side)//2)):
        ys, xs = np.indices((side,side)).reshape(2,-1)
        state = np.zeros((ys.size, outputs),dtype=np.float32)
        if bias is not None:
            state[:] = bias
        visits = terms = dense = expected_terms = 0
        for ky in range(kh):
            iy = (ys+top)*conv.stride[0] - conv.padding[0] + ky*conv.dilation[0]
            for kx in range(kw):
                ix = (xs+left)*conv.stride[1] - conv.padding[1] + kx*conv.dilation[1]
                valid, = np.nonzero((iy>=0)&(iy<codes.shape[1])&(ix>=0)&(ix<codes.shape[2]))
                inputs = codes[:,iy[valid],ix[valid]].T
                pos, channels = np.nonzero(inputs)
                levels = inputs[pos,channels].astype(np.int16)
                dense += inputs.size*outputs
                visits += levels.size*outputs
                planes = table[levels+127]
                expected_terms += int(np.count_nonzero(planes))*outputs
                for bit in range(planes.shape[1]):
                    for sign, update in ((1,np.add),(-1,np.subtract)):
                        active, = np.nonzero(planes[:,bit]==sign)
                        if active.size:
                            vector = bank[bit][ky,kx,channels[active]]
                            update.at(state,valid[pos[active]],vector)
                            terms += vector.size
        assert terms==expected_terms and visits<=dense
        target = reference[0,:,top:top+side,left:left+side].cpu().numpy().reshape(outputs,-1).T
        error = float(np.max(np.abs(state-target)))
        peak = float(np.max(np.abs(target)))
        assert error<=2e-4*max(1.,peak),(error,peak,top,left)
        tiles.append(dict(top=top,left=left,width=side,height=side,outputs=outputs,peak=peak,error=error,
                          eligible_synapses=dense,nonzero_synapses=visits,executed_add_sub=terms))
    return tiles
