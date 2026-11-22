"""Deterministic, inference-only FP32 deep fusion on packed nonzero channels."""
import ctypes as ct
from pathlib import Path
import torch

from opencood.models.sub_modules.int_exact_fusion_cache import ExactFusionCache


_KERNEL = r'''
extern "C" __global__ void fusion_nonzero(
    const int *row, const int *channel, const float *value, const float *weight,
    const long long *sites, int height, int width, float *out, int *visited) {
    int j = blockIdx.x, site = (int)sites[j], oc = threadIdx.x;
    int cy = site / width, cx = site % width, count = 0;
#ifdef ACCUMULATE
    float sum = out[site*128+oc];
#else
    float sum = 0.f;
#endif
    for (int dy = -1; dy <= 1; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= height) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= width) continue;
            int pixel = y * width + x;
            int end = row[pixel+1], tap = (dy+1)*3 + dx+1;
            count += end-row[pixel];
            for (int i = row[pixel]; i < end; ++i) {
                int w = (tap * 128 + channel[i]) * 128 + oc;
#ifdef SIGN_ONLY
                sum = value[i] > 0 ? __fadd_rn(sum, weight[w]) : __fsub_rn(sum, weight[w]);
#else
                sum = __fmaf_rn(value[i], weight[w], sum);
#endif
            }
        }
    }
    out[site*128+oc] = sum;
    if (oc == 0) visited[j] = count;
}
'''


def check(code, operation):
    if code:
        raise RuntimeError(f'{operation} failed with CUDA status {code}')


class NonzeroFusionKernel:
    """Compile with PyTorch's bundled NVRTC and launch on the current stream."""
    def __init__(self, source=_KERNEL):
        lib=Path(torch.__file__).resolve().parent/'lib'
        paths=sorted(lib.glob('libnvrtc*.so*'))
        libs=[p for p in paths if 'builtins' not in p.name]
        self.rtc=ct.CDLL(str(libs[0]) if libs else 'libnvrtc.so')
        self.driver=ct.CDLL('libcuda.so.1')
        torch.cuda.current_stream()
        prog=ct.c_void_p()
        check(self.rtc.nvrtcCreateProgram(ct.byref(prog),source.encode(),b'fusion.cu',0,None,None),'create program')
        major,minor=torch.cuda.get_device_capability()
        opt=(ct.c_char_p*1)(f'--gpu-architecture=compute_{major}{minor}'.encode())
        try:
            status=self.rtc.nvrtcCompileProgram(prog,1,opt)
            size=ct.c_size_t()
            if status:
                self.rtc.nvrtcGetProgramLogSize(prog,ct.byref(size))
                log=ct.create_string_buffer(size.value)
                self.rtc.nvrtcGetProgramLog(prog,log)
                raise RuntimeError(log.value.decode())
            check(self.rtc.nvrtcGetPTXSize(prog,ct.byref(size)),'PTX size')
            ptx=ct.create_string_buffer(size.value)
            check(self.rtc.nvrtcGetPTX(prog,ptx),'PTX')
        finally:
            self.rtc.nvrtcDestroyProgram(ct.byref(prog))
        self.module=ct.c_void_p()
        self.function=ct.c_void_p()
        check(self.driver.cuModuleLoadData(ct.byref(self.module),ptx),'load module')
        check(self.driver.cuModuleGetFunction(ct.byref(self.function),self.module,b'fusion_nonzero'),'find kernel')

    def run(self,deep,weights,sites,output):
        if not len(sites):return 0
        _,c,h,w=deep.shape
        pixels=deep[0].permute(1,2,0).contiguous().view(-1,c)
        csr=pixels.to_sparse_csr()
        row,col=csr.crow_indices().int(),csr.col_indices().int()
        values=csr.values()
        visits=torch.empty(len(sites),device=deep.device,dtype=torch.int32)
        args=[ct.c_void_p(t.data_ptr()) for t in (row,col,values,weights,sites)]
        args.extend((ct.c_int(h),ct.c_int(w),ct.c_void_p(output.data_ptr()),ct.c_void_p(visits.data_ptr())))
        pointers=(ct.c_void_p*len(args))(*(ct.addressof(a) for a in args))
        stream=ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        check(self.driver.cuLaunchKernel(self.function,len(sites),1,1,128,1,1,0,stream,pointers,None),'fusion kernel')
        return int(visits.sum())


class NonzeroFusionCache(ExactFusionCache):
    def __init__(self,conv):
        super().__init__(conv)
        self.kernel=NonzeroFusionKernel()
        self.register_buffer('packed_weights',self.weight[:,256:].detach().permute(2,3,1,0).contiguous())

    def compute_deep(self,deep,ids,output):
        taps=self.kernel.run(deep,self.packed_weights,ids,output)
        return 0,taps*128,dict(nonzero_taps=taps)

