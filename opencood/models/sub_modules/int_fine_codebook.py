"""Four-level signed transmission and matched analog receiver for the fine edge."""
import torch
import torch.nn.functional as F

from opencood.models.sub_modules.int_fine_demand import QuotaFineEdge


def pack_symbols(symbols):
    shifts=torch.arange(8,device=symbols.device)*3
    codes=F.pad(symbols.long(),(0,(-symbols.numel())%8)).view(-1,8)
    word=(codes << shifts).sum(1)
    # Keep the last partial group byte-rounded, with no unused whole bytes.
    data=(word[:,None] >> (torch.arange(3,device=word.device)*8)).byte().flatten()
    return data[:(3*symbols.numel()+7)//8].contiguous()


def unpack_symbols(data,count):
    shifts=torch.arange(3,device=data.device)*8
    word=(F.pad(data.long(),(0,(-data.numel())%3)).view(-1,3) << shifts).sum(1)
    return ((word[:,None] >> (torch.arange(8,device=data.device)*3)) & 7).flatten()[:count]


class FineCodebook(QuotaFineEdge):
    def __init__(self,bounds,cap,levels=None,analog=False):
        super().__init__(bounds,'amplitude',cap)
        if levels is not None and (len(levels)!=4 or sorted(set(levels))!=list(levels) or min(levels)<1):
            raise ValueError('Expected four increasing positive integer amplitudes')
        self.book=levels
        self.analog=analog

    def deliver(self,receiver,ids,change):
        if self.book is None:
            sent,_=super().deliver(receiver,ids,change)
            hist=torch.bincount(sent.abs().long()).tolist()
            return sent,dict(magnitudes=hist)
        table=change.new_tensor(self.book)
        # Lower bin wins exact ties.
        rank=torch.bucketize(change.abs()*2,table[1:]+table[:-1])
        symbols=rank+4*(change<0).long()
        payload=pack_symbols(symbols)
        decoded=unpack_symbols(payload,len(ids))
        assert torch.equal(decoded,symbols)
        received=table[decoded%4]*torch.where(decoded>=4,-1,1).int()
        flat=receiver.view(-1)
        if self.analog:
            # Independent floating point accumulation in physical feature units.
            out=flat.float()*self.quantum
            out.index_add_(0,ids,received.float()*self.quantum)
            flat.copy_((out/self.quantum).int())
        else:
            flat.index_add_(0,ids,received)
        return received,dict(bytes=payload.numel(),codes=torch.bincount(decoded,minlength=8).tolist(),
                             abs_error=int((received-change).abs().sum()))

