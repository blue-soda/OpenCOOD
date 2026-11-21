"""A continuous-valued receiver with the same address budget as the event sender."""
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_fine_demand import QuotaFineEdge


class FloatQuotaReceiver(QuotaFineEdge):
    def __init__(self,bounds,cap):
        super().__init__(bounds,'amplitude',cap)
        self.quantum=1.

    def __call__(self,value):
        if self.phase=='refresh':
            self.levels=value.detach().clone()
            self.ledger=dict(refresh_levels=value.numel(),cache_bytes=value.numel()*value.element_size())
            return value
        return super().__call__(value)

    def aligned_receiver(self,area):
        return warp_bev(self.previous['fine_levels'],self.previous['pose'],self.pose,area)[0].contiguous()

    def target_levels(self,value):
        return value

    def deliver(self,receiver,ids,change):
        receiver.view(-1).index_add_(0,ids,change)
        return change,dict(bytes=change.numel()*change.element_size())

    def commit(self,state,drop):
        state['fine_levels']=self.levels
        self.read=self.levels=None

