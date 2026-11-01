"""Transient new-view repair; never stores a mixed or rewarped anchor."""
import torch
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_pyramid_cache import PackedPyramidStage
from opencood.models.sub_modules.int_pyramid_regions import coalesced_refresh


class PatchedLatePyramidDetector(AnchoredLatePyramidDetector):
    def __init__(self, base, period=2, dense_reference=False):
        super().__init__(base,period=period,coverage='zero',max_age=.25)
        self.patch_stage=PackedPyramidStage(self.core.deep_encoder,self.core.deep_update).eval()
        self.dense_reference=dense_reference
        self.last_patch_mask=None

    def reuse_deep(self, current, old, valid):
        # Tile1 preserves exactly the uncovered mask; larger execution regions
        # may be chosen without changing which output cells are written.
        requested=~valid
        self.last_patch_mask=requested.detach().clone()
        if not bool(requested.any()):
            return old,dict(active=False,actual_write_fraction=0.,executed_input_area_ratio=0.,stage_conv_ratio=0.,fallback=False)
        packed=self.patch_stage.pack(current,old,valid)
        result,written,stats=coalesced_refresh(self.patch_stage,packed,old,requested,tile=1,
                                              dense_reference=self.dense_reference)
        assert torch.equal(written,requested)
        return result,dict(active=True,**stats)
