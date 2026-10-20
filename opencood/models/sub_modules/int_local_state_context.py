"""Identity-initialized local context extension, inside the selectable update."""
import copy
import torch
from opencood.models.sub_modules.int_local_state import LocalStateDetector


class ContextLocalStateDetector(LocalStateDetector):
    version = 'migrated_local_state_context_v2'

    def __init__(self, teacher, config, memory_policy='recurrent'):
        super().__init__(teacher, config, memory_policy)
        stage = teacher.backbone.resnet.layer1
        if len(stage) != 4:
            raise ValueError('Context migration expects four original stage-one blocks')
        for index in (2, 3):
            block = copy.deepcopy(stage[index])
            if block.downsample is not None:
                raise ValueError('Context extension must preserve the state lattice')
            for conv in (block.conv1, block.conv2):
                if (conv.in_channels != 256 or conv.out_channels != 256 or
                        conv.stride != (1, 1) or conv.kernel_size != (3, 3) or
                        conv.padding != (1, 1) or conv.dilation != (1, 1) or conv.groups != 1):
                    raise ValueError('Unsupported context dependency')
            # The preceding state is nonnegative (BasicBlock ReLU). Zeroing
            # the new residual output therefore preserves the entire function.
            with torch.no_grad():
                block.bn2.weight.zero_()
                block.bn2.bias.zero_()
            for parameter in block.parameters():
                parameter.requires_grad_(True)
            self.update.blocks.add_module(str(index), block)
        self.update.radius = 8
        self.train(False)


def load_refinement_state(student, state):
    """Permit only the two declared new blocks to be absent in a base checkpoint."""
    missing, unexpected = student.load_state_dict(state, strict=False)
    allowed = {key for key in student.state_dict()
               if key.startswith(('update.blocks.2.', 'update.blocks.3.'))}
    # PyTorch supplies missing num_batches_tracked on old-style state loading.
    required = {key for key in allowed if not key.endswith('num_batches_tracked')}
    if unexpected or not required.issubset(set(missing)) or not set(missing).issubset(allowed):
        raise ValueError('Refinement checkpoint differs beyond the declared context extension')
