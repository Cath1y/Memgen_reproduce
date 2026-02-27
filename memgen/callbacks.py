import torch
from transformers import TrainerCallback

class EmptyCacheCallback(TrainerCallback):
    """
    A callback that empties the CUDA cache at the end of each step
    to prevent memory fragmentation and OOM.
    """
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()