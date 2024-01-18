import os
from pathlib import Path
from packaging import version
from transformers import Trainer, is_torch_tpu_available
from transformers.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init, deepspeed_load_checkpoint
from transformers.utils import is_sagemaker_mp_enabled, WEIGHTS_NAME, logging
from transformers.trainer_callback import TrainerState
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from typing import Optional, Union
from peft import PeftModel
import torch.distributed as dist
import numpy as np
import math
from transformers.trainer_pt_utils import get_model_param_count, get_parameter_names
import time
import torch
from accelerate import skip_first_batches
from accelerate import __version__ as accelerate_version
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)

class CustomizedTrainer(Trainer):
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        extend_layers = None, 
    ):
        self.extend_layers = extend_layers
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, 
        model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if self.extend_layers is not None:
                opt_model.requires_grad_(False)
                for n, p in opt_model.named_parameters():
                    for idx in self.extend_layers:
                        if 'layers.' + str(idx) + '.' in n:
                            p.requires_grad_(True)
            
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
