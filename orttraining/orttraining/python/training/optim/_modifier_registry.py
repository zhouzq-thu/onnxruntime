# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._apex_amp_modifier import ApexAMPModifier
from ._ds_modifier import DeepSpeedZeROModifier
from ._megatron_modifier import LegacyMegatronLMModifier


class _AccelerateDeepSpeedZeROModifier(DeepSpeedZeROModifier):
    """
    Modifier for wrapper of DeepSpeed Optimizer in accelerator.
    https://github.com/huggingface/accelerate/blob/7843286f2e1c50735d259fbc0084a7f1c85e00e3/src/accelerate/utils/deepspeed.py#L182C19-L182C19
    """
    def __init__(self, accelerator_optimizer, **kwargs) -> None:
        super().__init__(accelerator_optimizer.optimizer)

OptimizerModifierTypeRegistry = {
    "megatron.fp16.fp16.FP16_Optimizer": LegacyMegatronLMModifier,
    "deepspeed.runtime.zero.stage2.FP16_DeepSpeedZeroOptimizer": DeepSpeedZeROModifier,
    "deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer": DeepSpeedZeROModifier,
    "apex.amp.optimizer.unique_name_as_id": ApexAMPModifier,
    "accelerate.utils.deepspeed.DeepSpeedOptimizerWrapper": _AccelerateDeepSpeedZeROModifier,
}
