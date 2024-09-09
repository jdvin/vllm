from array import array
from dataclasses import dataclass, field
from typing import List, Optional, Mapping, Iterable, Tuple
from logging import getLogger

import torch
from torch import nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.inputs import INPUT_REGISTRY
from vllm.inputs.data import LLMInputs
from vllm.inputs.registry import InputContext
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalData, MultiModalDataDict, MultiModalInputs, MultiModalPlugin
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, IntermediateTensors, SequenceData

logger = getLogger(__name__)

def dummy_data_for_test_model(
    ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]
) -> tuple[SequenceData, MultiModalDataDict]:
    cfg = ctx.get_hf_config()
    mm_tokens = [-1] * cfg.text_n_positions * mm_counts["text_conditioning"]
    seq = SequenceData(array(VLLM_TOKEN_ID_ARRAY_TYPE, mm_tokens + ([1] * seq_len)))
    mmdd = {
        "text_conditioning": [mm_tokens],
    }
    return seq, mmdd


def input_processor_for_test_model(ctx: InputContext, llm_inputs: LLMInputs) -> LLMInputs:
    """Validate XTTS inputs and tokenize text.

    Must be done here because this step of the
    [vLLM input pipeline](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html)
    must return a `LLMInputs` instance where the `prompt_token_ids` field contains n tokens
    where n >= min kv cache size required for the request."""

    multi_modal_data = llm_inputs.get("multi_modal_data")
    if (
        multi_modal_data is None
        or "text_conditioning" not in multi_modal_data
        or not isinstance(multi_modal_data["text_conditioning"], str)
    ):
        raise TypeError(
            "XTTS output must be conditioned on text tokens."
            " Provide a list of token ids as `text_conditioning`"
            " in the `multi_modal_data` dictionary."
        )
    conf = ctx.get_hf_config()

    cond_text_tokens = list(range(10))
    assert isinstance(multi_modal_data, dict)
    multi_modal_data["text_conditioning"] = [cond_text_tokens]

    return LLMInputs(
        prompt_token_ids=[-1] * len(cond_text_tokens) + [999],
        prompt=None,
        multi_modal_data=multi_modal_data,
    )

class TextConditioningPlugin(MultiModalPlugin):
    """Plugin for text prompt input data.
    Since XTTS generates mel codes, they are it's 'native' modality."""

    def get_data_key(self) -> str:
        return "text_conditioning"

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        """
        Calculate the maximum number of tokens, corresponding to a single
        instance of multimodal data, that are passed to the language model.
        """
        return ctx.model_config.hf_config.text_n_positions

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data: MultiModalData[object],
    ) -> MultiModalInputs:
        """
        Return a dictionary to be passed as keyword arguments to
        :meth:`~torch.nn.Module.forward`. This is similar in concept to
        tokenizers and processors in HuggingFace Transformers.

        If the data is not supported, throw :exc:`TypeError`.
        """
        assert isinstance(data, list)
        return {
            "text_input_ids": torch.tensor(data, dtype=torch.long).squeeze(0),
        }


MULTIMODAL_REGISTRY.register_plugin(TextConditioningPlugin())

@MULTIMODAL_REGISTRY.register_input_mapper("text_conditioning")
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("text_conditioning")
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_test_model)
@INPUT_REGISTRY.register_input_processor(input_processor_for_test_model)
class TestModel(nn.Module):
    def __init__(self, config, cache_config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.cache_config = cache_config
        self.a = nn.Parameter(torch.tensor([1.0]))
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        seq_multimodal_tokens: List[int],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        speaker_conditioning_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.full(
            (input_ids.shape[0], self.config.hidden_size), 
            seq_multimodal_tokens[0],
            device=input_ids.device,
        ) 

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        out = torch.randn(
            (sampling_metadata.selected_token_indices.shape[0], 50256), 
            device=hidden_states.device)
        out[:, hidden_states[0][0].item()] = torch.ones(out[:, hidden_states[0][0].item()].shape, device=hidden_states.device) * 100
        return out

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        pass

