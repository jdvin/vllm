import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.model_executor.models import ModelRegistry
from vllm.sampling_params import SamplingParams

from .test_model.model import TestModel
from .test_model.config import TestModelConfig


ModelRegistry.register_model("TestModel", TestModel, is_multimodal=True)


def test_multimodal_metadata():
    # This test checks if stepping the LLM successfully runs iterations
    # and each request gets information about the request sequence mm tokens.
    engine_args = EngineArgs(
        model="./test_model",
        trust_remote_code=True,
        inject_mm_metadata=True,
        skip_tokenizer_init=True,
    )

    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(temperature=0.0)
    engine.add_request(
        "0",
        {
            "prompt_token_ids": [],
            "multi_modal_data": {"text_conditioning": ""},
        },
        sampling_params,
    )
    step1_out = engine.step()
    step2_out = engine.step()
    step3_out = engine.step()
    assert step2_out[0].outputs[0].token_ids[0] == 9
    assert step2_out[0].outputs[0].token_ids[0] == step2_out[0].outputs[0].token_ids[1]
