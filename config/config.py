from functools import lru_cache

from torch import float16
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import (
    HuggingFacePipelineCompatible,
    register_llm_provider,
)

def _load_model(model_name, device, num_gpus, debug=False):
    """Helper function to load the model."""
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update(
                    {
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(num_gpus)},
                    }
                )
    elif device == "mps":
        kwargs = {"torch_dtype": float16}
        # Avoid bugs in mps backend by not using in-place operations.
        print("mps not supported")
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, **kwargs
    )

    if device == "cuda" and num_gpus == 1:
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer

def get_gemma_2b_llm_from_path(model_path: str = "C:\\Users\\malli\\.cache\\huggingface\hub\\models--google--gemma-2b-it\\snapshots\\de144fb2268dee1066f515465df532c05e699d48"):
    """Loads the Gemma 2B LLM from a local path."""
#     device = "cuda"
    device = "cpu"
    num_gpus = 2  # making sure GPU-GPU are NVlinked, GPUs-GPUS with NVSwitch
    model, tokenizer = _load_model(model_path, device, num_gpus, debug=False)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )

    llm = HuggingFacePipelineCompatible(pipeline=pipe)
    return llm

# On the next line, change the Vicuna LLM instance depending on your needs
HFPipelineGemma = get_llm_instance_wrapper(
    llm_instance=get_gemma_2b_llm_from_path(), llm_type="hf_pipeline_gemma"
)

register_llm_provider("hf_pipeline_gemma", HFPipelineGemma)