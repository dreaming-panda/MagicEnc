import torch
torch.cuda.set_per_process_memory_fraction(0.5, 0)
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama import LlamaForCausalLM
name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
llm :LlamaForCausalLM = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16).cuda()
SEQ_LEN = 126976
input_ids = torch.randint(low=3, high=30000, size=(1, SEQ_LEN), device="cuda")

torch.cuda.synchronize()
t1 = time.perf_counter()
logits, cache = llm.prefilling(input_ids, chunk_size=4096, allow_fp16_qk_reduction=False, swap_memory=True, reserved_layers=0)
torch.cuda.synchronize()
t2 = time.perf_counter()
print("Prefill throughput {} token/s".format((SEQ_LEN/(t2 - t1))))

GEN_LEN = 256
input_ids = torch.randint(low=3, high=30000, size=(1, 1))
past_key_values = cache
torch.cuda.synchronize()
t1 = time.perf_counter()
with torch.inference_mode():
    for _ in range(GEN_LEN):
        output = llm(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
torch.cuda.synchronize()
t2 = time.perf_counter()
print("Decode throughput {} token/s".format((GEN_LEN/(t2 - t1))))



