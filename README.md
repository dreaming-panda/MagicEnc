# MagicEnc

## Introduction

MagicEnc is a lightweight, huggingface-compatible package for long context LLMs. The basic usage is to avoid CUDA OOM when encoding long contexts. 
With layer-wise automatic iterative encoding, MagicEnc can encode long context models within 24GB VRAM. The result of encoding, i.e. the prefilled KV cache is moved to
CPU RAM during encoding for later usage. Notice that MagicEnc generates exact results and do not conduct any approximation. Only batch size = 1 is supported. 

## Support Models and Performance

| Model      | Context | Encode Speed (token/s) |
| ----------- | ----------- |----------- |
| meta-llama/Meta-Llama-3.1-8B      | 128k      |     2470       |
| gradientai/Llama-3-8B-Instruct-Gradient-1048k   | 256k        |     1400   |

## Decoding Optimization

For meta-llama/Meta-Llama-3.1-8B(-Instruct) model, we can swap KV cache and model parameters to leverage GPU to compute GQA. With implementation of Huggingface, we can get 6.5tokens/s for meta-llama/Meta-Llama-3.1-8B(-Instruct) with 124k context. 



