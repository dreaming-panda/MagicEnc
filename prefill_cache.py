import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import Any, Dict, List, Optional, Tuple, Union
class PrefillCache(DynamicCache):
    def __init__(self, 
        batch_size :int,
        num_layers: int, 
        seq_len: int, 
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str,
        device: str,
        storage: str,
        chunk_size: int = 4096
        ):
        super().__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.storage = storage
        self.dtype = dtype
        self.device_h_cache :torch.Tensor = torch.zeros((1, self.seq_len, self.num_heads * self.head_dim), device=self.device, dtype=self.dtype)
        self.device_key_cache :torch.Tensor = torch.zeros((self.seq_len, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
        self.device_value_cache :torch.Tensor = torch.zeros((self.seq_len, self.num_kv_heads, self.head_dim), device=self.device, dtype=self.dtype)
        
        self.key_cache :list[torch.Tensor] = [torch.zeros((self.batch_size, self.num_kv_heads, self.seq_len, self.head_dim), device=self.storage, dtype=self.dtype, pin_memory=True) for _ in range(self.num_layers)]
        self.value_cache :list[torch.Tensor] = [torch.zeros((self.batch_size, self.num_kv_heads, self.seq_len, self.head_dim), device=self.storage, dtype=self.dtype, pin_memory=True) for _ in range(self.num_layers)]
        
        self.chunk_size = chunk_size
        self.num_chunk = ((self.seq_len // self.chunk_size ) if (self.seq_len % self.chunk_size  > 0) else (self.seq_len // self.chunk_size  - 1)) + 1

        self.chunk_start = [i * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end = [(i+1) * self.chunk_size for i in range(self.num_chunk)]
        self.chunk_end[-1] = self.seq_len
        self._seen_tokens = self.seq_len

    def save_kv_cache(self, k: torch.Tensor, v: torch.Tensor, start:int, end:int, layer_idx: int):
        
        
        self.device_key_cache[start:end,:,:].copy_(k[0].transpose(0,1), non_blocking=True)
        self.device_value_cache[start:end,:,:].copy_(v[0], non_blocking=True)

        self.key_cache[layer_idx][:,:, start:end,:].copy_(k, non_blocking=True)
        self.value_cache[layer_idx][:,:, start:end,:].copy_(v.transpose(1,2), non_blocking=True)
        return self.device_key_cache[:end], self.device_value_cache[:end]
    
    def save_h_cache(self, h: torch.Tensor, start:int, end:int):
        self.device_h_cache[:,start:end,:].copy_(h, non_blocking=True)
    

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states.to(self.key_cache[layer_idx].device)], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states.to(self.value_cache[layer_idx].device)], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def to_gpu(self):
        self.device_h_cache = None
        self.device_v_cache = None
        self.device_k_cache = None
        torch.cuda.empty_cache()

        for i in range(self.num_layers):
            self.key_cache[i] = self.key_cache[i].to(self.device)
            self.value_cache[i] = self.value_cache[i].to(self.device)
        
        self.device, self.storage = self.storage, self.device
    

    def partial_to_gpu(self, layers:int = 13):
        self.device_h_cache = None
        self.device_v_cache = None
        self.device_k_cache = None
        torch.cuda.empty_cache()

        for i in range(layers):
            self.key_cache[i] = self.key_cache[i].to(self.device)
            self.value_cache[i] = self.value_cache[i].to(self.device)






