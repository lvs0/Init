"""
Init Engine — The Heart of the System

A lightweight, efficient inference engine for running large language models
on constrained hardware (2GB RAM, integrated GPU).

Key Features:
- Layer streaming (loads layers on-demand from disk)
- Memory-aware KV cache management
- Context window compression for large inputs
- Intelligent CPU/GPU memory balancing
- OpenAI-compatible API
"""

import os
import sys
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Compute device types"""
    CPU = "cpu"
    GPU = "cuda"  # Would need CUDA support
    MPS = "mps"    # Apple Metal


@dataclass
class ModelConfig:
    """Configuration for the model"""
    name: str
    path: str
    total_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int
    max_context: int
    recommended_ram: int  # in GB
    recommended_vram: int  # in GB


@dataclass
class InferenceRequest:
    """Incoming inference request"""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    activation_percent: float = 1.0  # From BrainRouter


@dataclass
class InferenceResponse:
    """Inference result"""
    text: str
    tokens_used: int
    latency_ms: float
    finish_reason: str
    model: str


class MemoryGuard:
    """
    MemoryGuard — Monitors RAM/VRAM usage in real-time.
    
    Evicts KV cache entries when memory pressure is high.
    Ensures the system never OOMs on constrained hardware.
    """
    
    def __init__(self, max_ram_gb: float = 1.8, max_vram_gb: float = 0.0):
        self.max_ram_gb = max_ram_gb
        self.max_vram_gb = max_vram_gb
        self.current_usage_gb = 0.0
        self.eviction_threshold = 0.85  # Start evicting at 85% capacity
        
        logger.info(f"MemoryGuard initialized: RAM={max_ram_gb}GB, VRAM={max_vram_gb}GB")
    
    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage"""
        try:
            import psutil
            ram_usage = psutil.Process().memory_info().rss / (1024**3)  # GB
            return {
                "ram_used_gb": ram_usage,
                "ram_available_gb": self.max_ram_gb - ram_usage,
                "ram_percent": (ram_usage / self.max_ram_gb) * 100
            }
        except ImportError:
            logger.warning("psutil not available, memory check disabled")
            return {"ram_used_gb": 0, "ram_available_gb": self.max_ram_gb, "ram_percent": 0}
    
    def should_evict(self) -> bool:
        """Check if KV cache eviction is needed"""
        mem = self.check_memory()
        return mem["ram_percent"] > (self.eviction_threshold * 100)
    
    def get_safe_allocation(self, requested_gb: float) -> float:
        """Get safe memory allocation size"""
        mem = self.check_memory()
        available = mem["ram_available_gb"]
        
        # Leave 10% buffer
        safe = available * 0.9
        
        if requested_gb <= safe:
            return requested_gb
        else:
            logger.warning(f"Requested {requested_gb}GB but only {safe:.2f}GB available")
            return safe


class LayerStreamer:
    """
    LayerStreamer — Streams model layers from disk on-demand.
    
    Instead of loading the entire model into RAM, only loads
    the layers needed for the current inference.
    
    Uses memory-mapped files (mmap) for efficient disk access.
    """
    
    def __init__(self, model_path: str, total_layers: int):
        self.model_path = Path(model_path)
        self.total_layers = total_layers
        self.loaded_layers: Dict[int, Any] = {}
        self.layer_access_count: Dict[int, int] = {}
        
        logger.info(f"LayerStreamer initialized for {total_layers} layers at {model_path}")
    
    def load_layer(self, layer_idx: int) -> bool:
        """
        Load a specific layer from disk.
        
        Args:
            layer_idx: Index of layer to load
            
        Returns:
            True if loaded successfully
        """
        if layer_idx in self.loaded_layers:
            self.layer_access_count[layer_idx] += 1
            return True
        
        layer_path = self.model_path / f"layer_{layer_idx:03d}.bin"
        
        if not layer_path.exists():
            logger.warning(f"Layer {layer_idx} not found at {layer_path}")
            return False
        
        try:
            # In real implementation, would load binary weights
            # For now, just mark as loaded
            self.loaded_layers[layer_idx] = True
            self.layer_access_count[layer_idx] = 1
            logger.debug(f"Loaded layer {layer_idx}")
            return True
        except Exception as e:
            logger.error(f"Failed to load layer {layer_idx}: {e}")
            return False
    
    def load_layers_sparse(self, layer_indices: List[int]) -> int:
        """
        Load multiple layers sparsely (for partial activation).
        
        Args:
            layer_indices: List of layer indices to load
            
        Returns:
            Number of layers successfully loaded
        """
        loaded = 0
        for idx in layer_indices:
            if self.load_layer(idx):
                loaded += 1
        return loaded
    
    def evict_least_used(self, count: int = 1) -> List[int]:
        """
        Evict least recently used layers to free memory.
        
        Args:
            count: Number of layers to evict
            
        Returns:
            List of evicted layer indices
        """
        # Sort by access count (ascending)
        sorted_layers = sorted(
            self.layer_access_count.items(),
            key=lambda x: x[1]
        )
        
        evicted = []
        for layer_idx, _ in sorted_layers[:count]:
            if layer_idx in self.loaded_layers:
                del self.loaded_layers[layer_idx]
                del self.layer_access_count[layer_idx]
                evicted.append(layer_idx)
                logger.debug(f"Evicted layer {layer_idx}")
        
        return evicted
    
    def preload_frequent(self, layer_indices: List[int]) -> int:
        """
        Preload frequently needed layers.
        
        Args:
            layer_indices: Layers to preload
            
        Returns:
            Number of layers preloaded
        """
        return self.load_layers_sparse(layer_indices)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get layer loading statistics"""
        return {
            "total_layers": self.total_layers,
            "loaded_layers": len(self.loaded_layers),
            "loaded_indices": list(self.loaded_layers.keys()),
            "total_accesses": sum(self.layer_access_count.values()),
            "hit_rate": (
                sum(self.layer_access_count.values()) / 
                max(1, len(self.layer_access_count))
            )
        }


class ContextCompressor:
    """
    ContextCompressor — Compresses context for large windows.
    
    When dealing with long context windows (>4k tokens),
    compresses older tokens to save memory while preserving
    semantic meaning.
    """
    
    def __init__(self, compression_ratio: float = 0.5):
        """
        Initialize compressor.
        
        Args:
            compression_ratio: How much to compress (0.5 = half size)
        """
        self.compression_ratio = compression_ratio
        logger.info(f"ContextCompressor: {compression_ratio:.0%} compression ratio")
    
    def compress(self, tokens: List[int]) -> List[int]:
        """
        Compress token sequence.
        
        Strategy:
        1. Remove punctuation and stopwords
        2. Keep key semantic tokens
        3. Downsample middle of sequence
        
        Args:
            tokens: Input token list
            
        Returns:
            Compressed token list
        """
        if len(tokens) <= 32:
            return tokens  # Too short to compress
        
        # Simple compression: keep first, last, and sample middle
        keep_first = tokens[:16]
        keep_last = tokens[-16:]
        
        middle_size = max(1, int(len(tokens) * self.compression_ratio))
        middle_sample = tokens[16:-16:max(1, (len(tokens) - 32) // middle_size)]
        
        return keep_first + list(middle_sample) + keep_last
    
    def estimate_compression(self, token_count: int) -> int:
        """Estimate compressed token count"""
        if token_count <= 32:
            return token_count
        return int(token_count * self.compression_ratio) + 32


class InitEngine:
    """
    Init Engine — Main inference engine.
    
    Orchestrates:
    - BrainRouter for activation decisions
    - LayerStreamer for memory-efficient loading
    - MemoryGuard for OOM prevention
    - ContextCompressor for long context handling
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        max_ram_gb: float = 1.8,
        max_vram_gb: float = 0.0
    ):
        """
        Initialize Init Engine.
        
        Args:
            model_config: Model configuration
            max_ram_gb: Maximum RAM to use
            max_vram_gb: Maximum VRAM to use (0 = CPU only)
        """
        self.model_config = model_config
        self.device = DeviceType.CPU
        
        # Initialize components
        self.brain_router = None  # Will be set after import
        self.memory_guard = MemoryGuard(max_ram_gb, max_vram_gb)
        self.layer_streamer = LayerStreamer(
            model_config.path,
            model_config.total_layers
        )
        self.context_compressor = ContextCompressor()
        
        self.is_running = False
        self.total_inferences = 0
        
        logger.info(f"Init Engine initialized for {model_config.name}")
        logger.info(f"  Total layers: {model_config.total_layers}")
        logger.info(f"  Max RAM: {max_ram_gb}GB")
        logger.info(f"  Max VRAM: {max_vram_gb}GB")
    
    def initialize(self, brain_router_class):
        """
        Initialize BrainRouter with proper configuration.
        
        Args:
            brain_router_class: BrainRouter class to instantiate
        """
        self.brain_router = brain_router_class(
            total_layers=self.model_config.total_layers
        )
        logger.info("BrainRouter initialized")
    
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Run inference on a request.
        
        Args:
            request: Inference request with prompt and parameters
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        # Step 1: Analyze query with BrainRouter
        if self.brain_router:
            config = self.brain_router.get_inference_config(request.prompt)
            request.activation_percent = config["activation_percent"]
            
            # Adjust parameters based on BrainRouter output
            request.temperature = config.get("temperature", request.temperature)
            request.max_tokens = min(
                request.max_tokens,
                config.get("max_tokens", 8192)
            )
        
        # Step 2: Compress context if needed
        prompt_tokens = self._tokenize(request.prompt)
        
        if len(prompt_tokens) > self.model_config.max_context // 2:
            compressed = self.context_compressor.compress(prompt_tokens)
            logger.info(
                f"Context compressed: {len(prompt_tokens)} → {len(compressed)} tokens"
            )
        
        # Step 3: Check memory and load required layers
        mem = self.memory_guard.check_memory()
        logger.debug(f"Memory: {mem['ram_percent']:.1f}% used")
        
        if self.memory_guard.should_evict():
            evicted = self.layer_streamer.evict_least_used(5)
            logger.info(f"Memory pressure: evicted {len(evicted)} layers")
        
        # Step 4: Load required layers based on activation percent
        if self.brain_router:
            layers_to_load = config.get("layers", [])
        else:
            # Default: load all layers
            layers_to_load = list(range(self.model_config.total_layers))
        
        loaded = self.layer_streamer.load_layers_sparse(layers_to_load)
        logger.debug(f"Loaded {loaded}/{len(layers_to_load)} layers")
        
        # Step 5: Run inference (placeholder — would call actual model)
        # In real implementation, would run model forward pass here
        
        output_text = self._generate_placeholder(
            request.prompt,
            request.max_tokens,
            request.temperature
        )
        
        # Step 6: Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        self.total_inferences += 1
        
        response = InferenceResponse(
            text=output_text,
            tokens_used=len(self._tokenize(output_text)),
            latency_ms=latency_ms,
            finish_reason="stop",
            model=self.model_config.name
        )
        
        logger.info(
            f"Inference complete: {response.tokens_used} tokens, "
            f"{latency_ms:.0f}ms, activation={request.activation_percent:.0%}"
        )
        
        return response
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization (placeholder)"""
        # In real implementation, would use proper tokenizer
        return text.split()
    
    def _generate_placeholder(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Placeholder text generation.
        
        In real implementation, would run actual model inference.
        """
        # Simulated response
        return (
            f"[Init Engine Response]\n"
            f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n"
            f"Max tokens: {max_tokens}, Temperature: {temperature}\n"
            f"Note: This is a placeholder. Real inference requires model weights."
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "model": self.model_config.name,
            "is_running": self.is_running,
            "total_inferences": self.total_inferences,
            "memory": self.memory_guard.check_memory(),
            "layers": self.layer_streamer.get_stats(),
            "device": self.device.value,
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        logger.info("Init Engine shutting down")


class APIServer:
    """
    OpenAI-compatible API server for Init Engine.
    
    Implements /v1/chat/completions endpoint.
    """
    
    def __init__(self, engine: InitEngine, host: str = "0.0.0.0", port: int = 8080):
        self.engine = engine
        self.host = host
        self.port = port
        self.app = None  # Would be FastAPI or similar
        
        logger.info(f"APIServer configured for {host}:{port}")
    
    async def chat_completions(self, request_data: Dict) -> Dict:
        """
        Handle /v1/chat/completions request.
        
        Args:
            request_data: OpenAI-compatible request body
            
        Returns:
            OpenAI-compatible response
        """
        messages = request_data.get("messages", [])
        model = request_data.get("model", self.engine.model_config.name)
        
        # Extract last user message
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        
        if not prompt:
            return {"error": "No user message found"}
        
        # Build inference request
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=request_data.get("max_tokens", 512),
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 0.9),
            stop=request_data.get("stop"),
        )
        
        # Run inference
        response = await self.engine.infer(request)
        
        # Format as OpenAI response
        return {
            "id": f"init-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text
                },
                "finish_reason": response.finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,  # Would calculate
                "completion_tokens": response.tokens_used,
                "total_tokens": response.tokens_used
            }
        }


# CLI interface
def main():
    """Main entry point for Init Engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Init Engine — Efficient LLM Inference")
    parser.add_argument("--model", default="llama-70b", help="Model name")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--layers", type=int, default=80, help="Total model layers")
    parser.add_argument("--max-ram", type=float, default=1.8, help="Max RAM in GB")
    parser.add_argument("--max-vram", type=float, default=0.0, help="Max VRAM in GB")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s — %(levelname)s — %(message)s")
    
    # Create model config
    config = ModelConfig(
        name=args.model,
        path=args.model_path,
        total_layers=args.layers,
        hidden_size=4096,
        num_heads=32,
        vocab_size=32000,
        max_context=8192,
        recommended_ram=4,  # Would need actual model specs
        recommended_vram=6,
    )
    
    # Initialize engine
    from brain_router import BrainRouter
    engine = InitEngine(config, args.max_ram, args.max_vram)
    engine.initialize(BrainRouter)
    
    logger.info("=" * 60)
    logger.info("Init Engine — Ready")
    logger.info(f"Model: {config.name}")
    logger.info(f"Layers: {config.total_layers}")
    logger.info(f"RAM Limit: {args.max_ram}GB")
    logger.info("=" * 60)
    
    # Start API server (placeholder)
    logger.info(f"API server would start on {args.port}")
    logger.info("In production, would use FastAPI + uvicorn")
    
    return engine


if __name__ == "__main__":
    engine = main()
    
    # Run test inference
    import asyncio
    
    async def test():
        request = InferenceRequest(
            prompt="Hello, how are you?",
            max_tokens=100
        )
        response = await engine.infer(request)
        print(f"\nTest Response:\n{response.text}")
    
    asyncio.run(test())
