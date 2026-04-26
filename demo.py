#!/usr/bin/env python3
"""
Init Engine — Demo Script

Test the BrainRouter and InitEngine with sample queries.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from brain_router import BrainRouter
from engine import InitEngine, ModelConfig, InferenceRequest


def demo_brainrouter():
    """Demo BrainRouter query analysis"""
    print("\n" + "=" * 60)
    print("🧠 BrainRouter — Query Analysis Demo")
    print("=" * 60)
    
    router = BrainRouter(total_layers=80)
    
    test_queries = [
        # Simple
        "Hello, how are you?",
        "Merci beaucoup!",
        "Oui, exactement.",
        
        # Moderate
        "What is machine learning?",
        "Explain how photosynthesis works.",
        "Décris le processus de mitose.",
        
        # Complex
        "Compare transformer architectures vs RNNs for NLP.",
        "What are the implications of quantum computing on cryptography?",
        "Analyse la relation entre PIB et développement durable.",
        
        # Expert
        "Prove that P ≠ NP or provide a counterexample.",
        "Derive the Euler-Lagrange equation from first principles.",
        "Établis la démonstration formelle du théorème de Gödel.",
    ]
    
    results = []
    
    for query in test_queries:
        config = router.get_inference_config(query)
        
        complexity_emoji = {
            "TRIVIAL": "🟢",
            "SIMPLE": "🟡", 
            "MODERATE": "🟠",
            "COMPLEX": "🔵",
            "ADVANCED": "🟣",
            "EXPERT": "🔴",
        }
        
        emoji = complexity_emoji.get(config['complexity'], "⚪")
        
        print(f"\n{emoji} {config['complexity']}")
        print(f"   Query: {query[:55]}{'...' if len(query) > 55 else ''}")
        print(f"   Activation: {config['activation_percent']:.0%} | Layers: {len(config['layers'])}")
        print(f"   Temperature: {config['temperature']} | Max tokens: {config['max_tokens']}")
        
        results.append({
            'query': query[:50],
            'complexity': config['complexity'],
            'activation': config['activation_percent']
        })
    
    return results


async def demo_engine():
    """Demo InitEngine inference"""
    print("\n" + "=" * 60)
    print("⚙️ Init Engine — Inference Demo")
    print("=" * 60)
    
    config = ModelConfig(
        name="llama-70b-simulated",
        path="./models/llama-70b",
        total_layers=80,
        hidden_size=4096,
        num_heads=32,
        vocab_size=32000,
        max_context=8192,
        recommended_ram=4,
        recommended_vram=6,
    )
    
    engine = InitEngine(config, max_ram_gb=1.8)
    engine.initialize(BrainRouter)
    
    requests = [
        InferenceRequest(prompt="Hello", max_tokens=50),
        InferenceRequest(prompt="What is AI?", max_tokens=100, temperature=0.7),
        InferenceRequest(prompt="Explain quantum entanglement", max_tokens=200, temperature=0.9),
    ]
    
    for req in requests:
        print(f"\n📝 Request: {req.prompt[:40]}{'...' if len(req.prompt) > 40 else ''}")
        print(f"   Max tokens: {req.max_tokens}, Temp: {req.temperature}")
        
        response = await engine.infer(req)
        
        print(f"\n   ⏱️ Latency: {response.latency_ms:.0f}ms")
        print(f"   📊 Tokens: {response.tokens_used}")
        print(f"   📄 Response: {response.text[:100]}{'...' if len(response.text) > 100 else ''}")
    
    # Status
    status = engine.get_status()
    print(f"\n📊 Engine Status:")
    print(f"   Model: {status['model']}")
    print(f"   Inferences: {status['total_inferences']}")
    print(f"   Memory: {status['memory']['ram_percent']:.1f}%")
    print(f"   Layers loaded: {status['layers']['loaded_layers']}/{status['layers']['total_layers']}")


def main():
    """Main demo entry point"""
    print("\n" + "=" * 60)
    print("🚀 Init Engine — Full System Demo")
    print("=" * 60)
    print("\nTesting intelligent LLM inference on 2GB RAM hardware")
    print("Target: Lenovo X250 / 70B+ models / 10-100% activation\n")
    
    # Demo 1: BrainRouter
    demo_brainrouter()
    
    # Demo 2: Engine (requires model weights for full run)
    # Uncomment to run:
    # asyncio.run(demo_engine())
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download real model weights (Llama 70B or similar)")
    print("2. Place in ./models/llama-70b/")
    print("3. Run: python -m init.src.engine --model-path ./models/llama-70b --verbose")
    print("4. Access API at http://localhost:8080/v1/chat/completions")


if __name__ == "__main__":
    main()
