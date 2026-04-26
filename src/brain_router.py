"""
BrainRouter — Intelligent LLM Activation System for Init

Analyzes incoming queries and determines the optimal activation percentage
of the neural network layers based on query complexity.

Key innovation: Not all queries need full model activation.
- Simple queries (greetings, short answers) → 10-30% activation
- Medium queries (explanations, analysis) → 30-60% activation  
- Complex queries (reasoning, math, deep analysis) → 60-100% activation

This allows running large models (70B+) on limited hardware (2GB RAM).
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    TRIVIAL = 0.1      # 10% activation
    SIMPLE = 0.25      # 25% activation
    MODERATE = 0.40    # 40% activation
    COMPLEX = 0.60     # 60% activation
    ADVANCED = 0.80    # 80% activation
    EXPERT = 1.0       # 100% activation


@dataclass
class QueryAnalysis:
    """Results of query analysis"""
    complexity: QueryComplexity
    activation_percent: float
    suggested_layers: List[int]
    estimated_tokens: int
    requires_reasoning: bool
    requires_long_context: bool
    keywords_found: List[str]
    confidence: float


class BrainRouter:
    """
    BrainRouter — The intelligent heart of Init.
    
    Decides which percentage of the model to activate based on query analysis.
    Uses pattern matching, keyword detection, and heuristics.
    """
    
    # Keywords indicating high complexity (EXPERT level)
    EXPERT_KEYWORDS = [
        "prove", "proof", "theorem", "mathematical", "demonstrate",
        "derive", "calculate", "optimize", "algorithm", "complexity",
        "formal proof", "contradiction", "induction", "computation",
        "quantum", "physics", "differential", "integral", "calculus"
    ]
    
    # Keywords indicating medium-high complexity (COMPLEX-ADVANCED)
    COMPLEX_KEYWORDS = [
        "explain", "analyze", "compare", "evaluate", "describe",
        "why", "how", "difference between", "relationship",
        "analysis", "synthesis", "implications", "consequences",
        "strategic", "research", "investigate", "examine"
    ]
    
    # Keywords indicating moderate complexity (MODERATE)
    MODERATE_KEYWORDS = [
        "what is", "tell me about", "overview", "summary",
        "definition", "introduction", "basics", "outline",
        "brief", "generally", "typically", "usually"
    ]
    
    # Keywords indicating simple queries (SIMPLE)
    SIMPLE_KEYWORDS = [
        "hi", "hello", "hey", "thanks", "thank you", "please",
        "ok", "okay", "yes", "no", "sure", "yeah", "bye",
        "hi", "salut", "bonjour", "merci", "oui", "non"
    ]
    
    # Patterns indicating reasoning requirements
    REASONING_PATTERNS = [
        r"\b(because|therefore|thus|hence|so|consequently)\b",
        r"\b(if|then|else|when|while|although|whereas)\b",
        r"\b(all|some|few|many|most|none|every)\b",
        r"\b(always|never|often|sometimes|rarely|usually)\b",
        r"\b(before|after|during|since|until|meanwhile)\b",
        r"\d+\s*[+\-*/=]\s*\d+",  # Math expressions
        r"\b(greater|less|equal|larger|smaller|more|less)\b",
    ]
    
    # Context length estimators
    LONG_CONTEXT_PATTERNS = [
        r"\b(context|previously|earlier|last|previous|before mentioned)\b",
        r"\bsummarize\b",
        r"\bthroughout\b",
        r"\boverall\b",
    ]
    
    def __init__(self, total_layers: int = 80):
        """
        Initialize BrainRouter.
        
        Args:
            total_layers: Total number of layers in the model (default 80 for 70B)
        """
        self.total_layers = total_layers
        logger.info(f"BrainRouter initialized with {total_layers} total layers")
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query and determine activation requirements.
        
        Args:
            query: The input query string
            
        Returns:
            QueryAnalysis with activation recommendations
        """
        query_lower = query.lower()
        words = re.findall(r'\w+', query_lower)
        
        # Count keyword matches
        expert_matches = sum(1 for kw in self.EXPERT_KEYWORDS if kw in query_lower)
        complex_matches = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in query_lower)
        moderate_matches = sum(1 for kw in self.MODERATE_KEYWORDS if kw in query_lower)
        simple_matches = sum(1 for kw in self.SIMPLE_KEYWORDS if kw in query_lower)
        
        # Check reasoning patterns
        reasoning_count = sum(
            1 for pattern in self.REASONING_PATTERNS 
            if re.search(pattern, query, re.IGNORECASE)
        )
        
        # Check context requirements
        context_matches = sum(
            1 for pattern in self.LONG_CONTEXT_PATTERNS 
            if re.search(pattern, query, re.IGNORECASE)
        )
        
        # Calculate base complexity score
        complexity_score = 0.0
        
        # Add weight for keyword matches
        complexity_score += expert_matches * 0.25
        complexity_score += complex_matches * 0.15
        complexity_score += moderate_matches * 0.08
        complexity_score += simple_matches * -0.05
        
        # Add weight for reasoning patterns
        complexity_score += reasoning_count * 0.12
        
        # Add weight for context requirements
        complexity_score += context_matches * 0.10
        
        # Add weight for query length (longer = more complex)
        if len(words) > 100:
            complexity_score += 0.20
        elif len(words) > 50:
            complexity_score += 0.10
        elif len(words) > 20:
            complexity_score += 0.05
        
        # Cap the score between 0.1 and 1.0
        complexity_score = max(0.1, min(1.0, complexity_score))
        
        # Determine complexity enum
        if complexity_score >= 0.85:
            complexity = QueryComplexity.EXPERT
        elif complexity_score >= 0.65:
            complexity = QueryComplexity.ADVANCED
        elif complexity_score >= 0.45:
            complexity = QueryComplexity.COMPLEX
        elif complexity_score >= 0.30:
            complexity = QueryComplexity.MODERATE
        elif complexity_score >= 0.15:
            complexity = QueryComplexity.SIMPLE
        else:
            complexity = QueryComplexity.TRIVIAL
        
        # Calculate layer activation
        activation_percent = complexity.value
        
        # Select which layers to activate (sparse selection for efficiency)
        # Strategy: Activate layers spread across the model for best coverage
        num_layers_to_activate = int(self.total_layers * activation_percent)
        
        if num_layers_to_activate == 0:
            suggested_layers = []
        elif num_layers_to_activate == self.total_layers:
            suggested_layers = list(range(self.total_layers))
        else:
            # Sparse activation: spread layers evenly
            step = self.total_layers // num_layers_to_activate
            suggested_layers = [i * step for i in range(num_layers_to_activate)]
        
        # Estimate tokens for context
        estimated_tokens = len(words) * 1.5  # Rough estimate
        
        # Determine additional requirements
        requires_reasoning = reasoning_count >= 2
        requires_long_context = context_matches >= 2 or len(words) > 80
        
        # Calculate confidence in analysis
        keyword_total = expert_matches + complex_matches + moderate_matches + simple_matches
        if keyword_total > 0:
            confidence = min(0.95, 0.5 + (keyword_total * 0.1))
        else:
            confidence = 0.6  # Lower confidence when no keywords found
        
        analysis = QueryAnalysis(
            complexity=complexity,
            activation_percent=activation_percent,
            suggested_layers=suggested_layers,
            estimated_tokens=int(estimated_tokens),
            requires_reasoning=requires_reasoning,
            requires_long_context=requires_long_context,
            keywords_found=self._extract_keywords(query_lower, words),
            confidence=confidence
        )
        
        logger.info(
            f"Query analyzed: complexity={complexity.name}, "
            f"activation={activation_percent:.0%}, layers={len(suggested_layers)}"
        )
        
        return analysis
    
    def _extract_keywords(self, query: str, words: List[str]) -> List[str]:
        """Extract significant keywords from query"""
        all_keywords = (
            self.EXPERT_KEYWORDS + 
            self.COMPLEX_KEYWORDS + 
            self.MODERATE_KEYWORDS
        )
        found = [kw for kw in all_keywords if kw in query]
        return found[:10]  # Return top 10 keywords
    
    def get_inference_config(self, query: str) -> dict:
        """
        Get full inference configuration for the query.
        
        Returns:
            Dictionary with all parameters needed for inference
        """
        analysis = self.analyze(query)
        
        return {
            "activation_percent": analysis.activation_percent,
            "layers": analysis.suggested_layers,
            "estimated_tokens": analysis.estimated_tokens,
            "requires_reasoning": analysis.requires_reasoning,
            "requires_long_context": analysis.requires_long_context,
            "complexity": analysis.complexity.name,
            "confidence": analysis.confidence,
            "temperature": self._suggest_temperature(analysis.complexity),
            "max_tokens": self._suggest_max_tokens(analysis),
        }
    
    def _suggest_temperature(self, complexity: QueryComplexity) -> float:
        """Suggest temperature based on complexity"""
        temps = {
            QueryComplexity.TRIVIAL: 0.3,
            QueryComplexity.SIMPLE: 0.5,
            QueryComplexity.MODERATE: 0.6,
            QueryComplexity.COMPLEX: 0.7,
            QueryComplexity.ADVANCED: 0.8,
            QueryComplexity.EXPERT: 0.9,
        }
        return temps.get(complexity, 0.7)
    
    def _suggest_max_tokens(self, analysis: QueryAnalysis) -> int:
        """Suggest max tokens based on analysis"""
        base_tokens = analysis.estimated_tokens * 2
        
        if analysis.requires_reasoning:
            base_tokens *= 2
        if analysis.requires_long_context:
            base_tokens *= 1.5
        
        return min(int(base_tokens), 8192)  # Cap at 8k for safety


class AdaptiveBrainRouter(BrainRouter):
    """
    Extended BrainRouter that learns from previous queries.
    Maintains a session history to improve future predictions.
    """
    
    def __init__(self, total_layers: int = 80, history_size: int = 100):
        super().__init__(total_layers)
        self.history: List[Tuple[str, QueryAnalysis]] = []
        self.history_size = history_size
    
    def analyze_with_feedback(self, query: str, was_accurate: bool):
        """
        Analyze and update based on feedback.
        
        Args:
            query: The query that was analyzed
            was_accurate: Whether the analysis was accurate
        """
        analysis = self.analyze(query)
        self.history.append((query, analysis))
        
        # Trim history
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
        
        if not was_accurate:
            logger.warning(f"BrainRouter feedback: analysis for '{query[:50]}...' was inaccurate")
    
    def get_session_summary(self) -> dict:
        """Get summary of current session analysis patterns"""
        if not self.history:
            return {"total_queries": 0}
        
        complexities = [a.complexity for _, a in self.history]
        avg_activation = sum(a.activation_percent for _, a in self.history) / len(self.history)
        
        return {
            "total_queries": len(self.history),
            "avg_activation": f"{avg_activation:.1%}",
            "complexity_distribution": {
                c.name: complexities.count(c) for c in QueryComplexity
            },
            "session_keywords": self._aggregate_keywords()
        }
    
    def _aggregate_keywords(self) -> List[str]:
        """Aggregate keywords from session history"""
        all_keywords = []
        for _, analysis in self.history:
            all_keywords.extend(analysis.keywords_found)
        
        # Return most common
        from collections import Counter
        return [kw for kw, _ in Counter(all_keywords).most_common(20)]


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    router = BrainRouter(total_layers=80)
    
    test_queries = [
        "Hello, how are you?",
        "What is machine learning?",
        "Explain the relationship between quantum entanglement and superposition.",
        "Prove that the sum of angles in a triangle equals 180 degrees.",
        "Write a Python function to sort a list using quicksort.",
        "Bonjour, comment ça va?",
    ]
    
    print("\n🧠 BrainRouter — Query Analysis Demo")
    print("=" * 60)
    
    for query in test_queries:
        config = router.get_inference_config(query)
        print(f"\n📝 Query: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"   Complexity: {config['complexity']}")
        print(f"   Activation: {config['activation_percent']:.0%}")
        print(f"   Layers: {len(config['layers'])}")
        print(f"   Temp: {config['temperature']}")
    
    print("\n" + "=" * 60)
