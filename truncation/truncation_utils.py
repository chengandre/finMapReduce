"""
Utilities for document truncation strategies.

Provides token-aware truncation methods that fit documents within
model context windows while preserving important content.
"""

import sys
import os
from typing import Dict, Tuple, Optional, List
from abc import ABC, abstractmethod

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TruncationStrategy(ABC):
    """Abstract base class for document truncation strategies."""
    
    @abstractmethod
    def truncate(self, text: str, max_tokens: int) -> Tuple[str, Dict]:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Full document text
            max_tokens: Maximum tokens allowed
            
        Returns:
            Tuple of (truncated_text, statistics_dict)
        """
        pass


class StartTruncationStrategy(TruncationStrategy):
    """Keep the beginning of the document, truncate the end."""
    
    def truncate(self, text: str, max_tokens: int) -> Tuple[str, Dict]:
        from utils import num_tokens_from_string
        
        current_tokens = num_tokens_from_string(text, "cl100k_base")
        
        if current_tokens <= max_tokens:
            return text, {
                "strategy": "start",
                "truncated_tokens": current_tokens,
                "retention_rate": 1.0,
                "truncation_applied": False,
                "original_tokens": current_tokens
            }
        
        # Binary search for the right truncation point
        words = text.split()
        low, high = 0, len(words)
        best_text = ""
        best_tokens = 0
        
        while low <= high:
            mid = (low + high) // 2
            candidate_text = " ".join(words[:mid])
            candidate_tokens = num_tokens_from_string(candidate_text, "cl100k_base")
            
            if candidate_tokens <= max_tokens:
                best_text = candidate_text
                best_tokens = candidate_tokens
                low = mid + 1
            else:
                high = mid - 1
        
        return best_text, {
            "strategy": "start",
            "truncated_tokens": best_tokens,
            "retention_rate": best_tokens / current_tokens if current_tokens > 0 else 0.0,
            "truncation_applied": True,
            "original_tokens": current_tokens
        }


class EndTruncationStrategy(TruncationStrategy):
    """Keep the end of the document, truncate the beginning."""
    
    def truncate(self, text: str, max_tokens: int) -> Tuple[str, Dict]:
        from utils import num_tokens_from_string
        
        current_tokens = num_tokens_from_string(text, "cl100k_base")
        
        if current_tokens <= max_tokens:
            return text, {
                "strategy": "end",
                "truncated_tokens": current_tokens,
                "retention_rate": 1.0,
                "truncation_applied": False,
                "original_tokens": current_tokens
            }
        
        # Binary search for the right truncation point
        words = text.split()
        low, high = 0, len(words)
        best_text = ""
        best_tokens = 0
        
        while low <= high:
            mid = (low + high) // 2
            candidate_text = " ".join(words[-mid:]) if mid > 0 else ""
            candidate_tokens = num_tokens_from_string(candidate_text, "cl100k_base")
            
            if candidate_tokens <= max_tokens:
                best_text = candidate_text
                best_tokens = candidate_tokens
                low = mid + 1
            else:
                high = mid - 1
        
        return best_text, {
            "strategy": "end",
            "truncated_tokens": best_tokens,
            "retention_rate": best_tokens / current_tokens if current_tokens > 0 else 0.0,
            "truncation_applied": True,
            "original_tokens": current_tokens
        }


class SmartTruncationStrategy(TruncationStrategy):
    """
    Intelligent truncation that tries to preserve important sections.
    
    This is a placeholder for future enhancement - currently falls back
    to start truncation but could be extended to:
    - Preserve sections with high keyword density
    - Keep financial tables and key metrics
    - Maintain document structure (headers, etc.)
    """
    
    def truncate(self, text: str, max_tokens: int) -> Tuple[str, Dict]:
        from utils import num_tokens_from_string
        
        current_tokens = num_tokens_from_string(text, "cl100k_base")
        
        if current_tokens <= max_tokens:
            return text, {
                "strategy": "smart",
                "truncated_tokens": current_tokens,
                "retention_rate": 1.0,
                "truncation_applied": False,
                "original_tokens": current_tokens
            }
        
        # For now, use start truncation as fallback
        # TODO: Implement smart selection of important sections
        start_strategy = StartTruncationStrategy()
        truncated_text, stats = start_strategy.truncate(text, max_tokens)
        
        # Update strategy name in stats
        stats["strategy"] = "smart"
        
        return truncated_text, stats


class TruncationManager:
    """
    Manages document truncation using different strategies.
    
    Provides a unified interface for truncating documents to fit
    within token limits while preserving content based on strategy.
    """
    
    _strategies = {
        "start": StartTruncationStrategy,
        "end": EndTruncationStrategy,
        "smart": SmartTruncationStrategy
    }
    
    def __init__(self, strategy: str = "start", max_tokens: int = 100000):
        """
        Initialize truncation manager.
        
        Args:
            strategy: Truncation strategy ("start", "end", "smart")
            max_tokens: Maximum tokens to allow in truncated document
        """
        if strategy not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")
        
        self.strategy_name = strategy
        self.strategy = self._strategies[strategy]()
        self.max_tokens = max_tokens
    
    def truncate_document(self, text: str) -> Tuple[str, Dict]:
        """
        Truncate a document using the configured strategy.
        
        Args:
            text: Full document text
            
        Returns:
            Tuple of (truncated_text, truncation_statistics)
        """
        if not text:
            return "", {
                "strategy": self.strategy_name,
                "truncated_tokens": 0,
                "retention_rate": 0.0,
                "truncation_applied": False,
                "original_tokens": 0
            }
        
        return self.strategy.truncate(text, self.max_tokens)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available truncation strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """
        Register a custom truncation strategy.
        
        Args:
            name: Name to register the strategy under
            strategy_class: Strategy class (must inherit from TruncationStrategy)
        """
        if not issubclass(strategy_class, TruncationStrategy):
            raise TypeError(f"{strategy_class} must inherit from TruncationStrategy")
        
        cls._strategies[name] = strategy_class


def estimate_prompt_tokens(question: str, template_overhead: int = 500) -> int:
    """
    Estimate tokens needed for prompt template + question.
    
    Args:
        question: Question text
        template_overhead: Estimated tokens for prompt template overhead
        
    Returns:
        Estimated prompt tokens
    """
    from utils import num_tokens_from_string
    
    question_tokens = num_tokens_from_string(question, "cl100k_base")
    return question_tokens + template_overhead


def calculate_max_document_tokens(context_window: int, 
                                question: str, 
                                buffer_tokens: int = 2000,
                                template_overhead: int = 500) -> int:
    """
    Calculate maximum tokens available for document content.
    
    Args:
        context_window: Model's maximum context window
        question: Question text
        buffer_tokens: Safety buffer for response tokens
        template_overhead: Estimated tokens for prompt template
        
    Returns:
        Maximum tokens available for document
    """
    prompt_tokens = estimate_prompt_tokens(question, template_overhead)
    available_tokens = context_window - prompt_tokens - buffer_tokens
    
    return max(1000, available_tokens)  # Ensure minimum viable document size


def validate_truncation_config(strategy: str, 
                             context_window: int, 
                             max_document_tokens: Optional[int] = None) -> Dict[str, str]:
    """
    Validate truncation configuration and return any warnings.
    
    Args:
        strategy: Truncation strategy name
        context_window: Model context window size
        max_document_tokens: Optional override for max document tokens
        
    Returns:
        Dictionary with validation results and warnings
    """
    warnings = []
    errors = []
    
    # Check strategy validity
    if strategy not in TruncationManager.get_available_strategies():
        available = TruncationManager.get_available_strategies()
        errors.append(f"Unknown strategy '{strategy}'. Available: {available}")
    
    # Check context window size
    if context_window < 4000:
        warnings.append(f"Context window ({context_window}) is very small. Consider using a larger model.")
    elif context_window > 2000000:
        warnings.append(f"Context window ({context_window}) is very large. Token counting may be slow.")
    
    # Check max document tokens
    if max_document_tokens is not None:
        if max_document_tokens >= context_window:
            errors.append(f"max_document_tokens ({max_document_tokens}) must be less than context_window ({context_window})")
        elif max_document_tokens < 1000:
            warnings.append(f"max_document_tokens ({max_document_tokens}) is very small. Results may be poor.")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }