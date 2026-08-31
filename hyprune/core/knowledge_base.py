"""
Knowledge base for tracking failed hypotheses to prevent re-generation.
"""
from __future__ import annotations
import json
import os
from typing import Any
from hyprune.core.hypothesis import Hypothesis

class NegativeKnowledgeBase:
    """Stores failed hypotheses to prevent re-generation."""
    
    def __init__(self) -> None:
        # Maps task_hash -> list of failed hypothesis operations and levels
        self.failed_patterns: dict[str, list[dict[str, Any]]] = {}
    
    def record_failure(self, task_hash: str, hypothesis: Hypothesis) -> None:
        """Record a failed hypothesis for a specific task."""
        if task_hash not in self.failed_patterns:
            self.failed_patterns[task_hash] = []
            
        pattern = {
            "signature": hypothesis.signature(),
            "operations": hypothesis.operations,
            "abstraction_level": hypothesis.abstraction_level.value
        }
        self.failed_patterns[task_hash].append(pattern)
        
    def is_redundant(self, hypothesis: Hypothesis, task_hash: str) -> bool:
        """
        Check if a hypothesis is redundant (too similar to a failed one).
        Uses Jaccard similarity on operation sets with a threshold of 0.9.
        """
        if task_hash not in self.failed_patterns:
            return False
            
        current_ops = set(hypothesis.operations)
        
        for pattern in self.failed_patterns[task_hash]:
            # Exact signature match
            if pattern["signature"] == hypothesis.signature():
                return True
                
            # Similarity match
            failed_ops = set(pattern["operations"])
            intersection = len(current_ops.intersection(failed_ops))
            union = len(current_ops.union(failed_ops))
            
            if union > 0:
                similarity = intersection / union
                if similarity >= 0.9:
                    return True
                    
        return False
        
    def get_failures(self, task_hash: str) -> list[str]:
        """Get a list of failed signatures for a task."""
        if task_hash not in self.failed_patterns:
            return []
        return [pattern["signature"] for pattern in self.failed_patterns[task_hash]]
        
    def clear(self, task_hash: str) -> None:
        """Clear the failure records for a specific task."""
        if task_hash in self.failed_patterns:
            del self.failed_patterns[task_hash]
    
    def save(self, path: str) -> None:
        """Save the knowledge base to a JSON file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.failed_patterns, f, indent=2)
            
    def load(self, path: str) -> None:
        """Load the knowledge base from a JSON file."""
        if not os.path.exists(path):
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            self.failed_patterns = json.load(f)
