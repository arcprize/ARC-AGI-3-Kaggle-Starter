"""
Core data structures for representing rules and hypotheses in HyPrune.
"""
from __future__ import annotations
import dataclasses
import uuid
import enum
from typing import Optional, Any
import numpy as np


class AbstractionLevel(enum.Enum):
    """Level of abstraction for a given hypothesis."""
    PIXEL = 'pixel'
    OBJECT = 'object'
    SPATIAL = 'spatial'
    COMPOSITIONAL = 'compositional'


@dataclasses.dataclass
class Hypothesis:
    """Represents a hypothesized rule for solving an ARC task."""
    id: str  # UUID
    description: str  # Natural language description of the rule
    program: Optional[str]  # Executable Python implementation
    confidence: float  # 0.0 - 1.0
    evidence_for: list[str]  # IDs of training pairs it explains
    evidence_against: list[str]  # IDs of training pairs it contradicts
    parent_id: Optional[str]  # Lineage tracking
    source: str  # Which teacher model or 'template' or 'student'
    abstraction_level: AbstractionLevel
    operations: list[str]  # e.g. ['rotate_90', 'fill_color']
    preconditions: dict[str, Any]  # When this rule applies
    generation: int  # Which refinement iteration
    
    @classmethod
    def create(
        cls, 
        description: str,
        source: str,
        abstraction_level: AbstractionLevel,
        operations: list[str],
        preconditions: Optional[dict[str, Any]] = None,
        program: Optional[str] = None,
        confidence: float = 0.5,
        parent_id: Optional[str] = None,
        generation: int = 0
    ) -> Hypothesis:
        """Factory method to create a new Hypothesis."""
        return cls(
            id=str(uuid.uuid4()),
            description=description,
            program=program,
            confidence=confidence,
            evidence_for=[],
            evidence_against=[],
            parent_id=parent_id,
            source=source,
            abstraction_level=abstraction_level,
            operations=operations.copy(),
            preconditions=preconditions or {},
            generation=generation
        )
    
    def is_alive(self) -> bool:
        """Check if the hypothesis is still considered viable."""
        return self.confidence > 0.1
    
    def consistency_score(self) -> float:
        """Calculate the consistency score based on evidence."""
        total_evidence = len(self.evidence_for) + len(self.evidence_against)
        if total_evidence == 0:
            return 0.0
        return len(self.evidence_for) / total_evidence
    
    def signature(self) -> str:
        """Generate a canonical signature for deduplication."""
        sorted_ops = ",".join(sorted(self.operations))
        return f"{self.abstraction_level.value}:{sorted_ops}"


@dataclasses.dataclass
class PairResult:
    """Result of applying a hypothesis to a specific input-output pair."""
    pair_id: str
    match: bool
    predicted: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclasses.dataclass
class ConsistencyReport:
    """Report detailing how well a hypothesis performed across task pairs."""
    hypothesis_id: str
    pairs_matched: int
    pairs_total: int
    results: list[PairResult]
    has_program_results: bool  # True if tested via program execution
    
    @property
    def score(self) -> float:
        """Calculate the overall consistency score."""
        if self.pairs_total == 0:
            return 0.0
        return self.pairs_matched / self.pairs_total
        
    @property
    def is_perfect(self) -> bool:
        """Check if the hypothesis perfectly solved all pairs."""
        if self.pairs_total == 0:
            return False
        return self.pairs_matched == self.pairs_total


@dataclasses.dataclass
class HypothesisBank:
    """Container for managing a beam of hypotheses through pruning rounds."""
    hypotheses: list[Hypothesis] = dataclasses.field(default_factory=list)
    pruned: list[Hypothesis] = dataclasses.field(default_factory=list)
    generation: int = 0
    
    def add(self, h: Hypothesis) -> None:
        """Add a hypothesis to the bank."""
        self.hypotheses.append(h)
        
    def kill(self, h_id: str) -> None:
        """Move a hypothesis to the pruned list and lower its confidence."""
        for i, h in enumerate(self.hypotheses):
            if h.id == h_id:
                h.confidence = 0.0
                self.pruned.append(self.hypotheses.pop(i))
                break
                
    def alive(self) -> list[Hypothesis]:
        """Get all currently alive hypotheses."""
        return [h for h in self.hypotheses if h.is_alive()]
        
    def best(self) -> Optional[Hypothesis]:
        """Get the hypothesis with the highest consistency score."""
        alive_hypotheses = self.alive()
        if not alive_hypotheses:
            return None
        return max(alive_hypotheses, key=lambda h: h.consistency_score())
        
    def diversity_score(self) -> float:
        """
        Calculate a diversity metric based on pairwise Jaccard distance 
        of operation sets among alive hypotheses.
        """
        alive_hypotheses = self.alive()
        n = len(alive_hypotheses)
        if n < 2:
            return 0.0
            
        total_distance = 0.0
        comparisons = 0
        
        for i in range(n):
            set1 = set(alive_hypotheses[i].operations)
            for j in range(i + 1, n):
                set2 = set(alive_hypotheses[j].operations)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                
                if union > 0:
                    jaccard_similarity = intersection / union
                    jaccard_distance = 1.0 - jaccard_similarity
                    total_distance += jaccard_distance
                comparisons += 1
                
        if comparisons == 0:
            return 0.0
            
        return total_distance / comparisons
