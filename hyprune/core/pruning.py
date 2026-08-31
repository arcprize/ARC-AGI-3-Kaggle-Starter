from __future__ import annotations
from typing import Any, Tuple, List
from .hypothesis import Hypothesis, ConsistencyReport

class PruningEngine:
    """Eliminates hypotheses based on simulation results."""
    
    def __init__(self, prune_threshold: float = 0.3):
        self.prune_threshold = prune_threshold
    
    def prune(
        self, hypotheses: List[Hypothesis], reports: List[ConsistencyReport]
    ) -> Tuple[List[Hypothesis], List[Hypothesis]]:
        """Returns (survivors, killed).
        
        Hard prune: if hypothesis has a program and it fails ANY training pair, kill it.
        Soft prune: if confidence drops below threshold, kill it.
        """
        survivors = []
        killed = []
        
        report_map = {r.hypothesis_id: r for r in reports}
        
        for hyp in hypotheses:
            report = report_map.get(hyp.id)
            
            # Hard prune: has program but failed a training pair
            if report and report.has_program_results and not report.overall_match:
                killed.append(hyp)
                continue
                
            # Soft prune: confidence below threshold
            if hyp.confidence < self.prune_threshold:
                killed.append(hyp)
                continue
                
            survivors.append(hyp)
            
        return survivors, killed
    
    def diversity_filter(self, survivors: List[Hypothesis], max_k: int = 8) -> List[Hypothesis]:
        """Keep at most max_k hypotheses, maximizing diversity.
        
        Cluster by operation type, keep highest confidence from each cluster.
        If still too many, keep top-k by confidence.
        """
        if len(survivors) <= max_k:
            return survivors
            
        # Cluster by operation type or fallback to string class representation
        clusters: dict[Any, List[Hypothesis]] = {}
        for hyp in survivors:
            key = tuple(getattr(hyp, 'operations', []))
            if not key:
                key = getattr(hyp, 'type', 'unknown')
            
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(hyp)
            
        diverse_survivors = []
        
        # Take highest confidence from each cluster
        for key in clusters:
            clusters[key].sort(key=lambda h: h.confidence, reverse=True)
            diverse_survivors.append(clusters[key][0])
            
        # If we have more than max_k diverse, sort by confidence and take max_k
        if len(diverse_survivors) > max_k:
            diverse_survivors.sort(key=lambda h: h.confidence, reverse=True)
            return diverse_survivors[:max_k]
            
        # If we have less than max_k, fill with remaining highest confidence
        diverse_survivors_set = {h.id for h in diverse_survivors}
        remaining = [h for h in survivors if h.id not in diverse_survivors_set]
        remaining.sort(key=lambda h: h.confidence, reverse=True)
        
        needed = max_k - len(diverse_survivors)
        diverse_survivors.extend(remaining[:needed])
        
        return diverse_survivors
    
    def rank(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Rank hypotheses by confidence, breaking ties by diversity."""
        # Simple ranking by confidence. For stable sort and simple diversity tie-breaker,
        # sort by len of operations (simpler hypotheses preferred on tie), then confidence.
        return sorted(
            hypotheses,
            key=lambda h: (h.confidence, -len(getattr(h, 'operations', []))),
            reverse=True
        )
