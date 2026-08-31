"""
HyPrune evaluation harness.
"""
from __future__ import annotations
import dataclasses
import json
import numpy as np
import time
from typing import Optional, Any
from ..tasks.loader import ArcTask

@dataclasses.dataclass
class TaskResult:
    task_id: str
    correct: bool
    predicted_output: Optional[np.ndarray]
    expected_output: Optional[np.ndarray]
    num_hypotheses_generated: int
    num_pruning_rounds: int
    num_surviving_hypotheses: int
    time_seconds: float
    final_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "correct": self.correct,
            "predicted_output": self.predicted_output.tolist() if self.predicted_output is not None else None,
            "expected_output": self.expected_output.tolist() if self.expected_output is not None else None,
            "num_hypotheses_generated": self.num_hypotheses_generated,
            "num_pruning_rounds": self.num_pruning_rounds,
            "num_surviving_hypotheses": self.num_surviving_hypotheses,
            "time_seconds": self.time_seconds,
            "final_confidence": self.final_confidence
        }

@dataclasses.dataclass
class BenchmarkResult:
    results: list[TaskResult]
    
    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)
        
    @property
    def mean_hypotheses(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.num_hypotheses_generated for r in self.results) / len(self.results)
        
    @property
    def mean_pruning_rounds(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.num_pruning_rounds for r in self.results) / len(self.results)
        
    @property
    def mean_time(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.time_seconds for r in self.results) / len(self.results)
    
    def summary(self) -> str:
        """Human-readable summary table."""
        lines = [
            "Evaluation Benchmark Summary",
            "==========================",
            f"Total Tasks: {len(self.results)}",
            f"Accuracy:    {self.accuracy * 100:.1f}%",
            f"Mean Hypotheses: {self.mean_hypotheses:.1f}",
            f"Mean Pruning Rounds: {self.mean_pruning_rounds:.1f}",
            f"Mean Time/Task: {self.mean_time:.2f}s",
            "",
            "Failed Tasks:",
            "------------"
        ]
        
        failed = [r for r in self.results if not r.correct]
        if not failed:
            lines.append("None! Perfect score.")
        else:
            for r in failed:
                lines.append(f"- {r.task_id} (conf: {r.final_confidence:.2f}, time: {r.time_seconds:.1f}s)")
                
        return "\n".join(lines)
    
    def save(self, path: str) -> None:
        """Save results to JSON."""
        data = {
            "summary": {
                "total_tasks": len(self.results),
                "accuracy": self.accuracy,
                "mean_hypotheses": self.mean_hypotheses,
                "mean_pruning_rounds": self.mean_pruning_rounds,
                "mean_time": self.mean_time
            },
            "results": [r.to_dict() for r in self.results]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


class EvaluationHarness:
    """Runs the HyPrune system on a set of ARC tasks and measures performance."""
    
    def __init__(self, system: Any):
        self.system = system  # The HyPrune pipeline
    
    def evaluate_task(self, task: ArcTask) -> TaskResult:
        """Evaluate a single task."""
        start_time = time.time()
        
        try:
            if not task.test_pairs:
                raise ValueError(f"Task {task.id} has no test pairs.")
                
            prediction, metadata = self.system.predict(task)
            expected = task.test_pairs[0].output_grid
            
            correct = False
            if expected.size > 0 and prediction.shape == expected.shape:
                correct = np.array_equal(prediction, expected)
                
            elapsed = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                correct=correct,
                predicted_output=prediction,
                expected_output=expected,
                num_hypotheses_generated=metadata.get('num_hypotheses_generated', 0),
                num_pruning_rounds=metadata.get('num_pruning_rounds', 0),
                num_surviving_hypotheses=metadata.get('num_surviving_hypotheses', 0),
                time_seconds=elapsed,
                final_confidence=metadata.get('final_confidence', 0.0)
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return TaskResult(
                task_id=task.id,
                correct=False,
                predicted_output=None,
                expected_output=task.test_pairs[0].output_grid if task.test_pairs else None,
                num_hypotheses_generated=0,
                num_pruning_rounds=0,
                num_surviving_hypotheses=0,
                time_seconds=elapsed,
                final_confidence=0.0
            )

    def evaluate_tasks(self, tasks: list[ArcTask], max_workers: int = 1) -> BenchmarkResult:
        """Evaluate multiple tasks. Could be parallelized in future."""
        results = []
        for task in tasks:
            result = self.evaluate_task(task)
            results.append(result)
            
        return BenchmarkResult(results=results)
