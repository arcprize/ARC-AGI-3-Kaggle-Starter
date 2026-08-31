from __future__ import annotations
import dataclasses
import subprocess
import sys
import json
import time
import textwrap
import numpy as np
from typing import Optional, Any
from pathlib import Path

from .hypothesis import Hypothesis, ConsistencyReport, PairResult

@dataclasses.dataclass
class TrainingPair:
    id: str
    input_grid: np.ndarray
    output_grid: np.ndarray

@dataclasses.dataclass
class SimulationResult:
    hypothesis_id: str
    pair_id: str
    match: bool
    predicted_grid: Optional[np.ndarray]
    error: Optional[str]
    execution_time_ms: float

class DeterministicSimulator:
    """Executes hypothesis programs in a sandboxed subprocess."""
    
    def __init__(self, timeout_seconds: float = 5.0, max_output_size: int = 1_000_000):
        self.timeout = timeout_seconds
        self.max_output_size = max_output_size
    
    def execute(self, hypothesis_program: str, input_grid: np.ndarray) -> np.ndarray:
        """Run a hypothesis program on an input grid, return predicted output.
        
        The program receives 'input_grid' as a numpy array and must set
        'output_grid' as a numpy array.
        """
        script = self._build_sandbox_script(hypothesis_program, input_grid)
        
        try:
            # Run the script in a subprocess
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Subprocess execution failed: {result.stderr}")
            
            if len(result.stdout) > self.max_output_size:
                raise RuntimeError(f"Output size exceeded limit ({len(result.stdout)} > {self.max_output_size})")
                
            output_data = json.loads(result.stdout)
            if "error" in output_data:
                raise RuntimeError(f"Sandbox error: {output_data['error']}")
            if "output_grid" not in output_data:
                raise RuntimeError("Script did not produce 'output_grid' in JSON output")
                
            return np.array(output_data["output_grid"])
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Execution timed out after {self.timeout} seconds")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON output: {e}")
        except Exception as e:
            raise RuntimeError(f"Execution error: {e}")
    
    def verify_against_pairs(
        self, hypothesis: Any, pairs: list[TrainingPair]
    ) -> list[SimulationResult]:
        """Test a hypothesis against all training pairs."""
        results = []
        
        for pair in pairs:
            start_time = time.time()
            match = False
            predicted_grid = None
            error = None
            
            if not getattr(hypothesis, 'program', None):
                error = "Hypothesis has no executable program"
            else:
                try:
                    predicted_grid = self.execute(hypothesis.program, pair.input_grid)
                    if np.array_equal(predicted_grid, pair.output_grid):
                        match = True
                except Exception as e:
                    error = str(e)
            
            execution_time_ms = (time.time() - start_time) * 1000.0
            
            results.append(SimulationResult(
                hypothesis_id=hypothesis.id,
                pair_id=pair.id,
                match=match,
                predicted_grid=predicted_grid,
                error=error,
                execution_time_ms=execution_time_ms
            ))
            
        return results
    
    def _build_sandbox_script(self, program: str, input_grid: np.ndarray) -> str:
        """Build a self-contained Python script for sandboxed execution."""
        input_list = input_grid.tolist()
        input_json = json.dumps(input_list)
        
        script = textwrap.dedent(f"""\
            import json
            import numpy as np

            def execute_sandbox():
                try:
                    # Load input grid
                    input_data = json.loads('''{input_json}''')
                    input_grid = np.array(input_data)
                    
                    # Initialize output_grid as None
                    output_grid = None
                    
                    # Local environment for execution
                    local_env = {{
                        'input_grid': input_grid,
                        'output_grid': output_grid,
                        'np': np
                    }}
                    
                    # Execute the hypothesis program
                    program_code = '''{program}'''
                    exec(program_code, {{'__builtins__': {{'range': range, 'len': len, 'list': list, 'int': int, 'float': float, 'bool': bool, 'enumerate': enumerate, 'zip': zip, 'min': min, 'max': max, 'sum': sum, 'abs': abs, 'all': all, 'any': any, 'round': round, 'dict': dict, 'set': set, 'tuple': tuple, 'print': print}}}}, local_env)
                    
                    # Retrieve output_grid
                    if 'output_grid' in local_env and local_env['output_grid'] is not None:
                        out = local_env['output_grid']
                        if isinstance(out, np.ndarray):
                            out_list = out.tolist()
                        else:
                            out_list = list(out)
                        print(json.dumps({{'output_grid': out_list}}))
                    else:
                        print(json.dumps({{'error': 'output_grid not set'}}))
                except Exception as e:
                    print(json.dumps({{'error': str(e)}}))

            if __name__ == '__main__':
                execute_sandbox()
        """)
        return script

class ConsistencyChecker:
    """Checks whether a hypothesis is consistent with training pairs
    using the DeterministicSimulator for executable hypotheses."""
    
    def __init__(self, simulator: DeterministicSimulator):
        self.simulator = simulator
    
    def check(self, hypothesis: Any, pairs: list[TrainingPair]) -> ConsistencyReport:
        """Returns a ConsistencyReport for the hypothesis."""
        if not getattr(hypothesis, 'program', None):
            return ConsistencyReport(
                hypothesis_id=hypothesis.id,
                has_program_results=False,
                pair_results=[],
                overall_match=False
            )
            
        simulation_results = self.simulator.verify_against_pairs(hypothesis, pairs)
        
        pair_results = []
        overall_match = True
        
        for res in simulation_results:
            pair_results.append(PairResult(
                pair_id=res.pair_id,
                match=res.match,
                predicted_grid=res.predicted_grid,
                error=res.error,
                execution_time_ms=res.execution_time_ms
            ))
            if not res.match:
                overall_match = False
                
        return ConsistencyReport(
            hypothesis_id=hypothesis.id,
            has_program_results=True,
            pair_results=pair_results,
            overall_match=overall_match
        )
