#!/usr/bin/env python3
"""
Profile the generate_answer function to identify performance bottlenecks.

This script provides multiple profiling approaches:
1. cProfile for overall function-level profiling
2. Custom timing decorators for component-level analysis (Generator, Retriever, Evaluator, Extractor)
3. Detailed breakdown of MCTS search phases

Usage:
    # Basic profiling with a sample question
    python -m scripts.profile_generate_answer --question "Who is the president of the United States?"

    # Profile with config overrides
    python -m scripts.profile_generate_answer --config configs/infer/base.yaml --question "..."

    # Profile with full cProfile output
    python -m scripts.profile_generate_answer --question "..." --full-profile

    # Profile with specific number of rollouts
    python -m scripts.profile_generate_answer --question "..." --num-rollouts 3
"""

import argparse
import asyncio
import cProfile
import functools
import io
import os
import pstats
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    """Stores timing statistics for a component."""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def record(self, elapsed: float):
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)

    def __str__(self) -> str:
        if self.call_count == 0:
            return f"{self.name}: No calls"
        return (
            f"{self.name}:\n"
            f"  Calls: {self.call_count}\n"
            f"  Total: {self.total_time:.3f}s\n"
            f"  Avg:   {self.avg_time:.3f}s\n"
            f"  Min:   {self.min_time:.3f}s\n"
            f"  Max:   {self.max_time:.3f}s"
        )


class ProfilerContext:
    """Global context for collecting profiling data."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.stats: Dict[str, TimingStats] = {}
        self.call_stack: List[str] = []
        self.phase_times: Dict[str, float] = defaultdict(float)
        self.enabled = True
    
    def reset(self):
        self.stats.clear()
        self.call_stack.clear()
        self.phase_times.clear()
    
    def get_or_create_stats(self, name: str) -> TimingStats:
        if name not in self.stats:
            self.stats[name] = TimingStats(name=name)
        return self.stats[name]
    
    def record_phase(self, phase: str, elapsed: float):
        self.phase_times[phase] += elapsed
    
    def print_summary(self):
        """Print a formatted summary of all timing stats."""
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)
        
        if not self.stats:
            print("No profiling data collected.")
            return
        
        # Sort by total time descending
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda s: s.total_time,
            reverse=True
        )
        
        total_measured = sum(s.total_time for s in sorted_stats)
        
        print(f"\n{'Component':<40} {'Calls':>8} {'Total(s)':>10} {'Avg(s)':>10} {'%':>8}")
        print("-" * 70)
        
        for stat in sorted_stats:
            pct = (stat.total_time / total_measured * 100) if total_measured > 0 else 0
            print(
                f"{stat.name:<40} {stat.call_count:>8} "
                f"{stat.total_time:>10.3f} {stat.avg_time:>10.3f} {pct:>7.1f}%"
            )
        
        print("-" * 70)
        print(f"{'Total measured time:':<40} {' ':>8} {total_measured:>10.3f}s")
        
        # Phase breakdown if available
        if self.phase_times:
            print("\n" + "-" * 70)
            print("MCTS PHASE BREAKDOWN")
            print("-" * 70)
            for phase, t in sorted(self.phase_times.items(), key=lambda x: -x[1]):
                print(f"  {phase:<35}: {t:.3f}s")


# Global profiler context
profiler = ProfilerContext()


@contextmanager
def timed_section(name: str):
    """Context manager for timing a code section."""
    if not profiler.enabled:
        yield
        return
    
    start = time.perf_counter()
    profiler.call_stack.append(name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        profiler.call_stack.pop()
        profiler.get_or_create_stats(name).record(elapsed)


T = TypeVar('T')


def timed(name: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to time function execution."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        label = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with timed_section(label):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            with timed_section(label):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Monkey-patching for agent profiling
# ---------------------------------------------------------------------------

def patch_agents_for_profiling():
    """
    Monkey-patch agent classes to add profiling instrumentation.
    This allows us to profile without modifying the original source code.
    """
    from state_aware_rag.agents.agents import (
        GeneratorAgent,
        RetrievalAgent,
        ExtractorAgent,
        EvaluatorAgent,
        BaseAgent,
    )
    
    # Patch BaseAgent.execute to track all agent executions
    original_execute = BaseAgent.execute
    
    async def profiled_execute(self, instance_id, parameters, **kwargs):
        agent_name = self.__class__.__name__
        fn_name = parameters.get("generate_fn") or parameters.get("evaluate_fn") or "execute"
        label = f"{agent_name}.{fn_name}"
        
        with timed_section(label):
            result = await original_execute(self, instance_id, parameters, **kwargs)
        return result
    
    BaseAgent.execute = profiled_execute
    
    # Patch the run methods for more granular profiling
    original_generator_run = GeneratorAgent.run
    original_retriever_run = RetrievalAgent.run
    original_extractor_run = ExtractorAgent.run
    original_evaluator_run = EvaluatorAgent.run
    
    def make_profiled_run(original_run, agent_type: str):
        @functools.wraps(original_run)
        def profiled_run(self, instance_id, parameters, **kwargs):
            fn_name = parameters.get("generate_fn") or parameters.get("evaluate_fn") or "run"
            label = f"{agent_type}.run.{fn_name}"
            with timed_section(label):
                return original_run(self, instance_id, parameters, **kwargs)
        return profiled_run
    
    GeneratorAgent.run = make_profiled_run(original_generator_run, "GeneratorAgent")
    RetrievalAgent.run = make_profiled_run(original_retriever_run, "RetrievalAgent")
    ExtractorAgent.run = make_profiled_run(original_extractor_run, "ExtractorAgent")
    EvaluatorAgent.run = make_profiled_run(original_evaluator_run, "EvaluatorAgent")
    
    print("[Profiler] Agent classes patched for profiling.")


def patch_mcts_for_profiling():
    """Patch MCTS methods to profile search phases."""
    from state_aware_rag.planners.MCTS.backbone import MCTS
    
    original_select = MCTS._select
    original_expand = MCTS._expand
    original_simulate = MCTS._simulate
    original_backpropagate = MCTS._backpropagate
    original_do_rollout = MCTS.do_rollout
    
    def profiled_select(self, node):
        with timed_section("MCTS._select"):
            return original_select(self, node)
    
    def profiled_expand(self, node, rollout_id=None):
        with timed_section("MCTS._expand"):
            return original_expand(self, node, rollout_id)
    
    def profiled_simulate(self, node, rollout_id=None):
        with timed_section("MCTS._simulate"):
            return original_simulate(self, node, rollout_id)
    
    def profiled_backpropagate(self, path, reward):
        with timed_section("MCTS._backpropagate"):
            return original_backpropagate(self, path, reward)
    
    def profiled_do_rollout(self, node, rollout_id=None):
        with timed_section(f"MCTS.do_rollout[{rollout_id}]"):
            return original_do_rollout(self, node, rollout_id)
    
    MCTS._select = profiled_select
    MCTS._expand = profiled_expand
    MCTS._simulate = profiled_simulate
    MCTS._backpropagate = profiled_backpropagate
    MCTS.do_rollout = profiled_do_rollout
    
    print("[Profiler] MCTS class patched for profiling.")


def patch_reasoning_node_for_profiling():
    """Patch ReasoningNode methods to profile node operations."""
    from state_aware_rag.planners.reasoning_node import ReasoningNode
    
    original_find_children = ReasoningNode.find_children
    original_reward = ReasoningNode.reward
    
    def profiled_find_children(self, rollout_id=None):
        node_type = self.node_type.name if hasattr(self.node_type, 'name') else str(self.node_type)
        with timed_section(f"ReasoningNode.find_children[{node_type}]"):
            return original_find_children(self, rollout_id)
    
    def profiled_reward(self):
        with timed_section("ReasoningNode.reward"):
            return original_reward(self)
    
    ReasoningNode.find_children = profiled_find_children
    ReasoningNode.reward = profiled_reward
    
    print("[Profiler] ReasoningNode class patched for profiling.")


# ---------------------------------------------------------------------------
# Main profiling logic
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)


def run_with_cprofile(func: Callable, *args, **kwargs):
    """Run a function with cProfile and return results + profile stats."""
    pr = cProfile.Profile()
    pr.enable()
    try:
        result = func(*args, **kwargs)
    finally:
        pr.disable()
    
    # Create stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    
    return result, ps, s


def print_cprofile_summary(ps: pstats.Stats, stream: io.StringIO, top_n: int = 30):
    """Print a summary of cProfile results."""
    print("\n" + "=" * 70)
    print("cProfile RESULTS (Top {} by cumulative time)".format(top_n))
    print("=" * 70)
    
    # Get fresh output
    s = io.StringIO()
    ps.stream = s
    ps.print_stats(top_n)
    print(s.getvalue())


def profile_generate_answer(
    question: str,
    config_path: str,
    mode: str = "mcts",
    num_rollouts: int = 3,
    max_depth: int = 5,
    top_k: int = 3,
    full_profile: bool = False,
    verbose: bool = False,
):
    """
    Profile the generate_answer function with the given configuration.
    """
    from state_aware_rag.agents.agents import (
        GeneratorAgent,
        RetrievalAgent,
        ExtractorAgent,
        EvaluatorAgent,
    )
    from inference import generate_answer
    
    # Reset profiler
    profiler.reset()
    
    # Load config
    config = load_config(config_path)
    
    # Initialize agents
    print("\n[Profiler] Initializing agents...")
    with timed_section("Agent initialization"):
        generator = GeneratorAgent(config=config["agents"]["generator"])
        retriever = RetrievalAgent(config=config["agents"]["retriever"])
        extractor = ExtractorAgent(config=config["agents"]["extractor"])
        evaluator = EvaluatorAgent(config=config["agents"]["evaluator"])
    
    # Prepare search config
    search_config = {
        "max_depth": max_depth,
        "num_rollouts": num_rollouts,
        "top_k": top_k,
        "exploration_weight": 1.0,
        "use_golden_answer": False,
        "save_tree": False,
        "verbose": verbose,
    }
    
    print(f"\n[Profiler] Profiling generate_answer with:")
    print(f"  Question: {question[:100]}{'...' if len(question) > 100 else ''}")
    print(f"  Mode: {mode}")
    print(f"  Rollouts: {num_rollouts}")
    print(f"  Max Depth: {max_depth}")
    print(f"  Top K: {top_k}")
    
    # Run with cProfile
    def run_generate():
        return generate_answer(
            question=question,
            generator=generator,
            evaluator=evaluator,
            extractor=extractor,
            retriever=retriever,
            question_id="profile_test",
            golden_answer=None,
            mode=mode,
            search=search_config,
        )
    
    print("\n[Profiler] Starting profiled execution...")
    start_time = time.perf_counter()
    
    result, ps, stream = run_with_cprofile(run_generate)
    
    total_time = time.perf_counter() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("EXECUTION RESULT")
    print("=" * 70)
    print(f"Total wall-clock time: {total_time:.3f}s")
    print(f"Predicted answer: {result.get('pred', 'N/A')[:200]}")
    
    # Print custom profiling summary
    profiler.print_summary()
    
    # Print cProfile summary
    if full_profile:
        print_cprofile_summary(ps, stream, top_n=50)
    else:
        print_cprofile_summary(ps, stream, top_n=20)
    
    # Save profile to file for later analysis
    profile_output_path = PROJECT_ROOT / "profiling_results"
    profile_output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    profile_file = profile_output_path / f"profile_{timestamp}.prof"
    ps.dump_stats(str(profile_file))
    print(f"\n[Profiler] Full profile saved to: {profile_file}")
    print("[Profiler] To view interactively, run:")
    print(f"  snakeviz {profile_file}")
    print("  or")
    print(f"  python -m pstats {profile_file}")
    
    return result, ps


def main():
    parser = argparse.ArgumentParser(
        description="Profile the generate_answer function to identify bottlenecks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default="What is the capital of France and who is the current president?",
        help="Question to use for profiling"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(PROJECT_ROOT / "configs/infer/base.yaml"),
        help="Path to inference config YAML"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["mcts", "cot"],
        default="mcts",
        help="Search mode (mcts or cot)"
    )
    parser.add_argument(
        "--num-rollouts", "-n",
        type=int,
        default=3,
        help="Number of MCTS rollouts (default: 3 for faster profiling)"
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=5,
        help="Maximum search depth"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Top-k for retrieval"
    )
    parser.add_argument(
        "--full-profile", "-f",
        action="store_true",
        help="Show full cProfile output (top 50 instead of 20)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output during search"
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Disable monkey-patching (only use cProfile)"
    )
    
    args = parser.parse_args()
    
    # Apply patches before importing the main modules
    if not args.no_patch:
        patch_agents_for_profiling()
        patch_mcts_for_profiling()
        patch_reasoning_node_for_profiling()
    
    # Run profiling
    try:
        profile_generate_answer(
            question=args.question,
            config_path=args.config,
            mode=args.mode,
            num_rollouts=args.num_rollouts,
            max_depth=args.max_depth,
            top_k=args.top_k,
            full_profile=args.full_profile,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n[Profiler] Interrupted by user.")
        profiler.print_summary()
        sys.exit(1)
    except Exception as e:
        print(f"\n[Profiler] Error during profiling: {e}")
        profiler.print_summary()
        raise


if __name__ == "__main__":
    main()
