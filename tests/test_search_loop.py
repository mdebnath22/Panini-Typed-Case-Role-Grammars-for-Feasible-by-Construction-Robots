"""Smoke test for M3 search loop. Not a strict correctness assertion on the
fitness numbers (those are stochastic physics) — asserts the machinery runs,
produces a monotonically non-decreasing evolutionary best-so-far curve, and
that the benchmark's paired statistic is computable."""

from experiments.m_search_loop import run, benchmark, evaluate_genome, random_genome
import numpy as np


def test_evo_history_nondecreasing():
    r = run(budget=20, seed=0, verbose=False)
    hist = r["evo"]["history"]
    assert all(hist[i] <= hist[i + 1] for i in range(len(hist) - 1)), \
        "evolutionary best-so-far must be monotonically non-decreasing"


def test_gate_rejects_without_physics_call():
    # a genome with an absurdly large mass budget requirement should be
    # rejectable by the gate before any physics call — just check the gate
    # path doesn't crash and returns the sentinel penalty
    rng = np.random.default_rng(0)
    from sulba.feasibility import FeasibilitySpec
    g = random_genome(rng)
    fit, info = evaluate_genome(g, use_gate=True,
                                spec=FeasibilitySpec(mass_budget=0.001))
    assert fit == -1e3
    assert info.get("error") == "infeasible"


def test_benchmark_runs():
    b = benchmark(n_seeds=2, budget=10, verbose=False)
    assert len(b["evo_fits"]) == 2
    assert len(b["rnd_fits"]) == 2
    assert b["wins"] in (0, 1, 2)


if __name__ == "__main__":
    test_evo_history_nondecreasing()
    print("  ok  test_evo_history_nondecreasing")
    test_gate_rejects_without_physics_call()
    print("  ok  test_gate_rejects_without_physics_call")
    test_benchmark_runs()
    print("  ok  test_benchmark_runs")
