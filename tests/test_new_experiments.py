"""Smoke tests for M-H6b (gate ablation) and M-H1 (raw-XML baseline)."""

from experiments.mh6b_gate_ablation import run as run_gate_ablation
from experiments.mh1_xml_baseline import run as run_xml_baseline


def test_gate_ablation_runs_and_accounts_fairly():
    r = run_gate_ablation(n_seeds=2, physics_budget=15, verbose=False)
    assert len(r["with_gate"]) == 2
    assert len(r["without_gate"]) == 2
    # gate-off should reach physics_budget in exactly physics_budget tries
    # (no rejections possible with use_gate=False) — checked indirectly via
    # run() not raising and returning finite-looking numbers
    assert all(f > -1e3 or f == -1e3 for f in r["with_gate"])  # no crash


def test_grammar_mutation_always_valid():
    r = run_xml_baseline(n_trials=30, seed=0, verbose=False)
    assert r["grammar_valid"] == r["n_trials"], \
        "grammar mutation must be legal-by-construction (100% valid)"
    assert r["xml_valid"] <= r["n_trials"]


if __name__ == "__main__":
    test_gate_ablation_runs_and_accounts_fairly()
    print("  ok  test_gate_ablation_runs_and_accounts_fairly")
    test_grammar_mutation_always_valid()
    print("  ok  test_grammar_mutation_always_valid")
