"""CLI for the federated learning platform."""

from __future__ import annotations

import argparse
import json

from flp.config import FLConfig
from flp.orchestrator import run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flp", description="Federated learning platform (simulation)")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run a federated training job")
    train.add_argument("--clients", type=int, default=6)
    train.add_argument("--rounds", type=int, default=8)
    train.add_argument("--participation", type=float, default=0.8)
    train.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox", "trimmed_mean"])
    train.add_argument("--no-dp", action="store_true", help="disable differential privacy")
    train.add_argument("--epsilon-budget", type=float, default=50.0)
    train.add_argument("--stragglers", default="", help="comma-separated client ids that straggle")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--json", action="store_true", help="print the full result as JSON")

    demo = sub.add_parser("demo", help="short demo run (1 straggler, DP on)")
    demo.add_argument("--rounds", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        config = FLConfig(
            n_rounds=args.rounds,
            straggler_client_ids=(5,),
            dp_enabled=True,
        )
    else:
        stragglers = tuple(int(x) for x in args.stragglers.split(",") if x.strip()) if args.stragglers else ()
        config = FLConfig(
            n_clients=args.clients,
            n_rounds=args.rounds,
            participation_fraction=args.participation,
            strategy=args.strategy,
            dp_enabled=not args.no_dp,
            dp_epsilon_budget=args.epsilon_budget,
            straggler_client_ids=stragglers,
            seed=args.seed,
        )
    result = run_training(config)
    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    print(f"Federated training: {config.strategy}, {config.n_clients} clients, "
          f"{config.n_rounds} rounds, DP={'on' if config.dp_enabled else 'off'}")
    print(f"initial loss {result.initial_loss:.4f} -> final loss {result.final_loss:.4f}, "
          f"accuracy {result.final_accuracy:.4f}")
    for r in result.rounds:
        print(f"  round {r.round_no}: loss {r.loss:.4f} acc {r.accuracy:.4f} "
              f"participants {r.participants} dropped {r.dropped} "
              f"eps {r.dp_epsilon_spent:.2f} up {r.bytes_up}B")
    if result.halted_reason:
        print(f"halted: {result.halted_reason}")
    print(f"promoted model version: {result.registry.promoted_version} "
          f"(best acc {result.registry.best_accuracy:.4f})")
    print(f"total comm: {result.comm.total} bytes "
          f"(up {result.comm.total_up}, down {result.comm.total_down})")
    if result.fairness:
        print(f"fairness: per-client acc spread {result.fairness.spread:.4f} "
              f"(min {result.fairness.min_accuracy:.4f}, max {result.fairness.max_accuracy:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
