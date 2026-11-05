"""End-to-end demo: federated training with 1 straggler and DP enabled.

Expected output: loss decreases across rounds, the straggler client is
dropped when sampled, and training completes inside the DP budget.
"""

from __future__ import annotations

from flp.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", "--rounds", "8"]))
