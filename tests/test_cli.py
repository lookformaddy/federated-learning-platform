"""CLI tests."""

from __future__ import annotations

from flp.cli import build_parser, main


def test_cli_train_no_dp(capsys):
    rc = main(["train", "--clients", "4", "--rounds", "2", "--no-dp", "--seed", "7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "final loss" in out
    assert "promoted model version" in out


def test_cli_demo(capsys):
    rc = main(["demo", "--rounds", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "final loss" in out


def test_cli_train_json(capsys):
    import json

    rc = main(["train", "--clients", "4", "--rounds", "2", "--no-dp",
               "--seed", "7", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["rounds"]) == 2


def test_cli_rejects_bad_strategy():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--strategy", "bogus"])
