import json

from tools import compile_project_catalogue as catalogue


def test_cli_repeats_real_lowering_failure_after_catalogue_summary(monkeypatch, tmp_path, capsys):
    manifest = {"units": [{
        "qualified_name": "manager", "status": "failed",
        "error_type": "ConcordanceRefusal", "error": "producer 73 lacks physical rank",
        "failure": "units/manager/failure.json",
    }]}
    monkeypatch.setattr(catalogue, "compile_project_product", lambda *args, **kwargs: manifest)
    assert catalogue.main(["--source", str(tmp_path / "work.py"), "--output", str(tmp_path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    report = json.loads(lines[-1])
    assert "catalogue complete:" in lines[-2]
    assert report["error"] == "producer 73 lacks physical rank"
    assert report["error_type"] == "ConcordanceRefusal"
    assert report["phase"] == "lowering"


def test_cli_native_build_failure_is_visible_and_fails_without_cases(monkeypatch, tmp_path, capsys):
    failure = tmp_path / "native-failure.json"
    failure.write_text(json.dumps({"error_type": "RuntimeError", "error": "mixed index Add refused"}))
    manifest = {"units": [{
        "qualified_name": "manager", "status": "complete",
        "native_status": "failed", "native_failure": failure.name,
    }]}
    monkeypatch.setattr(catalogue, "compile_project_product", lambda *args, **kwargs: manifest)
    assert catalogue.main(["--source", str(tmp_path / "work.py"), "--output", str(tmp_path), "--emit-native"]) == 1
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["error"] == "mixed index Add refused"
    assert report["phase"] == "native"
    assert report["failure"] == str(failure)
