import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lab", ROOT / "lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_saves_diagnosis_for_recorded_responses(self):
        import json
        cases = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        records = json.loads((ROOT / "examples/4060-q8-2026-11-22/quality.json").read_text(encoding="utf-8"))
        proc = Mock()
        proc.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(lab.socket, "socket") as socket_factory, \
                 patch.object(lab.subprocess, "Popen", return_value=proc), \
                 patch.object(lab, "wait_ready"), \
                 patch.object(lab, "request_json", side_effect=[r["response"] for r in records]):
                socket_factory.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
                rows = lab.evaluate(out / "fake-server", out / "fake.gguf", cases, out, 4)
            saved = json.loads((out / "quality.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, rows)
            self.assertEqual([r["passed"] for r in rows], [r["passed"] for r in records])
            follow = next(r for r in saved if r["id"] == "follow")
            self.assertEqual(follow["expected"], "完成")
            self.assertEqual(follow["answer"], "好的。")
            self.assertEqual(follow["failure_reason"], "exact_mismatch")
            self.assertIsNone(saved[0]["failure_reason"])
            proc.terminate.assert_called_once()

    def test_report_shows_failure_evidence_and_escapes_answers(self):
        import json
        manifest = json.loads((ROOT / "examples/4060-q8-2026-11-22/manifest.json").read_text(encoding="utf-8"))
        rows = [{"id": "follow", "passed": False, "expected": "完成", "answer": "好的。|<x>\n第二行",
                 "failure_reason": "exact_mismatch", "wall_s": 0.1}]
        with tempfile.TemporaryDirectory() as tmp:
            lab.report(Path(tmp), manifest, {}, rows)
            text = (Path(tmp) / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("通过 0/1", text)
            self.assertIn("| follow | 完成 | 好的。&#124;&lt;x&gt;<br>第二行 | exact_mismatch |", text)

    def test_report_reads_legacy_records_without_inventing_diagnosis(self):
        import json
        source = ROOT / "examples/4060-q8-2026-11-22"
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        rows = json.loads((source / "quality.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            lab.report(Path(tmp), manifest, {}, rows)
            text = (Path(tmp) / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("通过 7/8", text)
            self.assertIn("| follow | 未记录（旧格式） | 好的。 | 未记录（旧格式） |", text)

    def test_diagnosis_preserves_real_follow_failure(self):
        import json
        cases = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        case = next(row for row in cases if row["id"] == "follow")
        rows = json.loads((ROOT / "examples/4060-q8-2026-11-22/quality.json").read_text(encoding="utf-8"))
        record = next(row for row in rows if row["id"] == "follow")
        result = lab.diagnose_answer(case, record["answer"], record["response"]["choices"][0]["finish_reason"])
        self.assertEqual(result, {"passed": False, "failure_reason": "exact_mismatch"})

    def test_diagnosis_accepts_exact_answer_with_whitespace(self):
        case = {"rule": "exact", "expected": "完成"}
        result = lab.diagnose_answer(case, " 完成\n", "stop")
        self.assertEqual(result, {"passed": True, "failure_reason": None})

    def test_diagnosis_rejects_truncated_correct_text(self):
        case = {"rule": "exact", "expected": "完成"}
        result = lab.diagnose_answer(case, "完成", "length")
        self.assertEqual(result, {"passed": False, "failure_reason": "unexpected_finish_reason"})

    def test_diagnosis_keeps_json_grading(self):
        case = {"rule": "json", "expected": {"age": 20}}
        self.assertEqual(lab.diagnose_answer(case, '{"age":20}', "stop"),
                         {"passed": True, "failure_reason": None})
        self.assertEqual(lab.diagnose_answer(case, '{"age":"20"}', "stop"),
                         {"passed": False, "failure_reason": "answer_mismatch"})

    def test_exact_rejects_explanation(self):
        case = {"rule": "exact", "expected": "42"}
        self.assertTrue(lab.check_answer(case, "42\n"))
        self.assertFalse(lab.check_answer(case, "答案是42"))

    def test_json_checks_structure_and_values(self):
        case = {"rule": "json", "expected": {"name": "小林", "age": 20}}
        self.assertTrue(lab.check_answer(case, '{"age":20,"name":"小林"}'))
        for answer in ('```json\n{}\n```', '{"name":"小林","age":"20"}', '{}'):
            self.assertFalse(lab.check_answer(case, answer))

    def test_invalid_arguments_do_not_create_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result"
            result = subprocess.run([sys.executable, str(ROOT / "lab.py"),
                                     "--bin-dir", tmp, "--model", str(Path(tmp) / "missing.gguf"),
                                     "--out", str(out)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            suffix = ".exe" if lab.os.name == "nt" else ""
            for name in ("llama-bench" + suffix, "llama-server" + suffix, "model.gguf"):
                (Path(tmp) / name).touch()
            out = Path(tmp) / "run"
            out.mkdir()
            evidence = out / "evidence.txt"
            evidence.write_text("original")
            with self.assertRaises(FileExistsError):
                lab.main(["--bin-dir", tmp, "--model", str(Path(tmp) / "model.gguf"), "--out", str(out)])
            self.assertEqual(evidence.read_text(), "original")

    def test_server_exit_is_not_mistaken_for_ready(self):
        proc = Mock()
        proc.poll.return_value = 1
        proc.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "exited"):
            lab.wait_ready(proc, "http://127.0.0.1:1")

    def test_loopback_permission_failure_is_immediate(self):
        proc = Mock()
        proc.poll.return_value = None
        with patch.object(lab, "request_json", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                lab.wait_ready(proc, "http://127.0.0.1:1")

    def test_benchmark_uses_model_directory_not_unicode_argv(self):
        import argparse
        import json
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "中文目录" / "model.gguf"
            args = argparse.Namespace(threads=4, prompt=128, generate=64, repetitions=3, timeout=5)
            rows = [{"avg_ts": 10}, {"avg_ts": 20}]
            with patch.object(lab, "capture", return_value={"exit_code": 0, "stdout": json.dumps(rows), "stderr": ""}) as run:
                lab.benchmark(Path(tmp) / "llama-bench", model, "cpu", args, Path(tmp))
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index("-m") + 1], "model.gguf")
            self.assertEqual(run.call_args.kwargs["cwd"], model.parent)


if __name__ == "__main__":
    unittest.main()
