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
