"""Reproducible llama.cpp benchmark + small task evaluation. Python stdlib only."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_env():
    env = os.environ.copy()
    cuda = env.get("CUDA_PATH")
    if cuda:
        env["PATH"] = os.pathsep.join([str(Path(cuda) / "bin" / "x64"),
                                       str(Path(cuda) / "bin"), env.get("PATH", "")])
    return env


def capture(cmd, timeout=30, cwd=None):
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=runtime_env(), timeout=timeout, cwd=cwd)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def check_answer(case, text):
    """Strict task checks: formatting failures are intentionally preserved."""
    rule = case["rule"]
    if rule == "exact":
        return text.strip() == case["expected"]
    if rule == "json":
        try:
            actual = json.loads(text)
            return actual == case["expected"]
        except (ValueError, TypeError):
            return False
    raise ValueError(f"unsupported rule: {rule}")


def diagnose_answer(case, text, finish_reason):
    """Return a grading outcome and a machine-readable failure reason."""
    if finish_reason != "stop":
        return {"passed": False, "failure_reason": "unexpected_finish_reason"}

    if case["rule"] == "exact":
        if text.strip() == case["expected"]:
            return {"passed": True, "failure_reason": None}
        else:
            return {"passed": False, "failure_reason": "exact_mismatch"}

    passed = check_answer(case, text)
    return {"passed": passed, "failure_reason": None if passed else "answer_mismatch"}


def request_json(url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    # These endpoints are loopback; do not send them through system HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def wait_ready(proc, base, timeout=180):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see server.log")
        try:
            if request_json(base + "/health", timeout=2).get("status") == "ok":
                return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, PermissionError) or getattr(reason, "winerror", None) == 10013:
                raise PermissionError("loopback access denied by execution environment") from exc
        time.sleep(0.25)
    raise TimeoutError("server startup timed out")


def evaluate(server, model, cases, out, threads):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    # Some Windows llama.cpp tools use narrow argv for file names: keep -m ASCII
    # and use the Unicode-aware process working directory to reach the model.
    cmd = [str(server), "-m", model.name, "-ngl", "99", "-c", "2048",
           "-t", str(threads), "--host", "127.0.0.1", "--port", str(port),
           "--reasoning-budget", "0", "-np", "1"]
    rows = []
    with (out / "server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=model.parent, stdout=log, stderr=subprocess.STDOUT, env=runtime_env(),
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            wait_ready(proc, base)
            for case in cases:
                payload = {"messages": [{"role": "user", "content": case["prompt"]}],
                           "temperature": 0, "seed": 1234, "max_tokens": 128,
                           "stream": False, "cache_prompt": False,
                           "chat_template_kwargs": {"enable_thinking": False}}
                start = time.perf_counter()
                response = request_json(base + "/v1/chat/completions", payload)
                wall = time.perf_counter() - start
                choice = response["choices"][0]
                answer = choice["message"].get("content") or ""
                diagnosis = diagnose_answer(case, answer, choice["finish_reason"])
                passed = diagnosis["passed"]
                rows.append({"id": case["id"], "request": payload, "response": response,
                             "answer": answer, "expected": case["expected"], "rule": case["rule"],
                             "passed": passed, "failure_reason": diagnosis["failure_reason"],
                             "wall_s": wall})
                save(out / "quality.json", rows)
                print(f"quality {case['id']}: {'PASS' if passed else 'FAIL'}", flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    return rows


def benchmark(bench, model, device, args, out):
    cmd = [str(bench), "-m", model.name, "-t", str(args.threads),
           "-p", str(args.prompt), "-n", str(args.generate), "-r", str(args.repetitions),
           "-o", "json", "-ngl", "0" if device == "cpu" else "99",
           "-dev", "none" if device == "cpu" else "CUDA0"]
    if device == "cpu":
        cmd += ["-nkvo", "1"]
    # Save portable invocation; file identities are recorded separately by SHA256.
    portable = [bench.name if x == str(bench) else model.name if x == str(model) else x for x in cmd]
    result = capture(cmd, timeout=args.timeout, cwd=model.parent)
    save(out / f"bench-{device}-process.json", {"command": portable, **result})
    if result["exit_code"]:
        raise RuntimeError(f"{device} benchmark failed: exit {result['exit_code']}")
    rows = json.loads(result["stdout"])
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("expected separate prompt-processing and generation benchmark rows")
    for row in rows:
        if row["avg_ts"] <= 0:
            raise ValueError("invalid benchmark throughput")
    save(out / f"bench-{device}.json", rows)
    print(f"benchmark {device}: completed", flush=True)
    return rows


def report_cell(value):
    """Keep recorded answers from becoming Markdown table or HTML markup."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return html.escape(text).replace("|", "&#124;").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def report(out, manifest, benches, quality):
    lines = ["# 实验报告", "", f"- 时间（UTC）：{manifest['created_utc']}",
             f"- 模型：{manifest['model']['name']}", f"- SHA256：`{manifest['model']['sha256']}`",
             f"- 状态：{manifest['status']}", "", "## 性能", "",
             "llama-bench 局部吞吐；不包括模型加载、分词、采样和 HTTP 请求的完整耗时。",
             "pp = prompt processing（输入处理）；tg = token generation（逐 token 生成）。",
             "以下 ± 为重复实验标准差，不是置信区间。", "",
             "| 设备配置 | 测试 | token/s（均值 ± 标准差） |", "|---|---|---|"]
    for device, rows in benches.items():
        for row in rows:
            kind = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
            lines.append(f"| {device} | {kind} | {row['avg_ts']:.2f} ± {row['stddev_ts']:.2f} |")
    lines += ["", "## 固定任务检查", "",
              f"通过 {sum(r['passed'] for r in quality)}/{len(quality)}。检查在 GPU 配置下运行。",
              "小型自建题集只检查指定任务和格式，不能代表通用能力，也不是困惑度评测。",
              "", "| 题号 | 通过 | HTTP 完整响应耗时（秒） |", "|---|---|---|"]
    for row in quality:
        lines.append(f"| {row['id']} | {row['passed']} | {row['wall_s']:.3f} |")
    if quality:
        lines += ["", f"完整响应耗时中位数：{statistics.median(r['wall_s'] for r in quality):.3f} 秒。",
                  "不同题目的输入输出长度不同，此处仅作运行观察，不能用于公平的模型速度比较。"]
    failures = [row for row in quality if not row["passed"]]
    lines += ["", "## 失败明细", ""]
    if failures:
        lines += ["失败类型描述可直接检查的现象，不用于归因量化或模型内部机制。",
                  "旧记录没有保存的字段显示为未记录，不根据当前题集猜测历史期望值。", "",
                  "| 题号 | 期望回答 | 实际回答 | 失败原因 |", "|---|---|---|---|"]
        for row in failures:
            cells = (row["id"], row.get("expected", "未记录（旧格式）"), row["answer"],
                     row.get("failure_reason") or "未记录（旧格式）")
            lines.append("| " + " | ".join(report_cell(cell) for cell in cells) + " |")
    else:
        lines.append("本次已记录任务中没有失败项；不代表所有可能任务都正确。")
    lines += ["", "## 条件与局限", "", "- 单机、串行请求；质量评测禁用提示词缓存，关闭思考模式。",
              "- 同一个模型的 CPU/GPU 对比保持模型哈希和基准参数一致；CPU 为指定线程数，未调优。",
              "- 尚未控制电源、温度、GPU 频率和后台负载；不将结果推广为硬件极限。",
              "- 未测 TTFT、TPOT、并发和峰值显存；不据此宣称生产部署能力。",
              "- 未开展同源量化对比，不能宣称量化加速收益；未修改推理内核。",
              "- server.log 和完整请求响应保留，用于核对卸载层数与失败案例。", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory; never overwrites existing runs")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "gpu"], default=["gpu"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--generate", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    for name in ("threads", "prompt", "generate", "repetitions", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    suffix = ".exe" if os.name == "nt" else ""
    bench = (args.bin_dir / ("llama-bench" + suffix)).resolve()
    server = (args.bin_dir / ("llama-server" + suffix)).resolve()
    model = args.model.resolve()
    for path in (bench, server, model):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases_path = ROOT / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "model": {"name": model.name, "bytes": model.stat().st_size, "sha256": sha256(model)},
                "binaries": {p.name: sha256(p) for p in (bench, server)},
                "code_sha256": sha256(Path(__file__)), "cases_sha256": sha256(cases_path),
                "python": platform.python_version(), "platform": platform.platform(),
                "config": {k: v for k, v in vars(args).items() if k not in ("bin_dir", "model", "out")}}
    benches, quality = {}, []
    save(out / "manifest.json", manifest)
    try:
        save(out / "version.json", capture([str(server), "--version"]))
        save(out / "gpu-before.json", capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv"]))
        quality = evaluate(server, model, cases, out, args.threads)
        for device in dict.fromkeys(args.devices):
            benches[device] = benchmark(bench, model, device, args, out)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(manifest["error"], file=sys.stderr)
    finally:
        save(out / "manifest.json", manifest)
        if not quality and (out / "quality.json").exists():
            quality = json.loads((out / "quality.json").read_text(encoding="utf-8"))
        report(out, manifest, benches, quality)
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
