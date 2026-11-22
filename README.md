本项目面向内存与显存受限的本地大语言模型部署，基于 llama.cpp，探索量化等方法在资源占用、回答质量和推理速度之间的权衡，减少部署对高规格硬件的依赖。

当前已完成 Qwen3-1.7B Q8_0 的本地推理评测流程，包含固定任务检查、同模型 CPU/GPU 性能对照、原始结果保存与报告生成。后续将接入同源高精度模型与不同量化版本，评估资源节省及质量、速度变化。

面向 llama.cpp 的小模型推理评测工具：运行固定任务检查，在同一个 GGUF 模型上比较 CPU/GPU 吞吐，保留原始证据并生成报告。

## 已验证的第一版

2026-11-22，在 RTX 4060 Laptop 8GB / i9-14900HX / Windows 上完成实测。Qwen3-1.7B Q8_0，同一个模型文件，4线程、pp128、tg64、各重复3次：

| 配置 | pp128（token/s） | tg64（token/s） |
|---|---:|---:|
| GPU | 7058.47 ± 641.78 | 113.14 ± 2.14 |
| CPU | 108.92 ± 2.65 | 29.81 ± 0.30 |

固定任务检查通过7/8，失败回答照实保留；上述±为重复实验标准差。一次实验、每项3次重复，不代表稳定硬件极限，也不是量化加速比。报告和原始记录见[实测示例](examples/4060-q8-2026-11-22/REPORT.md)。本地7项自动测试通过；GitHub Actions配置已提供，远程执行状态以Actions页面为准。

**当前范围：单模型评测与同精度 CPU/GPU 对照。真实模型量化、困惑度、并发优化尚未包含。**

## 为什么做这个项目

模型“跑起来”之后，需要回答：结果是否符合任务要求、测量覆盖哪些阶段、别人能否重复实验？本项目用固定输入、模型和程序哈希、原始输出及失败记录来回答这些问题。

## 环境

- Python 3.10+，工具代码只使用标准库。
- 自行构建的 llama.cpp，提供 `llama-bench` 与 `llama-server`。
- CUDA 构建、NVIDIA GPU、可用的 `nvidia-smi`；当前质量检查使用 GPU 配置。
- 用户自行准备 GGUF 模型。本仓库不分发模型或 llama.cpp 源码和二进制。
- Windows 如果 CUDA DLL 不在 PATH，请设置正确的 `CUDA_PATH`；Linux 正常配置 CUDA 运行环境。

参考：[llama.cpp 构建](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)、[官方基准工具](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)、[服务接口](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)。参数以实际锁定版本的帮助为准。

## 运行

以下在本仓库目录执行，将路径替换为本机路径：

```powershell
python lab.py --bin-dir /path/to/llama.cpp/bin --model /path/to/model.gguf --devices gpu cpu --out results/run-001
```

默认：4 个 CPU 线程、pp128、tg64、各重复5次；质量题集8题，temperature=0、seed=1234、最多128个输出token，禁用提示词缓存和思考模式。启动的服务仅监听127.0.0.1，退出时关闭；不会操作其他已运行的服务。

复现实测示例时追加`--repetitions 3`。示例命令中的`/path/to/...`是占位路径，替换为本机实际路径。Windows工具对中文路径的兼容修复：通过`cwd`进入模型目录，再用模型文件名传入`-m`；不需要复制大模型文件。

每次使用新的输出目录，旧实验不覆盖。检查进程退出码，并读取 `manifest.json` 的 `status`。`completed` 表示流程完成；**不代表全部质量题通过**。

```powershell
python -m unittest discover -s tests -v
```

## 输出

| 文件 | 内容 |
|---|---|
| manifest.json | 模型、程序、代码及题集SHA256；环境与参数；完成/失败状态 |
| quality.json | 完整请求、回复、任务判定和HTTP完整响应时间 |
| server.log | 加载、GPU卸载、缓存与服务日志 |
| bench-*.json | llama-bench结构化结果 |
| bench-*-process.json | 可移植命令、退出码、原始标准输出与错误输出 |
| REPORT.md | 根据本次原始结果生成的报告 |

使用相同模型哈希和相同测试参数比较 CPU/GPU。CPU使用`-dev none -ngl 0 -nkvo 1`，GPU使用`-dev CUDA0 -ngl 99`。这同时改变执行设备与内存放置，不能解释成单个算子的加速。

## 指标解释

- **pp**：prompt processing，输入处理；**tg**：token generation，逐token生成。
- llama-bench是局部吞吐，报告均值和重复实验标准差，不包含加载、分词、采样和HTTP完整开销。
- 质量检查的HTTP耗时为非流式完整响应时间，**不是首token时延（TTFT）**；题目长短不同，不能据此公平比较速度。
- 严格检查格式，保留失败回答。8题只用于冒烟和回归，不能当成通用准确率。
- SHA256锁定文件；本机历史构建自报commit unknown时如实保留，通过二进制哈希辅助定位，不伪造版本号。

## 复现限制

电源、温度、GPU频率和后台负载仍可能影响结果。尚未报告峰值显存、置信区间、多并发结果或生产部署能力。不同规模模型和不同量化一起变化时，不能将速度变化归因于量化。

现有模型来源：[Qwen3-1.7B官方GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF)。本机Q8_0文件SHA256为`061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`。复现时以实际下载文件哈希为准。

## 后续工作（未完成）

1. 接入同一高精度来源的Q8/Q4、困惑度与质量案例，核对台式机原始日志。
2. 流式TTFT/TPOT、并发和显存采样。
3. 用分析工具验证瓶颈假设，再选择一处源码修改。

## 贡献说明

初版工具代码、测试和文档由AI协助编写，初版运行由AI执行。学习者的解释、修改与实验设计另行记录，不将AI执行写为独立完成。个人简历、联系方式、私人聊天、录音、虚拟环境及模型文件均不属于本仓库发布内容。

工具代码采用MIT许可证。模型与llama.cpp各自遵循其原有许可证，本仓库许可证不替代它们。
