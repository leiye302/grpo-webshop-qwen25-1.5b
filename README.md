# WebShop GRPO — Qwen2.5-1.5B, 4×A100 80GB

独立的 **Qwen2.5-1.5B-Instruct 纯 GRPO** 复现仓库，直接使用 [SDAR 公开代码](https://github.com/ZJU-REAL/SDAR) 的 WebShop GRPO baseline。无 SFT、OPSD、Teacher、skill、RETHINK、Analyzer 或付费 API。另一个模型尺寸独立发布，不在本仓库顺序训练。

## 固定配方

| 项目 | 设置 |
| --- | --- |
| 初始化 | Qwen/Qwen2.5-1.5B-Instruct，固定公开 revision |
| 上游 | SDAR `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3` |
| 原版参照 | `examples/grpo_trainer/run_webshop_8gpu.sh`，四卡资源适配；另固定验证任务 |
| GPU | 4×A100 **80GB**；4 FSDP ranks，2个 TP=2 生成副本 |
| 训练 | 150 updates；全局16组×8轨迹=128 episodes/update；LR=1e-6 |
| PPO | 全局决策行 mini-batch64；micro8/GPU；原生GRPO/clip/token-mean |
| 正则 | 原生reference KL=.01 low_var_kl；entropy=.001；非法动作惩罚=.1 |
| 交互 | max15步；history2；prompt4096；response512；训练temperature1 |
| 验证 | 初始及每5步；固定同一组128任务；temperature=.4，do_sample=true |
| 保存 | 每20步及最终150；完整保存、SHA校验后才轮换，保留最近两份 |

**环境保留 SDAR 默认：1000商品的小环境、synthetic goals。** 不是全量商品环境，也不是 canonical human500 测试集。原版每次验证重新抽128个索引；本仓库将原版第一次验证的128个环境任务绑定写成 [固定清单](configs/fixed_validation128.json)，此后所有checkpoint、两个模型尺寸、重启续跑和独立评测都使用同一组。仅改变验证选题，不改变训练采样、loss、prompt、动作解析或奖励。详细边界见 [复现合同](docs/REPRODUCTION.md)。

两种尺寸只改变 backbone/对应模型权重与入口标识，其余公共配置相同。四卡下全局 mini64→每rank16，micro8→2次原生梯度累积；不增加全局采样量，也不承诺八卡/四卡逐位数值相同。

## 1. 干净服务器安装

需要 Linux x86-64、Python **3.10**、OpenJDK **11**、兼容 CUDA12.8 的驱动和四张已获配 A100 80GB。默认32个可用逻辑CPU；原生256个环境actor会各自加载商品、NLP及Java搜索，建议主机内存256–512GiB，优先512GiB。建议至少250GiB空闲盘，必须容纳旧两份断点加第三份新写入的峰值。此为容量预算，不是实测最低值。

以下示例使用公开 Conda；可用等价的 micromamba。环境和资产都由公开来源准备，无维护者本机文件或私人服务器依赖。

```bash
git clone https://github.com/leiye302/grpo-webshop-qwen25-1.5b.git
cd grpo-webshop-qwen25-1.5b
conda create -n webshop-grpo python=3.10 openjdk=11 -c conda-forge -y
conda activate webshop-grpo
export JAVA_HOME="$CONDA_PREFIX"
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip check

cp .env.example .env
# 仅编辑自己获配的卡和CPU数。默认四卡0,1,2,3、CPU32。
set -a
source .env
set +a

# 下载固定版本的公开商品/指令文件、NLP模型，运行原生Lucene索引程序。
python src/assets.py --download --nlp --index --verify
python src/env_smoke.py
python src/setup.py --download-model 1.5b

python src/setup.py --verify
python scripts/check_release.py
python -m pytest tests -q
bash scripts/train.sh --dry-run
```

不运行原版脚本里的私人代理、环境路径、整机清理或 wandb key。FlashAttention需要匹配wheel或CUDA Toolkit/nvcc编译。不要再运行上游未约束的全量依赖脚本覆盖此环境。所有下载均固定公开版本/SHA；文本环境不需要下载图像特征，也不需要启动 Chrome 或外部购物网站。

`env_smoke.py` 在CPU真实执行搜索、点选商品、配置选项和购买，用已知目标验证解析/奖励链路。它读取目标答案仅用于环境单测，**不参与模型训练/评测，不代表Actor成功率**。没有显卡也可准备完整环境。

## 2. 训练、续跑和导出

```bash
# 本入口固定1.5b；用tmux或调度器保持进程。
bash scripts/train.sh

# 意外退出后，确认旧控制器/worker都退出，再从最新完整断点续跑。
bash scripts/train.sh --resume

# 最终150步完成后导出HF模型。
python src/run.py export --size 1.5b
python src/report.py
```

也可使用 `bash scripts/run_both.sh` 自动训练→导出→生成报告；这个兼容文件名在本独立仓库**只运行1.5b**。无需外层 torchrun。脚本不停止其他作业、不连接已有Ray集群、不自动抢卡。拒绝覆盖旧run；同一run使用控制器排他锁。

第一次GPU运行仍需核实4rank、2个TP2副本、真实rollout、有限loss/grad、实际参数更新、首次完整存盘及GPU恢复。不要仅凭CPU测试认定150步训练已验收。

## 3. 验证口径与结果

原版从每个worker已打乱 goals 的前500个索引中抽128个；固定seed只决定整个抽题序列，不会使每轮抽到同一组。本版将seed1000的**第一次抽题**预先写入固定清单，绑定worker编号、worker seed和目标索引。每次只在验证reset期间复现这一抽样并核对清单，之后恢复该验证环境原有私有RNG；不操作训练环境或全局模型RNG。

同一组任务在update0、5、10直到150重复评测；1.5B和3B使用字节一致的清单。首次记录 `fixed_validation_identity.json`；以后含续跑均核对初始任务观察的SHA，不一致立即报错，不悄悄换题。每次summary记录清单SHA、`selection_sha256`及初始观察SHA，跨仓库比较时应三者一致。

训练仍从索引500起随机采样。不同worker的goals有各自seed，因此**固定验证任务不等于已证明与训练任务全局语义隔离**，不能宣称是canonical human500/严格zero-shot评测。验证仍为每任务一次完整自由rollout，最多15步、history2、response512，温度0.4；没有改成贪心，所以**固定题目不保证每次模型输出或分数完全相同**。

- 成功率：原生wrapper仅当终局WebShop原始score恰为1时给reward10、won=true；其余reward0。
- 连续score：另保留原生 `task_score` 的0–1值和0–100显示值；不把部分匹配当成功。
- 原生宽松动作解析不改。完整think/action协议、reasoning出现率仅为遥测，不作训练门禁。

输出在 `runs/grpo_1.5b/`：`metrics.jsonl`、`evaluation/update_000`及其后每5步的summary/逐决策token文本和ID、原生验证日志、20步断点、最终150步和HF导出。初始base成绩来自update0，不预填成绩、不从测试曲线挑最优checkpoint冒充最终结果。

```bash
# 独立评测导出模型：使用同一固定128条任务，使用新输出目录。
bash scripts/eval.sh --model "$PWD/runs/grpo_1.5b/hf150" \
  --run-dir "$PWD/runs/grpo_1.5b_final_eval"
python src/report.py
```

## 4. 安全保存与公开验证

新断点先在同盘暂存目录写齐四rank的model/optimizer/extra状态、data.pt、config/tokenizer；检验归档CRC、大小和逐文件SHA，原子发布后才更新latest并清理旧断点。保存和恢复都按磁盘完整断点扫描，只保留最近两份；失败、损坏、异绑定及软链接产物留作证据，不当作有效断点也不自动删除。HF导出不轮换。必须预留第三份写入空间。

原生optimizer、scheduler/RNG及dataloader保留；原版不快照WebShop/Ray环境内部随机状态，所以续跑不承诺与未中断采样逐位相同。四rank断点不能直接用八rank续跑或反过来。

固定验证清单及实现也纳入续跑绑定；旧随机验证版的run不能静默迁到本版继续。保留旧run和其原代码；新实验使用新run目录。验证固定并不意味着训练环境的续跑采样被改为固定。

GitHub Actions在干净Linux/Python3.10/Java11中公开获取上游和数据，检查源码/配方/隐私、原生GRPO实际CPU梯度、存盘故障与CPU恢复、真实WebShop搜索/购买/奖励、入口配置。**这不是四卡GPU/vLLM/NCCL或完整150步训练验证**；没有捏造成绩。完整来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
