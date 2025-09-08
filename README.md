# auto\_bidsify · LLM 主导 / 无 LLM 也可运行

> 把**任意无序文件夹或 .zip** 自动整理为**最小可用的 BIDS 目录**。支持 **LLM 自动分配 subject**；即便没有 LLM，也能用“Smart stub”按 `--nsubjects` 自动分配。

---

## ✨ Core Features

* **无正则提取 subject**：

  * **在线（LLM）**：由 ChatGPT 基于 `all_files` 自动给出 `assignment_rules`（subject → 匹配列表）。
  * **离线（Smart stub）**：不调用 LLM，也会依据 `--nsubjects` 与文件清单生成 `assignment_rules` 与 `mappings`。
* **原格式保留**：`.nirs/.snirf/.mat/.csv/.tsv/.txt/...` 等原样保留，按 BIDS 命名落位；其余全部进 `derivatives/`。
* **三件套自动处理**：`README.md / participants.tsv / dataset_description.json`（大小写不敏感）直接复制到 BIDS 根，输出统一为小写名。
* **大小写无关匹配**：`assignment_rules` 和 `mappings` 的通配符匹配均不区分大小写。
* **结果可追溯**：生成 `conversion_log.json`（映射日志）与 `BIDSMap_with_manifest.yaml`（内含最终文件清单与目录树）。

---

## 🧰 Dependencies \& Installation

* **Python**：3.9+（建议 3.10/3.11）
* **Python package**：

  * `pyyaml`（YAML 读写）
  * `openai`（可选，调用 ChatGPT）

```bash
pip install pyyaml openai
```

* **（可选）bids-validator**：用于结果校验

```bash
npm install -g bids-validator
```

* **（可选）OpenAI Key**（若使用 LLM）

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> 网络受限环境可设置代理：`export HTTPS_PROXY=http://host:port`

---

## 📁 代码结构

```
auto_bidsify/
├─ cli.py                # 命令行入口：scan → map → exec → validate（或 full 一键）
├─ ingest.py             # 接收目录/zip；zip 解压到 out/_staging/extracted
├─ scan.py               # 扫描：扩展名统计、抽样、收集 all_files；写入 user_hints.n_subjects
├─ llm_openai.py         # 规则生成（LLM）；失败或 --llm stub 时走“智能 stub”
├─ rules.py              # 规则 I/O 与小型模板引擎（{subject}/{ext}/...）
├─ execmap.py            # 执行：按 assignment_rules 赋 subject，套模板落位；其余进 derivatives/
├─ validate.py           # bids-validator 包装；无则返回桩报告
└─ utils.py              # 工具函数：建目录/拷贝/彩色输出
```

---

## 🚀 Quick Start

### 方案一：一键全流程

```bash
python3 cli.py full \
  --in /path/to/dataverse_files.zip \
  --out /path/to/bids_out \
  --llm openai --model gpt-5-mini \
  --nsubjects 18
```

> 如果无法调用 LLM，把 `--llm openai` 换成 `--llm stub` 也能跑通（离线“智能 stub”）。

### 方案二：分步执行

```bash
# 1) 扫描：生成 evidence_bundle.json（含 all_files + user_hints.n_subjects）
python3 cli.py scan --in /path/to/input --out /path/to/bids_out --nsubjects 18

# 2) 规则：由 LLM 或“智能 stub”生成 BIDSMap.yaml（含 assignment_rules + mappings）
python3 cli.py map --evidence /path/to/bids_out/evidence_bundle.json \
                   --out /path/to/bids_out --llm openai --model gpt-5-mini

# 3) 执行：按规则拷贝/落位；未匹配的全部进 derivatives/
python3 cli.py exec --in /path/to/input --out /path/to/bids_out \
                    --rules /path/to/bids_out/BIDSMap.yaml

# 4) 校验（可选）
python3 cli.py validate --out /path/to/bids_out
```

---

## 🔑 Command Line Parameters

* `--in`：输入目录或 `.zip`（zip 将解压到 `out/_staging/extracted/`）。
* `--out`：BIDS 输出目录。
* `--llm {openai|stub}`：使用 ChatGPT 生成规则，或使用**智能 stub**离线生成规则。
* `--model`：如 `gpt-5-mini`、`gpt-5` 等。
* `--nsubjects`：受试者总数（**强烈建议提供**），写入 `evidence_bundle.json.user_hints.n_subjects`。

---

## 🧭 Pipeline

1. **ingest**（`ingest.py`）

   * `.zip`：解压至 `out/_staging/extracted/`；原始 zip 复制到 `derivatives/orig/archives/` 便于溯源。
   * 目录：直接作为输入根。

2. **scan**（`scan.py`）

   * 识别多种 NIRS 格式：`.nirs/.snirf/.mat/.csv/.tsv/.txt/.xlsx`（可扩展）。
   * 输出 `evidence_bundle.json`：

     * `all_files`：**全量相对路径**（后续 subject 分配的关键）
     * `user_hints.n_subjects`：来自 `--nsubjects` 的提示
     * `counts/samples/trio_found`：扩展名统计/代表样本/三件套是否存在

3. **map**（`llm_openai.py`）

   * **在线（LLM）**：将 `evidence_bundle.json` 发给 ChatGPT（Responses API），产出 YAML：

     * `subjects.total / labels`
     * `assignment_rules`（每个 subject 的匹配列表：glob 或显式路径）
     * `mappings`（多模态/多后缀的匹配与 `bids_out` 模板，使用 `{subject}`、`{ext}`）
     * `policy`（仅保留必需内容在 BIDS 根；其余进 `derivatives/`）
     * `questions`（待确认项）
   * **离线（智能 stub）**：无需 LLM 也会生成可用 YAML：

     * 从 `all_files` 中筛 NIRS 候选（后缀/关键词），按字母序取前 `N` 个（`N = --nsubjects`），**一对一分配**给 `sub-01..sub-N`；
     * 生成 `assignment_rules + mappings`，使 BIDS 主体至少覆盖 `min(N, 候选数)` 个文件。

4. **exec**（`execmap.py`）

   * 用 `assignment_rules`（大小写无关 `fnmatch`）确定每个文件的 `subject`。
   * 模板上下文可用：`{subject}`、`{ext}`、`{filename}`、`{parentname}`、`{relpath}`。
   * 将匹配到的文件复制进 `sub-{subject}/<modality>/...`；未匹配的文件（且非三件套）全部路由到 `derivatives/`。
   * 生成：

     * `conversion_log.json`（输入→输出映射）
     * `BIDSMap_with_manifest.yaml`（在规则末尾附 `manifest`：文件总数、清单、ASCII 树）

5. **validate**（`validate.py`）

   * 有 `bids-validator`：输出真实校验报告；
   * 无：返回桩报告（不阻塞流程）。

---

## 🧩 YAML Structure Sample

```yaml
bids_version: "1.10.0"
policy:
  keep_user_trio: true
  route_residuals_to_derivatives: true
  derivatives_buckets:
    docs: ["**/*.pdf","**/*.html","**/*.htm","**/*.md"]
    intermediate: ["**/*.log","**/*.tmp","**/*.bak"]
    _misc: ["**/*"]
entities:
  subject: { pad: 2, sanitize: true }
  session: { omit_if_single: true }
  run: { omit_if_single: true }
subjects:
  total: 18
  labels: ["01","02", ...]
assignment_rules:
  - subject: "01"
    match: ["**/BZZ014*.nirs", "**/sub01/*"]
  - subject: "02"
    match: ["**/BZZ003*.nirs"]
  # ...
mappings:
  - modality: nirs
    match:
      - pattern: "**/*.nirs"
        bids_out: "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"
      - pattern: "**/*.snirf"
        bids_out: "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"
      - pattern: "**/*.mat"
        bids_out: "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"
      - pattern: "**/*.csv"
        bids_out: "sub-{subject}/nirs/sub-{subject}_task-rest_nirs{ext}"
    sidecar:
      TaskName: "rest"
      SamplingFrequency: "unknown"
participants:
  respect_user_file: true
  fallback_generate_if_missing: false
questions:
  - "Confirm task labels if not rest."
```

> 说明：
>
> * **不使用正则**；subject 由 `assignment_rules` 的命中直接决定。
> * `bids_out` 使用 `{subject}` 与 `{ext}`（原扩展名保留）。
> * 需要为某 subject 追加更多文件？直接在该 subject 的 `match` 列表中新增 glob 或具体文件路径即可。

---

## 🔎 调试与常见问题

* **`[plan] 0 files`**：

  * `BIDSMap.yaml` 可能没有 `assignment_rules` 或 `mappings`；
  * 如走 LLM 仍产空，先用 `--llm stub` 验证流程；
  * 检查 `--nsubjects` 是否传入；`all_files` 是否被写入 evidence。

* **文件都进了 `derivatives/_misc/`**：

  * `assignment_rules` 没命中该文件；或 `mappings` 没覆盖该后缀。
  * 解决：在 `assignment_rules` 增加匹配；在 `mappings.match` 增加该后缀的规则。

* **大小写差异**：

  * 执行器匹配是大小写无关的；
  * 三件套输出名固定为小写：`readme.md / participants.tsv / dataset_description.json`。

* **同一 subject 多文件**：

  * 在同一 `subject` 的 `match` 中加多条；或使用更宽的 glob（如 `**/sub01/*`）。

* **需要严格 SNIRF**：

  * 当前示例为“保留原格式”；可在执行阶段接入转换器（如 mne-bids 等），把 `.nirs/.mat/.csv` 转为 `.snirf` 后再落位。

---

## 🔧 可扩展点

* **识别更多格式**：在 `scan.py` 中扩展 `NIRS_EXT` 与 `NIRS_NAME_HINTS`。
* **更“聪明”的离线分配**：将 stub 的“顺序分配”升级为“按父目录/前缀聚类后分配”。
* **真实数据转换**：在 `execmap.py` 的拷贝处接入转换工具链（DICOM→NIfTI、EEG→BIDS、fNIRS→SNIRF）。
* **隐私过滤**：扫描阶段剔除潜在 PHI 文本或进行脱敏。

---

## 📦 Outputs

```
bids_out/
├─ readme.md
├─ participants.tsv
├─ dataset_description.json
├─ sub-01/
│  └─ nirs/
│     └─ sub-01_task-rest_nirs.nirs
├─ sub-02/
│  └─ nirs/
│     └─ sub-02_task-rest_nirs.csv
├─ derivatives/
│  ├─ orig/
│  │  └─ archives/
│  │     └─ dataverse_files.zip
│  ├─ docs/
│  └─ _misc/
├─ BIDSMap.yaml
├─ BIDSMap_with_manifest.yaml   # manifest: files + tree_text
└─ conversion_log.json
```



