# auto\_bidsify · 全新版本 README（LLM 主导的 subject 分配 | 无正则）

> 目标：把**任意无序文件夹或 .zip** 自动整理为**最小可用的 BIDS 树**。
>
> 亮点：
>
> * **无需你写正则**从文件名抽取 subject；改为 **LLM 基于文件清单自动分配**（支持 `--nsubjects` 指定受试者总数）。
> * **原格式保留**（`.nirs/.snirf/.mat/.csv/.tsv/.txt/...`），仅命名与落位遵循 BIDS；其它全部路由到 `derivatives/`。
> * **离线兜底**：即使没法调用 LLM，也会用“智能 stub”根据 `--nsubjects` 自动生成 `assignment_rules` 与 `mappings`，确保能跑通。

---

## 🧰 用到的工具

* **Python 3.9+**（建议 3.10/3.11）
* **PyYAML**：读写 YAML
* **OpenAI Python SDK**（`responses` API）：让 ChatGPT 产出 BIDS 规则（可选）
* **（可选）bids-validator**：验证 BIDS 结果

安装：

```bash
pip install pyyaml openai
# 可选（推荐在容器/CI 中安装）
npm install -g bids-validator
```

如果使用 LLM：

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

---

## 📁 代码结构

```
auto_bidsify/
├─ cli.py                # 命令行入口：scan → map → exec → validate（或 full 一键）
├─ ingest.py             # 接受目录或 .zip；zip 解压到 out/_staging
├─ scan.py               # 扫描：统计扩展名、抽样、收集 all_files；写 user_hints.n_subjects
├─ llm_openai.py         # 调用 ChatGPT；失败则“智能 stub”自动生成 assignment_rules+mappings
├─ rules.py              # 规则 I/O & 迷你模板引擎（{subject}/{ext}/…）
├─ execmap.py            # 执行：按 assignment_rules 赋 subject，套模板落位；其余进 derivatives/
├─ validate.py           # bids-validator 包装；无则返回桩报告
└─ utils.py              # 通用：建目录/拷贝/彩色输出
```

---

## 🚀 快速开始

### 一键全流程（推荐）

```bash
python3 cli.py full \
  --in /path/to/dataverse_files.zip \
  --out /path/to/bids_out \
  --llm openai --model gpt-5-mini \
  --nsubjects 18
```

> 无法联网或没 Key？把 `--llm openai` 换为 `--llm stub`，也能跑通（按顺序把前 18 个 NIRS 文件分配给 sub-01..sub-18）。

### 分步执行（便于排查）

```bash
# 1) 扫描：生成 evidence_bundle.json（含 all_files + user_hints.n_subjects）
python3 cli.py scan --in /path/to/input --out /path/to/bids_out --nsubjects 18

# 2) 规则：LLM 或智能 stub 产出 BIDSMap.yaml（含 assignment_rules + mappings）
python3 cli.py map --evidence /path/to/bids_out/evidence_bundle.json \
                   --out /path/to/bids_out --llm openai --model gpt-5-mini

# 3) 执行：按规则放入 BIDS，剩余全部进 derivatives/
python3 cli.py exec --in /path/to/input --out /path/to/bids_out \
                    --rules /path/to/bids_out/BIDSMap.yaml

# 4) 验证（可选）
python3 cli.py validate --out /path/to/bids_out
```

---

## 🔑 关键参数与输入输出

* `--in`：输入目录或 `.zip`。若是 zip，会解压到 `out/_staging/extracted/` 后处理；执行完将**原始 zip**复制到 `derivatives/orig/archives/`。
* `--out`：BIDS 输出目录。
* `--llm {openai|stub}`：用 ChatGPT 生成规则或使用**智能 stub**（离线）。
* `--model`：如 `gpt-5-mini`、`gpt-5` 等。
* `--nsubjects`：受试者总数（**强烈建议提供**）。会被写到 `evidence_bundle.json.user_hints.n_subjects`，指导 LLM 或 stub 生成分配。

主要产物：

* `BIDSMap.yaml`：规则本体（含 `subjects/assignment_rules/mappings/policy/...`）
* `conversion_log.json`：输入→输出映射日志
* `BIDSMap_with_manifest.yaml`：在规则末尾嵌入 `manifest`（最终文件清单 + 目录树）
* `derivatives/orig/archives/`：保留原始 zip

---

## 🧭 工作流程（Pipeline）

1. **ingest**（`ingest.py`）

   * 目录：直接使用。
   * `.zip`：解压到 `out/_staging/extracted/`；保留原 zip 以追溯。

2. **scan**（`scan.py`）

   * 识别多种 NIRS 格式：`.nirs/.snirf/.mat/.csv/.tsv/.txt/.xlsx`（可扩展）。
   * 生成 `evidence_bundle.json`：

     * `counts`：扩展名分布
     * `samples`：代表样本（小片头/表头）
     * `trio_found`：是否存在三件套（大小写不敏感）；存在则后续**原样复制**到根
     * `all_files`：**全量相对路径**（供 LLM 聚类/分配 subject）
     * `user_hints.n_subjects`：来自 `--nsubjects`

3. **map**（`llm_openai.py`）

   * **在线**：把 `evidence_bundle.json` 发给 ChatGPT（Responses API），产出 **YAML**：

     * `subjects.total / labels`
     * `assignment_rules`：每个 subject 的 `match`（glob 或显式文件路径）
     * `mappings`：每个模态的匹配 + 输出模板（使用 `{subject}`、`{ext}`）
     * `policy`：只保留 BIDS 必需 + 三件套在根；其余全部入 `derivatives/`
     * `questions`：待确认项
   * **离线/失败**：启用“**智能 stub**”——

     * 从 `all_files` 里挑出 NIRS 相关文件，按顺序分配给 `sub-01..sub-N`；
     * 生成 `assignment_rules + mappings`，能立即把文件放入 `sub-xx/nirs/`。

4. **exec**（`execmap.py`）

   * 根据 `assignment_rules`（**大小写无关** glob）给每个文件确定 `subject`。
   * 模板上下文：`{filename} / {parentname} / {relpath} / {subject} / {ext}`。
   * 将匹配到的文件复制进 BIDS（演示版只 copy；生产可接入转换器）。
   * 将**未匹配**的文件（且非三件套）路由到 `derivatives/` 的各桶：`docs/orig/intermediate/_misc`。
   * 输出 `conversion_log.json` 与 `BIDSMap_with_manifest.yaml`（含 `manifest.tree_text`）。

5. **validate**（`validate.py`）

   * 有 `bids-validator`：输出真实报告；
   * 无：返回“桩”（无错误），不阻塞流程。

---

## 🧩 YAML 示例（节选）

> 由 LLM 或智能 stub 生成的 `BIDSMap.yaml` 大致结构如下：

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
> * `assignment_rules` 不用你写正则；LLM/Stub 会用 **glob** 或**显式路径**分配。
> * `bids_out` 用 `{subject}` 和 `{ext}`（保留原扩展名）。
> * 三件套大小写不敏感识别，输出统一为小写名：`readme.md/participants.tsv/dataset_description.json`。

---

## 🧪 调试与排错

1. `[plan] 0 files` ➜ 规则缺少 `mappings` 或 `assignment_rules` 没匹配到。

   * 打开 `BIDSMap.yaml` 确认两者存在。
   * 看 `conversion_log.json` 的 `planned`，是否为空；
   * 若用 `--llm openai` 仍为空，先用 `--llm stub` 验证流程。
2. 文件没进 BIDS、全进 `derivatives/_misc/`：

   * 检查模式大小写（执行器对 glob **不区分大小写**，一般不是原因）。
   * 检查是否遗漏了某种扩展名（可在 `mappings.match` 里补一条）。
3. 受试者数量对不上：

   * 确认 `--nsubjects` 是否传递；
   * 手动编辑 `assignment_rules`，为同一 subject 增加多条 `match`。
4. 需要严格 BIDS-NIRS（SNIRF）：

   * 先保留原格式在 BIDS 或 `derivatives/`；
   * 接入转换器后，把 `bids_out` 改成输出 `.snirf` 并补 sidecar 字段。

---

## 🔧 可扩展点

* **识别更多格式**：在 `scan.py` 的 `NIRS_EXT` 与 `NIRS_NAME_HINTS` 中添加；
* **高级分配逻辑**（离线 stub）：可将“顺序分配”升级为“同前缀/同父目录聚类”。
* **真实转换**：在 `execmap.py` 的 copy 位置接入 `dcm2niix`、`mne-bids` 等。
* **隐私过滤**：扫描时剔除潜在 PHI 文本片段。

---

## 📦 产出示例

```
bids_out/
├─ readme.md                     # 若输入根存在（大小写不敏感）
├─ participants.tsv              # 若输入根存在
├─ dataset_description.json      # 若输入根存在
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
├─ BIDSMap_with_manifest.yaml    # 含 manifest：files + tree_text
└─ conversion_log.json
```

---

## ❓FAQ

**Q：没有 LLM 能跑吗？**
A：可以。使用 `--llm stub`，会基于 `--nsubjects` 自动生成 `assignment_rules` 与 `mappings`。

**Q：文件名没有 subject 线索怎么办？**
A：LLM 会根据**目录/批次/前缀**做聚类分配；从 stub 起步也能先“一个文件对一个 subject”，再人工微调 `assignment_rules`。

**Q：为什么把很多文件放进 `derivatives/`？**
A：保持 BIDS 主体**最小**，便于校验与下游复用；原始与文档仍被完整保留在 `derivatives/`。

---

## 📄 许可证

示例代码用于研究/教学场景。感谢 BIDS 社区与 OpenAI SDK。

