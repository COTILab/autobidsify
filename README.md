# auto\_bidsify · 使用说明（README）

> 一句话目标：把**任意无序的文件夹或 .zip 压缩包**，自动整理成**最小 BIDS 主体**；
> **仅**保留 BIDS 必需内容 +（如有）用户三件套 `README.md / participants.tsv / dataset_description.json` 在 BIDS 根，
> **其余所有文件**（包含其他 `.md/.pdf/.html` 等说明材料）一律放入 `derivatives/` 分类存放。
> 规则由 **ChatGPT（Responses API）**自动生成（可离线回退到 stub），执行由**确定性脚本**完成。

---

## 用到了什么工具

* **Python 3.9+**：整套脚本语言环境
* **OpenAI Python SDK（Responses API）**：向 ChatGPT 发送“证据包”，让模型产出 **BIDSMap.yaml**（规则）
  （无 API Key 或离线时会回退到**本地 stub 规则**）
* **（可选）bids-validator**：验证生成的 BIDS 目录结构（若未安装则返回“桩”报告以保证流程可跑）

> 真实数据转换（如 DICOM→NIfTI、EEG/MEG/fNIRS 等）在本演示中**只做 copy/rename**，
> 你可在执行阶段替换为 `dcm2niix`、`mne-bids` 等工具调用。

---

## 代码目录结构

把下列文件放在同一目录：

```
auto_bidsify_v3/
├─ cli.py                # 命令行入口：scan → map → exec → validate（或 full 一键）
├─ ingest.py             # 输入摄取：支持目录或 .zip；zip 会解压到 staging
├─ scan.py               # 扫描器：抽样文件头信息 + 统计 + 三件套存在性 → evidence_bundle.json
├─ rules.py              # 规则 I/O + 轻量模板引擎（{filename|regex...}）
├─ llm_openai.py         # 调用 ChatGPT（Responses API）生成 BIDSMap（含 stub 回退）
├─ execmap.py            # 执行器：按 BIDSMap 落地；其余全部路由到 derivatives/
├─ validate.py           # 验证器封装：调用 bids-validator（无则返回桩）
└─ utils.py              # 常用工具：建目录/拷贝/彩色输出
```

---

## 快速开始

### 1) 安装依赖

```bash
pip install openai pyyaml
# （可选）安装 BIDS Validator（建议在容器/CI 中）
npm install -g bids-validator
```

### 2) 设置 OpenAI API Key（如果使用 ChatGPT 生成规则）

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> 若不设置，将自动使用**本地 stub 规则**（仍可完成“最小 BIDS 主体 + 其余入 derivatives”）。

### 3) 一键运行（推荐先用小数据尝试）

* **输入是文件夹：**

```bash
python3 cli.py full --in /path/to/messy_folder --out /path/to/bids_out --llm openai --model gpt-5-mini
```

* **输入是 .zip：**

```bash
python3 cli.py full --in /path/to/messy.zip --out /path/to/bids_out --llm openai --model gpt-5
```

### 4) 分步运行（便于排查问题）

```bash
# 1) 扫描：生成证据包 evidence_bundle.json 到 --out
python3 cli.py scan --in /path/to/input --out /path/to/bids_out

# 2) 规则：用 ChatGPT（或 stub）将证据包转换为 BIDSMap.yaml
python3 cli.py map --evidence /path/to/bids_out/evidence_bundle.json --out /path/to/bids_out --llm openai --model gpt-5-mini

# 3) 执行：按规则构建最小 BIDS 主体；其余全部路由到 derivatives/
python3 cli.py exec --in /path/to/input --out /path/to/bids_out --rules /path/to/bids_out/BIDSMap.yaml

# 4) 验证（如已安装 bids-validator）
python3 cli.py validate --out /path/to/bids_out
```

---

## 工作流程（Pipeline）

### 流程总览

1. **ingest（摄取）**

   * 接受**目录**或\*\*.zip\*\*。
   * 如果是 .zip，解压到 `out/_staging/extracted/`（不改动原压缩包），并记录原始 zip 路径用于后续溯源。

2. **scan（扫描）**

   * 统计后缀分布、每类抽样 3–5 个代表文件；
   * 读取**少量头信息**（文本前几百字符、表格头+几行、其他取前 4KB 指纹；若内部还有 zip，只列清单不再解压）；
   * 检测 BIDS 三件套（`README.md/participants.tsv/dataset_description.json`）**是否位于输入根目录**；
   * 产出 **`evidence_bundle.json`**（小而全，便于模型理解）。

3. **map（规则生成）**

   * 把证据包交给 **ChatGPT Responses API**，由模型输出 **`BIDSMap.yaml`**（仅 YAML，无围栏与解释）。
   * 如无 API Key 或 API 出错 → **回退到本地 stub**（仍包含核心策略与路由）。

4. **exec（执行）**

   * 读取 `BIDSMap.yaml`，对输入树做 pattern 匹配，使用**迷你模板引擎**生成 BIDS 目标路径；
   * **复制**（演示版）或**转换**（生产替换为真实工具）到目标；
   * 若用户在输入根提供了 BIDS 三件套，**原样复制**到 BIDS 根（不覆盖、不改写）；
   * **所有未映射到 BIDS 的文件**（且不是三件套）→ **统一路由到 `derivatives/`**，并按 `orig/docs/intermediate/_misc` 分类；
   * 如果输入原本是 .zip，**原始压缩包**也会复制到 `derivatives/orig/archives/` 以保留溯源。

5. **validate（验证）**

   * 若系统安装了 `bids-validator`，运行并返回 JSON 报告；
   * 否则返回“桩”，保证离线环境也能完整跑通。

### 路由策略（核心约束）

* **BIDS 根只保留：**

  * 映射到 BIDS 的**必需文件**（按 BIDSMap rules）；
  * **用户三件套**（如存在）：`README.md / participants.tsv / dataset_description.json`。
* **其余文件**（含**其他** `.md/.pdf/.html` 等说明材料）→ **一律进入 `derivatives/`** 对应分桶。
* `derivatives/docs` 默认包含 `**/*.md`（但 `README.md` 属“三件套”，留在 BIDS 根）。

---

## 输入数据的三种情况如何处理

1. **无序目录**

   * 直接作为“准备好的输入根”使用，后续照常扫描、规则生成与执行。

2. **.zip 压缩包**

   * 自动解压到 staging 目录，扫描/执行都在 staging 内进行；
   * 执行完成后把**原始 .zip**复制到 `derivatives/orig/archives/` 以保留追溯。

3. **三件套可有可无**

   * 如果**有**（可能只有其中 1–2 个），视为**参考与真相**，原样复制到 BIDS 根；
   * 如果**没有**，也不会阻塞——模型仍会根据其他文件推断规则；
   * 任何**其他**描述性文件（如 `METHODS.md`、论文 PDF、网页 HTML）将自动进入 `derivatives/docs/`。

---

## LLM（ChatGPT）在这里具体做什么？

* **输入**：由 `scan.py` 生成的 `evidence_bundle.json`（包括后缀统计、抽样头信息、三件套存在性）。
* **输出**：一份 **`BIDSMap.yaml`**，包含：

  * `mappings`: 每个模态的匹配规则（文件 pattern）与命名模板（`bids_out`）；
  * `policy`: 三件套保留、**非必需全部入 derivatives** 的硬约束；
  * `entities`: `subject/session/run` 的省略/格式策略；
  * `questions`: 对不确定项的清单（如 `task` 标签或 license 需要人工确认）。

> **重要**：LLM 只负责\*\*“写规则”**；所有实际的拷贝/命名/路由在本地脚本中**确定性\*\*执行，结果可审计与复现。

---

## 规则与模板引擎（`rules.py`）

* 规则中常见表达式（用于 `bids_out`）：
  `"{filename|regex:'sub-(\\d+)',1|pad2|or:'01'}"`

  * `regex:'pattern',N`：用正则提取第 N 个分组
  * `pad2`：将数字补足两位
  * `sanitize`：去掉非法字符
  * `or:default`：为空时给默认值
* 你可以把模型生成的 YAML 作为**长效配方**复用，也可以在此基础上手动微调。

---

## 验证（`validate.py`）

* 若已安装 `bids-validator`：会在 `validate` 步骤**真验证**，返回 error/warning 明细；
* 若未安装：返回“桩”结果（无错误），方便离线快速开发与演示。
* 建议在 CI 或容器环境固定 `bids-validator`，确保每次输出都能通过校验（或至少错误为 0）。

---

## 如何把演示版升级为“生产可用”

1. **替换真实转换器**（`execmap.py` 第 3 步 copy 的位置）：

   * MRI：`dcm2niix`（DICOM → NIfTI + JSON）
   * EEG/MEG/fNIRS：`mne-bids` / 专用 SDK
2. **增强扫描**（`scan.py`）：解析 DICOM Tag、SNIRF metaDataTags、EEG 通道/采样率等**只读头信息**；
3. **生成辅助 TSV/JSON**：根据规则自动生成 `events.tsv`、`channels.tsv`、`coordsystem.json` 等；
4. **输出报告**：把转换日志、路由统计、验证摘要输出成 HTML，便于汇报与审计；
5. **Prompt 强化**：加入 few-shot 示例、schema 检查（YAML Schema/JSON Schema），提升 LLM 产出稳定性；
6. **隐私过滤**：扫描阶段屏蔽可能的 PHI（如姓名/生日），不进入证据包或提示词。

---

## 常见问答（FAQ）

**Q1：没有三件套也能产生合格的 BIDS 吗？**
A：可以。模型会根据文件名/抽样头信息推断命名与模态；`ses/run` 在单次情况下省略；`task` 必要时给默认（如 fMRI `rest`）。不确定项会写到 `questions` 里，便于你后续人工补全。

**Q2：其他 `.md/.pdf/.html` 会不会把 `README.md` 路由走了？**
A：不会。`README.md` 被明确当作三件套的一部分，**保留在 BIDS 根**；其他 `.md`（例如 `NOTES.md`）会进入 `derivatives/docs/`。

**Q3：输入是 .zip，会不会把原始压缩包删掉？**
A：不会。我们在 staging 解压操作；执行完后会**复制原始 .zip**到 `derivatives/orig/archives/`，以保留溯源链路。

**Q4：为什么要把“非必需”都放到 `derivatives/`？**
A：保持 BIDS 主体**干净、最小**，便于 `bids-validator` 校验与下游分析；同时通过 `derivatives/` 完整保留原始与说明材料，兼顾合规与追溯。

---

## 最终outputs（示例）

```
bids_out/
├─ README.md
├─ participants.tsv
├─ dataset_description.json
├─ sub-01/
│  ├─ anat/
│  │  └─ sub-01_T1w.nii.gz         # 示例
│  └─ eeg/
│     └─ sub-01_task-rest_eeg.edf  # 示例
├─ derivatives/
│  ├─ orig/
│  │  └─ archives/
│  │     └─ messy.zip              # 输入是zip时复制过来
│  ├─ docs/
│  │  ├─ NOTES.md
│  │  └─ paper.pdf
│  ├─ intermediate/
│  └─ _misc/
├─ BIDSMap.yaml
├─ evidence_bundle.json
└─ conversion_log.json
```

---

## 运行失败怎么排查？

1. 先跑 `scan` 看 `evidence_bundle.json` 是否生成；
2. 跑 `map` 看 `BIDSMap.yaml` 是否输出，是否为空（无 API Key 会生成 stub）；
3. 跑 `exec` 查看 `conversion_log.json` 与 `derivatives/` 路由是否符合预期；
4. （可选）跑 `validate` 看报错定位；如有错误，把 `evidence_bundle.json + BIDSMap.yaml + 错误信息` 发给 LLM 让其“产出最小补丁”。

---

## 许可证与致谢

* 感谢 BIDS 社区工具与规范、OpenAI 提供的模型与 SDK。

