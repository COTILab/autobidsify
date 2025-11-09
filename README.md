# auto-bidsify

---

## 系统概述

auto-bidsify 是一个**全自动神经影像数据标准化系统**，能够将各种格式的神经影像数据（MRI、fNIRS 等）自动转换为符合 BIDS（Brain Imaging Data Structure）标准的数据集。

### 核心能力

- ✅ **全自动转换**：从原始数据到标准 BIDS 数据集，一键完成
- ✅ **智能识别**：自动检测受试者数量、模态类型、文件结构
- ✅ **多模态支持**：MRI（T1w/T2w/BOLD/DWI）、fNIRS、混合数据集
- ✅ **智能重命名**：自动生成符合 BIDS 规范的文件名
- ✅ **文档提取**：从 PDF/DOCX 提取实验参数和元数据
- ✅ **质量保证**：多层验证确保数据完整性和准确性

### 适用场景

1. **单中心研究**：单个实验室采集的神经影像数据
2. **多中心研究**：多个站点的数据（如 Cambridge、Beijing、NYC）
3. **历史数据整理**：回顾性整理已有的非标准数据
4. **数据共享准备**：准备公开发布的数据集
5. **大型队列研究**：如 CamCAN、UK Biobank 等大型项目

---

## 核心设计理念

### 1. 智能与确定性分离

**理念**：将"理解"和"执行"分离，确保数据转换的精确性。

- **LLM 负责**：语义理解、文档分析、模式识别、决策制定
- **Python 负责**：精确计算、数学运算、文件操作、数据转换

**为什么这样设计？**
- 神经影像数据是科研数据，不能有任何精度损失
- LLM 擅长理解复杂的描述和文档，但不适合做精确计算
- Python 代码执行确定性操作，保证每次运行结果一致

**实际例子**：
- LLM 从 PDF 中读取"TR = 2.25s"，理解这是重复时间参数
- Python 将 2.25 精确写入 NIfTI 文件头，确保数值完全正确

---

### 2. 数据最小化原则

**理念**：只让 LLM 接触必要的元数据摘要，保护隐私和敏感信息。

**实施方式**：
- 不发送完整文件列表给 LLM（可能有几万个文件）
- 只发送代表性样本（前 50 个文件路径）
- 不发送原始影像数据（NIfTI/DICOM 文件内容）
- 只发送统计摘要（如"3763 个受试者，每人平均 3.1 个文件"）

**效果**：
- API 调用成本从 ~128K tokens 降低到 ~5K tokens
- 保护受试者隐私（不泄露完整文件结构）
- 提高处理速度（减少数据传输量）

---

### 3. 多层闸门机制

**理念**：在关键节点设置质量检查点，确保数据质量。

**三类闸门**：

1. **阻塞闸门（Block）**：必须解决才能继续
   - 缺少必填字段（如 dataset_description.json 的 Name）
   - 无法识别受试者数量且用户未提供
   - 文件命名完全混乱无法分类

2. **警告闸门（Warn）**：建议处理但可继续
   - 检测到的受试者数量可疑（如平均每人只有 1 个文件）
   - 缺少推荐字段（如 README.md）
   - License 字段格式不标准但可自动修正

3. **信息闸门（Info）**：仅提供信息
   - 文档中提到多个作者但只提取到部分
   - 建议用户手动检查某些字段

---

### 4. 渐进式降级策略

**理念**：当某个环节失败时，系统不会崩溃，而是优雅降级。

**降级链**：

```
最优方案：LLM 生成完整配置
    ↓ 失败
备选方案：使用启发式规则推断
    ↓ 失败
最低方案：使用默认值 + 警告用户
```

**实际例子**：

**文件命名降级**：
1. 首选：使用 LLM 生成的 filename_rules
2. 备选：使用 Python 启发式推断（检测关键词）
3. 保底：保持原文件名 + 警告

**受试者检测降级**：
1. 首选：Python 正则匹配（高置信度）
2. 备选：文档内容分析（中置信度）
3. 保底：启发式估算（低置信度）
4. 终极：阻塞并要求用户提供

---

## 完整工作流程

### 系统架构概览

```
输入数据
   ↓
┌─────────────────────────────────────────┐
│  第一阶段：数据摄取 (Ingest)               │
│  • 解压压缩包或引用目录                    │
│  • 记录数据位置和类型                      │
│  • 不复制数据（性能优化）                   │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  第二阶段：证据收集 (Evidence)             │
│  • 扫描文件结构                           │
│  • 提取文档内容（PDF/DOCX/TXT）            │
│  • 自动检测受试者数量                      │
│  • 识别数据模态                           │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  第三阶段：分类分流 (Classification)       │
│  [仅混合模态数据需要]                      │
│  • LLM 分析文件属性                       │
│  • 分流到 MRI/fNIRS 池                   │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  第四阶段：三件套生成 (Trio)               │
│  • dataset_description.json             │
│  • README.md                            │
│  • participants.tsv                     │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  第五阶段：计划制定 (Plan)                 │
│  • 分析文件结构模式                        │
│  • 生成受试者映射规则                      │
│  • 制定文件命名规则                        │
│  • 创建 BIDSPlan.yaml                    │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  第六阶段：确定性执行 (Execute)            │
│  • 应用文件命名规则                        │
│  • 组织目录结构                           │
│  • 复制和重命名文件                        │
│  • 生成 sidecar JSON                     │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  第七阶段：验证 (Validate)                │
│  • BIDS 结构验证                         │
│  • 必填字段检查                           │
│  • 生成验证报告                           │
└─────────────────────────────────────────┘
   ↓
标准 BIDS 数据集
```

---

## 七阶段详解

### 第一阶段：数据摄取 (Ingest)

**目标**：将输入数据准备好供后续处理。

**输入**：
- 压缩文件（.zip、.tar.gz）
- 或目录路径

**处理流程**：

1. **检测输入类型**
   - 如果是压缩文件 → 解压到 `_staging/extracted/`
   - 如果是目录 → **不复制**，直接记录路径（性能优化）

2. **生成摄取记录**
   - 创建 `ingest_info.json`
   - 记录数据位置、类型、时间戳

**输出**：
- `_staging/ingest_info.json`：包含数据路径信息

**为什么不复制目录？**
- 大型数据集可能有几百 GB，复制需要很长时间
- 直接引用原始位置，节省磁盘空间和时间
- 只有压缩文件才需要解压（否则无法访问）

**实际例子**：
```
输入：camcan_data.zip (50 GB)
操作：解压到 _staging/extracted/
时间：约 5-10 分钟

输入：/data/camcan/ (目录)
操作：仅记录路径，不复制
时间：< 1 秒
```

---

### 第二阶段：证据收集 (Evidence)

**目标**：收集所有必要的元数据，为后续决策提供依据。

**收集内容**：

1. **文件结构信息**
   - 递归扫描所有文件
   - 按扩展名分类统计
   - 生成代表性样本列表

2. **文档内容提取**
   - 完整读取 PDF/DOCX/TXT 文件内容
   - 提取实验方案、参数、描述
   - 记录元数据来源

3. **自动受试者检测**（核心功能）
   
   **三层策略**：
   
   **Layer 1: Python 正则匹配**
   - 搜索文件路径中的模式
   - 支持 7 种命名模式：
     * `sub-01/`, `sub-025/` (BIDS 标准)
     * `Cambridge_sub06272/` (站点前缀)
     * `subject_001/` (Subject 前缀)
     * `s001/`, `001/` (简短格式)
   - 提取唯一 ID 集合
   - 计算平均文件数
   - **高置信度判断**：平均每人 ≥2 个文件
   
   **Layer 2: 文档内容分析**
   - 在 PDF/DOCX 中搜索关键短语
   - 模式：
     * "650 subjects"
     * "n = 650"
     * "sample size: 650"
   - **中置信度判断**：文档明确提到
   
   **Layer 3: 启发式估算**
   - 检测到模式但分布可疑
   - **低置信度判断**：需要用户验证
   
   **完全失败**：
   - 无法检测 → 要求用户提供

4. **模态识别**
   - 检测文件扩展名
   - 推断数据类型（MRI/fNIRS/混合）

5. **Trio 文件检查**
   - 检测输入数据是否已有 trio 文件
   - 如果存在，提升到输出根目录

**输出**：
- `_staging/evidence_bundle.json`：完整的证据包
  * 文件统计
  * 文档内容
  * 受试者检测结果
  * 模态信息

**性能优化**：

**用户提供 --nsubjects 时**：
- ✅ 跳过整个检测逻辑
- ✅ 立即使用用户值
- ✅ 节省 10-30 秒（大数据集）

**实际例子**：

```
数据集：CamCAN (3763 受试者, 11,598 个文件)

检测过程：
1. Python 正则匹配
   → 找到模式：Cambridge_sub06272, Beijing_sub82980...
   → 提取 3763 个唯一 ID
   → 平均每人 3.1 个文件 ✓

2. 置信度：HIGH
3. 最终结果：使用 3763（auto-detected）
```

---

### 第三阶段：分类分流 (Classification)

**触发条件**：仅当 `--modality=mixed` 或未指定模态时运行。

**目标**：将混合模态数据分流到不同处理通道。

**为什么需要分类？**
- MRI 和 fNIRS 的转换逻辑完全不同
- 混合在一起处理会导致错误
- 分流后可以并行处理（未来可扩展）

**分类策略**：

1. **文档分析优先**
   - LLM 读取 protocol.pdf
   - 识别描述："fNIRS optodes"、"3T MRI scanner"
   - 根据描述分类文件

2. **文件特征次之**
   - 扩展名：`.snirf` → fNIRS, `.nii.gz` → MRI
   - 文件大小：MRI 文件通常更大
   - 元数据字段：DICOM 标签、SNIRF 字段

3. **路径模式最后**
   - 目录名：`fmri/`, `nirs_data/`
   - 文件名关键词：`bold`, `rest`, `hbo`

**输出**：
- `_staging/nirs_pool/`：fNIRS 文件
- `_staging/mri_pool/`：MRI 文件
- `_staging/unknown/`：无法分类的文件
- `_staging/classification_plan.json`：分类依据

**单模态优化**：

如果指定 `--modality=mri` 或 `--modality=nirs`：
- ✅ 完全跳过此阶段
- ✅ 节省 1 次 LLM 调用
- ✅ 更快完成

**实际例子**：

```
输入：100 个文件（50 个 .nii.gz + 50 个 .snirf）

LLM 分析：
- protocol.pdf 提到 "3T Siemens scanner" + "fNIRS Artinis system"
- 结论：混合数据集

分类结果：
- mri_pool: 50 个 .nii.gz 文件
- nirs_pool: 50 个 .snirf 文件
- unknown: 0 个文件

下一步：MRI 和 fNIRS 分别进入各自的转换通道
```

---

### 第四阶段：三件套生成 (Trio)

**目标**：生成 BIDS 必需的三个核心文件。

**为什么叫"三件套"？**
- 这是 BIDS 规范要求的最基础文件
- 没有这三个文件，数据集不是有效的 BIDS
- 类似于"身份证、户口本、出生证明"

---

#### 文件 1: dataset_description.json

**作用**：数据集的"身份证"，描述数据集基本信息。

**必填字段**：
- `Name`：数据集名称
- `BIDSVersion`：BIDS 规范版本（1.10.0）
- `License`：数据使用许可

**可选但推荐字段**：
- `Authors`：作者列表（必须是数组）
- `Funding`：资助信息（必须是数组）
- `EthicsApprovals`：伦理批准（必须是数组）
- `ReferencesAndLinks`：相关文献

**生成流程**：

1. **LLM 提取信息**
   - 从 PDF/DOCX 中提取作者名单
   - 识别资助号码（如 NIH R01-xxx）
   - 查找伦理批准号
   - 提取数据集描述

2. **智能合并**
   - 如果文件已存在 → 合并新旧信息
   - 保留用户手动编辑的字段
   - 只更新缺失或错误的字段

3. **格式验证和修正**
   - 检查字段类型（Authors 必须是数组）
   - 自动转换错误格式
   - License 模糊匹配和标准化

**License 标准化示例**：
```
输入变体               → 标准输出
"CC BY 4.0"           → "CC-BY-4.0"
"cc-by 4.0"           → "CC-BY-4.0"
"Creative Commons 4"  → "CC-BY-4.0"
"MIT License"         → "MIT"
```

**作者提取策略**：

优先级 1：显式作者列表
- 搜索："Authors:", "Contributors:", "Author list:"

优先级 2：文档末尾
- 如果提到"authors at end"，搜索最后 20%

优先级 3：作者署名段落
- 识别带机构编号的作者名（如 "John Smith¹, Jane Doe²"）

优先级 4：部分提取 + 问题
- 只能提取部分 → 生成 Info 问题建议用户补充

**输出**：
- `dataset_description.json`

**质量检查**：
- ❌ 缺少 Name → BLOCK（阻塞）
- ❌ 缺少 License → BLOCK（阻塞）
- ⚠️ Authors 是字符串而非数组 → 自动修正
- ℹ️ 文档提到多作者但只提取到部分 → Info 问题

---

#### 文件 2: README.md

**作用**：数据集的"说明书"，提供详细描述和使用指南。

**推荐包含内容**：
- 数据集概述
- 实验设计
- 采集参数
- 数据处理流程
- 使用限制
- 引用方式

**生成流程**：

1. **LLM 提取和综合**
   - 从文档中提取实验描述
   - 整理采集参数
   - 提取使用条款
   - 综合成 Markdown 格式

2. **灵活生成**
   - LLM 可以直接返回 Markdown 文本
   - 或返回 JSON 结构再转换
   - 根据内容自动调整格式

3. **已有文件保护**
   - 如果已存在 README → 不覆盖
   - 检测各种命名变体（readme.txt, README.rst）

**输出**：
- `README.md`

**特点**：
- 使用 temperature=0.4（较高创造性）
- 允许 LLM 自由发挥写作风格
- 生成人类可读的描述性文本

---

#### 文件 3: participants.tsv

**作用**：受试者信息表，类似"花名册"。

**基本格式**：
```
participant_id    age    sex    group
sub-01           25     F      control
sub-02           67     M      patient
sub-03           34     F      control
```

**生成策略**：

**小型数据集（≤100 受试者）**：
- Trio 阶段生成简单版本
- 只有 participant_id 列
- 使用检测到的受试者列表

**大型数据集（>100 受试者）**：
- Trio 阶段**跳过**
- 推迟到 Plan 阶段生成
- Plan 阶段有完整的受试者分析结果

**多站点数据集**：
- Plan 阶段生成
- 包含 site 列（如 Cambridge, Beijing）
- 从文件名提取站点信息

**输出**：
- `participants.tsv`（可能在 Trio 或 Plan 阶段生成）

**质量检查**：
- ℹ️ 大数据集 → 推迟到 Plan 阶段
- ℹ️ 小数据集 → Trio 阶段生成基础版本

---

### 第五阶段：计划制定 (Plan)

**目标**：制定详细的转换计划，类似"施工图纸"。

**这是系统的"大脑"，做出所有关键决策。**

**核心任务**：

#### 1. 受试者分析（核心中的核心）

**Python 确定性提取**：

```
扫描所有文件路径 → 正则匹配模式 → 提取唯一 ID

输入：
- Cambridge_sub06272/scan_mprage.nii.gz
- Cambridge_sub06272/scan_rest.nii.gz
- Beijing_sub82980/scan_mprage.nii.gz
- NYC_sub12345/scan_mprage.nii.gz

输出：
subject_records = [
  {original_id: "Cambridge_sub06272", numeric_id: "06272", site: "Cambridge"},
  {original_id: "Beijing_sub82980", numeric_id: "82980", site: "Beijing"},
  {original_id: "NYC_sub12345", numeric_id: "12345", site: "NYC"}
]

has_site_info: True
needs_standardization: True
```

**为什么在 Plan 阶段做这个？**
- Plan 阶段可以访问**完整文件列表**
- Evidence 阶段只有样本，不适合精确分析
- Plan 阶段有足够信息做准确决策

#### 2. 标准化决策

**决策树**：

```
IF 检测到站点前缀（如 Cambridge_sub06272）
  → 标准化：YES
  → 策略：提取站点到 participants.tsv
  → 示例：Cambridge_sub06272 → sub-06272 (site: Cambridge)

IF 已经是 BIDS 格式（如 sub-01）
  → 标准化：NO
  → 保持原始 ID

IF 非标准格式（如 subject001）
  → 标准化：YES
  → 策略：转换为 sub-XX 格式
```

#### 3. 文件命名规则生成（新增核心功能）

**目标**：为每种文件类型生成 BIDS 命名规则。

**LLM 分析文件名并生成规则**：

```yaml
mappings:
  - modality: mri
    match: ["**/*mprage*.nii.gz"]
    filename_rules:
      - pattern: ".*anonymized.*"
        bids_name: "sub-{subject}_acq-anonymized_T1w.nii.gz"
      - pattern: ".*skull.*strip.*"
        bids_name: "sub-{subject}_acq-skullstripped_T1w.nii.gz"
      - pattern: ".*"
        bids_name: "sub-{subject}_T1w.nii.gz"
  
  - modality: mri
    match: ["**/*rest*.nii.gz"]
    filename_rules:
      - pattern: ".*run[_-]?(\\d+).*"
        bids_name: "sub-{subject}_task-rest_run-{run}_bold.nii.gz"
      - pattern: ".*"
        bids_name: "sub-{subject}_task-rest_bold.nii.gz"
```

**规则特点**：
- 按优先级排序（具体规则在前，通用规则在后）
- 支持正则表达式捕获组（如提取 run 编号）
- 使用占位符（`{subject}`, `{run}`, `{session}`）

#### 4. 生成 participants.tsv（如果 Trio 未生成）

**多站点数据集示例**：

```python
# Python 生成 TSV
participant_id    site
sub-06272        Cambridge
sub-82980        Beijing
sub-12345        NYC
sub-43358        Cambridge
```

**提取逻辑**：
- 从 `Cambridge_sub06272` 提取站点名 "Cambridge"
- 从 `Cambridge_sub06272` 提取数字 ID "06272"
- 生成标准 ID `sub-06272`
- 记录站点信息到 site 列

#### 5. 生成 BIDSPlan.yaml

**计划文件结构**：

```yaml
standardization:
  apply: true
  strategy: "site_to_participants"
  reason: "Multi-site dataset detected"

subjects:
  labels: ["06272", "82980", "12345", ...]  # 示例，Python 会补全

assignment_rules:
  - subject: "06272"
    original: "Cambridge_sub06272"
    site: "Cambridge"
    match: ["**/Cambridge_sub06272/**"]

mappings:
  - modality: mri
    match: ["**/*.nii.gz"]
    format_ready: true
    convert_to: none
    bids_out: "sub-{subject}/anat/sub-{subject}_T1w.nii.gz"
    filename_rules: [...]

defaults:
  mri:
    RepetitionTime: 2.25
    EchoTime: 0.03
```

**输出**：
- `_staging/BIDSPlan.yaml`：主计划文件
- `_staging/subject_analysis.json`：受试者分析详情
- `participants.tsv`：受试者表（如果 Trio 阶段未生成）

**质量检查**：
- 如果无法确定受试者结构 → BLOCK
- 如果文件命名完全混乱 → BLOCK
- 生成警告和建议

---

### 第六阶段：确定性执行 (Execute)

**目标**：按照 Plan 执行文件转换和组织。

**核心特点**：
- ✅ 完全确定性（每次运行结果一致）
- ✅ 不调用 LLM（避免不确定性）
- ✅ 应用精确的数学变换
- ✅ 保证数据完整性

**执行流程**：

#### 步骤 1: 组织 Trio 文件

```
复制：
  dataset_description.json → bids_compatible/
  README.md → bids_compatible/
  participants.tsv → bids_compatible/
```

#### 步骤 2: 处理数据文件（核心）

**对每个文件**：

1. **匹配规则**
   - 检查文件是否匹配 mapping 的 pattern
   - 确定该文件属于哪个 mapping

2. **确定受试者**
   - 从文件路径或文件名提取受试者 ID
   - 查找 assignment_rules 匹配项
   - 应用标准化映射

3. **应用文件命名规则**
   - 按优先级尝试 filename_rules
   - 匹配正则表达式模式
   - 提取捕获组（如 run 编号）
   - 替换占位符生成 BIDS 文件名

4. **确定目标路径**
   - 根据模态确定子目录（anat/func/dwi/nirs）
   - 组合：`sub-{subject}/{subdir}/{bids_filename}`

5. **执行文件操作**
   - 创建目标目录
   - 复制文件到新位置
   - 复制关联的 sidecar JSON（如果存在）

**命名转换示例**：

```
输入文件：
  Cambridge_sub06272/scan_mprage_anonymized.nii.gz

匹配规则：
  pattern: ".*anonymized.*"
  bids_name: "sub-{subject}_acq-anonymized_T1w.nii.gz"

提取信息：
  subject: "06272" (从 assignment_rules 查找)
  
替换占位符：
  sub-{subject} → sub-06272
  
输出路径：
  bids_compatible/sub-06272/anat/sub-06272_acq-anonymized_T1w.nii.gz
```

**启发式后备逻辑**：

如果没有 filename_rules 或没有规则匹配：

```python
# Python 启发式推断
检测关键词：
  "mprage" or "t1" → T1w
  "rest" or "bold" → task-rest_bold
  "skull" or "strip" → acq-skullstripped

构建文件名：
  parts = ["sub-06272"]
  if "rest" in filename: parts.append("task-rest")
  if "skull" in filename: parts.append("acq-skullstripped")
  parts.append("T1w")
  
  result = "_".join(parts) + ".nii.gz"
  # sub-06272_acq-skullstripped_T1w.nii.gz
```

#### 步骤 3: 组织派生文件

```
不属于核心数据的文件 → derivatives/
  - 处理后的数据
  - 分析结果
  - 可视化图表
```

#### 步骤 4: 生成报告

**生成文件**：

1. `_staging/conversion_log.json`
   - 每个文件的转换记录
   - 源路径 → 目标路径
   - 成功/失败状态

2. `_staging/BIDSManifest.yaml`
   - 最终数据集的文件清单
   - 目录树 ASCII 可视化
   - 文件统计

**输出**：
- `bids_compatible/`：标准 BIDS 数据集
  - 正确的目录结构
  - BIDS 规范的文件名
  - 完整的元数据文件

**日志示例**：

```
[2/3] Organizing BIDS data files...
  [1/2] Processing mri files...
    Found 6 files
      scan_mprage_anonymized.nii.gz
        → sub-06272_acq-anonymized_T1w.nii.gz
      scan_mprage_skullstripped.nii.gz
        → sub-06272_acq-skullstripped_T1w.nii.gz
      scan_rest.nii.gz
        → sub-06272_task-rest_bold.nii.gz

✓ BIDS Dataset Created
Location: outputs/run2/bids_compatible
Files organized: 6
Failures: 0
```

---

### 第七阶段：验证 (Validate)

**目标**：确保生成的数据集符合 BIDS 规范。

**验证策略**：

#### 优先使用官方验证器

```bash
bids-validator bids_compatible/
```

如果系统安装了 `bids-validator`：
- 运行完整的 BIDS 验证
- 检查所有规范要求
- 生成详细的错误和警告报告

#### 后备内置验证

如果没有 `bids-validator`：

**检查项目**：

1. **必需文件**
   - ✓ dataset_description.json 存在
   - ✓ 包含 Name 字段
   - ✓ 包含 BIDSVersion 字段
   - ✓ 包含 License 字段

2. **推荐文件**
   - ⚠️ README.md（建议）
   - ⚠️ participants.tsv（建议）

3. **数据结构**
   - ✓ 至少有一个 sub-XX 目录
   - ✓ 受试者目录包含模态子目录
   - ✓ 文件名符合 BIDS 模式

4. **字段类型**
   - ✓ Authors 是数组
   - ✓ Funding 是数组
   - ✓ License 值在白名单中

**输出**：
- 验证报告（JSON 格式）
- 错误列表（如果有）
- 警告列表（如果有）
- 数据集统计

**报告示例**：

```
=== Validating BIDS Dataset ===
Location: outputs/run2/bids_compatible

✓ No BIDS errors found
⚠ 2 warnings:
  1. MISSING_README: README is recommended
  2. PARTICIPANTS_TSV_RECOMMENDED: participants.tsv is recommended

Dataset summary:
  Subjects: 3763
  Total files: 11,598

✓ Validation complete
```

---

## 使用指南

### 快速开始

#### 方式 1: 一键完整流程（推荐）

```bash
python cli.py full \
  --input /path/to/data \
  --output outputs/my_bids \
  --model gpt-4o
```

**适用场景**：
- 首次使用
- 标准数据集
- 想要最简单的操作

#### 方式 2: 分步执行（高级）

```bash
# 步骤 1
python cli.py ingest --input /path/to/data --output outputs/my_bids

# 步骤 2
python cli.py evidence --output outputs/my_bids --modality mri

# 步骤 3（可选，仅混合模态）
python cli.py classify --output outputs/my_bids --model gpt-4o

# 步骤 4
python cli.py trio --output outputs/my_bids --model gpt-4o --file all

# 步骤 5
python cli.py plan --output outputs/my_bids --model gpt-4o

# 步骤 6
python cli.py execute --output outputs/my_bids

# 步骤 7
python cli.py validate --output outputs/my_bids
```

**适用场景**：
- 需要检查中间结果
- 某个步骤需要人工干预
- 调试和开发

---

### 参数说明

#### 通用参数

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|--------|
| `--input` | 路径 | 输入数据（文件或目录） | 必填 |
| `--output` | 路径 | 输出目录 | 必填 |
| `--model` | 字符串 | LLM 模型名称 | gpt-4o |

#### 可选参数

| 参数 | 类型 | 说明 | 默认值 |
|-----|------|------|--------|
| `--nsubjects` | 整数 | 受试者数量（自动检测） | None |
| `--modality` | 枚举 | 数据模态：mri/nirs/mixed | None |
| `--describe` | 字符串 | 数据集描述文本 | None |

#### Trio 特殊参数

| 参数 | 说明 |
|-----|------|
| `--file all` | 生成全部三个文件（默认） |
| `--file dataset_description` | 仅生成 dataset_description.json |
| `--file readme` | 仅生成 README.md |
| `--file participants` | 仅生成 participants.tsv |

---

### 典型使用场景

#### 场景 1: 标准单中心 MRI 数据

```bash
python cli.py full \
  --input /data/study_mri.zip \
  --output outputs/study_bids \
  --modality mri \
  --model gpt-4o
```

**数据特点**：
- 单个中心采集
- 纯 MRI 数据
- 标准文件命名（sub-01/, sub-02/）

**处理时间**：
- 100 个受试者：约 2-3 分钟
- 500 个受试者：约 5-8 分钟
- 不包括 LLM API 调用时间

---

#### 场景 2: 多中心大型队列研究

```bash
python cli.py full \
  --input /data/camcan/ \
  --output outputs/camcan_bids \
  --nsubjects 3763 \
  --modality mri \
  --model gpt-4o \
  --describe "CamCAN: Cambridge Centre for Ageing and Neuroscience"
```

**数据特点**：
- 多个站点（Cambridge, Beijing, NYC）
- 数千个受试者
- 站点前缀命名（Cambridge_sub06272）

**处理流程**：
1. 自动识别站点信息
2. 提取站点到 participants.tsv
3. 标准化为 sub-XXXXX 格式
4. 保留站点列便于分析

**提供 --nsubjects 的好处**：
- 跳过自动检测（节省 10-30 秒）
- 直接使用准确数量

---

#### 场景 3: fNIRS 功能成像数据

```bash
python cli.py full \
  --input /data/nirs_study.zip \
  --output outputs/nirs_bids \
  --modality nirs \
  --model gpt-4o
```

**数据特点**：
- fNIRS 数据（.snirf 或 CSV 表格）
- 需要转换为 SNIRF 格式

**特殊处理**：
- CSV → SNIRF 转换
- 生成 channels.tsv
- 生成 optodes.tsv
- 创建 sidecar JSON

---

#### 场景 4: 混合模态数据

```bash
python cli.py full \
  --input /data/multimodal.zip \
  --output outputs/multi_bids \
  --modality mixed \
  --model gpt-4o
```

**数据特点**：
- 同时包含 MRI 和 fNIRS
- 需要智能分类

**额外步骤**：
- 运行 Classification 阶段
- 分流到不同处理通道
- 最终合并为统一数据集

---

#### 场景 5: 重新生成 Trio 文件

```bash
# 只重新生成 dataset_description.json
python cli.py trio \
  --output outputs/my_bids \
  --model gpt-4o \
  --file dataset_description

# 重新生成全部三个
python cli.py trio \
  --output outputs/my_bids \
  --model gpt-4o \
  --file all
```

**适用场景**：
- 原始文档更新了
- 需要修正某个 trio 文件
- 只需要更新部分文件

---

## 高级特性

### 1. 自动受试者检测

**无需手动指定 --nsubjects！**

**检测能力**：

✅ **高置信度模式**：
- `sub-01/`, `sub-025/`, `sub-650/`
- `Cambridge_sub06272/`, `Beijing_sub82980/`
- `subject_001/`, `participant_025/`
- `s001/`, `s025/`
- `001/`, `025/`, `100/`

✅ **文档分析**：
- PDF 中的 "n = 650"
- DOCX 中的 "650 subjects"
- 协议文本中的明确提及

✅ **启发式估算**：
- 检测到模式但分布异常
- 给出估算值 + 警告

❌ **完全失败**：
- 文件命名完全混乱
- 无法从文档推断
- 阻塞并要求用户提供

**性能优化**：

如果你明确知道受试者数量：

```bash
python cli.py full \
  --input /data/huge_dataset/ \
  --output outputs/bids \
  --nsubjects 3763 \
  --model gpt-4o
```

✅ 完全跳过检测逻辑
✅ 节省 10-30 秒（大数据集）
✅ 立即开始处理

---

### 2. 智能文件重命名

**从混乱到规范**：

```
输入：
  scan_mprage_anonymized.nii.gz      （不规范）
  scan_mprage_skullstripped.nii.gz  （不规范）
  scan_rest.nii.gz                   （不规范）

输出：
  sub-06272_acq-anonymized_T1w.nii.gz     （BIDS 标准）
  sub-06272_acq-skullstripped_T1w.nii.gz  （BIDS 标准）
  sub-06272_task-rest_bold.nii.gz         （BIDS 标准）
```

**命名规则**：

LLM 在 Plan 阶段生成规则：

```yaml
filename_rules:
  - pattern: ".*anonymized.*"
    bids_name: "sub-{subject}_acq-anonymized_T1w.nii.gz"
  - pattern: ".*skull.*"
    bids_name: "sub-{subject}_acq-skullstripped_T1w.nii.gz"
  - pattern: ".*rest.*"
    bids_name: "sub-{subject}_task-rest_bold.nii.gz"
```

Execute 阶段应用规则：
1. 匹配文件名与 pattern
2. 提取占位符值（subject, run, session）
3. 替换生成 BIDS 文件名
4. 如果没有规则匹配 → 启发式推断

**支持的 BIDS 实体**：
- `sub-XX`：受试者
- `ses-XX`：采集时段
- `task-XX`：任务名称
- `run-XX`：重复编号
- `acq-XX`：采集变体
- `dir-XX`：方向
- `echo-XX`：回波编号

---

### 3. 多站点数据处理

**自动识别站点前缀**：

```
文件结构：
  Cambridge_sub06272/
  Cambridge_sub43358/
  Beijing_sub82980/
  NYC_sub12345/

识别结果：
  站点：Cambridge, Beijing, NYC
  受试者：4 个
  需要标准化：是
```

**标准化策略**：

```
原始 ID              → 标准 ID     站点信息
Cambridge_sub06272  → sub-06272   Cambridge
Beijing_sub82980    → sub-82980   Beijing
NYC_sub12345        → sub-12345   NYC
```

**生成 participants.tsv**：

```
participant_id    site         age    sex
sub-06272        Cambridge     45     F
sub-82980        Beijing       52     M
sub-12345        NYC           38     F
```

**目录结构**：

```
bids_compatible/
├── dataset_description.json
├── README.md
├── participants.tsv          ← 包含 site 列
├── sub-06272/
│   └── anat/
│       └── sub-06272_T1w.nii.gz
├── sub-82980/
│   └── anat/
│       └── sub-82980_T1w.nii.gz
└── sub-12345/
    └── anat/
        └── sub-12345_T1w.nii.gz
```

---

### 4. License 智能标准化

**BIDS 要求的 License 格式**：

严格的标准格式（如 `CC-BY-4.0`），但用户输入可能是：
- "CC BY 4.0"（空格）
- "cc-by 4.0"（小写）
- "Creative Commons Attribution 4.0"（全称）
- "MIT License"

**自动标准化**：

```python
输入各种变体        → 标准输出
"CC BY 4.0"        → "CC-BY-4.0"
"cc by 4.0"        → "CC-BY-4.0"
"CC-BY 4.0"        → "CC-BY-4.0"
"CCBY4.0"          → "CC-BY-4.0"
"MIT License"      → "MIT"
"Apache 2"         → "Apache-2.0"
```

**标准化算法**：
1. 移除空格和连字符
2. 转大写
3. 查找映射表
4. 返回标准格式

**支持的 License**：
- 开放数据：PDDL, CC0, PD
- Creative Commons：CC-BY-4.0, CC-BY-SA-4.0, CC-BY-NC-4.0
- 开源软件：MIT, BSD-2-Clause, BSD-3-Clause, GPL-2.0, GPL-3.0
- 其他：MPL, Apache-2.0, CDDL-1.0
- 自定义：Non-Standard

---

### 5. 文档内容提取

**支持的文档格式**：
- ✅ PDF（使用 pdfplumber 或 PyPDF2）
- ✅ DOCX（使用 python-docx）
- ✅ TXT/MD/RST（纯文本）

**提取内容**：

从实验方案 PDF 中提取：
- 受试者数量："n = 650"
- 作者列表：完整的作者署名
- 采集参数：TR, TE, 体素大小
- 伦理批准：IRB 编号
- 资助信息：Grant 编号
- 实验设计：任务描述

**大文档处理**：
- PDF > 10 MB → 只读前 50 页
- 文本 > 1 MB → 截断
- 保持合理的 API 调用大小

**实际例子**：

```
输入：protocol.pdf (150 页)

提取：
  Page 5: "This study includes 650 participants..."
  Page 12: "TR = 2250 ms, TE = 2.99 ms, voxel size = 1mm isotropic"
  Page 18: "IRB approval: 2023-001-XYZ"
  Page 145: "Authors: John Smith, Jane Doe, ..." (作者列表)

使用：
  → dataset_description.json 的 Authors 字段
  → README.md 的方法部分
  → JSON sidecar 的 RepetitionTime 字段
```

---

### 6. 温度参数优化

**不同任务需要不同的"创造性"**。

**温度值设置**：

| 任务 | 温度 | 原因 |
|-----|------|------|
| Classification | 0.15 | 需要一致性，避免随机分类 |
| Dataset Description | 0.1 | 精确提取字段，不能创造 |
| README | 0.4 | 允许创造性写作 |
| Participants | 0.2 | 平衡准确性和灵活性 |
| BIDS Plan | 0.15 | 需要精确决策 |

**推理模型特殊处理**：

o1/o3/gpt-5 系列模型：
- ❌ 不支持 temperature 参数
- ✅ 自动检测并跳过
- ✅ 使用 max_completion_tokens 控制

```python
if model.startswith("o1") or model.startswith("o3"):
    # 推理模型
    api_params["max_completion_tokens"] = 16000
    # 不设置 temperature
else:
    # 标准模型
    api_params["temperature"] = 0.15  # 任务特定值
```

---

## 常见问题

### Q1: 需要什么环境？

**Python 版本**：
- Python 3.8 或更高

**必需依赖**：
```bash
pip install openai pyyaml nibabel numpy scipy h5py
```

**可选依赖**：
```bash
# PDF 提取
pip install pdfplumber PyPDF2

# DOCX 提取
pip install python-docx

# BIDS 验证
npm install -g bids-validator
```

---

### Q2: 数据会被上传到 OpenAI 吗？

**不会上传原始影像数据！**

**只发送**：
- 文件路径（不含数据内容）
- 文件统计（数量、大小）
- 文档内容（PDF/DOCX 的文本）
- 代表性样本列表（~50 个文件名）

**不发送**：
- NIfTI/DICOM 文件内容
- 受试者的影像数据
- 完整的文件列表（>1000 个文件时）

**隐私保护**：
- 文件名中如果包含敏感信息，会在发送前匿名化
- 采用数据最小化原则
- 只发送绝对必要的信息

---

### Q3: 处理大数据集需要多长时间？

**时间估算**：

| 数据集规模 | 文件数 | 处理时间 | 主要耗时 |
|----------|--------|---------|---------|
| 小型 | < 1,000 | 2-5 分钟 | LLM 调用 |
| 中型 | 1,000-10,000 | 5-15 分钟 | 文件扫描 + LLM |
| 大型 | 10,000-50,000 | 15-45 分钟 | 文件扫描 |
| 超大型 | > 50,000 | 45-120 分钟 | 主要是文件 I/O |

**不包括**：
- 网络传输时间（下载数据）
- 解压时间（如果输入是压缩包）
- DICOM → NIfTI 转换（如果需要）

**优化建议**：
- 如果输入是目录，不要压缩（避免解压时间）
- 如果知道受试者数量，提供 --nsubjects（跳过检测）
- 使用 SSD 存储（提高 I/O 速度）

---

### Q4: 系统检测的受试者数量不准确怎么办？

**原因可能**：
1. 文件命名不规范
2. 缺失部分受试者的数据
3. 包含测试/练习数据

**解决方法**：

```bash
# 方法 1: 手动指定数量
python cli.py full \
  --input data/ \
  --output bids_out \
  --nsubjects 650 \
  --model gpt-4o

# 方法 2: 检查检测详情
cat outputs/bids_out/_staging/evidence_bundle.json | grep -A 20 "subject_detection"

# 方法 3: 分步执行，在 evidence 阶段检查
python cli.py evidence --output bids_out
# 查看检测结果
cat bids_out/_staging/evidence_bundle.json
# 如果不准确，重新运行 evidence 提供 --nsubjects
python cli.py evidence --output bids_out --nsubjects 650
```

---

### Q5: 如何处理缺失的文件？

**场景**：某些受试者缺少某些模态的数据。

**BIDS 规范允许**：
- 不是每个受试者都要有所有模态
- 可以有 sub-01/anat/ 但没有 sub-01/func/

**系统处理**：
- ✅ 自动处理不完整数据
- ✅ 只转换存在的文件
- ✅ 在 participants.tsv 中标记所有受试者

**实际例子**：

```
输入：
  sub-01/anat/T1w.nii.gz  ✓
  sub-01/func/bold.nii.gz ✓
  sub-02/anat/T1w.nii.gz  ✓
  sub-02/func/            ✗ (缺失)

输出：
  sub-01/anat/sub-01_T1w.nii.gz  ✓
  sub-01/func/sub-01_task-rest_bold.nii.gz ✓
  sub-02/anat/sub-02_T1w.nii.gz  ✓
  (sub-02 没有 func/ 目录，但仍在 participants.tsv 中)
```

---

### Q6: 如何处理多次扫描（多 run）？

**BIDS 支持 run 实体**：
- `run-1`, `run-2`, `run-3`...

**系统检测方法**：

1. **从文件名提取**：
   ```
   rest_run1.nii.gz → run-1
   rest_run2.nii.gz → run-2
   bold_r01.nii.gz  → run-1
   ```

2. **使用正则捕获组**：
   ```yaml
   filename_rules:
     - pattern: ".*run[_-]?(\\d+).*"
       bids_name: "sub-{subject}_task-rest_run-{run}_bold.nii.gz"
   ```

3. **自动编号**：
   如果检测不到 run 编号：
   - 同一受试者的多个同类文件
   - 自动分配 run-1, run-2, run-3

**输出示例**：

```
bids_compatible/sub-01/func/
├── sub-01_task-rest_run-1_bold.nii.gz
├── sub-01_task-rest_run-1_bold.json
├── sub-01_task-rest_run-2_bold.nii.gz
└── sub-01_task-rest_run-2_bold.json
```

---

### Q7: 支持哪些 MRI 序列类型？

**解剖像**：
- T1w（T1 加权）
- T2w（T2 加权）
- T2starw（T2* 加权）
- FLAIR（流体衰减反转恢复）
- PD（质子密度）
- PDw（质子密度加权）
- inplaneT1/inplaneT2（平面内参考）
- angio（血管造影）

**功能像**：
- BOLD（血氧水平依赖）
- 支持 task-based 和 resting-state

**弥散像**：
- DWI（弥散加权成像）
- 支持多个 b 值和方向

**场图**：
- fieldmap（场图）
- phasediff（相位差）
- magnitude（幅度图）

**其他**：
- SWI（磁敏感加权成像）
- MTR（磁化传递率）

---

### Q8: 生成的文件名不对怎么办？

**问题定位**：

1. **检查 BIDSPlan.yaml**
   ```bash
   cat outputs/bids_out/_staging/BIDSPlan.yaml
   ```
   查看 filename_rules 是否合理

2. **检查 conversion_log.json**
   ```bash
   cat outputs/bids_out/_staging/conversion_log.json
   ```
   查看实际的命名转换记录

**常见问题**：

❌ **规则太宽泛**：
```yaml
# 问题：pattern 匹配所有文件
filename_rules:
  - pattern: ".*"
    bids_name: "sub-{subject}_T1w.nii.gz"
```

✅ **解决：增加具体规则**：
```yaml
filename_rules:
  - pattern: ".*mprage.*"
    bids_name: "sub-{subject}_T1w.nii.gz"
  - pattern: ".*rest.*"
    bids_name: "sub-{subject}_task-rest_bold.nii.gz"
```

**手动修正**：

1. 编辑 BIDSPlan.yaml 中的 filename_rules
2. 重新运行 execute：
   ```bash
   python cli.py execute --output outputs/bids_out
   ```

---

### Q9: 如何添加自定义元数据？

**方法 1: 手动编辑生成的文件**

生成基础版本后：
```bash
# 编辑 dataset_description.json
nano outputs/bids_out/dataset_description.json

# 添加字段
{
  "Name": "My Study",
  "License": "CC-BY-4.0",
  "Authors": ["John Doe"],
  "CustomField": "Custom Value"  ← 新增
}
```

**方法 2: 使用 --describe 参数**

```bash
python cli.py full \
  --input data/ \
  --output bids_out \
  --model gpt-4o \
  --describe "Study design: Cross-sectional. Task: Resting-state fMRI. Scanner: Siemens 3T."
```

LLM 会将这些信息整合到 README.md 和 dataset_description.json

**方
