# Experimental Reasoning (ER) Benchmark选择与系统设计方案

## Context

设计一种名为 **Experimental Reasoning / Tactical Reasoning** 的**模型无关agentic scaffold**，模仿科学家做实验的推理过程：反应式（而非预测式），基于行动反馈驱动推理，结合NL CoT + Program-of-Thought + 工具增强 + 形式化推理，并通过provenance证据链和context管理模块记录完整实验轨迹。

## 核心设计原则

1. **模型无关 (Model-Agnostic)**: ER是scaffold/架构贡献，不是模型优化。任何LLM接入ER后都应获得性能提升。论文需展示 Model × Benchmark 矩阵证明一致性改善。
2. **通用性优先 (Generality-First)**: 核心ER循环零benchmark特定代码。领域战术可插拔，但核心loop对所有任务通用。不为单一benchmark极限调优。
3. **学术论文级严谨**: 5个benchmark + 3个模型族 + 消融实验 + SOTA对比，目标ICLR/NeurIPS投稿。

---

## 一、Benchmark选择方案（5个，同心圆策略）

```
               +-- tau-bench ----+  Ring 3: 非科学领域通用性
               |                 |
          +----+-- MLE-bench ----+----+  Ring 2: 邻近科学领域
          |                           |
     +----+-- ScienceAgentBench ------+----+  Ring 2: 多学科科学
     |                                     |
+----+-- BixBench + HeurekaBench ----------+----+  Ring 1: 核心生信/生物
+---------------------------------------------------+
```

### 选择理由与匹配度评分

| # | Benchmark | 领域 | 当前SOTA | 分数 | 剩余空间 | ER匹配度 | 选择理由 |
|---|-----------|------|---------|------|---------|---------|---------|
| 1 | **BixBench** | 生物信息学 | K-Dense 34.4% | 34.4% | ~66pp | ★★★★★ | 直接匹配ER核心领域；多步分析轨迹；已有68个生信工具 |
| 2 | **HeurekaBench** | 单细胞生物 | Claude-4-Sonnet 2.58/5.0 | 44% MCQ | ~56pp | ★★★★★ | 显式科学发现流程；Biomni交互环境；巨大提升空间 |
| 3 | **ScienceAgentBench** | 多学科科学 | o1-preview 42.2% | 42.2% | ~58pp | ★★★★☆ | 4学科通用性验证；现有SOTA是暴力重试，ER有结构优势 |
| 4 | **MLE-bench** | ML工程/Kaggle | MLE-STAR 43.9% | 43.9% | ~56pp | ★★★★☆ | 实验方法论天然匹配；AIDE→MLE-STAR已证明迭代方法有效 |
| 5 | **tau-bench** | 客服工具调用 | o4-mini 56% (airline) | 56%/86% | ~44pp | ★★★☆☆ | 证明非科学领域通用性；策略合规需要证据追踪 |

### 被排除的候选及原因

- **AstaBench**: 11个子benchmark + InspectAI + DVC集成成本过高，第一版后再考虑
- **DSBench**: 与MLE-bench方法论重叠度高，MLE-bench更有挑战性和影响力
- **MedAgentBench**: SOTA已69.67%，提升空间有限且需要医学专业知识
- **数学benchmark** (GPQA/OlympiadBench): 纯知识/数学推理不需要实验过程，ER优势不明显

---

## 二、当前SOTA系统弱点深度分析

### 1. BixBench — K-Dense Analyst (34.4%)

**系统架构**: 10个固定专业子agent (data loader, statistician, visualizer等) + Gemini 2.5 Pro骨干

**致命弱点**:
| 弱点 | 描述 | ER如何解决 |
|------|------|-----------|
| 刚性管线 | 10个子agent按固定顺序执行，不根据中间数据调整 | ER反应式循环：每步结果决定下一步策略 |
| 无证据链 | 最终答案错误时无法追溯哪个中间结论出错 | Prov-O式provenance图：每个结论可追溯到源数据 |
| 子agent通信退化 | 自由文本传递导致信息压缩+幻觉在10跳间累积 | CMM结构化证据传递：JSON证据记录替代自由文本 |
| 无假设管理 | 不形成、测试、修正假设，执行计划然后希望正确 | Hypothesis Manager主动管理假设生命周期 |

**Kepler (33.4%)弱点**: 单agent在20+步代码执行后上下文退化

### 2. HeurekaBench — Claude-4-Sonnet (2.58/5.0)

**系统架构**: 通用LLM + 代码执行，机会性使用Biomni数据库

**致命弱点**:
| 弱点 | 描述 | ER如何解决 |
|------|------|-----------|
| 无结构化发现周期 | 将insight→question→analysis视为平坦序列而非迭代循环 | ER的实验循环直接映射三阶段流程 |
| 44GB数据无策略采样 | 尝试分析全部或随机采样 | 假设驱动：先形成假设，再针对性查询数据子集 |
| 跨论文无证据聚合 | 13篇Nature/Cell论文独立分析 | CMM跨论文知识图谱，证据跨论文聚合 |
| 领域工具利用不足 | 不知道何时/为何查询Biomni数据库 | 生信战术库指导何时调用哪个数据库 |

### 3. ScienceAgentBench — o1-preview + self-debug (42.2%)

**系统架构**: 单LLM调用 + 错误消息反馈重试循环(最多10次)

**致命弱点** (源码分析 `agent.py`):
| 弱点 | 描述 | ER如何解决 |
|------|------|-----------|
| 暴力调试 | `step()`方法简单将error消息反馈LLM重试10次 | 假设驱动调试：分析为何失败→假设原因→针对性修复 |
| 无探索性数据分析 | 收到数据集描述后直接写完整程序 | ORIENT阶段：先EDA理解数据，再形成编码假设 |
| 上下文截断 | 系统消息截断到`context_cutoff-2000`，丢失领域知识 | CMM三级压缩：活跃假设L1 + 摘要L2 + 完整存档L3 |
| >10x成本低回报 | o1内部CoT消耗大量token但仅19-41%忠实 | ER结构化推理替代自由形式CoT，更高token效率 |

### 4. MLE-bench — MLE-STAR + Gemini 2.5 Pro (43.9%)

**系统架构**: 网络搜索→检索模型→消融引导优化→集成

**致命弱点**:
| 弱点 | 描述 | ER如何解决 |
|------|------|-----------|
| 搜索噪声 | 检索的Kaggle解决方案常过时或不相关 | 假设驱动搜索：先分析数据特征，再有针对性搜索 |
| 无方向消融 | 穷举变体而非假设驱动 | 每个消融是一个实验，有假设、预期、记录 |
| 无实验日志 | 不记录尝试了什么/成功了什么/失败了什么 | CMM实验日志：防止重复实验，支持回溯分析 |
| 机械集成 | 简单平均而非基于证据选择互补模型 | 证据驱动集成：基于实验日志选择互补性模型 |

### 5. tau-bench — o4-mini/Claude 3.7 (56% airline)

**系统架构**: ReAct/few-shot工具调用agent

**致命弱点**:
| 弱点 | 描述 | ER如何解决 |
|------|------|-----------|
| 策略合规漂移 | 多轮对话后丢失当前适用策略 | CMM策略状态追踪器：每次行动前显式验证 |
| 无系统性信息收集 | 反应式调用工具而非系统性收集 | 信息收集战术：决策前系统性查询所有相关状态 |
| 跨轮错误累积 | 早期错误API调用腐蚀后续推理 | 证据链+回溯机制：发现错误时追溯源头修正 |

---

## 三、ER系统架构设计

### 3.1 核心架构

```
+=========================================================================+
|                     EXPERIMENTAL REASONING SYSTEM                        |
+=========================================================================+
|                                                                          |
|  +------------------+    +--------------------+    +------------------+  |
|  |   REASONING CORE |<-->|  CONTEXT MANAGEMENT|<-->|  ACTION ENGINE  |  |
|  |                  |    |    MODULE (CMM)     |    |                 |  |
|  |  - Hypothesis    |    |                    |    |  - FSM控制器     |  |
|  |    Manager       |    |  - Evidence Store  |    |  - Tool Router  |  |
|  |  - Evidence      |    |  - Provenance Graph|    |  - Code Executor|  |
|  |    Evaluator     |    |  - Observation Log |    |  - Env Adapter  |  |
|  |  - Tactic        |    |  - Hypothesis      |    |                 |  |
|  |    Selector      |    |    Registry        |    |                 |  |
|  +--------+---------+    +--------+-----------+    +--------+--------+  |
|           |                       |                         |            |
|           +-----------+-----------+-------------+-----------+            |
|                       |                         |                        |
|               +-------v--------+       +--------v--------+              |
|               | FORMAT ENGINE  |       | DOMAIN TACTICS  |              |
|               |                |       |                 |              |
|               | - NL CoT       |       | - 生信战术      |              |
|               | - Program-of-  |       | - 数据科学战术  |              |
|               |   Thought      |       | - ML工程战术    |              |
|               | - Structured   |       | - 策略合规战术  |              |
|               |   JSON/Table   |       |                 |              |
|               +----------------+       +-----------------+              |
+=========================================================================+
```

### 3.2 ER循环（核心差异化）

**当前系统**: Plan → Execute → Maybe Retry
**ER系统**: Observe → Hypothesize → Execute → Interpret → Adapt → Iterate

```
                    ┌─────────┐
                    │  ORIENT │ ◄── 理解任务+初始观察
                    └────┬────┘
                         │
                    ┌────▼─────┐
              ┌────►│HYPOTHESIZE│ ◄── 基于观察形成/修正假设
              │     └────┬─────┘
              │          │
              │     ┌────▼─────┐
              │     │EXPERIMENT │ ◄── 选择战术，执行工具/代码
              │     └────┬─────┘
              │          │
              │     ┌────▼────┐
              │     │ OBSERVE  │ ◄── 记录结果到CMM
              │     └────┬────┘
              │          │
              │     ┌────▼─────┐    假设被支持？
              │     │ EVALUATE │───────────────┐
              │     └────┬─────┘               │
              │          │                     │
              │    假设被反驳/需修正        ┌───▼────┐
              └──────────┘                 │CONCLUDE │ ◄── 生成带证据链的最终答案
                                           └────────┘
```

**关键区别**:
- 状态转移由**证据条件**驱动（非LLM自由选择）
- 每步产生结构化证据记录（非自由文本）
- 假设生命周期管理：proposed → testing → supported/refuted → revised
- 上下文不会丢失：CMM三级压缩保持完整provenance

### 3.3 核心组件规格

#### A. Context Management Module (CMM) — "Agent的Zotero"

**数据模型** (扩展自 `/home/lingzhi/Code/1-Context-Engineering/life-long-memory/src/db.py` 的SQLite schema):

```sql
-- 假设注册表
hypotheses (id, statement, status, confidence, parent_id, created_at, updated_at)

-- 证据存储
evidence (id, source_action_id, content, evidence_type, created_at)

-- 证据-假设关联
evidence_hypothesis (evidence_id, hypothesis_id, relation, strength)
-- relation: 'supports' | 'contradicts' | 'neutral'

-- 行动日志
actions (id, tactic_name, input_summary, output_summary, tokens_used, duration_ms)

-- Prov-O 溯源图
provenance (subject_id, subject_type, predicate, object_id, object_type)
-- predicate: 'wasGeneratedBy' | 'wasDerivedFrom' | 'used' | 'wasInformedBy'

-- 观察日志
observations (id, action_id, content, content_type, truncated_summary)
```

**三级上下文管理**:
- **L1 (活跃上下文, 放入prompt)**: 当前假设 + 最近3条证据 + 当前战术状态
- **L2 (压缩摘要, 可查询)**: 先前实验的关键发现摘要
- **L3 (完整存档, SQLite)**: 完整provenance图，按需查询

#### B. Hypothesis Manager

```python
@dataclass
class Hypothesis:
    id: str
    statement: str           # "该数据集中RNA表达量与疾病状态显著相关"
    status: Literal["proposed", "testing", "supported", "refuted", "revised"]
    confidence: float        # [0, 1]
    evidence_for: list[str]  # evidence IDs
    evidence_against: list[str]
    parent_id: str | None    # 如果是修正版本
```

操作: `propose()` → `test()` → `update_confidence()` → `refute()/support()` → `revise()`

#### C. Domain Tactics Library

每个领域有一套战术目录（扩展自 `/home/lingzhi/Code/3-Action/chain-of-action/src/core/action_type.py` 的ActionCatalog模式）:

**生信战术** (BixBench, HeurekaBench):
- `explore_data`: 数据profiling + 分布可视化 + 异常检测
- `sequence_analysis`: BLAST/MMseqs2 → 命中分析 → GO迁移
- `statistical_test`: 选择统计检验 → 执行 → 解释p值
- `database_query`: 策略性Biomni/UniProt/KEGG查询
- `cross_paper_synthesis`: 跨论文证据聚合

**ML工程战术** (MLE-bench):
- `baseline_model`: 建立性能基线
- `feature_experiment`: 假设驱动的特征工程
- `model_experiment`: 有假设的模型架构/超参实验
- `ablation_analysis`: 系统性消融 + 证据记录
- `ensemble_selection`: 基于证据的互补模型选择

**策略合规战术** (tau-bench):
- `gather_state`: 系统性收集客户状态
- `verify_policy`: 显式验证适用策略
- `execute_with_evidence`: 带证据依据的行动执行
- `error_recovery`: 追溯错误源头 + 修正

#### D. Format Engine

基于FSM状态自动选择最优推理格式（Cohen's d = 1.58的研究发现）:

| FSM状态 | 推理格式 | 理由 |
|---------|---------|------|
| ORIENT | NL CoT | 理解任务需要语义推理 |
| HYPOTHESIZE | NL CoT + Structured JSON | 假设需要自然语言，但注册需要结构化 |
| EXPERIMENT | Program-of-Thought + Tool calls | 执行需要代码和工具 |
| OBSERVE | Structured Table | 结果需要结构化记录 |
| EVALUATE | Tactic Format | 证据评估需要原子步骤声明 |
| CONCLUDE | NL CoT + Evidence Chain | 综合最终答案需要语义 + 证据 |

### 3.4 ER组件 → Benchmark优势映射

| 组件 | BixBench | HeurekaBench | ScienceAgentBench | MLE-bench | tau-bench |
|------|----------|-------------|-------------------|-----------|-----------|
| **Hypothesis Manager** | 生物假设驱动数据分析 | 驱动insight生成周期 | 编码前假设数据结构 | 假设驱动模型选择 | 追踪客户场景适用性 |
| **Evidence Evaluator** | 评估统计结果vs生物假设 | 聚合13篇论文的证据 | 评估代码输出vs预期 | 因果评估模型性能 | 验证策略合规证据 |
| **CMM Provenance** | 追溯多步分析错误来源 | 跨论文知识图谱 | 调试历史防止重复错误 | 实验日志防止冗余 | 对话状态完整追踪 |
| **CMM 压缩** | 20+步轨迹不丢失上下文 | 44GB数据策略性处理 | 跨调试周期维持上下文 | 跨75个竞赛维持日志 | 跨多轮维持策略状态 |
| **Domain Tactics** | 68个生信工具编排 | Biomni数据库战术 | 多学科EDA/建模/调试 | ML实验方法论 | 策略合规检查流程 |

---

## 四、预期性能目标

### 核心论文主张 (Paper Claim)

> "Experimental Reasoning scaffold consistently improves agent performance across multiple model families and task domains, demonstrating that structured hypothesis-evidence-tactic reasoning is a model-agnostic architectural contribution."

### 评估矩阵设计 (Model × Benchmark)

论文需展示：**同一个ER scaffold，接入不同模型，在不同benchmark上均有提升**。

**测试模型族** (至少3个，展示模型无关性):
- **Claude Sonnet 4.x** — agentic任务表现强，代表Anthropic系
- **GPT-4o / GPT-5** — 代表OpenAI系，工具调用能力强
- **Gemini 2.5 Pro** — 代表Google系，长上下文优势
- (可选) **DeepSeek-R1 / Qwen3-235B** — 代表开源系

**预期结果矩阵** (ER scaffold vs. 各模型的vanilla/ReAct基线):

| Benchmark | 基线方法 | Claude+ER vs Claude基线 | GPT+ER vs GPT基线 | Gemini+ER vs Gemini基线 | 平均提升 |
|-----------|---------|----------------------|-------------------|----------------------|---------|
| **BixBench** | Zero-shot/ReAct | +4-8pp | +3-7pp | +4-8pp | +4-7pp |
| **HeurekaBench** | Generic agent | +6-11pp MCQ | +5-9pp | +5-10pp | +5-10pp |
| **ScienceAgentBench** | Self-debug | +4-8pp | +3-7pp | +4-8pp | +4-7pp |
| **MLE-bench** | AIDE baseline | +3-6pp | +2-5pp | +3-6pp | +3-6pp |
| **tau-bench airline** | ReAct agent | +6-12pp | +5-10pp | +5-10pp | +5-10pp |

**关键**: 不追求在每个benchmark上击败绝对SOTA（那些SOTA往往使用特定模型+特定优化），而是证明 **ER scaffold对任何模型都能带来consistent improvement**。如果某些组合同时超越了SOTA，那是额外加分。

### 保守估计与SOTA对比

| Benchmark | 当前SOTA | SOTA系统 | ER+最佳模型目标 | 对比 |
|-----------|---------|---------|---------------|------|
| **BixBench** | 34.4% | K-Dense (10 agents) | 38-42% | 有望超越 |
| **HeurekaBench MCQ** | 44% | Claude-4-Sonnet | 50-55% | 有望超越 |
| **ScienceAgentBench** | 42.2% | o1-preview (>10x cost) | 46-50% | 可能超越，且成本更低 |
| **MLE-bench** | 43.9% | MLE-STAR (成熟scaffold) | 46-50% | 接近或持平 |
| **tau-bench airline** | 56% | o4-mini | 62-68% | 有望超越 |

### 估计依据

- **BixBench +4-8pp**: context engineering可恢复10-15pp (研究发现)，减去实现开销。K-Dense的10-agent通信退化是已确认的结构问题
- **HeurekaBench +6-11pp**: 56pp headroom，结构化发现周期直接映射benchmark设计意图
- **ScienceAgentBench +4-8pp**: `agent.py`的盲目重试是最弱环节，假设驱动调试是直接结构化替代
- **MLE-bench +2-6pp**: MLE-STAR已是成熟scaffold，ER的增量来自实验日志防冗余
- **tau-bench +6-12pp**: 策略合规追踪是新能力，直接修复已识别的主要失败模式

### 论文核心图表规划

1. **Table 1**: Model × Benchmark improvement matrix (main result)
2. **Table 2**: ER vs. SOTA systems per benchmark
3. **Figure 1**: ER架构图
4. **Figure 2**: 消融实验 — 逐个移除ER组件的影响
5. **Figure 3**: 案例研究 — ER的假设演化轨迹 vs. ReAct的平坦执行
6. **Table 3**: Token效率对比 — ER vs. brute-force approaches (ScienceAgentBench)

---

## 五、实现路线图（通用性优先）

**策略调整**: 不再逐个benchmark串行开发，改为 **通用核心先行 + 早期多benchmark验证 + 迭代扩展**。

### Phase 0: 通用ER核心 (Week 1-3) — 零benchmark特定代码
- **P0.1** CMM模块 — 扩展 `life-long-memory` SQLite schema（通用的hypothesis/evidence/provenance表）
- **P0.2** Hypothesis Manager — 通用数据类 + 优先队列 + 生命周期（不含任何领域知识）
- **P0.3** ER循环引擎 — 通用状态机 ORIENT→HYPOTHESIZE→EXPERIMENT→OBSERVE→EVALUATE→CONCLUDE
- **P0.4** Format Engine — 基于FSM状态的格式选择（通用规则）
- **P0.5** Environment Adapter接口 — `execute(action) -> observation` 抽象接口
- **P0.6** Model Adapter接口 — 统一API支持Claude/GPT/Gemini切换

### Phase 1: 早期双benchmark验证 (Week 4-5) — 验证通用性
同时适配2个不同领域的benchmark，用同一个ER核心，**不修改核心代码**：
- **P1.1** BixBench环境适配器 (Docker + Jupyter) — 科学/生信领域
- **P1.2** tau-bench环境适配器 (API tool calling) — 非科学领域
- **P1.3** 通用战术库（仅通用战术：explore_data, form_hypothesis, execute_test, evaluate_evidence, debug, synthesize）
- **P1.4** 用Claude + GPT两个模型族分别在两个benchmark的dev set上运行
- **P1.5** 验证核心问题：ER循环是否对两个完全不同的领域都有效？是否模型无关？
- **里程碑**: 两个benchmark均观察到vs基线的改善（即使很小），且两个模型均有效

### Phase 2: 领域战术扩展 (Week 6-8) — 可插拔领域知识
在通用核心不变的前提下，添加可插拔的领域战术模块：
- **P2.1** 生信战术模块 (BixBench) — 利用68个现有工具
- **P2.2** 策略合规战术模块 (tau-bench) — 策略状态追踪
- **P2.3** 科学数据分析战术模块 (ScienceAgentBench) — EDA + 假设编码 + 假设调试
- **P2.4** ScienceAgentBench环境适配器 — 替换 `agent.py` 的暴力重试
- **P2.5** 三个benchmark的dev set全面评估 + 失败模式分析
- **里程碑**: 3个benchmark验证；领域战术提供额外提升（vs仅通用战术）

### Phase 3: 规模扩展 (Week 9-11) — 补全benchmark组合
- **P3.1** MLE-bench环境适配器 + ML实验战术
- **P3.2** HeurekaBench/Biomni环境适配器 + 科学发现战术
- **P3.3** 第三个模型族(Gemini)接入评估
- **P3.4** 5个benchmark的dev set全面评估
- **里程碑**: 5 benchmarks × 3 models 的完整评估矩阵

### Phase 4: 全量评估 + 消融 + 论文 (Week 12-16)
- **P4.1** 5个benchmark全量运行（非dev set）
- **P4.2** 消融实验设计与执行：
  - ER-Full vs ER-NoHypothesis vs ER-NoProvenance vs ER-NoTactics vs ER-NoFormat vs Vanilla ReAct
  - 每个消融配置 × 3模型 × 5benchmark
- **P4.3** Token效率分析（ER vs brute-force approaches）
- **P4.4** 案例研究：展示ER假设演化轨迹
- **P4.5** 论文撰写

---

## 六、关键可复用代码

| 组件 | 来源路径 | 复用方式 |
|------|---------|---------|
| CMM数据库 | `/home/lingzhi/Code/1-Context-Engineering/life-long-memory/src/db.py` | 扩展SQLite schema加入hypotheses/evidence/provenance表 |
| FSM引擎 | `/home/lingzhi/Code/3-Action/finite-state-agent/src/core/machine.py` | 改造为证据条件状态转移的ER循环 |
| 战术目录 | `/home/lingzhi/Code/3-Action/chain-of-action/src/core/action_type.py` | ActionCatalog模式用于领域战术注册 |
| 生信工具 | `/home/lingzhi/Code/3-Reasoning/tactical-reasoning/Tools/` | 68个工具直接用于BixBench/HeurekaBench |
| ToolUniverse | `/home/lingzhi/Code/6-Bioinformatics/skill-bio/Github/ToolUniverse/` | MCP集成的生信API |
| 战术格式 | `/home/lingzhi/Code/3-Reasoning/tactical-reasoning/design/reasoning/reasoning_tactic_prompt/` | 适配为Format Engine的EVALUATE状态格式 |
| BixBench参考 | `/home/lingzhi/Code/7-Benchmark/data/BixBench/bixbench/zero_shot.py` | 环境适配器参考 |
| ScienceAgent参考 | `/home/lingzhi/Code/7-Benchmark/data/ScienceAgentBench/agent.py` | 替换目标：暴力重试循环 |
| tau-bench参考 | `/home/lingzhi/Code/7-Benchmark/data/tau-bench/tau_bench/agents/` | Agent.solve()接口适配 |

---

## 七、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| LLM无法遵循ER循环结构 | 中 | 高 | JSON schema强制输出格式；系统（非LLM）提取结构化观察 |
| 长任务上下文耗尽 | 高 | 高 | CMM三级压缩(L1/L2/L3)；预算感知战术选择 |
| ER开销在简单任务上降低性能 | 中 | 中 | ORIENT阶段复杂度分类器；简单任务短路到直接执行 |
| 领域战术库不完整 | 中 | 中 | 先建通用战术(EDA/假设/调试/验证)，领域特定增量添加 |
| Benchmark环境集成困难 | 低中 | 中 | 每阶段建一个适配器，渐进式集成；利用现有接口 |
| 评估成本/时间过高 | 高 | 中 | 先dev set小规模；渐进扩展；预算上限；可并行化 |
| 多agent开销(MARBLE 3-agent上限) | 低 | — | ER设计为单agent+CMM，非多agent；避免MARBLE陷阱 |
| 模型无关性不成立 | 中 | 高 | 某些模型可能无法遵循ER结构。Phase 1双模型双benchmark检查点提前暴露问题；对弱遵循模型降级到简化ER |
| 通用核心不够通用 | 中 | 高 | Phase 1就同时验证科学+非科学两个领域；若通用战术不足，说明核心循环需要调整而非添加特定代码 |

---

## 八、验证方案

### 通用性验证（最关键）
1. **Phase 1检查点**: 同一个ER核心（零benchmark特定代码）在BixBench和tau-bench上均观察到改善，且Claude和GPT两个模型族均有效。**如果失败 → 重新审视ER循环设计**
2. **Phase 2检查点**: 添加领域战术后，3个benchmark均有进一步提升。**如果领域战术没有额外收益 → 简化架构**

### 模型无关性验证
3. **每个benchmark运行3个模型**: Claude Sonnet 4.x + GPT-4o/5 + Gemini 2.5 Pro
4. **主张成立条件**: 至少在80%的 (model × benchmark) 组合上，ER > 该模型的ReAct基线
5. **报告不一致性**: 如果某模型在某benchmark上ER反而降低性能，分析原因并报告

### 消融实验
6. 逐个移除ER组件，测量各组件在不同benchmark上的贡献：
   - ER-Full（完整系统）
   - ER-NoHypothesis（移除假设管理，退化为ReAct+CMM）
   - ER-NoProvenance（移除证据链，仅保留观察日志）
   - ER-NoTactics（移除领域战术，仅通用动作）
   - ER-NoFormat（固定NL CoT格式，不切换）
   - Vanilla ReAct（完全移除ER，纯ReAct基线）

### 最终评估
7. 5个benchmark全量评估 + 与Performance.md中SOTA对比
8. Token效率分析：ER vs 暴力重试方法(ScienceAgentBench)的token消耗对比
9. 案例研究：选择3个代表性任务，可视化ER的假设演化轨迹
