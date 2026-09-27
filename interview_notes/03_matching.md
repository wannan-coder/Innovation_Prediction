# 面试专题 03 · 匹配原理：企业↔专家、企业↔成果

> 用途：回答“企业 business 去匹配专家，到底用哪些字段、怎么算的”“成果匹配是不是一样”这类问题。
> 前置：`01_data.md`（数据是什么）、`02_data_origin_and_cleaning.md`（数据从哪来/怎么洗）。
> 一句话总览：**两个匹配都是“企业查询文本 vs 候选文本”的向量相似度检索；区别是专家匹配额外融合了 TF-IDF 关键词分，成果匹配只用了余弦相似度。**

---

## 1. 匹配总览

平台里有两个独立匹配：

| 匹配 | 输入 | 输出 | 给模型提供什么 |
|---|---|---|---|
| 企业 → 专家 | 企业画像字段 | Top-K 专家 | 专家的 `application` → 作为 `research_field`（研究趋势） |
| 企业 → 成果 | 企业画像字段 | Top-K 成果 | 成果标题+介绍 → 作为 `achievement`（技术抓手） |

两者都是**逐企业处理**：算出一家企业的查询向量，和候选库逐条比相似度，取 Top-K。两家企业的匹配互相独立。

---

## 2. 企业 → 专家 匹配（逐字段拆解）

**对应代码**：`data_process/enterprises_match_experts.py`

### 2.1 用哪些字段

**企业侧（查询）拼成一条 query 字符串**，字段是：

```text
business + category + category_big + category_middle
```

> 注意：专家匹配的企业查询**不含 `org_name`**（企业名）。因为它想把“业务内容”而不是“公司名字”拿去匹配。

**专家侧（候选）拼成一条文本**，字段是：

```text
research_field + application
```

例如：

```text
企业 query：工业机器人研发；高端装备制造；制造业；智能制造设备
专家 文本：工业视觉检测,机器学习,智能机器人；智能制造,质量检测;机器人,自动化产线
```

两边都先过 `normalize_text`（NFKC、合并空白、统一标点）。

### 2.2 第一步：向量语义相似度

1. 用中文句向量模型编码：默认 `shibing624/text2vec-base-chinese`（`SentenceTransformer`）。
2. 专家库向量一次性编码并缓存为 `.npy`（缓存名含模型名 + 文本 MD5，避免错误复用）。
3. 把专家向量和专家向量都做 L2 归一化，再点乘 = 余弦相似度：

```text
sim_sem = cos(企业向量, 专家向量)
```

4. 分块计算：专家库每 4096 条一块搬到 GPU，企业查询每 1024 条一批编码，控制显存。
5. 把余弦从 `[-1, 1]` 映射到 `[0, 1]`：

```text
sim_sem_norm = (sim_sem + 1) / 2
```

### 2.3 第二步：TF-IDF 关键词相似度

为什么加：embedding 擅长“语义相近”，但对具体技术名词（材料名、算法名）不够敏感；关键词命中能补这一块。

做法：

1. 用 `jieba.posseg` 分词，只保留名词/动名词/动词/英文数字词，过滤停用词表和单字词。
2. 对“专家文本”和“企业文本”一起训练 `TfidfVectorizer(min_df=2, ngram_range=(1,1))`。
3. 取企业的 TF-IDF 向量，与全部专家 TF-IDF 向量做点乘（`TfidfVectorizer` 默认 L2 归一化，所以点乘即**TF-IDF 空间余弦**）：

```text
sim_kw = dot(企业TF-IDF向量, 专家TF-IDF向量)     # 再裁剪到 [0,1]
```

### 2.4 第三步：融合成最终分数

```text
final_score = (1 - alpha) * sim_sem_norm + alpha * sim_kw
```

仓库默认：

```text
alpha = keyword_alpha = 0.2
```

即 **80% 语义相似度 + 20% 关键词相关度**。

### 2.5 第四步：阈值 + Top-K

```text
threshold = primary_threshold = 0.73
```

- 只保留 `final_score >= 0.73` 的候选；
- 在满足阈值的候选里取 Top-K（默认 `top_k = 3`）；
- 一个都没达到阈值 → 该企业**没有匹配专家**（返回空）。

> 专家匹配**只有一个阈值**，没有次阈值降级（这一点和成果匹配不同）。

### 2.6 输出

```json
{
  "enterprise_index": 17,
  "matches": [
    {
      "index": 208,
      "score": 0.84,
      "expert": {
        "user_name": "张教授",
        "research_field": "工业视觉检测,机器学习,智能机器人",
        "application": "智能制造,质量检测;机器人,自动化产线"
      }
    }
  ]
}
```

之后 `append_matches_to_inputs.py` 取专家的 `application`，聚合成企业输入里的 `research_field`。

---

## 3. 企业 → 成果 匹配（逐字段拆解）

**对应代码**：`data_process/enterprises_match_achievements.py`

### 3.1 用哪些字段

**企业侧（查询）**字段是：

```text
org_name + business + category + category_big + category_middle
```

> 注意：成果匹配的企业查询**包含 `org_name`**（这点和专家匹配不同）。

**成果侧（候选）**——脚本按以下顺序拼接“存在的字段”：

```text
title + analyse_contect + application + application_field_scenario
+ main_function + main_advantage + scene_label + chain_label
```

> **诚实提醒**：清洗脚本 `achievements_full_clean.py` 把其中一部分列删掉了，且存在字段拼写差异（如 `application` vs 清洗里的 `aplication_field_scenario`）。因此**实际参与匹配的主体往往是 `title + analyse_contect`**。面试时可以说“脚本会拼接存在字段，核心是标题和成果介绍”，不要说死了所有字段都参与。

### 3.2 相似度计算：只用余弦

```text
sim = cos(企业向量, 成果向量)
```

- 同样的句向量模型、归一化、分块计算（成果库每 4096 条一块）。
- **没有** TF-IDF 关键词融合。

### 3.3 阈值：主 + 次双阈值

```text
primary_threshold   = 0.8
secondary_threshold = 0.73
```

- 先在主阈值（0.8）上取 Top-K；
- 如果主阈值下一个都没有，**降级**到次阈值（0.73）再取 Top-K；
- 仍然没有 → 该企业没有匹配成果（返回空）。

### 3.4 输出

```json
{
  "enterprise_index": 17,
  "matches": [
    {
      "index": 326,
      "score": 0.83,
      "achievement": {
        "title": "基于深度学习的工业产品表面缺陷检测方法",
        "analyse_contect": "本成果针对工业零部件表面缺陷检测问题……"
      }
    }
  ]
}
```

之后拼接为输入里的 `achievement` 字段，格式如：

```text
成果1：基于深度学习的工业产品表面缺陷检测方法————本成果针对……
```

---

## 4. 两个匹配的差异对比（面试重点）

| 维度 | 企业→专家 | 企业→成果 |
|---|---|---|
| 企业查询字段 | `business + category + category_big + category_middle`（**不含 org_name**） | `org_name + business + category + category_big + category_middle`（**含 org_name**） |
| 候选字段 | `research_field + application` | `title + analyse_contect`（+ 其余存在字段） |
| 相似度 | 余弦 + **TF-IDF 关键词融合** | **只用余弦** |
| 融合权重 | `0.8 语义 + 0.2 关键词` | 无 |
| 阈值 | 单阈值 `0.73` | 主 `0.8` + 次 `0.73` 降级 |
| Top-K | 3 | 3 |
| 模型 | `shibing624/text2vec-base-chinese` | 同左 |
| 输出字段 | `expert{user_name,research_field,application}` | `achievement{title,analyse_contect}` |

### 为什么两边策略不同（可讲的理由）

- 专家库条目短、术语密集，纯语义容易把“听起来相关但技术方向不同”的人召回，所以叠加关键词分更稳。
- 成果匹配条目较长、包含场景描述，语义信息更充分，先用余弦 + 主阈值，命中不够再降级到次阈值，能在“宁缺毋滥”和“尽量给候选”之间折中。

> 面试官若追问“为什么阈值不一致、能不能统一”——答：当前是经验阈值，两边分数口径不同（专家那侧还经过 `(x+1)/2` 映射和加权），**不能直接比较**；规范做法是在人工标注的匹配集上调参并报告 Precision@K / Recall@K / MRR。

---

## 5. 工程细节（容易被追问）

### 5.1 为什么要缓存 embedding

候选库（专家、成果）变化频率低于企业。缓存后每次只编码新企业查询，省掉重复编码。

### 5.2 为什么要分块 + 分批

- 候选向量常驻 **CPU**，按 4096 条一块搬到 GPU 算相似度 → 显存占用可控。
- 企业按 1024 条一批编码 → 避免一次性把所有企业装进显存。

### 5.3 复杂度与扩展

- 每个企业要对全部候选算一遍，复杂度约 `O(N_企业 × N_候选 × d)`。
- 候选上百万时要用 **FAISS / Milvus / pgvector** 建 ANN 索引，再对小候选做精排（Cross-Encoder / 规则）。

### 5.4 已知技术债

- 两套匹配各自实现，**逻辑重复**，可抽成公共模块。
- 阈值、模型名、批大小**硬编码在脚本里**，应集中到配置。
- 成果匹配的注释声称“向量相似度与文本相关度融合”，但**实际代码只用余弦**（文件名下的 docstring 与实现不一致）。
- 两边分数口径不一致，阈值不可直接比较。

---

## 6. 高频追问与参考回答

**Q1：企业是怎么匹配到专家的？**
> 把企业的 business、category、category_big、category_middle 拼成查询文本，把专家的 research_field、application 拼成候选文本，用中文句向量模型算余弦相似度；同时用 jieba + TF-IDF 算关键词相似度，按 0.8/0.2 融合成最终分，取阈值 0.73 以上的 Top-3。

**Q2：为什么不只用 business？**
> 单靠经营范围容易过窄或表述不规范，叠加行业分类（大/中类）能让查询更稳定、更全面。

**Q3：专家匹配为什么要加 TF-IDF？**
> 语义模型对同义改写友好，但对具体技术名词不够敏感；关键词分能提升术语精确命中，降低“语义像、方向不对”的误召回。

**Q4：成果匹配和专家匹配一样吗？**
> 框架一样（企业查询 vs 候选的向量检索），但成果匹配只用余弦、不融合关键词，并且有主/次双阈值降级；企业查询也多了 org_name。

**Q5：阈值 0.73 / 0.8 怎么来的？**
> 经验值，基于分数分布与抽样。规范做法是用人工标注匹配集调参，报告 Precision@K / Recall@K / MRR。不能说成理论最优。

**Q6：怎么防止匹配错误拖累生成？**
> 阈值 + Top-K 控制候选质量；匹配结果只作为“参考上下文”，最终由大模型综合判断；后续可加 Cross-Encoder 精排和人工校验。

**Q7：两个匹配为什么字段口径不同？**
> 历史演进不同、由不同脚本实现，属于一致性技术债，后续会统一成同一套“查询构造 + 相似度 + 阈值”配置。

---

## 7. 自测题

1. 企业→专家匹配，企业侧用哪几个字段？专家侧用哪几个字段？
2. 企业→成果匹配，企业侧字段和专家匹配有什么不同？
3. 专家匹配的最终分数公式是什么？`alpha=0.2` 代表什么？
4. TF-IDF 分数为什么可以直接用点乘当余弦用？
5. 为什么专家的余弦要先做 `(x+1)/2` 映射？
6. 为什么专家匹配只有一个阈值，而成果匹配有主/次两个阈值？
7. 两个匹配各自默认的阈值和 Top-K 是多少？
8. 匹配结果分别变成模型输入里的哪个字段？
9. 为什么候选向量要放 CPU 分块搬 GPU？
10. 说出至少两个本节提到的技术债。

---

## 附：源码位置

| 主题 | 文件 |
|---|---|
| 企业→专家匹配 | `data_process/enterprises_match_experts.py` |
| 企业→成果匹配 | `data_process/enterprises_match_achievements.py` |
| 匹配结果拼进输入 | `data_process/append_matches_to_inputs.py` |
| 文本归一化 / 缓存 / 分块 | 两个匹配脚本内的 `normalize_text` / `load_model` / `_match_one_enterprise` |
