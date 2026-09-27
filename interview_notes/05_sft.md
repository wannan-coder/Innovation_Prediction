# 面试专题 05 · SFT 篇（原理 + 损失函数 + 数据构造 + 落地调参）

> 本文合并原 05/06/07 三篇，并并入基础概念，一篇讲透 SFT。
> 阅读顺序：第 0 部分先修概念 → 第 1 部分原理 → 第 2 部分本项目实现 → 第 3 部分落地调参 → 第 4 部分面试题。
> 前置：`01_data.md`（数据）、`03_matching.md`（匹配）。
> 本文公式全部用纯文本，任何 Markdown 阅读器都能显示。

---

# 第 0 部分：先修概念（范式 vs 范围、学习率、PEFT）

> SFT 是否高频考点：**会，而且高频必问**，分概念层 / 实现层 / 调优层三种问法，详见第 4 部分。

很多概念混淆，是因为把**两个不同维度**当成一回事。

## 0.1 两个正交维度

**维度 A：训练范式（学什么、用什么数据、什么损失）**

| 范式 | 数据 | 目标 |
|---|---|---|
| 预训练 Pretrain | 海量无标注文本 | 学语言和知识 |
| 继续预训练 CPT | 领域无标注文本 | 领域适应 |
| **SFT 指令微调** | **指令-回答配对** | **学按指令回答** |
| RLHF / DPO | 偏好对 | 对齐人类偏好 |

**维度 B：微调范围（到底改哪些参数）**

| 方式 | 改哪些参数 |
|---|---|
| 全参微调 Full FT | 模型**所有**参数 |
| **PEFT（参数高效微调）** | 只改**少量新增/选定**参数（如 LoRA） |
| 冻结部分层 | 只改后面几层，前面冻结 |

**关键：两个维度是独立组合的。**

```text
“SFT” 说的是维度 A（目标），不决定改哪些参数。
“LoRA” 说的是维度 B（范围），不决定损失和数据。
```

所以“做 SFT”可以有两种实现：

```text
SFT + 全参微调  -> 所有参数都按 SFT 损失更新
SFT + LoRA     -> 底座冻结，只更新 LoRA 小矩阵（本项目）
```

> 这就是“**SFT 是范式，不指定层；更新范围由训练方式决定**”的意思。

## 0.2 SFT 到底改不改模型

**会改，SFT 一定改变模型行为**，区别只是改哪些参数：

| 方式 | 参数变化 | 行为 |
|---|---|---|
| 全参微调 | 所有权重都变 | 模型变了 |
| LoRA | 底座不动，只新增 A、B | 有效权重 `W + ΔW` 变了 → 行为也变了 |

类比：全参=重写整篇文章；LoRA=原文不动、只在旁边贴便签；读者看到的是“原文+便签”，内容确实变了。

> 不能说“LoRA 没改模型”；准确说法是**底座参数冻结，LoRA 参数被训练，模型的有效计算发生了改变**。

## 0.3 学习率

学习率 = **每次沿梯度走多大一步**：`新参数 = 旧参数 - 学习率 × 梯度`。类比下山时每步跨多远。

| 学习率 | 现象 |
|---|---|
| 太大 | 震荡、loss 爆炸、发散、跳过最优 |
| 太小 | 收敛慢、训练久、可能卡在差的解 |
| 合适 | loss 平稳下降，验证指标稳步改善 |

为什么要 warmup + 衰减：训练初期参数远离最优、梯度大，用大学习率容易崩，所以先小后大（warmup）；后期用更小学习率细调收敛。本项目 `warmup_frac=0.1` + 线性衰减 + 梯度裁剪 `1.0`。

| 场景 | 典型学习率 |
|---|---|
| 从头预训练 | 1e-4 ~ 3e-4（配大 batch） |
| 全参 SFT | 1e-5 ~ 2e-5 |
| LoRA | 1e-4 ~ 2e-4 |
| 本项目 | 1e-5（偏保守，欠拟合可上调） |

## 0.4 什么是 PEFT

**PEFT = Parameter-Efficient Fine-Tuning，参数高效微调**：冻结大部分底座，只训练少量新增/选定参数，用小成本逼近全参效果。

好处：省显存/存储、易多任务切换、减少灾难性遗忘、小数据可训。

常见方法（**不止 LoRA 一种**）：

| 方法 | 一句话 |
|---|---|
| **LoRA** | 学低秩增量 `ΔW = B·A`（本项目） |
| QLoRA | 4bit 量化底座 + LoRA，显存更省 |
| Adapter | 层间插入小前馈模块 |
| Prefix-Tuning | 每层学一段虚拟前缀 |
| P-Tuning / Prompt Tuning | 只学输入端虚拟 prompt |
| BitFit | 只训 bias 参数 |
| IA3 | 只学对激活的缩放向量 |

记法：**PEFT 是一大类“只改一点点”的方法，LoRA 是其中最主流的一种。**

## 0.5 常见误解澄清

| 误解 | 正确 |
|---|---|
| “SFT 不改模型” | SFT 一定改行为；LoRA 下改的是新增矩阵，有效权重变了 |
| “PEFT 就是 SFT” | PEFT 是“改哪些参数”，SFT 是“学什么目标”，两回事 |
| “LoRA 是一种损失函数” | 不是；它是参数化方式，损失还是 SFT 的交叉熵 |
| “学习率越大越快越好” | 太大会发散，太小会太慢 |

## 0.6 速记

```text
范式(学什么) = SFT
范围(改哪里) = PEFT 中的 LoRA
学习率(走多大) = 1e-5
```

---

---

# 第 1 部分：SFT 原理

## 1.1 SFT 是什么

**SFT = Supervised Fine-Tuning，监督微调**，也叫**指令微调（Instruction Tuning）**。

> 用“输入（指令）→ 输出（期望回答）”的示范数据继续训练预训练模型，让它学会**按指令回答**。

解决的问题：预训练模型“会说话”但不一定“听话”——会续写文本，但不会稳定地“给答案、守格式、遵指令”。

## 1.2 预训练 vs SFT

**预训练**：在大量文本上预测下一个 token，学“整段文本分布”，拥有语言能力和世界知识。

```text
L_pretrain = - (1/T) * Σ_{t=1..T} log P( token_t | token_<t )
```

**SFT**：给定提示 \(x\)，最大化答案 \(y\) 的条件似然：

```text
L_SFT(θ) = - (1/n) * Σ_{t=1..n} log P_θ( y_t | x, y_<t )

x = 提示(prompt) = system + user
y = 答案(response) = assistant
n = 答案 token 数
```

**它和预训练是同一个损失（交叉熵），只差两点**：数据换成“指令-回答”配对；**只对回答算 loss**。

## 1.3 为什么 SFT 能让模型“听话”

- 知识来自预训练；SFT **不灌新知识，而是重塑条件分布**，把“给定指令该输出什么”的概率抬高。
- 本质是对专家示范的**行为克隆（模仿学习）**：让它模仿“好回答长什么样”。
- 因为只优化示范似然，它**不能直接优化“人更喜欢”**，那是 RLHF/DPO 的事。

> 面试加分句：SFT 主要教**格式、风格、指令遵循**；新知识靠预训练、继续预训练（CPT）或 RAG。

## 1.4 损失函数怎么设计（核心）

### 1.4.1 因果 LM 预测下一个 token

每个位置输出一个词表大小的 logits，损失是“正确 token 的负对数概率”：

```text
loss_t = - log softmax(logits_t)[ 真实的下一个 token ]
```

### 1.4.2 token 级例子

```text
输入 token:  x1  x2  x3 | y1  y2  y3  <eos>
预测对齐:    位置 x3 -> y1,  y1 -> y2,  y2 -> y3,  y3 -> <eos>
```

```text
L = - [ logP(y1|x) + logP(y2|x,y1) + logP(y3|x,y1,y2) + logP(<eos>|x,y1,y2,y3) ] / 4
```

> **分母是回答 token 数（4），不是整段长度**，否则提示越长回答权重被稀释越严重。

### 1.4.3 用 labels + `-100` 实现“对齐 + 屏蔽”

```text
input_ids:  x1  x2  x3 | y1  y2  y3  <eos>
labels   :  -100 -100 -100 | y1  y2  y3  <eos>
                          ^ 从回答开始才算 loss
```

`-100` 是 PyTorch `CrossEntropyLoss` 的默认 `ignore_index`，表示该位置不参与 loss。

### 1.4.4 为什么只对回答算 loss（4 条理由）

1. **任务是条件生成** \(P(y|x)\)，提示是条件不是目标。
2. **避免模型学“复述输入”**（防止学成复读机）。
3. **避免长度偏差**：不同样本提示长度差异大，计入会带偏梯度。
4. **契合数据设计**：system 是模板、user 是检索拼接内容，不是要学的答案。

### 1.4.5 哪些 token 参与 loss

`labels[:prompt_len] = -100` 后，**回答段及其结束符**都参与：

```text
... | y1 y2 y3 <|im_end|>
```

`<|im_end|>` 参与 loss 很重要：教模型“何时停止”，否则生成不收尾。

### 1.4.6 伪代码（面试可手写）

```python
full_text  = render(messages, add_generation_prompt=False)
input_ids  = tokenize(full_text)
prompt_len = len(tokenize(render(messages[:-1], add_generation_prompt=True)))

labels = input_ids.copy()
labels[:prompt_len] = -100          # 屏蔽 prompt

loss = CrossEntropyLoss(ignore_index=-100)(logits[..., :-1, :], labels[..., 1:])
# 实际用 HF：model(input_ids, labels=labels) 内部完成移位与求和
```

## 1.5 SFT 与其他训练阶段的区别

| 阶段 | 目标 | 数据 | loss 范围 | 能力 |
|---|---|---|---|---|
| 预训练 Pretrain | 学语言/知识 | 海量无标注文本 | 所有 token | 会续写 |
| 继续预训练 CPT | 领域适应 | 领域无标注文本 | 所有 token | 懂领域语言 |
| **SFT / 指令微调** | **学指令遵循/格式** | 指令-回答配对 | **只回答 token** | 会按指令答 |
| RLHF / DPO | 对齐人类偏好 | 偏好对/打分 | 偏好目标 | 更符合偏好 |

一句话：**Pretrain 学知识 → SFT 学格式和指令 → RLHF/DPO 学偏好。**

---

# 第 2 部分：本项目的数据与实现

## 2.1 偏好数据 → SFT 数据

偏好数据（`generate_preference_dataset.py`）：

```json
{
  "instruction": "……为该公司制定未来2-3年的研究方向。",
  "input": "企业信息：……研究趋势：……研究成果：……",
  "chosen": "【方向一】……【战略价值】……【技术路线】……",
  "rejected": "【方向一】……（宽泛、不可落地）"
}
```

`preference_to_sft.py` **只取 chosen** → `output`，补上领域 `system`：

```json
{ "system": "<领域系统提示词>", "instruction": "……", "input": "……", "output": "<chosen>" }
```

> **为什么只取 chosen**：当前做的是 SFT（只学“好答案长什么样”）；rejected 已保留，后续可做 DPO。
> **口径**：不能说“已经做了 DPO”。

## 2.2 ChatML 对话模板

Qwen 用 ChatML 风格：

```text
<|im_start|>system
你是……<|im_end|>
<|im_start|>user
（instruction + input）<|im_end|>
<|im_start|>assistant
（output）<|im_end|>
```

- `<|im_start|>` / `<|im_end|>` 是**特殊 token**，各占一个 id；`<|im_end|>` 同时是 eos。
- 用 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=...)` 生成：
  - 训练完整对话：`add_generation_prompt=False`
  - 生成：`True`（末尾补 `<|im_start|>assistant\n`）

**环境坑**：`apply_chat_template` 要 `jinja2>=3.1`；环境受限时手写等价 ChatML 模板，但必须与官方模板逐字核对。

## 2.3 SFTDataset 逐段

对应 `lora_training/dataset.py`：

1. **过滤**：`output` 非空的才保留。
2. **拼消息**：`system` + `user=instruction + "\n" + input` + `assistant=output`。
3. **两次 tokenize**：
   - 完整对话 → `input_ids`（`max_length=3072` 截断）；
   - 只到 assistant 开头（`add_generation_prompt=True`）→ `prompt_len`（定位回答起点）。
4. **mask**：`labels[:prompt_len] = -100`。
5. **边界**：`prompt_len` 超过序列长度则截到序列长度；若整个 label 全是 `-100`（回答被截没）则返回 `None`。

## 2.4 collate / 动态 padding / DataLoader

```python
batch = [x for x in batch if x is not None]     # 过滤被截断样本
input_ids     -> pad_token_id 补齐
attention_mask-> 0 补齐
labels        -> -100 补齐
```

- **动态 padding**：按 batch 内最长样本补齐，省显存。
- `shuffle=True`、`drop_last=True`、`num_workers=0`。
- 配上 `batch_size=1`、`gradient_accumulation_steps=8`（真实 batch=8）。

## 2.5 技术债：训练/评测模板不一致

| 位置 | user 拼法 |
|---|---|
| 训练 `dataset.py` | `instruction + "\n" + input` |
| 评测 `evaluator.py` / `utils.py` | `instruction + "\n\n输入：\n" + input` |

训练和评测看到的输入格式不同，会同时影响 PPL 和生成。**修复**：抽公共 `build_messages()`，训练评测共用。

---

# 第 3 部分：SFT 落地

## 3.1 调了哪些参数（`lora_training/config.yaml`）

| 参数 | 值 | 作用 |
|---|---|---|
| `max_length` | 3072 | 截断长度 |
| `batch_size` | 1 | 单卡 micro-batch |
| `gradient_accumulation_steps` | 8 | 真实 batch=8 |
| `epochs` | 3 | 训练轮数 |
| `learning_rate` | 1e-5 | 学习率 |
| `weight_decay` | 0.01 | 正则，防过拟合 |
| `warmup_frac` | 0.1 | 预热 10% |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `save_steps` / `logging_steps` | 100 / 10 | 保存/日志 |
| `r` / `lora_alpha` / `dropout` | 8 / 16 / 0.05 | LoRA 配置 |
| `target_modules` | q,k,v,o,gate,up,down | 注入哪些层 |
| `eval_steps` | 20 | 验证间隔 |

优化器 **AdamW**、调度 **linear + warmup**、**手写训练循环**、按验证 loss 存 `best`。

## 3.2 怎么调能变好（先判断欠/过拟合）

| 症状 | 处方 |
|---|---|
| 欠拟合（不守格式、loss 高） | ↑lr（1e-5→5e-5/1e-4）、↑epoch、↑r、加 target_modules、查数据质量 |
| 过拟合（loss 极低、输出模板化、泛化差） | ↓epoch、↓lr、↑dropout、↓r、加数据、早停 best |
| OOM | ↑梯度累积、↓max_length、gradient checkpointing |
| 震荡/发散 | 加 warmup、梯度裁剪、↓lr |
| 格式不统一 | 统一模板、清洗数据、补格式样本 |

**方法**：单变量消融 + 留出集评估（验证 loss + 格式遵循率 + 人工抽样），多随机种子报告均值方差。

## 3.3 微调在模型的哪些层

SFT 是**范式**，不指定层；更新范围由训练方式决定。本项目是**参数高效微调**：

```text
冻结整个底座，只在每个 Transformer 层插入 LoRA：
  Attention:  q_proj  k_proj  v_proj  o_proj
  MLP:        gate_proj  up_proj  down_proj
不训: embed_tokens, norm, lm_head
```

## 3.4 为什么本项目选 SFT

1. 任务以**格式与领域表达**为主（要学【方向】【战略价值】【技术路线】协议）。
2. 有**示范数据**（LLM 生成的 chosen）。
3. **无偏好标注/奖励模型**，RLHF 成本高收益不确定。
4. **单卡可训、稳定便宜**。
5. rejected 已保留，**DPO 留作下一步消融**。

## 3.5 为什么不用 RAG？RAG 是不是更好？

先给结论：**这个项目其实已经用了“检索”，只是检索发生在前置的数据构造阶段，而不是生成时的在线 RAG。**

- **RAG**：把外部知识检索出来拼进 prompt，让**模型不改参数**就能用新知识。
- **本项目**：先做「企业↔专家 / 企业↔成果」检索，把匹配到的研究方向与成果拼进输入（`research_field`、`achievement`），再让 SFT 后的模型生成。相当于**“检索增强的输入构造 + SFT”**。

**为什么还要 SFT，而不只做 RAG**：
1. 任务难点是**固定输出协议**（方向/战略价值/技术路线）和领域表达，这属于**行为**问题；RAG 改不了模型行为，base 模型仍会跑偏。
2. RAG 擅长**注入事实、降低幻觉、更新知识**；SFT 擅长**学格式、风格、指令遵循**。两者解决不同问题。
3. 本项目用示范数据做 SFT 最直接；RAG 仍需要一个“会按格式回答”的底座。

**RAG 是不是更好？** 取决于目标：
- 要**新鲜知识 / 可溯源 / 减少幻觉** → RAG 更合适；
- 要**统一格式 / 领域表达** → SFT 更合适；
- **最佳实践是两者结合**，本项目已经是“检索 + 微调”的形态。

> 面试口径：不能说“项目没做检索”——前置的企业↔专家/成果匹配就是检索；从“知识（RAG）vs 行为（SFT）”两个角度解释两者互补即可。

## 3.5 遇到的问题与解决

### 代码中真实存在

| 问题 | 解决 |
|---|---|
| 训练/评测模板不一致 | 抽公共 `build_messages()` |
| assistant 被截断（label 全 -100） | 返回 None，collate 过滤 |
| 空 batch（collate 返回 `{}`） | 训练循环跳过空 batch |
| 梯度范数是近似（忽略交叉项） | 累积完成后对 `p.grad` 算真实范数 |
| 日志 loss 是最后一个 micro-batch | 累加求平均 |
| `drop_last` + 尾累积窗口丢数据 | 正确处理最后一个窗口 |
| `pad=eos` 告警 | 正确传 `attention_mask` |

### 调参常见

| 问题 | 解决 |
|---|---|
| 过拟合（loss→0.02） | 早停 best + dropout + 减 epoch + 补数据 |
| 欠拟合（不守格式） | 提 lr、加 target_modules、增 r |
| 灾难性遗忘 | LoRA + 小 lr + 混通用数据 |
| 生成不停止 | 保证 `<|im_end|>` 参与 loss |

### 工程环境

- `apply_chat_template` 要 jinja2≥3.1 → 升级或手写模板。
- 自写 LoRA 时 bf16/fp32 dtype 不匹配 → forward 里 `.to(x.dtype)`。
- transformers 版本不认识新模型架构 → 换匹配版本。

### STAR 式讲“困难”

> **问题**：训练 loss 降到很低后，生成开始模板化，留出企业上泛化变差。
> **定位**：对比训练/验证曲线，确认过拟合；数据量偏小、epoch 偏多。
> **解决**：按验证集存 best、减 epoch、增 LoRA dropout、补数据。
> **结果**：留出集格式遵循率保持，生成多样性回升。

---

# 第 4 部分：面试题与自测

## 4.1 概念层

**Q：什么是 SFT？和预训练区别？**
> SFT 在“指令-回答”配对数据上监督微调，学按指令回答；预训练在海量无标注文本上预测下一 token。损失形式相同，SFT 只对回答算 loss。

**Q：SFT 和 RLHF/DPO 区别？**
> SFT 只模仿示范答案；RLHF 用偏好数据 + 奖励模型 + PPO；DPO 用偏好对直接优化、无需奖励模型。通常先 SFT 再对齐。

**Q：SFT 能灌新知识吗？**
> 不可靠，主要学格式/风格/指令遵循；新知识靠 CPT 或 RAG。

## 4.2 实现层

**Q：损失函数是什么？**
> 回答部分的条件最大似然（交叉熵）：`L = -(1/n) Σ log P(y_t | x, y_<t)`。

**Q：为什么只对回答算 loss？**
> 任务是 P(y|x)，提示是条件；避免复述输入、避免长度偏差、让梯度集中。

**Q：`-100` 是什么？**
> `CrossEntropyLoss` 的 `ignore_index` 默认值，屏蔽 system/user。

**Q：怎么对齐“预测下一 token”？**
> 位置 t 预测 t+1；把 prompt 位置设 -100，框架内部 logits/labels 各错一位算 CE。

**Q：为什么分母是回答 token 数？**
> 按整段长度归一化会让长提示样本权重变小。

**Q：结束符要算 loss 吗？**
> 要，`<|im_end|>` 参与 loss，模型才学会停止。

## 4.3 调优层

**Q：调了哪些参数？**
> 见 3.1；模型侧 LoRA r=8/alpha=16/dropout=0.05，注入 q/k/v/o + gate/up/down。

**Q：怎么调更好？**
> 先判欠/过拟合，再动对应参数；单变量消融 + 留出集评估。

**Q：微调在哪些层？**
> 冻结底座，只在注意力和 MLP 投影加 LoRA；embedding/norm/lm_head 不训。

**Q：训几个 epoch？**
> 1~3；多了过拟合，表现为输出模板化、泛化差。

**Q：怎么防灾难性遗忘？**
> LoRA、小 lr、混通用数据、只训回答。

**Q：loss 不降 / 训练好但推理差怎么查？**
> 查 lr、数据格式、mask 起点、模板一致性、batch；训练好推理差大概率是模板/拼接不一致。

**Q：SFT 怎么评估？**
> 留出 PPL + 生成人工评审 + 任务指标，不能只看训练 loss。

## 4.4 自测题

1. 一句话定义 SFT，并说出和预训练的异同。
2. 写出 SFT 损失公式并解释每项。
3. 为什么只对回答算 loss？至少三条理由。
4. `-100` 从哪来、干什么？
5. 为什么结束符也要参与 loss？
6. 为什么归一化用回答 token 数？
7. CPT 为什么反而对全序列算 loss？
8. 本项目微调在哪些层？哪些不训？为什么？
9. 用一句话区分 SFT / RLHF / DPO。
10. 训练/评测模板不一致会造成什么后果、怎么修？
11. 欠/过拟合分别调哪些参数？
12. 用 STAR 讲一个你遇到的训练问题。

---

## 附：源码位置

| 主题 | 文件 / 函数 |
|---|---|
| 偏好转 SFT | `data_process/preference_to_sft.py` |
| 偏好数据生成 | `data_process/generate_preference_dataset.py` |
| SFT 数据集与 mask | `lora_training/dataset.py` → `SFTDataset`、`collate_fn`、`get_dataloader` |
| 训练超参数 | `lora_training/config.yaml` |
| 优化器/调度/裁剪/早停 | `lora_training/trainer.py` |
| 评测拼接差异 | `model_eval/evaluator.py`、`model_eval/utils.py` |
