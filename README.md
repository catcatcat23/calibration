
# 分类模型校准方法总结（含 Mix-n-Match 方法）

## 一、背景与动机
深度学习模型常常**过度自信（overconfident）**，输出的置信度并不等于真实概率。  
因此，校准（Calibration）方法的目标是**将预测概率修正为更符合真实分布的概率**，而不改变分类准确率。

理想的校准方法应同时满足：
1. **保持准确率（Accuracy-preserving）**
2. **数据高效（Data-efficient）**
3. **表达能力强（High expressive power）**

传统方法往往无法同时满足三者。论文提出了 **Mix-n-Match 策略**，结合 **集成 (Ensemble)** 和 **组合 (Composition)** 来改进。

---

## 二、传统校准方法

### 参数化方法（Parametric）
参数化方法通过少量参数调整整体分布，**数据高效但表达能力有限**。

- **温度缩放 (Temperature Scaling, TS)**  

```math
\sigma_{\text{SM}}(z; T)_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}, \quad T > 0
````

* 当 $T<1$：拉尖（更自信）
* 当 $T>1$：拉平（更保守）

---

* **向量缩放 (Vector Scaling)**

```math
\sigma_{\text{VS}}(z)_k = \frac{\exp(a_k z_k + b_k)}{\sum_j \exp(a_j z_j + b_j)}
```

---

* **矩阵缩放 (Matrix Scaling)**

```math
\sigma_{\text{MS}}(z) = \text{softmax}(Wz + b)
```

---

### 非参数化方法（Non-Parametric）

非参数化方法灵活，**表达能力强但数据效率低**。

* **直方图分箱 (Histogram Binning)**：分区间取真实正确率。
* **保序回归 (Isotonic Regression, IR)**：学习单调递增的分段函数。
* **Beta 校准 (Beta Calibration)**：用 Beta 分布拟合概率。
* **BBQ (Bayesian Binning into Quantiles)**：贝叶斯组合分箱方法。

---

## 三、Mix-n-Match 方法（论文贡献）

### 1. 准确率保持的校准映射

定义：

```math
T(z) = \frac{(g(z_1), g(z_2), \ldots, g(z_L))}{\sum_{l=1}^L g(z_l)}
```

其中 $g$ 是严格单调递增函数。
➡ 保证了类别顺序不变，因此不会影响分类准确率。

---

### 2. 集成方法（Ensemble）

#### **(a) 模型集成（Model Ensemble）**

```math
T(z) = \sum_{j=1}^M w_j T(z; \theta_j), \quad \sum_j w_j = 1
```

* 提升表达能力。

**实例：集成温度缩放 (Ensemble Temperature Scaling, ETS)**

```math
T(z; w, t) = w_1 T(z; t) + w_2 z + w_3 \frac{1}{L}
```

* 三个分量：

  * 标准 TS（自由温度 $t$）
  * $t=1$：原始预测
  * $t=\infty$：均匀分布
* ETS 在数据高效、保持准确率的同时，**比 TS 表达更强**。

---

#### **(b) 数据集成（Data Ensemble）**

* 针对非参数化方法（如 IR），将所有类别的预测与标签打包一起训练一个校准函数（IRM）。
* 数据效率显著提升。
* 但表达能力略弱于一对多 (IROvA)。

---

### 3. 组合方法（Composition）

* 先用参数化方法做“粗校准”，再用非参数化方法精修。
* 优点：

  * 参数化 → 数据效率高
  * 非参数化 → 表达能力强
* 组合后兼具两者优势。

例如：**IROvA + TS (IROvA-TS)**。

---

## 四、校准性能评估改进

### 1. 传统指标：ECE（Expected Calibration Error）

```math
ECE_d(f) = \mathbb{E}\,\|Z - \pi(Z)\|_d^d
```

常用直方图估计，缺点：

* 对分箱敏感
* 数据效率低

---

### 2. 改进方法：KDE-based ECE

使用核密度估计 (Kernel Density Estimation, KDE) 替代直方图：

```math
\tilde{ECE}_d(f) = \int \|z - \tilde{\pi}(z)\|_d^d \tilde{p}(z)\,dz
```

* 理论上无偏、一致
* 在数据量少时更可靠

---

## 五、实验结果与结论

* **ETS**：在 CIFAR-100、ImageNet 等复杂任务上显著优于 TS。
* **IRM**：比 IROvA 数据效率更高，但表达能力略逊。
* **IROvA-TS**：在大数据集下表现最佳，但不保持准确率。
* **KDE-ECE**：比直方图 ECE 更稳定，尤其在小样本情况下。

### 指导建议：

* **数据少 & 需保持准确率**：推荐 ETS 或 IRM。
* **数据多 & 允许轻微准确率下降**：推荐 IROvA-TS。
* **评估指标**：KDE-ECE 更优，或使用校准增益 (Calibration Gain)。

---

## 六、总结对比表

| 方法       | 类型     | 保持准确率 | 数据效率 | 表达能力 | 适用场景    |
| -------- | ------ | ----- | ---- | ---- | ------- |
| TS       | 参数化    | ✅     | ✅    | 低    | 小数据集    |
| ETS      | 集成参数化  | ✅     | ✅    | 高    | 通用推荐    |
| IRM      | 集成非参数化 | ✅     | 中    | 中    | 数据有限时   |
| IROvA    | 非参数化   | ❌     | 低    | 高    | 大数据集    |
| IROvA-TS | 组合     | ❌     | 中    | 高    | 大数据复杂任务 |
| KDE-ECE  | 评估方法   | -     | ✅    | -    | 小样本评估   |



# 实验设计文档：Focal→TS→IOP‑OI / IROvA，度量使用 KDE‑based ECE

> 目的：在你已有两组基线（**CE→TS** 与 **Focal→TS**）之上，补充更强的后校准（**IOP‑OI** 或 **IROvA**），并将评价指标从传统分箱 ECE 拓展为 **KDE‑based ECE**（含 overall / class‑wise / multi‑class 版本），系统比较在 30k 校准集规模下的效果与稳定性。

---

## 0. 记号与问题设置（统一符号，易于复现实验）
- K 类分类任务；样本 \((x_i, y_i)\)，其中 \(y_i \in \{1,\dots,K\}\)。
- 模型未校准 **logits**：\(\mathbf z_i\in\mathbb R^K\)；softmax 概率：\(\hat{\mathbf p}_i=\mathrm{softmax}(\mathbf z_i)\)，分量 \(\hat p_i^{(k)}\) 表示第 \(k\) 类的预测概率。
- **预测类别**：\(\hat y_i=\arg\max_k \hat p_i^{(k)}\)；**置信度**（top‑1 概率）：\(c_i=\max_k \hat p_i^{(k)}\)。
- **指示变量**：\(\mathbb 1[\cdot]\)；**准确性**（top‑1 是否命中）：\(\mathrm{acc}_i=\mathbb 1[\hat y_i=y_i]\)。
- 校准器（后处理）记 \(g(\cdot)\)：输入 logits 或概率，输出校准后的概率 \(\hat{\mathbf q}_i\)。

---

## 1. 数据与划分
为避免信息泄露，建议如下 4 路划分：
1. **Train**（用于训练主干网络）
2. **Fine‑tune**（用于 Focal Loss 微调，若与 Train 重合亦可，但需固定超参）
3. **Calib‑Learn（= 30k）**：**仅**用于拟合校准器（TS / IOP‑OI / IROvA 的参数）
4. **Test**：仅用于最终比较，禁止参与任何选择

> 若需要对比不同超参（如带宽 h、γ、IOP‑OI 结构等），可在 Calib‑Learn 内部再做 **5‑fold CV** 或 hold‑out（Calib‑Select）用于模型选择，最后在整 30k 上重训被选配置再去 Test 测试。

---

## 2. 实验分组（方法清单）
**已实现基线**（复现留存）：
- **B0：CE→TS**（交叉熵训练 + 全局温度缩放）
- **B1：Focal(γ)→TS**（Focal 微调 + TS）

**新增方法**（本次要比较，统一以 Focal→TS 为起点）：
- **P‑A：Focal→TS→IOP‑OI**（保持 top‑k 的高表达力校准，*不改变准确率*）
- **P‑B：Focal→TS→IROvA**（one‑vs‑all Isotonic + 归一化；可能改变排序与 top‑1）

**可选消融**（小规模即可）：
- 顺序对比：**Focal→IOP‑OI→TS** 与 **Focal→IROvA→TS**（验证顺序敏感性）
- γ 消融：Focal 的 \(\gamma\in\{2,3,5\}\) 或使用 FLSD‑53

> 统一说明：TS/IOP‑OI/IROvA 都只在 **Calib‑Learn(30k)** 上拟合；主干网络在 Test 时完全冻结。

---

## 3. 方法细节与可复用定义

### 3.1 Focal 微调（回顾）
- 多类 Focal Loss：\(\displaystyle L_{\text{FL}}=-\sum_{i}\,(1-\hat p_{i}^{(y_i)})^{\gamma}\,\log \hat p_{i}^{(y_i)}\)。
- 建议：短程微调 5–15 epoch，学习率为原端到端训练的 1/10；\(\gamma=3\) 起步（或 FLSD‑53）。

### 3.2 温度缩放（TS，标量 \(T>0\)）
- 对 logits：\(\mathbf z\mapsto \mathbf z/T\)，\(\hat{\mathbf q}=\mathrm{softmax}(\mathbf z/T)\)。
- 在 Calib‑Learn 上最小化 NLL 选择 \(T\) 或直接网格搜索 \(T\in[0.5,3.0]\)。

### 3.3 IOP‑OI（顺序保持 + 置换不变）实现范式
**目标**：对每个样本的输出向量做**类内保持次序**（不改 top‑k）且**对类别置换不敏感**的可学习映射。

一种工程化可复现的实现（值域单调、类别无关）：
1) **按样本排序**：令 \(s\) 为按值降序排列的 logits：\(z_{(1)}\ge\dots\ge z_{(K)}\)。
2) **共享的单调函数** \(f_\theta: \mathbb R\to\mathbb R\)：用**分段线性单调曲线**（M 段折线）或单调样条表示；用非负斜率参数化（如斜率 = softplus(weight)）。
3) **逐位变换**：\(z'_{(r)}=f_\theta\big(z_{(r)}\big)\)（对所有 r 使用同一个 \(f_\theta\)）。
4) **还原顺序**回到原类别索引；\(\hat{\mathbf q}=\mathrm{softmax}(\mathbf z')\)。
5) **训练目标**：在 Calib‑Learn 最小化 NLL（可加 \(L_2\) 正则与小的 TV 正则使曲线更平滑）。

> 该实现保证：对同一样本的分量做**同一单调函数**，因此天然**保持次序**（IOP）；函数不依赖类别标签、仅依赖数值与“名次”，从而**对类别置换不敏感**（OI）。

**超参建议**：M（折线段）\(\in\{8,16\}\)；正则 \(\lambda\in[10^{-5},10^{-3}]\)；优化器优先 LBFGS（收敛快），最大迭代 100–200。

### 3.4 IROvA（IR one‑vs‑all）实现范式
**思想**：对每类做一维保序回归，再把得到的“类独立概率”归一化为多类分布。

1) 选分数：\(s_i^{(k)}=z_i^{(k)}\) 或 margin \(z_i^{(k)}-\max_{j\ne k}z_i^{(j)}\)。
2) 对每个类 k，拟合单调非降函数 \(g_k\) 使 \(\hat r_i^{(k)}=g_k\big(s_i^{(k)}\big)\approx \Pr(y=k\mid s^{(k)})\)。实现可用 **PAV**（Pool‑Adjacent‑Violators）或分段线性单调网络。
3) 归一化：\(\displaystyle \hat q_i^{(k)}=\frac{\hat r_i^{(k)}}{\sum_{j=1}^K \hat r_i^{(j)}}\)。
4) 以 NLL 最小化为主（可加平滑先验，防止某些区间过拟合）。

> 注意：IROvA **不保证**保持向量内部的相对次序，**可能改变 top‑1**；实验需同步报告 top‑1 是否变化及准确率差异。

---

## 4. KDE‑based ECE（从分箱到无参平滑）

### 4.1 直觉与总体定义
- 经典 **ECE**：把置信度 \(c\) 分箱，比较每箱内“平均准确率”与“平均置信度”的差异；缺点是**依赖分箱**、对阈值敏感。
- **KDE‑based ECE**：利用核密度估计（KDE）在 \([0,1]\) 上**平滑估计**\(\pi(c)=\Pr(\text{正确}\mid \text{置信度}=c)\)，再与 \(c\) 对齐，避免分箱敏感性。

### 4.2 KDE 估计式（overall 版本）
给定 \((c_i,\mathrm{acc}_i)\)（top‑1 置信度与是否命中）：
- 选核函数 \(K(\cdot)\)（常用高斯核）与带宽 \(h>0\)。
- **条件正确率的核回归**（Nadaraya–Watson）：
\[
\hat \pi(c)\;=\;\frac{\sum_{i=1}^n K\!\left(\frac{c-c_i}{h}\right)\,\mathrm{acc}_i}{\sum_{i=1}^n K\!\left(\frac{c-c_i}{h}\right)}.
\]
- **ECE 的核化近似**：用样本分布近似 \(\mathbb{E}[\cdot]\)，
\[
\mathrm{KDE\text{-}ECE}\;=\;\frac{1}{n}\sum_{i=1}^n \Big|\hat\pi(c_i)\;{-}\;c_i\Big|.\quad \text{（建议 LOO 版，见下）}
\]
> **LOO（Leave‑One‑Out）**：为减小内插偏差，在计算 \(\hat\pi(c_i)\) 时从和式中**排除样本 i**；记作 \(\hat\pi^{\setminus i}(c_i)\)。

### 4.3 Class‑wise 与 Multi‑class KDE‑ECE
- **Class‑wise KDE‑ECE**：仅在“**被预测为类 k**”的样本集合 \(\mathcal I_k=\{i:\hat y_i=k\}\) 上，令 \(c_i=\hat p_i^{(k)}\)、\(\mathrm{acc}_i=\mathbb 1[y_i=k]\)，按 4.2 公式各自得到 \(\mathrm{KDE\text{-}ECE}_k\)，最终加权平均：
\[\mathrm{KDE\text{-}ECE}_{\text{cls}}=\sum_{k=1}^K \frac{|\mathcal I_k|}{n}\,\mathrm{KDE\text{-}ECE}_k.\]
- **Multi‑class KDE‑ECE**：把每条样本的**所有类概率**都纳入：构造 \(n\times K\) 组对 \((p_{i}^{(k)},\mathbb 1[y_i=k])\)，对每个类 k 独立做 4.2 的核回归并取平均：
\[\mathrm{KDE\text{-}ECE}_{\text{multi}}=\frac{1}{K}\sum_{k=1}^K\Bigg(\frac{1}{n}\sum_{i=1}^n\big|\hat\pi_k(p_i^{(k)})-p_i^{(k)}\big|\Bigg).\]
> **π 的含义**：\(\pi_k(p)=\Pr(y=k\mid \hat p^{(k)}=p)\)。Class‑wise 关注模型“**认定为 k**”时的可靠性；Multi 关注**对每个类概率刻度**的整体可靠性。

### 4.4 带宽与边界处理
- **带宽 h**：优先用 **5‑fold CV** 在 Calib‑Learn 上最小化 LOO‑NLL 来选 h；备选 Silverman 经验式（对 \([0,1]\) 可先对 \(c\) 做 **logit 变换**）
  \[h_{\text{silver}}=1.06\,\hat\sigma\,n^{-1/5}.\]
- **边界效应**：\(c\in[0,1]\)，高斯核会“溢出”边界；可用 **镜像扩展**（对 \([-1,2]\) 反射）或 **Beta 核** 替代。

### 4.5 可靠性曲线（KDE 版）
- 在网格 \(\{c_g\}_{g=1}^G\subset[0,1]\) 上画 \((c_g,\hat\pi(c_g))\) 与对角线 \(y=x\)。
- Class‑wise 版本：每个 k 一条曲线；Multi 版本：对每个 k 分别画，再叠加平均。

---

## 5. 评价指标与统计显著性
**主指标**：KDE‑ECE（overall / class‑wise / multi）

**辅指标**：
- 分箱 ECE（15 或 20 分箱，便于与已有工作对照）
- NLL、Brier score、Top‑1 准确率
- 可靠性曲线与**覆盖‑宽度**（若做置信区间）

**置信区间/显著性**：
- **有放回 bootstrap**（1,000 次）对 **样本级**重采样，给出各指标 95% CI。
- **配对比较**：报告方法 A−B 的差值分布（如 KDE‑ECE 差），若 CI 不含 0 视为显著。

---

## 6. 统一训练与选择协议（避免“挑模型”）
1) 所有方法的**超参选择**（如 TS 的 T、IOP‑OI 的段数与正则、IROvA 的平滑强度、KDE 的带宽 h）都在 **Calib‑Learn(30k)** 内通过 CV 或 hold‑out 确定；
2) 确定后在整 30k 上**重训校准器**，再在 **Test** 上一次性评估；
3) 每种方法跑 **3 次不同随机种子**（影响优化初值/mini‑batch），汇总均值±标准差，并给 bootstrap CI。

---

## 7. 报告模板（建议小节结构）
**7.1 实验设置**：数据集、划分、主干、Focal 超参、Calib‑Learn 尺度、KDE 细节（核/带宽）。

**7.2 主结果表**（Test 上）：
- 行：B0、B1、P‑A(IOP‑OI)、P‑B(IROvA)
- 列：KDE‑ECE(overall / cls / multi)、ECE(分箱)、NLL、Top‑1、是否改动 Top‑1（✓/✗）

**7.3 可靠性图**：overall 与 per‑class 的 KDE 曲线（可挑若干代表类）。

**7.4 消融与顺序**：Focal γ、顺序互换（IOP‑OI/ IROvA 前后 TS）。

**7.5 讨论**：谁在 30k 校准规模下最稳？IOP‑OI 是否在**不改 top‑1**的前提下优于 TS？IROvA 是否带来更强的刻度修正但伴随的排序风险？

---

## 8. 实施清单（可直接落地）
**步骤 S1：产出校准前的缓存**
- 用主干（或 Focal 微调后主干）在 Calib‑Learn 与 Test 上，缓存 logits 与标签（\(\mathbf z_i, y_i\)）。

**步骤 S2：拟合 TS**（若已完成可复用）
- 网格或 NLL 最优，得到 \(T^*\)。

**步骤 S3：拟合 IOP‑OI**（以 TS 后的 logits 为输入）
- 构造分段线性单调函数 \(f_\theta\)；
- 目标：最小化 NLL（LBFGS，迭代≤200，早停看 Calib‑Select）；
- 输出：参数 \(\theta^*\)。

**步骤 S4：拟合 IROvA**
- 为每类拟合 \(g_k\)（PAV/单调层），并保存归一化方案；
- 输出：\(\{g_k^*\}_{k=1}^K\)。

**步骤 S5：选择 KDE 带宽 h**
- 5‑fold CV 最小化 LOO‑NLL 选 h；并固定核（高斯）与边界处理（镜像或 Beta 核）。

**步骤 S6：在 Test 上评估**
- 计算 KDE‑ECE（overall/cls/multi）+ 其他辅指标；
- 画 KDE 可靠性曲线；
- bootstrap 置信区间 + 配对差值。

---

## 9. 伪代码片段（关键计算）

### 9.1 KDE‑based ECE（LOO 版，overall）
```pseudo
inputs: confidences c[1..n] ∈ [0,1], accuracy a[1..n] ∈ {0,1}, bandwidth h
for i in 1..n:
  num = 0; den = 0
  for j in 1..n, j≠i:
    w = K((c[i]-c[j])/h)
    num += w * a[j]
    den += w
  pi_hat[i] = num / (den + eps)
KDE_ECE = mean_i |pi_hat[i] - c[i]|
```

### 9.2 Class‑wise KDE‑ECE（对每个被预测的类）
```pseudo
for each class k:
  I_k = { i : argmax p_i == k }
  compute KDE_ECE_k using pairs (c_i = p_i[k], a_i = 1[y_i==k]) for i in I_k
KDE_ECE_cls = sum_k (|I_k|/n) * KDE_ECE_k
```

### 9.3 Multi‑class KDE‑ECE（全概率向量）
```pseudo
for each class k:
  pairs = { (p_i[k], 1[y_i==k]) for i in 1..n }
  compute KDE_ECE_k over pairs
KDE_ECE_multi = (1/K) * sum_k KDE_ECE_k
```

### 9.4 IOP‑OI（分段线性单调曲线）训练主循环
```pseudo
# logits_after_TS: z ∈ R^{n×K}
for step in 1..max_iter:
  # 排序与反排序索引 (per-sample)
  z_sorted, idx = sort_descending(z, dim=1)
  z_prime = piecewise_monotone(z_sorted; θ)  # 共享单调函数
  z_prime = unsort_by(idx, z_prime)
  q = softmax(z_prime, dim=1)
  loss = NLL(q, y) + λ * regularizer(θ)
  update θ via LBFGS/Adam
```

### 9.5 IROvA（PAV 实现的 one‑vs‑all）
```pseudo
for k in 1..K:
  s = z[:,k]  or  z[:,k] - max_{j≠k} z[:,j]
  r_k = isotonic_regression_fit(s, 1[y==k])  # 得到单调函数 g_k
# 归一化
def predict_IROvA(z):
  r = [ g_k( score_k(z) ) for k in 1..K ]
  q = r / sum(r)
  return q
```

---

## 10. 复现实验环境与日志
- 固定随机种子（如 3407 / 2025 / 9527），记录：框架版本、GPU/CPU、数据版本哈希、划分清单。
- 保存：TS 的 \(T^*\)、IOP‑OI 的 \(\theta^*\)、IROvA 的 \(g_k^*\)、KDE 的 h；
- 输出：每组合成品概率向量（以便离线复算所有指标）。

---

## 11. 风险与对策
- **IROvA 的排序风险**：同步报告 top‑1 变化比例；若明显下降，考虑先 TS/VS 再 IROvA，或弱化单调函数的自由度（分段更少、加平滑）。
- **KDE 的带宽敏感**：务必用 CV 选 h，并做 **h±20%** 的敏感性分析；对边界密集区域（接近 0/1）建议镜像扩展。
- **数据漂移**：若 Test 与 Calib‑Learn 分布差异大，优先报告 **Class‑wise KDE‑ECE** 及 per‑class 曲线（更能看出受影响的类别）。

---

## 12. 你需要交付的可视化
- 1 张总表 + 2 张 KDE 可靠性图（overall 与 per‑class 合辑）
- 1 张顺序消融（TS↔IOP‑OI / TS↔IROvA）对比条形图（KDE‑ECE）
- 附录给出 h 的 CV 曲线与 IOP‑OI 学到的单调函数示意（折线）

---





