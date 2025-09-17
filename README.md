
## 1. 背景与动机

深度模型常**过度自信**，即置信度不等于真实概率。\*\*校准（Calibration）\*\*旨在尽量不影响准确率的前提下，让预测概率更接近真实概率。

理想校准方法应同时具备：**保持准确率**、**数据高效**、**表达力强**。单一传统方法难兼顾三者；**Mix-n-Match** 通过\*\*集成（Ensemble）**与**组合（Composition）\*\*综合提升。

---

## 2. 传统校准方法

### 2.1 参数化（数据高效、表达有限）

* **温度缩放（TS）**

$$
\sigma_{\mathrm{SM}}(z;T)_k=\frac{\exp(z_k/T)}{\sum_{j}\exp(z_j/T)},\quad T>0.
$$

$T<1$ 分布变尖，$T>1$ 分布变平。

* **向量缩放（VS）**

$$
\sigma_{\mathrm{VS}}(z)_k=\frac{\exp(a_k z_k + b_k)}{\sum_{j}\exp(a_j z_j + b_j)}.
$$

* **矩阵缩放（MS）**

$$
\sigma_{\mathrm{MS}}(z)=\mathrm{softmax}(Wz+b).
$$


### 2.2 非参数化（表达强、数据效率低）

直方图分箱（Histogram Binning）、保序回归（IR）、Beta 校准、BBQ 等。

---

## 3. Mix-n-Match：核心思想

### 3.1 保持准确率的校准映射

对同一样本各分量施加**同一严格单调增**函数 $g$ 后再归一化：

$$
T(z)=\frac{\big(g(z_1),\,\dots,\,g(z_L)\big)}{\sum_{\ell=1}^{L} g(z_\ell)}.
$$

类别排序不变，**不改动 top-1**，从而基本不影响准确率。

### 3.2 集成（Ensemble）

**(a) 模型集成**（把多个保准确率的映射加权）

$$
\mathbf q(z)=\sum_{j=1}^{M} w_j\,T\!\left(z;\theta_j\right),\qquad \sum_{j} w_j=1.
$$

**例：集成温度缩放（ETS）**（在概率空间混合）

$$
\mathbf q_{\mathrm{ETS}}(z;w,t)=
w_1\,\sigma_{\mathrm{SM}}(z/t)+
w_2\,\sigma_{\mathrm{SM}}(z/1)+
w_3\,\tfrac{1}{L}\mathbf 1,
\quad
w_1+w_2+w_3=1.
$$

**(b) 数据集成（IRM）**
把所有类别样本汇总学习**一个**非参数函数，数据效率高；相比 IROvA 表达略弱。

### 3.3 组合（Composition）

先参数化“粗校准”，再非参数化“精修”（如 **IROvA→TS** 或 **TS→IROvA**）。组合通常更强；是否保持准确率取决于具体实现。

---

## 4. 指标：从 ECE 到 KDE-based ECE

### 4.1 经典 ECE

令 $Z$ 为置信度，$\pi(Z)$ 为条件正确率函数：

$$
\mathrm{ECE}_d(f)=\mathbb{E}\big[\|Z-\pi(Z)\|_d^{\,d}\big].
$$

常用**分箱**估计，易受分箱数量与边界影响。

### 4.2 KDE-based ECE（核回归替代分箱）

在 $[0,1]$ 上用核回归估计 $\tilde\pi(z)$ 与 $\tilde p(z)$：

$$
\widetilde{\mathrm{ECE}}_d(f)=\int \big\|z-\tilde\pi(z)\big\|_d^{\,d}\,\tilde p(z)\,dz.
$$

**样本化（LOO 推荐）**
给定 $(c_i,\mathrm{acc}_i)$，Nadaraya–Watson 回归

$$
\hat\pi(c)=\frac{\sum_{i=1}^{n} K\!\big((c-c_i)/h\big)\,\mathrm{acc}_i}{\sum_{i=1}^{n} K\!\big((c-c_i)/h\big)},
$$

并用 LOO 方式计算

$$
\mathrm{KDE\text{-}ECE}=\frac{1}{n}\sum_{i=1}^{n}\big|\hat\pi^{\setminus i}(c_i)-c_i\big|.
$$

### 4.3 Class-wise 与 Multi-class

* **Class-wise**：


在 $\mathcal{I}_k=\{\,i:\hat y_i=k\,\}$ 上，$c_i=\hat p_i^{(k)}$、$\mathrm{acc}_i=\mathbf{1}_{\{y_i=k\}}$，分别得到 $\mathrm{KDE\text{-}ECE}_k$，再加权：

$$
\mathrm{KDE\text{-}ECE}_{\mathrm{cls}}
=\sum_{k=1}^{K}\frac{|\mathcal{I}_k|}{n}\,\mathrm{KDE\text{-}ECE}_k .
$$


### 4.4 带宽与边界

* 带宽 $h$：在 Calib-Learn 上用 5-fold CV 最小化 LOO-NLL 选择；或 Silverman 经验式

  $$
  h_{\mathrm{silver}}=1.06\,\hat\sigma\,n^{-1/5}.
  $$
* 边界：对 $[0,1]$ 建议镜像扩展或 Beta 核以减轻外溢。

### 4.5 可靠性曲线（KDE 版）

在网格 $\{c_g\}_{g=1}^{G}\subset[0,1]$ 上作点 $(c_g,\hat\pi(c_g))$ 与对角线 $y=x$；Class-wise 为每个类一条；Multi-class 为每类作图并可叠加平均。

---

## 5. 实验设计（与你现有基线对齐）

### 5.1 数据与划分

* **Train**：训练主干
* **Fine-tune**：Focal 微调（可与 Train 重合，注意固定超参）
* **Calib-Learn（30k）**：仅拟合 TS / IOP-OI / IROvA
* **Test**：只做最终评估

### 5.2 方法分组

* **B0：CE→TS**
* **B1：Focal($\gamma$)→TS**
* **P-A：Focal→TS→IOP-OI**（保持次序与置换不变，力求不改准确率）
* **P-B：Focal→TS→IROvA**（一对多 IR + 归一化）

> 可选消融：顺序互换（TS↔IOP-OI / TS↔IROvA）、Focal 的 $\gamma\in\{2,3,5\}$ 或 FLSD-53。

### 5.3 关键实现摘要

* **Focal 微调**（多类）

  
  $L_{\mathrm{FL}}=-\sum_{i}\big(1-\hat p_i^{(y_i)}\big)^{\gamma}\,\log \hat p_i^{(y_i)}.$
  
* **温度缩放（TS）**：$$\mathbf z\mapsto \mathbf z/T$$，T 由 Calib-Learn 上的 NLL 选定。
* **IOP-OI**（顺序保持 + 置换不变）：
  每样本对 $\mathbf z$ 降序排序得 $z_{(1)}\ge\dots\ge z_{(K)}$，用**同一**单调函数 $f_\theta$ 映射 $z'_{(r)}=f_\theta(z_{(r)})$，再还原原类别顺序并 softmax；Calib-Learn 上最小化 NLL 学 $\theta$（分段线性单调、LBFGS、平滑正则）。
* **IROvA**：每类 $k$ 拟合单调函数 $g_k$ 近似 $\Pr(y=k\mid s^{(k)})$（$s^{(k)}$ 可用 $z^{(k)}$ 或 margin），再归一化 $\hat q^{(k)}=\hat r^{(k)}/\sum_{j}\hat r^{(j)}$。

### 5.4 指标与统计

* **主指标**：KDE-ECE（overall / class-wise / multi）
* **辅指标**：分箱 ECE（15–20 bins）、NLL、Brier、Top-1
* **显著性**：样本级 bootstrap（1000 次，95% CI），配对差值 CI 不含 0 视为显著

---

## 6. 实务建议（30k 校准集）

* **需保持准确率、数据偏少**：优先 ETS 或 IRM。
* **数据充足、允许轻微准确率波动**：考虑 IROvA-TS 或 IROvA。
* **评估**：优先用 KDE-ECE，配合可靠性曲线与 NLL/Brier。

---

## 7. 对比表

| 方法       | 类型    | 保持准确率  | 数据效率 | 表达力 | 适用场景    |
| -------- | ----- | ------ | ---- | --- | ------- |
| TS       | 参数化   | ✅      | ✅    | 低   | 小数据/基线  |
| ETS      | 集成参数化 | ✅(通常)  | ✅    | 高   | 通用推荐    |
| IRM      | 集成非参  | ✅      | 中-高  | 中   | 数据有限    |
| IROvA    | 非参数化  | （实现相关） | 低    | 高   | 大数据     |
| IROvA-TS | 组合    | （实现相关） | 中    | 高   | 大数据复杂任务 |
| KDE-ECE  | 评估方法  | —      | ✅    | —   | 小样本评估   |

> 备注：若 IROvA 各类共享同一单调函数且仅做概率域保序再归一化，则可保持 top-1；若对每类独立拟合 $g_k$ 再归一化，则可能改变 top-1。

---

## 8. 伪代码（演示逻辑）

**KDE-ECE（LOO, overall）**

```
inputs: confidences c[1..n] in [0,1], accuracy a[1..n] in {0,1}, bandwidth h
for i in 1..n:
  num, den = 0, 0
  for j in 1..n, j != i:
    w = K((c[i]-c[j]) / h)
    num += w * a[j]
    den += w
  pi_hat[i] = num / (den + eps)
KDE_ECE = mean_i |pi_hat[i] - c[i]|
```

**Class-wise KDE-ECE**

```
for each class k:
  I_k = { i : argmax p_i == k }
  compute KDE_ECE_k on (c_i = p_i[k], a_i = 1[y_i==k]) for i in I_k
KDE_ECE_cls = sum_k (|I_k|/n) * KDE_ECE_k
```

**Multi-class KDE-ECE**

```
for each class k:
  pairs = { (p_i[k], 1[y_i==k]) for i in 1..n }
  compute KDE_ECE_k over pairs
KDE_ECE_multi = (1/K) * sum_k KDE_ECE_k
```

**IOP-OI（分段线性单调）**

```
z_sorted, idx = sort_descending(z, dim=1)
z_prime = piecewise_monotone(z_sorted; theta)  # shared, monotone
z_prime = unsort_by(idx, z_prime)
q = softmax(z_prime)
loss = NLL(q, y) + lambda * smooth_reg(theta)
optimize theta (LBFGS/Adam)
```


---

# 📘 基于 KDE 的四种校准评估方法

模型的**置信度 (confidence)** 与**准确性 (accuracy)** 是否一致，是衡量模型是否校准 (calibrated) 的关键指标。
理想情况下：


$P(y = \hat y \mid c = z) = z, \quad \forall z \in [0,1]$

也就是预测置信度 $z$ 应等于真实精度。
传统的 **ECE (Expected Calibration Error)** 依赖直方图分桶，结果会受 bin 边界影响；而 **KDE (Kernel Density Estimation)** 能平滑估计概率分布，更稳健。

---

## 1️⃣ Pred-class-wise ECE（预测类口径）

### 定义

对每个类别 $k$，只考虑 **被预测为该类** 的样本：

$$
\mathcal{I}_k^{pred} = \{ i \mid \hat y_i = k \}
$$

* 横轴：该类的预测概率

  
  $c_i = p_i^{(k)}$

* 纵轴：是否预测正确

  $a_i = \mathbf{1}[y_i = k]$

### KDE 估计

$$
\pi_k(z) = \frac{\sum_{i \in \mathcal{I}_k^{pred}} a_i K_h(z - c_i)}{\sum_{i \in \mathcal{I}_k^{pred}} K_h(z - c_i)}
$$

### 误差

$$
\mathrm{ECE}_k = \mathbb{E}_{i \in \mathcal{I}_k^{pred}} \Big| \pi_k(c_i) - c_i \Big|
$$

返回每个类别的 ECE 值。

**特点**

* 每类一条曲线（K 条线绘制在同一图上）。
* 适合诊断：哪些类过置信 / 欠置信。

---

## 2️⃣ True-class-wise ECE（真实类口径）

### 定义

对每个类别 $k$，考虑 **所有样本**，分析该类概率分布是否合理：

* 横轴：预测的该类概率

  $c_i = p_i^{(k)}$
  
* 纵轴：是否真实属于该类

  $a_i = \mathbf{1}[y_i = k]$

### KDE 估计

$$
\pi_k(z) = \frac{\sum_{i=1}^N a_i K_h(z - p_i^{(k)})}{\sum_{i=1}^N K_h(z - p_i^{(k)})}
$$

### 误差

$$
\mathrm{ECE}_k = \mathbb{E}_i \Big| \pi_k(p_i^{(k)}) - p_i^{(k)} \Big|
$$

**特点**

* 每类一条曲线（K 条线）。
* 是 **one-vs-rest** 视角，反映真实类的概率刻度是否合理。
* 可用于发现某些类系统性低估或高估。

---

## 3️⃣ Multiclass ECE（多类整体口径）

### 定义

对每个样本 $i$，同时考虑它的所有类别概率：

$$
\text{gap}_i = \sum_{k=1}^K \Big| \pi_k(p_i^{(k)}) - p_i^{(k)} \Big|
$$

### 误差

$$
\mathrm{ECE}_{multi} = \frac{1}{N}\sum_{i=1}^N \text{gap}_i
$$

$$
\mathrm{MCE}_{multi} = \max_i \text{gap}_i
$$

**特点**

* 输出一个全局指标（无曲线）。
* 定量衡量整体校准水平。
* 不区分类别，适合整体比较模型效果。

---

## 4️⃣ Overall (Top-1) ECE（整体 top-1 口径）

### 定义

只看每个样本的 top-1 概率：

$$
c_i = \max_k p_i^{(k)}, \quad a_i = \mathbf{1}[\hat y_i = y_i]
$$

* 横轴：top-1 置信度
* 纵轴：是否预测正确

### KDE 估计

$$
\pi(z) = \frac{\sum_{i=1}^N a_i K_h(z - c_i)}{\sum_{i=1}^N K_h(z - c_i)}
$$

### 误差

$$
\mathrm{ECE}_{overall} = \frac{1}{N} \sum_{i=1}^N \Big| \pi(c_i) - c_i \Big|
$$

$$
\mathrm{MCE}_{overall} = \max_i \Big| \pi(c_i) - c_i \Big|
$$

**特点**

* 只画一条曲线（最经典的 reliability diagram）。
* 与标准 ECE 定义最接近。
* 适合快速评估整体校准。

---

## 📊 总结对比

| 方法              | 曲线数量  | 样本选择              | 适用场景         |
| --------------- | ----- | ----------------- | ------------ |
| Pred-class-wise | K 条曲线 | 被预测为该类的样本         | 分析预测类是否过/欠置信 |
| True-class-wise | K 条曲线 | 所有样本（one-vs-rest） | 分析真实类的概率刻度偏差 |
| Multiclass      | 无（指标） | 每个样本所有类别概率        | 整体全局校准指标     |
| Overall (Top-1) | 1 条曲线 | 每个样本的最大概率（top-1）  | 快速评估整体校准     |

---
