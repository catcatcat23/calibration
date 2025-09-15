
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


