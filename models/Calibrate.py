import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class _IsoMap(nn.Module):
    """irova:单调回归的分段常数映射：y = step(x)。boundaries:[M-1], values:[M]"""
    def __init__(self):
        super().__init__()
        self.register_buffer("boundaries", torch.empty(0))
        self.register_buffer("values", torch.empty(0))

    @torch.no_grad()
    def set_params(self, boundaries: torch.Tensor, values: torch.Tensor):
        self.boundaries = boundaries.detach().float().reshape(-1)
        self.values = values.detach().float().reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.values.numel() == 0:
            raise RuntimeError("iROVA 映射尚未拟合（values 为空）")
        idx = torch.searchsorted(self.boundaries, x, right=True)  # [N], 范围 0..M-1
        return self.values[idx]  # 逐样本取所在台阶的常数

class Calibrate_Model(nn.Module):
    """
    calibrate_method:
      - 'ts'      : 温度缩放，forward 返回 logits/T
      - 'ets'     : ETS(三组件)，forward 返回 log(q) 作为 "logits"
      - 'ets_pc'  : ETS + 按类温度（每类一个 T）
    非惰性：选择 ETS 时立即创建 α 参数（需传入类别数）
    """
    def __init__(self, model, T=None):
        super().__init__()
        self.model = model
        self.T = nn.Parameter(torch.tensor(1.0 if T is None else T, dtype=torch.float32))

        # 统一模式开关（默认 TS），对外一律“像 logits”
        self.calib_method = 'ts'
        self._returns_log_prob = False  # 我们统一把输出当 logits 用（ETS 返回 log(q)）

        # ===== ETS 相关（非惰性）=====
        self._etsm_enabled = False
        self._etsm_per_class = False
        self._etsm_eps = 1e-8
        self._etsm_w_logits = None
        self._etsm_alpha_raw_bos = None
        self._etsm_alpha_raw_bom = None
        self._etsm_init_alpha = 1.0
        # 记录类别数，便于校验
        self._num_classes_bos = None
        self._num_classes_bom = None
        
        # ===== iROVA 相关 =====
        self._irova_enabled = False
        self._irova_use_ts = False     # 是否先做 TS（即 logits/T 再 softmax）
        self._irova_eps = 1e-8
        self._irova_bos = None         # nn.ModuleList([_IsoMap]*K_bos)
        self._irova_bom = None         # nn.ModuleList([_IsoMap]*K_bom)
        self._irova_fitted_bos = False
        self._irova_fitted_bom = False


    # ========= 主入口：由 args 驱动，非惰性创建 α ========= 🔧
    def set_calibrate_method(self, method: str,
                             per_class: bool = False,
                             init_T: float = 1.0,
                             num_classes_bom: int = None,
                             num_classes_bos: int = None):
        """
        参数：
          method: 'ts' | 'ets' | 'ets_pc'
          per_class: ETS 是否按类温度
          init_T: ETS 初始 T（α=1/T）
          num_classes_bom/bos: 对应头的类别数（ETS 下必需；TS 可忽略）
        """
        method = (method or 'ts').lower()
              # …先把 ETS 标记清零…
        self._etsm_enabled = False
        self._returns_log_prob = False
        self._irova_enabled = False
        self._irova_use_ts = False
        if method == 'ts':
            self.calib_method = 'ts'
            self._returns_log_prob = False
            self._etsm_enabled = False
            return
        if method in ('ets', 'ets_pc', 'ets-pc'):
            # ETS / ETS_PC
            if num_classes_bom is None and num_classes_bos is None:
                raise ValueError("ETS 需要提供 num_classes_bom 或 num_classes_bos（至少一个头的 K）")

            self._num_classes_bos = num_classes_bos
            self._num_classes_bom = num_classes_bom
            self._etsm_per_class = bool(per_class or method in ('ets_pc','ets-pc'))
            self._etsm_eps = 1e-8
            self._etsm_enabled = True
            self._etsm_init_alpha = 1.0 / max(init_T, 1e-6)

            # 立即创建参数（非惰性） 🔧
            self._create_ets_params()

            self.calib_method = 'ets_pc' if self._etsm_per_class else 'ets'
            self._returns_log_prob = False  # 对外仍然当作 logits 用
            return
        
        if method == 'irova_ts':
            self.calib_method = 'irova_ts'
            self._irova_enabled = True
            self._irova_use_ts = True
            self._num_classes_bos = num_classes_bos
            self._num_classes_bom = num_classes_bom
            if num_classes_bos is not None and self._irova_bos is None:
                self._irova_bos = nn.ModuleList([_IsoMap() for _ in range(num_classes_bos)])
            if num_classes_bom is not None and self._irova_bom is None:
                self._irova_bom = nn.ModuleList([_IsoMap() for _ in range(num_classes_bom)])
            return
        
        if method == 'irova':
            self.calib_method = 'irova'
            self._irova_enabled = True
            self._irova_use_ts = False
            self._num_classes_bos = num_classes_bos
            self._num_classes_bom = num_classes_bom
            if num_classes_bos is not None and self._irova_bos is None:
                self._irova_bos = nn.ModuleList([_IsoMap() for _ in range(num_classes_bos)])
            if num_classes_bom is not None and self._irova_bom is None:
                self._irova_bom = nn.ModuleList([_IsoMap() for _ in range(num_classes_bom)])
            return 
        
    def _infer_device(self):
        # 取已有参数的 device；优先骨干，其次 T
        for p in self.model.parameters():
            return p.device
        return self.T.device
    # ========= 立即创建 ETS 参数（非惰性） ========= 🔧
    def _create_ets_params(self):
        # 3 个混合权重（softmax 约束）
        self._etsm_w_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        device = self._infer_device()
        init_raw = float(np.log(np.exp(self._etsm_init_alpha) - 1.0))  # α = softplus(raw)+eps
        # bos 头
        if self._num_classes_bos is not None:
            if self._etsm_per_class:
                self._etsm_alpha_raw_bos = nn.Parameter(torch.full(
                    (self._num_classes_bos,), init_raw, dtype=torch.float32, device=device))
            else:
                self._etsm_alpha_raw_bos = nn.Parameter(torch.tensor([init_raw], dtype=torch.float32, device=device))
        else:
            self._etsm_alpha_raw_bos = None

        # bom 头
        if self._num_classes_bom is not None:
            if self._etsm_per_class:
                self._etsm_alpha_raw_bom = nn.Parameter(torch.full(
                    (self._num_classes_bom,), init_raw, dtype=torch.float32,device=device))
            else:
                self._etsm_alpha_raw_bom = nn.Parameter(torch.tensor([init_raw], dtype=torch.float32,device=device))
        else:
            self._etsm_alpha_raw_bom = None

    # ========= forward：统一调用，不做惰性初始化 =========
    def forward(self, img1, img2):
        if self.calib_method.startswith('ets') and self._etsm_enabled:
            return self._forward_ets_as_logits(img1, img2)
        
        if self.calib_method.startswith('ts'):

            # TS：返回 logits / T
            logits_bos, logits_bom = self.model(img1, img2)
            out_bos = None if logits_bos is None else logits_bos / self.T
            out_bom = None if logits_bom is None else logits_bom / self.T
            self._returns_log_prob = False
        if self._irova_enabled:  # 覆盖 irova / irova_ts
            return self._forward_irova_as_logits(img1, img2)
        return out_bos, out_bom

    def calibrate_test(self, img1, img2):
        self.eval()
        with torch.no_grad():
            return self(img1, img2)

    # ========= 冻结骨干，只学校准参数 =========
    def pre_fintune(self):
        for p in self.model.parameters():
            p.requires_grad = False

        if self.calib_method == 'ts':
            self.T.requires_grad = True
        elif self.calib_method.startswith('ets'):
            # ETS：w 与 α 可训练（非惰性时必定已存在）
            assert self._etsm_w_logits is not None, "ETS 参数未初始化；请先 set_calibrate_method(..., num_classes_*)"
            self._etsm_w_logits.requires_grad = True
            if self._etsm_alpha_raw_bos is not None:
                self._etsm_alpha_raw_bos.requires_grad = True
            if self._etsm_alpha_raw_bom is not None:
                self._etsm_alpha_raw_bom.requires_grad = True
            # 如需同时学校准全局 T，可打开：
            # self.T.requires_grad = True
        elif self.calib_method.startswith('irova'):
            # iROVA：T 可训练
            self.T.requires_grad = True
            # iROVA 映射不可微，故不训练
            if self._irova_bos is not None:
                for m in self._irova_bos:
                    for p in m.parameters():
                        p.requires_grad = False
            if self._irova_bom is not None:
                for m in self._irova_bom:
                    for p in m.parameters():
                        p.requires_grad = False
    # ========= 其他工具（保持你原来的） =========
    def compile_model(self):
        self.model = torch.jit.script(self.model)

    def grid_search_set(self, T_range, dt):
        self.candidate_T_arange = np.arange(T_range[0], T_range[1] + dt, dt)
        self.T_index = 0

    def reflash_T(self):
        assert self.T_index < len(self.candidate_T_arange), "已超过可选T范围"
        self.T = nn.Parameter(torch.tensor(self.candidate_T_arange[self.T_index], dtype=torch.float32))

    def reset_index(self, T_idx=None):
        self.T_index = T_idx if T_idx is not None else (self.T_index + 1)

    def __len__(self):
        print(f"可选T范围为: {len(self.candidate_T_arange)}")
        return len(self.candidate_T_arange)

    # ========= ETS 计算 =========
    def _etsm_softplus_pos(self, x):  # α>0
        return F.softplus(x) + self._etsm_eps

    def _etsm_ts_prob(self, p, alpha):
        # p^{alpha} / sum p^{alpha}，alpha: 标量或 [K]
        if alpha.ndim == 0:
            a = alpha.view(1, 1)
        else:
            a = alpha.view(1, -1)
        p_pow = torch.clamp(p, min=self._etsm_eps) ** a
        return p_pow / p_pow.sum(dim=1, keepdim=True)

    def _forward_ets_as_logits(self, img1, img2):
        """
        ETS：
          logits -> p -> q = w1*TS(p;α) + w2*p + w3*uniform
          返回 log(q) 作为 "logits"（CE/Focal 直接可用）
        """
        assert self._etsm_enabled, "请先 set_calibrate_method('ets'/'ets_pc', ...)"
        logits_bos, logits_bom = self.model(img1, img2)
        w = torch.softmax(self._etsm_w_logits, dim=0)  # [3]

        out_bos = None
        if logits_bos is not None:
            if self._num_classes_bos is not None:
                assert logits_bos.shape[1] == self._num_classes_bos, \
                    f"bos 类别数不一致：logits:{logits_bos.shape[1]} vs 配置:{self._num_classes_bos}"
            if self._etsm_alpha_raw_bos is not None:
                alpha_bos = self._etsm_softplus_pos(self._etsm_alpha_raw_bos)
                p = F.softmax(logits_bos, dim=1)
                ts = self._etsm_ts_prob(p, alpha_bos)
                uni = torch.full_like(p, 1.0 / p.shape[1])
                q = w[0] * ts + w[1] * p + w[2] * uni
                out_bos = torch.log(q.clamp_min(self._etsm_eps))
            else:
                out_bos = None  # 未配置 bos

        out_bom = None
        if logits_bom is not None:
            if self._num_classes_bom is not None:
                assert logits_bom.shape[1] == self._num_classes_bom, \
                    f"bom 类别数不一致：logits:{logits_bom.shape[1]} vs 配置:{self._num_classes_bom}"
            if self._etsm_alpha_raw_bom is not None:
                alpha_bom = self._etsm_softplus_pos(self._etsm_alpha_raw_bom)
                p = F.softmax(logits_bom, dim=1)
                ts = self._etsm_ts_prob(p, alpha_bom)
                uni = torch.full_like(p, 1.0 / p.shape[1])
                q = w[0] * ts + w[1] * p + w[2] * uni
                out_bom = torch.log(q.clamp_min(self._etsm_eps))
            else:
                out_bom = None  # 未配置 bom

        self._returns_log_prob = False  # 对外当 logits
        return out_bos, out_bom
    
      # ===== iROVA：PAV 拟合一条单调函数，输出边界与台阶值 =====

    @staticmethod
    def _pav_isotonic_fit(x_np: np.ndarray, y_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        PAV 拟合单调函数，返回 (boundaries, values)
        - boundaries: [M-1] 分段点
        - values: [M] 每段的常数值
        输入：x_np∈[0,1]^N（自信度），y_np∈{0,1}^N（是否为该类）
        输出：boundaries:[M-1], values:[M]（非降阶梯）
        """
        x = np.asarray(x_np, dtype=np.float64)
        y = np.asarray(y_np, dtype=np.float64)
        oder = np.argsort(x, kind='mergesort')

        x , y = x[oder], y[oder]
        # 初始化
    
        sum_w = np.ones_like(y)
        sum_y = y.copy()
        mean = sum_y / np.maximum(sum_w, 1e-12)

        # 栈实现 PAV
        sw, sy, sm, L, R = [], [], [], [], []
        for i in range(len(x)):
            sw.append(sum_w[i])
            sy.append(sum_y[i])
            sm.append(mean[i])
            L.append(i)
            R.append(i)
            # 合并
            while len(sm) >= 2 and sm[-2] > sm[-1] + 1e-12:
                w2, y2, m2, l2, r2 = sw.pop(), sy.pop(), sm.pop(), L.pop(), R.pop()
                w1, y1, m1, l1, r1 = sw.pop(), sy.pop(), sm.pop(), L.pop(), R.pop()
                w = w1 + w2; yv = y1 + y2; m = yv / max(w, 1e-12)
                sw.append(w); sy.append(yv); sm.append(m); L.append(l1); R.append(r2)
        
        M = len(sm)
        boundaries = []
        for k in range(M - 1):
            b = (x[R[k]] + x[L[k + 1]]) / 2.0
            boundaries.append(b)
        boundaries = np.array(boundaries, dtype=np.float64)
        values = np.clip(np.array(sm, dtype=np.float64), 0.0, 1.0
        )




        return boundaries, values
    

    @torch.no_grad()
    def fit_irova(self, head: str, logits: torch.Tensor, labels: torch.Tensor, use_temperature: bool = None):
        """
        在验证集上拟合 iROVA 阶梯：
          head: 'bos' 或 'bom'
          logits: [N,K]（该头的 logits）
          labels: [N]（0..K-1）
          use_temperature: 若为 None，随当前模式（irova_ts=True, irova=False）
        """
     
        if use_temperature is None:
            use_temperature = self._irova_use_ts

        N, K = logits.shape
        device = logits.device

        if head.lower() == 'bos':
            if self._irova_bos is None:
                self._irova_bos = nn.ModuleList([_IsoMap() for _ in range(K)])
        
        elif head.lower() == 'bom':
            if self._irova_bom is None:
                self._irova_bom = nn.ModuleList([_IsoMap() for _ in range(K)])
        else:
            raise ValueError("head 必须为 'bos' 或 'bom'")
        
        if use_temperature:
            p = F.softmax(logits / self.T, dim=1)
        else:
            p = F.softmax(logits, dim=1)
        y = labels.long().view(-1).detach().cpu().numpy()


        p_np = p.detach().cpu().numpy()
        for k in range(K):
            x_k = p_np[:, k]
            y_k = (y == k).astype(np.float64)
            bnd, val = self._pav_isotonic_fit(x_k, y_k)
            bnd_t = torch.tensor(bnd, dtype=torch.float32, device=device)
            val_t = torch.tensor(val, dtype=torch.float32, device=device)
            if head.lower() == 'bos':
                self._irova_bos[k].set_params(bnd_t, val_t)
            else:
                self._irova_bom[k].set_params(bnd_t, val_t)
        
        if head.lower() == 'bos':
            self._irova_fitted_bos = True

        else:
            self._irova_fitted_bom = True
    
    def _forward_irova_as_logits(self, img1, img2):
        """
        iROVA：
          logits -> p -> q = iROVA(p)
          返回 log(q) 作为 "logits"（CE/Focal 直接可用）
        """
        logits_bos, logits_bom = self.model(img1, img2)

        out_bos = None
        if logits_bos is not None and self._irova_bos is not None:
            if self._num_classes_bos is not None:
                assert logits_bos.shape[1] == self._num_classes_bos, \
                    f"bos 类别数不一致：logits:{logits_bos.shape[1]} vs 配置:{self._num_classes_bos}"
                
            if not self._irova_fitted_bos or self._irova_bos is None:
                raise RuntimeError("bos 头的 iROVA 尚未拟合，请先调用 fit_irova('bos', ...)")
            

            p = F.softmax(logits_bos / self.T if self._irova_use_ts else logits_bos, dim=1)
            qs = [self._irova_bos[k](p[:, k]) for k in range(logits_bos.shape[1])]
            q_tilde = torch.stack(qs, dim=1)
            denom = q_tilde.sum(dim=1, keepdim=True).clamp_min(self._irova_eps)
            q = q_tilde / denom
            out_bos = torch.log(q.clamp_min(self._irova_eps))

        out_bom = None
        if logits_bom is not None and self._irova_bom is not None:
            if self._num_classes_bom is not None:
                assert logits_bom.shape[1] == self._num_classes_bom, \
                    f"bom 类别数不一致：logits:{logits_bom.shape[   1]} vs 配置:{self._num_classes_bom}"
            if not self._irova_fitted_bom or self._irova_bom is None:
                raise RuntimeError("bom 头的 iROVA 尚未拟合，请先调用 fit_irova('bom', ...)")
            
            p = F.softmax(logits_bom / self.T if self._irova_use_ts else logits_bom, dim=1)
            qs = [self._irova_bom[k](p[:, k]) for k in range(logits_bom.shape[1])]
            q_tilde = torch.stack(qs, dim=1)
            denom = q_tilde.sum(dim=1, keepdim=True).clamp_min(self._irova_eps)
            q = q_tilde / denom
            out_bom = torch.log(q.clamp_min(self._irova_eps))
            
        self._returns_log_prob = False  # 对外当 logits
        return out_bos, out_bom

        

# import torch
# import torch.nn as nn
# import numpy as np

# class TemperatureCalibrator(nn.Module):
#     """
#     通用温度缩放校准层：
#     - mode='learn'：学习（可微），T>0 通过 softplus 约束
#     - mode='grid' ：网格搜索（不可微），内部维护候选 T 列表并逐个取值
#     - 支持两个头（bos/bom）各自拥有 T，也可选择共享
#     """
#     def __init__(self, model, share_T=False, init_T=1.0, mode='learn', eps=1e-6):
#         super().__init__()
#         self.model = model
#         self.mode = mode
#         self.share_T = share_T
#         self.eps = eps

#         # 以 u 参数化：T = softplus(u) + eps 以保证 T>0
#         init_u = np.log(np.exp(init_T - eps) - 1.0) if init_T > eps else 0.0
#         init_u = float(init_u)

#         if share_T:
#             self.u_shared = nn.Parameter(torch.tensor([init_u], dtype=torch.float32))
#             self.u_bos = None
#             self.u_bom = None
#         else:
#             self.u_shared = None
#             self.u_bos = nn.Parameter(torch.tensor([init_u], dtype=torch.float32))
#             self.u_bom = nn.Parameter(torch.tensor([init_u], dtype=torch.float32))

#         # grid 模式相关
#         self._grid_vals_bos = None
#         self._grid_vals_bom = None
#         self._grid_idx = 0

#     def _softplus_T(self, u):
#         return torch.nn.functional.softplus(u) + self.eps  # 标量 Tensor

#     def _get_Ts(self):
#         """返回当前 bos/bom 的标量温度（Tensor），根据是否共享决定。"""
#         if self.share_T:
#             T = self._softplus_T(self.u_shared)
#             return T, T
#         else:
#             T_bos = self._softplus_T(self.u_bos)
#             T_bom = self._softplus_T(self.u_bom)
#             return T_bos, T_bom

#     @torch.no_grad()
#     def compile_model(self, try_script=False):
#         if try_script:
#             try:
#                 self.model = torch.jit.script(self.model)
#             except Exception:
#                 # 按需降级或忽略
#                 pass

#     def forward(self, img1, img2):
#         logits_bos, logits_bom = self.model(img1, img2)

#         if self.mode == 'learn':
#             # 可微学习：使用 softplus(u) 的当前值
#             T_bos, T_bom = self._get_Ts()
#         elif self.mode == 'grid':
#             # 网格搜索：从预先设好的候选表里读取（常量，不需要梯度）
#             assert (self._grid_vals_bos is not None) and (self._grid_vals_bom is not None), \
#                 "请先调用 set_grid(start, stop, num) 设定候选 T"
#             T_bos = torch.tensor([self._grid_vals_bos[self._grid_idx]], 
#                                  dtype=torch.float32, device=logits_bos.device if logits_bos is not None else None)
#             T_bom = torch.tensor([self._grid_vals_bom[self._grid_idx]], 
#                                  dtype=torch.float32, device=logits_bom.device if logits_bom is not None else None)
#         else:
#             raise ValueError(f"Unknown mode: {self.mode}")

#         out_bos = logits_bos / T_bos if logits_bos is not None else None
#         out_bom = logits_bom / T_bom if logits_bom is not None else None
#         return out_bos, out_bom

#     # ====== learn 模式下的便捷方法 ======
#     def freeze_backbone(self):
#         for p in self.model.parameters():
#             p.requires_grad = False
#         # 仅温度相关参数参与训练
#         if self.share_T:
#             self.u_shared.requires_grad = True
#         else:
#             self.u_bos.requires_grad = True
#             self.u_bom.requires_grad = True

#     def temperatures(self):
#         """返回当前数值化后的 (T_bos, T_bom)，用于打印/记录。"""
#         with torch.no_grad():
#             if self.share_T:
#                 T = self._softplus_T(self.u_shared).item()
#                 return T, T
#             else:
#                 return self._softplus_T(self.u_bos).item(), self._softplus_T(self.u_bom).item()

#     # ====== grid 模式下的便捷方法 ======
#     def set_grid(self, start=0.5, stop=3.0, num=26, share_values=True):
#         """
#         设定候选 T 列表。使用 linspace 更稳妥。
#         - share_values=True：两个头用同一组候选；否则可分别设置。
#         """
#         vals = np.linspace(start, stop, num).astype(np.float32)
#         self._grid_vals_bos = vals
#         self._grid_vals_bom = vals if share_values else vals.copy()
#         self._grid_idx = 0

#     def next_T(self):
#         """
#         进入下一组候选 T（仅 grid 模式使用）。
#         返回 False 表示已用尽。
#         """
#         if self.mode != 'grid':
#             return False
#         self._grid_idx += 1
#         if self._grid_vals_bos is None:
#             return False
#         return self._grid_idx < len(self._grid_vals_bos)

#     def grid_len(self):
#         return 0 if self._grid_vals_bos is None else len(self._grid_vals_bos)
