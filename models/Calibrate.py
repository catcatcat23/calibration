import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Calibrate_Model(nn.Module):
    def __init__(self, model, T = None):
        super().__init__()
        self.model = model

        if T is not None:
            self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32))  # 使其成为 nn.Parameter
        else:
            self.T = nn.Parameter(torch.ones(1, dtype=torch.float32))  # 初始化为 1

    def compile_model(self):
        self.model = torch.jit.script(self.model)

    def forward(self, img1, img2):
        logits_output_bos, logits_output_bom = self.model(img1, img2)

        if logits_output_bos is not None:
            output_bos = logits_output_bos / self.T
        else:
            output_bos = None
        if logits_output_bom is not None:
            output_bom = logits_output_bom / self.T 
        else:
            output_bom = None

        return output_bos, output_bom
        
        # return output_bos / self.T, output_bom / self.T   # 进行温度缩放

    def calibrate_test(self, img1, img2):
        self.eval()  
        with torch.no_grad():  
            logits_output_bos, logits_output_bom = self(img1, img2)
        return logits_output_bos, logits_output_bom
    
    def pre_fintune(self):
        for param in self.model.parameters():
            param.requires_grad = False  # 冻结模型参数
        self.T.requires_grad = True  # 确保 T 仍然可训练
    
    def grid_search_set(self, T_range, dt):
        self.candidate_T_arange = np.arange(T_range[0], T_range[1] + dt, dt)
        self.T_index = 0
    
    def reflash_T(self):
        assert self.T_index < len(self.candidate_T_arange), "已超过可选T范围"
        self.T = nn.Parameter(torch.tensor(self.candidate_T_arange[self.T_index], dtype=torch.float32))
    
    def reset_index(self, T_idx = None):
        if T_idx is not None:
            self.T_index = T_idx
        else:
            self.T_index += 1
            
    def __len__(self):
        print(f"可选T范围为: {len(self.candidate_T_arange)}")
        return len(self.candidate_T_arange)
    

        # ======  ETS（三组件凸组合）实现 ======
    def enable_ets_mix(self,
                    per_class: bool = False,   # False: 标量温度；True: 按类温度 α_l
                    init_T: float = 1.0,       # 初始温度（若 per_class=True 则所有类同值）
                    eps: float = 1e-8):
        """
        启用 ETS（三组件：TS + identity + uniform）。不改原 forward 的行为。
        之后使用 forward_ets_mix / calibrate_test_ets_mix。
        """
        self._etsm_enabled = True
        self._etsm_per_class = bool(per_class)
        self._etsm_eps = float(eps)
        # 可训练的权重（3 个），用 softmax 保证在单纯形内
        self._etsm_w_logits = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
        # 温度用 α=1/T 参数化，正约束：α = softplus(a_raw) + eps
        # 按类或标量在第一次前向看到类别数后再惰性创建
        self._etsm_alpha_raw_bos = None
        self._etsm_alpha_raw_bom = None
        self._etsm_init_alpha = 1.0 / max(init_T, 1e-6)
        self._etsm_inited_bos = False
        self._etsm_inited_bom = False

    def _etsm_softplus_pos(self, x):
        return F.softplus(x) + self._etsm_eps

    def _etsm_ts_prob(self, p, alpha):
        """
        概率域 TS：p^{alpha} / sum p^{alpha}
        - p: [N,K]
        - alpha: 标量或 [K]
        """
        if alpha.ndim == 0:
            a = alpha.view(1, 1)
        else:
            a = alpha.view(1, -1)  # [1,K]
        p_pow = torch.clamp(p, min=self._etsm_eps) ** a
        return p_pow / p_pow.sum(dim=1, keepdim=True)

    def _etsm_lazy_init(self, logits, which='bos'):
        """
        惰性创建 α 参数：标量或长度为 K 的向量
        """
        K = logits.shape[1]
        if which == 'bos' and (not self._etsm_inited_bos):
            if self._etsm_per_class:
                self._etsm_alpha_raw_bos = torch.nn.Parameter(
                    torch.full((K,), fill_value=float(np.log(np.exp(self._etsm_init_alpha)-1.0)),
                            dtype=torch.float32)
                )
            else:
                self._etsm_alpha_raw_bos = torch.nn.Parameter(
                    torch.tensor([float(np.log(np.exp(self._etsm_init_alpha)-1.0))], dtype=torch.float32)
                )
            self._etsm_inited_bos = True
        if which == 'bom' and (not self._etsm_inited_bom):
            if self._etsm_per_class:
                self._etsm_alpha_raw_bom = torch.nn.Parameter(
                    torch.full((K,), fill_value=float(np.log(np.exp(self._etsm_init_alpha)-1.0)),
                            dtype=torch.float32)
                )
            else:
                self._etsm_alpha_raw_bom = torch.nn.Parameter(
                    torch.tensor([float(np.log(np.exp(self._etsm_init_alpha)-1.0))], dtype=torch.float32)
                )
            self._etsm_inited_bom = True

    def forward_ets_mix(self, img1, img2):
        """
        ETS 前向：
        1) 得到原 logits -> 概率 p
        2) TS(p; α) + identity(p) + uniform 三者按 w 混合
        3) 返回 log(混合概率) 作为 logits-like
        """
        assert getattr(self, "_etsm_enabled", False), "请先调用 enable_ets_mix(...)"

        logits_bos, logits_bom = self.model(img1, img2)
        w = torch.softmax(self._etsm_w_logits, dim=0)  # [3], >=0 且和为 1
        # 头1（bos）
        out_bos = None
        if logits_bos is not None:
            self._etsm_lazy_init(logits_bos, 'bos')
            alpha_bos = self._etsm_softplus_pos(self._etsm_alpha_raw_bos)  # 标量或 [K]
            p = F.softmax(logits_bos, dim=1)
            ts = self._etsm_ts_prob(p, alpha_bos)
            uni = torch.full_like(p, 1.0 / p.shape[1])
            q = w[0] * ts + w[1] * p + w[2] * uni
            out_bos = torch.log(q.clamp_min(self._etsm_eps))
        # 头2（bom）
        out_bom = None
        if logits_bom is not None:
            self._etsm_lazy_init(logits_bom, 'bom')
            alpha_bom = self._etsm_softplus_pos(self._etsm_alpha_raw_bom)
            p = F.softmax(logits_bom, dim=1)
            ts = self._etsm_ts_prob(p, alpha_bom)
            uni = torch.full_like(p, 1.0 / p.shape[1])
            q = w[0] * ts + w[1] * p + w[2] * uni
            out_bom = torch.log(q.clamp_min(self._etsm_eps))

        return out_bos, out_bom

    def calibrate_test_ets_mix(self, img1, img2):
        self.eval()
        with torch.no_grad():
            return self.forward_ets_mix(img1, img2)

    def ets_mix_pre_fintune(self):
        """
        只训练 ETS 参数（w 与 α），冻结骨干。
        """
        for p in self.model.parameters():
            p.requires_grad = False
        self._etsm_w_logits.requires_grad = True
        if self._etsm_inited_bos and (self._etsm_alpha_raw_bos is not None):
            self._etsm_alpha_raw_bos.requires_grad = True
        if self._etsm_inited_bom and (self._etsm_alpha_raw_bom is not None):
            self._etsm_alpha_raw_bom.requires_grad = True


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
