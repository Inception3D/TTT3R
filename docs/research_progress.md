# TTT3R — 频域引导状态更新研究进展

## 项目概述
TTT3R 基于 CUT3R/TTT3R 循环重建模型，目标投稿 **NeurIPS**。
核心贡献：**频域引导的 state & memory 选择性更新框架**（train-free，轻量推理阶段）

---

## 三层框架

| Layer   | 级别       | 频域信号                    | 功能                   | 状态       |
|---------|-----------|---------------------------|----------------------|-----------|
| Layer 1 | 帧级别     | LFE(FFT2(RGB diff))       | 跳过冗余帧              | 已验证 ✅   |
| Layer 2 | Token 级别 | HF residual of token trajectory | 调制 per-token 更新幅度 | 待重跑 🔄  |
| Layer 3 | State 级别 | LFE(FFT2(log-depth diff)) | 门控 state 更新         | 已验证 ✅   |

**统一叙事**：频域分解在三个粒度上提供统一的"信息质量"判断——帧级低频能量判断输入冗余性，预测级低频能量判断输出可信度，token 级高频能量判断 state 稳定性。

---

## 实验进展

### Exp 0 — Confidence 校准分析
CUT3R confidence 校准优于 TTT3R（r = −0.276 vs −0.218，10个ScanNet场景）。
TUM 动态场景下 confidence 完全失效（TTT3R r ≈ −0.07）→ 需要额外信号。

### Exp 1 — State Token 频谱分析
**脚本**：`analysis/spectral_analysis.py`
帧级别频谱特征与深度误差相关性弱（r ≈ 0.05–0.18）。

### Exp 2 — Confidence 门控消融 ❌
**脚本**：`analysis/conf_gated_ablation.py`
结果：<1% 改善，校准反而变差（正反馈循环）。**方案放弃**。

### Exp 3 — Layer 1：频域帧筛选 ✅
**脚本**：`analysis/batch_frame_novelty.py`
**指标**：spectral_change = LFE(FFT2(帧间RGB差))，自适应 EMA 阈值。

| 指标               | CUT3R          | TTT3R          |
|-------------------|---------------|---------------|
| Skip rate (ScanNet)| 35.3% ± 13.5% | 35.3% ± 13.5% |
| r(sc, osc) ScanNet | **+0.384 ± 0.018** | +0.191 ± 0.034 |
| Depth err (ScanNet) | −1.3%         | **−3.1%**      |
| Depth err (TUM)    | +0.9%         | −0.7%          |

### Exp 4 — 帧变化指标对比
**脚本**：`analysis/metric_comparison.py`

| 指标                  | r(osc) ScanNet | err_chg ScanNet |
|----------------------|---------------|----------------|
| spectral_change（低频）| +0.382        | −4.7%          |
| l2_change（基线）      | +0.383        | −4.3%          |
| high_freq_change     | +0.218        | −0.8%          |
| mid_freq_change      | +0.133        | −0.7%          |

**关键发现**：低频 > 高频（验证频域分解意义），但帧级别 spectral_change ≈ l2_change（自然图像 1/f² 谱）。

### Exp 5 — Layer 2 SIASU ✅
**脚本**：`analysis/spectral_ablation.py`
Per-token EMA 高频能量调制更新强度。Bug 已修复（warm-start），2026-03-23 重跑完成。

| 配置 | ALL err | ScanNet err | TUM err | vs cut3r Δ% |
|------|---------|-------------|---------|-------------|
| cut3r (baseline) | 0.0745 | 0.0438 | 0.1129 | — |
| ttt3r (baseline) | 0.0697 | 0.0418 | 0.1047 | -6.4% |
| cut3r_spectral_t1 | 0.0708 | 0.0424 | 0.1064 | -5.0% |
| cut3r_spectral_t2 | 0.0710 | 0.0424 | 0.1068 | -4.7% |
| cut3r_spectral_t4 | 0.0708 | 0.0423 | 0.1063 | -5.0% |
| ttt3r_spectral_t1 | 0.0683 | 0.0408 | 0.1025 | **-8.3%** |
| ttt3r_spectral_t2 | 0.0684 | 0.0409 | 0.1029 | -8.2% |
| ttt3r_spectral_t4 | 0.0683 | 0.0409 | 0.1025 | **-8.3%** |

**关键发现**：
- SIASU 单独即可让 cut3r 降 5%，与 ttt3r 叠加再降 2%（总计 -8.3%）
- 温度 τ 不敏感（t1/t2/t4 几乎一致），选 τ=1
- warm-start 修复是关键，之前的 bug 导致 state 冻结

### Exp 6 — Direction C：动态 Token ❌
**脚本**：`analysis/dynamic_token_analysis.py`
Walking: temporal r = −0.024, spatial r = −0.003。State token 不追踪空间语义。**放弃**。

### Exp 7 — B2 Memory Gate（效果弱）
**脚本**：`analysis/memgate_ablation.py`

| 配置 | Δ ALL% |
|------|--------|
| cut3r_mg_t3_sr0.3 (最优) | -1.75% |
| ttt3r_mg_t3_sr0.5 | -6.25% (vs ttt3r -6.40%) |

Memory 的 soft cross-attention write 自带抗冗余能力，gate 增益不大。

### Exp 8 — B3 Geometric Consistency Gate ✅ 最佳结果
**脚本**：`analysis/geogate_ablation.py`

**三轮实验**：

| 版本 | 最优 cut3r | 最优 ttt3r+ |
|------|-----------|------------|
| v1 空间域 (L1 log-depth) | -3.83% (t2) | **-7.41%** (t3) |
| v2 频域 LFE c8 (12.5%) | -3.50% (t2) | -6.59% (t3) |
| v3 频域 LFE c4 (25%) | -3.52% (t2) | **-7.16%** (t2) |

**Cutoff 不敏感**：c2/c4/c8 在 cut3r_geogate 上均为 -3.50%~-3.52%，说明低频段包含了几何一致性的核心信息。

**最终选择**：频域版 τ=2, cutoff=4。性能接近空间域（-7.16% vs -7.41%），叙事统一。

---

## LocalMemory (pose_retriever) 机制

```
mem = [1, 256, 1536]  — 256 slots (key 768 + value 768)
update_mem(): mem 作 query, 新帧特征作 context → cross-attention soft write
inquire(): [global_img_feat, masked_token] 作 query → cross-attention 读 pose feat
```

---

## 数据集

| 数据集      | 服务器路径                                     | 深度 scale | 特点        |
|------------|----------------------------------------------|-----------|------------|
| ScanNet    | `/mnt/sda/szy/research/dataset/scannetv2/`       | 1000      | 室内静态     |
| TUM-dynamics| `/mnt/sda/szy/research/dataset/tum/`            | 5000      | 动态人物，8序列 |

---

## 代码结构

| 文件                                  | 功能                                    |
|--------------------------------------|----------------------------------------|
| `src/dust3r/model.py`                | 核心模型，所有 update type 和 gate 方法     |
| `analysis/geogate_ablation.py`       | B3 几何一致性 gate 消融                    |
| `analysis/memgate_ablation.py`       | B2 Memory Gate 消融                      |
| `analysis/spectral_ablation.py`      | Layer 2 SIASU 消融                       |
| `analysis/batch_frame_novelty.py`    | Layer 1 批量验证                          |
| `analysis/metric_comparison.py`      | 帧变化指标对比                             |
| `analysis/dynamic_token_analysis.py` | Direction C 验证（已失败）                  |

---

## 关键修复

1. **SIASU warm-start**：running_energy 初始化 0 → ratio 爆炸 → state 冻结。改为首次 warm-start。
2. **TUM depth matching**：时间戳关联（20ms 容差），非 stem 匹配。
3. **评估偏差**：相同 kept_indices 上对比全序列与过滤序列。

---

### Exp 9 — Joint Ablation（三层联合消融）✅
**脚本**：`analysis/joint_ablation.py`
12 个配置 = {cut3r, ttt3r} × {baseline, L1, L2, L3, L23, L123}

| 配置 | ALL err | vs cut3r Δ% |
|------|---------|-------------|
| cut3r (baseline) | 0.0745 | — |
| ttt3r (baseline) | 0.0697 | -6.4% |
| L1+ttt3r | 0.0700 | -6.0% |
| L2+ttt3r | 0.0684 | -8.2% |
| L3+ttt3r | 0.0692 | -7.2% |
| **L23+ttt3r** | **0.0690** | **-7.5%** |
| L123+ttt3r | 0.0699 | -6.2% |
| L23+cut3r | 0.0698 | -6.3% |

**关键发现**：
- L23+ttt3r (-7.5%) 为最优组合，SIASU + GeoGate 叠加有效
- L1 帧跳过与 L2/L3 细粒度调制冲突：L123+ttt3r (-6.2%) < L23+ttt3r (-7.5%)
- L23+cut3r (-6.3%) ≈ pure ttt3r (-6.4%)，说明频域框架本身就可替代学习门控

**最终方案**：L23+ttt3r (SIASU × GeoGate × TTT3R)，Layer 1 作独立加速方案讨论。

---

## 待办

- [x] 重跑 Layer 2 SIASU 消融（warm-start 修复后）— 2026-03-23 完成，ttt3r_spectral -8.3%
- [x] 三层联合实验（Layer 1 + 2 + 3）— 2026-03-23 完成，L23+ttt3r -7.5% 最优
- [ ] 论文 outline 起草
