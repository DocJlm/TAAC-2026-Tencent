# TAAC-2026-Tencent

腾讯广告算法大赛 2026 学术赛道单模型方案。

![Leaderboard result](assets/leaderboard_result.png)

## 比赛结果

| 赛道 | 排名 | 最佳分数 | 最佳提交时间 |
|---|---:|---:|---|
| Academic Track | 360 | 0.827503 | 2026-05-18 12:04:56 |

本仓库开源的是我在比赛中最稳定、线上效果最好的单模型版本。代码基于
`v43.1` / `v37.2` 这条实验线，这也是在大量特征、结构和训练策略实验之后，
表现最可靠的一版。

仓库不包含训练数据、模型 checkpoint 或任何比赛平台私有文件。

## 赛题简介

这个任务是一个大规模广告转化率预测问题。对于每一次候选广告曝光，模型会
收到多种不同类型的信息：

- 用户画像特征；
- 商品 / 广告特征；
- dense 数值特征；
- 四个业务域的用户行为序列；
- 时间戳和时间桶特征；
- 二分类转化标签。

模型需要预测一次广告曝光发生转化的概率，排行榜使用 AUC 作为评价指标。
在真实广告系统中，这类模型就是排序模型：它负责判断某个时刻给某个用户展示
哪条广告更可能产生转化。

## 方法

最终提交模型保留了一个紧凑的统一排序主干，重点建模稳定有效的用户时间信号
和 dense-pair 信号。

主要组件：

- **PCVRHyFormer backbone**：统一处理用户、广告、dense 和序列特征。
- **RankMixer NS tokenizer**：处理非序列的用户 / 广告特征。
- **DensePair compressor**：处理对齐的 dense/int 字段对：
  `62,63,64,65,66,89,90,91`。
- **Exposure time context**：建模当前曝光时间。
- **Multi-resolution exposure time**：同时建模粗粒度和细粒度时间模式。
- **Calendar time embeddings**：建模本地日历结构。
- **Calendar user activity cross**：建模用户在不同时段的活跃模式。
- **User field coverage time context**：建模用户画像完整度以及时间相关的用户侧信号。
- **Raw-AUC checkpoint selection**：按验证集原始 AUC 选择 checkpoint，并保留诊断日志。

这次比赛里最重要的经验是：不是模块越多越好。很多更大的分支能提升本地验证集，
但会伤害线上 AUC。最终稳定方案尽量保持主模型的表征结构不被破坏，只加入在线上
反复验证过更稳定的信号。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── assets/
│   ├── leaderboard_result.png
│   └── leaderboard_result.svg
├── docs/
│   └── VERSION_NOTES.md
├── eval/
│   ├── dataset.py
│   ├── infer.py
│   └── model.py
├── scripts/
│   └── run_v43_1.sh
└── src/
    ├── dataset.py
    ├── infer.py
    ├── model.py
    ├── ns_groups.json
    ├── train.py
    ├── trainer.py
    └── utils.py
```

## 环境

代码面向 TAAC 平台的 PyTorch 运行环境开发。本地环境可以这样准备：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

比赛平台会通过环境变量提供真实训练数据路径，例如：
`TRAIN_DATA_PATH`、`TRAIN_CKPT_PATH` 和 `TRAIN_LOG_PATH`。

## 训练

使用开源的最佳版本训练脚本：

```bash
bash scripts/run_v43_1.sh
```

关键超参数如下：

```bash
--d_model 80
--num_heads 5
--num_queries 2
--dense_pair_compressor
--exposure_time_context
--multi_res_exposure_time
--calendar_time_embeddings
--cross_calendar_time_context
--calendar_user_activity_cross
--user_field_coverage_time_context
--checkpoint_selection auc
```

## 评估文件

`eval/` 文件夹包含比赛平台评估上传所需的三个文件：

```text
eval/dataset.py
eval/model.py
eval/infer.py
```

这个目录故意保持最小化，不包含训练专用文件。

## 实验经验

比较有效的方向：

- 用户侧字段覆盖；
- 曝光时间和日历时间建模；
- 对齐 dense/int 字段的 DensePair 建模；
- 保持 checkpoint 和导出链路一致。

在我的实验中不稳定的方向：

- 大型 extra context 分支；
- 直接融合 final logit 的侧分支；
- 全量替换 tokenizer；
- 复杂 query-memory 检索；
- 激进的序列 reservoir 采样；
- 大型 DIN / SMoE 分支；
- 对大 dense 字段做原始高阶交叉。

## 免责声明

这是一个比赛研究代码仓库。官方 TAAC 数据集不会在本仓库中重新分发。使用本代码时，
请遵守比赛规则和数据集许可。
