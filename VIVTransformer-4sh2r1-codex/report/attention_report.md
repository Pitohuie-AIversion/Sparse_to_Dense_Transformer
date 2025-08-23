# 注意力机制对比分析报告
本报告聚合了批量实验的训练与测试日志，按注意力机制汇总最佳结果并进行横向对比。
## 总体排名（Top-5，按测试损失）
| 排名 | 注意力 | 平均测试损失 | 最终测试损失 | 最佳验证损失 | 轮数 | 源目录 | Loss配置 |
|---:|:--|--:|--:|--:|--:|:--|:--|
| 1 | sparse | 0.380726 | 0.375148 | 0.382926 | 1 | dated:20250814_093108 | - |
| 2 | relative | 0.555391 | 0.556663 | 0.551188 | 1 | dated:20250814_093108 | - |
| 3 | lsh | 0.559291 | 0.560872 | 0.554736 | 1 | dated:20250814_093108 | - |
| 4 | muse | 0.579450 | 0.581355 | 0.574924 | 1 | dated:20250814_093108 | - |
| 5 | se | 0.580671 | 0.582564 | 0.576159 | 1 | dated:20250814_093108 | - |

## 详细对比（每类注意力的最佳一次）
| 注意力 | 平均测试损失 | 最终测试损失 | 最佳验证损失 | 轮数 | 训练时长(s) | 参数量 | FLOPs | 源目录 | Loss配置 | base_weight | topk |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|:--|--:|--:|
| sparse | 0.380726 | 0.375148 | 0.382926 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| relative | 0.555391 | 0.556663 | 0.551188 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| lsh | 0.559291 | 0.560872 | 0.554736 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| muse | 0.579450 | 0.581355 | 0.574924 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| se | 0.580671 | 0.582564 | 0.576159 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| eca | 0.580724 | 0.582489 | 0.576139 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| sge | 0.583803 | 0.585847 | 0.579246 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| external | 0.589789 | 0.591623 | 0.585192 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| coord | 0.591174 | 0.592992 | 0.586588 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| residual | 0.592173 | 0.594064 | 0.587553 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| shuffle | 0.592401 | 0.594279 | 0.587763 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| cbam | 0.593262 | 0.595162 | 0.588670 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| self | 0.644017 | 0.645741 | 0.638322 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| simplified_self | 0.645049 | 0.647556 | 0.640584 | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| aft | - | - | - | - | - | - | - | dated:20250814_093108 | - | - | - |
| ufo | - | nan | - | 1 | - | - | - | dated:20250814_093108 | - | - | - |
| a2 | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| axial | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| bam | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| coatnet | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| cot | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| crisscross | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| crossformer | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| danet | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| dat | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| emsa | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| gfnet | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| halo | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| moa | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| mobilevit | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| mobilevitv2 | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| outlook | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| parnet | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| polarized | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| psa | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| s2 | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| sk | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| triplet | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |
| vip | - | - | - | - | - | - | - | loss_config_0 | loss_config_0 | - | - |

## 数据来源与解析说明
- loss_logs/loss_log.txt：解析最终/最佳损失与轮数。
- test_results/test_loss_log.txt：解析批次级测试损失与平均值。
- test_result_*.txt：解析一次性汇报的测试损失。
- research_data/comprehensive_metrics_*.json：若存在，补充训练时长、模型复杂度与FLOPs。
- failed_attention_log.txt：若存在，列出训练失败的注意力机制（来自训练脚本 main.py）。
