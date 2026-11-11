# 训练、模型与跨平台

## CPU Critic

`research/neuro_sandbox_critic.py` 保留基本特征与弱标签扰动；`research/critic_v4.py` 增加节奏、强拍和弦音、未解决外音、音域、级进、跳进恢复和乐句轮廓等特征，共 33 维。

`artifacts/training_v4/training_features.npz` 保存 498 行弱标签特征、5 个锚点的特征/评分/权重，以及特征名称。旧训练 MIDI 已按整理要求删除；评分锚点 JSON 仍保存旧路径、哈希和用户原评分，用于历史溯源，不应把旧路径当成仍可访问的文件。

```bash
python research/train.py --output outputs/training
```

偏好模型使用弱标签及高权重人工锚点。技术模型将“去变音”弱标签 55 映射为技术目标 90；这不改变人类评分。输出为 CPU sklearn 模型，与 GPU 品牌无关，但依赖 sklearn/Python 兼容环境。

历史校准报告仍保留原样。当前缓存没有逐行作品组 ID，重拟合命令不会把训练误差伪装成交叉验证；重新做作品级验证需要重新整理外部语料及分组。

## 可选神经基线

先在目标设备上安装厂商匹配的 PyTorch；MUSA 需匹配的 torch_musa。然后：

```bash
python -m pip install -r requirements-neural.txt
python research/train_neural.py --device auto --epochs 600
```

这是对当前 33 维缓存的全量拟合演示，不输出独立泛化指标。`artifacts/neural/` 的权重与报告属于早期云端 MUSA 基线，其输入宽度以自身报告的 `feature_names` 为准，不能和新特征缓存混用。

硬件路由顺序为 MUSA → CUDA/ROCm → MPS → CPU；显式指定不可用设备应报错。统一 FP32；不依赖 flash-attn、xformers、bitsandbytes。保存前将权重移回 CPU，再以 safetensors 保存；加载时先读入 CPU，再放到目标设备。

浮点权重可跨设备搬运，但驱动、torch/torch_musa 配套版本、算子支持和数值差异仍需在目标平台确认。MIDI/WAV 导出和 sklearn Critic 均不要求 GPU。
