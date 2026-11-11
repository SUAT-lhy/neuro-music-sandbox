# 神经音乐沙盒 · 灯火归来

用可解释的符号音乐特征、质量评分与真实试听反馈，探索音乐创作闭环。当前保留的唯一范例是 **《灯火归来》短版**：钢琴主旋律、十度流水琶音、踏板和适度混响。

**先听音乐：** [WAV](examples/light-and-return/light_and_return_short.wav) · [MIDI](examples/light-and-return/light_and_return_short.mid)

最终范例经过逐句人工反馈与代码修订；不是训练后的神经生成器自动作曲。评分器是独立保留的研究组件，没有替代听众评价。项目没有采集 EEG、fMRI 或生理反应数据。

## 范例结构

| 时间（约） | 内容 |
| --- | --- |
| 0–7.06 秒 | 保留《远处的灯》前两小节 |
| 7.06–10.59 秒 | 一小节过渡 |
| 10.59–15.73 秒 | 《等你归来》的抓耳主题，约 5.14 秒 |
| 15.73–29.45 秒 | G 大调尾声 |
| 末尾 | 钢琴与混响自然衰减 |

原有曲名是本项目迭代中的原创片段名称。钢琴事件保存在 `piano/score.json`；此文件是当前作品的独立源谱，不依赖已删除的早期试听版本。主题末尾用一小节 2/4 衔接，再回到 4/4。开头 68 BPM，主题起切换为 70 BPM。

## 目录

```text
piano/                       当前曲谱、MIDI 导出、采样钢琴渲染
examples/light-and-return/   唯一保留的 WAV、MIDI 和渲染信息
research/                    特征提取、CPU Critic、可选神经基线、跨卡设备路由
artifacts/training_v4/        Critic、技术底线模型、特征缓存、历史校准报告
artifacts/neural/             历史 MUSA MLP 权重与对应报告
data/feedback/               原始听感反馈与评分锚点
docs/                        思路、演进、训练说明、数据来源和清理记录
```

## 导出与渲染

建议 Python 3.10–3.12。从项目根目录执行：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m piano.export
```

MIDI 输出到 `outputs/light-and-return/`。导出只需普通 CPU。编辑 `piano/score.json` 中的音符、力度或踏板位置后，再运行导出即可。

生成新的 WAV 需要系统安装 **FluidSynth**，并自行取得 **FluidR3_GM.sf2**：

```bash
# Ubuntu/Debian 示例
sudo apt-get install fluidsynth fluid-soundfont-gm
python -m piano.export --render
```

Windows 安装 FluidSynth 并加入 PATH 后，可指定本地音源：

```powershell
python -m piano.export --render --soundfont 'D:/SoundFonts/FluidR3_GM.sf2'
```

也可设置 `PIANO_SOUNDFONT` 环境变量。仓库不包含音源本体。MIDI CC64 会直接交给合成器，保留声部、时值、力度与延音；WAV 为立体声 22.05 kHz / PCM16。渲染后只作整首常数增益调整。更换音源或 FluidSynth 版本可能改变音频结果。

## 评分器与训练

```bash
python -m pip install -r requirements-research.txt
python research/train.py
```

这会用保留的 498 行弱标签特征和 5 个人类评分锚点重新拟合两个 CPU 模型，结果写入 `outputs/training/`。它不下载乐谱，也不生成新的独立验证误差。第三方语料、旧试听 MIDI 和下载缓存已清理；源作品分组信息见历史报告，特征缓存不足以重新做作品级交叉验证。

历史检查点由 scikit-learn 1.7.2 的云端环境导出。跨 scikit-learn 版本加载 pickle 不可靠；在当前环境重新拟合，或使用 [历史模型环境约束](requirements-models.txt)。只加载可信来源的 pickle/joblib。

可选 PyTorch MLP 的设备检测支持 MUSA、CUDA/ROCm、MPS、CPU；训练使用 FP32，保存 CPU `state_dict` 为 safetensors。保留厂商匹配的 torch/torch_musa，不在依赖文件中替换它们。详见 [训练与跨卡说明](docs/TRAINING.md)。权重可移植不等于所有后端算子和数值行为完全一致。

## 实验结论与限制

- 机械重复、高节奏熵都可能被评分器误奖；高代理分数没有保证好听。
- 历史 5 个锚点的样本内 MAE 约 0.10；留一评分 MAE 约 11.94，与恒定 50 分基线的 12.00 接近，尚不能证明偏好泛化。
- 最终进展主要来自听众指出具体时间段、保留有效动机、定点修改旋律与和声。
- 原始文档包含早期设想和未经本项目验证的神经科学表述，不作为当前实现能力或科学结论。

详见 [项目思路](docs/IDEA.md)、[迭代记录](docs/EVOLUTION.md)、[数据与第三方说明](docs/DATA_AND_CREDITS.md)。未为项目代码和原创范例指定开放许可证；第三方材料适用各自条款。
