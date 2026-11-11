# 数据、音源和权利说明

## 范例音乐

唯一保留的试听范例为本项目经用户逐句反馈修订的原创钢琴短曲《灯火归来》。其名称、动机与结构来自本次项目迭代；没有将《秋日私语》等商业作品的原谱或录音放入仓库。

`examples/light-and-return/` 的 WAV 是 FluidSynth + FluidR3_GM Acoustic Grand Piano 渲染结果，保留 CC64 延音和适度混响。原始渲染信息中的执行机器绝对路径已去除，音频和 MIDI 保持原文件。

## 历史训练材料

曾使用或读取 music21 内置语料、Nottingham、POP909、When-in-Rome/OpenScore 及古典乐谱镜像。此次整理不再分发这些原始语料、源仓库副本、训练 MIDI 切片或旧试听音乐。

- music21: https://www.music21.org/
- Nottingham: https://github.com/jukedeck/nottingham-dataset
- POP909: https://github.com/music-x-lab/POP909-Dataset
- When-in-Rome: https://github.com/MarkGotham/When-in-Rome

历史来源清单和下载决策保留于 `docs/provenance/`；其中路径指向整理前的位置，仅用于审计。重新下载时应分别检查各数据集、具体乐谱版本和衍生内容的许可。项目未为第三方数据统一授予许可。

## 音源与引擎

- FluidSynth: https://www.fluidsynth.org/
- FluidR3_GM 音源本体不包含在仓库中；安装方式见 README。
- 音源的原版权和条款见 [FluidR3_GM_COPYRIGHT.txt](third_party/FluidR3_GM_COPYRIGHT.txt)。

项目代码、用户原始文档和原创音乐尚未指定开放许可证。仓库可访问性不等于自动授予再分发或商业使用许可；第三方内容仍适用各自条款。
