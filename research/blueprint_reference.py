"""
Neuro-Aesthetic Music Sandbox & Semi-Supervised Critic (双轨神经审美打分器)
===========================================================================
架构目标：
1. 输入：8~16小节的【主旋律 (Melody) + 和弦 (Chords)】双轨符号音乐 (music21.stream.Score)
2. 特征提取：将音符转化为 5 维神经张力心电图 (5D Neuro-Tension Curves)
3. 半监督学习：利用少量大师精华切片 (Golden Positives)，自动通过“抹平变音手术”和
   “破坏解决手术”合成平庸样本 (Hard Negatives) 与噪音样本 (Noise)，训练审美打分器。

依赖安装：
pip install music21 numpy scikit-learn scipy
"""

import copy
import random
import numpy as np
from music21 import stream, note, chord, pitch, interval, key, meter
from sklearn.ensemble import GradientBoostingRegressor
from scipy.spatial.distance import cosine

# ==========================================
# 模块一：5 维神经沙盒特征提取器 (Feature Extractor)
# ==========================================
class NeuroSandboxExtractor:
    """
    将一段【主旋律 + 和弦】切片转化为 5 维审美特征向量与心电图序列（统一归一化到 C大调 / A小调）
    """
    def __init__(self):
        # C大调 / A小调自然音级集合 (Pitch Classes: C=0, D=2, E=4, F=5, G=7, A=9, B=11)
        self.diatonic_pcs = {0, 2, 4, 5, 7, 9, 11}
        # 五度圈位置映射，用于计算和弦横向离调距离
        self.circle_of_fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6,
                                 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}

    def extract_5d_features(self, melody_stream: stream.Part, chord_stream: stream.Part) -> dict:
        """
        提取 5 大维度的序列心电图及统计指纹
        """
        mel_notes = [n for n in melody_stream.flatten().notesAndRests]
        chords = [c for c in chord_stream.flatten().getElementsByClass('Chord')]

        # 1. 横向旋律信息量与意外度 (IC_melody) & 4. 呼吸与重力拱形 (G_contour)
        ic_curve, contour_score = self._calc_melody_ic_and_contour(mel_notes)

        # 2. 横向和声离调张力 (T_chord: 捕捉 1% 的借用和弦/副属变音)
        chord_tension_curve, smooth_voice_leading = self._calc_harmonic_tension(chords)

        # 3. 纵向音程撕裂度与半音叹息解决 (D_vertical: 捕捉倚音摩擦 -> 半音融化)
        vertical_friction_curve, sigh_resolution_bonus = self._calc_vertical_friction_and_resolution(
            melody_stream, chord_stream
        )

        # 5. 动机模进自相似度 (S_motif: 捕捉《秋日私语》式的宿命模进感)
        motif_similarity = self._calc_motif_self_similarity(melody_stream)

        # 汇总为用于半监督模型训练的特征向量 (12维高阶统计指纹)
        # 核心逻辑：好曲子必须有“适度的摩擦尖峰 (Max/Std)” + “极高的解决率与模进率”
        feature_vector = np.array([
            np.mean(ic_curve),
            np.std(ic_curve),                      # 旋律起伏丰富度
            np.mean(chord_tension_curve),          # 平均和声色彩度
            np.max(chord_tension_curve),           # 🔥 黄金分割点是否有离调神来之笔！(1%反常)
            smooth_voice_leading,                  # 🔥 离调和弦是否通过半音丝滑连接！(防噪音锁)
            np.mean(vertical_friction_curve),      # 平均纵向张力
            np.max(vertical_friction_curve),       # 强拍最大撕裂度
            sigh_resolution_bonus,                 # 🔥 核心指标：高摩擦外音后是否紧跟半音下行解决！
            contour_score,                         # 呼吸感与大跳反向回落得分
            motif_similarity                       # 动机模进自相似度 (99%规律锁)
        ], dtype=np.float32)

        return {
            "vector": feature_vector,
            "details": {
                "1_Melodic_IC_Mean": round(float(np.mean(ic_curve)), 3),
                "2_Harmonic_Surprise_Peak": round(float(np.max(chord_tension_curve)), 3),
                "2b_Voice_Leading_Smoothness": round(float(smooth_voice_leading), 3),
                "3_Sigh_Resolution_Bonus": round(float(sigh_resolution_bonus), 3),
                "4_Breath_and_Gravity_Score": round(float(contour_score), 3),
                "5_Motif_Sequence_Similarity": round(float(motif_similarity), 3),
            }
        }

    def _calc_melody_ic_and_contour(self, mel_notes):
        """计算旋律信息量 (基于 Narmour 预期模型简化版) 与 呼吸重力法则"""
        pitches = []
        rest_count = 0
        total_ql = 0.0

        for elem in mel_notes:
            total_ql += elem.quarterLength
            if elem.isRest or elem.quarterLength >= 2.0:
                rest_count += 1  # 长音或休止符均视为有效换气点
            if isinstance(elem, note.Note):
                pitches.append(elem.pitch.midi)

        if len(pitches) < 4:
            return [0.0], 0.0

        ic_list = []
        leap_penalties = 0
        leap_recoveries = 0

        for i in range(1, len(pitches)):
            diff = pitches[i] - pitches[i - 1]
            abs_diff = abs(diff)
            # 音程越大、或者踩在调外黑键上，信息量(意外度)越高
            is_chromatic = 1.5 if (pitches[i] % 12) not in self.diatonic_pcs else 0.0
            ic = (abs_diff * 0.25) + is_chromatic
            ic_list.append(ic)

            # 检查大跳 (>= 6 半音) 后是否反向级进回落 (Gap-Fill 物理重力法则)
            if abs_diff >= 6 and i + 1 < len(pitches):
                next_diff = pitches[i + 1] - pitches[i]
                if np.sign(next_diff) != np.sign(diff) and abs(next_diff) <= 4:
                    leap_recoveries += 1
                else:
                    leap_penalties += 1  # 连续乱飙高音不回落，重罚！

        # 计算换气率 (理想状态：每 8~16 拍有 1~3 次换气)
        measures = max(1.0, total_ql / 4.0)
        breath_ratio = min(1.0, rest_count / (measures * 0.5))
        gravity_score = (leap_recoveries + 1.0) / (leap_recoveries + leap_penalties * 2.5 + 1.0)

        contour_score = 0.5 * breath_ratio + 0.5 * gravity_score
        return ic_list, contour_score

    def _calc_harmonic_tension(self, chords):
        """计算和弦离调张力 (捕捉 #C / Abmaj7 / Fm 等调外神来之笔) 与 声部连接丝滑度"""
        if not chords:
            return [0.0], 0.0

        tensions = []
        smooth_transitions = 0

        for i, c in enumerate(chords):
            pcs = [p.pitchClass for p in c.pitches]
            # 统计和弦中包含多少个调外变音 (Borrowed / Secondary Dominant notes)
            non_diatonic_count = sum(1 for pc in pcs if pc not in self.diatonic_pcs)
            # 七和弦/九和弦自带色彩加成
            extension_bonus = 0.4 if len(pcs) >= 4 else 0.0
            tension = non_diatonic_count * 1.8 + extension_bonus
            tensions.append(tension)

            # 检查声部连接丝滑度 (与前一个和弦是否有共同音或半音平滑移动)
            if i > 0:
                prev_pcs = set(p.pitchClass for p in chords[i - 1].pitches)
                curr_pcs = set(pcs)
                common_tones = len(prev_pcs.intersection(curr_pcs))
                # 检查是否有半音滑动 (例如 G -> #G -> A 或 Bb -> A)
                semitone_moves = sum(
                    1 for p1 in prev_pcs for p2 in curr_pcs if abs(p1 - p2) in (1, 11)
                )
                if common_tones >= 1 or semitone_moves >= 1:
                    smooth_transitions += 1

        smooth_score = smooth_transitions / max(1, len(chords) - 1)
        return tensions, smooth_score

    def _calc_vertical_friction_and_resolution(self, melody_stream, chord_stream):
        """
        核心审美探测器：计算旋律与底层和弦的纵向摩擦度，
        并专门奖励【强拍高摩擦/变音 -> 紧接着下行半音/全音解决到和弦内音】的“肖邦/秋日私语式叹息”！
        """
        mel_notes = [n for n in melody_stream.flatten().notes]
        chords = [c for c in chord_stream.flatten().getElementsByClass('Chord')]
        if not mel_notes or not chords:
            return [0.0], 0.0

        frictions = []
        resolved_sighs = 0
        unresolved_clashes = 0

        for i, m_note in enumerate(mel_notes):
            offset = m_note.offset
            # 找到当前时刻正在发声的和弦
            active_chord = chords[0]
            for c in chords:
                if c.offset <= offset < c.offset + c.quarterLength:
                    active_chord = c
                    break

            chord_pcs = {p.pitchClass for p in active_chord.pitches}
            root_pc = active_chord.root().pitchClass
            mel_pc = m_note.pitch.pitchClass

            # 判断旋律音与和弦的关系
            if mel_pc in chord_pcs:
                # 落在三音、七音上比落在死板的根音上更有唯美色彩
                friction = 0.2 if mel_pc == root_pc else 0.6
                is_dissonant = False
            else:
                # 落在和弦外音（倚音/挂留音/九音）或调外变音上 -> 产生强烈撕裂感！
                is_chromatic = (mel_pc not in self.diatonic_pcs)
                friction = 2.5 if is_chromatic else 1.8
                is_dissonant = True

            # 如果当前和弦本身含有调外变音，且旋律刚好踩在那个变音上（如《秋日私语》A7配#C），同样视为高张力点
            if mel_pc not in self.diatonic_pcs:
                friction = max(friction, 2.2)
                is_dissonant = True

            frictions.append(friction)

            # 🔥 检查是否发生了“先揪心、后融化”的半音/级进解决 (Appoggiatura Resolution)
            if is_dissonant and (i + 1 < len(mel_notes)):
                next_note = mel_notes[i + 1]
                semitone_step = next_note.pitch.midi - m_note.pitch.midi
                # 如果下一个音通过 1~2 个半音平滑移动（尤其是下行叹息 -1, -2）
                if abs(semitone_step) in (1, 2):
                    resolved_sighs += 1.5 if semitone_step < 0 else 1.0
                elif abs(semitone_step) >= 5:
                    # 踩在刺耳音上不仅不解决，反而乱跳 -> 噪音重罚！
                    unresolved_clashes += 1.5

        sigh_bonus = (resolved_sighs * 2.0) / max(1.0, len(mel_notes) * 0.25 + unresolved_clashes * 3.0)
        return frictions, min(3.0, sigh_bonus)

    def _calc_motif_self_similarity(self, melody_stream):
        """
        计算第 1~2 小节的核心动机与后续小节的【音程轮廓模进相似度】
        （例如《秋日私语》开头连续 4 次平移同一个四音动机，此项得分接近满分 1.0）
        """
        measures = list(melody_stream.getElementsByClass('Measure'))
        if len(measures) < 4:
            # 若未显式分小节，则按每 4 拍切分窗口
            notes = [n for n in melody_stream.flatten().notes]
            if len(notes) < 8:
                return 0.5
            chunk_size = max(3, len(notes) // 4)
            chunks = [notes[i:i + chunk_size] for i in range(0, len(notes) - chunk_size + 1, chunk_size)]
        else:
            chunks = [list(m.flatten().notes) for m in measures if len(m.flatten().notes) >= 2]

        if len(chunks) < 2:
            return 0.5

        def get_interval_contour(note_list):
            if len(note_list) < 2:
                return np.zeros(4)
            diffs = [note_list[j+1].pitch.midi - note_list[j].pitch.midi for j in range(len(note_list)-1)]
            # 截取或补齐到固定长度 4，比较相对音程走向（捕获模进 Sequence）
            vec = np.array(diffs[:4] + [0]*(4 - len(diffs[:4])), dtype=float)
            return vec

        seed_motif = get_interval_contour(chunks[0])
        similarities = []
        for idx in range(1, len(chunks)):
            target_motif = get_interval_contour(chunks[idx])
            if np.linalg.norm(seed_motif) == 0 or np.linalg.norm(target_motif) == 0:
                sim = 0.5
            else:
                # 方向相同、步幅相近即视为完美模进
                sim = 1.0 - (cosine(seed_motif, target_motif) / 2.0)
            similarities.append(max(0.0, sim))

        return float(np.mean(similarities))


# ==========================================
# 模块二：半监督“外科阉割与噪音”自动生成器
# ==========================================
class SemiSupervisedAugmenter:
    """
    核心半监督机制：不需要人工去找烂曲子！
    直接拿 98 分的大师正样本做两种手术：
    1. 阉割手术 (de_chromatize)：抹掉所有调外变音与倚音，变成 55 分的“平庸白开水样本”。
    2. 破坏手术 (corrupt_noise)：把高潮音乱飙八度、破坏半音解决、打乱模进，变成 10 分的“噪音样本”。
    """
    Diantonic_Snap = {1: 0, 3: 2, 6: 5, 8: 7, 10: 9}  # 将所有黑键强行抹平回白键

    @classmethod
    def create_mediocre_version(cls, mel_part: stream.Part, chord_part: stream.Part):
        """生成 55 分平庸版：抹杀那 1% 的神来之笔（所有黑键/离调和弦全部改成普通白键三和弦）"""
        mel_copy = copy.deepcopy(mel_part)
        chord_copy = copy.deepcopy(chord_part)

        for n in mel_copy.flatten().notes:
            pc = n.pitch.pitchClass
            if pc in cls.Diantonic_Snap:
                n.pitch.midi -= 1  # 抹去半音变音

        for c in chord_copy.flatten().getElementsByClass('Chord'):
            new_pitches = []
            for p in c.pitches[:3]:  # 砍掉七音/九音色彩，只留干瘪三和弦
                if p.pitchClass in cls.Diantonic_Snap:
                    p.midi -= 1
                new_pitches.append(p)
            c.pitches = new_pitches

        return mel_copy, chord_copy

    @classmethod
    def create_noisy_version(cls, mel_part: stream.Part, chord_part: stream.Part):
        """生成 10 分烂俗/噪音版：模拟突然乱飙高音、破坏模进、制造不解决的刺耳碰撞"""
        mel_copy = copy.deepcopy(mel_part)
        chord_copy = copy.deepcopy(chord_part)

        notes = list(mel_copy.flatten().notes)
        for i, n in enumerate(notes):
            if i % 3 == 1:
                # 模拟毫无铺垫地突然飙高音或乱跳 (+11 或 +14 半音)
                n.pitch.midi += random.choice([11, 13, -9])

        return mel_copy, chord_copy


# ==========================================
# 模块三：半监督审美打分器 (Neuro-Aesthetic Critic)
# ==========================================
class NeuroAestheticCritic:
    def __init__(self):
        self.extractor = NeuroSandboxExtractor()
        self.model = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42)
        self.is_trained = False

    def train_semi_supervised(self, golden_masterpieces: list):
        """
        输入：黄金大师切片列表 [(melody_part, chord_part), ...]
        自动通过半监督对比增强扩展数据集并训练打分器
        """
        X, y = [], []

        for mel, chd in golden_masterpieces:
            # 1. 大师原版 -> 黄金正样本 (96 ~ 99 分)
            feat_gold = self.extractor.extract_5d_features(mel, chd)["vector"]
            for _ in range(5):
                noise = np.random.normal(0, 0.02, size=feat_gold.shape)
                X.append(feat_gold + noise)
                y.append(random.uniform(96.0, 99.5))

            # 2. 外科阉割版（抹去 1% 变音与色彩和弦） -> 平庸样本 (52 ~ 58 分)
            mel_med, chd_med = SemiSupervisedAugmenter.create_mediocre_version(mel, chd)
            feat_med = self.extractor.extract_5d_features(mel_med, chd_med)["vector"]
            for _ in range(5):
                noise = np.random.normal(0, 0.02, size=feat_med.shape)
                X.append(feat_med + noise)
                y.append(random.uniform(52.0, 58.0))

            # 3. 破坏逻辑版（乱飙高音、不解决、无模进） -> 噪音负样本 (8 ~ 18 分)
            for _ in range(5):
                mel_bad, chd_bad = SemiSupervisedAugmenter.create_noisy_version(mel, chd)
                feat_bad = self.extractor.extract_5d_features(mel_bad, chd_bad)["vector"]
                X.append(feat_bad)
                y.append(random.uniform(8.0, 18.0))

        self.model.fit(np.array(X), np.array(y))
        self.is_trained = True

    def evaluate(self, mel_part: stream.Part, chord_part: stream.Part) -> dict:
        """给任意一段【主旋律 + 和弦】打出审美总分 (0~100) 并输出可解释性诊断报告"""
        res = self.extractor.extract_5d_features(mel_part, chord_part)
        score = float(self.model.predict([res["vector"]])[0])
        score = max(0.0, min(100.0, score))
        return {
            "Aesthetic_Score": round(score, 2),
            "5D_Neuro_Diagnostics": res["details"]
        }


# ==========================================
# 模块四：内置《秋日私语》与肖邦式切片生成器（开箱即测！）
# ==========================================
def build_a_comme_amour_slice():
    """
    内置生成《秋日私语》(A Comme Amour) 核心主歌切片（已统一移调至 A小调 以对齐沙盒坐标系）：
    和弦进行：Dm7 -> G7 -> Cmaj7 -> Fmaj7 -> Bm7b5 -> E7(#G变音!) -> Am
    主旋律：经典的四音叹息模进瀑布流，在 E7 处踩中 #G (Midi=68) 制造揪心感，随后半音解决回 A (Midi=69)！
    """
    mel = stream.Part()
    chd = stream.Part()

    # 左手五度圈七和弦瀑布流（每小节 4 拍）
    chord_data = [
        (["D3", "F3", "A3", "C4"], 4.0),   # Dm7
        (["G2", "F3", "G3", "B3"], 4.0),   # G7
        (["C3", "E3", "G3", "B3"], 4.0),   # Cmaj7
        (["F2", "E3", "A3", "C4"], 4.0),   # Fmaj7
        (["B2", "D3", "F3", "A3"], 4.0),   # Bm7b5
        (["E2", "D3", "G#3", "B3"], 4.0),  # 🔥 E7 (含调外神来之笔 #G 导音变音！)
        (["A2", "C3", "E3", "A3"], 8.0),   # Am (完美回归解决)
    ]
    offset = 0.0
    for notes_list, dur in chord_data:
        c = chord.Chord(notes_list, quarterLength=dur)
        chd.insert(offset, c)
        offset += dur

    # 右手主旋律：严格遵循“四音叹息模进” + “黄金分割点 #G 揪心变音与半音解决”
    # 每一组模进：[休止1拍, 高音1拍, 下行二度1拍, 上行回绕0.5拍, 下行三度解决0.5拍]
    melody_motifs = [
        # Measure 1 (on Dm7): F5 -> E5 -> F5 -> D5
        (None, 1.0), ("F5", 1.0), ("E5", 1.0), ("F5", 0.5), ("D5", 0.5),
        # Measure 2 (on G7): E5 -> D5 -> E5 ->B4 (模进下移一度)
        (None, 1.0), ("E5", 1.0), ("D5", 1.0), ("E5", 0.5), ("B4", 0.5),
        # Measure 3 (on Cmaj7): D5 -> C5 -> D5 -> G4 (模进继续下移一度)
        (None, 1.0), ("D5", 1.0), ("C5", 1.0), ("D5", 0.5), ("G4", 0.5),
        # Measure 4 (on Fmaj7): C5 -> B4 -> C5 -> A4 (强拍 B4 撞 Fmaj7 产生极美增四度摩擦后半音上行解决!)
        (None, 1.0), ("C5", 1.0), ("B4", 1.0), ("C5", 0.5), ("A4", 0.5),
        # Measure 5 (on Bm7b5): B4 -> A4 -> B4 -> F4
        (None, 1.0), ("B4", 1.0), ("A4", 1.0), ("B4", 0.5), ("F4", 0.5),
        # Measure 6 (on E7): 🔥 神来之笔！踩在 #G4 变音上撕裂摩擦，紧接着在下一小节滑入 A4！
        (None, 1.0), ("B4", 1.0), ("A4", 1.0), ("G#4", 1.0),
        # Measure 7~8 (on Am): 完美叹息落地 + 长音换气呼吸
        ("A4", 4.0), (None, 4.0)
    ]
    m_offset = 0.0
    for pitch_name, dur in melody_motifs:
        if pitch_name is None:
            mel.insert(m_offset, note.Rest(quarterLength=dur))
        else:
            mel.insert(m_offset, note.Note(pitch_name, quarterLength=dur))
        m_offset += dur

    return mel, chd


if __name__ == "__main__":
    print("=" * 72)
    print("🚀 启动【双轨神经审美打分器 (Neuro-Aesthetic Critic)】半监督训练与盲测...")
    print("=" * 72)

    # 1. 载入黄金正样本切片（示例：内置的《秋日私语》核心切片）
    autumn_mel, autumn_chd = build_a_comme_amour_slice()
    critic = NeuroAestheticCritic()
    critic.train_semi_supervised([(autumn_mel, autumn_chd)])

    # 2. 生成三组对比测试样本：
    # (A) 原版《秋日私语》精华段
    # (B) 被抹掉 #G 离调变音与七和弦色彩的“平庸白开水版”
    # (C) 模拟普通 AI 突然乱飙高音、破坏模进的“噪音版”
    med_mel, med_chd = SemiSupervisedAugmenter.create_mediocre_version(autumn_mel, autumn_chd)
    bad_mel, bad_chd = SemiSupervisedAugmenter.create_noisy_version(autumn_mel, autumn_chd)

    for label, (m, c) in [
        ("🏆 测试样本 A: 《秋日私语》原版黄金切片 (99%模进规律 + 1% #G离调半音叹息)", (autumn_mel, autumn_chd)),
        ("😐 测试样本 B: 外科阉割版 (抹去 #G 变音与七和弦，变成普通白键口水歌)", (med_mel, med_chd)),
        ("💥 测试样本 C: 乱飙高音噪音版 (无铺垫大跳 + 刺耳外音不解决)", (bad_mel, bad_chd))
    ]:
        report = critic.evaluate(m, c)
        print(f"\n{label}")
        print(f"  👉 最终审美打分: 【 {report['Aesthetic_Score']} / 100 分 】")
        print(f"  📊 5维神经心电图诊断: {report['5D_Neuro_Diagnostics']}")
