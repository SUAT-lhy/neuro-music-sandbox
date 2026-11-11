"""Interpretable symbolic-music proxy critic; synthetic labels are not listener ratings."""
import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
from music21 import chord, note, stream
from sklearn.ensemble import GradientBoostingRegressor

BASE_FEATURE_NAMES = ['ic_mean', 'ic_std', 'harmonic_mean', 'harmonic_peak',
                 'voice_leading', 'friction_mean', 'friction_peak',
                 'resolution', 'breath_gravity', 'motif', 'tension_release', 'rhythm_similarity']
RHYTHM_FEATURE_NAMES=['duration_entropy','ioi_entropy','cross_bar_rhythm_change','syncopation_ratio',
                     'rest_ratio','harmonic_change_rate','bar_repetition_ratio','strong_beat_clash_ratio']
FEATURE_NAMES=BASE_FEATURE_NAMES+RHYTHM_FEATURE_NAMES
DIATONIC = {0, 2, 4, 5, 7, 9, 11}


def active_chord(chords, offset):
    return next((c for c in reversed(chords)
                 if float(c.offset) <= offset < float(c.offset + c.quarterLength)), None)


class NeuroSandboxExtractor:
    def extract_5d_features(self, melody_stream, chord_stream):
        events = list(melody_stream.flatten().notesAndRests)
        if any(isinstance(n, chord.Chord) for n in events):
            raise ValueError('Melody must be monophonic; use the score loader skyline reduction.')
        notes = [n for n in events if isinstance(n, note.Note)]
        chords = list(chord_stream.flatten().getElementsByClass(chord.Chord))
        if len(notes) < 4 or not chords:
            raise ValueError('Need at least four melody notes and one sounding accompaniment chord.')
        duration = max(float(melody_stream.highestTime), float(chord_stream.highestTime))
        pitches = np.array([n.pitch.midi for n in notes])
        diffs = np.diff(pitches)
        # Simplified interval expectation cost, not a fitted IDyOM information model.
        ic = np.r_[0., np.abs(diffs) * .25] + np.array([1.5 * (p % 12 not in DIATONIC) for p in pitches])
        tension = np.array([1.8 * sum(p.pitchClass not in DIATONIC for p in c.pitches)
                            + .4 * (len(set(c.pitchClasses)) >= 4) for c in chords])
        leading = []
        for a, b in zip(chords, chords[1:]):
            aa, bb = set(a.pitchClasses), set(b.pitchClasses)
            distance = np.mean([min(min((x-y) % 12, (y-x) % 12) for y in bb) for x in aa])
            leading.append(np.exp(-distance / 2))
        friction, resolved, eligible = [], 0., 0
        for i, n in enumerate(notes):
            c = active_chord(chords, float(n.offset))
            if c is None:
                friction.append(0.)
                continue
            pc = n.pitch.pitchClass
            outside = pc not in set(c.pitchClasses)
            chromatic = pc not in DIATONIC
            f = (2.5 if chromatic else 1.8) if outside else (.2 if pc == c.root().pitchClass else .6)
            friction.append(max(f, 2.2) if chromatic else f)
            if outside or chromatic:
                eligible += 1
                if i + 1 < len(notes):
                    nxt = notes[i+1]
                    nc = active_chord(chords, float(nxt.offset))
                    step = nxt.pitch.midi - n.pitch.midi
                    # Require an immediate, bounded-time step into a currently sounding chord tone.
                    contiguous = abs(float(nxt.offset - n.offset - n.quarterLength)) < 1e-6
                    if contiguous and float(nxt.offset-n.offset) <= 2 and abs(step) in (1, 2) and nc and nxt.pitch.pitchClass in nc.pitchClasses:
                        resolved += 1
        resolution = resolved / max(1, eligible)
        leaps = [i for i, d in enumerate(diffs[:-1]) if abs(d) >= 6]
        recovery = np.mean([diffs[i] * diffs[i+1] < 0 and 0 < abs(diffs[i+1]) <= 4 for i in leaps]) if leaps else .5
        breath = sum(isinstance(n, note.Rest) or n.quarterLength >= 2 for n in events) / len(events)
        gravity = float(.5 * recovery + .5 * min(1., breath * 4))
        # Two-bar windows use the actual meter, rather than arbitrary note-count chunks.
        from music21 import meter
        signatures = list(melody_stream.recurse().getElementsByClass(meter.TimeSignature))
        bar = float(signatures[0].barDuration.quarterLength) if signatures else 4.
        windows = []
        for start in np.arange(0, duration, bar * 2):
            ns = [n for n in notes if start <= float(n.offset) < start + bar * 2]
            if len(ns) >= 3:
                windows.append((np.diff([n.pitch.midi for n in ns]), [float(n.quarterLength) for n in ns]))
        sims, rhythms = [], []
        if windows:
            a, ar = windows[0]
            for b, br in windows[1:]:
                k = min(len(a), len(b))
                sims.append(float(np.exp(-np.mean(np.abs(a[:k]-b[:k])) / 3)) * k / max(len(a),len(b)))
                k = min(len(ar), len(br))
                rhythms.append(float(np.exp(-np.mean(np.abs(np.array(ar[:k])-br[:k])))) * k / max(len(ar),len(br)))
        motif = float(np.mean(sims)) if sims else 0.
        rhythm = float(np.mean(rhythms)) if rhythms else 0.
        release = max(0., float(np.max(tension) - tension[-1]))
        vector = np.array([ic.mean(), ic.std(), tension.mean(), tension.max(),
                           np.mean(leading) if leading else 0., np.mean(friction), max(friction),
                           resolution, gravity, motif, release, rhythm], dtype=np.float32)
        from music_descriptors import rhythm_stats
        timing=rhythm_stats(melody_stream,chord_stream)
        vector=np.r_[vector,np.array([timing[name] for name in RHYTHM_FEATURE_NAMES],dtype=np.float32)]
        beats = np.arange(0, duration, 1.)
        curves = []
        for t in beats:
            ni = [i for i,n in enumerate(notes) if float(n.offset) <= t < float(n.offset+n.quarterLength)]
            ci = [i for i,c in enumerate(chords) if float(c.offset) <= t < float(c.offset+c.quarterLength)]
            window=[n for n in events if t-bar<float(n.offset)<=t+bar]
            local_breath=sum(n.isRest and n.quarterLength>=.125 or n.quarterLength>=2 for n in window)/max(1,len(window))
            local_indices=[i for i,n in enumerate(notes) if t-bar<float(n.offset)<=t+bar]
            local_recovery=[]
            for i in local_indices:
                if 0<i<len(notes)-1 and abs(diffs[i-1])>=6:
                    local_recovery.append(diffs[i-1]*diffs[i]<0 and abs(diffs[i])<=4)
            local_gravity=.5*min(1.,local_breath*4)+.5*(float(np.mean(local_recovery)) if local_recovery else .5)
            current_bar=int(t//bar)
            local=[n for n in notes if current_bar*bar<=float(n.offset)<(current_bar+1)*bar]
            seed_notes=[n for n in notes if float(n.offset)<bar]
            a=np.diff([n.pitch.midi for n in seed_notes]);b=np.diff([n.pitch.midi for n in local]);k=min(len(a),len(b))
            local_motif=float(np.exp(-np.mean(abs(a[:k]-b[:k]))/3)*k/max(len(a),len(b))) if k else 0.
            curves.append([float(ic[ni[0]]) if ni else 0., float(tension[ci[-1]]) if ci else 0.,
                           float(friction[ni[0]]) if ni else 0.,local_gravity,local_motif])
        if not np.isfinite(vector).all():
            raise ValueError('Non-finite feature values')
        return {'vector': vector, 'details': dict(zip(FEATURE_NAMES, map(float, vector))),
                'curves': {'beat_offsets': beats.tolist(), 'channels': ['IC','T_chord','D_vertical','G_contour','S_motif'],
                           'values': curves}, 'resolved_events': resolved, 'eligible_events': eligible}


class SemiSupervisedAugmenter:
    @staticmethod
    def create_mechanical_version(mel_part,chord_part,kind='uniform'):
        from music_descriptors import bar_length
        mel=stream.Part();chd=copy.deepcopy(chord_part);bar=bar_length(mel_part)
        from music21 import meter
        sig=list(mel_part.recurse().getElementsByClass(meter.TimeSignature))
        if sig:mel.insert(0,copy.deepcopy(sig[0]))
        source=list(mel_part.flatten().notes);duration=float(mel_part.highestTime)
        if kind=='uniform':
            # Retain the pitch sequence, remove expressive durations and rests.
            for i,t in enumerate(np.arange(0,duration,.5)):
                mel.insert(float(t),note.Note(source[i%len(source)].pitch.midi,quarterLength=min(.5,duration-t)))
        else:
            # Repeat the first bar exactly, while leaving the real harmony unchanged.
            origin=int(float(source[0].offset)//bar)*bar
            motif=[n for n in mel_part.flatten().notesAndRests if origin<=float(n.offset)<origin+bar]
            for start in np.arange(0,duration,bar):
                for n in motif:
                    relative=float(n.offset)-origin;t=float(start+relative)
                    if t<duration:
                        obj=copy.deepcopy(n);obj.quarterLength=min(float(n.quarterLength),bar-relative,duration-t)
                        if obj.quarterLength>0:mel.insert(t,obj)
        return mel,chd

    @staticmethod
    def create_mediocre_version(mel_part, chord_part):
        mel, chd = copy.deepcopy(mel_part), copy.deepcopy(chord_part)
        for n in mel.flatten().notes:
            if n.pitch.pitchClass not in DIATONIC:
                n.pitch.midi -= 1
        for c in chd.flatten().getElementsByClass(chord.Chord):
            ps = list(c.pitches[:3])
            for p in ps:
                if p.pitchClass not in DIATONIC:
                    p.midi -= 1
            c.pitches = ps
        return mel, chd

    @staticmethod
    def create_noisy_version(mel_part, chord_part, rng=None):
        rng = rng or np.random.default_rng(42)
        mel, chd = copy.deepcopy(mel_part), copy.deepcopy(chord_part)
        for i,n in enumerate(mel.flatten().notes):
            if i % 3 == 1:
                n.pitch.midi = int(np.clip(n.pitch.midi + rng.choice([11,13,-9]), 0,127))
        return mel, chd


class NeuroAestheticCritic:
    def __init__(self, seed=42):
        self.seed = seed
        self.extractor = NeuroSandboxExtractor()
        self.model = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=seed, loss='squared_error')
        self.is_trained = False
        self.feature_names=list(FEATURE_NAMES)

    def vector(self,mel,chd):
        features=self.extractor.extract_5d_features(mel,chd)
        return np.array([features['details'][name] for name in self.feature_names],dtype=np.float32)

    def training_rows(self, pairs):
        rng = np.random.default_rng(self.seed)
        x, y = [], []
        for mel,chd in pairs:
            variants = [(mel,chd,98.), (*SemiSupervisedAugmenter.create_mediocre_version(mel,chd),55.)]
            variants += [(*SemiSupervisedAugmenter.create_noisy_version(mel,chd,rng),12.) for _ in range(8)]
            variants += [(*SemiSupervisedAugmenter.create_mechanical_version(mel,chd,kind),40.) for kind in ['uniform','loop']]
            original = self.vector(mel,chd)
            for m,c,label in variants:
                v = self.vector(m,c)
                if label in (40,55) and np.allclose(original,v):
                    continue  # Do not assign contradictory labels to unchanged transformations.
                x.append(v); y.append(label)
        return np.array(x), np.array(y)

    def train_semi_supervised(self, golden_masterpieces):
        x,y = self.training_rows(golden_masterpieces)
        if not len(x):
            raise ValueError('No training samples')
        self.model.fit(x,y)
        self.is_trained = True
        return x,y

    def evaluate(self, mel_part, chord_part):
        if not self.is_trained:
            raise RuntimeError('Train or load a critic before evaluation')
        res = self.extractor.extract_5d_features(mel_part,chord_part)
        vec=[res['details'][name] for name in self.feature_names]
        return {'Aesthetic_Score': round(float(np.clip(self.model.predict([vec])[0],0,100)),2),
                '5D_Neuro_Diagnostics': res['details'], 'curves': res['curves']}

    def save(self, path):
        # CPU estimator; pickle/joblib files must only be loaded from trusted sources.
        joblib.dump({'schema':2,'feature_names':self.feature_names,'model':self.model,'seed':self.seed},path)

    @classmethod
    def load(cls,path):
        payload = joblib.load(path)
        if payload['schema'] not in (1,2) or payload['feature_names'] not in (FEATURE_NAMES,BASE_FEATURE_NAMES):
            raise ValueError('Feature schema mismatch')
        obj = cls(payload['seed']); obj.model = payload['model']; obj.is_trained = True;obj.feature_names=payload['feature_names']
        return obj


def build_a_comme_amour_slice():
    """Blueprint's synthetic circle-of-fifths demo, NOT a verified original transcription."""
    from blueprint_reference import build_a_comme_amour_slice as build
    return build()


def export_midi(pair,path):
    score = stream.Score()
    for name,part in zip(['Melody','Chords'],pair):
        p = copy.deepcopy(part); p.partName = name; score.insert(0,p)
    score.write('midi',fp=str(path))


def demo(output='artifacts/demo'):
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    gold = build_a_comme_amour_slice()
    critic = NeuroAestheticCritic(); critic.train_semi_supervised([gold]); critic.save(out/'critic_demo.pkl')
    variants = {'synthetic_original':gold, 'flattened':SemiSupervisedAugmenter.create_mediocre_version(*gold),
                'noise':SemiSupervisedAugmenter.create_noisy_version(*gold,np.random.default_rng(2026))}
    report = {'evaluation_type':'training-anchor demonstration, not blind validation','samples':{}}
    for label,pair in variants.items():
        report['samples'][label] = critic.evaluate(*pair)
        export_midi(pair,out/f'{label}.mid')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v['Aesthetic_Score'] for k,v in report['samples'].items()},ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output',default='artifacts/demo')
    demo(p.parse_args().output)
