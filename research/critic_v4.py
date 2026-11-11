"""Voice-aware, human-weighted Critic. Earlier checkpoints remain untouched."""
import copy
from pathlib import Path
import joblib
import numpy as np
from music21 import note,stream,meter
from neuro_sandbox_critic import NeuroSandboxExtractor,NeuroAestheticCritic,FEATURE_NAMES,active_chord
from voice_rules import diagnose

VOICE_FEATURE_NAMES=['strong_beat_chord_tone_ratio','unresolved_dissonance_rate','pitch_range_semitones',
 'stepwise_motion_ratio','leap_recovery_rate','large_leap_rate','multiple_leap_bar_ratio',
 'arch_climax_lift','arch_return_drop','effective_pitch_range','range_preference','stepwise_preference','fragmentation_rate']
V4_FEATURE_NAMES=FEATURE_NAMES+VOICE_FEATURE_NAMES


class VoiceExtractor(NeuroSandboxExtractor):
    def extract_5d_features(self,mel,chd):
        result=super().extract_5d_features(mel,chd);voice=diagnose(mel,chd)
        result['details'].update({name:float(voice[name]) for name in VOICE_FEATURE_NAMES})
        result['vector']=np.array([result['details'][name] for name in V4_FEATURE_NAMES],dtype=np.float32)
        result['voice_diagnostics']=voice
        return result


def chaotic_version(mel,chd,rng):
    """Varied rhythm and diatonic pitches, but non-chord accented tones and unresolved jumps."""
    from music_descriptors import bar_length
    from music21 import chord
    output=stream.Part();sig=list(mel.recurse().getElementsByClass(meter.TimeSignature))
    output.insert(0,copy.deepcopy(sig[0]) if sig else meter.TimeSignature('4/4'))
    bar=bar_length(mel);duration=float(mel.highestTime);cs=list(chd.flatten().getElementsByClass(chord.Chord))
    scale=[p for p in range(60,85) if p%12 in {0,2,4,5,7,9,11}];previous=72
    t=0.
    while t<duration-1e-6:
        remaining=min(bar-t%bar,duration-t)
        choices=[d for d in [.25,.5,1,1.5] if d<=remaining+1e-6]
        length=float(rng.choice(choices)) if choices else remaining
        c=active_chord(cs,t)
        pool=[p for p in scale if abs(p-previous)>=7 and (c is None or p%12 not in c.pitchClasses)]
        if not pool:pool=[p for p in scale if abs(p-previous)>=7]
        pitch=int(rng.choice(pool));output.insert(t,note.Note(pitch,quarterLength=length));previous=pitch;t+=length
    return output,copy.deepcopy(chd)


class VoiceCritic(NeuroAestheticCritic):
    def __init__(self,seed=42):
        super().__init__(seed);self.extractor=VoiceExtractor();self.feature_names=list(V4_FEATURE_NAMES)
        from sklearn.ensemble import GradientBoostingRegressor
        self.model=GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=.07,random_state=seed)

    def training_rows(self,pairs):
        x,y=super().training_rows(pairs);extra=[];rng=np.random.default_rng(self.seed+777)
        for mel,chd in pairs:
            for _ in range(2):extra.append(self.vector(*chaotic_version(mel,chd,rng)))
        return np.concatenate([x,np.array(extra)]),np.r_[y,np.full(len(extra),30.)]

    def save(self,path):
        joblib.dump({'schema':4,'feature_names':self.feature_names,'model':self.model,'seed':self.seed},path)

    @classmethod
    def load(cls,path):
        value=joblib.load(path)
        if value['schema']!=4 or value['feature_names']!=V4_FEATURE_NAMES:raise ValueError('Not a v4 voice Critic checkpoint')
        obj=cls(value['seed']);obj.model=value['model'];obj.is_trained=True;return obj
