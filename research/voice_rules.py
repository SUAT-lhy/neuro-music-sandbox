"""Auditable melody constraints: sounding strong beats, passing tones, leaps and arch."""
import numpy as np
from music21 import chord,meter,note
from music_descriptors import bar_length


def strong_offsets(signature):
    if signature=='4/4':return [0.,2.]
    if signature=='3/4':return [0.]
    if signature=='6/8':return [0.,1.5]
    return [0.]


def active(chords,t):
    return next((c for c in reversed(chords) if float(c.offset)<=t<float(c.offset+c.quarterLength)-1e-7),None)


def diagnose(mel,harmony):
    notes=[n for n in mel.flatten().notes if isinstance(n,note.Note)]
    cs=list(harmony.flatten().getElementsByClass(chord.Chord));bar=bar_length(mel)
    signatures=list(mel.recurse().getElementsByClass(meter.TimeSignature));signature=signatures[0].ratioString if signatures else '4/4'
    pitches=np.array([n.pitch.midi for n in notes]);durations=np.array([float(n.quarterLength) for n in notes]);diffs=np.diff(pitches)
    if len(notes)<4:raise ValueError('Voice diagnostic needs four notes')
    strong_eligible=0;strong_bad=[]
    for start in np.arange(0,float(mel.highestTime)-1e-5,bar):
        for offset in strong_offsets(signature):
            t=start+offset;n=next((n for n in notes if float(n.offset)<=t<float(n.offset+n.quarterLength)-1e-7),None);c=active(cs,t)
            if n is not None and c is not None:
                strong_eligible+=1
                if n.pitch.pitchClass not in c.pitchClasses:strong_bad.append(float(t))
    dissonant=[];unresolved=[]
    for i,n in enumerate(notes):
        c=active(cs,float(n.offset))
        if c is None or n.pitch.pitchClass in c.pitchClasses:continue
        dissonant.append(i)
        if i==0 or i==len(notes)-1:unresolved.append(i);continue
        before=abs(int(diffs[i-1]));after=abs(int(diffs[i]));nxt=notes[i+1];nc=active(cs,float(nxt.offset))
        gap=float(nxt.offset-n.offset-n.quarterLength)
        if not (before in [1,2] and after in [1,2] and gap<=.125 and nc is not None and nxt.pitch.pitchClass in nc.pitchClasses):unresolved.append(i)
    leaps=[i for i,d in enumerate(diffs) if abs(d)>=7];unrecovered=[];large_by_bar={}
    for i,d in enumerate(diffs):
        if abs(d)>5:
            index=int(float(notes[i+1].offset)//bar);large_by_bar[index]=large_by_bar.get(index,0)+1
        if abs(d)>=7 and (i+1>=len(diffs) or not (d*diffs[i+1]<0 and abs(diffs[i+1]) in [1,2])):unrecovered.append(i)
    sections=[]
    for section in range(4):
        ids=[i for i,n in enumerate(notes) if section*4*bar<=float(n.offset)<(section+1)*4*bar]
        sections.append(float(np.average(pitches[ids],weights=durations[ids])) if ids else float(np.mean(pitches)))
    peak_bars=sorted(set(int(float(n.offset)//bar)+1 for n in notes if n.pitch.midi==int(pitches.max())))
    order=np.argsort(pitches);cumulative=np.cumsum(durations[order])/durations.sum()
    def quantile(q):return int(pitches[order][min(np.searchsorted(cumulative,q),len(order)-1)])
    total=float(mel.highestTime)
    return {'strong_beat_chord_tone_ratio':1-len(strong_bad)/max(1,strong_eligible),
            'unresolved_dissonance_rate':len(unresolved)/max(1,len(dissonant)),
            'pitch_range_semitones':int(np.ptp(pitches)),'stepwise_motion_ratio':float(np.mean(np.isin(abs(diffs),[1,2]))),
            'leap_recovery_rate':1-len(unrecovered)/max(1,len(leaps)),
            'large_leap_rate':float(np.mean(abs(diffs)>5)),
            'multiple_leap_bar_ratio':sum(c>1 for c in large_by_bar.values())/max(1,total/bar),
            'arch_climax_lift':sections[2]-sections[0],'arch_return_drop':sections[2]-sections[3],
            'effective_pitch_range':quantile(.9)-quantile(.1),
            'range_preference':max(0.,1-max(10-int(np.ptp(pitches)),int(np.ptp(pitches))-16,0)/12),
            'stepwise_preference':max(0.,1-max(.65-float(np.mean(np.isin(abs(diffs),[1,2]))),float(np.mean(np.isin(abs(diffs),[1,2])))-.85,0)/.5),
            'fragmentation_rate':sum(float(n.quarterLength)<.5-1e-5 for n in notes)/len(notes),
            'section_centroids':sections,'peak_bars':peak_bars,'strong_beat_violations':strong_bad,
            'unresolved_note_indices':unresolved,'unrecovered_leap_indices':unrecovered,
            'large_leaps_per_bar':large_by_bar,'pitch_min':int(pitches.min()),'pitch_max':int(pitches.max())}


def passes(diagnostics,pop=False):
    d=diagnostics
    return bool(d['strong_beat_chord_tone_ratio']==1 and d['unresolved_dissonance_rate']==0 and
                d['leap_recovery_rate']==1 and d['multiple_leap_bar_ratio']==0 and
                (13 if pop else 10)<=d['pitch_range_semitones']<=19 and d['effective_pitch_range']>=8 and
                .65<=d['stepwise_motion_ratio']<=.85 and 5<=d['arch_climax_lift']<=8 and
                d['arch_return_drop']>=4 and all(9<=bar<=12 for bar in d['peak_bars']))
