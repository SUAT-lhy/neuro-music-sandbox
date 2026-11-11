"""Timing, contour and harmonic descriptors independent of key/transposition."""
from collections import Counter
import numpy as np
from music21 import chord,meter,note


def entropy(values):
    if not len(values):return 0.
    counts=np.array(list(Counter(values).values()),dtype=float);p=counts/counts.sum()
    return float(-(p*np.log2(p)).sum())


def bar_length(part):
    signatures=list(part.recurse().getElementsByClass(meter.TimeSignature))
    return float(signatures[0].barDuration.quarterLength) if signatures else 4.


def rhythm_stats(mel,chd):
    events=list(mel.flatten().notesAndRests);notes=[n for n in events if isinstance(n,note.Note)]
    bar=bar_length(mel);total=max(float(mel.highestTime),1.)
    # Quantisation suppresses floating-point residues and articulation microgaps.
    durations=[max(1,round(float(n.quarterLength)*12)) for n in notes]
    iois=[max(1,round(float(b.offset-a.offset)*12)) for a,b in zip(notes,notes[1:])]
    patterns=[]
    for start in np.arange(0,total-1e-6,bar):
        patterns.append(tuple((round(float(n.offset-start)*12),round(float(n.quarterLength)*12),n.isRest)
                              for n in events if start<=float(n.offset)<start+bar and (not n.isRest or n.quarterLength>=.125)))
    variation=sum(a!=b for a,b in zip(patterns,patterns[1:]))/max(1,len(patterns)-1)
    repeat=1-len(set(patterns))/max(1,len(patterns))
    sync=sum(abs(float(n.offset)-round(float(n.offset)))>.05 and
             float(n.offset+n.quarterLength)>np.ceil(float(n.offset))+.05 for n in notes)/max(1,len(notes))
    rest=sum(float(n.quarterLength) for n in events if n.isRest and n.quarterLength>=.125)/total
    chords=list(chd.flatten().getElementsByClass(chord.Chord))
    pcs=[tuple(sorted(set(c.pitchClasses))) for c in chords]
    changes=sum(a!=b for a,b in zip(pcs,pcs[1:]))/max(1,total/bar)
    signatures=list(mel.recurse().getElementsByClass(meter.TimeSignature))
    signature=signatures[0] if signatures else meter.TimeSignature('4/4')
    compound=signature.denominator==8 and signature.numerator>=6 and signature.numerator%3==0
    strong_times=np.arange(0,total,1.5 if compound else 1.)
    clashes,eligible=0,0
    for t in strong_times:
        sounding=next((n for n in notes if float(n.offset)<=t<float(n.offset+n.quarterLength)),None)
        harmony=next((c for c in reversed(chords) if float(c.offset)<=t<float(c.offset+c.quarterLength)),None)
        if sounding is not None and harmony is not None:
            eligible+=1;clashes+=sounding.pitch.pitchClass not in harmony.pitchClasses
    return dict(duration_entropy=entropy(durations),ioi_entropy=entropy(iois),cross_bar_rhythm_change=float(variation),
                syncopation_ratio=float(sync),rest_ratio=float(rest),harmonic_change_rate=float(changes),
                bar_repetition_ratio=float(repeat),strong_beat_clash_ratio=clashes/max(1,eligible))
