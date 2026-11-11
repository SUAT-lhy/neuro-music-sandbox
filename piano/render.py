"""Render MIDI through a sampled piano with actual CC64 and moderate stereo reverb."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf


def render(source,target,soundfont='/usr/share/sounds/sf2/FluidR3_GM.sf2',reverb=True,master=True):
    executable=shutil.which('fluidsynth')
    if executable is None:raise RuntimeError('FluidSynth is required; no additive-synthesis fallback is used for sampled-piano rendering')
    font=Path(soundfont)
    if not font.exists():raise FileNotFoundError(font)
    target=Path(target);target.parent.mkdir(exist_ok=True,parents=True)
    settings={'room_size':.48,'damping':.42,'width':.65,'level':.23}
    for gain in [.45,.28,.16]:
        command=[executable,'-ni','-q','-r','22050','-g',str(gain),'-R','1' if reverb else '0','-C','0',
                 '-o','synth.polyphony=256','-o',f"synth.reverb.room-size={settings['room_size']}",
                 '-o',f"synth.reverb.damp={settings['damping']}",'-o',f"synth.reverb.width={settings['width']}",
                 '-o',f"synth.reverb.level={settings['level']}",'-T','wav','-F',str(target),str(font),str(source)]
        completed=subprocess.run(command,check=True,capture_output=True,text=True)
        audio,rate=sf.read(target,always_2d=True)
        clipped=float(np.mean(abs(audio)>=.9999))
        if clipped==0:break
    else:raise RuntimeError('Piano renderer still clips after gain reduction')
    raw_peak=float(abs(audio).max());master_gain=.88/raw_peak if master and raw_peak>0 else 1.
    audio*=master_gain
    if master:
        sf.write(target,audio,rate,subtype='PCM_16')
        audio,rate=sf.read(target,always_2d=True)
    metadata={'engine':'FluidSynth','soundfont':str(font),'soundfont_sha256':hashlib.sha256(font.read_bytes()).hexdigest(),
              'program':'Acoustic Grand Piano (GM program 0)','sample_rate':rate,'channels':audio.shape[1],
              'seconds':len(audio)/rate,'peak':float(abs(audio).max()),'rms':float(np.sqrt(np.mean(audio**2))),
              'clipped_sample_fraction':clipped,'gain':gain,'reverb_enabled':reverb,'reverb_settings':settings,
              'raw_peak':raw_peak,'master_gain':master_gain,'master_gain_db':float(20*np.log10(master_gain)),
              'mastering':'constant gain only; no compressor, limiter or change to relative dynamics',
              'pedal':'CC64 interpreted directly by the synthesizer; MIDI not imported or quantized by music21',
              'command':command}
    target.with_suffix('.render.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata

