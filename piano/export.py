"""Export the approved score, optionally rendering with a sampled piano."""
import argparse
import json
import os
from pathlib import Path
from .midi import write_midi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', type=Path, default=Path(__file__).with_name('score.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/light-and-return'))
    parser.add_argument('--render', action='store_true', help='Also render WAV using FluidSynth')
    parser.add_argument('--soundfont', default=os.environ.get(
        'PIANO_SOUNDFONT', '/usr/share/sounds/sf2/FluidR3_GM.sf2'))
    args = parser.parse_args()
    score = json.loads(args.score.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'light_and_return_short.mid'
    write_midi(score, path)
    print(path)
    if args.render:
        from .render import render
        render(path, path.with_suffix('.wav'), soundfont=args.soundfont)
        print(path.with_suffix('.wav'))


if __name__ == '__main__':
    main()
