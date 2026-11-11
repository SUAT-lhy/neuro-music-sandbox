"""Write an event score directly, preserving notes, pedal and tempo changes."""
import mido

PPQ = 960


def write_midi(score, path):
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    events = [(0, 0, mido.MetaMessage('track_name', name=score['title']))]
    for entry in score['tempo_map']:
        events.append((round(entry['quarter'] * PPQ), 1,
                       mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(entry['bpm']))))
    for entry in score['meter_map']:
        events.append((round(entry['quarter'] * PPQ), 2, mido.MetaMessage(
            'time_signature', numerator=entry['numerator'], denominator=entry['denominator'])))
    for entry in score['key_map']:
        events.append((round(entry['quarter'] * PPQ), 3,
                       mido.MetaMessage('key_signature', key=entry['key'])))
    end = round((score['duration_quarters'] + score.get('release_quarters', 4)) * PPQ)
    events.append((end, 9, mido.MetaMessage('end_of_track')))
    _append(meta, events)
    for channel, notes in [(0, score['right']), (1, score['left'] + score['inner'])]:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change', channel=channel, program=0))
        track.append(mido.Message('control_change', channel=channel, control=7,
                                  value=110 if channel == 0 else 82))
        track.append(mido.Message('control_change', channel=channel, control=10,
                                  value=72 if channel == 0 else 53))
        events = []
        for position in score['pedal_positions']:
            tick = round(position * PPQ)
            events.append((tick, 1, mido.Message('control_change', channel=channel, control=64, value=0)))
            if position < score['duration_quarters']:
                events.append((tick, 2, mido.Message('control_change', channel=channel, control=64, value=127)))
        for note in notes:
            start = round(note['start'] * PPQ)
            stop = round((note['start'] + note['duration']) * PPQ)
            if not 0 <= start < stop <= end:
                raise ValueError('Note outside score duration')
            events.append((start, 3, mido.Message('note_on', channel=channel,
                                                  note=note['pitch'], velocity=note['velocity'])))
            events.append((stop, 0, mido.Message('note_off', channel=channel,
                                                 note=note['pitch'], velocity=0)))
        events.append((end, 9, mido.MetaMessage('end_of_track')))
        _append(track, events)
    mid.save(path)


def _append(track, events):
    previous = 0
    for tick, _, message in sorted(events, key=lambda entry: (entry[0], entry[1])):
        track.append(message.copy(time=tick - previous))
        previous = tick
