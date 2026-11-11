"""Read performed MIDI exactly; never quantize note timing during preview or evaluation."""
from music21 import converter


def read_midi(path):
    return converter.parse(path,quantizePost=False,forceSource=True)
