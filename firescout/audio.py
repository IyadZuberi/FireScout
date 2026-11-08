"""Casualty sounds the robot's microphones pick up, played on the console speakers.

Voice clips are made once with Windows speech synthesis (SAPI, offline) and a synthesised laboured-breathing
loop; each clip is stored at 3 loudness levels so playback matches the level the mic array measured.
Build:  python audio.py      Playback: winsound (Windows stdlib); silently disabled elsewhere.
"""
from __future__ import annotations

import os
import subprocess
import time
import wave

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'audio')
SR = 22050
LEVELS = {'far': 0.22, 'mid': 0.5, 'near': 1.0}
PHRASES = {
    'help_f': ('Help! Help me, please!', 'medium', 'slow'),
    'here_f': ("I'm in here... help!", 'medium', 'slow'),
    'help_m': ('Help! Somebody help me!', 'x-low', 'slow'),
    'here_m': ("In here! I can't breathe...", 'x-low', 'slow'),
}


def _save(path: str, x: np.ndarray) -> None:
    x = np.clip(x, -1, 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype('<i2').tobytes())


def _load(path: str) -> np.ndarray:
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32767


def _lowpass(x: np.ndarray, k: int) -> np.ndarray:
    """Moving-average low-pass: far / through-wall voices lose their highs."""
    return np.convolve(x, np.ones(k) / k, mode='same') if k > 1 else x


def _breathing(seconds: float = 6.4, seed: int = 0) -> np.ndarray:
    """Laboured breathing: band-limited noise, slow inhale / longer exhale, a little throat rasp."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    breath = _lowpass(noise, 6) - _lowpass(noise, 60)          # crude band-pass ~300 Hz - 3.5 kHz
    cyc = 3.2
    ph = (t % cyc) / cyc
    env = np.where(ph < 0.38, np.sin(np.pi * ph / 0.38) * 0.6, np.where(ph < 0.45, 0.0,
                   np.sin(np.pi * (ph - 0.45) / 0.55)))
    rasp = 1 + 0.35 * np.sin(2 * np.pi * 34 * t)
    x = breath * env ** 1.5 * rasp
    return 0.9 * x / np.abs(x).max()


def _robot_voice(key: str, text: str) -> None:
    """Robot loudspeaker announcement: a different voice (Zira), clipped band like a small speaker."""
    raw = os.path.join(DIR, f'_{key}.wav')
    ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$v=$s.GetInstalledVoices()|%{$_.VoiceInfo.Name}|?{$_ -like '*Zira*'}; if($v){$s.SelectVoice($v)}; "
          "$s.Rate=-1; $f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050,"
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
          f"$s.SetOutputToWaveFile('{raw}',$f); $s.Speak('{text}'); $s.Dispose()")
    subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=True)
    x = _load(raw)
    os.remove(raw)
    x = _lowpass(x, 2) - _lowpass(x, 40)  # tinny loudspeaker band
    _save(os.path.join(DIR, f'{key}.wav'), 0.95 * x / max(1e-6, np.abs(x).max()))


def build() -> None:
    os.makedirs(DIR, exist_ok=True)
    from brain import SAY_TEXT
    _robot_voice('robot_evac', SAY_TEXT['EVACUATE'])
    _robot_voice('robot_shelter', SAY_TEXT['SHELTER'])
    for key, (text, pitch, rate) in PHRASES.items():
        raw = os.path.join(DIR, f'_{key}.wav')
        ssml = (f"<speak version='1.0' xml:lang='en-GB'><prosody pitch='{pitch}' rate='{rate}'>{text}"
                f"</prosody></speak>")
        ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "$v=$s.GetInstalledVoices()|%{$_.VoiceInfo.Name}|?{$_ -like '*Hazel*'}; if($v){$s.SelectVoice($v)}; "
              "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050,"
              "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
              f"$s.SetOutputToWaveFile('{raw}',$f); $s.SpeakSsml(\"{ssml}\"); $s.Dispose()")
        subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=True)
        x = _load(raw)
        os.remove(raw)
        x = 0.95 * x / max(1e-6, np.abs(x).max())
        for lvl, gain in LEVELS.items():
            _save(os.path.join(DIR, f'{key}_{lvl}.wav'), _lowpass(x, {'far': 5, 'mid': 2, 'near': 1}[lvl]) * gain)
    br = _breathing()
    for lvl, gain in LEVELS.items():
        _save(os.path.join(DIR, f'breath_{lvl}.wav'), br * gain)


def level_name(db: float) -> str:
    return 'near' if db >= 55 else 'mid' if db >= 42 else 'far'


class Player:
    """Plays one clip at a time, asynchronously. on=False or no winsound -> no-op."""

    def __init__(self, on: bool = True):
        try:
            import winsound
            self.ws = winsound
        except ImportError:
            self.ws = None
        self.on = on and self.ws is not None and os.path.isdir(DIR)
        self.busy_until = 0.0

    def play(self, clip: str, db: float, seconds: float = 2.6) -> bool:
        now = time.monotonic()
        if not self.on or now < self.busy_until:
            return False
        path = os.path.join(DIR, f'{clip}_{level_name(db)}.wav')
        if not os.path.exists(path):
            return False
        self.ws.PlaySound(path, self.ws.SND_FILENAME | self.ws.SND_ASYNC | self.ws.SND_NODEFAULT)
        self.busy_until = now + seconds
        return True

    def stop(self):
        if self.ws is not None:
            self.ws.PlaySound(None, 0)


if __name__ == '__main__':
    build()
    print(sorted(os.listdir(DIR)))
    assert level_name(60) == 'near' and level_name(45) == 'mid' and level_name(31) == 'far'
