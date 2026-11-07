import numpy as np

from brain import Brain
from report import methane
from test_brain import obs, person_frame


def test_methane_has_seven_lines():
    b = Brain(12, 8, (1.0, 4.0))
    b.step(obs())
    lines = methane(b).strip().splitlines()
    assert [l[:2] for l in lines] == ['M:', 'E:', 'T:', 'H:', 'A:', 'N:', 'E:']


def test_zero_casualties_never_clear():
    b = Brain(12, 8, (1.0, 4.0))
    for k in range(10):
        b.step(obs(t=k * 0.05))
    text = methane(b).lower()
    assert 'clear' not in text
    assert 'human search still required' in text


def test_pinned_and_possible_listed():
    b = Brain(12, 8, (2.0, 4.0))
    th, dp = person_frame()
    for k in range(4):
        b.step(obs(t=k * 0.05, pose=(2.0, 4.0, 0.0), thermal=th, depth=dp))
    for k in range(20):
        b.step(obs(t=0.25 + k * 0.05, pose=(2.0, 4.0, 0.0), thermal=th, depth=dp, breath=0.9))
    text = methane(b)
    assert 'N: 1 confirmed' in text and 'breathing' in text
