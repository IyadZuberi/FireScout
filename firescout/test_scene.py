import mujoco

from scene import generate_layout, load


def test_layout_deterministic():
    assert generate_layout(3) == generate_layout(3)


def test_layout_has_required_things():
    for seed in range(20):
        L = generate_layout(seed)
        kinds = [t.kind for t in L.things]
        assert 1 <= kinds.count('fire') <= 2
        assert kinds.count('shelf') == 1
        assert kinds.count('decoy') >= 2
        assert kinds.count('casualty') <= 2


def test_things_clear_of_start():
    for seed in range(20):
        L = generate_layout(seed)
        for t in L.things:
            assert (t.xy[0] - L.start[0]) ** 2 + (t.xy[1] - L.start[1]) ** 2 >= 1.0


def test_mjcf_loads_and_steps():
    m, d = load(generate_layout(1))
    assert m.ncam == 2  # thermal + chase (radar is a batched ray cast in sensors.py)
    for _ in range(10):
        mujoco.mj_step(m, d)


def test_robot_moves_with_ctrl():
    m, d = load(generate_layout(1))
    x0 = d.qpos[0]
    d.ctrl[0] = 0.5
    for _ in range(200):
        mujoco.mj_step(m, d)
    assert d.qpos[0] > x0 + 0.3


def test_presets_counts_and_load():
    from scene import PRESETS
    for k, preset in enumerate(PRESETS):
        L = generate_layout(k, preset=preset, n_casualties=3, n_fires=2)
        kinds = [t.kind for t in L.things]
        assert kinds.count('casualty') == 3 and kinds.count('fire') == 2, preset
        assert L.width == PRESETS[preset]['width'] and len(L.rooms) == len(PRESETS[preset]['rooms'])
        assert L.preset == preset
        load(L)


def test_some_casualties_call_out():
    cas = [t for s in range(15) for t in generate_layout(s, n_casualties=3).things if t.kind == 'casualty']
    assert any(t.responsive for t in cas) and not all(t.responsive for t in cas)


def test_flames_are_many_tongues():
    L = generate_layout(2, n_fires=1)
    xml = __import__('scene').build_mjcf(L)
    assert xml.count('material="flame') >= 10
