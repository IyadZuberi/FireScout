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
    assert m.nsensor == 36 and m.ncam == 2  # thermal + chase
    for _ in range(10):
        mujoco.mj_step(m, d)


def test_robot_moves_with_ctrl():
    m, d = load(generate_layout(1))
    x0 = d.qpos[0]
    d.ctrl[0] = 0.5
    for _ in range(200):
        mujoco.mj_step(m, d)
    assert d.qpos[0] > x0 + 0.3
