import numpy as np

from brain import Grid, astar, cost_map, frontiers, integrate_ranges, rescue_route


# ---------------- Task 3: map + planners

def open_grid(w=4, h=4):
    g = Grid(w, h)
    g.occ[:] = 0
    return g


def test_astar_avoids_no_go():
    g = open_grid()
    g.heat[6:10, 0:12] = 600  # hot band with a gap on the right (x 3..4 m)
    c = cost_map(g)
    p = astar(c, g.to_cell(0.5, 0.5), g.to_cell(0.5, 3.5))
    assert p is not None and all(np.isfinite(c[i, j]) for i, j in p)
    assert max(j for _, j in p) >= 12  # went through the gap


def test_astar_none_when_fully_blocked():
    g = open_grid()
    g.heat[6:10, :] = 600
    assert astar(cost_map(g), g.to_cell(0.5, 0.5), g.to_cell(0.5, 3.5)) is None


def test_fire_cell_itself_is_no_go():
    g = open_grid()
    g.heat[8, 8] = 600  # a single small fire blob
    assert not np.isfinite(cost_map(g)[8, 8])


def test_integrate_ranges_marks_wall():
    g = Grid(6, 6)
    r = np.full(36, 6.0)
    r[0] = 2.0  # ray 0 points along robot heading (+x)
    integrate_ranges(g, (1.0, 3.0, 0.0), r)
    assert g.occ[g.to_cell(3.0, 3.0)] == 1
    assert g.occ[g.to_cell(2.0, 3.0)] == 0


def test_frontiers_found_at_known_unknown_border():
    g = Grid(4, 4)
    g.occ[0:4, 0:4] = 0
    f = frontiers(g)
    assert f and all(i <= 3 and j <= 3 for i, j in f)


def test_rescue_route_orders_targets():
    g = open_grid(8, 4)
    c = cost_map(g)
    start = g.to_cell(0.5, 2.0)
    near, far = g.to_cell(3.0, 2.0), g.to_cell(7.0, 2.0)
    order, path = rescue_route(c, start, [far, near])
    assert order == [1, 0] and path[0] == start and path[-1] == far
