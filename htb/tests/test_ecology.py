import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from wildcast.ecology import kernel_coordinates
from wildcast.landscape_context import cost_graph, movement_cost
from wildcast.marine import past_months


def test_past_month_selection_crosses_year_without_using_origin_month():
    seconds = pd.to_datetime(["2020-01-01T00:00Z", "2020-03-31T23:59Z"]).astype("int64").to_numpy() / 1e9
    np.testing.assert_array_equal(past_months(seconds), [2019 * 12 + 10, 2020 * 12])


def test_actual_phylogeny_missing_tips_stay_zero_and_coordinates_deterministic():
    distance = np.asarray([[np.nan, np.nan, np.nan], [np.nan, 0., 2.], [np.nan, 2., 0.]])
    observed = np.asarray([False, True, True])
    coordinates = kernel_coordinates(distance, observed)
    assert (coordinates[0] == 0).all()
    np.testing.assert_allclose(coordinates, kernel_coordinates(distance, observed))
    assert np.linalg.norm(coordinates[1] - coordinates[2]) > 0


def test_phylogeny_extension_does_not_refit_on_heldout_species():
    distances = np.asarray([[0., 2., 3.], [2., 0., 4.], [3., 4., 0.]])
    training = np.asarray([True, True, False])
    full = kernel_coordinates(distances, np.ones(3, bool), training=training)
    fit = kernel_coordinates(distances[:2, :2], np.ones(2, bool))
    np.testing.assert_allclose(full[:2], fit, atol=1e-7)
    assert np.isfinite(full).all() and (full[2] != 0).any()


def test_local_cost_path_detours_and_guilds_have_finite_preferences():
    cost = np.ones((7, 7)); cost[1:6, 3] = 100
    graph = cost_graph(cost, resolution=1)
    distance = dijkstra(graph, indices=3 * 7)[3 * 7 + 6]
    assert 6 < distance < 15  # routes around the barrier
    land = np.zeros((3, 3), bool); land[:, 0] = True
    costs = [movement_cost(land, np.zeros((3, 3)), np.zeros((3, 3)), mode) for mode in [0, 1, 2]]
    assert costs[0][0, 2] > costs[0][0, 0]  # terrestrial water penalty
    assert costs[2][0, 0] > costs[2][0, 2]  # seal land penalty
    assert all(np.isfinite(c).all() and (c > 0).all() for c in costs)
