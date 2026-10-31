import networkx as nx
import numpy as np

from carepulse.geo.travel import EstimatedTravel, RoadNetworkTravel, haversine_km


def test_haversine_known_distance():
    # Birmingham New Street to Coventry station is roughly 27 km in a straight line
    d = haversine_km(52.4778, -1.8990, 52.4008, -1.5134)
    assert 26 < d < 28


def test_estimated_matrix_shape_and_diagonal():
    T = EstimatedTravel().matrix([52.48, 52.50, 52.45], [-1.90, -1.85, -1.95])
    assert T.shape == (3, 3)
    assert (np.diag(T) == 0).all()
    assert (T == T.T).all()


def test_road_network_uses_one_way_streets():
    G = nx.MultiDiGraph(crs="epsg:4326")
    G.add_node(1, x=-1.90, y=52.48)
    G.add_node(2, x=-1.89, y=52.48)
    G.add_node(3, x=-1.895, y=52.49)
    G.add_edge(1, 2, travel_time=60)    # one-way 1 -> 2
    G.add_edge(2, 3, travel_time=120)
    G.add_edge(3, 1, travel_time=120)
    T = RoadNetworkTravel(G, parking_min=0).matrix(np.array([52.48, 52.48]), np.array([-1.90, -1.89]))
    assert T[0, 1] == 1     # direct, one minute
    assert T[1, 0] == 4     # has to go round, four minutes
