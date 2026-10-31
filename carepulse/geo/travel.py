"""Travel-time providers.

Two providers share one interface, `matrix(points) -> minutes`:

* RoadNetworkTravel: real OpenStreetMap drive network (via OSMnx). Shortest
  paths use each road's speed, so a motorway counts as faster than a side road.
* EstimatedTravel: straight-line distance times a detour factor, at an average
  urban speed. Used when no road data is available (for example in CI).

Every run records which provider it used, so published numbers always say
whether they came from real roads or from the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


@dataclass
class EstimatedTravel:
    """Straight-line km x detour factor, at an average speed, plus parking time.

    1.35 is a common urban road-detour factor; 24 km/h is a typical average
    for short urban car trips including junctions. Both are assumptions.
    """
    detour: float = 1.35
    speed_kmh: float = 24.0
    parking_min: float = 3.0
    name: str = "estimated (straight line x 1.35 at 24 km/h)"

    def matrix(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        km = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) * self.detour
        minutes = km / self.speed_kmh * 60 + np.where(km > 0.05, self.parking_min, 0.0)
        return np.rint(minutes).astype(int)


class RoadNetworkTravel:
    """Shortest drive times on an OpenStreetMap road graph.

    Build once with `RoadNetworkTravel.download("Birmingham, UK", path)` on a
    machine with internet access, then load the saved file with
    `RoadNetworkTravel.load(path)`.
    """

    def __init__(self, graph, parking_min: float = 3.0, name: str = "OpenStreetMap road network"):
        self.G = graph
        self.parking_min = parking_min
        self.name = name

    @classmethod
    def download(cls, place: str, path: str | Path):
        import osmnx as ox
        G = ox.graph_from_place(place, network_type="drive")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        ox.save_graphml(G, path)
        return cls(G)

    @classmethod
    def load(cls, path: str | Path):
        import osmnx as ox
        return cls(ox.load_graphml(path))

    def _nearest_nodes(self, lat, lon):
        import osmnx as ox
        return ox.distance.nearest_nodes(self.G, X=list(lon), Y=list(lat))

    def matrix(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        import networkx as nx
        nodes = list(self._nearest_nodes(lat, lon))
        n = len(nodes)
        out = np.zeros((n, n))
        uniq = {u: i for i, u in enumerate(dict.fromkeys(nodes))}
        cache = {}
        for u in uniq:
            cache[u] = nx.single_source_dijkstra_path_length(self.G, u, weight="travel_time")
        far = 3600.0  # unreachable pairs (one-way traps): treat as an hour
        for i, u in enumerate(nodes):
            d = cache[u]
            for j, v in enumerate(nodes):
                out[i, j] = d.get(v, far)
        minutes = out / 60 + np.where(out > 0, self.parking_min, 0.0)
        return np.rint(minutes).astype(int)


def default_provider(graph_path: str | Path = "data/birmingham_drive.graphml"):
    p = Path(graph_path)
    if p.exists():
        return RoadNetworkTravel.load(p)
    return EstimatedTravel()
