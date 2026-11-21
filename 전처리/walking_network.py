"""Seoul walking distances from the Geofabrik 2025-01-01 OpenStreetMap snapshot.

Map data © OpenStreetMap contributors, ODbL 1.0. Distances are metres in EPSG:5186.
"""
from pathlib import Path
import numpy as np
import osmium
import shapely
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

PUBLIC = {"yes", "designated", "permissive", "official"}
RESTRICTED = {"no", "private", "customers", "permit", "delivery", "agricultural", "forestry"}
EXCLUDED = {"motorway", "motorway_link", "trunk", "trunk_link", "raceway", "cycleway",
            "construction", "proposed", "abandoned", "platform", "bus_guideway", "busway"}


def accessible(tags):
    road = tags.get("highway")
    foot = tags.get("foot")
    if not road or road in {"construction", "proposed", "abandoned"} or tags.get("area") == "yes":
        return False
    if foot in PUBLIC:
        return True
    return (road not in EXCLUDED and foot not in RESTRICTED and tags.get("access") not in RESTRICTED
            and tags.get("motorroad") != "yes" and tags.get("sidewalk") != "separate")


class WalkWays(osmium.SimpleHandler):
    def __init__(self, bounds):
        super().__init__()
        self.west, self.south, self.east, self.north = bounds
        self.coords = {}
        self.edges = set()
        self.blocked = set()

    def node(self, node):
        if not node.tags or not node.location.valid():
            return
        x, y = node.location.lon, node.location.lat
        if not (self.west <= x <= self.east and self.south <= y <= self.north):
            return
        tags = node.tags
        if tags.get("foot") in RESTRICTED or (tags.get("foot") not in PUBLIC and
                (tags.get("access") in RESTRICTED or tags.get("barrier") in {"wall", "fence"})):
            self.blocked.add(node.id)

    def way(self, way):
        if not accessible(way.tags):
            return
        previous = None
        for node in way.nodes:
            if not node.location.valid() or node.ref in self.blocked:
                previous = None
                continue
            x, y = node.location.lon, node.location.lat
            if not (self.west <= x <= self.east and self.south <= y <= self.north):
                previous = None
                continue
            self.coords[node.ref] = (x, y)
            if previous is not None and previous != node.ref:
                self.edges.add(tuple(sorted((previous, node.ref))))
            previous = node.ref


def graph_of(edges, lengths, n):
    a, b = edges.T
    return coo_matrix((np.r_[lengths, lengths], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()


def station_distances(stations, pbf):
    lon = stations["경도"].to_numpy()
    lat = stations["위도"].to_numpy()
    ways = WalkWays((lon.min()-.07, lat.min()-.055, lon.max()+.07, lat.max()+.055))
    ways.apply_file(str(Path(pbf)), locations=True, idx="flex_mem")
    keys = sorted(ways.coords)
    index = {key: i for i, key in enumerate(keys)}
    tx = Transformer.from_crs(4326, 5186, always_xy=True)
    xy = np.column_stack(tx.transform(*np.array([ways.coords[key] for key in keys]).T))
    edges = np.array([(index[a], index[b]) for a, b in sorted(ways.edges)], dtype=np.int32)
    lengths = np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1)
    _, component = connected_components(graph_of(edges, lengths, len(xy)), directed=False)
    main = np.bincount(component).argmax()
    keep = (component[edges[:, 0]] == main) & (lengths > 0)
    edges, lengths = edges[keep], lengths[keep]
    # Each station joins the nearest segment on the main component at its projected point.
    point_xy = np.column_stack(tx.transform(lon, lat))
    points = shapely.points(point_xy)
    segments = shapely.linestrings(xy[edges])
    tree = shapely.STRtree(segments)
    nearest = tree.nearest(points)
    along = shapely.line_locate_point(segments[nearest], points)
    access = shapely.distance(segments[nearest], points)
    ends = np.flatnonzero(~np.isin(np.arange(len(edges)), np.unique(nearest)))
    extra_edges, extra_lengths = [], []
    for segment in np.unique(nearest):
        station_rows = np.flatnonzero(nearest == segment)
        order = station_rows[np.argsort(along[station_rows], kind="stable")]
        nodes = np.r_[edges[segment, 0], len(xy) + order, edges[segment, 1]]
        positions = np.r_[0., along[order], lengths[segment]]
        extra_edges.extend(zip(nodes[:-1], nodes[1:]))
        extra_lengths.extend(np.maximum(np.diff(positions), 0.))
    graph = graph_of(np.vstack([edges[ends], extra_edges]), np.r_[lengths[ends], extra_lengths],
                     len(xy) + len(points))
    targets = np.arange(len(xy), graph.shape[0])
    distances = np.empty((len(points), len(points)))
    for start in range(0, len(points), 12):
        subset = targets[start:start+12]
        distances[start:start+len(subset)] = dijkstra(graph, indices=subset)[:, targets]
    distances += access[:, None] + access[None, :]
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)
    assert np.isfinite(distances).all() and (distances >= 0).all()
    stats = {"eligible_segments": len(ways.edges), "main_component_segments": len(edges),
             "stations": len(points), "snap_median_m": float(np.median(access)),
             "snap_p95_m": float(np.quantile(access, .95)), "snap_max_m": float(access.max())}
    return distances, access, stats
