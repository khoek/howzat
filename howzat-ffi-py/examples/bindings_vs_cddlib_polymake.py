#!/usr/bin/env python3

from __future__ import annotations

import random
import shutil
import statistics
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cdd
import howzat
import numpy as np

KIND = "drum"
SEED = 123
N = 10
V = 11
REPEATS = 10
BACKEND = "howzat-dd:f64"


def standard_simplex_vertices(dim: int) -> np.ndarray:
    bot = np.zeros((dim + 1, dim), dtype=np.float64)
    bot[np.arange(dim) + 1, np.arange(dim)] = 1.0
    return bot


def embed_with_height(points: np.ndarray, height: float) -> np.ndarray:
    out = np.empty((points.shape[0], points.shape[1] + 1), dtype=np.float64)
    out[:, :-1] = points
    out[:, -1] = height
    return out


def drum_vertices(top: np.ndarray, bot: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([embed_with_height(top, 1.0), embed_with_height(bot, 0.0)], axis=0)
    )


def sample_random_vertices(dim: int, count: int, seed: int) -> tuple[np.ndarray, int]:
    if dim == 1 and count > 2:
        raise ValueError("cannot sample >2 non-redundant vertices in 1D")

    rng = random.Random(seed)
    points: list[list[float]] = []
    attempts = 0

    while len(points) < count:
        attempts += 1
        candidate = [-10.0 + 20.0 * rng.random() for _ in range(dim)]
        rows = [[1.0, *p] for p in points]
        rows.append([1.0, *candidate])
        if not cdd.redundant_rows(cdd.matrix_from_array(rows, rep_type=cdd.RepType.GENERATOR)):
            points.append(candidate)

    return np.asarray(points, dtype=np.float64), attempts


def sample_vertices(kind: str, seed: int, n: int, v: int) -> tuple[np.ndarray, int]:
    if kind == "points":
        verts, attempts = sample_random_vertices(n, v, seed)
        return np.ascontiguousarray(verts), attempts
    if kind == "drum":
        top, attempts = sample_random_vertices(n, v, seed)
        return drum_vertices(top, standard_simplex_vertices(n)), attempts
    raise ValueError(f"unsupported kind {kind!r} (expected 'points' or 'drum')")


def normalize_sets(sets) -> list[list[int]]:
    return [sorted(s) for s in sets]


def graph_to_lists(graph: object) -> list[list[int]]:
    node_count = graph.node_count()
    return [graph.neighbors(i) for i in range(node_count)]


def fr_edges(
    facets_to_vertices: list[list[int]],
    facet_adjacency: list[list[int]],
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    keys = [tuple(face) for face in facets_to_vertices]
    edges: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for i, neigh in enumerate(facet_adjacency):
        for j in neigh:
            if j <= i:
                continue
            a, b = keys[i], keys[j]
            edges.add((a, b) if a <= b else (b, a))
    return edges


def undirected_edges(adjacency: list[list[int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, neigh in enumerate(adjacency):
        for j in neigh:
            if j <= i:
                continue
            edges.add((i, j))
    return edges


def cdd_solve(verts: np.ndarray, exact: bool) -> dict[str, object]:
    array = (
        [[Fraction(1, 1), *(Fraction.from_float(float(x)) for x in row)] for row in verts]
        if exact
        else [[1.0, *(float(x) for x in row)] for row in verts]
    )
    mat = cdd.matrix_from_array(array, rep_type=cdd.RepType.GENERATOR)
    poly = cdd.polyhedron_from_matrix(mat)
    return {
        "facets_to_vertices": normalize_sets(cdd.copy_incidence(poly)),
        "facet_adjacency": normalize_sets(cdd.copy_adjacency(poly)),
        "vertex_adjacency": normalize_sets(cdd.copy_input_adjacency(poly)),
    }


def maybe_polymake_solve(verts: np.ndarray) -> dict[str, object] | None:
    polymake = shutil.which("polymake")
    if polymake is None:
        return None

    def fmt(x: float) -> str:
        if not np.isfinite(x):
            raise ValueError(f"non-finite coordinate: {x!r}")
        return repr(float(x))

    pts = "[" + ",".join("[" + ",".join(["1.0", *(fmt(x) for x in row)]) + "]" for row in verts) + "]"
    script = (
        'print "###HOWZAT_BEGIN###\\n";'
        f"my $pts = {pts};"
        'my $p = new Polytope<Float>(POINTS=>$pts);'
        'print "VERTICES\\n";'
        "print $p->VERTICES;"
        'print "V_ADJ\\n";'
        "print $p->GRAPH->ADJACENCY;"
        'print "F_ADJ\\n";'
        "print $p->DUAL_GRAPH->ADJACENCY;"
        'print "V_IN_F\\n";'
        "print $p->VERTICES_IN_FACETS;"
        'print "###HOWZAT_END###\\n";'
    )

    proc = subprocess.run([polymake, "-A", "polytope", script], text=True, capture_output=True)
    if proc.returncode != 0 or "ERROR:" in proc.stderr:
        raise RuntimeError(f"polymake failed:\n{proc.stderr}\n{proc.stdout}")

    stdout = proc.stdout.splitlines()
    try:
        begin = stdout.index("###HOWZAT_BEGIN###")
        end = stdout.index("###HOWZAT_END###")
    except ValueError:
        raise RuntimeError(f"failed to parse polymake output:\n{proc.stdout}") from None

    def parse_set(line: str) -> list[int]:
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            raise ValueError(f"expected set line, got {line!r}")
        inner = line[1:-1].strip()
        return [] if not inner else [int(tok) for tok in inner.split()]

    section = None
    vertices_hom: list[list[float]] = []
    vertex_adjacency: list[list[int]] = []
    facet_adjacency: list[list[int]] = []
    facets_to_vertices: list[list[int]] = []

    for raw in stdout[begin + 1 : end]:
        raw = raw.strip()
        if not raw:
            continue
        if raw in ("VERTICES", "V_ADJ", "F_ADJ", "V_IN_F"):
            section = raw
            continue
        if section == "VERTICES":
            vertices_hom.append([float(tok) for tok in raw.split()])
        elif section == "V_ADJ":
            vertex_adjacency.append(parse_set(raw))
        elif section == "F_ADJ":
            facet_adjacency.append(parse_set(raw))
        elif section == "V_IN_F":
            facets_to_vertices.append(parse_set(raw))

    if not vertices_hom:
        raise RuntimeError("polymake returned no vertices")

    return {
        "vertices_hom": vertices_hom,
        "vertex_adjacency": normalize_sets(vertex_adjacency),
        "facet_adjacency": normalize_sets(facet_adjacency),
        "facets_to_vertices": normalize_sets(facets_to_vertices),
        "polymake_bin": str(Path(polymake)),
    }


def verify(verts: np.ndarray, how: object, cdd_exact: dict[str, object], poly: dict[str, object] | None) -> None:
    how_facets = normalize_sets(how.facets_to_vertices)
    how_f_adj = normalize_sets(graph_to_lists(how.facet_adjacency))
    how_v_adj = normalize_sets(graph_to_lists(how.vertex_adjacency))

    if sorted(map(tuple, how_facets)) != sorted(map(tuple, cdd_exact["facets_to_vertices"])):
        raise RuntimeError("facet sets mismatch between howzat and pycddlib (exact)")
    if fr_edges(how_facets, how_f_adj) != fr_edges(cdd_exact["facets_to_vertices"], cdd_exact["facet_adjacency"]):
        raise RuntimeError("facet adjacency (FR graph) mismatch between howzat and pycddlib (exact)")
    if undirected_edges(how_v_adj) != undirected_edges(cdd_exact["vertex_adjacency"]):
        raise RuntimeError("vertex adjacency mismatch between howzat and pycddlib (exact)")

    print(
        "pycddlib exact match=OK "
        f"FR_edges={len(fr_edges(how_facets, how_f_adj))} V_edges={len(undirected_edges(how_v_adj))}"
    )

    if poly is None:
        print("polymake: unavailable")
        return

    verts_h = np.empty((verts.shape[0], verts.shape[1] + 1), dtype=np.float64)
    verts_h[:, 0] = 1.0
    verts_h[:, 1:] = verts

    poly_vertices = np.asarray(poly["vertices_hom"], dtype=np.float64)
    if poly_vertices.shape != verts_h.shape:
        raise RuntimeError(f"polymake vertices shape mismatch: {poly_vertices.shape} != {verts_h.shape}")

    mapping: dict[int, int] = {}
    used: set[int] = set()
    tol = 1e-9
    for i, row in enumerate(poly_vertices):
        for j, want in enumerate(verts_h):
            if j in used:
                continue
            if np.max(np.abs(row - want)) <= tol:
                mapping[i] = j
                used.add(j)
                break
        else:
            raise RuntimeError("polymake vertex order mismatch: unable to map vertices back to input indices")

    poly_facets = [[mapping[v] for v in face] for face in poly["facets_to_vertices"]]
    poly_v_adj = [[] for _ in range(verts.shape[0])]
    for i, neigh in enumerate(poly["vertex_adjacency"]):
        poly_v_adj[mapping[i]] = [mapping[v] for v in neigh]

    if sorted(map(tuple, how_facets)) != sorted(map(tuple, normalize_sets(poly_facets))):
        raise RuntimeError("facet sets mismatch between howzat and polymake")
    if fr_edges(how_facets, how_f_adj) != fr_edges(normalize_sets(poly_facets), poly["facet_adjacency"]):
        raise RuntimeError("facet adjacency (FR graph) mismatch between howzat and polymake")
    if undirected_edges(how_v_adj) != undirected_edges(normalize_sets(poly_v_adj)):
        raise RuntimeError("vertex adjacency mismatch between howzat and polymake")

    print(f"polymake match=OK bin={poly['polymake_bin']}")


def main() -> int:
    verts, attempts = sample_vertices(KIND, SEED, N, V)
    print(
        f"running sample kind={KIND} seed={SEED} attempts={attempts} n={N} v={V} "
        f"verts={verts.shape[0]} dim={verts.shape[1]}",
        flush=True,
    )

    backend = howzat.Backend(BACKEND)
    how_wall: list[float] = []
    how_backend: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        how = backend.solve(verts)
        how_wall.append(time.perf_counter() - start)
        how_backend.append(how.total_seconds)

    cdd_exact_wall: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        cdd_exact = cdd_solve(verts, exact=True)
        cdd_exact_wall.append(time.perf_counter() - start)

    cdd_float_wall: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        cdd_float = cdd_solve(verts, exact=False)
        cdd_float_wall.append(time.perf_counter() - start)

    poly_wall: list[float] | None = None
    poly = None
    if shutil.which("polymake") is not None:
        poly_wall = []
        for _ in range(min(REPEATS, 3)):
            start = time.perf_counter()
            poly = maybe_polymake_solve(verts)
            poly_wall.append(time.perf_counter() - start)

    print(f"sample kind={KIND} seed={SEED} attempts={attempts} n={N} v={V} verts={verts.shape[0]} dim={verts.shape[1]}")
    if KIND == "drum":
        print(f"drum split top={V} bot={N + 1}")
    print(f"backend={BACKEND}")
    print(f"howzat.solve facets={how.facets} ridges={how.ridges}")
    print(
        "howzat.solve "
        f"wall_med={statistics.median(how_wall):.6f}s backend_med={statistics.median(how_backend):.6f}s "
        f"call_overhead_med={statistics.median([w - b for w, b in zip(how_wall, how_backend)]):+.6f}s"
    )
    print(f"pycddlib exact wall_med={statistics.median(cdd_exact_wall):.6f}s")
    print(f"pycddlib float wall_med={statistics.median(cdd_float_wall):.6f}s")
    if poly_wall is not None:
        print(f"polymake wall_med={statistics.median(poly_wall):.6f}s")

    verify(verts, how, cdd_exact, poly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
