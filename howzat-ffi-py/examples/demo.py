#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Callable
from collections import deque
import time

import howzat
import numpy as np

SANTOS_TOP_BASE = np.asarray(
    [
        [18.0, 0.0, 0.0, 0.0],
        [-18.0, 0.0, 0.0, 0.0],
        [0.0, 18.0, 0.0, 0.0],
        [0.0, -18.0, 0.0, 0.0],
        [0.0, 0.0, 45.0, 0.0],
        [0.0, 0.0, -45.0, 0.0],
        [0.0, 0.0, 0.0, 45.0],
        [0.0, 0.0, 0.0, -45.0],
        [15.0, 15.0, 0.0, 0.0],
        [-15.0, 15.0, 0.0, 0.0],
        [15.0, -15.0, 0.0, 0.0],
        [-15.0, -15.0, 0.0, 0.0],
        [0.0, 0.0, 30.0, 30.0],
        [0.0, 0.0, -30.0, 30.0],
        [0.0, 0.0, 30.0, -30.0],
        [0.0, 0.0, -30.0, -30.0],
        [0.0, 10.0, 40.0, 0.0],
        [0.0, -10.0, 40.0, 0.0],
        [0.0, 10.0, -40.0, 0.0],
        [0.0, -10.0, -40.0, 0.0],
        [10.0, 0.0, 0.0, 40.0],
        [-10.0, 0.0, 0.0, 40.0],
        [10.0, 0.0, 0.0, -40.0],
        [-10.0, 0.0, 0.0, -40.0],
    ],
    dtype=np.float64,
)

SANTOS_BOT_BASE = np.asarray(
    [
        [0.0, 0.0, 0.0, 18.0],
        [0.0, 0.0, 0.0, -18.0],
        [0.0, 0.0, 18.0, 0.0],
        [0.0, 0.0, -18.0, 0.0],
        [45.0, 0.0, 0.0, 0.0],
        [-45.0, 0.0, 0.0, 0.0],
        [0.0, 45.0, 0.0, 0.0],
        [0.0, -45.0, 0.0, 0.0],
        [0.0, 0.0, 15.0, 15.0],
        [0.0, 0.0, 15.0, -15.0],
        [0.0, 0.0, -15.0, 15.0],
        [0.0, 0.0, -15.0, -15.0],
        [30.0, 30.0, 0.0, 0.0],
        [-30.0, 30.0, 0.0, 0.0],
        [30.0, -30.0, 0.0, 0.0],
        [-30.0, -30.0, 0.0, 0.0],
        [40.0, 0.0, 10.0, 0.0],
        [40.0, 0.0, -10.0, 0.0],
        [-40.0, 0.0, 10.0, 0.0],
        [-40.0, 0.0, -10.0, 0.0],
        [0.0, 40.0, 0.0, 10.0],
        [0.0, 40.0, 0.0, -10.0],
        [0.0, -40.0, 0.0, 10.0],
        [0.0, -40.0, 0.0, -10.0],
    ],
    dtype=np.float64,
)

WILLIAMSON_K1_MOTIF = np.asarray(
    [
        [0.0, 0.0, 3.0, 3.0, 1.0],
        [98.0, 0.0, 1.0, 0.0, 1.0],
        [100.0, 0.0, 0.0, 0.0, 1.0],
        [75.0, 75.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

WILLIAMSON_K4_MOTIF = np.asarray(
    [
        [0.0, 0.0, 3.0, 3.0, 1.0],
        [98.0, 0.0, 1.0, 0.0, 1.0],
        [100.0, 0.0, 0.0, 0.0, 1.0],
        [92.625_394_156_403_93, 23.393_453_858_754_036, 0.0, 0.0, 1.0],
        [85.693_360_076_337_29, 45.047_070_135_648_63, 0.0, 0.0, 1.0],
        [79.594_659_910_781_46, 62.140_358_293_966_72, 0.0, 0.0, 1.0],
        [75.0, 75.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def embed_with_height(points: np.ndarray, height: float) -> np.ndarray:
    out = np.empty((points.shape[0], points.shape[1] + 1), dtype=np.float64)
    out[:, :-1] = points
    out[:, -1] = height
    return out


def drum_vertices(top: np.ndarray, bot: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([embed_with_height(top, 1.0), embed_with_height(bot, 0.0)], axis=0)
    )


def williamson_bases(motif: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    orbit: list[tuple[float, float, float, float, float]] = []
    queue: deque[tuple[float, float, float, float, float]] = deque()

    for row in motif:
        v = tuple(float(x) for x in row)
        if v not in orbit:
            orbit.append(v)
            queue.append(v)

    while queue:
        v = queue.popleft()
        for i in range(4):
            flipped = list(v)
            flipped[i] = -flipped[i]
            candidate = tuple(flipped)
            if candidate not in orbit:
                orbit.append(candidate)
                queue.append(candidate)

        tau = (v[2], v[3], v[1], v[0], -v[4])
        if tau not in orbit:
            orbit.append(tau)
            queue.append(tau)

    top: list[list[float]] = []
    bot: list[list[float]] = []
    for v in orbit:
        (top if v[4] > 0.0 else bot).append([v[0], v[1], v[2], v[3]])

    return np.asarray(top, dtype=np.float64), np.asarray(bot, dtype=np.float64)


def bfs_distance(adjacency: object, start: int, goal: int) -> int:
    if start == goal:
        return 0

    dist = [-1] * len(adjacency)
    dist[start] = 0
    q: deque[int] = deque([start])

    while q:
        u = q.popleft()
        next_dist = dist[u] + 1
        for v in adjacency.neighbors(u):
            if dist[v] != -1:
                continue
            dist[v] = next_dist
            if v == goal:
                return next_dist
            q.append(v)

    raise RuntimeError("disconnected facet graph")


def print_stats(label: str, solve: Callable[[], object]) -> object:
    then = time.perf_counter()
    out = solve()
    print(f"{label}:")
    print(f"  time={time.perf_counter() - then:.6f}s")
    print(
        f"  verts={out.vertices} dim={out.dimension} facets={out.facets} ridges={out.ridges} backend={out.spec}"
    )
    return out


def base_facets(out: object, verts: np.ndarray) -> tuple[int, int]:
    top = tuple(map(int, np.flatnonzero(verts[:, -1] == 1.0)))
    bot = tuple(map(int, np.flatnonzero(verts[:, -1] == 0.0)))
    faces = {tuple(face): i for i, face in enumerate(out.facets_to_vertices)}
    try:
        return faces[top], faces[bot]
    except KeyError as e:
        raise RuntimeError("failed to locate top/bot base facets") from e


def calc_width(verts: np.ndarray, out: object) -> int:
    return bfs_distance(out.facet_adjacency, *base_facets(out, verts))


def assert_width_is(label: str, verts: np.ndarray, out: object, true_width: int) -> int:
    computed_width = calc_width(verts, out)
    assert true_width == computed_width, f"{true_width} != {computed_width}"
    print(f"{label}: width={computed_width}")


def main() -> int:
    # Assemble the vertex arrays
    santos = drum_vertices(SANTOS_TOP_BASE, SANTOS_BOT_BASE)
    will1 = drum_vertices(*williamson_bases(WILLIAMSON_K1_MOTIF))
    will4 = drum_vertices(*williamson_bases(WILLIAMSON_K4_MOTIF))

    # To solve one of these vertex sets, simply run:
    out = howzat.solve(santos)

    # There are high performance helpers to compute e.g. drum width from the FR graph, but for
    # simplicity here let's just compute the width here in python:
    width = calc_width(santos, out)

    # The solve above used the default backend I picked out. Below we'll demo running a few
    # different solves of a few polytopes with several backends by specifying their algorithm
    # represented as a string. You might want to do this to switch from an exact to an inexact (say,
    # f64 mode)---though I have made several improvements to the DD core which appear to make this
    # much, much less necessary in practice---or to compare the performance of different backends.
    # See my attached graph for the scaling performance of these as the polytopes grow.
    #
    # For the full list of backend pipeline config parameters---including setting vector
    # normalization mode ("max", "min", "none", and for rational numeric types "gcd"), using higher
    # than 64-bit (though still finite) floating point numbers, or changing the float fuzzy equality
    # epsilon value (e.g. "f64[eps(1e-12)]"), consult the python package documentation or just ask
    # me. :)
    #
    # Each invocation is wrapped in a `print_stats()` to print some simple time-taken stats:

    # - Solve Santos (`santos`) with default backend (spec: "howzat-dd[purify[snap]]:f64[eps[1e-12]]" (default))
    santos_out = print_stats(
        "santos",
        lambda: howzat.solve(santos)
    )

    # - Solve Williamson-k1 (`will1`) with explicit backend selection which disables "normal
    #   snapping" as a countermeasure against accumulated f64 error (spec: "howzat-dd:f64").
    will1_plain_out = print_stats(
        "williamson-k1 (snapping off)",
        lambda: howzat.Backend("howzat-dd:f64").solve(will1),
    )

    # - Solve Williamson-k1 (`will1`) again via solving in f64, then re-solving the
    #   combinatorial output exactly with rationals and repairing if necessary
    #   (spec: "howzat-dd:f64-repair[gmprat]"). Of course, as you can see repair is not
    #   required in this case.
    will1_repair_out = print_stats(
        "williamson-k1 (snapping off, repair)",
        lambda: howzat.Backend("howzat-dd:f64-repair[gmprat]").solve(will1),
    )

    # - Solve Williamson-k4 (`will4`) via the default backend (spec: "howzat-dd[purify[snap]]:f64[eps[1e-12]]").
    #
    #   TIME: ~0.063s on my machine
    will4_default_out = print_stats(
        "williamson-k4 (default backend)",
        lambda: howzat.solve(will4)
    )

    # - Solve Williamson-k4 (`will4`) again via an exact version of the previous DD algorithm
    #   (spec: "howzat-dd:gmprat").
    #
    #   TIME: ~0.443s on my machine
    will4_exact_dd_out = print_stats(
        "williamson-k4 (howzat-dd:gmprat)",
        lambda: howzat.Backend("howzat-dd:gmprat").solve(will4),
    )

    # - Solve Williamson-k4 (`will4`) again exactly via LRS (spec: "howzat-lrs:rug").
    #
    #   TIME: ~0.494s on my machine
    will4_exact_lrs_out = print_stats(
        "williamson-k4 (howzat-lrs:rug)",
        lambda: howzat.Backend("howzat-lrs:rug").solve(will4),
    )

    # - Solve Williamson-k4 (`will4`) again via pycddlib-like exact cddlib backend, which is my
    #   best guess for what Geordie had previously been using (spec: "cddlib:gmprational").
    #
    #   NOTE: Despite having many other problems I'd be very interested to talk about, the FR graph
    #         solver built into this library is busted and has catastrophic scaling. The variant
    #         ("cddlib+hlbl:gmprational") replaces that algorithm with the one in `hullabaloo` (our
    #         library), solving that particular problem.
    #
    #   TIME: ~1.567s on my machine
    will4_cdd_out = print_stats(
        "williamson-k4 (cddlib:gmprational)",
        lambda: howzat.Backend("cddlib:gmprational").solve(will4),
    )

    # - Solve Williamson-k4 (`will4`) exactly again via lrslib (spec: "lrslib+hlbl")
    #
    #   NOTE: This one uses my hullabaloo incidence graph calculation library,
    #         because lrslib doesn't ship with a builtin for that AFAICS.
    #
    #   TIME: ~0.476s on my machine
    will4_lrs_out = print_stats(
        "williamson-k4 (lrslib+hlbl)",
        lambda: howzat.Backend("lrslib+hlbl").solve(will4),
    )

    assert_width_is("santos", santos, santos_out, 6)
    assert_width_is("williamson-k1 (snapping off)", will1, will1_plain_out, 6)
    assert_width_is("williamson-k1 (snapping off, repair)", will1, will1_repair_out, 6)
    assert_width_is("williamson-k4 (default backend)", will4, will4_default_out, 9)
    assert_width_is("williamson-k4 (howzat-dd:gmprat)", will4, will4_exact_dd_out, 9)
    assert_width_is("williamson-k4 (howzat-lrs:rug)", will4, will4_exact_lrs_out, 9)
    assert_width_is("williamson-k4 (cddlib:gmprational)", will4, will4_cdd_out, 9)
    assert_width_is("williamson-k4 (lrslib+hlbl)", will4, will4_lrs_out, 9)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
