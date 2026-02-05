#!/usr/bin/env python3

from __future__ import annotations

import random
import re
import statistics
import subprocess
import time
from pathlib import Path

import cdd
import howzat
import numpy as np

KIND = "drum"
SEED0 = 123
N = 10
V = 11
REPEATS = 10
BACKEND = "howzat-dd:f64"


def hirsch_bin() -> Path:
    return Path(__file__).resolve().parents[2] / "target" / "release-lto" / "hirsch"


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


def run_bench_times_seconds(kind: str, seed: int, n: int, v: int, backend_spec: str, repeats: int) -> list[float]:
    bin_path = hirsch_bin()
    if not bin_path.is_file():
        raise RuntimeError(f"missing `{bin_path}`; build with `cargo build -p kompute-hirsch --profile release-lto`")

    cmd = [
        str(bin_path),
        "sandbox",
        "bench",
        "--kind",
        kind,
        "--seed",
        str(seed),
        "--repeats",
        str(repeats),
        "--num-workers",
        "1",
        "--verbose",
        "--backend",
        f"^{backend_spec}",
        "--option",
        f"n={n}",
        "--option",
        f"v={v}",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)

    out: list[float] = []
    for raw in proc.stdout.splitlines():
        raw = raw.strip()
        if not raw.startswith("bench run "):
            continue
        m = re.search(r"\btime=([0-9.]+)(ns|us|µs|ms|s)\b", raw)
        if m is None:
            raise RuntimeError(f"failed to parse time from bench line:\n{raw}")

        value = float(m.group(1))
        unit = m.group(2)
        if unit == "s":
            out.append(value)
        elif unit == "ms":
            out.append(value / 1_000.0)
        elif unit in ("us", "µs"):
            out.append(value / 1_000_000.0)
        elif unit == "ns":
            out.append(value / 1_000_000_000.0)
        else:
            raise RuntimeError(f"unknown time unit {unit!r} in bench line:\n{raw}")

    if len(out) != repeats:
        raise RuntimeError(f"expected {repeats} bench runs, got {len(out)}:\n{proc.stdout}")
    return out


def main() -> int:
    samples = [sample_vertices(KIND, SEED0 + i, N, V) for i in range(REPEATS)]
    backend = howzat.Backend(BACKEND)

    wall: list[float] = []
    backend_seconds: list[float] = []
    for verts, _ in samples:
        start = time.perf_counter()
        out = backend.solve(verts)
        wall.append(time.perf_counter() - start)
        backend_seconds.append(out.total_seconds)

    bench_seconds = run_bench_times_seconds(KIND, SEED0, N, V, BACKEND, REPEATS)
    bench_med = statistics.median(bench_seconds)
    verts0, attempts0 = samples[0]

    print(
        f"sample kind={KIND} seed0={SEED0} repeats={REPEATS} n={N} v={V} "
        f"verts={verts0.shape[0]} dim={verts0.shape[1]}"
    )
    print(f"sample attempts0={attempts0}")
    if KIND == "drum":
        print(f"drum split top={V} bot={N + 1}")
    print(f"backend={BACKEND}")
    print(
        "python howzat.Backend.solve "
        f"wall_med={statistics.median(wall):.6f}s wall_min={min(wall):.6f}s wall_max={max(wall):.6f}s "
        f"backend_med={statistics.median(backend_seconds):.6f}s backend_min={min(backend_seconds):.6f}s "
        f"backend_max={max(backend_seconds):.6f}s "
        f"call_overhead_med={statistics.median([w - b for w, b in zip(wall, backend_seconds)]):+.6f}s"
    )
    print(
        f"hirsch sandbox bench time_med={bench_med:.6f}s time_min={min(bench_seconds):.6f}s "
        f"time_max={max(bench_seconds):.6f}s"
    )
    print(f"ratio backend/bench={statistics.median(backend_seconds) / bench_med:.3f}x")
    print(f"ratio wall/bench={statistics.median(wall) / bench_med:.3f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
