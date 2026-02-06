#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass


NS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

REPEATS_BY_N = {
    4: 10_000,
    5: 1_000,
    6: 1_000,
    7: 1_000,
    8: 100,
    9: 100,
    10: 10,
    11: 10,
    12: 10,
    13: 10,
    14: 10,
}


SCHEDULE_TABLE = """
# backend                                n4  n5  n6  n7  n8  n9 n10 n11 n12 n13 n14
lrslib+hlbl:gmpint                       5   5   5   5   5   5   5   3   3   2   1
ppl+hlbl:gmpint                          5   5   5   5   5   5   5   3   3   2   1
cddlib:gmprational                       5   5   5   5   5   5   5   1   1   0   0
cddlib:gmpfloat                          5   5   5   5   5   5   5   1   1   0   0
cddlib:f64                               5   5   5   5   5   5   5   1   1   0   0
cddlib+hlbl:gmprational                  5   5   5   5   5   5   5   3   3   2   0
cddlib+hlbl:gmpfloat                     5   5   5   5   5   5   5   3   3   2   1
cddlib+hlbl:f64                          5   5   5   5   5   5   5   3   3   2   1
howzat-dd:f64                            5   5   5   5   5   5   5   3   3   2   1
howzat-dd:f64-repair[gmprat]             5   5   5   5   5   5   5   3   3   2   1
howzat-dd[purify[snap]]:f64              5   5   5   5   5   5   5   3   3   2   1
howzat-dd[purify[snap]]:f64-repair[gmprat] 5   5   5   5   5   5   5   3   3   2   1
howzat-dd:gmprat                         5   5   5   5   5   5   5   3   3   2   1
howzat-lrs:rug                           5   5   5   5   5   5   5   3   3   2   1
howzat-lrs:dashu                         5   5   5   5   5   5   5   3   3   2   1
""".strip()


@dataclass(frozen=True)
class Segment:
    extra_lo: int
    extra_hi: int


def parse_segments(spec: str) -> list[Segment]:
    spec = spec.strip()
    if spec == "0":
        return []

    if spec.isdigit():
        max_extra = int(spec)
        if max_extra <= 0:
            return []
        return [Segment(extra_lo=1, extra_hi=max_extra)]

    if "," in spec:
        endpoints = [int(x) for x in spec.split(",") if x.strip() != ""]
        if len(endpoints) < 2:
            raise ValueError(f"invalid segment spec {spec!r}: need >=2 comma-separated integers")
        if endpoints[0] != 1:
            raise ValueError(f"invalid segment spec {spec!r}: first endpoint must be 1")
        if endpoints[1] < endpoints[0]:
            raise ValueError(f"invalid segment spec {spec!r}: non-increasing endpoints")
        segments = [Segment(extra_lo=endpoints[0], extra_hi=endpoints[1])]
        prev_end = endpoints[1]
        for end in endpoints[2:]:
            lo = prev_end + 1
            hi = end
            if hi < lo:
                raise ValueError(f"invalid segment spec {spec!r}: non-increasing endpoints")
            segments.append(Segment(extra_lo=lo, extra_hi=hi))
            prev_end = hi
        return segments

    if "-" in spec:
        lo_s, hi_s = spec.split("-", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        if lo <= 0 or hi <= 0 or hi < lo:
            raise ValueError(f"invalid segment spec {spec!r}")
        return [Segment(extra_lo=lo, extra_hi=hi)]

    raise ValueError(f"invalid segment spec {spec!r}")


def parse_schedule_table(table: str) -> list[tuple[str, dict[int, list[Segment]]]]:
    schedule: list[tuple[str, dict[int, list[Segment]]]] = []
    for line_no, raw_line in enumerate(table.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        backend = parts[0]
        cells = parts[1:]
        if len(cells) != len(NS):
            raise ValueError(
                f"schedule row {line_no} ({backend}) has {len(cells)} cells, expected {len(NS)}"
            )

        by_n: dict[int, list[Segment]] = {}
        for n, spec in zip(NS, cells):
            by_n[n] = parse_segments(spec)
        schedule.append((backend, by_n))
    return schedule


def shard_id(n: int, seg: Segment) -> str:
    if seg.extra_lo == seg.extra_hi:
        return f"n{n}-e{seg.extra_lo}"
    return f"n{n}-e{seg.extra_lo}-{seg.extra_hi}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GitHub Actions matrix for benches.")
    parser.add_argument("--max-entries", type=int, default=256, help="Fail if include length exceeds this.")
    args = parser.parse_args()

    schedule = parse_schedule_table(SCHEDULE_TABLE)
    include: list[dict[str, object]] = []
    for backend, by_n in schedule:
        for n in NS:
            repeats = REPEATS_BY_N.get(n)
            if repeats is None:
                raise ValueError(f"missing repeats mapping for n={n}")
            for seg in by_n[n]:
                include.append(
                    {
                        "backend": backend,
                        "n": n,
                        "repeats": repeats,
                        "extra_lo": seg.extra_lo,
                        "extra_hi": seg.extra_hi,
                        "shard": shard_id(n, seg),
                    }
                )

    if len(include) > args.max_entries:
        raise ValueError(f"matrix include has {len(include)} entries > {args.max_entries}")

    matrix = {"include": include}
    print(json.dumps(matrix, separators=(",", ":")))
    print(f"generated matrix entries: {len(include)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
