#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import collate_and_plot as plot_util


RESULT_RE = re.compile(r"^RESULT\s+(?P<body>.*)$")


@dataclass
class ResultRow:
    kind: str
    seed0: int
    mode: str
    n: int
    v: int
    repeats: int
    backend: str
    total_s: float
    avg_s: float
    fails: int
    fallbacks: int
    errors: int
    mismatches: int
    issues: int
    mem_avg_bytes: int
    mem_max_bytes: int


def parse_result_lines(lines: Iterable[str]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for line in lines:
        m = RESULT_RE.match(line.rstrip("\n"))
        if not m:
            continue
        body = m.group("body").strip()
        fields: dict[str, str] = {}
        for tok in body.split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            fields[k] = v

        def req(name: str) -> str:
            if name not in fields:
                raise ValueError(f"missing field '{name}' in RESULT line: {line!r}")
            return fields[name]

        mode = fields.get("mode", "adjacency")
        fails = int(req("fails"))
        mismatches = int(req("mismatches"))
        repeats = int(req("repeats"))
        errors = int(fields.get("errors", "0"))
        issues_s = fields.get("issues")
        issues = int(issues_s) if issues_s is not None else min(repeats, fails + mismatches)
        rows.append(
            ResultRow(
                kind=req("kind"),
                seed0=int(req("seed0")),
                mode=mode,
                n=int(req("n")),
                v=int(req("v")),
                repeats=repeats,
                backend=req("backend"),
                total_s=float(req("total_s")),
                avg_s=float(req("avg_s")),
                fails=fails,
                fallbacks=int(req("fallbacks")),
                errors=errors,
                mismatches=mismatches,
                issues=issues,
                mem_avg_bytes=int(req("mem_avg_bytes")),
                mem_max_bytes=int(req("mem_max_bytes")),
            )
        )
    return rows


def read_all_results(input_dir: Path) -> list[ResultRow]:
    logs = sorted(input_dir.rglob("*.log"))
    out: list[ResultRow] = []
    for path in logs:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            out.extend(parse_result_lines(lines))
        except Exception as exc:
            print(f"W: failed to parse {path}: {exc}")
    return out


def write_csv(rows: list[ResultRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "kind",
                "seed0",
                "mode",
                "n",
                "v",
                "repeats",
                "backend",
                "issues",
                "issue_rate",
                "mismatches",
                "mismatch_rate",
                "fails",
                "fail_rate",
                "errors",
                "error_rate",
                "fallbacks",
                "fallback_rate",
                "avg_s",
                "total_s",
            ]
        )
        for r in sorted(
            rows, key=lambda r: (r.kind, r.seed0, r.mode, r.backend, r.n, r.v, r.repeats)
        ):
            issue_rate = (r.issues / r.repeats) if r.repeats else 0.0
            mismatch_rate = (r.mismatches / r.repeats) if r.repeats else 0.0
            fail_rate = (r.fails / r.repeats) if r.repeats else 0.0
            error_rate = (r.errors / r.repeats) if r.repeats else 0.0
            fallback_rate = (r.fallbacks / r.repeats) if r.repeats else 0.0
            w.writerow(
                [
                    r.kind,
                    r.seed0,
                    r.mode,
                    r.n,
                    r.v,
                    r.repeats,
                    r.backend,
                    r.issues,
                    f"{issue_rate:.12f}",
                    r.mismatches,
                    f"{mismatch_rate:.12f}",
                    r.fails,
                    f"{fail_rate:.12f}",
                    r.errors,
                    f"{error_rate:.12f}",
                    r.fallbacks,
                    f"{fallback_rate:.12f}",
                    f"{r.avg_s:.9f}",
                    f"{r.total_s:.9f}",
                ]
            )


def plot_issue_fraction_by_n(rows: list[ResultRow], out_dir: Path) -> None:
    plot_util.ensure_plotly()
    import plotly.graph_objects as go

    out_dir.mkdir(parents=True, exist_ok=True)
    by_mode: dict[str, list[ResultRow]] = {}
    for row in rows:
        by_mode.setdefault(row.mode, []).append(row)

    pattern_shapes = ["", "/", "x"]
    if not by_mode:
        fig = plot_util.plotly_empty_figure(
            "Failure/mismatch rate vs ambient dimension",
            "",
            "Ambient dimension <i>n</i>",
            "Fraction of runs with issues",
        )
        plot_util.write_plotly_images(
            fig,
            out_dir / "issues_by_n.svg",
            out_dir / "issues_by_n.png",
        )
        plot_util.flush_plotly_images()
        return

    for mode, mode_rows in sorted(by_mode.items(), key=lambda kv: kv[0]):
        mode_rows = [r for r in mode_rows if r.kind == "drum" and r.v == r.n + 1]
        if not mode_rows:
            continue

        seeds = sorted({r.seed0 for r in mode_rows})
        ns = sorted({r.n for r in mode_rows})
        backends = plot_util.backend_order({r.backend for r in mode_rows})
        color_map = plot_util.plotly_color_map(backends)
        label_map = {backend: plot_util.backend_legend_label(backend) for backend in backends}

        by_backend_seed_n: dict[tuple[str, int, int], ResultRow] = {}
        denom_by_backend_n: dict[tuple[str, int], int] = {}
        issues_by_backend_n: dict[tuple[str, int], int] = {}
        for r in mode_rows:
            by_backend_seed_n[(r.backend, r.seed0, r.n)] = r
            key = (r.backend, r.n)
            denom_by_backend_n[key] = denom_by_backend_n.get(key, 0) + r.repeats
            issues_by_backend_n[key] = issues_by_backend_n.get(key, 0) + r.issues

        max_total = 0.0
        for key, denom in denom_by_backend_n.items():
            if denom <= 0:
                continue
            issues = issues_by_backend_n.get(key, 0)
            max_total = max(max_total, issues / denom)

        seeds_desc = ", ".join(
            f"{seed}={pattern_shapes[i % len(pattern_shapes)] or 'solid'}"
            for i, seed in enumerate(seeds)
        )
        subtitle = (
            f"kind=drum mode={mode} v=n+1 (issues=fail|error|mismatch; "
            f"stack: seed0={seeds_desc}; height is aggregate issue fraction)"
        )

        fig = go.Figure()
        fig.update_layout(barmode="stack", bargap=0.18, bargroupgap=0.04)

        for backend_idx, backend in enumerate(backends):
            color = color_map.get(backend, "#444444")
            for seed_idx, seed0 in enumerate(seeds):
                xs: list[int] = []
                ys: list[float] = []
                customdata: list[list[float]] = []
                for n in ns:
                    row = by_backend_seed_n.get((backend, seed0, n))
                    if row is None:
                        continue
                    denom = denom_by_backend_n.get((backend, n), 0)
                    if denom <= 0:
                        continue
                    issue_rate = (row.issues / row.repeats) if row.repeats else 0.0
                    contrib = (row.issues / denom) if denom else 0.0
                    xs.append(n)
                    ys.append(contrib)
                    customdata.append(
                        [
                            float(row.issues),
                            float(row.repeats),
                            float(issue_rate),
                            float(row.fails),
                            float(row.errors),
                            float(row.mismatches),
                            float(row.fallbacks),
                        ]
                    )

                if not xs:
                    continue

                pattern = pattern_shapes[seed_idx % len(pattern_shapes)]
                marker = (
                    dict(color=color, line=dict(color=color, width=1))
                    if pattern == ""
                    else dict(
                        color="#ffffff",
                        line=dict(color=color, width=1),
                        pattern=dict(
                            shape=pattern,
                            fgcolor=color,
                            bgcolor="rgba(0,0,0,0)",
                            solidity=0.25,
                            size=5,
                        ),
                    )
                )

                showlegend = seed_idx == 0
                trace = go.Bar(
                    x=xs,
                    y=ys,
                    offsetgroup=backend,
                    legendgroup=backend,
                    name=label_map.get(backend, backend),
                    marker=marker,
                    customdata=customdata,
                    hovertemplate=(
                        f"backend=%{{legendgroup}}<br>seed0={seed0}<br>n=%{{x}}"
                        "<br>issue_rate=%{customdata[2]:.6g}"
                        "<br>issues=%{customdata[0]:.0f} repeats=%{customdata[1]:.0f}"
                        "<br>fails=%{customdata[3]:.0f} errors=%{customdata[4]:.0f}"
                        "<br>mismatches=%{customdata[5]:.0f} fallbacks=%{customdata[6]:.0f}<extra></extra>"
                    ),
                    showlegend=showlegend,
                )
                if showlegend:
                    trace.legendrank = backend_idx
                fig.add_trace(trace)

        fig.update_xaxes(dtick=1)
        y_max = max_total * 1.1 if max_total > 0 else 1.0
        fig.update_yaxes(range=[0.0, min(1.0, y_max)])
        plot_util.plotly_apply_backend_legend_labels(fig, label_map)
        plot_util.plotly_apply_style(
            fig,
            "Failure/mismatch rate vs ambient dimension",
            subtitle,
            "Ambient dimension <i>n</i>",
            "Fraction of runs with issues",
            log_y=False,
            show_legend=True,
            legend_items=len(backends),
        )
        plot_util.write_plotly_images(
            fig,
            out_dir / f"issues_by_n_{mode}.svg",
            out_dir / f"issues_by_n_{mode}.png",
        )

    plot_util.flush_plotly_images()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("artifacts_fr"))
    ap.add_argument("--out-dir", type=Path, default=Path("bench_fr_errors_out"))
    args = ap.parse_args()

    rows = read_all_results(args.input_dir)
    out_dir: Path = args.out_dir
    write_csv(rows, out_dir / "fr_errors.csv")
    plot_issue_fraction_by_n(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
