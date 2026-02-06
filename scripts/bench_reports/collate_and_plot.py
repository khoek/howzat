#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RESULT_RE = re.compile(r"^RESULT\s+(?P<body>.*)$")


@dataclass(frozen=True)
class ResultKey:
    kind: str
    mode: str
    n: int
    v: int
    repeats: int
    backend: str


@dataclass
class ResultRow:
    kind: str
    mode: str
    n: int
    v: int
    repeats: int
    backend: str
    total_s: float
    avg_s: float
    fails: int
    fallbacks: int
    mem_avg_bytes: int
    mem_max_bytes: int


@dataclass(frozen=True)
class PlotKey:
    kind: str
    mode: str
    n: int
    v: int
    backend: str


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
        rows.append(
            ResultRow(
                kind=req("kind"),
                mode=mode,
                n=int(req("n")),
                v=int(req("v")),
                repeats=int(req("repeats")),
                backend=req("backend"),
                total_s=float(req("total_s")),
                avg_s=float(req("avg_s")),
                fails=int(req("fails")),
                fallbacks=int(req("fallbacks")),
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
            text = path.read_text(encoding="utf-8", errors="replace").splitlines()
            out.extend(parse_result_lines(text))
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
                "mode",
                "n",
                "v",
                "repeats",
                "backend",
                "total_s",
                "avg_s",
                "fails",
                "fallbacks",
                "mem_avg_bytes",
                "mem_max_bytes",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.kind,
                    r.mode,
                    r.n,
                    r.v,
                    r.repeats,
                    r.backend,
                    f"{r.total_s:.9f}",
                    f"{r.avg_s:.9f}",
                    r.fails,
                    r.fallbacks,
                    r.mem_avg_bytes,
                    r.mem_max_bytes,
                ]
            )


def ensure_matplotlib():
    raise RuntimeError("matplotlib plots were removed; use plotly instead")


def ensure_plotly():
    import pandas  # noqa: F401
    import plotly.express  # noqa: F401
    import plotly.graph_objects  # noqa: F401
    import plotly.io  # noqa: F401


PLOT_BASE_WIDTH = 1920
PLOT_BASE_HEIGHT = 1080
PLOT_PNG_SCALE = 2
PLOT_HTML_DEFAULT_WIDTH = "100%"
PLOT_HTML_DEFAULT_HEIGHT = "95vh"
PLOT_LINE_WIDTH = 2
PLOT_MARKER_SIZE = 8
PLOT_MARKER_LINE_WIDTH = 0
PLOT_MARKER_SYMBOL = "x"

_PLOTLY_IMAGE_QUEUE: list[tuple[object, Path, Path, Path]] = []


def write_plotly_images(fig, out_svg: Path, out_png: Path) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_html = out_svg.with_suffix(".html")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    _PLOTLY_IMAGE_QUEUE.append((fig, out_svg, out_png, out_html))
    if len(_PLOTLY_IMAGE_QUEUE) >= 25:
        flush_plotly_images()


def flush_plotly_images() -> None:
    ensure_plotly()
    import plotly.io as pio

    if not _PLOTLY_IMAGE_QUEUE:
        return

    figs = [fig for fig, _, _, _ in _PLOTLY_IMAGE_QUEUE]
    svgs = [str(out_svg) for _, out_svg, _, _ in _PLOTLY_IMAGE_QUEUE]
    pngs = [str(out_png) for _, _, out_png, _ in _PLOTLY_IMAGE_QUEUE]

    pio.write_images(figs, svgs, format="svg", width=PLOT_BASE_WIDTH, height=PLOT_BASE_HEIGHT)
    pio.write_images(
        figs,
        pngs,
        format="png",
        width=PLOT_BASE_WIDTH,
        height=PLOT_BASE_HEIGHT,
        scale=PLOT_PNG_SCALE,
    )
    for fig, _, _, out_html in _PLOTLY_IMAGE_QUEUE:
        fig.update_layout(width=None, height=None, autosize=True)
        fig.write_html(
            out_html,
            include_plotlyjs="cdn",
            full_html=True,
            config=dict(responsive=True),
            default_width=PLOT_HTML_DEFAULT_WIDTH,
            default_height=PLOT_HTML_DEFAULT_HEIGHT,
        )
    _PLOTLY_IMAGE_QUEUE.clear()


def backend_group_name(backend: str) -> str:
    if backend.startswith("cddlib+hlbl:"):
        return "cddlib+hlbl"
    if backend.startswith("cddlib:"):
        return "cddlib"
    if backend.startswith("howzat-dd"):
        return "howzat-dd"
    if backend.startswith("howzat-lrs"):
        return "howzat-lrs"
    if backend.startswith("lrslib+hlbl:"):
        return "lrslib+hlbl"
    if backend.startswith("ppl+hlbl:"):
        return "ppl+hlbl"
    return "other"


def backend_sort_key(backend: str) -> tuple[int, str]:
    group_order = {
        "cddlib+hlbl": 0,
        "cddlib": 1,
        "howzat-dd": 2,
        "howzat-lrs": 3,
        "lrslib+hlbl": 4,
        "ppl+hlbl": 5,
        "other": 6,
    }
    group = backend_group_name(backend)
    return (group_order[group], backend)


def backend_order(backends: Iterable[str]) -> list[str]:
    return sorted(backends, key=backend_sort_key)


def backend_legend_label(backend: str) -> str:
    split = backend.rfind("-repair[")
    if split != -1:
        return f"{backend[:split]}<br>&nbsp;&nbsp;&nbsp;&nbsp;{backend[split:]}"
    return backend


def plotly_apply_backend_legend_labels(fig, label_map: dict[str, str]) -> None:
    for trace in getattr(fig, "data", []):
        showlegend = getattr(trace, "showlegend", True)
        if showlegend is False:
            continue
        key = getattr(trace, "legendgroup", None) or getattr(trace, "name", None)
        if not key:
            continue
        label = label_map.get(key)
        if label:
            trace.name = label


def backend_line_dash_map(backends_in_order: list[str]) -> dict[str, str]:
    return {backend: ("1px,3px" if i % 2 == 1 else "solid") for i, backend in enumerate(backends_in_order)}


def plotly_apply_backend_dashes(fig, dash_map: dict[str, str]) -> None:
    for trace in getattr(fig, "data", []):
        backend = getattr(trace, "legendgroup", None) or getattr(trace, "name", None)
        if not backend:
            continue
        dash = dash_map.get(backend)
        if dash:
            trace.line.dash = dash


def plotly_reorder_dotted_traces_last(fig, dash_map: dict[str, str], backend_order: list[str]) -> None:
    rank = {backend: idx for idx, backend in enumerate(backend_order)}
    traces = list(getattr(fig, "data", ()))
    for trace in traces:
        backend = getattr(trace, "legendgroup", None) or getattr(trace, "name", None)
        if backend in rank:
            trace.legendrank = rank[backend]

    solid: list[object] = []
    dotted: list[object] = []
    for trace in traces:
        backend = getattr(trace, "legendgroup", None) or getattr(trace, "name", None)
        dash = dash_map.get(backend, "solid")
        if dash != "solid":
            dotted.append(trace)
        else:
            solid.append(trace)
    fig.data = tuple(solid + dotted)


def plotly_remove_markers_from_legend(fig, backend_order: list[str]) -> None:
    ensure_plotly()
    import plotly.graph_objects as go

    rank = {backend: idx for idx, backend in enumerate(backend_order)}
    new_traces: list[object] = []
    for trace in list(getattr(fig, "data", ())):
        showlegend = getattr(trace, "showlegend", True)
        if showlegend is False:
            continue
        if getattr(trace, "type", None) != "scatter":
            continue
        mode = getattr(trace, "mode", "") or ""
        if "markers" not in mode:
            continue
        backend = getattr(trace, "legendgroup", None) or getattr(trace, "name", None)
        if not backend:
            continue
        line = getattr(trace, "line", None)
        legend_trace = go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=getattr(trace, "name", backend),
            legendgroup=backend,
            showlegend=True,
            hoverinfo="skip",
            line=dict(
                color=getattr(line, "color", None),
                dash=getattr(line, "dash", None),
                width=getattr(line, "width", None),
            ),
        )
        legend_rank = rank.get(backend)
        if legend_rank is not None:
            legend_trace.legendrank = legend_rank
        trace.showlegend = False
        new_traces.append(legend_trace)

    if new_traces:
        fig.add_traces(new_traces)


def plotly_color_map(backends_in_order: list[str]) -> dict[str, str]:
    ensure_plotly()
    import plotly.express as px

    group_order = [
        "cddlib+hlbl",
        "cddlib",
        "howzat-dd",
        "howzat-lrs",
        "lrslib+hlbl",
        "ppl+hlbl",
        "other",
    ]
    by_group: dict[str, list[str]] = defaultdict(list)
    for backend in backends_in_order:
        by_group[backend_group_name(backend)].append(backend)

    colors: dict[str, str] = {}
    groups_present = [g for g in group_order if by_group.get(g)]
    denom = max(1, len(groups_present))
    centers = {group: (idx + 0.5) / denom for idx, group in enumerate(groups_present)}
    for group in groups_present:
        backends = by_group[group]
        count = len(backends)
        if count == 1:
            positions = [centers[group]]
        else:
            span = min(0.22, 0.08 + 0.03 * (count - 1))
            start = max(0.0, centers[group] - span / 2.0)
            end = min(1.0, centers[group] + span / 2.0)
            step = (end - start) / (count - 1)
            positions = [start + i * step for i in range(count)]
        sampled = px.colors.sample_colorscale(px.colors.sequential.Turbo, positions)
        if len(backends) != len(sampled):
            raise ValueError(
                f"Color count mismatch for group {group}: "
                f"{len(backends)} backends, {len(sampled)} colors"
            )
        for backend, color in zip(backends, sampled):
            colors[backend] = color
    for backend in backends_in_order:
        colors.setdefault(backend, "#444444")
    return colors


def plotly_seconds_tick_text(value_s: float) -> str:
    if value_s <= 0:
        return "0s"
    if value_s < 1e-3:
        return f"{value_s * 1e6:g}µs"
    if value_s < 1:
        return f"{value_s * 1e3:g}ms"
    return f"{value_s:g}s"


def plotly_set_log_time_ticks(fig, values_s: Iterable[float]) -> None:
    vals = [v for v in values_s if v > 0]
    if not vals:
        return
    exp_min = math.floor(math.log10(min(vals)))
    exp_max = math.ceil(math.log10(max(vals)))
    exp_min = max(exp_min, -12)
    exp_max = min(exp_max, 12)
    tickvals = [10**k for k in range(exp_min, exp_max + 1)]
    ticktext = [plotly_seconds_tick_text(v) for v in tickvals]
    fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)


def plotly_set_log_numeric_ticks(fig, values: Iterable[float]) -> None:
    vals = [v for v in values if v > 0]
    if not vals:
        return
    exp_min = math.floor(math.log10(min(vals)))
    exp_max = math.ceil(math.log10(max(vals)))
    exp_min = max(exp_min, -12)
    exp_max = min(exp_max, 12)
    tickvals = [10**k for k in range(exp_min, exp_max + 1)]

    def fmt(v: float) -> str:
        if v >= 1_000_000 and v % 1_000_000 == 0:
            return f"{int(v / 1_000_000)}M"
        if v >= 1_000 and v % 1_000 == 0:
            return f"{int(v / 1_000)}k"
        return f"{v:g}"

    ticktext = [fmt(v) for v in tickvals]
    fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)


def plotly_legend_item_count(fig) -> int:
    seen: set[str] = set()
    for trace in getattr(fig, "data", []):
        showlegend = getattr(trace, "showlegend", True)
        if showlegend is False:
            continue
        key = getattr(trace, "legendgroup", None) or getattr(trace, "name", None)
        if key:
            seen.add(key)
    return len(seen)


def plotly_apply_style(
    fig,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    *,
    log_y: bool,
    show_legend: bool,
    width: int = PLOT_BASE_WIDTH,
    height: int = PLOT_BASE_HEIGHT,
    legend_items: int | None = None,
) -> None:
    legend_items = legend_items if legend_items is not None else plotly_legend_item_count(fig)
    legend_font_size = 7 if legend_items > 12 else 8
    margin_right = 185 if show_legend and legend_items > 0 else 40

    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        paper_bgcolor="#f6f7fb",
        plot_bgcolor="#ffffff",
        margin=dict(l=80, r=margin_right, t=60, b=80),
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=14,
            color="#2a3f5f",
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            title_text="",
            font=dict(size=legend_font_size),
            itemsizing="constant",
            traceorder="normal",
            tracegroupgap=0,
            entrywidth=170,
            entrywidthmode="pixels",
        ),
    )
    fig.add_annotation(
        text=title,
        x=0.0,
        y=1.0,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        xshift=4,
        yshift=22,
        showarrow=False,
        font=dict(size=23, color="#111827"),
    )
    if subtitle:
        fig.add_annotation(
            text=subtitle,
            x=0.0,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            xshift=4,
            yshift=6,
            showarrow=False,
            font=dict(size=11, color="#6b7280"),
        )

    fig.update_xaxes(
        title_text=x_label,
        showgrid=False,
        ticks="outside",
        ticklen=6,
        title_font=dict(size=16),
        tickfont=dict(size=13),
    )

    yaxis_minor = (
        dict(showgrid=True, gridcolor="rgba(0,0,0,0.07)", gridwidth=1, dtick="D1") if log_y else None
    )
    fig.update_yaxes(
        title_text=y_label,
        type="log" if log_y else "linear",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        gridwidth=1,
        minor=yaxis_minor,
        ticks="outside",
        ticklen=6,
        title_font=dict(size=16),
        tickfont=dict(size=13),
    )


def plotly_empty_figure(
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
) -> object:
    ensure_plotly()
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text="No results parsed.",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="#666"),
    )
    plotly_apply_style(
        fig,
        title,
        subtitle,
        x_label,
        y_label,
        log_y=False,
        show_legend=False,
    )
    return fig


def best_rows_by_config(rows: list[ResultRow]) -> list[ResultRow]:
    best: dict[PlotKey, ResultRow] = {}
    for r in rows:
        key = PlotKey(kind=r.kind, mode=r.mode, n=r.n, v=r.v, backend=r.backend)
        existing = best.get(key)
        if existing is None or r.repeats > existing.repeats:
            best[key] = r
    return list(best.values())


def standard_drum_rows(rows: list[ResultRow]) -> list[ResultRow]:
    plot_rows = best_rows_by_config(rows)
    standard = [r for r in plot_rows if r.kind == "drum" and r.v == r.n + 1]
    return standard if standard else plot_rows


def fixed_n_rows(rows: list[ResultRow], n: int) -> list[ResultRow]:
    plot_rows = best_rows_by_config(rows)
    return [r for r in plot_rows if r.kind == "drum" and r.n == n]


def plot_time_stacked_solve_adj_by_n(
    rows: list[ResultRow],
    v_offset: int,
    backend_order: list[str],
    color_map: dict[str, str],
    label_map: dict[str, str],
    out_svg: Path,
    out_png: Path,
) -> None:
    ensure_plotly()
    import plotly.graph_objects as go

    plot_rows = best_rows_by_config(rows)

    rep: dict[tuple[str, int, int, str], ResultRow] = {}
    adj: dict[tuple[str, int, int, str], ResultRow] = {}
    for r in plot_rows:
        if r.kind != "drum" or r.v != r.n + v_offset:
            continue
        key = (r.kind, r.n, r.v, r.backend)
        if r.mode == "representation":
            rep[key] = r
        elif r.mode == "adjacency":
            adj[key] = r

    joined: dict[tuple[str, int], tuple[float, float]] = {}
    for key in sorted(rep.keys() | adj.keys()):
        r_rep = rep.get(key)
        r_adj = adj.get(key)
        total = r_adj.avg_s if r_adj is not None else (r_rep.avg_s if r_rep is not None else 0.0)
        if total <= 0:
            continue
        solve_raw = r_rep.avg_s if r_rep is not None else total
        solve = min(solve_raw, total) if solve_raw > 0 else total
        extra_adj = max(0.0, total - solve)
        backend = key[3]
        joined[(backend, key[1])] = (solve, extra_adj)

    ns = sorted({n for _, n in joined})
    present_backends = {backend for backend, _ in joined}
    backends = [b for b in backend_order if b in present_backends]

    title = "Mean time vs ambient dimension"
    subtitle = (
        f"v=n+{v_offset} (height=log(total); solve=solid, adj=hatched; segment share is raw time ratio)"
        if v_offset != 1
        else "v=n+1 (height=log(total); solve=solid, adj=hatched; segment share is raw time ratio)"
    )
    x_label = "Ambient dimension <i>n</i>"
    y_label = "log total time (raw share)"
    if not joined:
        fig = plotly_empty_figure(title, subtitle, x_label, y_label)
        write_plotly_images(fig, out_svg, out_png)
        return

    totals = [solve + extra_adj for solve, extra_adj in joined.values() if solve + extra_adj > 0]
    if not totals:
        fig = plotly_empty_figure(title, subtitle, x_label, y_label)
        fig.update_annotations(text="No positive timings parsed.")
        write_plotly_images(fig, out_svg, out_png)
        return

    fig = go.Figure()
    fig.update_layout(barmode="stack", bargap=0.18, bargroupgap=0.04)

    for backend in backends:
        color = color_map.get(backend, "#444")
        xs: list[int] = []
        solve_y: list[float] = []
        adj_y: list[float] = []
        customdata: list[list[float]] = []

        for n in ns:
            times = joined.get((backend, n))
            if times is None:
                continue
            solve, extra_adj = times
            total = solve + extra_adj
            if total <= 0:
                continue
            total_log = math.log10(max(total * 1e6, 1.0))
            if total_log <= 0:
                continue
            xs.append(n)
            solve_y.append((max(0.0, solve) / total) * total_log)
            adj_y.append((max(0.0, extra_adj) / total) * total_log)
            customdata.append([solve, extra_adj, total])

        fig.add_trace(
            go.Bar(
                x=xs,
                y=solve_y,
                offsetgroup=backend,
                legendgroup=backend,
                name=label_map.get(backend, backend),
                marker=dict(color=color, line=dict(color=color, width=1)),
                customdata=customdata,
                hovertemplate=(
                    "backend=%{legendgroup}<br>n=%{x}<br>solve=%{customdata[0]:.6g}s"
                    "<br>adj=%{customdata[1]:.6g}s<br>total=%{customdata[2]:.6g}s<extra></extra>"
                ),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Bar(
                x=xs,
                y=adj_y,
                offsetgroup=backend,
                legendgroup=backend,
                marker=dict(
                    color="#ffffff",
                    line=dict(color=color, width=1),
                    pattern=dict(
                        shape="/",
                        fgcolor=color,
                        bgcolor="rgba(0,0,0,0)",
                        solidity=0.25,
                        size=5,
                    ),
                ),
                customdata=customdata,
                hovertemplate=(
                    "backend=%{legendgroup}<br>n=%{x}<br>solve=%{customdata[0]:.6g}s"
                    "<br>adj=%{customdata[1]:.6g}s<br>total=%{customdata[2]:.6g}s<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_xaxes(dtick=1)
    plotly_apply_style(
        fig,
        title,
        subtitle,
        x_label,
        y_label,
        log_y=False,
        show_legend=True,
        legend_items=len(backends),
    )
    totals_for_ticks = [solve + extra_adj for solve, extra_adj in joined.values() if solve + extra_adj > 0]
    vals = [v for v in totals_for_ticks if v > 0]
    if vals:
        exp_min = math.floor(math.log10(max(min(vals), 1e-6)))
        exp_max = math.ceil(math.log10(max(vals)))
        tickvals: list[float] = []
        ticktext: list[str] = []
        for exp in range(exp_min, exp_max + 1):
            for m in range(1, 10):
                val_s = m * (10**exp)
                tickvals.append(math.log10(max(val_s * 1e6, 1.0)))
                ticktext.append(plotly_seconds_tick_text(val_s) if m == 1 else "")
        fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    write_plotly_images(fig, out_svg, out_png)


def plot_time(
    rows: list[ResultRow],
    color_map: dict[str, str],
    backend_order: list[str],
    label_map: dict[str, str],
    dash_map: dict[str, str],
    out_svg: Path,
    out_png: Path,
) -> None:
    ensure_plotly()
    import plotly.express as px

    title = "Mean solve time vs ambient dimension"
    subtitle = (
        "Solving V-rep -> H-rep for a random v=n+1 vertex drum in dim n with a standard simplex base"
    )

    plot_rows = standard_drum_rows(rows)

    data = [
        {"n": r.n, "avg_s": r.avg_s, "backend": r.backend}
        for r in plot_rows
        if r.avg_s > 0
    ]
    if not data:
        fig = plotly_empty_figure(title, subtitle, "Ambient dimension <i>n</i>", "Time taken")
        write_plotly_images(fig, out_svg, out_png)
        return

    fig = px.line(
        data,
        x="n",
        y="avg_s",
        color="backend",
        color_discrete_map=color_map,
        category_orders={"backend": backend_order},
    )
    for trace in fig.data:
        trace.legendgroup = trace.name
    fig.update_traces(
        mode="lines+markers",
        line=dict(width=PLOT_LINE_WIDTH),
        marker=dict(
            symbol=PLOT_MARKER_SYMBOL,
            size=PLOT_MARKER_SIZE,
            line=dict(width=PLOT_MARKER_LINE_WIDTH),
        ),
    )
    plotly_apply_backend_dashes(fig, dash_map)
    fig.update_xaxes(dtick=1)
    plotly_apply_backend_legend_labels(fig, label_map)
    plotly_reorder_dotted_traces_last(fig, dash_map, backend_order)
    plotly_remove_markers_from_legend(fig, backend_order)
    plotly_apply_style(
        fig,
        title,
        subtitle,
        "Ambient dimension <i>n</i>",
        "Time taken",
        log_y=True,
        show_legend=True,
        legend_items=len({row["backend"] for row in data}),
    )
    plotly_set_log_time_ticks(fig, [row["avg_s"] for row in data])
    write_plotly_images(fig, out_svg, out_png)


def plot_memory(
    rows: list[ResultRow],
    color_map: dict[str, str],
    backend_order: list[str],
    label_map: dict[str, str],
    dash_map: dict[str, str],
    out_svg: Path,
    out_png: Path,
) -> None:
    ensure_plotly()
    import plotly.express as px
    import plotly.graph_objects as go

    title = "Memory usage vs ambient dimension"
    subtitle = (
        "Solving V-rep -> H-rep for a random v=n+1 vertex drum in dim n with a standard simplex base"
        " (avg is dashed)"
    )

    plot_rows = standard_drum_rows(rows)
    backend_dash_map = {backend: "solid" for backend in backend_order}

    series_avg: dict[str, list[tuple[int, float]]] = defaultdict(list)
    series_max: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in plot_rows:
        avg_mib = r.mem_avg_bytes / (1024.0 * 1024.0)
        max_mib = r.mem_max_bytes / (1024.0 * 1024.0)
        backend = r.backend
        if avg_mib > 0:
            series_avg[backend].append((r.n, avg_mib))
        if max_mib > 0:
            series_max[backend].append((r.n, max_mib))

    if not series_max:
        fig = plotly_empty_figure(title, subtitle, "Ambient dimension <i>n</i>", "Memory (MiB)")
        write_plotly_images(fig, out_svg, out_png)
        return

    max_data = [
        {"n": n, "mem_mib": mib, "backend": backend}
        for backend, pts in series_max.items()
        for n, mib in pts
    ]
    fig = px.line(
        max_data,
        x="n",
        y="mem_mib",
        color="backend",
        color_discrete_map=color_map,
        category_orders={"backend": backend_order},
    )
    for trace in fig.data:
        trace.legendgroup = trace.name
    fig.update_traces(
        mode="lines+markers",
        line=dict(width=PLOT_LINE_WIDTH),
        marker=dict(
            symbol=PLOT_MARKER_SYMBOL,
            size=PLOT_MARKER_SIZE,
            line=dict(width=PLOT_MARKER_LINE_WIDTH),
        ),
    )
    plotly_apply_backend_dashes(fig, backend_dash_map)
    fig.update_xaxes(dtick=1)

    for backend in backend_order:
        pts = series_avg.get(backend)
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        base_dash = backend_dash_map.get(backend, "solid")
        avg_dash = "dash" if base_dash == "solid" else "dashdot"
        fig.add_trace(
            go.Scatter(
                x=[n for n, _ in pts_sorted],
                y=[mib for _, mib in pts_sorted],
                mode="lines+markers",
                line=dict(
                    width=PLOT_LINE_WIDTH,
                    dash=avg_dash,
                    color=color_map.get(backend, "#444"),
                ),
                marker=dict(
                    symbol=PLOT_MARKER_SYMBOL,
                    size=PLOT_MARKER_SIZE,
                    line=dict(width=PLOT_MARKER_LINE_WIDTH),
                ),
                legendgroup=backend,
                showlegend=False,
                hovertemplate=(
                    f"backend={backend}<br>n=%{{x}}<br>avg=%{{y:.6g}} MiB<extra></extra>"
                ),
            )
        )

    plotly_apply_backend_legend_labels(fig, label_map)
    plotly_reorder_dotted_traces_last(fig, backend_dash_map, backend_order)
    plotly_remove_markers_from_legend(fig, backend_order)
    plotly_apply_style(
        fig,
        title,
        subtitle,
        "Ambient dimension <i>n</i>",
        "Memory (MiB)",
        log_y=True,
        show_legend=True,
        legend_items=len(series_max),
    )
    plotly_set_log_numeric_ticks(fig, [row["mem_mib"] for row in max_data])
    write_plotly_images(fig, out_svg, out_png)


def plot_time_fixed_n(
    rows: list[ResultRow],
    n: int,
    color_map: dict[str, str],
    backend_order: list[str],
    label_map: dict[str, str],
    dash_map: dict[str, str],
    out_svg: Path,
    out_png: Path,
) -> None:
    ensure_plotly()
    import plotly.express as px

    plot_rows = fixed_n_rows(rows, n)
    if not plot_rows:
        return

    title = f"Mean solve time vs vertex count (n={n})"
    subtitle = "Solving V-rep -> H-rep for a random drum in dim n with a standard simplex base"
    data = [
        {"v": r.v, "avg_s": r.avg_s, "backend": r.backend}
        for r in plot_rows
        if r.avg_s > 0
    ]
    if not data:
        fig = plotly_empty_figure(title, subtitle, "Vertex count <i>v</i>", "Time taken")
        write_plotly_images(fig, out_svg, out_png)
        return

    fig = px.line(
        data,
        x="v",
        y="avg_s",
        color="backend",
        color_discrete_map=color_map,
        category_orders={"backend": backend_order},
    )
    for trace in fig.data:
        trace.legendgroup = trace.name
    fig.update_traces(
        mode="lines+markers",
        line=dict(width=PLOT_LINE_WIDTH),
        marker=dict(
            symbol=PLOT_MARKER_SYMBOL,
            size=PLOT_MARKER_SIZE,
            line=dict(width=PLOT_MARKER_LINE_WIDTH),
        ),
    )
    plotly_apply_backend_dashes(fig, dash_map)
    fig.update_xaxes(dtick=1)
    plotly_apply_backend_legend_labels(fig, label_map)
    plotly_reorder_dotted_traces_last(fig, dash_map, backend_order)
    plotly_remove_markers_from_legend(fig, backend_order)
    plotly_apply_style(
        fig,
        title,
        subtitle,
        "Vertex count <i>v</i>",
        "Time taken",
        log_y=True,
        show_legend=True,
        legend_items=len({row["backend"] for row in data}),
    )
    plotly_set_log_time_ticks(fig, [row["avg_s"] for row in data])
    write_plotly_images(fig, out_svg, out_png)


def plot_memory_fixed_n(
    rows: list[ResultRow],
    n: int,
    color_map: dict[str, str],
    backend_order: list[str],
    label_map: dict[str, str],
    dash_map: dict[str, str],
    out_svg: Path,
    out_png: Path,
) -> None:
    ensure_plotly()
    import plotly.express as px
    import plotly.graph_objects as go

    plot_rows = fixed_n_rows(rows, n)
    if not plot_rows:
        return

    title = f"Memory usage vs vertex count (n={n})"
    subtitle = "Solving V-rep -> H-rep for a random drum in dim n with a standard simplex base (avg is dashed)"
    backend_dash_map = {backend: "solid" for backend in backend_order}

    series_avg: dict[str, list[tuple[int, float]]] = defaultdict(list)
    series_max: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in plot_rows:
        avg_mib = r.mem_avg_bytes / (1024.0 * 1024.0)
        max_mib = r.mem_max_bytes / (1024.0 * 1024.0)
        backend = r.backend
        if avg_mib > 0:
            series_avg[backend].append((r.v, avg_mib))
        if max_mib > 0:
            series_max[backend].append((r.v, max_mib))

    if not series_max:
        fig = plotly_empty_figure(title, subtitle, "Vertex count <i>v</i>", "Memory (MiB)")
        write_plotly_images(fig, out_svg, out_png)
        return

    max_data = [
        {"v": v, "mem_mib": mib, "backend": backend}
        for backend, pts in series_max.items()
        for v, mib in pts
    ]
    fig = px.line(
        max_data,
        x="v",
        y="mem_mib",
        color="backend",
        color_discrete_map=color_map,
        category_orders={"backend": backend_order},
    )
    for trace in fig.data:
        trace.legendgroup = trace.name
    fig.update_traces(
        mode="lines+markers",
        line=dict(width=PLOT_LINE_WIDTH),
        marker=dict(
            symbol=PLOT_MARKER_SYMBOL,
            size=PLOT_MARKER_SIZE,
            line=dict(width=PLOT_MARKER_LINE_WIDTH),
        ),
    )
    plotly_apply_backend_dashes(fig, backend_dash_map)
    fig.update_xaxes(dtick=1)

    for backend in backend_order:
        pts = series_avg.get(backend)
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        base_dash = backend_dash_map.get(backend, "solid")
        avg_dash = "dash" if base_dash == "solid" else "dashdot"
        fig.add_trace(
            go.Scatter(
                x=[v for v, _ in pts_sorted],
                y=[mib for _, mib in pts_sorted],
                mode="lines+markers",
                line=dict(
                    width=PLOT_LINE_WIDTH,
                    dash=avg_dash,
                    color=color_map.get(backend, "#444"),
                ),
                marker=dict(
                    symbol=PLOT_MARKER_SYMBOL,
                    size=PLOT_MARKER_SIZE,
                    line=dict(width=PLOT_MARKER_LINE_WIDTH),
                ),
                legendgroup=backend,
                showlegend=False,
                hovertemplate=(
                    f"backend={backend}<br>v=%{{x}}<br>avg=%{{y:.6g}} MiB<extra></extra>"
                ),
            )
        )

    plotly_apply_backend_legend_labels(fig, label_map)
    plotly_reorder_dotted_traces_last(fig, backend_dash_map, backend_order)
    plotly_remove_markers_from_legend(fig, backend_order)
    plotly_apply_style(
        fig,
        title,
        subtitle,
        "Vertex count <i>v</i>",
        "Memory (MiB)",
        log_y=True,
        show_legend=True,
        legend_items=len(series_max),
    )
    plotly_set_log_numeric_ticks(fig, [row["mem_mib"] for row in max_data])
    write_plotly_images(fig, out_svg, out_png)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--out-dir", type=Path, default=Path("bench-reports"))
    args = ap.parse_args()

    rows_raw = read_all_results(args.input_dir)
    last_by_key: dict[ResultKey, ResultRow] = {}
    for r in rows_raw:
        key = ResultKey(
            kind=r.kind,
            mode=r.mode,
            n=r.n,
            v=r.v,
            repeats=r.repeats,
            backend=r.backend,
        )
        last_by_key[key] = r

    rows = sorted(
        last_by_key.values(),
        key=lambda r: (r.kind, r.mode, r.backend, r.n, r.v, r.repeats),
    )

    backend_order_list = backend_order({r.backend for r in best_rows_by_config(rows)})
    color_map = plotly_color_map(backend_order_list)
    label_map = {backend: backend_legend_label(backend) for backend in backend_order_list}
    dash_map = backend_line_dash_map(backend_order_list)

    out_dir: Path = args.out_dir
    write_csv(rows, out_dir / "results.csv")

    for v_offset in range(1, 6):
        plot_time_stacked_solve_adj_by_n(
            rows,
            v_offset,
            backend_order_list,
            color_map,
            label_map,
            out_dir / f"time_stacked_solve_adj_v_nplus{v_offset}.svg",
            out_dir / f"time_stacked_solve_adj_v_nplus{v_offset}.png",
        )

    modes = sorted({r.mode for r in rows})
    for mode in modes:
        mode_rows = [r for r in rows if r.mode == mode]
        plot_time(
            mode_rows,
            color_map,
            backend_order_list,
            label_map,
            dash_map,
            out_dir / f"time_{mode}.svg",
            out_dir / f"time_{mode}.png",
        )
        plot_memory(
            mode_rows,
            color_map,
            backend_order_list,
            label_map,
            dash_map,
            out_dir / f"memory_{mode}.svg",
            out_dir / f"memory_{mode}.png",
        )

        by_n_dir = out_dir / "by_n" / mode
        ns = sorted({r.n for r in best_rows_by_config(mode_rows) if r.kind == "drum"})
        for n in ns:
            if len({r.v for r in fixed_n_rows(mode_rows, n)}) < 2:
                continue
            plot_time_fixed_n(
                mode_rows,
                n,
                color_map,
                backend_order_list,
                label_map,
                dash_map,
                by_n_dir / f"time_by_v_n{n}.svg",
                by_n_dir / f"time_by_v_n{n}.png",
            )
            plot_memory_fixed_n(
                mode_rows,
                n,
                color_map,
                backend_order_list,
                label_map,
                dash_map,
                by_n_dir / f"memory_by_v_n{n}.svg",
                by_n_dir / f"memory_by_v_n{n}.png",
            )

    flush_plotly_images()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
