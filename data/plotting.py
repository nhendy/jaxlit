"""Plotly trace builders for the canonical `Scenario`.

Lives in the package rather than in a script so every viewer shares one set of
conventions -- colours, the observed/future split, the yaw spoke on each box --
and so a second viewer can import them normally instead of loading a script by
path.
"""

from typing import Any

import attrs
import jax
import numpy as np
import plotly.graph_objects as go
import tabulate

from commmon.geometry import box_corners
from data.schema import Scenario
from data.specs import AgentType, PolylineType

# Map elements read as context, agents as figure -- so the map is drawn in
# muted greys and the accent is spent on the ego and the observed/future split.
_MAP_STYLE = {
    PolylineType.LANE_CENTER: ("#9aa4b2", 1.0, "dot"),
    PolylineType.ROAD_EDGE: ("#5b6472", 1.8, "solid"),
    PolylineType.LANE_BOUNDARY_SOLID: ("#8b94a3", 1.2, "solid"),
    PolylineType.LANE_BOUNDARY_DASHED: ("#8b94a3", 1.2, "dash"),
    PolylineType.CROSSWALK: ("#c98a3a", 1.6, "solid"),
    PolylineType.STOP_LINE: ("#b3402a", 2.0, "solid"),
    PolylineType.SPEED_BUMP: ("#c98a3a", 1.6, "dot"),
    PolylineType.UNKNOWN: ("#c3c9d4", 1.0, "solid"),
}
_AGENT_COLOR = {
    AgentType.VEHICLE: "#3f7fbf",
    AgentType.PEDESTRIAN: "#2f9e6f",
    AgentType.CYCLIST: "#b06fc9",
    AgentType.MOTORCYCLIST: "#c9863f",
    AgentType.BUS: "#4a5fa8",
    AgentType.STATIC: "#98a1ad",
    AgentType.UNKNOWN: "#b8bfc9",
}
_EGO_COLOR = "#d1453b"
# Boxes are drawn over the map and over each other, so the fill has to stay
# faint enough that the outline, not the fill, is what you read.
_BOX_FILL_ALPHA = 0.18


def _map_traces(scenario: Scenario) -> list[go.Scatter]:
    points = np.asarray(scenario.vector_map.points)
    types = np.asarray(scenario.vector_map.types)
    mask = np.asarray(scenario.polyline_mask)

    # One trace per type rather than per polyline: a scenario has hundreds of
    # polylines, and hundreds of traces makes Plotly crawl. NaN separators break
    # the line between polylines within a single trace.
    traces = []
    for kind in np.unique(types[mask]):
        style_color, width, dash = _MAP_STYLE.get(
            PolylineType(int(kind)), _MAP_STYLE[PolylineType.UNKNOWN]
        )
        xs: list[float] = []
        ys: list[float] = []
        for poly in points[mask & (types == kind)]:
            xs.extend([*poly[:, 0], np.nan])
            ys.extend([*poly[:, 1], np.nan])
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=style_color, width=width, dash=dash),
                name=PolylineType(int(kind)).name.lower().replace("_", " "),
                hoverinfo="skip",
                legendgroup="map",
            )
        )
    return traces


def print_tree(tree: Any) -> None:
    rows = [
        [jax.tree_util.keystr(path, simple=True, separator="."), leaf.shape, leaf.dtype]
        for path, leaf in jax.tree.leaves_with_path(tree)
    ]
    print(tabulate.tabulate(rows, headers=["path", "shape", "dtype"]))


def _rgba(color: str, alpha: float) -> str:
    """`#rrggbb` -> `rgba(...)`, so one palette can serve both lines and fills."""
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


@attrs.frozen(kw_only=True)
class AgentStyle:
    """How one agent is drawn, shared by every trace that belongs to it."""

    color: str
    label: str
    # Plotly legend group: clicking the agent's legend entry toggles every
    # trace in it -- box, observed track, true future, and any forecast.
    group: str
    is_ego: bool


def agent_style(scenario: Scenario, i: int) -> AgentStyle:
    types = np.asarray(scenario.agents.agent_type)
    is_ego = i == int(np.asarray(scenario.ego_index))
    kind = AgentType(int(types[i]))
    return AgentStyle(
        color=_EGO_COLOR
        if is_ego
        else _AGENT_COLOR.get(kind, _AGENT_COLOR[AgentType.UNKNOWN]),
        label=f"{'EGO ' if is_ego else ''}{kind.name.lower()} #{i}",
        group=f"agent-{i}",
        is_ego=is_ego,
    )


def _bbox_trace(corners: np.ndarray, *, style: AgentStyle, detail: str) -> go.Scatter:
    """One filled trace for a stack of boxes, each with a spoke marking its yaw.

    `corners` is anything `box_corners` returns -- `(4, 2)` for a single box, or
    any leading dims for a batch (every agent at one step, or one agent over
    time). They all land in a *single* trace, joined by NaN separators, for the
    same reason the map does it: a trace per box makes Plotly crawl.

    A rectangle alone is yaw-ambiguous -- it looks identical rotated by pi -- so
    each box also gets a spoke from its centre to the midpoint of the front
    edge. That is what separates a car driving away from one driving at you.
    `fill="toself"` treats each NaN-separated run as its own polygon, so the
    two-point spoke encloses no area and only the box body is filled.
    """
    xs: list[float] = []
    ys: list[float] = []
    for box in np.asarray(corners).reshape(-1, 4, 2):
        outline = np.concatenate([box, box[:1]])  # closed: last corner -> first
        centre = box.mean(axis=0)
        nose = (box[1] + box[2]) / 2.0  # midpoint of the front-right/front-left edge
        xs.extend([*outline[:, 0], np.nan, centre[0], nose[0], np.nan])
        ys.extend([*outline[:, 1], np.nan, centre[1], nose[1], np.nan])

    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=style.color, width=2.0 if style.is_ego else 1.2),
        fill="toself",
        fillcolor=_rgba(style.color, _BOX_FILL_ALPHA),
        name=f"{style.label} box",
        legendgroup=style.group,
        showlegend=False,
        hovertemplate=f"{style.label}<br>{detail}<extra></extra>",
    )


def _agent_traces(
    scenario: Scenario,
    show_future: bool,
    show_boxes: bool,
    future_color: str | None = None,
) -> list[go.Scatter]:
    position = np.asarray(scenario.agents.position)
    heading = np.asarray(scenario.agents.heading)
    extent = np.asarray(scenario.agents.extent)
    valid = np.asarray(scenario.agents.valid)
    split = scenario.spec.num_history

    traces = []
    for i in range(position.shape[0]):
        if not valid[i].any():
            continue
        style = agent_style(scenario, i)
        color, label, is_ego = style.color, style.label, style.is_ego

        obs = valid[i, :split]
        if show_boxes and obs.any():
            # The last *valid* observed step, not `split - 1`: agents drop in and
            # out mid-window, and heading is zeroed where invalid, so the fixed
            # index would point a box the wrong way.
            now = int(np.flatnonzero(obs)[-1])
            traces.append(
                _bbox_trace(
                    box_corners(position[i, now], heading[i, now], extent[i]),
                    style=style,
                    detail=f"t={now}, {extent[i, 0]:.1f}x{extent[i, 1]:.1f} m",
                )
            )
        traces.append(
            go.Scatter(
                x=position[i, :split][obs, 0],
                y=position[i, :split][obs, 1],
                mode="lines+markers",
                line=dict(color=color, width=3 if is_ego else 1.6),
                marker=dict(size=4 if is_ego else 2.5),
                name=label,
                legendgroup=style.group,
                showlegend=True,
                hovertemplate=f"{label}<br>observed<extra></extra>",
            )
        )
        if show_future:
            fut = valid[i, split:]
            traces.append(
                go.Scatter(
                    x=position[i, split:][fut, 0],
                    y=position[i, split:][fut, 1],
                    mode="lines",
                    # Future is the thing a model predicts, so it is drawn
                    # dashed -- visually "not given" rather than just another line.
                    line=dict(
                        color=future_color or color,
                        width=3.0 if is_ego else 2.2,
                        dash="dash",
                    ),
                    name=f"{label} future",
                    legendgroup=style.group,
                    showlegend=False,
                    hovertemplate=f"{label}<br>future<extra></extra>",
                )
            )
    return traces


def scenario_traces(
    scenario: Scenario,
    *,
    show_future: bool = True,
    show_boxes: bool = True,
    future_color: str | None = None,
) -> list[go.Scatter]:
    """Map, then agents: everything needed to draw one unbatched scenario.

    `future_color` draws every agent's true future in one colour instead of
    the agent's own -- for plots that compare it against a prediction.
    """
    return [
        *_map_traces(scenario),
        *_agent_traces(scenario, show_future, show_boxes, future_color),
    ]
