"""Shared look: colors, Plotly template, KPI tiles, sections, chart frames and report tables."""

import plotly.graph_objects as go
import plotly.io as pio
from dash import dcc, html

PAPER, PANEL, BORDER = "#FAF8F4", "#FFFFFF", "#E4E0D8"
TEXT, TEXT_2, NAVY = "#1A1A1A", "#5F6368", "#0B1F3A"
POS, NEG, BAND, GRID = "#2E7D5B", "#B03A2E", "#ECEAE4", "#EEEBE5"
LIABILITY = NAVY

PORTFOLIOS = ["A", "B", "C"]
PORTFOLIO_NAMES = {"A": "A Conservative IG", "B": "B Yield seeking", "C": "C Optimized ALM"}
PORTFOLIO_COLORS = {"A": "#6F7276", "B": "#C8892B", "C": "#1F5F8B"}

BUCKET_ORDER = ["TSY", "AGENCY_MBS", "IG", "HY", "RMBS", "CMBS", "ABS", "CLO"]
BUCKET_NAMES = {"TSY": "Treasuries", "AGENCY_MBS": "Agency MBS", "IG": "IG corporates",
                "HY": "HY corporates", "RMBS": "Non-agency RMBS", "CMBS": "CMBS", "ABS": "ABS", "CLO": "CLO"}
# Fixed order, validated for CVD separation with in-bar labels as secondary encoding.
BUCKET_COLORS = dict(zip(BUCKET_ORDER, ["#2f6db3", "#d0703a", "#2a9a72", "#d29a1e",
                                        "#c76a93", "#4f7f2a", "#6655b8", "#c24a45"]))

DISCLAIMER = "Educational model on illustrative, synthetic data. Not investment advice, not a statutory RBC result."
CAPTION = "Illustrative, synthetic data."
FONT = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"

pio.templates["alm_report"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, size=13, color=TEXT),
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    margin=dict(l=8, r=8, t=8, b=8),
    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=BORDER, ticks="outside",
               tickcolor=BORDER, automargin=True),
    yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, automargin=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12),
                bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(family=FONT, size=12, color=TEXT)),
    bargap=0.25, bargroupgap=0.08,
    separators=".,",
))
pio.templates.default = "alm_report"

GRAPH_CONFIG = {"displaylogo": False, "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# ---------- number formatting ----------

def fmt(v, spec=".3f") -> str:
    """Format with a Python spec; minus signs as true minus for print-quality tables."""
    return format(v, spec).replace("-", "−")


def fmt_usd_k(v_musd, spec=".1f") -> str:
    """USD millions -> '$12.3k'."""
    return "$" + fmt(v_musd * 1000, spec) + "k"


# ---------- building blocks ----------

def page_header(title: str, lede=None):
    return html.Div(className="page-header", children=[html.H1(title)] + ([html.P(lede, className="lede")] if lede else []))


def section(title: str, *children, note=None):
    head = [html.H2(title)] + ([html.Span(note, className="section-note")] if note else [])
    return html.Section(className="section", children=[html.Div(head, className="section-head"), *children])


def kpi(label: str, value: str, unit: str = "", context: str = "", id_prefix=None):
    ids = (lambda part: {"id": f"{id_prefix}-{part}"}) if id_prefix else (lambda part: {})
    return html.Div(className="kpi", children=[
        html.Div(label, className="kpi-label"),
        html.Div([html.Span(value, **ids("value")), html.Span(unit, className="kpi-unit")], className="kpi-value"),
        html.Div(context, className="kpi-context", **ids("context")),
    ])


def chart(title: str, fig: go.Figure, caption: str = "", height: int = 340, graph_id=None):
    """Chart in a panel: title with units above, 'Illustrative, synthetic data' caption below."""
    fig.update_layout(height=height)
    graph_kwargs = {"id": graph_id} if graph_id else {}
    return html.Div(className="panel", children=[
        html.Div(title, className="chart-title"),
        dcc.Graph(figure=fig, config=GRAPH_CONFIG, style={"height": f"{height}px"}, **graph_kwargs),
        html.Div(" ".join(filter(None, [caption, CAPTION])), className="chart-caption"),
    ])


def swatch(color: str):
    return html.Span(className="swatch", style={"background": color})


def report_table(header: list, rows: list, numeric_cols=(), breaches=None):
    """Plain report table. rows: list of lists of strings / components.

    numeric_cols: column indexes right-aligned with tabular numbers.
    breaches: set of (row, col) cells shown in red as limit breaches.
    """
    breaches = breaches or set()
    cls = lambda j: "num" if j in numeric_cols else None
    thead = html.Thead(html.Tr([html.Th(h, className=cls(j)) for j, h in enumerate(header)]))
    body = []
    for i, row in enumerate(rows):
        cells = []
        for j, v in enumerate(row):
            c = " ".join(filter(None, [cls(j), "breach" if (i, j) in breaches else None]))
            cells.append(html.Td(v, className=c or None))
        body.append(html.Tr(cells))
    return html.Div(className="table-wrap", children=html.Table(className="report", children=[thead, html.Tbody(body)]))


def net_krd_figure(tenors, net_k, tol_k, color: str) -> go.Figure:
    """Net key rate DV01 by tenor ($k per bp) with the per-tenor tolerance band drawn around zero."""
    labels = [f"{int(t)}Y" for t in tenors]
    fig = go.Figure()
    fig.add_bar(x=labels, y=2 * tol_k, base=-tol_k, name="Tolerance band", marker_color=BAND, width=0.8,
                customdata=tol_k, hovertemplate="%{x}: tolerance \u00b1%{customdata:.1f} $k/bp<extra></extra>")
    fig.add_bar(x=labels, y=net_k, name="Net key rate DV01 (assets \u2212 liabilities)", marker_color=color,
                width=0.45, hovertemplate="%{x}: %{y:+.2f} $k/bp<extra></extra>")
    fig.add_hline(y=0, line=dict(color=NAVY, width=1))
    fig.update_layout(barmode="overlay", yaxis_title="$k per bp", xaxis_title="Key tenor")
    return fig


def portfolio_selector(id_: str, value: str = "C"):
    return dcc.RadioItems(id=id_, value=value, className="segmented", inputClassName="segmented-input",
                          options=[{"label": PORTFOLIO_NAMES[p], "value": p} for p in PORTFOLIOS])
