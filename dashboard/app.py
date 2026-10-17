"""Dash app: sidebar navigation, page routing and the /health route.

Run locally:  python -m dashboard.app
Production:   gunicorn dashboard.app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
"""

import os
from pathlib import Path

from dash import Dash, Input, Output, dcc, html

from dashboard import data, solve
from dashboard.components import DISCLAIMER
from dashboard.pages import capital, cash_flows, methodology, optimizer, overview, rate_risk, robustness, stress

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600"
         "&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap")
REPO_URL = "https://github.com/weixunxie/insurance-alm-model-portfolio"

# (path, nav label, layout function)
PAGES = [
    ("/", "Overview", overview.layout),
    ("/optimizer", "Optimizer", optimizer.layout),
    ("/rate-risk", "Rate risk (KRD)", rate_risk.layout),
    ("/cash-flows", "Cash flows", cash_flows.layout),
    ("/capital", "Capital (NAIC C-1)", capital.layout),
    ("/stress", "Stress tests", stress.layout),
    ("/robustness", "Robustness", robustness.layout),
    ("/methodology", "Methodology", methodology.layout),
]

app = Dash(__name__, title="ALM Model Portfolio", update_title=None, suppress_callback_exceptions=True,
           external_stylesheets=[FONTS], assets_folder=str(Path(__file__).parent / "assets"))
server = app.server


@server.route("/health")
def health():
    return ("ok", 200) if data.is_ready() else ("warming up", 503)


def sidebar():
    return html.Nav(className="sidebar", children=[
        html.Div("ALM Model Portfolio", className="brand"),
        html.Div("NAIC-constrained insurance ALM", className="brand-sub"),
        html.Div(className="nav", children=[
            dcc.Link(label, href=path, id=f"nav-{i}", className="nav-link") for i, (path, label, _) in enumerate(PAGES)]),
    ])


def footer():
    return html.Footer(className="footer", children=[
        "Weixun Xie · ", html.A("github.com/weixunxie/insurance-alm-model-portfolio", href=REPO_URL)])


app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar(),
    html.Main(className="main", children=[
        html.Div(className="content", children=[html.Div(DISCLAIMER, className="disclaimer"),
                                                html.Div(id="page")]),
        footer(),
    ]),
])


@app.callback([Output("page", "children")] + [Output(f"nav-{i}", "className") for i in range(len(PAGES))],
              Input("url", "pathname"))
def route(pathname):
    match = next((i for i, (path, _, _) in enumerate(PAGES) if path == pathname), 0)
    classes = ["nav-link active" if i == match else "nav-link" for i in range(len(PAGES))]
    return [PAGES[match][2]()] + classes


solve.warm()   # build the base model, base-case LP data, solution and frontier once per process
data.warm()    # load outputs and mark the process ready for /health

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8050)), debug=False)
