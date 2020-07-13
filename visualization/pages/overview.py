import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from utils import Header, make_dash_table

import pandas as pd
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()


df_allocations = pd.read_csv(DATA_PATH.joinpath("df_allocations.csv"))
df_essentials = pd.read_csv(DATA_PATH.joinpath("df_essentials.csv"))
df_port = pd.read_csv(DATA_PATH.joinpath("df_port.csv"))
df_holdings = pd.read_csv(DATA_PATH.joinpath("df_holdings.csv"))
df_benchmark = pd.read_csv(DATA_PATH.joinpath("df_benchmark.csv"))
df_composition = pd.read_csv(DATA_PATH.joinpath("df_composition.csv"))
df_performance = pd.read_csv(DATA_PATH.joinpath("df_performance.csv"))


def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    html.Div(
                        [
                            html.H5("Investment Objective"),
                            html.P(
                                "\
                            The portfolio's objective is to achieve a balance of current income and long-term capital appreciation, with a small bias towards capital appreciation. It invests primarily in a diversified mix of equity, fixed income and commodity managed by us and by other ETF fund managers.",
                                style={"color": "#ffffff"},
                                className="row",
                            ),
                        ],
                        className="product",
                    ),
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(["Reasons for Investing"], className="subtitle padded"),
    
                                    html.P(
                                        "• Invests in a mix of ETF diversified by asset\
                                            class, geographic region, economic sector and\
                                            investment style, aiming to maximize returns while\
                                            managing risk."
                                    ),
                                    html.P(
                                        "• Rigorous portfolio construction by an experienced\
                                            team combined with regular monitoring and daily\
                                            cash flow rebalancing help to ensure each portfolio\
                                            stays on track."
                                    ),
                                    html.P(
                                        "• From creating the optimal asset mix and selecting\
                                            funds, to their monitoring and rebalancing, each\
                                            portfolio delivers convenience and simplicity."
                                    ),
                                ],
                                className="six columns",
                                style={"color": "#696969"},
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Average Annual Performance",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-1",
                                        figure={
                                            "data": [
                                                go.Bar(
                                                    x=df_performance["Date"],
                                                    y=df_performance["Portfolio"],
                                                    marker={
                                                        "color": "#97151c",
                                                        "line": {
                                                            "color": "rgb(255, 255, 255)",
                                                            "width": 2,
                                                        },
                                                    },
                                                    name="All-weather Portfolio",
                                                ),
                                                go.Bar(
                                                    x=df_performance["Date"],
                                                    y=df_performance["Benchmark"],
                                                    marker={
                                                        "color": "#dddddd",
                                                        "line": {
                                                            "color": "rgb(255, 255, 255)",
                                                            "width": 2,
                                                        },
                                                    },
                                                    name="S&P Risk Parity Benchmark",
                                                ),
                                            ],
                                            "layout": go.Layout(
                                                autosize=False,
                                                bargap=0.35,
                                                font={"family": "Raleway", "size": 10},
                                                height=200,
                                                hovermode="closest",
                                                legend={
                                                    "x": -0.0228945952895,
                                                    "y": -0.189563896463,
                                                    "orientation": "h",
                                                    "yanchor": "top",
                                                },
                                                margin={
                                                    "r": 0,
                                                    "t": 20,
                                                    "b": 10,
                                                    "l": 30,
                                                },
                                                showlegend=True,
                                                title="",
                                                width=330,
                                                xaxis={
                                                    "autorange": True,
                                                    "range": [-0.5, 4.5],
                                                    "showline": True,
                                                    "title": "",
                                                    "type": "category",
                                                },
                                                yaxis={
                                                    "autorange": True,
                                                    "range": [0, 22.9789473684],
                                                    "showgrid": True,
                                                    "showline": True,
                                                    "title": "",
                                                    "type": "linear",
                                                    "zeroline": False,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Risk Potential", className="subtitle padded"
                                    ),
                                    html.Img(
                                        src=app.get_asset_url("risk_reward.png"),
                                        className="risk-reward",
                                    ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Fund Essentials",
                                        className="subtitle padded",
                                    ),
                                    html.Table(make_dash_table(df_essentials)),
                                ],
                                className="six columns",
                            ),
                            
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Allocations (%)"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table(df_allocations)),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        ["Composition"],
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                       figure={
                                           'data': [{
                                                    'labels': df_composition['Class'],
                                                    'values': df_composition['Portfolio'],
                                                    'marker': {
                                                      'colors': [
                                                        '#97140c',
                                                        '#ff9900',
                                                        '#f8e0b0',
                                                        '#93b5cf',
                                                        '#1ba784',
                                                        '#ad6598',
                                                        '#696969',
                                                      ]
                                                    },
                                                    'hole':.3,
                                                    'type': 'pie',
                                                    'name': "Portfolio",
                                                    'hoverinfo': 'label+percent+name',
                                                    # 'sort': false,
                                                  }],

                                                  'layout': {
                                                    'autosize': True,
                                                    'width': 350,
                                                    'height':200,
                                                    'font': {"family": "Raleway", "size": 10},

                                                    'margin':{
                                                        "r": 0,
                                                        "t": 50,
                                                        "b": 0,
                                                        "l": 10,
                                                    },
                                                  }

                                
                                       },
                                       config={"displayModeBar": False},
                                    ),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row",
                        style={"margin-bottom": "15px"},
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Growth of $100K", className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-4",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=df_port["Date"],
                                                    y=df_port["accountCAD"],
                                                    line={"color": "#ff9900"},
                                                    mode="lines",
                                                    name="CAD Account",
                                                ),
                                                go.Scatter(
                                                    x=df_port["Date"],
                                                    y=df_port["accountUSD"],
                                                    line={"color": "#f8e0b0"},
                                                    mode="lines",
                                                    name="USD Account",
                                                ),
                                                go.Scatter(
                                                    x=df_port["Date"],
                                                    y=df_port["portfolio"],
                                                    line={"color": "#97140c"},
                                                    mode="lines",
                                                    name="All-weather Portfolio",
                                                ),
                                                go.Scatter(
                                                    x=df_benchmark["Date"],
                                                    y=df_benchmark["Benchmark"],
                                                    line={"color": "#b5b5b5"},
                                                    mode="lines",
                                                    name="S&P Risk Parity Benchmark",
                                                ),
                                            ],
                                            "layout": go.Layout(
                                                autosize=True,
                                                width=700,
                                                height=200,
                                                font={"family": "Raleway", "size": 10},
                                                margin={
                                                    "r": 30,
                                                    "t": 30,
                                                    "b": 30,
                                                    "l": 30,
                                                },
                                                showlegend=True,
                                                titlefont={
                                                    "family": "Raleway",
                                                    "size": 10,
                                                },
                                                xaxis={
                                                    "autorange": True,
                                                    "range": [
                                                        "2007-12-31",
                                                        "2018-03-06",
                                                    ],
                                                    "rangeselector": {
                                                        "buttons": [
                                                            {
                                                                "count": 1,
                                                                "label": "1Y",
                                                                "step": "year",
                                                                "stepmode": "backward",
                                                            },
                                                            {
                                                                "count": 3,
                                                                "label": "3Y",
                                                                "step": "year",
                                                                "stepmode": "backward",
                                                            },
                                                            {
                                                                "count": 5,
                                                                "label": "5Y",
                                                                "step": "year",
                                                            },
                                                            {
                                                                "label": "All",
                                                                "step": "all",
                                                            },
                                                        ]
                                                    },
                                                    "showline": True,
                                                    "type": "date",
                                                    "zeroline": False,
                                                },
                                                yaxis={
                                                    "autorange": True,
                                                    "range": [
                                                        18.6880162434,
                                                        278.431996757,
                                                    ],
                                                    "showline": True,
                                                    "type": "linear",
                                                    "zeroline": False,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "Portfolio Holdings"
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            html.Table(
                                                make_dash_table(df_holdings),
                                                className="tiny-header",
                                            )
                                        ],
                                        style={"overflow-x": "auto"},
                                    ),
                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
