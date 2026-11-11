"""Standalone, locally viewable HTML preview. No web hosting required."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def save_html(test: pd.DataFrame, report: dict, out_path):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=.13,
                        subplot_titles=("Фактическая выручка и прогноз (отложенный период)",
                                        "Число заказов в день"),
                        row_heights=[.72, .28])
    fig.add_trace(go.Scatter(x=test["date"], y=test["revenue"], name="Факт", mode="lines",
                             line=dict(color="#2563eb", width=2.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test["date"], y=test["predicted"], name="Прогноз (" + report["selected_predictor"] + ")",
                             mode="lines", line=dict(color="#0f766e", dash="dash", width=2)), row=1, col=1)
    bad = test[test["alert"]]
    fig.add_trace(go.Scatter(x=bad["date"], y=bad["revenue"], name="Проверить", mode="markers",
                             marker=dict(color="#dc2626", size=11, symbol="circle-open", line=dict(width=2.5)),
                             text=[f"Отклонение от прогноза: {v:,.0f}" for v in bad["shortfall_vs_forecast"]],
                             hovertemplate="%{x|%d.%m.%Y}<br>%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=test["date"], y=test["orders"], name="Заказы", marker_color="#64748b"), row=2, col=1)
    fig.update_layout(
        title=dict(text="G-Analytics · Business Pulse ML<br><sup>Учебный прототип · не интерпретировать как потерянную прибыль</sup>"),
        template="plotly_white", height=800, margin=dict(t=120, b=60, l=60, r=40),
        legend=dict(orientation="h", y=1.03, x=0), hovermode="x unified",
    )
    fig.update_yaxes(title_text="Продажи, валюта источника", row=1, col=1)
    fig.update_yaxes(title_text="Заказы", row=2, col=1)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
