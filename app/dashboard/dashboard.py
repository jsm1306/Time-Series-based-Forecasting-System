import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/forecast"
MODEL_COLORS = {
    "XGBoost": "#003f5c",
    "LSTM": "#58508d",
    "SARIMA": "#bc5090",
    "Prophet": "#ff6361",
}
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "trained_models" / "model_registry.json"
DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "processed_timeseries.csv"
MODEL_NAMES = ["XGBoost", "LSTM", "SARIMA", "Prophet"]


def load_registry() -> Dict[str, Any]:
    """Load the model registry data from disk."""
    if not REGISTRY_PATH.exists():
        return {}

    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_registry_states(registry_path: Path) -> List[str]:
    """Load available states from the model registry."""
    registry = load_registry()
    return sorted(registry.keys())


def load_registry_entry(state: str) -> Dict[str, Any]:
    """Load registry entry for the given state."""
    registry = load_registry()
    return registry.get(state, {})


def load_state_history(state: str, points: int = 30) -> pd.DataFrame:
    """Load recent historical data for the selected state."""
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    state_df = df[df["State"] == state].sort_values("Date").reset_index(drop=True)
    return state_df.tail(points)[["Date", "Total"]].copy()


def build_forecast_payload(state: str, periods: int, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Prepare the request payload for the forecasting API."""
    payload: Dict[str, Any] = {"state": state, "forecast_periods": periods}
    if model_name:
        payload["model_name"] = model_name
    return payload


def call_forecast_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call the FastAPI forecast endpoint."""
    response = requests.post(API_URL, json=payload, timeout=10)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        body = response.text.strip()
        raise ValueError(
            f"Forecast API HTTP {response.status_code}: {body or response.reason}"
        ) from http_err
    return response.json()


def format_predictions(response: Dict[str, Any]) -> pd.DataFrame:
    """Convert API response predictions into a DataFrame."""
    periods = response.get("forecast_periods", 0)
    predictions = response.get("predictions", [])
    if not predictions:
        return pd.DataFrame(columns=["Period", "Prediction"])

    prediction_values = [float(value) for value in predictions]
    return pd.DataFrame(
        {
            "Period": list(range(1, len(prediction_values) + 1)),
            "Prediction": prediction_values,
        }
    )


def set_page_config() -> None:
    """Configure the Streamlit page layout and title."""
    st.set_page_config(
        page_title="Enterprise Forecasting Analytics",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    """Render the dashboard header and overview text."""
    st.title("Enterprise Sales Forecasting Dashboard")
    st.markdown(
        "Modern ML operations dashboard for enterprise forecasting. Select a state and forecasting mode to inspect model output, compare alternatives, and review predicted sales performance."
    )
    st.divider()


def render_sidebar(states: List[str]) -> Dict[str, Any]:
    """Render sidebar controls for state selection and forecast mode."""
    st.sidebar.header("Forecast Controls")
    selected_state = st.sidebar.selectbox("State", options=states)
    selected_mode = st.sidebar.radio(
        "Forecasting Mode",
        options=["Auto Best Model", "Manual Model Selection"],
        index=0,
    )
    selected_model = ""
    if selected_mode == "Manual Model Selection":
        selected_model = st.sidebar.selectbox("Model", options=MODEL_NAMES)
    forecast_periods = st.sidebar.slider(
        "Forecast horizon (days)", min_value=1, max_value=24, value=8
    )
    generate = st.sidebar.button("Generate Forecast")
    compare_models = st.sidebar.button("Compare All Models")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Enterprise-grade forecasting dashboard using FastAPI backend services.")

    return {
        "state": selected_state,
        "mode": selected_mode,
        "model": selected_model,
        "forecast_periods": forecast_periods,
        "generate": generate,
        "compare_models": compare_models,
    }


def render_response_details(response: Dict[str, Any]) -> None:
    """Render response metadata and model explanation."""
    st.markdown("### Forecast Details")
    details = {
        "State": response.get("state", "N/A"),
        "Model Used": response.get("model_used", "N/A"),
        "Forecast Horizon": response.get("forecast_periods", 0),
        "Generated At": response.get("generated_at", "N/A"),
    }

    for label, value in details.items():
        st.markdown(f"**{label}:** {value}")


def render_model_explanation(mode: str, selected_model: str, response: Dict[str, Any]) -> None:
    """Render the model explanation section."""
    selected_label = response.get("model_used", selected_model or "Auto Best Model")
    if mode == "Auto Best Model":
        st.markdown(f"#### 🏆 Best Model Selected: {selected_label}")
        reason = "Selected automatically based on registry model ranking and validation performance."
    else:
        st.markdown(f"#### Model Selected: {selected_label}")
        reason = "User-selected model override enabled."

    st.markdown("### Model Explanation")
    st.markdown(f"**Reason:** {reason}")


def render_metrics(response: Dict[str, Any], df: pd.DataFrame) -> None:
    """Render performance metric cards based on forecast output."""
    average_sales = float(df["Prediction"].mean()) if not df.empty else 0.0
    max_sales = float(df["Prediction"].max()) if not df.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", response.get("model_used", "N/A"))
    col2.metric("Horizon", response.get("forecast_periods", 0))
    col3.metric("Avg Predicted Sales", f"{average_sales:,.2f}")
    col4.metric("Max Predicted Sales", f"{max_sales:,.2f}")


def render_forecast_chart(df: pd.DataFrame) -> None:
    """Render the forecast-only plotly line chart."""
    if df.empty:
        st.warning("No forecast data available to display.")
        return

    fig = px.line(
        df,
        x="Period",
        y="Prediction",
        title="Forecasted Sales Over Time",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Forecast Period",
        yaxis_title="Predicted Sales",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_historical_vs_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    """Render combined historical and forecast time series chart."""
    if history_df.empty and forecast_df.empty:
        st.warning("Historical and forecast data are unavailable for comparison.")
        return
    if history_df.empty:
        st.warning("Historical data is unavailable for comparison.")
        return
    if forecast_df.empty:
        st.warning("Forecast data is unavailable for comparison.")
        return

    future_dates = pd.date_range(start=history_df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=len(forecast_df))
    forecast_plot_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Sales": forecast_df["Prediction"].astype(float),
            "Type": "Forecast",
        }
    )
    history_plot_df = history_df.rename(columns={"Total": "Sales"})
    history_plot_df["Type"] = "Historical"

    chart_df = pd.concat([history_plot_df, forecast_plot_df], ignore_index=True)
    fig = px.line(
        chart_df,
        x="Date",
        y="Sales",
        color="Type",
        title="Historical Sales vs Future Forecast",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        legend_title="Series",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison_table(state: str) -> pd.DataFrame:
    """Build a comparison table for model benchmarking."""
    entry = load_registry_entry(state)
    all_models = entry.get("all_models", {})
    rows = []
    for model_name in MODEL_NAMES:
        metrics = all_models.get(model_name, {})
        rmse = metrics.get("rmse", np.nan)
        mae = metrics.get("mae", np.nan)
        mape = metrics.get("mape", np.nan)

        if model_name == entry.get("best_model") and not metrics:
            rmse = entry.get("rmse", np.nan)
            mae = entry.get("mae", np.nan)
            mape = entry.get("mape", np.nan)

        if model_name == entry.get("best_model"):
            status = "Best Model"
        elif metrics:
            status = "Available"
        else:
            status = "No metrics"

        rows.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "Status": status,
            }
        )

    df = pd.DataFrame(rows)
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
    df["MAPE"] = pd.to_numeric(df["MAPE"], errors="coerce")
    return df


def render_benchmark_section(state: str) -> None:
    """Render model benchmark comparison and MAPE bar chart."""
    st.markdown("### Model Benchmark Comparison")
    benchmark_df = render_model_comparison_table(state)
    benchmark_df["RMSE"] = pd.to_numeric(benchmark_df["RMSE"], errors="coerce")
    benchmark_df["MAE"] = pd.to_numeric(benchmark_df["MAE"], errors="coerce")
    benchmark_df["MAPE"] = pd.to_numeric(benchmark_df["MAPE"], errors="coerce")

    display_df = benchmark_df.copy()
    display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    display_df["MAE"] = display_df["MAE"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    display_df["MAPE"] = display_df["MAPE"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    st.table(display_df)

    numeric_df = benchmark_df.dropna(subset=["MAPE"])
    if not numeric_df.empty:
        if len(numeric_df) < len(benchmark_df):
            st.info(
                "Partial benchmark metrics are available. Only models with saved MAPE values are shown in the chart. "
                "Run the full pipeline to collect metrics for all models."
            )
        fig = px.bar(
            numeric_df,
            x="Model",
            y="MAPE",
            title="Forecasting Model Performance Comparison",
            text_auto=".2f",
        )
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="MAPE",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "MAPE comparison data is not available. "
            "This can happen when only the best model metrics are stored. "
            "Run the full forecasting pipeline to capture metrics for all models."
        )


def render_comparison_overlay(state: str, predictions: Dict[str, List[float]]) -> None:
    """Render comparison chart overlay for all model forecasts."""
    valid_predictions = {model: values for model, values in predictions.items() if values}
    if not valid_predictions:
        st.warning("No model comparison data available.")
        return

    entry = load_registry_entry(state)
    best_model = entry.get("best_model")
    best_metrics = entry.get("best_model_metrics", {})
    best_mape = float(best_metrics.get("mape", 0.0)) if best_metrics else 0.0

    periods = list(range(1, len(next(iter(valid_predictions.values()))) + 1))
    fig = go.Figure()

    if best_model in valid_predictions and best_mape > 0:
        best_values = np.array(valid_predictions[best_model], dtype=float)
        confidence_pct = min(max(best_mape, 0.05), 0.25)
        upper = best_values * (1 + confidence_pct)
        lower = best_values * (1 - confidence_pct)
        color = MODEL_COLORS.get(best_model, "#636efa")

        # Convert hex color to RGBA for Plotly fill color compatibility
        hex_color = color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        fill_rgba = f"rgba({r},{g},{b},0.18)"

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor=fill_rgba,
                line=dict(width=0),
                name=f"{best_model} confidence interval",
                showlegend=False,
            )
        )

    for model_name, values in valid_predictions.items():
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                mode="lines+markers",
                name=model_name,
                line=dict(color=MODEL_COLORS.get(model_name, None), width=3),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="Forecast Comparison Across Models",
        xaxis_title="Forecast Period",
        yaxis_title="Predicted Sales",
        template="plotly_white",
        legend_title="Model",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def collect_comparison_predictions(state: str, periods: int) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Collect forecast predictions from all supported models."""
    predictions: Dict[str, List[float]] = {}
    errors: Dict[str, str] = {}
    for model_name in MODEL_NAMES:
        payload = build_forecast_payload(state, periods, model_name=model_name)
        try:
            response = call_forecast_api(payload)
            values = [float(val) for val in response.get("predictions", [])]
            if values:
                predictions[model_name] = values
            else:
                errors[model_name] = "Forecast response returned no prediction values."
        except Exception as exc:
            errors[model_name] = str(exc)
    return predictions, errors


def main() -> None:
    """Main dashboard execution function."""
    set_page_config()
    render_header()

    states = load_registry_states(REGISTRY_PATH)
    if not states:
        st.error("Unable to load available states from the model registry.")
        return

    controls = render_sidebar(states)
    if not controls["generate"] and not controls["compare_models"]:
        st.info("Select options in the sidebar and click Generate Forecast or Compare All Models.")
        return

    state_history = load_state_history(controls["state"])
    registry_entry = load_registry_entry(controls["state"])

    if controls["generate"]:
        payload = build_forecast_payload(
            controls["state"],
            controls["forecast_periods"],
            controls["model"] if controls["mode"] == "Manual Model Selection" else None,
        )
        with st.spinner("Generating forecast, please wait..."):
            try:
                response = call_forecast_api(payload)
                df = format_predictions(response)
                st.success("Forecast generated successfully.")

                render_response_details(response)
                render_model_explanation(controls["mode"], controls["model"], response)
                render_metrics(response, df)

                st.markdown("---")
                st.subheader("Forecast Table")
                st.dataframe(df)

                st.markdown("---")
                render_forecast_chart(df)

                st.markdown("---")
                render_historical_vs_forecast(state_history, df)
            except requests.exceptions.RequestException:
                st.error("Unable to reach forecast API. Please ensure the FastAPI service is running.")
            except ValueError as error:
                st.warning(f"Invalid response: {error}")
            except Exception as error:
                st.error(f"An unexpected error occurred: {error}")

    if controls["compare_models"]:
        with st.spinner("Comparing models, please wait..."):
            comparison_predictions, comparison_errors = collect_comparison_predictions(
                controls["state"], controls["forecast_periods"]
            )
            if comparison_errors:
                st.warning("Some model forecasts failed to generate:")
                for model_name, error_text in comparison_errors.items():
                    st.write(f"- **{model_name}**: {error_text}")
            render_comparison_overlay(controls["state"], comparison_predictions)

    st.markdown("---")
    render_benchmark_section(controls["state"])


if __name__ == "__main__":
    main()
