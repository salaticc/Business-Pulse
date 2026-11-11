"""Load daily revenue from an explicitly marked demo CSV or UCI Online Retail XLSX.

Business definitions:
- gross positive line-item sales, excluding cancellation records
- not net revenue, profit, or recovered revenue
- zero transaction dates may represent closures or incomplete source data
"""
from pathlib import Path
from hashlib import sha256
import logging
import numpy as np
import pandas as pd


def validate_daily(daily: pd.DataFrame) -> None:
    """Validate model input even when callers bypass the file loader.

    Row lags mean calendar days only on a complete, unique daily grid.
    Sorting is allowed; imputation and target filtering are not implicit.
    """
    if not {"date", "revenue"}.issubset(daily.columns) or daily.empty:
        raise ValueError("Нужен непустой дневной ряд с date и revenue")
    dates = daily["date"]
    if not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().any():
        raise ValueError("date должен содержать корректные datetime-даты без пропусков")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Нужны календарные даты без времени и часового пояса")
    if dates.duplicated().any():
        raise ValueError("Для агрегированного CSV требуется ровно одна строка на дату")
    ordered = dates.sort_values()
    if not ordered.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("Пропущены календарные даты; нельзя считать пропуски нулевой выручкой")
    if not pd.api.types.is_numeric_dtype(daily["revenue"]):
        raise ValueError("revenue должен иметь числовой тип")
    try:
        values = daily["revenue"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("revenue должен быть числом") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("revenue должен быть конечным неотрицательным числом без пропусков")


def load_daily(
    path: str | Path, country: str = "United Kingdom", *, missing_days: str = "error",
) -> tuple[pd.DataFrame, dict]:
    if missing_days not in {"error", "zero"}:
        raise ValueError("missing_days должен быть error или zero")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}. См. README.md: как скачать датасет UCI.")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        needed = {"date", "revenue", "orders"}
        if not needed.issubset(df.columns):
            raise ValueError(f"В демонстрационном CSV ожидаются столбцы {needed}")
        df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
        df["revenue"] = pd.to_numeric(df["revenue"], errors="raise")
        df["orders"] = pd.to_numeric(df["orders"], errors="raise")
        if df["date"].isna().any():
            raise ValueError("Пропущена дата")
        if df["date"].duplicated().any():
            raise ValueError("Для агрегированного CSV требуется ровно одна строка на дату")
        for column in ("revenue", "orders"):
            values = df[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{column}: нужны конечные неотрицательные значения без пропусков")
        if not df["orders"].eq(np.floor(df["orders"])).all():
            raise ValueError("orders должен быть целым числом")
        demo = Path(__file__).resolve().parents[1] / "data/demo/daily_sales.csv"
        is_demo = demo.exists() and sha256(path.read_bytes()).digest() == sha256(demo.read_bytes()).digest()
        meta = {
            "source": "SYNTHETIC DEMO (not customer data)" if is_demo else "Daily CSV (provenance unverified)",
            "currency": "arbitrary units" if is_demo else "unspecified",
            "country": "n/a", "source_kind": "synthetic" if is_demo else "unverified",
        }
    elif path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(path, engine="openpyxl")
        columns = {"InvoiceNo", "InvoiceDate", "Country", "Quantity", "UnitPrice"}
        if not columns.issubset(raw.columns):
            raise ValueError(f"Отсутствуют необходимые колонки UCI: {columns - set(raw.columns)}")
        raw = raw.loc[raw["Country"].eq(country)].copy()
        if raw.empty:
            raise ValueError(f"Страна '{country}' не найдена в исходном датасете")
        raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
        raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
        raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")
        is_cancellation = raw["InvoiceNo"].astype("string").str.upper().str.startswith("C", na=False)
        is_valid = (
            ~is_cancellation & raw["InvoiceDate"].notna()
            & raw["InvoiceNo"].notna()
            & np.isfinite(raw["Quantity"]) & np.isfinite(raw["UnitPrice"])
            & raw["Quantity"].gt(0) & raw["UnitPrice"].gt(0)
        )
        cleaned = raw.loc[is_valid].copy()
        cleaned["date"] = cleaned["InvoiceDate"].dt.normalize()
        cleaned["line_gross"] = cleaned["Quantity"].astype(float) * cleaned["UnitPrice"].astype(float)
        daily_revenue = cleaned.groupby("date")["line_gross"].sum()
        daily_orders = cleaned.groupby("date")["InvoiceNo"].nunique()
        df = pd.DataFrame({"revenue": daily_revenue, "orders": daily_orders}).reset_index()
        meta = {
            "source": "UCI Online Retail (2010-12-01 to 2011-12-09)",
            "citation": "Chen, D. (2015). Online Retail. UCI ML Repository. https://doi.org/10.24432/C5BW33",
            "country": country, "currency": "GBP",
            "raw_rows_filtered_to_country": int(len(raw)),
            "valid_sales_lines": int(len(cleaned)),
            "cancellation_lines_excluded": int(is_cancellation.sum()),
            "invalid_non_cancellation_lines_excluded": int((~is_cancellation & ~is_valid).sum()),
            "exact_duplicate_rows_detected": int(raw.duplicated().sum()),
            "duplicate_policy": "reported, not silently removed: identical transaction lines may be legitimate",
            "source_kind": "uci_transactions",
        }
    else:
        raise ValueError("Нужен CSV (демо) или Excel XLSX (UCI Online Retail)")

    df = df.sort_values("date").set_index("date")
    if df.empty:
        raise ValueError("Нет транзакций после обработки")
    all_dates = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = all_dates.difference(df.index)
    if len(missing) and missing_days == "error":
        raise ValueError(
            f"Пропущены {len(missing)} календарных дат (первая {missing[0].date()}). "
            "Проверьте календарь и полноту загрузки; missing_days='zero' / --missing-days zero "
            "допускается только как явное исследовательское предположение."
        )
    if len(missing):
        logging.getLogger(__name__).warning("%s отсутствующих дат явно заполнены нулями; полнота данных не подтверждена", len(missing))
    df = df.reindex(all_dates, fill_value=0).rename_axis("date").reset_index()
    df["revenue"] = df["revenue"].astype(float)
    df["orders"] = df["orders"].astype(int)
    df["aov"] = df["revenue"] / df["orders"].replace(0, float("nan"))
    meta["days"] = int(len(df))
    meta["zero_order_days"] = int((df["orders"] == 0).sum())
    meta["missing_days_policy"] = missing_days
    meta["assumed_zero_dates"] = [str(d.date()) for d in missing]
    meta["completeness_verified"] = False
    meta["quality_note"] = "Daily continuity cannot establish ingestion completeness or distinguish closures; source edges may be partial."
    validate_daily(df)
    return df, meta
