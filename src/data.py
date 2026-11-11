"""Load daily revenue from an explicitly marked demo CSV or UCI Online Retail XLSX.

Business definitions:
- gross positive line-item sales, excluding cancellation records
- not net revenue, profit, or recovered revenue
- zero transaction dates may represent closures or incomplete source data
"""
from pathlib import Path
import pandas as pd


def load_daily(path: str | Path, country: str = "United Kingdom") -> tuple[pd.DataFrame, dict]:
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
        if df["date"].duplicated().any():
            raise ValueError("Для агрегированного CSV требуется ровно одна строка на дату")
        meta = {"source": "SYNTHETIC DEMO (not customer data)", "currency": "arbitrary units", "country": "n/a"}
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, engine="openpyxl")
        columns = {"InvoiceNo", "InvoiceDate", "Country", "Quantity", "UnitPrice"}
        if not columns.issubset(raw.columns):
            raise ValueError(f"Отсутствуют необходимые колонки UCI: {columns - set(raw.columns)}")
        raw = raw.loc[raw["Country"].eq(country)].copy()
        if raw.empty:
            raise ValueError(f"Страна '{country}' не найдена в исходном датасете")
        raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
        is_cancellation = raw["InvoiceNo"].astype("string").str.upper().str.startswith("C", na=False)
        is_valid = (
            ~is_cancellation & raw["InvoiceDate"].notna()
            & (pd.to_numeric(raw["Quantity"], errors="coerce") > 0)
            & (pd.to_numeric(raw["UnitPrice"], errors="coerce") > 0)
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
        }
    else:
        raise ValueError("Нужен CSV (демо) или Excel XLSX (UCI Online Retail)")

    df = df.sort_values("date").set_index("date")
    if df.empty:
        raise ValueError("Нет транзакций после обработки")
    # For this educational exercise: no transactions => 0 sales observed.
    # In real operations the ingestion completeness and branch opening schedule MUST be verified.
    all_dates = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(all_dates, fill_value=0).rename_axis("date").reset_index()
    df["revenue"] = df["revenue"].astype(float)
    df["orders"] = df["orders"].astype(int)
    df["aov"] = df["revenue"] / df["orders"].replace(0, float("nan"))
    meta["days"] = int(len(df))
    meta["zero_order_days"] = int((df["orders"] == 0).sum())
    return df, meta
