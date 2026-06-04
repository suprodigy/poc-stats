import pandas as pd
import numpy as np
from pathlib import Path


COLUMN_ALIASES = {
    "date": "date_partition",
    "user": "email",
    "user_id": "email",
    "credit": "usage_credit",
    "quantity": "usage_quantity",
    "unit": "usage_unit",
    "type": "usage_type",
    "division": "사업부",
    "department": "부서",
}

REQUIRED_COLUMNS = {"date_partition", "email", "usage_credit"}
OPTIONAL_COLUMNS = {"usage_type", "usage_quantity", "usage_unit", "사업부", "부서"}
VALID_UNITS = {"tokens", "counts", "duration_s"}


def load(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV or Excel file, normalize columns, validate schema. Returns (df, warnings)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype={"email": str})
    else:
        df = pd.read_csv(path, dtype={"email": str})

    df.columns = [c.strip() for c in df.columns]
    df = _apply_aliases(df)

    warnings = []
    _validate(df, warnings)
    df = _normalize(df, warnings)
    return df, warnings


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for alias, canonical in COLUMN_ALIASES.items():
        if alias in df.columns and canonical not in df.columns:
            rename[alias] = canonical
    return df.rename(columns=rename)


def _validate(df: pd.DataFrame, warnings: list):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    if "usage_unit" in df.columns:
        bad_units = set(df["usage_unit"].dropna().unique()) - VALID_UNITS
        if bad_units:
            warnings.append(f"알 수 없는 usage_unit 값: {bad_units} — 집계에서 제외됩니다.")


def _normalize(df: pd.DataFrame, warnings: list) -> pd.DataFrame:
    df["date_partition"] = pd.to_datetime(df["date_partition"], errors="coerce")
    bad_dates = df["date_partition"].isna().sum()
    if bad_dates:
        warnings.append(f"날짜 파싱 실패 {bad_dates}행 — 제거합니다.")
        df = df.dropna(subset=["date_partition"])

    df["usage_credit"] = pd.to_numeric(df["usage_credit"], errors="coerce").fillna(0.0)
    neg = (df["usage_credit"] < 0).sum()
    if neg:
        warnings.append(f"음수 usage_credit {neg}행 — 0으로 처리합니다.")
        df["usage_credit"] = df["usage_credit"].clip(lower=0)

    for col in ["usage_quantity", "usage_unit", "usage_type", "사업부", "부서"]:
        if col not in df.columns:
            df[col] = "unknown"

    df["usage_quantity"] = pd.to_numeric(df["usage_quantity"], errors="coerce").fillna(1.0)
    df["usage_unit"] = df["usage_unit"].fillna("unknown").astype(str)
    df["usage_type"] = df["usage_type"].fillna("unknown").astype(str)
    df["사업부"] = df["사업부"].fillna("unknown").astype(str)
    df["부서"] = df["부서"].fillna("unknown").astype(str)
    df["email"] = df["email"].fillna("unknown").astype(str)

    # Check suspicious token ratio if tokens unit present
    token_rows = df[df["usage_unit"] == "tokens"]
    if not token_rows.empty and "usage_quantity" in df.columns:
        total_qty = token_rows["usage_quantity"].sum()
        total_credit = token_rows["usage_credit"].sum()
        if total_qty > 0:
            ratio = total_credit / total_qty
            if ratio > 1.0:
                warnings.append(
                    f"토큰당 크레딧이 {ratio:.4f}로 높습니다. "
                    "usage_credit이 크레딧/토큰인지, 총 크레딧인지 확인하세요."
                )

    return df
