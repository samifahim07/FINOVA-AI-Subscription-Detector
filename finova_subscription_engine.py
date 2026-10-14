"""FINOVA AI — customer-level subscription intelligence (prototype).

Input: full raw bank transaction history and the trusted LightGBM .pkl bundle
saved from the notebook (keys: model, features, threshold).

The pipeline never uses is_subscription, label, category, subcategory, or
subscriptions.csv to make predictions. The model is a transaction-level
classifier; recurrence and billing figures are separate estimates.

Usage (PowerShell):
    py finova_subscription_engine.py --transactions transactions.csv \
       --model finova_subscription_model.pkl --customer-id C0000047

Only load model pickle files you trust. Never load user-uploaded pickle files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd

GROUP_COLS = ["customer_id", "account_id", "merchant_name"]
REQUIRED = GROUP_COLS + ["timestamp", "amount"]
BOOL_FEATURES = ["monthly_pattern", "similar_amount", "quarterly_pattern", "yearly_pattern"]
HISTORY_FEATURES = [
    "days_since_last_payment", "amount_difference",
    "previous_interval", "interval_consistency",
]
FREQUENCIES = {"Monthly": (1, 65), "Quarterly": (3, 195), "Yearly": (12, 740)}

DIGITAL = re.compile(
    r"netflix|spotify|amazon prime|apple icloud|icloud|disney\+|hulu|"
    r"youtube premium|adobe|github|google workspace|nytimes|new york times|"
    r"dropbox|microsoft 365|chatgpt|notion", re.I,
)
BILLS = re.compile(
    r"xfinity|state farm|verizon|at&t|t-mobile|conedison|national grid|"
    r"geico|blue cross|city water|water dept|insurance|comcast|internet|"
    r"electric|utility|childcare|daycare|phone plan", re.I,
)
HOUSING = re.compile(r"realty|rent|landlord|mortgage|property manage", re.I)
BANK_FEE = re.compile(r"overdraft|bank fee|service fee|late payment fee", re.I)


def payment_category(merchant: str, cash_withdrawal: bool = False) -> str:
    merchant = str(merchant).strip()
    if cash_withdrawal or merchant.lower() in {"atm", "cash withdrawal"}:
        return "Excluded: Cash Withdrawal"
    if BANK_FEE.search(merchant):
        return "Excluded: Banking Fee"
    if HOUSING.search(merchant):
        return "Housing Payment"
    if DIGITAL.search(merchant):
        return "Digital Subscription"
    if BILLS.search(merchant):
        return "Recurring Bill"
    return "Other / Unclassified"


def prepare_features(raw: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """Reproduce the EXACT 10 features used for the notebook's LightGBM.

    Returns the cleaned transaction frame with model inputs appended.
    All lags use earlier transactions only, including non-subscriptions.
    """
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing transaction columns: {missing}")
    df = raw.copy()
    if "transaction_id" not in df:
        df["transaction_id"] = [f"row_{i}" for i in range(len(df))]
    if "currency" not in df:
        df["currency"] = "UNKNOWN"
    if "is_cash_withdrawal" not in df:
        df["is_cash_withdrawal"] = False

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if df["timestamp"].isna().any() or df["amount"].isna().any():
        raise ValueError("timestamp / amount contains invalid values. Clean the input before prediction.")
    if df[GROUP_COLS].isna().any().any():
        raise ValueError("customer_id, account_id, merchant_name must not be missing.")
    df["currency"] = df["currency"].fillna("UNKNOWN").astype(str)
    # Interpret CSV representations of booleans safely (bool("False") is True).
    df["is_cash_withdrawal"] = (
        df["is_cash_withdrawal"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    df = df.sort_values(GROUP_COLS + ["timestamp"], kind="stable").reset_index(drop=True)
    group = df.groupby(GROUP_COLS, sort=False)
    last_date = group["timestamp"].shift(1)
    previous_amount = group["amount"].shift(1)
    df["days_since_last_payment"] = (df["timestamp"] - last_date).dt.total_seconds() / 86400
    df["prior_merchant_payments"] = group.cumcount()
    df["amount_difference"] = (df["amount"].abs() - previous_amount.abs()).abs()
    df["previous_interval"] = df.groupby(GROUP_COLS, sort=False)["days_since_last_payment"].shift(1)
    df["interval_consistency"] = (df["days_since_last_payment"] - df["previous_interval"]).abs()
    df["monthly_pattern"] = df["days_since_last_payment"].between(27, 32).astype(int)
    df["similar_amount"] = df["amount_difference"].le(2).astype(int)
    df["quarterly_pattern"] = df["days_since_last_payment"].between(85, 95).astype(int)
    df["yearly_pattern"] = df["days_since_last_payment"].between(350, 380).astype(int)
    df[HISTORY_FEATURES] = df[HISTORY_FEATURES].fillna(-1)
    df["amount"] = df["amount"].astype(float)

    available = set(df.columns)
    missing_features = sorted(set(expected_features) - available)
    if missing_features:
        raise ValueError(f"Model expects unknown features: {missing_features}. Use the original model bundle.")
    return df


def billing_frequency(payment_dates: pd.Series) -> tuple[str, float | None, float | None]:
    # Pay cycles are estimated from matching DEBITS, never from the ground-truth label.
    dates = pd.to_datetime(payment_dates).sort_values()
    gaps = dates.diff().dt.total_seconds().div(86400).dropna()
    if len(dates) < 3 or len(gaps) < 2:
        return "Unverified", None, None
    median = float(gaps.median())
    mad = float((gaps - median).abs().median())
    if 27 <= median <= 32 and mad <= 5:
        return "Monthly", median, mad
    if 85 <= median <= 95 and mad <= 10:
        return "Quarterly", median, mad
    if 350 <= median <= 380 and mad <= 20:
        return "Yearly", median, mad
    return "Unverified", median, mad


def next_due(last_date: pd.Timestamp, ref_date: pd.Timestamp, frequency: str) -> str | None:
    if frequency not in FREQUENCIES:
        return None
    months = FREQUENCIES[frequency][0]
    last = pd.Timestamp(last_date).normalize()
    ref = pd.Timestamp(ref_date).normalize()
    # Derived from the last observed payment, not a known actual billing contract.
    cycle = 1
    while cycle <= 1000:
        date = last + pd.DateOffset(months=months * cycle)
        if date >= ref:
            return date.strftime("%Y-%m-%d")
        cycle += 1
    return None


def money(value: float) -> float:
    return round(float(value), 2)


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class FinovaSubscriptionEngine:
    def __init__(self, trusted_model_path: str | Path):
        with open(trusted_model_path, "rb") as f:
            package = pickle.load(f)
        if not isinstance(package, dict) or not {"model", "features", "threshold"}.issubset(package):
            raise ValueError("Expected .pkl dict with model, features, threshold.")
        self.model = package["model"]
        self.features = list(package["features"])
        self.threshold = float(package["threshold"])

    def report(self, raw: pd.DataFrame, customer_id: str, as_of: str | None = None) -> dict:
        if not customer_id:
            raise ValueError("Specify an authenticated customer_id; never combine customers in a report.")
        selected = raw.loc[raw["customer_id"].astype(str) == str(customer_id)].copy()
        if selected.empty:
            raise ValueError(f"No history for customer {customer_id}")

        selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="coerce")
        if selected["timestamp"].isna().any():
            raise ValueError("Invalid timestamp for selected customer.")
        if as_of is None:
            ref_ts = selected["timestamp"].max()
        else:
            ref_ts = pd.Timestamp(as_of)
            # The date-only CLI argument includes that entire date for transactions.
            if len(as_of.strip()) == 10:
                ref_ts += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        selected = selected.loc[selected["timestamp"] <= ref_ts]
        if selected.empty:
            raise ValueError("No transactions on or before as_of.")
        reference = ref_ts.normalize()

        tx = prepare_features(selected, self.features)
        X = tx[self.features].copy()
        X["amount"] = X["amount"].abs()  # CRITICAL: exactly as training.
        X[BOOL_FEATURES] = X[BOOL_FEATURES].astype(int)
        probs = self.model.predict_proba(X)[:, 1]
        tx["subscription_probability"] = probs
        tx["ml_detected"] = (probs >= self.threshold) & (tx["amount"] < 0)

        # Remove obvious non-subscription movements from candidate outputs.
        candidates = tx.loc[tx["ml_detected"]].copy()
        if candidates.empty:
            return self._empty_report(customer_id, reference, len(tx))
        candidates["payment_category"] = [
            payment_category(merchant, bool(cash))
            for merchant, cash in zip(candidates["merchant_name"], candidates["is_cash_withdrawal"])
        ]
        excluded = candidates["payment_category"].str.startswith("Excluded:")
        excluded_count = int(excluded.sum())
        candidates = candidates.loc[~excluded]
        if candidates.empty:
            report = self._empty_report(customer_id, reference, len(tx))
            report["summary"]["excluded_candidate_transactions"] = excluded_count
            return report

        # Use transaction history for only those customer/account/merchant groups which
        # have an ML-positive payment. Use all their historical DEBITS, not just hits.
        keys = GROUP_COLS + ["currency"]
        candidate_ids = candidates[keys].drop_duplicates()
        history = tx.loc[tx["amount"] < 0].merge(candidate_ids, on=keys, how="inner")
        rows = []
        change_events = []
        for group_key, grp in history.groupby(keys, sort=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            cust, account, merchant, currency = group_key
            grp = grp.sort_values("timestamp")
            matching_predictions = candidates[
                (candidates["account_id"] == account) &
                (candidates["merchant_name"] == merchant) &
                (candidates["currency"] == currency)
            ]
            last_full = grp.iloc[-1]
            anchor_amount = abs(float(last_full["amount"]))
            amount_tolerance = max(2, 0.15 * anchor_amount)
            matched = grp.loc[(grp["amount"].abs() - anchor_amount).abs() <= amount_tolerance]
            frequency, median_gap, gap_deviation = billing_frequency(matched["timestamp"])
            latest_match = matched.iloc[-1]
            last_day = latest_match["timestamp"]
            days_idle = float((ref_ts - last_day).total_seconds() / 86400)
            if frequency == "Unverified":
                activity = "Frequency Unverified"
            elif days_idle <= FREQUENCIES[frequency][1]:
                activity = "Likely Active"
            else:
                activity = "Possibly Inactive"

            cat = payment_category(merchant, bool(grp["is_cash_withdrawal"].fillna(False).astype(bool).any()))
            max_prob = float(matching_predictions["subscription_probability"].max())
            if frequency != "Unverified":
                evidence = "High-Confidence Recurring"
            else:
                evidence = "Needs Review"
            due = next_due(last_day, reference, frequency) if activity == "Likely Active" else None
            days_until = (pd.Timestamp(due) - reference).days if due else None
            months_per_cycle = FREQUENCIES[frequency][0] if frequency in FREQUENCIES else None
            # Costs are only projected for recently-observed, regular groups.
            monthly = money(anchor_amount / months_per_cycle) if activity == "Likely Active" else None
            annual = money(anchor_amount * 12 / months_per_cycle) if activity == "Likely Active" else None
            rows.append({
                "customer_id": str(cust), "account_id": str(account),
                "merchant_name": str(merchant), "currency": str(currency),
                "payment_category": cat, "detection_status": evidence,
                "activity_status": activity, "model_probability_max": round(max_prob, 4),
                "ml_detected_payments": len(matching_predictions),
                "historical_matching_payments": len(matched),
                "median_interval_days": round(median_gap, 2) if median_gap is not None else None,
                "interval_deviation_days": round(gap_deviation, 2) if gap_deviation is not None else None,
                "billing_frequency": frequency,
                "latest_observed_amount": money(anchor_amount),
                "latest_observed_payment": last_day.strftime("%Y-%m-%d"),
                "next_estimated_billing_date": due,
                "days_until_billing": days_until,
                "estimated_monthly_cost": monthly, "estimated_yearly_cost": annual,
            })

            # Charge-change events in verified merchant histories only;
            # these are NOT proof of a subscription plan price increase.
            if frequency == "Unverified" or cat not in {"Digital Subscription", "Recurring Bill"}:
                continue
            prior = matched["amount"].abs().shift(1)
            change = matched["amount"].abs() - prior
            pct_change = (change / prior.replace(0, np.nan)) * 100
            recent = matched.loc[(change >= 1) & (pct_change >= 5)].copy()
            for idx, r in recent.iterrows():
                change_events.append({
                    "merchant_name": str(merchant), "payment_category": cat,
                    "currency": str(currency), "timestamp": r["timestamp"].strftime("%Y-%m-%d"),
                    "previous_amount": money(prior.loc[idx]),
                    "current_amount": money(abs(r["amount"])),
                    "increase_percent": round(float(pct_change.loc[idx]), 2),
                    "note": "Charge increase detected; verify whether price or usage changed.",
                })

        groups = pd.DataFrame(rows)
        # Different transaction currencies must never be added together.
        active = groups.loc[groups["activity_status"] == "Likely Active"].copy()

        def sums(df: pd.DataFrame, column: str) -> dict[str, float]:
            if df.empty:
                return {}
            return {str(k): money(v) for k, v in df.groupby("currency")[column].sum().items()}

        digital = active.loc[active["payment_category"] == "Digital Subscription"]
        bills = active.loc[active["payment_category"] == "Recurring Bill"]
        housing = active.loc[active["payment_category"] == "Housing Payment"]
        other = active.loc[active["payment_category"] == "Other / Unclassified"]
        upcoming = active.loc[active["days_until_billing"].between(0, 30)].copy()
        upcoming = upcoming.sort_values(["days_until_billing", "merchant_name"])
        changes = sorted(change_events, key=lambda d: d["timestamp"], reverse=True)

        def as_records(d: pd.DataFrame) -> list[dict]:
            if d.empty:
                return []
            return to_json_safe(d.to_dict(orient="records"))

        return {
            "metadata": {
                "customer_id": str(customer_id),
                "reference_date": reference.strftime("%Y-%m-%d"),
                "reference_type": "transaction-history snapshot, not live bank access",
                "classifier_threshold": self.threshold,
                "scanned_transactions": len(tx),
                "note": "Model scores apply to individual transactions. Group-level status and bill dates are heuristic estimates, not validated subscriptions.",
            },
            "summary": {
                "ml_candidate_transactions": int(tx["ml_detected"].sum()),
                "excluded_candidate_transactions": excluded_count,
                "customer_merchant_groups": len(groups),
                "high_confidence_recurring_groups": int((groups["detection_status"] == "High-Confidence Recurring").sum()),
                "needs_review_groups": int((groups["detection_status"] == "Needs Review").sum()),
                "likely_active_groups": len(active),
                "digital_monthly_by_currency": sums(digital, "estimated_monthly_cost"),
                "digital_yearly_by_currency": sums(digital, "estimated_yearly_cost"),
                "bills_monthly_by_currency": sums(bills, "estimated_monthly_cost"),
                "housing_monthly_by_currency": sums(housing, "estimated_monthly_cost"),
                "other_monthly_by_currency": sums(other, "estimated_monthly_cost"),
                "all_recurring_monthly_by_currency": sums(active, "estimated_monthly_cost"),
                "all_recurring_yearly_by_currency": sums(active, "estimated_yearly_cost"),
            },
            "digital_subscriptions": as_records(groups.loc[groups["payment_category"] == "Digital Subscription"]),
            "recurring_bills": as_records(groups.loc[groups["payment_category"] == "Recurring Bill"]),
            "housing_payments": as_records(groups.loc[groups["payment_category"] == "Housing Payment"]),
            "other_or_unclassified": as_records(groups.loc[groups["payment_category"] == "Other / Unclassified"]),
            "needs_review": as_records(groups.loc[groups["detection_status"] == "Needs Review"]),
            "upcoming_30_days": as_records(upcoming),
            "historical_charge_increases": changes,
        }

    def _empty_report(self, customer_id: str, reference: pd.Timestamp, scanned: int) -> dict:
        return {
            "metadata": {"customer_id": customer_id, "reference_date": reference.strftime("%Y-%m-%d"),
                         "classifier_threshold": self.threshold, "scanned_transactions": scanned},
            "summary": {"ml_candidate_transactions": 0, "excluded_candidate_transactions": 0,
                        "customer_merchant_groups": 0, "high_confidence_recurring_groups": 0,
                        "needs_review_groups": 0, "likely_active_groups": 0,
                        "digital_monthly_by_currency": {}, "digital_yearly_by_currency": {},
                        "bills_monthly_by_currency": {}, "housing_monthly_by_currency": {},
                        "other_monthly_by_currency": {}, "all_recurring_monthly_by_currency": {},
                        "all_recurring_yearly_by_currency": {}},
            "digital_subscriptions": [], "recurring_bills": [], "housing_payments": [],
            "other_or_unclassified": [], "needs_review": [], "upcoming_30_days": [],
            "historical_charge_increases": [],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="FINOVA customer subscription intelligence report")
    parser.add_argument("--transactions", required=True, help="Path to raw transactions.csv")
    parser.add_argument("--model", default="finova_subscription_model.pkl", help="Your trusted .pkl bundle")
    parser.add_argument("--customer-id", required=True, help="Customer whose transactions are authorized")
    parser.add_argument("--as-of", default=None, help="Snapshot date YYYY-MM-DD (default: customer's latest transaction)")
    parser.add_argument("--output", default="finova_subscription_report.json", help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.transactions)
    engine = FinovaSubscriptionEngine(args.model)
    result = engine.report(df, customer_id=args.customer_id, as_of=args.as_of)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINOVA subscription engine completed")
    print(json.dumps(to_json_safe(result["summary"]), indent=2))
    print(f"Customer report saved: {out.resolve()}")


if __name__ == "__main__":
    main()
