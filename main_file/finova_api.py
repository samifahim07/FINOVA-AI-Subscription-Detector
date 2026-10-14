"""FINOVA AI subscription report API (local, single-customer prototype).

The browser must not be given the API key. For production, replace this
single-customer API-key demo with per-user login + customer ownership checks.

Required environment variables:
    FINOVA_API_KEY         local test secret, 16+ characters
    FINOVA_CUSTOMER_ID     customer whose data this server is authorized to serve

Expected files next to this script:
    finova_subscription_engine.py
    finova_subscription_model.pkl
    transactions.csv

Run:  py finova_api.py
"""
from __future__ import annotations

import hmac
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from finova_subscription_engine import FinovaSubscriptionEngine, to_json_safe

HERE = Path(__file__).resolve().parent


def create_app() -> Flask:
    """Create a read-only API scoped to a single, server-configured customer."""
    api_key = os.getenv("FINOVA_API_KEY", "")
    customer_id = os.getenv("FINOVA_CUSTOMER_ID", "")
    if len(api_key) < 16:
        raise RuntimeError("Set FINOVA_API_KEY (at least 16 characters).")
    if not customer_id.strip():
        raise RuntimeError("Set FINOVA_CUSTOMER_ID (e.g. C0000047).")

    transactions_path = HERE / "transactions.csv"
    model_path = HERE / "finova_subscription_model.pkl"
    if not transactions_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            "Place transactions.csv and finova_subscription_model.pkl beside finova_api.py"
        )

    # Startup-only load. If the CSV or model changes, restart the process.
    data = pd.read_csv(transactions_path)
    if "customer_id" not in data.columns:
        raise ValueError("transactions.csv is missing customer_id.")
    customer_data = data.loc[data["customer_id"].astype(str) == customer_id].copy()
    del data
    if customer_data.empty:
        raise RuntimeError("Configured customer has no transactions in transactions.csv.")
    engine = FinovaSubscriptionEngine(model_path)  # Trusted local model, never upload a .pkl.

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    @lru_cache(maxsize=1)
    def build_report() -> dict:
        # as_of defaults to the latest observed transaction, NOT today's live bank state.
        return to_json_safe(engine.report(customer_data, customer_id=customer_id))

    @app.after_request
    def security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.before_request
    def authorize_request():
        if request.endpoint == "health":
            return None
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied, api_key):
            return jsonify({"error": "Unauthorized"}), 401
        return None

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "FINOVA subscription API"})

    @app.get("/api/subscriptions/report")
    def report():
        return jsonify(build_report())

    @app.get("/api/subscriptions/summary")
    def summary():
        result = build_report()
        return jsonify({"metadata": result["metadata"], "summary": result["summary"]})

    @app.get("/api/subscriptions/upcoming")
    def upcoming():
        result = build_report()
        return jsonify({"metadata": result["metadata"], "upcoming_30_days": result["upcoming_30_days"]})

    @app.get("/api/subscriptions/review")
    def review():
        result = build_report()
        return jsonify({"metadata": result["metadata"], "needs_review": result["needs_review"]})

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error("Report generation failed: %s", error)
        return jsonify({"error": "Could not generate subscription report."}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host="127.0.0.1",  # Local testing only; not exposed to the public internet.
        port=int(os.getenv("FINOVA_API_PORT", "5001")),
        debug=False,
    )
