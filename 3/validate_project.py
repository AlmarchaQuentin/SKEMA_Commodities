"""Independent checks of timing, targets, risk estimates and cash ledgers."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
root = args.root.resolve()
checks = []


def check(name, condition, detail=""):
    if not bool(condition):
        raise AssertionError(name + ": " + str(detail))
    checks.append({"check": name, "status": "passed", "detail": str(detail)})


db = pd.read_csv(root / "3. Final_DB/final_db.csv", parse_dates=["date"])
px = pd.read_csv(root / "2. Data Processing/prices_processed.csv", parse_dates=["date", "delivery", "expiry"])
check("Unique chronological sessions", db.date.is_monotonic_increasing and not db.date.duplicated().any(), len(db))
check("Three euro spreads reconcile", np.allclose(db.s13, db.s12 + db.s23))
check("Complete daily risk estimates in study sample", db[["vol_m1", "vol_m2", "vol_m3", "sd_ds12", "sd_ds23", "sd_ds13"]].notna().all().all())
check("Unverified source mappings remain explicit", not db.contracts_source_verified.any() and not db.switch_source_verified.any())

oct7 = px.loc[px.date.eq("2026-10-07")].iloc[0]
check("Expiry independent of truncated dataset end", oct7.expiry == pd.Timestamp("2026-10-29") and oct7.dte == 16, f"{oct7.expiry.date()}, {oct7.dte} UK business days")
for month, hours in [("2024-02", 696), ("2024-03", 743), ("2024-10", 745), ("2025-02", 672)]:
    values = px.loc[px.m1_contract.eq(month), "m1_hours"]
    check("Delivery MWh hours " + month, len(values) > 0 and values.eq(hours).all(), hours)

for prefix in ["storage", "lng", "flow", "wx", "fc"]:
    available = pd.to_datetime(db[f"{prefix}_available_date"])
    check(prefix + " available before signal", (available.isna() | available.le(db.date)).all())
for prefix, source_col, lag in [("storage", "storage_gas_day", 2), ("lng", "lng_gas_day", 2), ("flow", "flow_source_gas_day", 6), ("wx", "wx_source_date", 6), ("fc", "fc_source_issue_date", 1)]:
    source = pd.to_datetime(db[source_col])
    available = pd.to_datetime(db[f"{prefix}_available_date"])
    valid = source.notna()
    check(prefix + " declared safety lag", available[valid].sub(source[valid]).dt.days.eq(lag).all(), f"{lag} calendar days")

for date in ["2019-01-03", "2022-02-24", "2026-10-07"]:
    history = px.loc[px.date.le(date)].copy()
    for spread in ["s12", "s23", "s13"]:
        delta = history[spread].diff().loc[history.roll_day.ne(1)].dropna().tail(20)
        got = history[f"sd_d{spread}"].iloc[-1]
        check(f"Causal risk spot check {spread} {date}", len(delta) == 20 and np.isclose(got, delta.std()))

dates = db.date
for h in [1, 5, 10]:
    entry = pd.to_datetime(db[f"entry_date{h}"])
    exit_date = pd.to_datetime(db[f"exit_date{h}"])
    same = db.m1_contract.eq(db.m1_contract.shift(-(h + 1))) & db.m1_contract.eq(db.m1_contract.shift(-1))
    eligible = same & dates.shift(-(h + 1)).notna()
    check(f"Trade eligibility horizon {h}", db[f"trade_valid{h}"].eq(eligible).all(), int(eligible.sum()))
    check(f"Trade endpoints horizon {h}", entry[eligible].eq(dates.shift(-1)[eligible]).all() and exit_date[eligible].eq(dates.shift(-(h + 1))[eligible]).all())
    for spread in ["s12", "s23", "s13"]:
        expected = (db[spread].shift(-(h + 1)) - db[spread].shift(-1)).where(eligible)
        actual = db[f"d{spread}_trade{h}"]
        check(f"Execution-aligned target {spread} horizon {h}", np.allclose(actual, expected, equal_nan=True))
    check(f"No target beyond sample horizon {h}", exit_date[db[f"ds12_trade{h}"].notna()].le(dates.max()).all())

pred = pd.read_csv(root / "4. Modelling/model_predictions.csv", parse_dates=["date", "entry_date", "exit_date", "fit_end", "refit_date", "last_train_exit"])
check("Unique model predictions", not pred.duplicated(["date", "model", "horizon", "spread"]).any(), len(pred))
fitted = pred.model.ne("no_change")
check("Training labels completed before refit", pred.loc[fitted, "last_train_exit"].lt(pred.loc[fitted, "refit_date"]).all())
check("Refit precedes every signal", pred.refit_date.le(pred.date).all())
check("Training rows precede signal", pred.loc[fitted, "fit_end"].lt(pred.loc[fitted, "date"]).all())
check("Training warm-up", pred.loc[fitted, "train_n"].ge(365).all())
wide = pred.pivot(index=["date", "model", "horizon"], columns="spread", values="prediction")
check("Forecast triangle identity", wide[["s12", "s23", "s13"]].notna().all().all() and np.allclose(wide.s13, wide.s12 + wide.s23))
check("No-change benchmark", pred.loc[pred.model.eq("no_change"), "prediction"].eq(0).all())
lookup = db.set_index("date")
for (h, spread), rows in pred.groupby(["horizon", "spread"]):
    expected = rows.date.map(lookup[f"d{spread}_trade{h}"])
    check(f"Scored model target {spread} horizon {h}", np.allclose(rows.actual, expected, equal_nan=True))

cfg = json.loads((root / "4. Strategy/strategy_assumptions.json").read_text())
trades = pd.read_csv(root / "4. Strategy/strategy_trades.csv", parse_dates=["date", "entry_date", "exit_date"])
cash = pd.read_csv(root / "5. Backtest/per_trade.csv", parse_dates=["date", "entry_date", "exit_date"])
daily = pd.read_csv(root / "5. Backtest/daily_pnl.csv", parse_dates=["date"])
metrics = pd.read_csv(root / "5. Backtest/metrics.csv")
keys = ["model", "spread", "horizon", "sample", "cost_multiplier"]
account_keys = ["model", "spread", "horizon", "sample"]
check("Unique strategy trades", trades.trade_id.is_unique, len(trades))
check("Signal threshold uses assumed roundtrip cost", np.isclose(cfg["roundtrip_cost_eur_per_mwh"], 0.05))
check("Equal-MWh hedge in both delivery legs", np.allclose(trades.mw_a * trades.hours_a, trades.direction * trades.quantity_mwh_per_leg) and np.allclose(trades.mw_b * trades.hours_b, -trades.direction * trades.quantity_mwh_per_leg))
cash_reference = trades.mw_a * trades.hours_a * (trades.exit_a - trades.entry_a) + trades.mw_b * trades.hours_b * (trades.exit_b - trades.entry_b)
check("Two-leg cash profit reconciles", np.allclose(cash_reference, trades.gross_pnl_eur, atol=1e-7))
check("Spread profit reconciles", np.allclose(trades.direction * trades.quantity_mwh_per_leg * trades.actual, trades.gross_pnl_eur, atol=1e-7))
for spread, rows in trades.groupby("spread"):
    sd = rows.date.map(lookup[f"sd_d{spread}"])
    check(f"Signal-day risk {spread}", np.allclose(sd, rows.sd_signal))
    quantity = np.minimum(cfg["quantity_cap_mwh_per_leg"], cfg["risk_budget_eur"] / (sd * np.sqrt(rows.horizon)))
    check(f"Predetermined position size {spread}", np.allclose(quantity, rows.quantity_mwh_per_leg))
    threshold = np.maximum(cfg["roundtrip_cost_eur_per_mwh"], cfg["signal_sigma_multiple"] * sd * np.sqrt(rows.horizon))
    check(f"Predetermined signal threshold {spread}", np.allclose(threshold, rows.entry_threshold) and rows.prediction.abs().gt(threshold).all())
for key, rows in trades.groupby(account_keys):
    rows = rows.sort_values("entry_date")
    if not (rows.entry_date.iloc[1:].to_numpy() > rows.exit_date.iloc[:-1].to_numpy()).all():
        raise AssertionError(f"Overlapping trades: {key}")
check("No overlapping standalone trades", True)
positions = pd.Series(np.arange(len(db)), index=db.date)
check("Next-session execution and exact holding period", np.array_equal(trades.entry_date.map(positions), trades.date.map(positions) + 1) and np.array_equal(trades.exit_date.map(positions) - trades.entry_date.map(positions), trades.horizon))
for row in trades.itertuples(index=False):
    path = lookup.loc[row.date:row.exit_date]
    if not all(path[f"{leg}_contract"].nunique() == 1 for leg in [row.leg_a, row.leg_b]):
        raise AssertionError(f"Trade changes contract: {row.trade_id}")
check("Every trade keeps its delivery contracts", True)
check("All cost scenarios keep the same trades", cash.groupby("trade_id").size().eq(len(cfg["cost_multipliers"])).all() and cash.groupby("trade_id").quantity_mwh_per_leg.nunique().eq(1).all() and cash.groupby("trade_id").gross_pnl_eur.nunique().eq(1).all())
check("Roundtrip costs charge both legs twice", np.allclose(cash.cost_eur, cash.quantity_mwh_per_leg * cfg["roundtrip_cost_eur_per_mwh"] * cash.cost_multiplier))
check("Net trade cash profit", np.allclose(cash.net_pnl_eur, cash.gross_pnl_eur - cash.cost_eur))
check("Unique daily account observations", not daily.duplicated(keys + ["date"]).any(), len(daily))
check("Daily net profit", np.allclose(daily.net_pnl_eur, daily.gross_pnl_eur - daily.cost_eur))
total_cols = ["gross_pnl_eur", "cost_eur", "net_pnl_eur", "turnover_mwh", "turnover_eur"]
cash_totals = cash.groupby(keys)[total_cols].sum()
daily_totals = daily.groupby(keys)[total_cols].sum()
cash_totals = cash_totals.reindex(daily_totals.index).fillna(0)
check("All trade cash totals reconcile to daily accounts", np.allclose(cash_totals, daily_totals, atol=1e-6))
metric_totals = metrics.set_index(keys)[total_cols].reindex(daily_totals.index)
check("Published metrics reconcile to cash totals", np.allclose(metric_totals, daily_totals, atol=1e-6))
for key, rows in daily.groupby(keys):
    rows = rows.sort_values("date")
    reference = cfg["initial_capital_eur"] + rows.net_pnl_eur.cumsum()
    peaks = np.maximum.accumulate(np.r_[cfg["initial_capital_eur"], reference])[1:]
    if not (np.allclose(rows.equity_eur, reference) and np.allclose(rows.drawdown_eur, reference - peaks) and np.allclose(rows.drawdown_pct, reference / peaks - 1)):
        raise AssertionError(f"Cash equity or drawdown: {key}")
check("Every daily equity and drawdown path reconciles", True)
zeros = daily.model.eq("no_change")
check("No-change accounts remain flat", daily.loc[zeros, "net_pnl_eur"].eq(0).all() and daily.loc[zeros, "equity_eur"].eq(cfg["initial_capital_eur"]).all())
insolvent = metrics.nonpositive_equity_days.gt(0)
check("Capital exhaustion does not produce a valid Sharpe", metrics.loc[insolvent, "daily_cash_sharpe"].isna().all(), f"{int(insolvent.sum())} scenarios exhaust capital")
check("Capital exhaustion is explicitly reported", metrics.loc[insolvent, "funding_status"].eq("capital_exhausted").all())
check("Account metrics have unique keys", not metrics.duplicated(keys).any(), len(metrics))

report = {"status": "passed", "checks": checks, "rows": len(db), "predictions": len(pred), "standalone_trades": len(trades), "daily_account_records": len(daily), "cost_scenarios_exhausting_capital": int(insolvent.sum())}
(root / "Documentation/validation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Passed {len(checks)} independent checks; {len(db)} sessions; {len(pred)} predictions; {len(trades)} standalone trades.")
