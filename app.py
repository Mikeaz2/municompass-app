import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="MuniCompass — TEY & Fiscal Health Screener", layout="wide")
st.title("MuniCompass — TEY & Fiscal Health Screener")
st.caption("Find true after-tax value — where yield meets fiscal strength.")

STATE_TAX = {"FL":0.0, "TX":0.0, "NY":0.0685, "CA":0.093, "IL":0.0495, "NJ":0.0637}

# ---------------- Sidebar: inputs & filters ----------------
with st.sidebar:
    st.header("Investor Profile")
    fed_bracket = st.number_input("Federal tax bracket (%)", 0, 50, 24) / 100
    investor_state = st.selectbox("State of residence", ["FL","NY","CA","TX","IL","NJ"])
    horizon = st.selectbox("Investment horizon", ["3 yrs","5 yrs","7 yrs","10 yrs"])
    goal = st.selectbox("Optimization goal", ["Max after-tax yield","Balanced","Conservative"])

    st.header("Screener Filters")
    only_surplus = st.toggle("Exclude deficit states (surplus-only)", value=True)
    min_risk = st.slider("Min RiskScore", 0, 100, 60)
    min_adj_tey = st.slider("Min Adjusted TEY (%)", 0.0, 8.0, 4.0, step=0.1)

from pathlib import Path

@st.cache_data
def load_data():
    base = Path(__file__).parent  # Folder where app.py is located
    munis_path = base / "data" / "munis.csv"
    fiscal_path = base / "data" / "fiscal_states.csv"

    munis = pd.read_csv(munis_path)
    fiscal = pd.read_csv(fiscal_path)

    for c in ["coupon","price","ytm"]:
        munis[c] = munis[c].astype(float)
    for c in ["pension_score","budget_score","maturity_year"]:
        munis[c] = munis[c].astype(int)
    fiscal["surplus_flag"] = fiscal["surplus_flag"].astype(bool)
    return munis, fiscal

munis, fiscal = load_data()

# dynamic sidebar multiselects after data loads
with st.sidebar:
    state_filter = st.multiselect("Issuer state filter", sorted(munis["issuer_state"].unique()))
    type_filter  = st.multiselect("Type filter", sorted(munis["type"].unique()))

# ---------------- Finance helpers ----------------
def combined_tax_rate(fed, state): return fed + state - fed*state
def per_bond_tax_rate(fed, investor_state, issuer_state):
    return combined_tax_rate(fed, STATE_TAX.get(investor_state,0.0)) if investor_state==issuer_state else fed
def compute_tey(ytm, tax_rate):
    denom = max(1e-6, 1.0 - tax_rate)
    return (ytm / denom) * 100.0
def call_penalty(call_date_str):
    if isinstance(call_date_str, str) and call_date_str.strip().lower() != "no":
        try:
            call_dt = parser.parse(call_date_str).date()
            years = max(0.0, (call_dt - datetime.today().date()).days / 365.25)
            if years <= 1.5: return 0.30
            if years <= 3.0: return 0.20
            return 0.10
        except: return 0.20
    return 0.0
def pension_adj(score): return 0.10 if score>=75 else (0.00 if score>=60 else -0.15)
def budget_adj(score):  return 0.10 if score>=80 else (0.00 if score>=65 else -0.10)

# ---------------- Core calc: TEY, Risk, Tier ----------------
def build_calc(munis: pd.DataFrame, fiscal: pd.DataFrame) -> pd.DataFrame:
    df = munis.merge(
        fiscal[["state","surplus_flag","budget_score"]].rename(
            columns={"state":"issuer_state","budget_score":"state_budget_score"}),
        on="issuer_state", how="left"
    )

    df["budget_signal"] = np.where(
        df["state_budget_score"].notna(),
        (0.6*df["budget_score"] + 0.4*df["state_budget_score"]).round(0),
        df["budget_score"]
    )

    rows = []
    for _, r in df.iterrows():
        tr = per_bond_tax_rate(fed_bracket, investor_state, r["issuer_state"])
        tey = compute_tey(r["ytm"], tr)
        adj = -call_penalty(r["call_date"]) + pension_adj(r["pension_score"]) + budget_adj(r["budget_signal"])
        rows.append({**r.to_dict(),"TEY_%":round(tey,2),"Adj_%":round(adj,2),"TEY_Adj_%":round(tey+adj,2)})
    out = pd.DataFrame(rows)

    # RiskScore (0-100)
    def call_sub(call_date):
        pen = call_penalty(call_date)
        return max(0, 100 - int(pen*100))
    out["CallSub"] = out["call_date"].astype(str).apply(call_sub)
    out["RiskScore"] = (0.35*out["CallSub"] + 0.35*out["pension_score"] + 0.30*out["budget_signal"]).round(0)

    # Tiering by TEY_Adj% + RiskScore
    adj_med = out["TEY_Adj_%"].median() if len(out) else 0
    p40 = np.percentile(out["TEY_Adj_%"], 40) if len(out) else 0
    def tier_row(r):
        if r["RiskScore"]>=70 and r["TEY_Adj_%"]>=adj_med: return "A"
        if r["RiskScore"]>=55 and r["TEY_Adj_%"]>=p40:   return "B"
        return "C"
    out["Tier"] = out.apply(tier_row, axis=1)
    out["TierBadge"] = out["Tier"].map({"A":"🟢 A","B":"🟡 B","C":"🔴 C"})
    return out

df_calc = build_calc(munis, fiscal)

# Filters
mask = pd.Series(True, index=df_calc.index)
if only_surplus: mask &= df_calc["surplus_flag"].fillna(False)
mask &= (df_calc["RiskScore"] >= min_risk) & (df_calc["TEY_Adj_%"] >= min_adj_tey)
if state_filter: mask &= df_calc["issuer_state"].isin(state_filter)
if type_filter:  mask &= df_calc["type"].isin(type_filter)
view = df_calc[mask].copy()

# Sorting
if goal=="Max after-tax yield":
    view = view.sort_values(["TEY_Adj_%","TEY_%"], ascending=False)
elif goal=="Conservative":
    is_callable = view["call_date"].astype(str).str.lower().ne("no")
    view = view.assign(is_callable=is_callable).sort_values(
        ["Tier","is_callable","TEY_Adj_%"], ascending=[True,True,False]
    ).drop(columns="is_callable")
else:
    view = view.sort_values(["Tier","type","TEY_Adj_%"], ascending=[True,True,False])

view["Rank"] = view["TEY_Adj_%"].rank(method="min", ascending=False).astype(int)

# ---------- Filter summary chips ----------
chips = [
    f"{int(fed_bracket*100)}% Fed",
    f"Resident: {investor_state}",
    f"Surplus-only: {'On' if only_surplus else 'Off'}",
    f"Risk ≥ {min_risk}",
    f"Adj TEY ≥ {min_adj_tey:.1f}%"
]
st.write(" | ".join(chips))

# ---------- Screener table ----------
st.subheader("Optimized Screener")
cols = ["Rank","TierBadge","name","type","issuer_state","maturity_year",
        "ytm","TEY_%","Adj_%","TEY_Adj_%","RiskScore","pension_score","budget_signal","call_date"]
st.dataframe(view[cols], use_container_width=True, hide_index=True)

# ---------- Ladder (Plotly) ----------
st.divider()
st.subheader("Tax-Aware Ladder (by adjusted TEY)")
hmap = {"3 yrs":3, "5 yrs":5, "7 yrs":7, "10 yrs":10}
target_rungs = hmap[horizon]
mats = sorted(view["maturity_year"].unique())
if mats:
    if len(mats) >= target_rungs:
        idxs = np.linspace(0, len(mats)-1, target_rungs, dtype=int).tolist()
        rung_years = [mats[i] for i in idxs]
    else:
        rung_years = [mats[i % len(mats)] for i in range(target_rungs)]
    ladder = pd.concat([
        view[view["maturity_year"]==y].sort_values("TEY_Adj_%", ascending=False).head(1).assign(rung=y)
        for y in rung_years if not view[view["maturity_year"]==y].empty
    ])
    fig = px.bar(ladder, x="rung", y="TEY_Adj_%", color="Tier", text="Tier",
                 title=None, labels={"rung":"Maturity Year","TEY_Adj_%":"Adj TEY (%)"})
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=380, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No bonds match the current filters.")

# ---------- Downloads ----------
st.download_button("Download Screener (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                   file_name="municompass_screener.csv", mime="text/csv")
if mats:
    st.download_button("Download Ladder (CSV)", data=ladder[["name","issuer_state","maturity_year","ytm","TEY_%","Adj_%","TEY_Adj_%","Tier"]].to_csv(index=False).encode("utf-8"),
                       file_name="municompass_ladder.csv", mime="text/csv")

# ---------- Alerts ----------
st.divider()
st.subheader("Alerts (demo)")
st.success("TX ISD 2032 — Pension funding +3 → TEY +0.08%")
st.warning("CA Water Rev 2031 — Refunding filed → TEY −0.12%")
st.info("NYC GO 2029 — Budget revision −5 points → TEY −0.05%")

st.caption("Notes: Simplified tax logic (federal exempt; in-state avoids state tax). Risk signals are illustrative for prototype.")
