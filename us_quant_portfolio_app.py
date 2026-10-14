
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

st.set_page_config(page_title="Projek UTS Model Quantitative Sederhana", layout="wide")

st.title("Projek UTS Semester 1")
st.caption("Nama    : Dama Najib Raspati")
st.caption("NIM     : K1B026115")
st.caption("Kelas   : C")

# -----------------------------
# sidebar
# -----------------------------
st.sidebar.header("Parameter")

tickers = {
    "Saham Amerika": "AAPL",
    "Index Amerika": "^GSPC",
    "Emas": "GLD",
    "Kripto": "BTC-USD",
}

period = st.sidebar.selectbox(
    "Periode data historis",
    ["3y", "5y", "10y", "max"],
    index=1
)

rf = st.sidebar.number_input(
    "Risk-free rate (%/tahun)",
    min_value=0.0,
    max_value=15.0,
    value=4.0,
    step=0.25
) / 100

vol_limit = st.sidebar.slider(
    "Batas volatilitas (%/tahun)",
    min_value=5,
    max_value=80,
    value=15,
    step=1
) / 100

n_portfolios = st.sidebar.number_input(
    "Jumlah portofolio acak",
    min_value=500,
    max_value=20000,
    value=3000,
    step=500
)

# -----------------------------
# download data
# -----------------------------
@st.cache_data(ttl=3600)
def get_data(period):
    symbols = list(tickers.values())
    data = yf.download(
        symbols,
        period=period,
        auto_adjust=True,
        progress=False
    )["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(how="all").ffill().dropna()

    # Samakan nama kolom
    rename_map = {v: k for k, v in tickers.items()}
    data = data.rename(columns=rename_map)

    return data

try:
    prices = get_data(period)
except Exception as e:
    st.error(f"Gagal mengambil data Yahoo Finance: {e}")
    st.stop()

required = list(tickers.keys())
missing = [x for x in required if x not in prices.columns]

if missing:
    st.error(f"Data tidak tersedia untuk: {missing}")
    st.stop()

prices = prices[required].dropna()
daily_returns = prices.pct_change().dropna()

# -----------------------------
# statistik yang telah dianualisasi
# -----------------------------
mu = daily_returns.mean() * 252
cov = daily_returns.cov() * 252
vol_asset = daily_returns.std() * np.sqrt(252)

# -----------------------------
# portfolio functions
# -----------------------------
def portfolio_return(w):
    return float(np.dot(w, mu.values))

def portfolio_vol(w):
    return float(np.sqrt(w @ cov.values @ w))

def sharpe(w):
    v = portfolio_vol(w)
    if v == 0:
        return -999
    return (portfolio_return(w) - rf) / v

def random_portfolios(n):
    weights = np.random.dirichlet(np.ones(len(required)), size=n)
    rets = weights @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov.values, weights))
    sharpes = (rets - rf) / vols
    return weights, rets, vols, sharpes

# -----------------------------
# random assets
# -----------------------------
np.random.seed(42)
W, R, V, S = random_portfolios(int(n_portfolios))

# -----------------------------
# max sharpe portfolio
# -----------------------------
n = len(required)
x0 = np.repeat(1/n, n)

constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1}
]
bounds = [(0, 1)] * n

opt_sharpe = minimize(
    lambda w: -sharpe(w),
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 2000, "ftol": 1e-12}
)

w_sharpe = opt_sharpe.x
ret_sharpe = portfolio_return(w_sharpe)
vol_sharpe = portfolio_vol(w_sharpe)
sr_sharpe = sharpe(w_sharpe)

# -----------------------------
# return terbaik dibawah batas volatilitas
# -----------------------------
vol_constraint = {
    "type": "ineq",
    "fun": lambda w: vol_limit - portfolio_vol(w)
}

opt_cap = minimize(
    lambda w: -portfolio_return(w),
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints + [vol_constraint],
    options={"maxiter": 3000, "ftol": 1e-12}
)

cap_ok = opt_cap.success and portfolio_vol(opt_cap.x) <= vol_limit + 1e-5

if cap_ok:
    w_cap = opt_cap.x
    ret_cap = portfolio_return(w_cap)
    vol_cap = portfolio_vol(w_cap)
    sr_cap = sharpe(w_cap)
else:
    # Fallback: closest random portfolio below the cap
    feasible = V <= vol_limit
    if feasible.any():
        idx = np.argmax(R[feasible])
        wf = W[feasible][idx]
        w_cap = wf
        ret_cap = portfolio_return(w_cap)
        vol_cap = portfolio_vol(w_cap)
        sr_cap = sharpe(w_cap)
    else:
        w_cap = w_sharpe
        ret_cap = ret_sharpe
        vol_cap = vol_sharpe
        sr_cap = sr_sharpe

# -----------------------------
# bagian efficient frontier
# -----------------------------
target_vols = np.linspace(max(vol_asset.min() * 0.90, 0.01), max(V.max(), vol_asset.max()), 80)

frontier_vol = []
frontier_ret = []

for target in target_vols:
    con = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, t=target: portfolio_vol(w) - t}
    ]
    res = minimize(
        lambda w: -portfolio_return(w),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=con,
        options={"maxiter": 1000, "ftol": 1e-10}
    )
    if res.success:
        frontier_vol.append(portfolio_vol(res.x))
        frontier_ret.append(portfolio_return(res.x))

# -----------------------------
# bagian plot
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 6))

sc = ax.scatter(
    V * 100,
    R * 100,
    c=S,
    s=8,
    alpha=0.28,
    cmap="viridis",
    label=f"{int(n_portfolios):,} portofolio acak"
)

if frontier_vol:
    ax.plot(
        np.array(frontier_vol) * 100,
        np.array(frontier_ret) * 100,
        linewidth=2.5,
        label="Efficient frontier"
    )

ax.scatter(
    vol_sharpe * 100,
    ret_sharpe * 100,
    s=230,
    marker="o",
    edgecolors="white",
    linewidths=2,
    label="Sharpe maksimum"
)

ax.scatter(
    vol_cap * 100,
    ret_cap * 100,
    s=230,
    marker="o",
    edgecolors="white",
    linewidths=2,
    label=f"Pilihan — batas {vol_limit*100:.0f}%"
)

# Asset individual
for i, name in enumerate(required):
    ax.scatter(
        vol_asset.iloc[i] * 100,
        mu.iloc[i] * 100,
        s=80
    )
    ax.annotate(
        name,
        (vol_asset.iloc[i] * 100, mu.iloc[i] * 100),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9
    )

ax.axvline(
    vol_limit * 100,
    linestyle="--",
    linewidth=1.5,
    alpha=0.7
)

ax.set_xlabel("Volatilitas (%/tahun)")
ax.set_ylabel("Return (%/tahun)")
ax.set_title("Risk–Return Space")
ax.grid(alpha=0.20)
ax.legend(loc="best")
st.pyplot(fig, use_container_width=True)

# -----------------------------
# bagian metrics utama
# -----------------------------
st.subheader("Portofolio dengan batas volatilitas")

c1, c2, c3 = st.columns(3)
c1.metric("Return/tahun", f"{ret_cap*100:.2f}%")
c2.metric("Volatilitas", f"{vol_cap*100:.2f}%")
c3.metric("Sharpe", f"{sr_cap:.2f}")

st.subheader("Alokasi aset — portofolio pilihan")

alloc = pd.DataFrame({
    "Aset": required,
    "Ticker": list(tickers.values()),
    "Bobot": w_cap * 100
})

st.dataframe(
    alloc.style.format({"Bobot": "{:.2f}%"}),
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# sharpe portfolio
# -----------------------------
st.subheader("Portofolio Sharpe maksimum")

c1, c2, c3 = st.columns(3)
c1.metric("Return/tahun", f"{ret_sharpe*100:.2f}%")
c2.metric("Volatilitas", f"{vol_sharpe*100:.2f}%")
c3.metric("Sharpe", f"{sr_sharpe:.2f}")

sharpe_alloc = pd.DataFrame({
    "Aset": required,
    "Ticker": list(tickers.values()),
    "Bobot": w_sharpe * 100
})
st.dataframe(
    sharpe_alloc.style.format({"Bobot": "{:.2f}%"}),
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# statistics asset
# -----------------------------
st.subheader("Statistik masing-masing aset")

asset_stats = pd.DataFrame({
    "Return tahunan": mu * 100,
    "Volatilitas tahunan": vol_asset * 100,
    "Sharpe": (mu - rf) / vol_asset
})

st.dataframe(
    asset_stats.style.format({
        "Return tahunan": "{:.2f}%",
        "Volatilitas tahunan": "{:.2f}%",
        "Sharpe": "{:.2f}"
    }),
    use_container_width=True
)

st.caption(
    "Catatan: hasil adalah estimasi berdasarkan data historis. Return masa lalu bukan jaminan return masa depan. "
    "Model long-only: bobot 0–100% dan total bobot = 100%."
)

st.markdown("""
### Rumus utama

**Return portofolio**
Rₚ = Σ(wᵢ × Rᵢ)

**Volatilitas portofolio**
σₚ = √(wᵀΣw)

**Sharpe ratio**
Sharpe = (Rₚ − Rf) / σₚ

Data harian diannualisasi dengan 252 hari perdagangan.
""")