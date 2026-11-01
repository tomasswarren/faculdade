"""Lista 6 - VAR e Projecao Local: politica monetaria nos EUA (FRED, mensal, 1990-2019)."""
import json, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/img", exist_ok=True)
SEED, P_LAGS, H_IRF, H_LP = 12345, 6, 48, 12
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9, "axes.titlesize": 10})
C1, C2, C3 = "#1f4e79", "#c0392b", "#7f8c8d"

# ---------------------------------------------------------------- dados
FRED = {"PPIACO": "Índice de preços de commodities (PPI)", "INDPRO": "Produção industrial",
        "CPIAUCSL": "CPI (todos os itens)", "UNRATE": "Taxa de desemprego (%)",
        "FEDFUNDS": "Fed Funds rate (% a.a.)"}


def load(s):
    f = f"{HERE}/data/{s}.csv"
    if not os.path.exists(f):
        pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}").to_csv(f, index=False)
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    return d.iloc[:, 0].astype(float)


raw = pd.concat({s: load(s) for s in FRED}, axis=1).loc["1989-12-01":"2019-12-01"].dropna()
d = pd.DataFrame({"dlppi": 100 * np.log(raw.PPIACO).diff(), "dlip": 100 * np.log(raw.INDPRO).diff(),
                  "dlcpi": 100 * np.log(raw.CPIAUCSL).diff(), "ur": raw.UNRATE, "ffr": raw.FEDFUNDS}).dropna()
names = list(d.columns)
lab = {"dlppi": "Commodities (Δlog PPI)", "dlip": "Produção ind. (Δlog)", "dlcpi": "CPI (Δlog)",
       "ur": "Desemprego (%)", "ffr": "Fed Funds (%)"}
out = {"n_obs": int(len(d)), "start": str(d.index[0].date()), "end": str(d.index[-1].date()), "p": P_LAGS}

# ---------------------------------------------------------------- figuras de dados
fig, ax = plt.subplots(3, 2, figsize=(9, 7.5))
for a, (s, t) in zip(ax.flat, FRED.items()):
    a.plot(raw[s], color=C1, lw=1.2); a.set_title(t)
ax.flat[-1].axis("off")
fig.tight_layout(); fig.savefig(f"{HERE}/img/series_originais.png"); plt.close(fig)

fig, ax = plt.subplots(3, 2, figsize=(9, 7.5))
for a, c in zip(ax.flat, names):
    a.plot(d[c], color=C1, lw=1); a.axhline(0, color="k", lw=.5); a.set_title(lab[c])
ax.flat[-1].axis("off")
fig.tight_layout(); fig.savefig(f"{HERE}/img/series_transformadas.png"); plt.close(fig)

# testes de raiz unitaria
ur = {}
for c in names:
    ur[c] = {"adf_p": float(adfuller(d[c], regression="c", autolag="AIC")[1]),
             "kpss_p": float(kpss(d[c], regression="c", nlags="auto")[1])}
out["unit_root"] = ur

# ---------------------------------------------------------------- VAR
m = VAR(d)
sel = m.select_order(12)
out["lag_sel"] = {k: int(v) for k, v in sel.selected_orders.items()}
out["lag_table"] = {str(i): {"aic": float(a), "bic": float(b), "hqic": float(h)}
                    for i, (a, b, h) in enumerate(zip(*[sel.ics[k] for k in ("aic", "bic", "hqic")]))}
res = m.fit(P_LAGS)
out["stable"] = bool(res.is_stable()); out["max_root"] = float(np.max(np.abs(res.roots)))
out["max_root"] = float(1 / np.min(np.abs(res.roots)))  # statsmodels: roots do polinomio; |.|>1 => estavel
out["lb"] = {c: {"p12": float(acorr_ljungbox(res.resid[c], lags=[12], return_df=True).lb_pvalue.iloc[0]),
                 "p24": float(acorr_ljungbox(res.resid[c], lags=[24], return_df=True).lb_pvalue.iloc[0])} for c in names}
out["portmanteau_p"] = float(res.test_whiteness(nlags=24, adjusted=True).pvalue)
out["normality_p"] = float(res.test_normality().pvalue)

fig, ax = plt.subplots(5, 1, figsize=(8, 9))
for a, c in zip(ax, names):
    plot_acf(res.resid[c], lags=36, ax=a, title=f"ACF dos resíduos – {lab[c]}", color=C1, vlines_kwargs={"colors": C1})
    a.set_ylim(-.3, .3)
fig.tight_layout(); fig.savefig(f"{HERE}/img/acf_residuos.png"); plt.close(fig)

# ---------------------------------------------------------------- FRI (Cholesky), IC 90% por Monte Carlo
irf = res.irf(H_IRF)
Pchol = np.linalg.cholesky(res.sigma_u.values)
out["chol_sd_ffr"] = float(Pchol[-1, -1])
cum_idx = [0, 1, 2]  # variaveis em diferenca -> mostrar resposta acumulada (nivel)
orth = irf.orth_irfs.copy()
orth_cum = irf.orth_cum_effects
def boot_irf(res, reps=1000, seed=SEED):
    """Bootstrap de residuos (recursivo): percentis 5/95 das FRI ortogonalizadas (nivel e acumulada)."""
    rng = np.random.default_rng(seed)
    u = res.resid.values - res.resid.values.mean(0)
    A = [res.coefs[k] for k in range(P_LAGS)]; nu = res.params.loc["const"].values
    Y0 = d.values[:P_LAGS]; T = len(u)
    I, C = [], []
    for _ in range(reps):
        e = u[rng.integers(0, T, T)]
        Y = np.zeros((T + P_LAGS, 5)); Y[:P_LAGS] = Y0
        for t in range(P_LAGS, T + P_LAGS):
            Y[t] = nu + sum(A[k] @ Y[t - 1 - k] for k in range(P_LAGS)) + e[t - P_LAGS]
        rb = VAR(pd.DataFrame(Y, columns=names)).fit(P_LAGS).irf(H_IRF)
        I.append(rb.orth_irfs); C.append(rb.orth_cum_effects)
    q = lambda X: (np.percentile(X, 5, axis=0), np.percentile(X, 95, axis=0))
    return q(I), q(C)


(lo, hi), (lo_c, hi_c) = boot_irf(res)
out["ic_method"] = "bootstrap de resíduos, 1000 réplicas, percentis 5% e 95%"
point = np.where(np.isin(np.arange(5), cum_idx)[None, :, None], orth_cum, orth)
LO = np.where(np.isin(np.arange(5), cum_idx)[None, :, None], lo_c, lo)
HI = np.where(np.isin(np.arange(5), cum_idx)[None, :, None], hi_c, hi)
hh = np.arange(H_IRF + 1)

fig, ax = plt.subplots(5, 5, figsize=(14, 11), sharex=True)
for i in range(5):          # resposta de i
    for j in range(5):      # a choque j
        a = ax[i, j]
        a.fill_between(hh, LO[:, i, j], HI[:, i, j], color=C1, alpha=.18, lw=0)
        a.plot(hh, point[:, i, j], color=C1, lw=1.3); a.axhline(0, color="k", lw=.6)
        if i == 0: a.set_title(f"Choque: {lab[names[j]].split(' (')[0]}", fontsize=8)
        if j == 0: a.set_ylabel(lab[names[i]].split(" (")[0] + ("\n(acum.)" if i in cum_idx else ""), fontsize=8)
for a in ax[-1]: a.set_xlabel("meses")
fig.suptitle("FRI ortogonalizada (Cholesky), choque de 1 d.p. – IC 90% (bootstrap)", y=.995)
fig.tight_layout(); fig.savefig(f"{HERE}/img/fri_completa.png"); plt.close(fig)

fig, ax = plt.subplots(1, 5, figsize=(14, 2.8))
for i in range(5):
    a = ax[i]; a.fill_between(hh, LO[:, i, 4], HI[:, i, 4], color=C1, alpha=.18, lw=0)
    a.plot(hh, point[:, i, 4], color=C1, lw=1.5); a.axhline(0, color="k", lw=.6)
    a.set_title(lab[names[i]] + (" – acum." if i in cum_idx else "")); a.set_xlabel("meses")
fig.suptitle("Resposta a um choque de política monetária (+1 d.p. ≈ %.2f p.p. no Fed Funds)" % Pchol[-1, -1], y=1.03)
fig.tight_layout(); fig.savefig(f"{HERE}/img/fri_choque_monetario.png", bbox_inches="tight"); plt.close(fig)
out["irf_mp"] = {names[i]: {"h": [int(x) for x in (0, 6, 12, 24, 48)],
                            "point": [float(point[h, i, 4]) for h in (0, 6, 12, 24, 48)],
                            "lo": [float(LO[h, i, 4]) for h in (0, 6, 12, 24, 48)],
                            "hi": [float(HI[h, i, 4]) for h in (0, 6, 12, 24, 48)]} for i in range(5)}
# pico da resposta
out["irf_peaks"] = {names[i]: {"h": int(np.argmax(np.abs(point[:, i, 4]))),
                               "v": float(point[np.argmax(np.abs(point[:, i, 4])), i, 4])} for i in range(5)}

# ---------------------------------------------------------------- FEVD
fe = res.fevd(H_IRF).decomp   # (neqs, steps, neqs)
out["fevd"] = {str(hy): {names[i]: {names[j]: float(100 * fe[i, hy - 1, j]) for j in range(5)} for i in range(5)}
               for hy in (12, 24, 48)}
fig, ax = plt.subplots(1, 5, figsize=(14, 2.9), sharey=True)
cols = ["#8e44ad", "#27ae60", "#e67e22", "#7f8c8d", C2]
for i in range(5):
    ax[i].stackplot(np.arange(1, H_IRF + 1), 100 * fe[i].T, colors=cols, labels=[lab[n].split(" (")[0] for n in names], alpha=.85)
    ax[i].set_title(lab[names[i]]); ax[i].set_xlabel("meses"); ax[i].set_xlim(1, H_IRF)
    for x in (12, 24, 48): ax[i].axvline(x, color="k", lw=.5, ls=":")
ax[0].set_ylabel("% da variância"); ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(f"{HERE}/img/fevd.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Projecao Local (Jorda, 2005)
# Choque = inovacao no Fed Funds identificada recursivamente: ffr_t entra por ULTIMO; controles = valores
# contemporaneos das demais variaveis + P lags de todas. Coef. de ffr_t = resposta a 1 p.p. de inovacao.
lvl = pd.DataFrame({"dlppi": 100 * np.log(raw.PPIACO), "dlip": 100 * np.log(raw.INDPRO), "dlcpi": 100 * np.log(raw.CPIAUCSL),
                    "ur": raw.UNRATE, "ffr": raw.FEDFUNDS}).loc[d.index[0] - pd.offsets.MonthBegin(1):]
dl = lvl.copy()
Xall = d.copy()
lags = pd.concat([d.shift(k).add_suffix(f"_L{k}") for k in range(1, P_LAGS + 1)], axis=1)
ctrl = pd.concat([d[["dlppi", "dlip", "dlcpi", "ur"]], lags], axis=1)


def lp(yname, H):
    """Retorna DataFrame h x (beta, se, lo, hi) para a resposta de yname a 1 p.p. de choque em ffr."""
    rows = []
    for h in range(H + 1):
        if yname == "ffr":
            y = d["ffr"].shift(-h)
        elif yname == "ur":
            y = d["ur"].shift(-h)
        else:  # resposta acumulada do nivel: x_{t+h} - x_{t-1}
            y = lvl[yname].shift(-h).loc[d.index] - lvl[yname].shift(1).loc[d.index]
        if h == 0 and yname not in ("ffr",):
            rows.append((0.0, 0.0, 0.0, 0.0)); continue   # restricao de identificacao: efeito contemporaneo nulo
        X = pd.concat([d[["ffr"]], ctrl], axis=1)
        if h == 0 and yname == "ffr":
            rows.append((1.0, 0.0, 1.0, 1.0)); continue
        Z = sm.add_constant(X); df = pd.concat([y.rename("y"), Z], axis=1).dropna()
        r = sm.OLS(df["y"], df.drop(columns="y")).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        b, s = r.params["ffr"], r.bse["ffr"]
        rows.append((b, s, b - 1.645 * s, b + 1.645 * s))
    return pd.DataFrame(rows, columns=["b", "se", "lo", "hi"])


sd = Pchol[-1, -1]
LP = {n: lp(n, H_IRF) for n in names}
fig, ax = plt.subplots(1, 5, figsize=(14, 2.9))
for i, n in enumerate(names):
    r = LP[n].iloc[:H_LP + 1]; x = np.arange(H_LP + 1)
    ax[i].fill_between(x, r.lo, r.hi, color=C2, alpha=.2, lw=0); ax[i].plot(x, r.b, color=C2, lw=1.5, marker="o", ms=3)
    ax[i].axhline(0, color="k", lw=.6); ax[i].set_title(lab[n] + (" – acum." if n in ("dlppi", "dlip", "dlcpi") else "")); ax[i].set_xlabel("meses")
fig.suptitle("Projeção Local – resposta a +1 p.p. de inovação no Fed Funds (IC 90%, erros HAC Newey-West)", y=1.03)
fig.tight_layout(); fig.savefig(f"{HERE}/img/lp_fri.png", bbox_inches="tight"); plt.close(fig)

# comparacao LP x VAR (choque de 1 d.p. do VAR = sd p.p.)
fig, ax = plt.subplots(1, 4, figsize=(13, 3))
for a, n in zip(ax, ["dlip", "dlcpi", "ur", "ffr"]):
    i = names.index(n); r = LP[n]; h = np.arange(H_LP + 1)
    a.fill_between(h, sd * r.lo[:H_LP + 1], sd * r.hi[:H_LP + 1], color=C2, alpha=.18, lw=0)
    a.plot(h, sd * r.b[:H_LP + 1], color=C2, lw=1.5, label="LP (IC 90%)")
    a.fill_between(h, LO[:H_LP + 1, i, 4], HI[:H_LP + 1, i, 4], color=C1, alpha=.15, lw=0)
    a.plot(h, point[:H_LP + 1, i, 4], color=C1, lw=1.5, ls="--", label="VAR (IC 90%)")
    a.axhline(0, color="k", lw=.6); a.set_title(lab[n]); a.set_xlabel("meses")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("LP × VAR – choque de 1 d.p. em política monetária", y=1.04)
fig.tight_layout(); fig.savefig(f"{HERE}/img/lp_vs_var.png", bbox_inches="tight"); plt.close(fig)

out["lp"] = {n: {str(h): {"b": float(LP[n].b[h]), "lo": float(LP[n].lo[h]), "hi": float(LP[n].hi[h])}
                 for h in (0, 1, 3, 6, 9, 12)} for n in names}
out["sd_ffr"] = float(sd)
json.dump(out, open(f"{HERE}/resultados.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("ok", json.dumps({k: out[k] for k in ("n_obs", "lag_sel", "stable", "max_root", "portmanteau_p", "normality_p", "lb", "irf_peaks", "chol_sd_ffr")}, indent=1))
