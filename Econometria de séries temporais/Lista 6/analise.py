"""Lista 6 - VAR e Projecao Local: politica monetaria nos EUA (FRED, mensal).

Amostra principal: 1990-01 a 2007-12 (antes do limite inferior zero da taxa de juros).
Robustez: amostra completa 1990-2019 e ordenacao alternativa de Cholesky.
Execucao: python analise.py   (gera img/*.png e resultados.json)
"""
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
SEED, P_LAGS, H_IRF, H_LP = 12345, 4, 48, 12
END_MAIN, END_FULL = "2007-12-01", "2019-12-01"
N_BOOT, N_BOOT_BIAS = 1000, 500
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10, "axes.titlesize": 11})
C1, C2, C3 = "#1f4e79", "#c0392b", "#7f8c8d"
RECESSIONS = [("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06")]   # datacao do NBER

# ------------------------------------------------------------------ dados
FRED = {"PPIACO": "PPI – todas as commodities (índice, sem ajuste sazonal)",
        "INDPRO": "Produção industrial (índice, dessazonalizada)",
        "CPIAUCSL": "CPI – todos os itens (índice, dessazonalizado)",
        "UNRATE": "Taxa de desemprego (%, dessazonalizada)",
        "FEDFUNDS": "Fed Funds – taxa efetiva (% a.a.)"}


def load(s):
    f = f"{HERE}/data/{s}.csv"
    if not os.path.exists(f):
        pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}").to_csv(f, index=False)
    return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0].astype(float)


raw_all = pd.concat({s: load(s) for s in FRED}, axis=1).loc["1989-12-01":END_FULL].dropna()
lab = {"dlppi": "Commodities (Δln PPI)", "dlip": "Produção ind. (Δln)", "dlcpi": "CPI (Δln)",
       "ur": "Desemprego", "ffr": "Fed Funds"}
short = {k: v.split(" (")[0] for k, v in lab.items()}
DIFF = ("dlppi", "dlip", "dlcpi")
ORDER = ["dlip", "dlcpi", "dlppi", "ur", "ffr"]          # ordem de Christiano-Eichenbaum-Evans (1999)
ORDER_ALT = ["dlppi", "dlip", "dlcpi", "ur", "ffr"]      # ordem alternativa (commodities primeiro)


def deseason(s):
    """Remove a componente sazonal deterministica (medias mensais) de uma serie; preserva a media geral."""
    D = pd.get_dummies(s.index.month).astype(float).set_index(s.index)
    r = sm.OLS(s, D).fit()
    return s - r.fittedvalues + s.mean()


def transform(raw):
    t = pd.DataFrame({"dlppi": 100 * np.log(raw.PPIACO).diff(), "dlip": 100 * np.log(raw.INDPRO).diff(),
                      "dlcpi": 100 * np.log(raw.CPIAUCSL).diff(), "ur": raw.UNRATE, "ffr": raw.FEDFUNDS}).dropna()
    t["dlppi"] = deseason(t["dlppi"])                      # PPIACO nao e dessazonalizada no FRED
    return t


d_full = transform(raw_all)
d_main = transform(raw_all.loc[:END_MAIN])
raw = raw_all.loc[:END_MAIN]
d = d_main[ORDER]
out = {"n_obs": int(len(d)), "start": str(d.index[0].date()), "end": str(d.index[-1].date()), "p": P_LAGS,
       "n_obs_full": int(len(d_full)), "end_full": str(d_full.index[-1].date())}


def season(index):
    """11 dummies mensais (janeiro e a categoria omitida)."""
    return pd.get_dummies(index.month, drop_first=True).astype(float).set_index(index)


def shade(a, lo=None, hi=None):
    for s, e in RECESSIONS:
        a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#bbbbbb", alpha=.45, lw=0)


# ------------------------------------------------------------------ figuras de dados
fig, ax = plt.subplots(3, 2, figsize=(10, 8))
for a, (s, t) in zip(ax.flat, FRED.items()):
    a.plot(raw_all[s], color=C1, lw=1.2); a.set_title(t, fontsize=9); shade(a)
    a.axvline(pd.Timestamp(END_MAIN), color=C2, lw=1, ls="--")
    if s == "FEDFUNDS":
        a.axvspan(pd.Timestamp("2008-12"), pd.Timestamp("2015-11"), facecolor="none", edgecolor=C2, hatch="///", lw=0, alpha=.5)
        a.text(pd.Timestamp("2009-01"), raw_all[s].max() * .72, "juro ≈ 0", color=C2, fontsize=8)
ax.flat[-1].axis("off")
ax.flat[-1].text(0, .6, "Faixas cinza: recessões do NBER.\nLinha tracejada: fim da amostra de\nestimação (dez/2007). Hachurado: período\nde juro próximo de zero (dez/2008–dez/2015).",
                 fontsize=9, va="center")
fig.tight_layout(); fig.savefig(f"{HERE}/img/series_originais.png"); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
for a, c in zip(ax, DIFF):
    a.plot(d_main[c], color=C1, lw=1); a.axhline(0, color="k", lw=.5); a.set_title(lab[c]); shade(a)
fig.tight_layout(); fig.savefig(f"{HERE}/img/series_transformadas.png"); plt.close(fig)

# correlacoes cruzadas: corr(ffr_t, x_{t+k})
ks = np.arange(-24, 25)
fig, ax = plt.subplots(1, 4, figsize=(13, 2.8), sharey=True)
for a, c in zip(ax, [n for n in ORDER if n != "ffr"]):
    cc = [d["ffr"].corr(d[c].shift(-k)) for k in ks]
    a.bar(ks, cc, color=C1, width=.8); a.axhline(0, color="k", lw=.6)
    a.axhline(1.96 / np.sqrt(len(d)), color=C2, lw=.8, ls=":"); a.axhline(-1.96 / np.sqrt(len(d)), color=C2, lw=.8, ls=":")
    a.set_title(f"corr(Fed Funds$_t$, {short[c]}$_{{t+k}}$)", fontsize=9); a.set_xlabel("k (meses)")
fig.tight_layout(); fig.savefig(f"{HERE}/img/correlacao_cruzada.png"); plt.close(fig)

# ------------------------------------------------------------------ raiz unitaria (especificacao reportada)
ur = {}
for c in ORDER:
    a = adfuller(d[c], regression="c", autolag="AIC"); k = kpss(d[c], regression="c", nlags="auto")
    ur[c] = {"adf_p": float(a[1]), "adf_lags": int(a[2]), "kpss_p": float(k[1]), "kpss_stat": float(k[0]), "spec": "c"}
lvl_main = {"dlppi": 100 * np.log(raw.PPIACO), "dlip": 100 * np.log(raw.INDPRO), "dlcpi": 100 * np.log(raw.CPIAUCSL)}
ur_lvl = {}
for c, s in lvl_main.items():
    s = s.loc[d.index]
    a = adfuller(s, regression="ct", autolag="AIC"); k = kpss(s, regression="ct", nlags="auto")
    ur_lvl[c] = {"adf_p": float(a[1]), "adf_lags": int(a[2]), "kpss_p": float(k[1])}
out["unit_root"], out["unit_root_levels"] = ur, ur_lvl

# ------------------------------------------------------------------ nucleo do VAR (MQO com dummies sazonais)


def fit_var(Y, D, p):
    T, n = Y.shape
    Xd = np.hstack([np.ones((T - p, 1)), D[p:]])
    X = np.hstack([Xd] + [Y[p - k:T - k] for k in range(1, p + 1)])
    Yt = Y[p:]
    B = np.linalg.lstsq(X, Yt, rcond=None)[0]
    U = Yt - X @ B
    k0 = Xd.shape[1]
    A = np.array([B[k0 + n * (k - 1):k0 + n * k].T for k in range(1, p + 1)])
    return {"A": A, "det": B[:k0], "U": U, "Sigma": U.T @ U / (T - p - X.shape[1]), "Xd": Xd, "T": T - p}


def companion(A):
    p, n, _ = A.shape
    C = np.zeros((n * p, n * p)); C[:n] = np.hstack(list(A)); C[n:, :-n] = np.eye(n * (p - 1))
    return C


def maxroot(A):
    return float(np.max(np.abs(np.linalg.eigvals(companion(A)))))


def simulate(A, detmat, Y0, U, rng):
    """Bootstrap 'wild' (Rademacher aplicado ao vetor de residuos: preserva heterocedasticidade e correlacao contemporanea)."""
    p, n, _ = A.shape
    e = U * rng.choice([-1.0, 1.0], size=(len(U), 1))
    Y = np.zeros((len(U) + p, n)); Y[:p] = Y0
    for t in range(p, len(Y)):
        Y[t] = detmat[t - p] + e[t - p] + sum(A[k] @ Y[t - 1 - k] for k in range(p))
    return Y


def shrink(A, bias):
    """Correcao de vies de Kilian (1998): A - delta*bias, com delta reduzido ate a estabilidade."""
    if maxroot(A) >= 1:
        return A, 0.0
    delta = 1.0
    while delta > 0:
        Ac = A - delta * bias
        if maxroot(Ac) < 1:
            return Ac, delta
        delta -= .01
    return A, 0.0


def phi(A, H):
    p, n, _ = A.shape
    Ph = np.zeros((H + 1, n, n)); Ph[0] = np.eye(n)
    for h in range(1, H + 1):
        Ph[h] = sum(Ph[h - k] @ A[k - 1] for k in range(1, min(h, p) + 1))
    return Ph


def irf_from(A, Sigma, cum_idx, H=H_IRF):
    P = np.linalg.cholesky(Sigma)
    I = phi(A, H) @ P
    I[:, cum_idx, :] = np.cumsum(I[:, cum_idx, :], axis=0)       # variaveis em diferenca: resposta acumulada (nivel)
    return I, np.diag(P).copy()


def run_var(dd, p, reps=N_BOOT, seed=SEED):
    """VAR(p) + FRI de Cholesky; vies corrigido (Kilian) e IC por wild bootstrap. dd: DataFrame na ordem do Cholesky."""
    rng = np.random.default_rng(seed)
    Y = dd.values; D = np.zeros((len(dd), 0)); n = Y.shape[1]
    cum_idx = [i for i, c in enumerate(dd.columns) if c in DIFF]
    f0 = fit_var(Y, D, p)
    # etapa 1: estimar o vies de MQO
    bias_s = []
    for _ in range(N_BOOT_BIAS):
        Yb = simulate(f0["A"], f0["Xd"] @ f0["det"], Y[:p], f0["U"], rng)
        bias_s.append(fit_var(Yb, D, p)["A"])
    bias = np.mean(bias_s, 0) - f0["A"]
    Ac, delta = shrink(f0["A"], bias)
    # intercepto/dummies recalculados dado A corrigido
    tgt = Y[p:] - sum(Y[p - k:len(Y) - k] @ Ac[k - 1].T for k in range(1, p + 1))
    detc = np.linalg.lstsq(f0["Xd"], tgt, rcond=None)[0]
    Uc = tgt - f0["Xd"] @ detc
    point, dP = irf_from(Ac, f0["Sigma"], cum_idx)
    point_ols, dP_ols = irf_from(f0["A"], f0["Sigma"], cum_idx)
    # etapa 2: bootstrap a partir do modelo corrigido
    R = np.zeros((reps, H_IRF + 1, n, n)); DP = np.zeros((reps, n)); n_shrunk = 0
    for r in range(reps):
        Yb = simulate(Ac, f0["Xd"] @ detc, Y[:p], Uc, rng)
        fb = fit_var(Yb, D, p)
        Abc, dl = shrink(fb["A"], bias)
        n_shrunk += dl < 1
        R[r], DP[r] = irf_from(Abc, fb["Sigma"], cum_idx)
    return {"names": list(dd.columns), "cum_idx": cum_idx, "f0": f0, "Ac": Ac, "delta": delta, "bias": bias,
            "point": point, "dP": dP, "point_ols": point_ols, "R": R, "DP": DP, "n_shrunk": int(n_shrunk),
            "maxroot": maxroot(Ac), "maxroot_ols": maxroot(f0["A"]), "T": f0["T"]}


def unit_resp(V, shock, lo=5, hi=95):
    """Resposta a um choque de tamanho unitario na propria variavel (impacto = 1): ponto e IC 90% (percentis)."""
    j = V["names"].index(shock)
    pt = V["point"][:, :, j] / V["dP"][j]
    B = V["R"][:, :, :, j] / V["DP"][:, j][:, None, None]
    return pt, np.percentile(B, lo, axis=0), np.percentile(B, hi, axis=0)


def fevd_from(V, Sigma=None, Ac=None):
    names = V["names"]; n = len(names)
    P = np.linalg.cholesky(V["f0"]["Sigma"]); Ph = phi(V["Ac"], H_IRF) @ P
    sq = Ph ** 2
    res = {}
    for hy in (12, 24, 48):
        num = sq[:hy].sum(0); res[hy] = 100 * num / num.sum(1, keepdims=True)
    return res


# ------------------------------------------------------------------ escolha de defasagens
Dm = np.zeros((len(d), 0)); TT, nvar = d.shape; PMAX = 12
ic = {}
for p in range(1, PMAX + 1):
    Y = d.values[PMAX - p:]; Dp = Dm[PMAX - p:]            # mesma amostra efetiva (ultimas T-PMAX obs) para todo p
    f = fit_var(Y, Dp, p); Tn = f["T"]
    ldet = np.linalg.slogdet(f["U"].T @ f["U"] / Tn)[1]; k = nvar * nvar * p + nvar
    pm = float(VAR(d).fit(p).test_whiteness(nlags=24, adjusted=True).pvalue) if p < 24 else np.nan
    ic[p] = {"aic": ldet + 2 * k / Tn, "bic": ldet + np.log(Tn) * k / Tn, "hqic": ldet + 2 * np.log(np.log(Tn)) * k / Tn,
             "pm": pm}
out["lag_table"] = {str(p): {k: float(v) for k, v in x.items()} for p, x in ic.items()}
out["lag_sel"] = {k: int(min(ic, key=lambda p: ic[p][k])) for k in ("aic", "bic", "hqic")}
out["pmax"] = PMAX

# ------------------------------------------------------------------ diagnostico do VAR escolhido
res = VAR(d).fit(P_LAGS)
out["lb"] = {c: {"p12": float(acorr_ljungbox(res.resid[c], lags=[12], return_df=True).lb_pvalue.iloc[0]),
                 "p24": float(acorr_ljungbox(res.resid[c], lags=[24], return_df=True).lb_pvalue.iloc[0])} for c in ORDER}
out["portmanteau_p"] = float(res.test_whiteness(nlags=24, adjusted=True).pvalue)
out["normality_p"] = float(res.test_normality().pvalue)
# sem dessazonalizar a PPI (para mostrar o efeito da sazonalidade)
d_ns = pd.DataFrame({"dlip": 100 * np.log(raw.INDPRO).diff(), "dlcpi": 100 * np.log(raw.CPIAUCSL).diff(),
                     "dlppi": 100 * np.log(raw.PPIACO).diff(), "ur": raw.UNRATE, "ffr": raw.FEDFUNDS}).dropna()
res_nd = VAR(d_ns).fit(P_LAGS)
out["portmanteau_p_sem_dessaz"] = float(res_nd.test_whiteness(nlags=24, adjusted=True).pvalue)
out["lb_ppi_sem_dessaz"] = float(acorr_ljungbox(res_nd.resid["dlppi"], lags=[12], return_df=True).lb_pvalue.iloc[0])
out["ppi_sazonal_amp"] = float(d_ns["dlppi"].groupby(d_ns.index.month).mean().max() - d_ns["dlppi"].groupby(d_ns.index.month).mean().min())
out["cpi_acf"] = {str(k): float(res.resid["dlcpi"].autocorr(k)) for k in (1, 2, 11, 12, 13)}
out["n_eff"] = int(res.nobs)

fig, ax = plt.subplots(5, 1, figsize=(8.5, 10))
for a, c in zip(ax, ORDER):
    plot_acf(res.resid[c], lags=36, ax=a, title=f"ACF dos resíduos – {lab[c]}", color=C1, vlines_kwargs={"colors": C1})
    a.set_ylim(-.3, .3)
fig.tight_layout(); fig.savefig(f"{HERE}/img/acf_residuos.png"); plt.close(fig)

fig, ax = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
for a, c in zip(ax, ORDER):
    a.plot(res.resid[c], color=C1, lw=.8); a.axhline(0, color="k", lw=.5); a.set_ylabel(short[c], fontsize=8); shade(a)
fig.suptitle(f"Resíduos do VAR({P_LAGS}) ao longo do tempo (faixas: recessões do NBER)", y=.995)
fig.tight_layout(); fig.savefig(f"{HERE}/img/residuos_tempo.png"); plt.close(fig)

# ------------------------------------------------------------------ VAR principal + robustez
print("estimando VAR principal..."); V = run_var(d, P_LAGS)
print("ordenacao alternativa...");   V_alt = run_var(d_main[ORDER_ALT], P_LAGS, seed=SEED + 1)
print("amostra completa...");        V_full = run_var(d_full[ORDER], P_LAGS, seed=SEED + 2)
print("VAR(6)...");                 V_p6 = run_var(d, 6, seed=SEED + 3)
names = V["names"]; hh = np.arange(H_IRF + 1)
out.update({"max_root": V["maxroot"], "max_root_ols": V["maxroot_ols"], "bias_delta": V["delta"],
            "n_shrunk": V["n_shrunk"], "n_boot": N_BOOT, "n_boot_bias": N_BOOT_BIAS, "stable": V["maxroot"] < 1,
            "chol_sd": {n: float(V["dP"][i]) for i, n in enumerate(names)}, "chol_sd_ffr": float(V["dP"][-1]),
            "max_root_full": V_full["maxroot"], "max_abs_diff_bias": float(np.max(np.abs(V["point"] - V["point_ols"])))})

ev = np.linalg.eigvals(companion(V["Ac"]))
fig, a = plt.subplots(figsize=(4.6, 4.6))
a.add_patch(plt.Circle((0, 0), 1, fill=False, color="k", lw=1)); a.scatter(ev.real, ev.imag, color=C1, s=22)
a.set_aspect("equal"); a.set_xlim(-1.1, 1.1); a.set_ylim(-1.1, 1.1); a.set_title("Autovalores da matriz companheira (30)")
fig.tight_layout(); fig.savefig(f"{HERE}/img/raizes.png"); plt.close(fig)

# ---- 7.a FRI 5x5 (choque de 1 desvio-padrao; escala comum por linha)
q = lambda V_: (np.percentile(V_["R"], 5, axis=0), np.percentile(V_["R"], 95, axis=0))
LO, HI = q(V)
fig, ax = plt.subplots(5, 5, figsize=(16, 12.5), sharex=True, sharey="row")
for i in range(5):
    for j in range(5):
        a = ax[i, j]
        a.fill_between(hh, LO[:, i, j], HI[:, i, j], color=C1, alpha=.2, lw=0)
        a.plot(hh, V["point"][:, i, j], color=C1, lw=1.5); a.axhline(0, color="k", lw=.6)
        a.tick_params(labelsize=8)
        if i == 0: a.set_title(f"Choque: {short[names[j]]}", fontsize=10, color=C2 if names[j] == "ffr" else "k")
        if j == 0: a.set_ylabel(short[names[i]] + ("\n(acum.)" if i in V["cum_idx"] else ""), fontsize=10)
for a in ax[-1]: a.set_xlabel("meses")
fig.suptitle("FRI ortogonalizada (Cholesky): choque de 1 desvio-padrão; IC 90% (wild bootstrap, vies corrigido)", y=.998)
fig.tight_layout(); fig.savefig(f"{HERE}/img/fri_completa.png"); plt.close(fig)

out["irf_matrix"] = {names[i]: {names[j]: {str(h): [float(V["point"][h, i, j]), float(LO[h, i, j]), float(HI[h, i, j])] for h in (0, 12, 24, 48)}
                                for j in range(5)} for i in range(5)}
# ---- choque monetario de 100 pb (1 p.p. no impacto), mesma escala da projecao local
pt_mp, lo_mp, hi_mp = unit_resp(V, "ffr")
fig, ax = plt.subplots(1, 5, figsize=(16, 3.2))
for i in range(5):
    a = ax[i]; a.fill_between(hh, lo_mp[:, i], hi_mp[:, i], color=C1, alpha=.2, lw=0)
    a.plot(hh, pt_mp[:, i], color=C1, lw=1.6, label="correção de viés"); a.axhline(0, color="k", lw=.6)
    a.plot(hh, V["point_ols"][:, i, 4] / V["dP"][4], color=C3, lw=1.1, ls="--", label="MQO")
    a.set_title(short[names[i]] + (" – acum." if i in V["cum_idx"] else "")); a.set_xlabel("meses")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Resposta a um choque monetário de +100 pb (1 p.p.) no Fed Funds; IC 90%", y=1.04)
fig.tight_layout(); fig.savefig(f"{HERE}/img/fri_choque_monetario.png", bbox_inches="tight"); plt.close(fig)
HS = (0, 6, 12, 24, 48)
out["irf_mp"] = {n: {"h": list(HS), "point": [float(pt_mp[h, i]) for h in HS], "lo": [float(lo_mp[h, i]) for h in HS],
                     "hi": [float(hi_mp[h, i]) for h in HS]} for i, n in enumerate(names)}
out["irf_peaks"] = {n: {"h": int(np.argmax(np.abs(pt_mp[:, i]))), "v": float(pt_mp[np.argmax(np.abs(pt_mp[:, i])), i])}
                    for i, n in enumerate(names)}
# primeiro horizonte com IC 90% excluindo zero (periodo e sinal)
sig = {}
for i, n in enumerate(names):
    ok = (lo_mp[:, i] > 0) | (hi_mp[:, i] < 0)
    sig[n] = [int(h) for h in hh[ok]]
out["irf_mp_signif_h"] = {n: {"first": (v[0] if v else None), "last": (v[-1] if v else None), "count": len(v)} for n, v in sig.items()}

# ---- robustez: choque monetario em 3 especificacoes
specs = [(V, "Principal: VAR(4), 1990–2007, ordem CEE", C1, "-"), (V_alt, "Ordem alternativa (commodities primeiro)", C2, "--"),
         (V_p6, "VAR(6) (mais defasagens)", "#8e44ad", "-."), (V_full, "Amostra completa 1990–2019", "#27ae60", ":")]
fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
rob = {}
for V_, lbl, col, ls in specs:
    p_, l_, h_ = unit_resp(V_, "ffr")
    for k, n in enumerate(names):
        i = V_["names"].index(n)
        ax[k].plot(hh, p_[:, i], color=col, ls=ls, lw=1.6, label=lbl)
        ax[k].fill_between(hh, l_[:, i], h_[:, i], color=col, alpha=.08, lw=0)
        rob.setdefault(lbl, {})[n] = {str(h): float(p_[h, i]) for h in (12, 24, 48)}
for k, n in enumerate(names):
    ax[k].axhline(0, color="k", lw=.6); ax[k].set_title(short[n] + (" – acum." if n in DIFF else "")); ax[k].set_xlabel("meses")
ax[0].legend(frameon=False, fontsize=7.5, loc="best")
fig.suptitle("Robustez: resposta a +100 pb no Fed Funds sob quatro especificações (IC 90%)", y=1.04)
fig.tight_layout(); fig.savefig(f"{HERE}/img/robustez.png", bbox_inches="tight"); plt.close(fig)
out["robustez"] = rob
out["irf_mp_full"] = {n: {"h": list(HS), "point": [float(unit_resp(V_full, "ffr")[0][h, V_full["names"].index(n)]) for h in HS]} for n in names}

# ---- FEVD (principal e ordem alternativa)
fe = fevd_from(V); fe_alt = fevd_from(V_alt)
out["fevd"] = {str(hy): {names[i]: {names[j]: float(fe[hy][i, j]) for j in range(5)} for i in range(5)} for hy in fe}
out["fevd_alt"] = {str(hy): {V_alt["names"][i]: {V_alt["names"][j]: float(fe_alt[hy][i, j]) for j in range(5)} for i in range(5)}
                   for hy in fe_alt}
Pm = np.linalg.cholesky(V["f0"]["Sigma"]); sqm = (phi(V["Ac"], H_IRF) @ Pm) ** 2
fig, ax = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
cols = {"dlip": "#27ae60", "dlcpi": "#e67e22", "dlppi": "#8e44ad", "ur": "#7f8c8d", "ffr": C2}
for i in range(5):
    cs = np.cumsum(sqm[:H_IRF, i, :], axis=0); sh = 100 * cs / cs.sum(1, keepdims=True)
    ax[i].stackplot(np.arange(1, H_IRF + 1), sh.T, colors=[cols[n] for n in names], labels=[short[n] for n in names], alpha=.85)
    ax[i].set_title(short[names[i]]); ax[i].set_xlabel("meses"); ax[i].set_xlim(1, H_IRF)
    for x in (12, 24, 48): ax[i].axvline(x, color="k", lw=.5, ls=":")
ax[0].set_ylabel("% da variância"); ax[-1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{HERE}/img/fevd.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ projecao local (Jorda, 2005)
lags = pd.concat([d.shift(k).add_suffix(f"_L{k}") for k in range(1, P_LAGS + 1)], axis=1)


def ydep(n, h):
    if n in DIFF:                                    # resposta acumulada do nivel do log: x_{t+h} - x_{t-1} = soma dos delta
        return sum(d[n].shift(-s) for s in range(h + 1))
    return d[n].shift(-h)


def lp(yname, shock, H=H_LP, cov="HC1"):
    """beta_h: efeito de +1 unidade de inovacao em `shock` (identificada recursivamente) sobre y_{t+h}.
    Controles: valores contemporaneos das variaveis ordenadas ANTES de `shock` e 6 defasagens de todas."""
    pos = ORDER.index(shock); contemp = [c for c in ORDER[:pos]]
    rows = []
    for h in range(H + 1):
        if h == 0 and yname in contemp:
            rows.append((np.nan,) * 4); continue
        X = pd.concat([d[[shock] + contemp], lags], axis=1); X = sm.add_constant(X)
        df = pd.concat([ydep(yname, h).rename("y"), X], axis=1).dropna()
        kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": h + 1}} if cov == "HAC" else {"cov_type": cov}
        r = sm.OLS(df["y"], df.drop(columns="y")).fit(**kw)
        b, s = r.params[shock], r.bse[shock]
        rows.append((b, s, b - 1.645 * s, b + 1.645 * s))
    return pd.DataFrame(rows, columns=["b", "se", "lo", "hi"])


# (i) equacao do Fed Funds (variavel mais endogena): resposta a inovacoes nas demais variaveis, com impacto contemporaneo
x = np.arange(H_LP + 1); LPT = {}; LPT_hac = {}; var_t = {}
fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), sharey=False)
for a, sk in zip(ax, [n for n in ORDER if n != "ffr"]):
    r = lp("ffr", sk); LPT[sk] = r; LPT_hac[sk] = lp("ffr", sk, cov="HAC")
    pv, lv, hv = unit_resp(V, sk); i = names.index("ffr")
    a.fill_between(x, r.lo, r.hi, color=C2, alpha=.2, lw=0); a.plot(x, r.b, color=C2, lw=1.6, marker="o", ms=3, label="LP (IC 90%, EHW)")
    a.fill_between(x, lv[:H_LP + 1, i], hv[:H_LP + 1, i], color=C1, alpha=.12, lw=0)
    a.plot(x, pv[:H_LP + 1, i], color=C1, lw=1.5, ls="--", label="VAR (IC 90%)")
    a.axhline(0, color="k", lw=.6); a.set_title(f"Inovação em {short[sk]} (+1 p.p.)"); a.set_xlabel("meses")
    var_t[sk] = [float(pv[h, i]) for h in range(H_LP + 1)]
ax[0].set_ylabel("resposta do Fed Funds (p.p.)"); ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Equação do Fed Funds por projeção local: resposta a inovações nas demais variáveis (impacto contemporâneo em h=0)", y=1.04)
fig.tight_layout(); fig.savefig(f"{HERE}/img/lp_taylor.png", bbox_inches="tight"); plt.close(fig)
out["lp_taylor"] = {sk: {"b": [float(v) for v in LPT[sk].b], "lo": [float(v) for v in LPT[sk].lo], "hi": [float(v) for v in LPT[sk].hi],
                         "var": var_t[sk], "se_ratio_hac_hc": float(np.nanmean(LPT_hac[sk].se / LPT[sk].se))} for sk in LPT}

# (ii) inovacao monetaria: resposta de todas as variaveis (h=0 nulo por identificacao, exceto no proprio juro)
LPM = {}
for n in names:
    if n == "ffr":
        r = lp(n, "ffr"); r.loc[0] = (1.0, 0.0, 1.0, 1.0)
    else:
        r = lp(n, "ffr"); r.loc[0] = (0.0, 0.0, 0.0, 0.0)
    LPM[n] = r
fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
for i, n in enumerate(names):
    r = LPM[n]
    ax[i].fill_between(x, r.lo, r.hi, color=C2, alpha=.2, lw=0); ax[i].plot(x, r.b, color=C2, lw=1.6, marker="o", ms=3)
    ax[i].axhline(0, color="k", lw=.6); ax[i].set_title(short[n] + (" – acum." if n in DIFF else "")); ax[i].set_xlabel("meses")
fig.suptitle("Projeção local: resposta a +1 p.p. de inovação monetária (IC 90%, erros robustos à heterocedasticidade)", y=1.04)
fig.tight_layout(); fig.savefig(f"{HERE}/img/lp_fri.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
for k, n in enumerate(names):
    i = names.index(n); r = LPM[n]
    ax[k].fill_between(x, r.lo, r.hi, color=C2, alpha=.15, lw=0); ax[k].plot(x, r.b, color=C2, lw=1.6, label="LP (IC 90%)")
    ax[k].fill_between(x, lo_mp[:H_LP + 1, i], hi_mp[:H_LP + 1, i], color=C1, alpha=.12, lw=0)
    ax[k].plot(x, pt_mp[:H_LP + 1, i], color=C1, lw=1.5, ls="--", label="VAR (IC 90%)")
    ax[k].axhline(0, color="k", lw=.6); ax[k].set_title(short[n] + (" – acum." if n in DIFF else "")); ax[k].set_xlabel("meses")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("LP × VAR: resposta a +100 pb de inovação monetária (mesma escala)", y=1.04)
fig.tight_layout(); fig.savefig(f"{HERE}/img/lp_vs_var.png", bbox_inches="tight"); plt.close(fig)
out["lp"] = {n: {str(h): {"b": float(LPM[n].b[h]), "lo": float(LPM[n].lo[h]), "hi": float(LPM[n].hi[h])} for h in (0, 1, 3, 6, 9, 12)}
             for n in names}
out["lp_full_series"] = {n: [float(v) for v in LPM[n].b] for n in names}
json.dump(out, open(f"{HERE}/resultados.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("ok")
