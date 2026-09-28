"""
Funcoes auxiliares para o Exercicio 4 (Econometria das Series Temporais Aplicada
a Macroeconomia e Financas). Reunidas aqui para serem reaproveitadas tanto na
secao empirica (Questao 1, IPCA) quanto na secao de simulacao (Questoes 2-4),
evitando duplicar a logica de previsao pseudo-fora-da-amostra e dos testes de
igualdade de capacidade preditiva (DMW e CW) nas duas partes do notebook.

Convencao de sinal usada em todo o arquivo: a perda diferencial e sempre definida
como d_t = e_bench^2 - e_model^2, onde "bench" e o modelo de referencia (em geral
o passeio aleatorio) e "model" e o modelo concorrente. Sob H1 (o modelo concorrente
preve melhor que o benchmark, ou seja sigma^2_bench > sigma^2_model), espera-se
d_t > 0 em media, de modo que a estatistica de teste e testada no quantil superior
(alternative="greater").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA


# ---------------------------------------------------------------------------
# Testes de igualdade de capacidade preditiva: Diebold-Mariano-West e Clark-West
# ---------------------------------------------------------------------------

def _hac_ttest_on_constant(series, h, alternative):
    """Regride `series` em uma constante e devolve a estatistica-t HAC (Newey-West)
    do intercepto, junto com o p-valor unilateral/bilateral e valores criticos
    convencionais da distribuicao t-Student com (n-1) graus de liberdade."""
    d = np.asarray(series, dtype=float)
    n = len(d)
    maxlags = max(h - 1, 0)
    x = np.ones((n, 1))
    res = sm.OLS(d, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True})
    stat = float(res.tvalues[0])
    dof = n - 1
    if alternative == "greater":
        pvalue = float(stats.t.sf(stat, dof))
    elif alternative == "less":
        pvalue = float(stats.t.cdf(stat, dof))
    else:
        pvalue = float(2 * stats.t.sf(abs(stat), dof))
    crit = {
        0.10: float(stats.t.ppf(0.90, dof)),
        0.05: float(stats.t.ppf(0.95, dof)),
        0.01: float(stats.t.ppf(0.99, dof)),
    }
    return {"stat": stat, "mean": float(d.mean()), "pvalue": pvalue, "dof": dof, "crit_values": crit}


def dmw_test(e_bench, e_model, h: int = 1, alternative: str = "greater"):
    """Estatistica de Diebold-Mariano-West para comparar dois vetores de erro de
    previsao de mesmo tamanho. `e_bench` e o erro do modelo de referencia (ex.:
    passeio aleatorio) e `e_model` o erro do modelo concorrente.
    """
    e_bench = np.asarray(e_bench, dtype=float)
    e_model = np.asarray(e_model, dtype=float)
    d = e_bench ** 2 - e_model ** 2
    out = _hac_ttest_on_constant(d, h=h, alternative=alternative)
    out["msfe_bench"] = float(np.mean(e_bench ** 2))
    out["msfe_model"] = float(np.mean(e_model ** 2))
    return out


def clark_west_test(y, f_bench, f_model, h: int = 1, alternative: str = "greater"):
    """Estatistica de Clark-West (2007) para comparar um modelo benchmark (em
    geral aninhado no modelo maior, como o passeio aleatorio) contra um modelo
    concorrente, corrigindo o vies gerado pelo ruido de estimacao de parametros
    extras. `f_bench` e `f_model` sao as previsoes pontuais (nao os erros).
    """
    y = np.asarray(y, dtype=float)
    f_bench = np.asarray(f_bench, dtype=float)
    f_model = np.asarray(f_model, dtype=float)
    e_bench = y - f_bench
    e_model = y - f_model
    adj = (f_bench - f_model) ** 2
    f_t = e_bench ** 2 - e_model ** 2 + adj
    out = _hac_ttest_on_constant(f_t, h=h, alternative=alternative)
    out["msfe_bench"] = float(np.mean(e_bench ** 2))
    out["msfe_model"] = float(np.mean(e_model ** 2))
    return out


# ---------------------------------------------------------------------------
# Previsao pseudo-fora-da-amostra: caso geral ARIMA(p,d,q) (Questao 1, IPCA)
# ---------------------------------------------------------------------------

def random_walk_forecast(series, R):
    """Previsao de passeio aleatorio sem drift: y_hat[t+1] = y[t]. Devolve as
    P = len(series) - R previsoes correspondentes aos alvos y[R], ..., y[N-1]."""
    y = np.asarray(series, dtype=float)
    return y[R - 1: len(y) - 1]


def arima_pseudo_forecast(series, order, R, scheme="recursive", window_size=None, trend="c"):
    """Previsoes 1-passo-a-frente pseudo-fora-da-amostra de um ARIMA(p,d,q),
    reestimado a cada novo ponto incorporado a janela.

    scheme="recursive": janela expansiva, sempre a partir da 1a observacao.
    scheme="rolling": janela movel de tamanho fixo `window_size` (padrao: R).
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    if window_size is None:
        window_size = R
    forecasts = np.full(n - R, np.nan)
    for k, i in enumerate(range(R, n)):
        if scheme == "recursive":
            train = y[:i]
        elif scheme == "rolling":
            train = y[i - window_size: i]
        else:
            raise ValueError("scheme deve ser 'recursive' ou 'rolling'")
        fit = ARIMA(train, order=order, trend=trend).fit()
        forecasts[k] = fit.forecast(steps=1)[0]
    return forecasts


# ---------------------------------------------------------------------------
# Simulacao AR(1) com inovacoes t-Student e previsao recursiva (Questoes 2-4)
# ---------------------------------------------------------------------------

def simulate_ar1_student(n=100, phi0=0.2, phi1=0.5, df=5, seed=None):
    """Simula yt = phi0 + phi1*y(t-1) + eps_t, eps_t ~ iid t-Student(df).
    O valor inicial e a media incondicional acrescida de uma inovacao, como
    pede o enunciado."""
    rng = np.random.default_rng(seed)
    eps = stats.t.rvs(df=df, size=n, random_state=rng)
    y = np.empty(n)
    mu = phi0 / (1 - phi1)
    y[0] = mu + eps[0]
    for t in range(1, n):
        y[t] = phi0 + phi1 * y[t - 1] + eps[t]
    return y, eps


def recursive_forecasts_ar1(y, R):
    """Previsao recursiva (janela expansiva) 1-passo-a-frente de um AR(1) com
    intercepto reestimado por OLS a cada nova observacao incorporada, e do
    passeio aleatorio, para os alvos de indice R, ..., len(y)-1 (0-indexado)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    P = n - R
    f_ar = np.full(P, np.nan)
    f_rw = np.full(P, np.nan)
    coefs = np.full((P, 2), np.nan)
    for k, i in enumerate(range(R, n)):
        y_train = y[:i]
        x = sm.add_constant(y_train[:-1])
        yy = y_train[1:]
        beta = np.linalg.lstsq(x, yy, rcond=None)[0]
        coefs[k] = beta
        f_ar[k] = beta[0] + beta[1] * y[i - 1]
        f_rw[k] = y[i - 1]
    return f_ar, f_rw, coefs


def true_model_forecast(y, R, phi0=0.2, phi1=0.5):
    """Previsao 1-passo-a-frente usando os coeficientes verdadeiros do processo
    gerador de dados: y_hat[t+1] = phi0 + phi1*y[t]."""
    y = np.asarray(y, dtype=float)
    return phi0 + phi1 * y[R - 1: len(y) - 1]


# ---------------------------------------------------------------------------
# Bootstrap semi-parametrico sob H0: sigma^2_rw = sigma^2_ar (Questao 4)
# ---------------------------------------------------------------------------

def semiparametric_bootstrap_rw_vs_ar(y, R, n_boot=1000, h=1, seed=None):
    """Bootstrap semi-parametrico sob H0 de que o passeio aleatorio e o
    verdadeiro processo gerador (sigma^2_rw = sigma^2_ar). Reamostra com
    reposicao as inovacoes de primeira diferenca da serie observada, reconstroi
    trajetorias sob H0 e repete a previsao recursiva (AR(1) vs RW) em cada
    replica, produzindo as distribuicoes bootstrap das estatisticas DMW e CW.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    n = len(y)
    e_rw = np.diff(y)  # inovacoes sob a nula de passeio aleatorio

    dmw_boot = np.empty(n_boot)
    cw_boot = np.empty(n_boot)
    for b in range(n_boot):
        e_star = rng.choice(e_rw, size=len(e_rw), replace=True)
        y_boot = np.empty(n)
        y_boot[0] = y[0]
        y_boot[1:] = y[0] + np.cumsum(e_star)

        f_ar_b, f_rw_b, _ = recursive_forecasts_ar1(y_boot, R)
        y_target_b = y_boot[R:]

        e_rw_b = y_target_b - f_rw_b
        e_ar_b = y_target_b - f_ar_b
        dmw_boot[b] = dmw_test(e_rw_b, e_ar_b, h=h)["stat"]
        cw_boot[b] = clark_west_test(y_target_b, f_rw_b, f_ar_b, h=h)["stat"]

    return dmw_boot, cw_boot


def bootstrap_summary(observed_stat, boot_dist):
    """Valores criticos empiricos (10/5/1%), p-valor (cauda superior) e demais
    estatisticas descritivas da distribuicao bootstrap de um teste unilateral
    (H1: estatistica grande e positiva)."""
    boot_dist = np.asarray(boot_dist, dtype=float)
    crit = {
        0.10: float(np.quantile(boot_dist, 0.90)),
        0.05: float(np.quantile(boot_dist, 0.95)),
        0.01: float(np.quantile(boot_dist, 0.99)),
    }
    pvalue = float(np.mean(boot_dist >= observed_stat))
    return {
        "observed": float(observed_stat),
        "crit_values": crit,
        "pvalue": pvalue,
        "boot_mean": float(boot_dist.mean()),
        "boot_std": float(boot_dist.std(ddof=1)),
    }
