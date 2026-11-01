"""Gera index.html a partir de resultados.json e das figuras em img/."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{HERE}/resultados.json", encoding="utf-8"))
names = ["dlppi", "dlip", "dlcpi", "ur", "ffr"]
lab = {"dlppi": "Commodities", "dlip": "Prod. industrial", "dlcpi": "CPI", "ur": "Desemprego", "ffr": "Fed Funds"}
f = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")

lag_rows = "".join(
    f"<tr><td>{k}</td><td>{f(v['aic'],3)}</td><td>{f(v['bic'],3)}</td><td>{f(v['hqic'],3)}</td></tr>"
    for k, v in R["lag_table"].items() if int(k) in (1, 2, 3, 4, 5, 6, 7, 8, 12))
ur_rows = "".join(f"<tr><td>{lab[n]}</td><td>{f(R['unit_root'][n]['adf_p'],3)}</td><td>{f(R['unit_root'][n]['kpss_p'],3)}</td></tr>" for n in names)
lb_rows = "".join(f"<tr><td>{lab[n]}</td><td>{f(R['lb'][n]['p12'],3)}</td><td>{f(R['lb'][n]['p24'],3)}</td></tr>" for n in names)


def fevd_table(hy):
    head = "".join(f"<th>{lab[n]}</th>" for n in names)
    rows = "".join(f"<tr><th>{lab[i]}</th>" + "".join(f"<td>{f(R['fevd'][str(hy)][i][j],1)}</td>" for j in names) + "</tr>" for i in names)
    return f"<table><thead><tr><th>Variável \\ choque</th>{head}</tr></thead><tbody>{rows}</tbody></table>"


irf_rows = ""
for n in names:
    v = R["irf_mp"][n]
    irf_rows += f"<tr><th>{lab[n]}</th>" + "".join(
        f"<td>{f(v['point'][k])}<br><small>[{f(v['lo'][k])}; {f(v['hi'][k])}]</small></td>" for k in range(5)) + "</tr>"
lp_rows = ""
for n in names:
    v = R["lp"][n]
    lp_rows += f"<tr><th>{lab[n]}</th>" + "".join(
        f"<td>{f(v[h]['b'])}<br><small>[{f(v[h]['lo'])}; {f(v[h]['hi'])}]</small></td>" for h in ("0", "1", "3", "6", "9", "12")) + "</tr>"

CODE_VAR = """lags = pd.concat([d.shift(k).add_suffix(f"_L{k}") for k in range(1, 7)], axis=1)
res = VAR(d).fit(6)                       # VAR(6), constante
irf = res.irf(48)                         # FRI ortogonalizada (Cholesky)
fe  = res.fevd(48).decomp                 # FEVD: (variavel, horizonte, choque)
# IC 90%: bootstrap de residuos (1000 replicas), percentis 5% e 95%"""
CODE_LP = """def lp(yname, H):
    for h in range(H + 1):
        y = <y_{t+h}>                     # nivel acumulado p/ variaveis em diferenca
        X = [ffr_t, dlppi_t, dlip_t, dlcpi_t, ur_t, 6 lags de todas, const]
        r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        beta_h, se_h = r.params["ffr"], r.bse["ffr"]      # IC 90%: beta +- 1,645 se"""

html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAR e Projeção Local – Lista 6</title>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']]}}}};</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml.min.js"></script>
<style>
:root{{--bg:#fafaf8;--fg:#1c1f23;--mut:#5d6672;--card:#fff;--bd:#e2e4e8;--ac:#1f4e79;--ac2:#c0392b;--code:#f3f4f6}}
@media(prefers-color-scheme:dark){{:root{{--bg:#14171a;--fg:#e8eaed;--mut:#9aa3ad;--card:#1d2126;--bd:#2c3238;--ac:#6fa8dc;--ac2:#e57368;--code:#23282e}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:0 16px 64px}}
header{{padding:40px 0 16px;border-bottom:1px solid var(--bd)}}h1{{margin:0 0 6px;font-size:1.9rem}}
.sub{{color:var(--mut);font-size:.95rem}}
h2{{margin:48px 0 8px;font-size:1.4rem;color:var(--ac)}}h3{{margin:28px 0 6px;font-size:1.05rem}}
.q{{background:var(--card);border:1px solid var(--bd);border-left:4px solid var(--ac);border-radius:8px;padding:6px 16px;margin:14px 0}}
.q>.n{{font-weight:700;color:var(--ac)}}
figure{{margin:16px 0}}figure img{{width:100%;height:auto;border:1px solid var(--bd);border-radius:8px;background:#fff}}
figcaption{{font-size:.85rem;color:var(--mut);margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;margin:12px 0}}th,td{{border:1px solid var(--bd);padding:5px 8px;text-align:center}}
thead th{{background:var(--code)}}tbody th{{text-align:left;background:var(--code)}}small{{color:var(--mut)}}
.tw{{overflow-x:auto}}pre{{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto;font-size:.82rem}}
details{{margin:10px 0}}summary{{cursor:pointer;color:var(--ac);font-weight:600}}
.note{{background:var(--code);border-radius:8px;padding:10px 14px;font-size:.92rem}}
nav a{{margin-right:14px;color:var(--ac);text-decoration:none;font-size:.9rem}}
</style></head><body><main>
<header><h1>VAR e Projeção Local: política monetária nos EUA</h1>
<div class="sub">UFMG – Econometria das Séries Temporais Aplicada à Macroeconomia e Finanças · Prof. Mauro Sayar Ferreira · Lista 6 (entrega 05/11/2026)<br>
Dados FRED, mensais, {R['start'][:7]} a {R['end'][:7]} ({R['n_obs']} observações) · Python (statsmodels)</div>
<nav><a href="#A">A. Dados</a><a href="#B">B. VAR</a><a href="#C">C. Estrutural</a><a href="#D">D. Projeção local</a><a href="#codigo">Código</a></nav></header>

<h2 id="A">A. Fase preparatória</h2>
<div class="q"><p><span class="n">1) Motivação.</span> Estudo o canal de transmissão da política monetária americana com cinco variáveis mensais: preço de <b>commodities</b> (PPI, que sinaliza inflação futura e ajuda a evitar o <i>price puzzle</i>), <b>produção industrial</b> (atividade), <b>CPI</b> (inflação), <b>taxa de desemprego</b> e <b>Fed Funds</b> (instrumento). A relação conjunta vem da regra de Taylor, $i_t=r^*+\\pi_t+\\phi_\\pi(\\pi_t-\\pi^*)+\\phi_y\\tilde y_t$: o Fed reage a inflação e atividade, e juros afetam atividade, desemprego e preços com defasagem. A amostra 1990–2019 evita a restrição de juro zero (2009–15) e a pandemia, que quebrariam a linearidade do modelo.</p></div>
<div class="q"><p><span class="n">2) Séries originais.</span></p>
<figure><img src="img/series_originais.png" alt="Séries originais"><figcaption>Séries em nível, como baixadas do FRED (PPIACO, INDPRO, CPIAUCSL, UNRATE, FEDFUNDS).</figcaption></figure></div>
<div class="q"><p><span class="n">3) Ajustes.</span> Commodities, produção industrial e CPI foram logaritmizadas e diferenciadas ($100\\,\\Delta\\ln x_t$, variação % mensal), pois são claramente tendenciais. Desemprego e Fed Funds ficam em nível: são limitadas e persistentes, e diferenciá-las descartaria informação sobre longo prazo. As séries do FRED usadas como CPIAUCSL e INDPRO já são dessazonalizadas.</p>
<div class="tw"><table><thead><tr><th>Série</th><th>ADF (p-valor)<br><small>H0: raiz unitária</small></th><th>KPSS (p-valor)<br><small>H0: estacionária</small></th></tr></thead><tbody>{ur_rows}</tbody></table></div>
<p class="note">O ADF rejeita raiz unitária para quase todas, exceto desemprego (p={f(R['unit_root']['ur']['adf_p'],2)}); o KPSS não rejeita estacionariedade para desemprego e rejeita para Fed Funds e CPI (alta persistência). Mantenho níveis para desemprego e juros e noto que as raízes do VAR ficam próximas de 1 (maior módulo {f(R['max_root'],3)}), o que é típico e aceitável para FRI com bandas por bootstrap.</p>
<figure><img src="img/series_transformadas.png" alt="Séries transformadas"><figcaption>Séries usadas no VAR.</figcaption></figure></div>

<h2 id="B">B. VAR em forma reduzida</h2>
<div class="q"><p><span class="n">4) Defasagens e restrições.</span> Estimo $y_t=c+\\sum_{{k=1}}^{{p}}A_k y_{{t-k}}+u_t$, $u_t\\sim(0,\\Sigma_u)$, com $y_t=(\\Delta p^{{com}},\\Delta ip,\\Delta cpi,ur,ffr)'$. Escolhi $p$ por critérios de informação até 12 defasagens (AIC e FPE → {R['lag_sel']['aic']}; BIC e HQ → {R['lag_sel']['bic']}) combinados à exigência de resíduos brancos: com $p=2$ restam autocorrelações; com $p=6$ os testes de Ljung-Box não rejeitam em nenhuma equação. Fiquei com <b>$p={R['p']}$</b>.</p>
<div class="tw"><table><thead><tr><th>p</th><th>AIC</th><th>BIC</th><th>HQ</th></tr></thead><tbody>{lag_rows}</tbody></table></div>
<p>Não impus restrições na forma reduzida: todas as variáveis são domésticas (EUA), então todos os choques podem afetar todas as variáveis; a parcimônia vem apenas da escolha de $p$. O modelo é estável (maior raiz da companion = {f(R['max_root'],3)} &lt; 1).</p></div>
<div class="q"><p><span class="n">5) Qualidade do ajuste.</span></p>
<figure><img src="img/acf_residuos.png" alt="ACF dos resíduos"><figcaption>Correlogramas dos resíduos de cada equação: praticamente todas as autocorrelações estão dentro da banda de 95%.</figcaption></figure>
<div class="tw"><table><thead><tr><th>Equação</th><th>Ljung-Box, 12 lags (p)</th><th>Ljung-Box, 24 lags (p)</th></tr></thead><tbody>{lb_rows}</tbody></table></div>
<p class="note"><b>Ressalvas.</b> Os resíduos de cada equação são brancos a 5% (o CPI fica no limite com 24 lags, p={f(R['lb']['dlcpi']['p24'],3)}), mas o teste portmanteau multivariado rejeita (p={f(R['portmanteau_p'],4)}) e a normalidade é fortemente rejeitada (caudas pesadas, crises). Por isso as bandas vêm de <i>bootstrap</i> de resíduos e não de fórmulas assintóticas gaussianas.</p></div>

<h2 id="C">C. Análise estrutural</h2>
<div class="q"><p><span class="n">6) Restrições.</span> Identificação recursiva (Cholesky): $\\Sigma_u=PP'$ com $P$ triangular inferior e $u_t=P\\varepsilon_t$, $\\varepsilon_t\\sim(0,I)$. A ordem é $(\\Delta p^{{com}},\\Delta ip,\\Delta cpi,ur,ffr)$. Isso impõe que: (i) commodities não reagem contemporaneamente a nada (preços de mercado, fortemente exógenos); (ii) atividade e preços ao consumidor reagem apenas a commodities dentro do mês (preços e produção rígidos no curto prazo); (iii) o Fed Funds, <b>por último</b>, reage a todos dentro do mês (o Fed observa dados correntes) mas o choque monetário afeta as demais variáveis só com defasagem de um mês. Como todas as variáveis são americanas, não há zeros adicionais na forma reduzida.</p></div>
<div class="q"><p><span class="n">7a) FRI.</span> Resposta de cada variável a cada choque estrutural de 1 desvio-padrão, com IC 90% por bootstrap ({R['ic_method']}). Para as variáveis em diferença mostro a resposta <b>acumulada</b> (nível do log, em pontos percentuais).</p>
<figure><img src="img/fri_completa.png" alt="FRI completa"><figcaption>Matriz 5×5: linhas = variável que responde; colunas = choque.</figcaption></figure>
<figure><img src="img/fri_choque_monetario.png" alt="FRI choque monetário"><figcaption>Foco no choque monetário (+{f(R['chol_sd_ffr'],2)} p.p. no Fed Funds no impacto).</figcaption></figure>
<div class="tw"><table><thead><tr><th>Resposta ao choque monetário</th><th>h=0</th><th>h=6</th><th>h=12</th><th>h=24</th><th>h=48</th></tr></thead><tbody>{irf_rows}</tbody></table></div></div>
<div class="q"><p><span class="n">7b) Interpretação.</span></p><ul>
<li><b>Choque monetário (Fed Funds):</b> o juro sobe {f(R['chol_sd_ffr'],2)} p.p. no impacto e chega a cerca de {f(R['irf_peaks']['ffr']['v'],2)} p.p. após ~{R['irf_peaks']['ffr']['h']} meses (persistência típica da regra de juros com suavização), voltando lentamente.</li>
<li><b>Desemprego:</b> cai ligeiramente (até {f(R['irf_peaks']['ur']['v'],3)} p.p. em ~{R['irf_peaks']['ur']['h']} meses, significativo a 90%) – sinal oposto ao previsto pela teoria; só em horizontes longos a resposta se aproxima de zero/positiva, sem significância. <b>Produção industrial</b> acumulada sobe e fica positiva. <b>CPI</b> sobe de forma lenta (<i>price puzzle</i> brando).</li>
<li><b>Diagnóstico:</b> o choque recursivo mensal capta em grande parte antecipação do Fed a informação que o modelo não observa (expectativas de inflação/atividade): o Fed sobe juros porque prevê aquecimento, e o VAR atribui isso ao "choque". Incluir commodities atenuou o puzzle mas não o eliminou. Resultado típico de identificação puramente recursiva; seria preciso choques narrativos (Romer-Romer) ou de alta frequência para limpar a identificação.</li>
<li><b>Outros choques:</b> choques de commodities elevam o CPI de forma persistente (repasse de custos), sem reação relevante do Fed Funds; choques de produção industrial reduzem o desemprego e elevam o Fed Funds (reação endógena da política à atividade).</li></ul></div>
<div class="q"><p><span class="n">8) FEVD.</span> Percentual da variância do erro de previsão de cada variável (linhas) atribuído a cada choque (colunas), em 12, 24 e 48 meses (1, 2 e 4 anos).</p>
<h3>1 ano</h3><div class="tw">{fevd_table(12)}</div><h3>2 anos</h3><div class="tw">{fevd_table(24)}</div><h3>4 anos</h3><div class="tw">{fevd_table(48)}</div>
<figure><img src="img/fevd.png" alt="FEVD"><figcaption>FEVD por horizonte (linhas pontilhadas: 12, 24 e 48 meses).</figcaption></figure>
<p><b>Comentários.</b> (i) Commodities, produção e CPI são dominados por choques próprios no curto prazo; o CPI deve cerca de {f(R['fevd']['12']['dlcpi']['dlppi'],0)}% da variância aos choques de commodities. (ii) O <b>desemprego</b> é explicado majoritariamente por choques de produção industrial ({f(R['fevd']['12']['ur']['dlip'],0)}% em 1 ano → {f(R['fevd']['48']['ur']['dlip'],0)}% em 4 anos), compatível com a lei de Okun. (iii) O <b>choque monetário</b> explica pouco da variação das demais variáveis (nunca passa de ~5%, ex.: {f(R['fevd']['48']['dlcpi']['ffr'],1)}% do CPI em 4 anos): a maior parte da variação do juro é sistemática (resposta endógena), não choque. (iv) O Fed Funds é explicado por seu próprio choque (de {f(R['fevd']['12']['ffr']['ffr'],0)}% a {f(R['fevd']['48']['ffr']['ffr'],0)}%) e, com o horizonte, pelos choques de produção ({f(R['fevd']['12']['ffr']['dlip'],0)}% → {f(R['fevd']['48']['ffr']['dlip'],0)}%), coerente com a regra de Taylor.</p></div>

<h2 id="D">D. Projeção local</h2>
<div class="q"><p><span class="n">9) Estimação e identificação.</span> A variável mais endógena do sistema é o <b>Fed Funds</b> (última na ordem, reage contemporaneamente a todas). Para cada horizonte $h=0,\\dots,12$ estimo (Jordà, 2005)
$$y_{{i,t+h}}-y_{{i,t-1}}\\;(\\text{{ou }} y_{{i,t+h}}\\text{{ para }} ur,ffr)=\\alpha_h+\\beta_h\\,ffr_t+\\gamma_h'\\,w_t+\\sum_{{k=1}}^{{6}}\\delta_{{h,k}}'\\,y_{{t-k}}+e_{{t+h}}^{{(h)}},\\quad w_t=(\\Delta p^{{com}}_t,\\Delta ip_t,\\Delta cpi_t,ur_t)',$$
e $\\beta_h$ é a FRI a uma inovação de +1 p.p. no Fed Funds. <b>Estratégia de identificação:</b> a mesma recursividade do VAR: controlar pelos valores <i>contemporâneos</i> das variáveis ordenadas antes do juro e por 6 defasagens de todas as variáveis faz $ffr_t$ ser, condicionalmente, a inovação monetária (ortogonal ao conjunto de informação do Fed no mês). Por construção, o efeito contemporâneo ($h=0$) nas demais variáveis é zero e no próprio juro é 1. As variáveis em diferença entram como variação acumulada do log entre $t-1$ e $t+h$, comparável à FRI acumulada.</p></div>
<div class="q"><p><span class="n">10) Resultado.</span> IC 90% = $\\hat\\beta_h\\pm1{{,}}645\\,\\widehat{{se}}$ com erros <b>HAC de Newey-West</b> ($h+1$ defasagens), robustos a heterocedasticidade e à autocorrelação MA($h$) induzida pelos resíduos sobrepostos.</p>
<figure><img src="img/lp_fri.png" alt="FRI por LP"><figcaption>FRI por Projeção Local, h=0..12 meses, +1 p.p. de inovação no Fed Funds.</figcaption></figure>
<div class="tw"><table><thead><tr><th>Resposta (p.p.)</th><th>h=0</th><th>h=1</th><th>h=3</th><th>h=6</th><th>h=9</th><th>h=12</th></tr></thead><tbody>{lp_rows}</tbody></table></div>
<figure><img src="img/lp_vs_var.png" alt="LP vs VAR"><figcaption>Comparação com o VAR (LP reescalonada para 1 d.p. = {f(R['sd_ffr'],2)} p.p.). As estimativas pontuais são muito próximas; o intervalo da LP é bem mais largo (menos restrita, menos eficiente).</figcaption></figure>
<p><b>Leitura.</b> O Fed Funds permanece elevado ~3 p.p. um ano depois de uma inovação de 1 p.p. (persistência). O desemprego cai de forma significativa nos primeiros meses (−0,16 a −0,28 p.p.) e a produção industrial acumula alta; o CPI primeiro cai (sem significância) e depois sobe (+0,97 p.p. em 12 meses, significativo). Ou seja, a LP reproduz os mesmos <i>puzzles</i> do VAR: sinal de que o problema é de <b>identificação</b>, não de método. A LP tem a vantagem de ser robusta a má especificação da dinâmica e à ordem do VAR, a custo de maior variância.</p></div>

<h2 id="codigo">Código e reprodução</h2>
<details open><summary>VAR, FRI e FEVD</summary><pre>{CODE_VAR}</pre></details>
<details><summary>Projeção local</summary><pre>{CODE_LP}</pre></details>
<p>Código completo: <code>analise.py</code> (baixa os dados do FRED, estima tudo, salva figuras e <code>resultados.json</code>) e <code>build_html.py</code> (gera esta página). Para reproduzir: <code>python analise.py &amp;&amp; python build_html.py</code>.</p>
<p class="sub">Dados: FRED, Federal Reserve Bank of St. Louis. Séries: PPIACO, INDPRO, CPIAUCSL, UNRATE, FEDFUNDS.</p>
</main></body></html>"""
open(f"{HERE}/index.html", "w", encoding="utf-8").write(html)
print("index.html ok", len(html))
