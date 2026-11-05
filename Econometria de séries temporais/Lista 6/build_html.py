# -*- coding: utf-8 -*-
"""Gera o relatorio HTML autocontido (figuras em base64, codigo completo embutido) a partir de resultados.json e img/."""
import base64, html, json, os, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{HERE}/resultados.json", encoding="utf-8"))
names = ["lip", "lcpi", "lppi", "ur", "ffr"]
lab = {"lip": "Produção industrial", "lcpi": "CPI", "lppi": "Commodities (PPI)", "ur": "Desemprego", "ffr": "Fed Funds"}
log_sfx = {"lip": " (ln×100)", "lcpi": " (ln×100)", "lppi": " (ln×100)", "ur": " (%)", "ffr": " (% a.a.)"}
f = lambda x, n=2: f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def pv(p):
    return "&lt; 0,001" if p < .001 else f(p, 3)


def kp(p):
    """p-valor do KPSS: a tabela so cobre 0,01 a 0,10 (valores truncados)."""
    return "≥ 0,10" if p >= .0999 else ("≤ 0,01" if p <= .0101 else f(p, 3))


def img(name, alt, cap, w=None):
    b = base64.b64encode(open(f"{HERE}/img/{name}", "rb").read()).decode()
    st = f' style="max-width:{w}px"' if w else ""
    return f'<figure><img src="data:image/png;base64,{b}" alt="{alt}"{st}><figcaption>{cap}</figcaption></figure>'


def ci(b, lo, hi, n=2):
    return f"{f(b, n)}<br><small>[{f(lo, n)}; {f(hi, n)}]</small>"


# ------------------------------------------------------------------ tabelas
sel = R["lag_sel"]
lag_rows = ""
for k, v in R["lag_table"].items():
    cls = ' class="hl"' if int(k) == R["p"] else ""
    lag_rows += f"<tr{cls}><td>{k}</td><td>{f(v['aic'], 3)}</td><td>{f(v['bic'], 3)}</td><td>{f(v['hqic'], 3)}</td><td>{pv(v['pm'])}</td></tr>"
ur_rows = "".join(
    f"<tr><td>{lab[n]}{log_sfx[n]}</td><td>{'constante e tendência' if R['unit_root'][n]['spec'] == 'ct' else 'constante'}</td><td>{pv(R['unit_root'][n]['adf_p'])}</td><td>{R['unit_root'][n]['adf_lags']}</td><td>{kp(R['unit_root'][n]['kpss_p'])}</td></tr>"
    for n in names)
lb_rows = "".join(f"<tr><td>{lab[n]}</td><td>{pv(R['lb'][n]['p12'])}</td><td>{pv(R['lb'][n]['p24'])}</td></tr>" for n in names)


def fevd_table(hy, key="fevd", order=names):
    head = "".join(f"<th>{lab[n]}</th>" for n in order)
    rows = "".join(f"<tr><th>{lab[i]}</th>" + "".join(f"<td>{f(R[key][str(hy)][i][j], 1)}</td>" for j in order) + "</tr>" for i in order)
    return f"<div class='tw'><table><thead><tr><th>Variável \\ choque</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>"


irf_rows = ""
for n in names:
    v = R["irf_mp"][n]
    irf_rows += f"<tr><th>{lab[n]}</th>" + "".join(ci(v['point'][k], v['lo'][k], v['hi'][k]).join(["<td>", "</td>"]) for k in range(5)) + "</tr>"
lp_rows = ""
for n in names:
    v = R["lp"][n]
    lp_rows += f"<tr><th>{lab[n]}</th>" + "".join(ci(v[h]['b'], v[h]['lo'], v[h]['hi']).join(["<td>", "</td>"]) for h in ("0", "1", "3", "6", "9", "12")) + "</tr>"
LT = R["lp_taylor"]
tay_rows = ""
for sk in ["lip", "lcpi", "lppi", "ur"]:
    v = LT[sk]
    tay_rows += f"<tr><th>Inovação em {lab[sk]}</th>" + "".join(ci(v['b'][h], v['lo'][h], v['hi'][h]).join(["<td>", "</td>"]) for h in (0, 1, 3, 6, 9, 12)) + "</tr>"
var_rows = "".join(f"<tr><th>Inovação em {lab[sk]} (VAR)</th>" + "".join(f"<td>{f(LT[sk]['var'][h])}</td>" for h in (0, 1, 3, 6, 9, 12)) + "</tr>" for sk in ["lip", "lcpi", "lppi", "ur"])
M = R["irf_matrix"]
sdv = R["chol_sd"]
sd_rows = "".join(f"<td>{f(sdv[n], 2)}</td>" for n in names)
FE = R["fevd"]; FA = R["fevd_alt"]
rb = R["robustez"]
rbk = list(rb.keys())
rb_rows = ""
for k in rbk:
    rb_rows += f"<tr><th>{html.escape(k)}</th>" + "".join(f"<td>{f(rb[k][n]['12'])}</td>" for n in names) + "</tr>"
IM = R["irf_mp"]; SG = R["irf_mp_signif_h"]; LPD = R["lp"]; PK = R["irf_peaks"]

code_main = html.escape(open(f"{HERE}/analise.py", encoding="utf-8").read())
code_build = html.escape(open(f"{HERE}/build_html.py", encoding="utf-8").read())

T = {}  # tokens preenchidos abaixo
page = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAR e Projeção Local – Lista 6</title>
<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']]}};</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml.min.js"></script>
<style>
:root{--bg:#fafaf8;--fg:#1c1f23;--mut:#5d6672;--card:#fff;--bd:#e2e4e8;--ac:#1f4e79;--ac2:#c0392b;--code:#f3f4f6}
@media(prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e8eaed;--mut:#9aa3ad;--card:#1d2126;--bd:#2c3238;--ac:#6fa8dc;--ac2:#e57368;--code:#23282e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:0 16px 64px}
header{padding:40px 0 16px;border-bottom:1px solid var(--bd)}h1{margin:0 0 6px;font-size:1.9rem}
.sub{color:var(--mut);font-size:.95rem}
h2{margin:48px 0 8px;font-size:1.4rem;color:var(--ac)}h3{margin:28px 0 6px;font-size:1.05rem}
.q{background:var(--card);border:1px solid var(--bd);border-left:4px solid var(--ac);border-radius:8px;padding:6px 16px;margin:14px 0}
.q .n{font-weight:700;color:var(--ac)}
figure{margin:16px 0}figure img{width:100%;height:auto;border:1px solid var(--bd);border-radius:8px;background:#fff;display:block;margin:auto}
figcaption{font-size:.85rem;color:var(--mut);margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:12px 0}th,td{border:1px solid var(--bd);padding:5px 8px;text-align:center}
thead th{background:var(--code)}tbody th{text-align:left;background:var(--code)}small{color:var(--mut)}tr.hl td{background:var(--code);font-weight:700}
.tw{overflow-x:auto}pre{background:var(--code);padding:12px;border-radius:8px;overflow:auto;font-size:.78rem;max-height:620px}
details{margin:10px 0}summary{cursor:pointer;color:var(--ac);font-weight:600}
.note{background:var(--code);border-radius:8px;padding:10px 14px;font-size:.92rem}
nav a{margin-right:14px;color:var(--ac);text-decoration:none;font-size:.9rem}
ol.ref{font-size:.88rem;color:var(--mut)}
</style></head><body><main>
<header><h1>VAR e Projeção Local: política monetária nos EUA</h1>
<div class="sub">UFMG – Econometria das Séries Temporais Aplicada à Macroeconomia e Finanças · Prof. Mauro Sayar Ferreira · Lista 6 (entrega em 05/11/2026)<br>
Dados do FRED, mensais, «start» a «end» («nobs» observações; «neff» efetivas no VAR) · Python (numpy, statsmodels)</div>
<nav><a href="#A">A. Dados</a><a href="#B">B. VAR</a><a href="#C">C. Estrutural</a><a href="#D">D. Projeção local</a><a href="#rob">Robustez</a><a href="#conc">Conclusão</a><a href="#codigo">Código</a></nav></header>

<div class="note" style="margin-top:18px"><b>Resumo.</b> Estima-se um VAR(«p») <b>com todas as variáveis em nível</b> (logaritmos de produção industrial, CPI e preços de commodities, taxa de desemprego e Fed Funds), com identificação recursiva (Cholesky), para a amostra de 1990 a 2007. Um choque monetário de 100 pontos-base eleva o juro a cerca de «pk_ffr» p.p. em seis meses e é seguido, no curto prazo, de aumento significativo da produção e das commodities e de queda do desemprego, sinais contrários à teoria. O CPI sobe, mas sem significância no VAR. A projeção local reproduz o padrão (com CPI significativo). Conclui-se que o principal problema é de identificação, e não de método. Todas as figuras, tabelas e o código completo constam deste documento.</div>

<h2 id="A">A. Fase preparatória</h2>
<div class="q"><p><span class="n">1) Motivação econômica.</span> O trabalho trata da transmissão da política monetária americana. A relação conjunta entre as variáveis decorre de três blocos: (i) a <b>regra de Taylor</b>, $i_t=r^*+\\pi_t+\\phi_\\pi(\\pi_t-\\pi^*)+\\phi_y\\tilde y_t$, que descreve a reação do Fed à inflação e à atividade; (ii) uma <b>curva IS</b>, pela qual a taxa de juros afeta a demanda e, com ela, a produção e o desemprego; e (iii) uma <b>curva de Phillips</b>, pela qual a demanda afeta os preços, sempre com defasagens. Daí a escolha de variáveis de atividade (produção industrial e desemprego), de preços (CPI) e do instrumento (Fed Funds, taxa efetiva). Inclui-se também o índice de preços ao produtor de todas as commodities (PPIACO), que antecipa pressões inflacionárias observadas pelo Fed e ajuda a mitigar o <i>price puzzle</i> (Sims, 1992). Cabe notar que o PPIACO é um índice amplo de preços ao produtor, e não um índice puro de commodities, como o CRB.</p>
<p><b>Amostra.</b> Estima-se o modelo de janeiro de 1990 a dezembro de 2007 («nobs» observações). A amostra exclui o período de juro próximo de zero (dezembro de 2008 a dezembro de 2015), no qual o instrumento fica praticamente imóvel e a política é conduzida por compra de ativos e <i>forward guidance</i>, o que torna a equação do juro mal especificada. A amostra completa (1990–2019, «nfull» observações) é usada apenas como análise de robustez. A taxa sombra de Wu e Xia (2016) seria uma alternativa para incluir o período de juro zero, mas não foi implementada.</p></div>

<div class="q"><p><span class="n">2) Séries originais.</span></p>
«fig_orig»</div>

<div class="q"><p><span class="n">3) Ajustes.</span> Produção industrial, CPI e preços de commodities foram logaritmizados ($100\,\ln x_t$); desemprego (%) e Fed Funds (% a.a.) permanecem como na fonte. <b>Todas as variáveis entram no VAR em nível</b>: as respostas das variáveis em logaritmo são, portanto, variações percentuais aproximadas em relação ao nível inicial, e as de desemprego e juro estão em pontos percentuais. As séries CPIAUCSL, INDPRO e UNRATE já são dessazonalizadas pela fonte. A série PPIACO <b>não é dessazonalizada</b> no FRED; por isso, retirou-se do seu logaritmo a componente sazonal determinística (dummies mensais, controlando por tendência linear) antes da estimação.</p>
<p>Os testes de raiz unitária (H0 do ADF: raiz unitária; H0 do KPSS: estacionariedade) foram aplicados nas séries em nível com a especificação indicada na tabela: constante e tendência para os logaritmos, que são tendenciais, e apenas constante para desemprego e juro. As defasagens do ADF foram escolhidas pelo critério de Akaike (a coluna indica o número selecionado) e o KPSS usa largura de banda automática. Como a tabela de valores críticos do KPSS cobre apenas de 0,01 a 0,10, os p-valores são truncados e aparecem como “≥ 0,10” ou “≤ 0,01”.</p>
<div class="tw"><table><thead><tr><th>Série em nível</th><th>Especificação</th><th>ADF (p-valor)</th><th>Defasagens ADF</th><th>KPSS (p-valor)</th></tr></thead><tbody>«ur_rows»</tbody></table></div>
<p class="note">Os logaritmos de produção, CPI e commodities não rejeitam raiz unitária no ADF e rejeitam estacionariedade no KPSS. O desemprego também não rejeita raiz unitária (ADF) e rejeita estacionariedade a 1% (KPSS); o Fed Funds tem evidência mista (ambos os testes rejeitam a 5%). Em suma, as cinco séries são muito persistentes, com raízes próximas de um. A estimação em nível é válida para fins de resposta ao impulso: Sims, Stock e Watson (1990) mostram que o VAR em nível estima consistentemente as funções de resposta mesmo na presença de raiz unitária ou de cointegração, sem exigir que se conheça a ordem de integração nem as relações de cointegração. O que se perde é a distribuição assintótica padrão de alguns testes, razão pela qual os intervalos de confiança são obtidos por <i>bootstrap</i>. A mesma especificação (nível) é aplicada a todas as variáveis, e eventuais relações de cointegração não são impostas.</p>
«fig_transf»
<p class="sub">Séries usadas no VAR (logaritmos ×100; PPI dessazonalizado). Desemprego e Fed Funds não sofrem transformação (ver figura do item 2).</p></div>

<h2 id="B">B. VAR em forma reduzida</h2>
<div class="q"><p><span class="n">4) Defasagens e restrições.</span> Estima-se, por mínimos quadrados ordinários equação a equação, $y_t=c+\\sum_{k=1}^{p}A_k y_{t-k}+u_t$, $u_t\\sim(0,\\Sigma_u)$, com $y_t=(ip_t,cpi_t,p^{com}_t,ur_t,ffr_t)'$ (logaritmos de produção, CPI e commodities; desemprego e juro em nível). Os critérios de informação foram calculados para $p=1,\\dots,«pmax»$ sobre a mesma amostra efetiva, e o teste portmanteau multivariado (24 defasagens) foi aplicado aos resíduos de cada modelo. O critério de Akaike escolhe $p=«aic»$, e os de Schwarz e Hannan-Quinn escolhem $p=«bic»$, porém o portmanteau rejeita a hipótese de resíduos brancos para $p=1,2$ e $3$ (a 5%). O critério adotado é o <b>menor $p$ cujos resíduos não rejeitam a hipótese de ruído branco</b>, o que leva a <b>$p=«p»$</b> (p-valor do portmanteau = «pm4»). O modelo com $p=6$ é estimado como análise de robustez (para ele, o portmanteau rejeita a 5%, p = «pm6»). A amostra efetiva do VAR(«p») tem «neff» observações (as «p» primeiras são perdidas como valores iniciais).</p>
<div class="tw"><table><thead><tr><th>p</th><th>AIC</th><th>BIC</th><th>HQ</th><th>Portmanteau(24)<br><small>p-valor</small></th></tr></thead><tbody>«lag_rows»</tbody></table></div>
<p>Não há restrições de exclusão na forma reduzida: todas as variáveis são domésticas e podem interagir entre si em qualquer defasagem. A parcimônia vem apenas da escolha de $p$.</p></div>

<div class="q"><p><span class="n">5) Qualidade do ajuste.</span> As figuras e testes a seguir avaliam se os resíduos estão branqueados.</p>
«fig_acf»
<div class="tw"><table><thead><tr><th>Equação</th><th>Ljung-Box, 12 defasagens (p)</th><th>Ljung-Box, 24 defasagens (p)</th></tr></thead><tbody>«lb_rows»</tbody></table></div>
<p><b>Resultados.</b> Os resíduos de produção, commodities, desemprego e Fed Funds são brancos a 5% (Ljung-Box e correlogramas). O portmanteau multivariado não rejeita a hipótese de resíduos brancos (p = «pm4»). <b>A equação do CPI constitui a exceção</b>: o teste de Ljung-Box rejeita (p «lbcpi12» com 12 defasagens e «lbcpi24» com 24), e o correlograma mostra autocorrelações em torno das defasagens 11 a 13 (valores de «acf11», «acf12» e «acf13» nas defasagens 11, 12 e 13), padrão característico de sazonalidade residual na série dessazonalizada do CPI. Essa limitação é assumida: a dessazonalização oficial do CPI é revista ao longo do tempo e deixa um resíduo sazonal que o VAR com «p» defasagens não absorve. Admite-se, portanto, que os intervalos referentes ao CPI podem ser ligeiramente imprecisos. Os resíduos rejeitam fortemente a normalidade (p ≈ 0, caudas pesadas), o que motiva o uso de <i>bootstrap</i> (e não de aproximações gaussianas) na construção dos intervalos de confiança. Os resíduos ao longo do tempo e as raízes da matriz companheira são apresentados a seguir: a maior raiz da matriz companheira é «maxroot_ols» em módulo, ligeiramente acima de 1, o que é esperado num VAR em nível com variáveis integradas (ver figura de raízes) e coerente com os testes de raiz unitária. As funções de resposta resultantes são persistentes e, no horizonte de 48 meses, permanecem limitadas.</p>
<div style="display:grid;grid-template-columns:1.6fr 1fr;gap:12px;align-items:center">«fig_res»«fig_raizes»</div>
«fig_cc»</div>

<h2 id="C">C. Análise estrutural</h2>
<div class="q"><p><span class="n">6) Restrições de identificação.</span> A identificação é recursiva (Cholesky): $\\Sigma_u=PP'$, com $P$ triangular inferior, $u_t=P\\varepsilon_t$ e $\\varepsilon_t\\sim(0,I)$. Como $\\Sigma_u$ tem $n(n+1)/2=15$ elementos distintos e $P$ tem $n^2=25$ elementos, são necessárias $25-15=10$ restrições, que são exatamente os dez zeros acima da diagonal de uma matriz triangular inferior $5\\times 5$: a identificação é <b>exata</b>. A ordem adotada segue Christiano, Eichenbaum e Evans (1999): produção industrial, CPI, commodities, desemprego e Fed Funds. Isso impõe que: (i) produção e preços ao consumidor, rígidos no curto prazo, não reagem dentro do mês a nenhuma das demais variáveis, e, por isso, são ordenados primeiro; (ii) o preço de commodities, flexível e negociado continuamente, reage dentro do mês à atividade e aos preços ao consumidor, razão pela qual vem depois deles, e o Fed observa seu comportamento corrente; (iii) o desemprego reage dentro do mês à produção e aos preços, mas não ao juro; (iv) o Fed Funds vem <b>por último</b>: a política reage contemporaneamente a todas as variáveis (o Fed observa os dados correntes), mas o choque monetário afeta as demais apenas com defasagem de um mês.</p>
<p>Uma propriedade relevante, demonstrada por Christiano, Eichenbaum e Evans (1999), é que, com o juro ordenado por último, a resposta ao <b>choque monetário</b> não depende da ordem das quatro primeiras variáveis. As respostas aos <b>demais</b> choques e a decomposição da variância, porém, dependem da ordem; isso é verificado na análise de robustez.</p></div>

<div class="q"><p><span class="n">7a) Funções de resposta ao impulso (FRI).</span> A figura apresenta a resposta de cada variável (linhas) a cada choque estrutural (colunas) de um desvio-padrão, com intervalos de confiança de 90%. As variáveis estão em nível: as respostas dos logaritmos são variações percentuais aproximadas, e as de desemprego e juro, pontos percentuais. Cada linha usa a mesma escala vertical. Os desvios-padrão dos choques (impacto sobre a própria variável) são, em % para os logaritmos e em p.p. para desemprego e juro:</p>
<div class="tw"><table><thead><tr><th></th>«sd_head»</tr></thead><tbody><tr><th>1 desvio-padrão</th>«sd_rows»</tr></tbody></table></div>
<p><b>Intervalos de confiança.</b> Utiliza-se o <i>bootstrap</i> <i>wild</i> (multiplicador de Rademacher aplicado ao vetor de resíduos, que preserva a heterocedasticidade e a correlação contemporânea), em linha com a rejeição da normalidade e a presença de caudas pesadas. Geram-se «nboot» séries a partir do modelo estimado, reestima-se o VAR e recalculam-se as funções de resposta; os intervalos de 90% são os percentis 5% e 95% das réplicas. O procedimento inclui a correção de viés de Kilian (1998), que se aplica apenas quando o modelo estimado é estável; como a maior raiz por MQO é «maxroot_ols» (≥ 1), a correção não é acionada e as estimativas pontuais são as de MQO. Esse é o caso esperado em VAR em nível com variáveis integradas, no qual o viés de pequena amostra é potencialmente relevante e os intervalos devem ser lidos com cautela, sobretudo nos horizontes longos.</p>
«fig_fri»</div>

<div class="q"><p><span class="n">7b) Interpretação.</span> A resposta ao <b>choque monetário</b> é apresentada para um aumento de <b>100 pontos-base</b> (1 p.p.) no Fed Funds no impacto, escala que permite a comparação direta com a projeção local do item 9. Choques de 1 desvio-padrão correspondem a apenas «sd_ffr_bp» pontos-base e produziriam números pouco legíveis.</p>
«fig_mp»
<div class="tw"><table><thead><tr><th>Resposta ao choque de +100 pb (% ou p.p.)</th><th>h=0</th><th>h=6</th><th>h=12</th><th>h=24</th><th>h=48</th></tr></thead><tbody>«irf_rows»</tbody></table></div>
<ul>
<li><b>Fed Funds:</b> o juro sobe 1 p.p. no impacto e atinge cerca de «pk_ffr» p.p. após «pkh_ffr» meses (significativo até o mês «last_ffr»), voltando a valores próximos de zero em cerca de dois anos. Um choque de 1 p.p. que se amplia para quase o dobro nos primeiros meses não é uma inovação exógena transitória: tem a feição de <b>início de um ciclo de aperto</b>, ou seja, de política sistemática.</li>
<li><b>Atividade:</b> a produção industrial <b>sobe</b> (até «pk_ip» % em cerca de seis meses; significativa entre os meses «f_ip» e «l_ip») e o desemprego <b>cai</b> (até «ur_abs» p.p.; significativo entre os meses «f_ur» e «l_ur»). Em ambos os casos o efeito se dissipa e perde significância depois, mas os sinais de curto prazo contrariam a previsão teórica de contração.</li>
<li><b>Preços:</b> as commodities <b>sobem</b> de forma significativa (até «com_pk» % em cerca de nove meses; significativo entre os meses «f_com» e «l_com»). O CPI tem resposta <b>positiva em todos os horizontes</b> («cpi12» % em 12 meses; intervalo de 90% de [«cpi12lo»; «cpi12hi»]), mas <b>nunca significativa</b> no VAR. Há, portanto, um sinal de <i>price puzzle</i> que o VAR em nível não consegue distinguir de zero; na projeção local ele é significativo (item 10).</li>
<li><b>Diagnóstico:</b> os sinais de atividade e de commodities (e o do CPI, ainda que impreciso) estão trocados simultaneamente, o que aponta para <b>erro de identificação</b>, e não para ausência de efeito monetário. A explicação usual é o <i>Fed information effect</i> (Romer e Romer, 2000; Nakamura e Steinsson, 2018; Jarociński e Karadi, 2020): o “choque” recursivo contém a reação do Fed a informação sobre a economia (previsões de inflação e atividade) que o VAR não observa, e o aperto é lido como sinal de economia forte. O fato de o juro se ampliar e permanecer elevado por cerca de um ano reforça a leitura de política sistemática, e não de inovação exógena. A inclusão das commodities não elimina o problema. Seria necessário recorrer a choques identificados por métodos narrativos ou de alta frequência. A amostra completa, que inclui o período de juro zero, <b>amplifica</b> o puzzle de atividade (a produção sobe «full_ip12» % em 12 meses), consistente com a má especificação da equação do juro naquele período.</li>
</ul>
<p><b>Demais choques</b> (ver matriz 5×5; os choques de 1 desvio-padrão, com identificação apenas recursiva, não têm todos nome econômico claro):</p>
<ul>
«other_shocks»
</ul></div>

<div class="q"><p><span class="n">8) Decomposição da variância do erro de previsão (FEVD).</span> Percentual da variância do erro de previsão de cada variável (linhas) atribuído a cada choque (colunas) em 12, 24 e 48 meses (1, 2 e 4 anos). A ordem é a mesma do item 6.</p>
<h3>1 ano</h3>«fe12»<h3>2 anos</h3>«fe24»<h3>4 anos</h3>«fe48»
«fig_fevd»
<p><b>Comentários.</b></p><ul>
<li><b>Choques próprios</b> dominam no curto prazo em produção («fe_ip_ip12»% em 1 ano), CPI («fe_cpi_cpi12»%) e desemprego («fe_ur_ur12»%), mas perdem peso no horizonte de 4 anos em produção («fe_ip_ip48»%) e CPI («fe_cpi_cpi48»%), pois as variáveis em nível acumulam efeitos de outros choques. O Fed Funds é explicado por seu próprio choque em «fe_ffr_ffr12»% em 1 ano e «fe_ffr_ffr48»% em 4 anos; no horizonte longo, passam a explicá-lo os choques de desemprego («fe_ffr_ur48»%) e de produção («fe_ffr_ip48»%), coerente com uma regra de Taylor.</li>
<li><b>Resultados dependentes da ordem.</b> Na ordem adotada, os choques de commodities respondem por apenas «fe_cpi_com12»% da variância do CPI em 1 ano, ao passo que os choques de CPI explicam «fe_com_cpi12»% da variância das commodities. Na ordem alternativa (commodities primeiro), a relação se inverte: «fa_cpi_com12»% da variância do CPI é atribuída a commodities. A correlação contemporânea entre os dois resíduos é alocada pela ordenação, e essas parcelas não devem ser lidas como causalidade econômica.</li>
<li><b>Desemprego:</b> os choques de produção explicam «fe_ur_ip12»% da variância em 1 ano e «fe_ur_ip48»% em 4 anos, resultado compatível com a lei de Okun, mas em parte mecânico: a produção precede o desemprego na ordem (em ambas as ordens testadas), de modo que a correlação contemporânea entre os dois resíduos é atribuída ao choque de produção.</li>
<li><b>Choque monetário:</b> explica pouco das demais variáveis (entre «mp_min»% e «mp_max»% em 4 anos): a maior parte da variação do juro é resposta sistemática, e não inovação.</li></ul></div>

<h2 id="D">D. Projeção local</h2>
<div class="q"><p><span class="n">9) Estimação e estratégia de identificação.</span> A <b>variável mais endógena</b> do sistema é o <b>Fed Funds</b>, que, por ser a última na ordem recursiva, é a única que sofre impacto contemporâneo de todas as demais. Estima-se, por projeção local (Jordà, 2005), a equação do Fed Funds para cada horizonte $h=0,1,\\dots,12$:
$$ffr_{t+h}=\\alpha_h+\\beta_h\\,x_t+\\gamma_h'\\,w_t+\\sum_{k=1}^{«p»}\\delta_{h,k}'\\,y_{t-k}+e^{(h)}_{t+h},$$
em que $x_t\\in\\{ip_t,cpi_t,p^{com}_t,ur_t\\}$ (em nível) é a variável cuja inovação constitui o choque e $w_t$ reúne os valores <b>contemporâneos</b> das variáveis ordenadas <b>antes</b> de $x_t$. Assim, $\\beta_0$ mede o <b>impacto contemporâneo</b> sobre o juro e é diferente de zero (a equação é uma regra de Taylor dinâmica estimada por projeção local), e $\\beta_h$ ($h=1,\\dots,12$) traça a trajetória subsequente.</p>
<p><b>Identificação.</b> Utiliza-se a mesma recursividade do VAR: ao controlar pelas defasagens de todas as variáveis e pelos valores contemporâneos das variáveis ordenadas antes de $x_t$, a variação remanescente de $x_t$ é a inovação (ortogonal ao conjunto de informação anterior na ordem de Cholesky). Os coeficientes têm, portanto, a mesma interpretação das FRI do VAR para um choque de 1 unidade (1% nas variáveis em logaritmo; 1 p.p. em desemprego) na variável $x_t$. Nesta ordem, a produção industrial é a primeira variável; logo, para o choque de produção, $w_t$ é vazio e $\\beta_0=P_{5,1}/P_{1,1}$.</p></div>
<div class="q"><p><span class="n">10) Resultados.</span> Os intervalos de confiança de 90% são $\\hat\\beta_h\\pm1{,}645\\,\\widehat{se}$, com erros-padrão <b>robustos à heterocedasticidade</b> (Eicker-Huber-White, HC1), como pede o enunciado. A inclusão de defasagens suficientes de controle torna esses erros-padrão adequados, com melhor cobertura que o estimador de Newey-West em amostras finitas (Montiel Olea e Plagborg-Møller, 2021). Como verificação, os erros de Newey-West resultaram entre «hac_min»% e «hac_max»% dos erros EHW.</p>
«fig_taylor»
<div class="tw"><table><thead><tr><th>Resposta do Fed Funds (p.p.)</th><th>h=0</th><th>h=1</th><th>h=3</th><th>h=6</th><th>h=9</th><th>h=12</th></tr></thead><tbody>«tay_rows»</tbody></table></div>
<p><b>Leitura.</b> O impacto contemporâneo é diferente de zero e economicamente pequeno: uma inovação de 1% na produção eleva o juro em «tay_ip0» p.p. no mesmo mês, e o efeito cresce até «tay_ip12» p.p. em 12 meses (intervalo [«tay_ip12lo»; «tay_ip12hi»]). Um aumento inesperado de 1 p.p. no desemprego reduz o juro de forma persistente («tay_ur12» p.p. em 12 meses), em linha com a regra de Taylor, ao passo que as inovações de CPI e de commodities não têm efeito significativo sobre o juro após o impacto. As estimativas por projeção local e por VAR (linha tracejada) são próximas, com intervalos da LP mais largos, como esperado de um estimador menos restrito.</p></div>
<div class="q"><p><span class="n">Complemento: resposta a uma inovação monetária.</span> Para comparação com o VAR, estima-se também o choque monetário por projeção local, com $ffr_t$ como regressor e controle pelos valores contemporâneos de todas as demais variáveis (todas ordenadas antes dele) e «p» defasagens. Por construção, o efeito contemporâneo nas demais variáveis é nulo, e sobre o próprio juro, igual a 1.</p>
«fig_lp»
<div class="tw"><table><thead><tr><th>Resposta (p.p.)</th><th>h=0</th><th>h=1</th><th>h=3</th><th>h=6</th><th>h=9</th><th>h=12</th></tr></thead><tbody>«lp_rows»</tbody></table></div>
«fig_lpvar»
<p><b>Leitura.</b> Uma inovação de 1 p.p. no Fed Funds chega a cerca de 1,5 p.p. nos primeiros meses e permanece acima do valor inicial por cerca de um ano: «lp_ffr12» p.p. após 12 meses (VAR: «ffr12»), padrão de política sistemática. O CPI sobe «lp_cpi12» % em 12 meses (intervalo [«lp_cpi12lo»; «lp_cpi12hi»]) e as commodities sobem «lp_com12» % (intervalo [«lp_com12lo»; «lp_com12hi»]): puzzles significativos, coerentes com o efeito informacional. O desemprego cai de forma significativa no primeiro mês («lp_ur1» p.p., intervalo [«lp_ur1lo»; «lp_ur1hi»]), e a produção sobe com defasagem e perde significância em seguida. A LP reproduz, portanto, os puzzles do VAR e torna o do CPI estatisticamente significativo.</p></div>

<h2 id="rob">Robustez</h2>
<div class="q"><p>A resposta ao choque monetário (+100 pb) é comparada em quatro especificações: a principal; a ordem alternativa, com commodities antes de produção e CPI; o VAR(6); e a amostra completa 1990–2019 (com o período de juro zero).</p>
«fig_rob»
<div class="tw"><table><thead><tr><th>Resposta em h=12</th>«rob_head»</tr></thead><tbody>«rb_rows»</tbody></table></div>
<p>A ordem alternativa reproduz exatamente a mesma resposta ao choque monetário, como previsto pela invariância à ordem das variáveis anteriores ao juro (Christiano, Eichenbaum e Evans, 1999). O VAR(6) mantém o mesmo padrão qualitativo, com resposta menor da produção («rob6_ip» contra «rob4_ip» % em 12 meses). A amostra completa gera respostas maiores de produção e commodities (produção: «robf_ip» %), o que reforça a decisão de excluir o período de juro zero da estimação principal. A decomposição da variância, ao contrário, <b>não</b> é robusta à ordem (item 8). Para referência, a FEVD na ordem alternativa a 1 ano:</p>
«fa12»</div>

<h2 id="conc">Conclusão</h2>
<div class="q"><p>Com identificação recursiva e variáveis em nível, o VAR e a projeção local produzem respostas qualitativamente iguais: um “choque” monetário de 100 pb eleva o juro a cerca de 1,5 a 1,8 p.p. nos primeiros meses e é acompanhado, no curto prazo, de aumento de produção e de commodities, queda do desemprego e alta de preços (esta significativa na projeção local e não significativa no VAR). O fato de dois estimadores distintos chegarem a respostas semelhantes indica que <b>o problema é de identificação, e não de método</b> (Plagborg-Møller e Wolf, 2021), isto é, o choque recursivo mistura inovação exógena com a reação sistemática do Fed a informação não observada pelo modelo. A equação do juro por projeção local confirma a reação sistemática do Fed (juro sobe com a produção, cai com o desemprego). Para obter efeitos monetários com sinais corretos seria necessário usar choques narrativos ou de alta frequência. Limitações assumidas: resíduo sazonal na equação do CPI, caudas pesadas, raiz maior que 1 no VAR em nível (sem correção de viés) e exclusão do período de juro zero.</p></div>

<h2>Referências</h2>
<ol class="ref">
<li>Christiano, L., Eichenbaum, M., Evans, C. (1999). Monetary Policy Shocks: What Have We Learned and to What End? <i>Handbook of Macroeconomics</i>.</li>
<li>Jarociński, M., Karadi, P. (2020). Deconstructing Monetary Policy Surprises. <i>AEJ: Macroeconomics</i>.</li>
<li>Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections. <i>AER</i>.</li>
<li>Kilian, L. (1998). Small-Sample Confidence Intervals for Impulse Response Functions. <i>REStat</i>.</li>
<li>Montiel Olea, J. L., Plagborg-Møller, M. (2021). Local Projection Inference Is Simpler and More Robust Than You Think. <i>Econometrica</i>.</li>
<li>Nakamura, E., Steinsson, J. (2018). High-Frequency Identification of Monetary Non-Neutrality: The Information Effect. <i>QJE</i>.</li>
<li>Plagborg-Møller, M., Wolf, C. (2021). Local Projections and VARs Estimate the Same Impulse Responses. <i>Econometrica</i>.</li>
<li>Romer, C., Romer, D. (2000). Federal Reserve Information and the Behavior of Interest Rates. <i>AER</i>.</li>
<li>Sims, C. (1992). Interpreting the macroeconomic time series facts: the effects of monetary policy. <i>European Economic Review</i>.</li>
<li>Sims, C., Stock, J., Watson, M. (1990). Inference in Linear Time Series Models with Some Unit Roots. <i>Econometrica</i>.</li>
<li>Wu, J. C., Xia, F. D. (2016). Measuring the Macroeconomic Impact of Monetary Policy at the Zero Lower Bound. <i>JMCB</i>.</li>
</ol>

<h2 id="codigo">Código e reprodução</h2>
<p>O código abaixo é o <b>código completo</b> usado em todas as estimações, figuras e tabelas (<code>analise.py</code>); o segundo bloco é o gerador deste documento (<code>build_html.py</code>). Os dados vêm do FRED (PPIACO, INDPRO, CPIAUCSL, UNRATE, FEDFUNDS) e ficam na pasta <code>data/</code>. Reprodução: <code>python analise.py &amp;&amp; python build_html.py</code> (sementes fixas).</p>
<details><summary>analise.py – dados, VAR, bootstrap, FRI, FEVD, projeção local e figuras</summary><pre>«code_main»</pre></details>
<details><summary>build_html.py – geração deste relatório</summary><pre>«code_build»</pre></details>
<p class="sub">Dados: FRED, Federal Reserve Bank of St. Louis.</p>
</main></body></html>"""


def pct(x):
    return f(x, 0)


def sig_range(n):
    s = SG[n]; return s["first"], s["last"]


fe = lambda hy, a, b: pct(FE[str(hy)][a][b])
fa = lambda hy, a, b: pct(FA[str(hy)][a][b])
ir = lambda n, k: IM[n]["point"][k]
mp48 = [FE["48"][n]["ffr"] for n in ["lip", "lcpi", "lppi", "ur"]]


def other_shocks():
    """Interpretacao dos choques nao monetarios a partir de irf_matrix (h=12; choque de 1 d.p.)."""
    def r(resp, sh, h=12):
        p, lo, hi = M[resp][sh][str(h)]
        return f"{f(p)} [{f(lo)}; {f(hi)}]"
    items = [
        f"<li><b>Choque de produção industrial</b> (demanda ou oferta agregada, sem distinção nesta identificação): a produção sobe de forma persistente ({r('lip','lip')} em 12 meses), o desemprego cai ({r('ur','lip')}) e o Fed Funds sobe ({r('ffr','lip')}), reação endógena da política à atividade, em linha com a regra de Taylor.</li>",
        f"<li><b>Choque de CPI</b> (choque de preço doméstico, por exemplo de <i>markup</i> ou de custos, com efeito permanente sobre o nível de preços): o CPI sobe no impacto e permanece alto ({r('lcpi','lcpi')}); o desemprego e o Fed Funds não reagem de forma significativa ({r('ur','lcpi')} e {r('ffr','lcpi')}).</li>",
        f"<li><b>Choque de commodities</b> (aumento do preço de insumos, condicional à atividade e ao CPI): o repasse ao CPI é pequeno ({r('lcpi','lppi')}); a produção cai ({r('lip','lppi')}), compatível com choque de oferta adverso; o Fed Funds não reage ({r('ffr','lppi')}).</li>",
        f"<li><b>Choque de desemprego</b> (variação da taxa de desemprego não explicada por produção, preços e commodities; não tem interpretação estrutural limpa nesta identificação, podendo refletir choque de oferta de trabalho, de pareamento ou de ciclo de emprego): o Fed Funds cai ({r('ffr','ur')} em 12 meses), como numa regra de Taylor com peso no desemprego; a resposta das commodities não é significativa ({r('lppi','ur')}).</li>",
        "<li>Os choques de <b>CPI e de desemprego</b> não têm identificação estrutural nítida; a rotulagem acima é apenas indicativa. Somente o choque monetário (último na ordem) tem interpretação estrutural direta.</li>"]
    return "\n".join(items)


T.update({
    "start": R["start"][:7], "end": R["end"][:7], "nobs": R["n_obs"], "neff": R["n_eff"], "nfull": R["n_obs_full"],
    "p": R["p"], "pmax": R["pmax"], "aic": sel["aic"], "pm4": f(R["lag_table"][str(R["p"])]["pm"], 3),
    "lbcpi12": pv(R["lb"]["lcpi"]["p12"]), "lbcpi24": pv(R["lb"]["lcpi"]["p24"]),
    "acf11": f(R["cpi_acf"]["11"]), "acf12": f(R["cpi_acf"]["12"]), "acf13": f(R["cpi_acf"]["13"]),
    "maxroot_ols": f(R["max_root_ols"], 3), "maxroot_c": f(R["max_root"], 4), "nbb": R["n_boot_bias"], "nboot": R["n_boot"], "delta": f(R["bias_delta"], 2),
    "pm6": f(R["lag_table"]["6"]["pm"], 3), "bic": sel["bic"], "fe_ur_ur12": fe(12, "ur", "ur"), "fe_ip_ip48": fe(48, "lip", "lip"), "fe_cpi_cpi48": fe(48, "lcpi", "lcpi"), "fe_ffr_ur48": fe(48, "ffr", "ur"), "com_pk": f(abs(PK["lppi"]["v"]), 1), "f_com": SG["lppi"]["first"], "l_com": SG["lppi"]["last"], "rob4_ip": f(rb[rbk[0]]["lip"]["12"]), "rob6_ip": f(rb[rbk[2]]["lip"]["12"]), "robf_ip": f(rb[rbk[3]]["lip"]["12"]), "sd_ffr_bp": f(100 * sdv["ffr"], 0),
    "pk_ffr": f(PK["ffr"]["v"]), "pkh_ffr": PK["ffr"]["h"], "last_ffr": SG["ffr"]["last"], "ffr12": f(ir("ffr", 2)),
    "pk_ip": f(PK["lip"]["v"]), "f_ip": SG["lip"]["first"], "l_ip": SG["lip"]["last"],
    "ur_abs": f(abs(ir("ur", 2)), 1), "f_ur": SG["ur"]["first"], "l_ur": SG["ur"]["last"],
    "cpi12": f(ir("lcpi", 2)), "cpi12lo": f(IM["lcpi"]["lo"][2]), "cpi12hi": f(IM["lcpi"]["hi"][2]),
    "f_cpi": SG["lcpi"]["first"], "l_cpi": SG["lcpi"]["last"],
    "com12": f(ir("lppi", 2)), "com12lo": f(IM["lppi"]["lo"][2]), "com12hi": f(IM["lppi"]["hi"][2]),
    "full_ip12": f(R["irf_mp_full"]["lip"]["point"][2]),
    "fe_ip_ip12": fe(12, "lip", "lip"), "fe_cpi_cpi12": fe(12, "lcpi", "lcpi"), "fe_ffr_ffr12": fe(12, "ffr", "ffr"),
    "fe_ffr_ffr48": fe(48, "ffr", "ffr"), "fe_ffr_ip48": fe(48, "ffr", "lip"),
    "fe_cpi_com12": fe(12, "lcpi", "lppi"), "fe_com_cpi12": fe(12, "lppi", "lcpi"), "fa_cpi_com12": fa(12, "lcpi", "lppi"),
    "fe_ur_ip12": fe(12, "ur", "lip"), "fe_ur_ip48": fe(48, "ur", "lip"),
    "mp_min": pct(min(mp48)), "mp_max": pct(max(mp48)),
    "hac_min": pct(100 * min(v["se_ratio_hac_hc"] for v in LT.values())), "hac_max": pct(100 * max(v["se_ratio_hac_hc"] for v in LT.values())),
    "tay_ip0": f(LT["lip"]["b"][0]), "tay_ip12": f(LT["lip"]["b"][12]), "tay_ip12lo": f(LT["lip"]["lo"][12]), "tay_ip12hi": f(LT["lip"]["hi"][12]),
    "tay_ur12": f(LT["ur"]["b"][12]), "tay_ppi12": f(LT["lppi"]["b"][12]),
    "lp_ffr12": f(LPD["ffr"]["12"]["b"]), "lp_cpi12": f(LPD["lcpi"]["12"]["b"]), "lp_cpi12lo": f(LPD["lcpi"]["12"]["lo"]), "lp_cpi12hi": f(LPD["lcpi"]["12"]["hi"]),
    "lp_com12": f(LPD["lppi"]["12"]["b"]), "lp_com12lo": f(LPD["lppi"]["12"]["lo"]), "lp_com12hi": f(LPD["lppi"]["12"]["hi"]),
    "lp_ur1": f(LPD["ur"]["1"]["b"]), "lp_ur1lo": f(LPD["ur"]["1"]["lo"]), "lp_ur1hi": f(LPD["ur"]["1"]["hi"]),
    "sd_head": "".join(f"<th>{lab[n]}</th>" for n in names), "sd_rows": sd_rows,
    "rob_head": "".join(f"<th>{lab[n]}</th>" for n in names), "rb_rows": rb_rows,
    "ur_rows": ur_rows, "lag_rows": lag_rows, "lb_rows": lb_rows, "irf_rows": irf_rows,
    "lp_rows": lp_rows, "tay_rows": tay_rows + var_rows,
    "fe12": fevd_table(12), "fe24": fevd_table(24), "fe48": fevd_table(48),
    "fa12": fevd_table(12, "fevd_alt", ["lppi", "lip", "lcpi", "ur", "ffr"]),
    "other_shocks": other_shocks(),
    "fig_orig": img("series_originais.png", "Séries originais", "Séries em nível, como baixadas do FRED. Faixas cinza: recessões do NBER; linha tracejada: fim da amostra de estimação; hachurado: período de juro próximo de zero.", 900),
    "fig_transf": img("series_transformadas.png", "Séries transformadas", "Séries em logaritmo (×100) usadas no VAR; PPI dessazonalizado."),
    "fig_acf": img("acf_residuos.png", "ACF dos resíduos", "Correlogramas dos resíduos de cada equação (bandas de 95%).", 760),
    "fig_res": img("residuos_tempo.png", "Resíduos", "Resíduos ao longo do tempo."),
    "fig_raizes": img("raizes.png", "Raízes", "Autovalores da matriz companheira (círculo: raio unitário)."),
    "fig_cc": img("correlacao_cruzada.png", "Correlação cruzada", "Correlações cruzadas entre o Fed Funds e as demais variáveis (linhas pontilhadas: banda de 95% sob ruído branco)."),
    "fig_fri": img("fri_completa.png", "FRI 5x5", "Matriz 5×5: linhas = variável que responde; colunas = choque. IC 90% por <i>wild bootstrap</i>."),
    "fig_mp": img("fri_choque_monetario.png", "FRI choque monetário", "Resposta a +100 pb de choque monetário (IC 90%)."),
    "fig_fevd": img("fevd.png", "FEVD", "FEVD por horizonte (linhas pontilhadas: 12, 24 e 48 meses)."),
    "fig_taylor": img("lp_taylor.png", "LP do Fed Funds", "Equação do Fed Funds por projeção local, h = 0 a 12: resposta a inovações de +1 unidade em cada variável (IC 90%, erros robustos à heterocedasticidade), comparada ao VAR."),
    "fig_lp": img("lp_fri.png", "FRI por LP", "Projeção local: resposta a +1 p.p. de inovação monetária (IC 90%)."),
    "fig_lpvar": img("lp_vs_var.png", "LP versus VAR", "Comparação LP × VAR, mesma escala (+100 pb)."),
    "fig_rob": img("robustez.png", "Robustez", "Resposta ao choque monetário (+100 pb) em quatro especificações."),
})
for k, v in T.items():
    page = page.replace(f"«{k}»", str(v))
import re
left = [t for t in re.findall(r"«[a-z0-9_]+»", page) if t not in ("«code_main»", "«code_build»")]
assert not left, left
page = page.replace("«code_main»", code_main).replace("«code_build»", code_build)
out = f"{HERE}/index.html"
open(out, "w", encoding="utf-8").write(page)
os.makedirs(f"{HERE}/entrega/codigo", exist_ok=True)
shutil.copy(out, f"{HERE}/entrega/relatorio.html")
for fn in ("analise.py", "build_html.py"):
    shutil.copy(f"{HERE}/{fn}", f"{HERE}/entrega/codigo/{fn}")
shutil.copytree(f"{HERE}/data", f"{HERE}/entrega/codigo/data", dirs_exist_ok=True)
print("ok", len(page))
