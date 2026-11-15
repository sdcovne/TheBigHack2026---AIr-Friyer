"""
Dunkelflaute Early Warning System — app Streamlit (v3, IT/EN).

Catena:  ① VEDERE (generatore AI + scenari di previsione) → ② MISURARE (gemello digitale)
         → ③ DECIDERE (azioni con scadenza, comunità) → ④ VERIFICARE (eventi reali 2024, inverni mai visti)
File nella stessa cartella: motore.py, testi.py, generatore_params.json, verifica_2024.csv, il CSV della simulazione.
Avvio:  streamlit run app.py
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import motore as M
from testi import TX, make_t, GIORNI

QUI = Path(__file__).parent
PARAMS_PATH = str(QUI / "generatore_params.json")
st.set_page_config(page_title="Dunkelflaute Early Warning", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# LINGUA (bandierina in alto a destra)
# ==============================================================================
if "lang" not in st.session_state:
    st.session_state["lang"] = "it"
_, col_lang = st.columns([10, 1.6])
with col_lang:
    lang = st.radio("Lingua / Language", ["it", "en"], key="lang", horizontal=True, label_visibility="collapsed",
                    format_func=lambda v: "🇮🇹 IT" if v == "it" else "🇬🇧 EN")
t = make_t(lang)
EN = lang == "en"

# ==============================================================================
# DESIGN SYSTEM — palette energetica, tema chiaro
# ==============================================================================
P = {
    "primario": "#0B5CAD", "testo": "#1B2A41", "testo_soft": "#5B6B7F", "bordo": "#DCE3EA",
    "sfondo": "#F6F8FB", "card": "#FFFFFF",
    "solare": "#F2A900", "eolico": "#1B9AAA", "accumulo": "#7B61C9", "fossile": "#8A94A6",
    "import": "#4C8BD6", "idro": "#2F6690", "riduzione": "#F4B183", "taglio": "#E07A1F",
    "non_serviti": "#D62839", "domanda_tot": "#1B2A41", "domanda_nec": "#A4161A", "evento": "#5B6C8F",
    "ok": "#2E9E5B", "warn": "#E07A1F", "crit": "#D62839", "storico": "#8A94A6", "sintetico": "#0B5CAD",
}
FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
:root {{ --primario:{P['primario']}; --testo:{P['testo']}; --testo-soft:{P['testo_soft']}; --bordo:{P['bordo']};
        --card:{P['card']}; --ok:{P['ok']}; --warn:{P['warn']}; --crit:{P['crit']}; }}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{ font-family:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif !important; }}
.stApp {{ background:{P['sfondo']}; color:var(--testo); }}
.block-container {{ padding-top:1.0rem; padding-bottom:3rem; max-width:1900px; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility:hidden; height:0; }}
[data-testid="stHeader"] {{ background:transparent; }}
div[role="radiogroup"][aria-label="Lingua / Language"] {{ justify-content:flex-end; }}

.hero {{ background:linear-gradient(120deg,#0B5CAD 0%,#1B7BB8 55%,#1B9AAA 100%); border-radius:16px; padding:24px 30px; color:#fff;
        display:flex; justify-content:space-between; align-items:flex-end; gap:24px; box-shadow:0 6px 24px rgba(11,92,173,.18); margin-bottom:18px; }}
.hero .eyebrow {{ font-size:.72rem; font-weight:700; letter-spacing:.14em; opacity:.85; text-transform:uppercase; }}
.hero h1 {{ font-size:1.9rem; font-weight:800; margin:4px 0 6px 0; color:#fff; padding:0; letter-spacing:-.01em; }}
.hero p {{ margin:0; opacity:.93; font-size:.95rem; max-width:820px; }}
.hero-chips {{ display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }}
.hero-chip {{ background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); padding:6px 12px; border-radius:999px; font-size:.78rem; font-weight:600; white-space:nowrap; }}

.chain {{ display:flex; gap:10px; margin:2px 0 18px 0; }}
.chain .step {{ flex:1; background:var(--card); border:1px solid var(--bordo); border-radius:12px; padding:10px 14px; }}
.chain .step b {{ color:var(--primario); font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; }}
.chain .step div {{ font-size:.82rem; color:var(--testo-soft); margin-top:2px; }}
.chain .arrow {{ align-self:center; color:var(--testo-soft); font-weight:700; }}

.demo {{ background:#FFF8E8; border:1px solid #F2D49B; border-left:6px solid {P['solare']}; border-radius:12px; padding:14px 18px; margin-bottom:14px; }}
.demo .t {{ font-weight:700; font-size:1.02rem; }} .demo .s {{ font-size:.9rem; color:#5B4A1F; margin-top:3px; }}

.section {{ margin:22px 0 10px 0; }}
.section .s-title {{ font-size:1.05rem; font-weight:700; color:var(--testo); }}
.section .s-cap {{ font-size:.82rem; color:var(--testo-soft); margin-top:2px; }}

.alert {{ overflow:visible; background:var(--card); border:1px solid var(--bordo); border-left:6px solid var(--lvl); border-radius:14px; padding:18px 22px; margin-bottom:6px; box-shadow:0 2px 10px rgba(27,42,65,.05); }}
.alert-head {{ display:flex; gap:18px; align-items:center; margin-bottom:14px; }}
.badge {{ background:var(--lvl); color:#fff; font-weight:800; font-size:.72rem; letter-spacing:.08em; padding:9px 14px; border-radius:8px; text-align:center; line-height:1.3; min-width:104px; }}
.alert-title {{ font-size:1.2rem; font-weight:700; color:var(--testo); }}
.alert-sub {{ font-size:.85rem; color:var(--testo-soft); margin-top:3px; }}
.alert-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; }}
.alert-cell {{ background:{P['sfondo']}; border:1px solid var(--bordo); border-radius:10px; padding:10px 14px; }}
.alert-cell .lbl {{ font-size:.68rem; color:var(--testo-soft); text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
.alert-cell .val {{ font-family:'JetBrains Mono',monospace; font-size:1.25rem; font-weight:600; color:var(--testo); margin-top:3px; }}

.alert-cell .sub {{ font-size:.74rem; color:var(--testo-soft); margin-top:3px; line-height:1.3; }}
.tip {{ position:relative; display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px; margin-left:6px;
       border-radius:50%; background:#E3EAF2; color:var(--primario); font-size:.68rem; font-weight:800; cursor:help; vertical-align:middle;
       text-transform:none; letter-spacing:0; }}
.tip .tiptext {{ visibility:hidden; opacity:0; position:absolute; z-index:50; top:22px; left:-10px; width:340px; background:#1B2A41; color:#fff;
       font-size:.8rem; font-weight:400; line-height:1.45; padding:10px 12px; border-radius:8px; box-shadow:0 6px 18px rgba(0,0,0,.18);
       transition:opacity .15s; text-align:left; }}
.tip:hover .tiptext, .tip:focus .tiptext {{ visibility:visible; opacity:1; }}
.alert-cell:nth-child(n+4) .tiptext, .act-card:nth-child(n+4) .tiptext {{ left:auto; right:-10px; }}
.now {{ display:flex; gap:12px; align-items:stretch; margin-bottom:14px; }}
.now-card {{ flex:1; background:var(--card); border:1px solid var(--bordo); border-radius:14px; padding:14px 18px; }}
.now-card.ev {{ border-color:{P['evento']}; background:#F1F3F8; }}
.now-card .lbl {{ font-size:.74rem; font-weight:700; color:var(--testo-soft); text-transform:uppercase; letter-spacing:.06em; }}
.now-card .big {{ font-size:1.35rem; font-weight:800; color:var(--testo); margin-top:4px; }}
.now-card .sub {{ font-size:.82rem; color:var(--testo-soft); margin-top:2px; }}
.now-arrow {{ align-self:center; font-family:'JetBrains Mono',monospace; font-weight:700; color:var(--primario); white-space:nowrap; }}
.acts-head {{ margin:4px 0 8px 0; }} .acts-head b {{ font-size:.95rem; margin-right:10px; }}
.acts-head span {{ font-size:.8rem; color:var(--testo-soft); }}
.acts {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-bottom:16px; }}
.act-card {{ background:var(--card); border:1px solid var(--bordo); border-top:4px solid #C9D2DC; border-radius:12px; padding:10px 12px; }}
.act-card.on {{ border-top-color:{P['ok']}; background:#F4FAF6; }}
.act-card.off {{ opacity:.62; }}
.act-by {{ font-family:'JetBrains Mono',monospace; font-size:.74rem; font-weight:700; color:var(--primario); }}
.act-name {{ font-size:.86rem; font-weight:700; margin-top:4px; line-height:1.3; }}
.act-eff {{ font-size:.76rem; color:var(--testo-soft); margin-top:4px; line-height:1.3; }}
.act-state {{ font-size:.76rem; font-weight:700; margin-top:6px; }}
.act-card.on .act-state {{ color:{P['ok']}; }} .act-card.off .act-state {{ color:var(--testo-soft); }}
.cmp {{ display:grid; grid-template-columns:1.4fr 1fr 1fr 0.9fr; background:var(--card); border:1px solid var(--bordo); border-radius:14px; overflow:visible; }}
.cmp > div {{ padding:11px 16px; border-bottom:1px solid #EEF2F6; }}
.cmp .c-head {{ background:{P['sfondo']}; }} .cmp .c-head b {{ display:block; font-size:.9rem; }}
.cmp .c-head span {{ font-size:.75rem; color:var(--testo-soft); }}
.cmp .c-head.act {{ background:#EAF4EE; }} .cmp .c-head.act b {{ color:{P['ok']}; }}
.cmp .c-lbl {{ font-size:.86rem; font-weight:600; }}
.cmp .c-val {{ font-family:'JetBrains Mono',monospace; font-size:1.02rem; }}
.cmp .c-val.act {{ background:#F6FBF8; font-weight:700; }}
.cmp .c-diff {{ font-family:'JetBrains Mono',monospace; font-size:.95rem; color:var(--testo-soft); }}
.cmp .c-diff.good {{ color:{P['ok']}; font-weight:700; }} .cmp .c-diff.bad {{ color:{P['crit']}; font-weight:700; }}
.hero-kpi {{ background:var(--card); border:1px solid var(--bordo); border-radius:14px; padding:18px 22px; height:100%; }}
.hero-kpi .lbl {{ font-size:.78rem; font-weight:700; color:var(--testo-soft); text-transform:uppercase; letter-spacing:.06em; }}
.hero-kpi .val {{ font-family:'JetBrains Mono',monospace; font-size:2.4rem; font-weight:600; color:{P['ok']}; line-height:1.15; margin-top:6px; }}
.hero-kpi .sub {{ font-size:.85rem; color:var(--testo-soft); margin-top:4px; }}

.msg {{ background:#F2F8FD; border:1px solid #CFE2F3; border-radius:14px; padding:16px 18px; font-size:.95rem; line-height:1.5; }}
.msg .lbl {{ font-size:.72rem; font-weight:700; color:var(--primario); text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }}

[data-testid="stMetric"] {{ background:var(--card); border:1px solid var(--bordo); border-radius:12px; padding:14px 16px; box-shadow:0 1px 6px rgba(27,42,65,.04); height:100%; }}
[data-testid="stMetricLabel"] p {{ font-size:.76rem !important; font-weight:600; color:var(--testo-soft); text-transform:uppercase; letter-spacing:.04em; }}
[data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace !important; font-weight:600; color:var(--testo); font-size:1.6rem !important; }}

.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--bordo); }}
.stTabs [data-baseweb="tab"] {{ height:44px; padding:0 18px; border-radius:10px 10px 0 0; font-weight:700; color:var(--testo-soft); }}
.stTabs [aria-selected="true"] {{ color:var(--primario) !important; background:var(--card); }}

[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid var(--bordo); }}
[data-testid="stSidebar"] .brand {{ font-weight:800; font-size:1.05rem; color:var(--testo); }}
[data-testid="stSidebar"] .brand-sub {{ font-size:.78rem; color:var(--testo-soft); margin-bottom:10px; }}
.sb-step {{ display:flex; align-items:center; gap:8px; margin:16px 0 6px 0; font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:var(--testo-soft); }}
.sb-step .n {{ background:var(--primario); color:#fff; width:20px; height:20px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:.72rem; }}

[data-testid="stDataFrame"] {{ border:1px solid var(--bordo); border-radius:12px; overflow:hidden; }}
.nota {{ font-size:.8rem; color:var(--testo-soft); }}
.fn {{ color:var(--primario); font-weight:800; text-decoration:none; margin-left:3px; font-size:.72rem; }}
.fn:hover {{ text-decoration:underline; }}
.notes {{ margin-top:34px; padding-top:14px; border-top:1px solid var(--bordo); }}
.notes-t {{ font-size:.9rem; font-weight:700; color:var(--testo); }}
.notes-c {{ font-size:.76rem; color:var(--testo-soft); margin-bottom:6px; }}
.notes ol {{ margin:0; padding-left:20px; }}
.notes li {{ font-size:.76rem; color:var(--testo-soft); line-height:1.5; margin-bottom:6px; scroll-margin-top:80px; }}
.notes li:target {{ background:#FFF8E8; }}
.footer-note {{ font-size:.75rem; color:var(--testo-soft); margin-top:28px; text-align:center; }}
@media (max-width:1100px) {{ .alert-grid, .acts {{ grid-template-columns:repeat(2,1fr); }} .hero, .chain, .now {{ flex-direction:column; align-items:stretch; }} }}
</style>
""", unsafe_allow_html=True)


def sezione(titolo, didascalia=None):
    cap = f'<div class="s-cap">{didascalia}</div>' if didascalia else ""
    st.markdown(f'<div class="section"><div class="s-title">{titolo}</div>{cap}</div>', unsafe_allow_html=True)


def sidebar_step(n, titolo):
    st.sidebar.markdown(f'<div class="sb-step"><span class="n">{n}</span>{titolo}</div>', unsafe_allow_html=True)


def fmt(x, dec=0):
    """Numero nel formato della lingua: 1.281,5 (IT) / 1,281.5 (EN)."""
    if x == float("inf"):
        return "> 30.000" if not EN else "> 30,000"
    s = f"{x:,.{dec}f}"
    return s if EN else s.replace(",", "§").replace(".", ",").replace("§", ".")


def fdata(ts):
    return ts.strftime("%d %b %H:%M") if EN else ts.strftime("%d/%m %H:%M")


def fgiorno(ts):
    mesi = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]
    return ts.strftime("%a %d %b") if EN else f"{ts.day} {mesi[ts.month - 1]}"


def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def stile(fig, titolo, altezza=440, y_suffix=" GW"):
    fig.update_layout(
        title=dict(text=titolo, font=dict(size=15, color=P["testo"], family=FONT), x=0, xanchor="left"),
        height=altezza, template="plotly_white", paper_bgcolor=P["card"], plot_bgcolor=P["card"],
        font=dict(family=FONT, size=12, color=P["testo"]), margin=dict(l=60, r=20, t=80, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor=P["bordo"], font=dict(family=FONT, size=12)),
        separators=".," if EN else ",.",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF2F6", linecolor=P["bordo"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF2F6", linecolor=P["bordo"], zeroline=False, ticksuffix=y_suffix)
    return fig


def mostra(fig):
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def etich(s):
    return s.get("etichetta_en", s["etichetta"]) if EN else s["etichetta"]


def az_nome(a):
    return a["nome_en"] if EN else a["nome"]


def az_eff(a):
    return a["effetto_en"] if EN else a["effetto"]


# ==============================================================================
# CALCOLI IN CACHE
# ==============================================================================
@st.cache_data(show_spinner=False)
def carica_params(percorso):
    p = Path(percorso)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def c_prep(df, pp):
    return M.prepara(df, carica_params(pp))


def scenari_disponibili(prep, params):
    s = {"dataset": M.scenario_dataset(prep)}
    if params:
        s.update(M.scenari_da_parametri(params))
    return s


def _scen(df, pp, key):
    prep = c_prep(df, pp)
    return prep, scenari_disponibili(prep, carica_params(pp))[key]


@st.cache_data(show_spinner=False)
def c_sim(df, pp, key, ip_t, x, piano):
    prep, scen = _scen(df, pp, key)
    return M.simula(M.applica_scenario(prep, scen), M.Ipotesi(**dict(ip_t)), M.leve_da_preavviso(x, piano))


@st.cache_data(show_spinner=False)
def c_ensemble(df, pp, key, ip_t, x, piano):
    prep, scen = _scen(df, pp, key)
    return M.ensemble(prep, scen, M.Ipotesi(**dict(ip_t)), x, piano=piano)


@st.cache_data(show_spinner=False)
def c_prob_curva(df, pp, key, ip_t, piano):
    prep, scen = _scen(df, pp, key)
    ip = M.Ipotesi(**dict(ip_t))
    return pd.DataFrame([dict(preavviso=x, prob=M.ensemble(prep, scen, ip, x, piano=piano)["prob_evento"]) for x in range(0, 15)])


@st.cache_data(show_spinner=False)
def c_curva(df, pp, key, ip_t):
    prep, scen = _scen(df, pp, key)
    return M.curva_preavviso(prep, scen, M.Ipotesi(**dict(ip_t)))


@st.cache_data(show_spinner=False)
def c_accumulo(df, pp, key, ip_t, x):
    prep, scen = _scen(df, pp, key)
    return M.accumulo_necessario(prep, scen, M.Ipotesi(**dict(ip_t)), x)


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown(f'<div class="brand">⚡ Dunkelflaute EWS</div><div class="brand-sub">{t("brand_sub")}</div>', unsafe_allow_html=True)
demo = st.sidebar.toggle(t("demo"), value=False, help=t("demo_help"))

df_input, fonte = None, None
for nome in ["simulazione_dunkelflaute_nov2029_gen2030.csv"]:
    for base in (QUI, Path.cwd()):
        if (base / nome).exists():
            df_input, fonte = pd.read_csv(base / nome), nome
            break
    if df_input is not None:
        break
if df_input is None:
    st.warning(t("no_file"), icon="📂")
    st.stop()

params = carica_params(PARAMS_PATH)
try:
    prep = c_prep(df_input, PARAMS_PATH)
except ValueError as e:
    st.error(t("bad_file", e=e), icon="⛔")
    st.stop()
if params is None:
    st.sidebar.warning(t("no_params"))
scenari = scenari_disponibili(prep, params)

scen_key, piano = "dataset", True

st.sidebar.markdown(f'<div class="sb-step">{t("sb_system2")}</div>', unsafe_allow_html=True)
E_max = st.sidebar.slider(t("storage_e"), 0, 3000, 80, step=10, format="%d GWh", help=t("storage_e_help"))
P_acc = st.sidebar.slider(t("storage_p"), 5, 150, 20, format="%d GW")
F_max = st.sidebar.slider(t("reserve"), 10, 100, 40, format="%d GW", help=t("reserve_help"))
with st.sidebar.expander(t("more_hyp")):
    scorta = st.number_input(t("fuel"), 500, 20000, 4000, step=500)
    soc_mercato = st.slider(t("soc_mkt"), 0.0, 1.0, 0.5, 0.05, format="%.2f", help=t("soc_mkt_help"))

finestra_gg = 21

# ---------------- modalità presentazione ----------------
S100 = "sintetico_100" if "sintetico_100" in scenari else "dataset"
PASSI = {
    1: ("d1", "d1s", dict(scen="dataset", X=7, piano=True, tab="see")),
    2: ("d2", "d2s", dict(scen="dataset", X=7, piano=False, tab="measure")),
    3: ("d3", "d3s", dict(scen="dataset", X=7, piano=True, tab="measure")),
    4: ("d4", "d4s", dict(scen=S100, X=7, piano=True, tab="decide")),
    5: ("d5", "d5s", dict(scen=S100, X=7, piano=True, tab="decide")),
}
passo, tab_primo, ov = None, "see", None
if demo:
    passo = st.radio("demo_step", list(PASSI.keys()), horizontal=True, format_func=lambda k: t(PASSI[k][0]),
                     label_visibility="collapsed")
    ov = PASSI[passo][2]
    scen_key, piano, tab_primo = ov["scen"], ov["piano"], ov["tab"]

ip = M.Ipotesi(E_max=float(E_max), P_acc=float(P_acc), F_max=float(F_max), scorta_base=float(scorta),
               soc_senza_precarica=float(soc_mercato))
ip_t = tuple(sorted(asdict(ip).items()))
scen = scenari[scen_key]
sc_att = M.applica_scenario(prep, scen)
t0, t1 = sc_att["t0"], sc_att["t1"]
ts = prep["df"]["timestamp"]
ts0, ts1 = ts.iloc[t0], ts.iloc[min(t1, len(ts) - 1)]


def giorno_lungo(x):
    g = GIORNI[lang][x.dayofweek]
    mesi = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]
    return f"{g} {x.day} {x.strftime('%b') if EN else mesi[x.month - 1]} {x.year}"


def tip(testo):
    return f'<span class="tip" tabindex="0">?<span class="tiptext">{testo}</span></span>'


# ==============================================================================
# HEADER (senza riquadri) + FLUSSO 1-3
# ==============================================================================
st.markdown(f"""
<div class="hero">
  <div>
    <div class="eyebrow">{t("eyebrow")}</div>
    <h1>Dunkelflaute Early Warning</h1>
    <p>{t("hero_p")}</p>
  </div>
</div>
<div class="chain">
  <div class="step"><b>{t("chain_see")}</b><div>{t("chain_see_d")}</div></div><span class="arrow">→</span>
  <div class="step"><b>{t("chain_measure")}</b><div>{t("chain_measure_d")}</div></div><span class="arrow">→</span>
  <div class="step"><b>{t("chain_decide")}</b><div>{t("chain_decide_d")}</div></div>
</div>
""", unsafe_allow_html=True)

if demo:
    st.markdown(f'<div class="demo"><div class="t">{t(PASSI[passo][0])}</div><div class="s">{t(PASSI[passo][1])}</div></div>',
                unsafe_allow_html=True)

# ==============================================================================
# BOX DI ALLERTA con il cursore dei giorni di anticipo
# ==============================================================================
box = st.container(border=True)
with box:
    preavviso = st.slider(t("lead_main"), 0, 14, 7, format="T−%d", disabled=demo, help=t("lead_main_help"), key="preavviso")
if demo:
    preavviso = ov["X"]

with st.spinner(t("spinner")):
    df_con, kpi_con = c_sim(df_input, PARAMS_PATH, scen_key, ip_t, preavviso, piano)
    df_senza, kpi_senza = c_sim(df_input, PARAMS_PATH, scen_key, ip_t, 0, False)
    ens = c_ensemble(df_input, PARAMS_PATH, scen_key, ip_t, preavviso, piano)
    curva = c_curva(df_input, PARAMS_PATH, scen_key, ip_t)

calo = M.calo_produzione(prep, sc_att)
protetti = max(0.0, kpi_senza["nec_non_serviti_GWh"] - kpi_con["nec_non_serviti_GWh"])
ore_ev = max(1, t1 - t0)
atteso_gw = float((prep["base_sol"] + prep["base_eol"])[t0:t1].mean())
previsto_gw = float((sc_att["df"].potenza_generata_solare + sc_att["df"].potenza_generata_eolica).values[t0:t1].mean())
ess_ev = float(sc_att["df"].potenza_consumata_necessaria.values[t0:t1].sum())
ns = kpi_con["nec_non_serviti_GWh"]
quota_scoperta = ns / ess_ev if ess_ev > 0 else 0.0

if ens["prob_evento"] < 0.1:
    col, badge, titolo = P["ok"], t("badge_green"), t("al_green")
elif ens["prob_evento"] < 0.3:
    col, badge, titolo = P["import"], t("badge_pre"), t("al_pre")
elif ns > 1.0:
    col, badge, titolo = P["crit"], t("badge_red"), t("al_red", v=fmt(ns))
else:
    col, badge, titolo = P["warn"], t("badge_orange"), t("al_orange")
celle = [
    (t("c_prob") + tip(t("tip_prob")), f"{ens['prob_evento']*100:.0f}%", t("prob_sub")),
    (t("c_dur"), f"{fmt(ens['durata_mediana_gg'])} {t('days')}", t("dur_sub", a=fgiorno(ts0))),
    (t("c_drop") + tip(t("tip_drop")), f"−{calo*100:.0f}%",
     t("drop_sub", a=fmt(atteso_gw), b=fmt(previsto_gw), c=fmt(atteso_gw - previsto_gw))),
    (t("c_gap") + tip(t("tip_gap")), f"{fmt(kpi_con['buco_GWh'])} GWh", t("gap_sub")),
    (t("c_risk2") + tip(t("tip_risk")), f"{quota_scoperta*100:.1f}%" if 0 < quota_scoperta < 0.1 else f"{quota_scoperta*100:.0f}%",
     t("risk_sub", a=fmt(ns), b=fmt(ess_ev))),
]
celle_html = "".join(f'<div class="alert-cell"><div class="lbl">{l}</div><div class="val">{v}</div><div class="sub">{s}</div></div>'
                     for l, v, s in celle)
with box:
    st.markdown(f"""
<div class="alert" style="--lvl:{col}">
  <div class="alert-head"><div class="badge">{badge}</div>
    <div><div class="alert-title">{titolo}</div>
    <div class="alert-sub">{t("al_sub2", x=preavviso, a=fdata(ts0), b=fdata(ts1))}</div></div></div>
  <div class="alert-grid">{celle_html}</div>
</div>""", unsafe_allow_html=True)

# ==============================================================================
# RISULTATO: oggi, arrivo dell'evento, agire oggi vs non fare nulla
# ==============================================================================
sezione(t("res_title"), t("res_cap2"))
oggi = ts0 - pd.Timedelta(days=preavviso)
n_att = len(M.azioni_attive(preavviso, piano))
n_tot = len(M.AZIONI)
if not piano:
    acts_sub = t("now_acts_off")
elif n_att == n_tot:
    acts_sub = t("now_acts_sub0")
else:
    acts_sub = t("now_acts_sub", n=n_tot - n_att)
st.markdown(f"""
<div class="now">
  <div class="now-card"><div class="lbl">{t("now_today")}</div><div class="big">{giorno_lungo(oggi)}</div>
    <div class="sub">{t("now_today_sub", x=preavviso) if preavviso else t("now_today_sub0")}</div></div>
  <div class="now-arrow">T−{preavviso} →</div>
  <div class="now-card ev"><div class="lbl">{t("now_event")}</div><div class="big">{giorno_lungo(ts0)}</div>
    <div class="sub">{t("now_event_sub", h=ts0.strftime("%H:%M"), d=fmt(ore_ev / 24))}</div></div>
  <div class="now-card"><div class="lbl">{t("now_acts")}</div><div class="big">{n_att} / {n_tot}</div>
    <div class="sub">{acts_sub}</div></div>
</div>""", unsafe_allow_html=True)

att_ids = {x["id"] for x in M.azioni_attive(preavviso, piano)}
NOTE_AZ = {az["id"]: i + 1 for i, az in enumerate(sorted(M.AZIONI, key=lambda x: -x["scadenza"]))}
schede = ""
for az in sorted(M.AZIONI, key=lambda x: -x["scadenza"]):
    on = az["id"] in att_ids
    stato = t("act_on") if on else (t("act_off") if piano else t("act_planoff"))
    n_nota = NOTE_AZ[az["id"]]
    schede += (f'<div class="act-card {"on" if on else "off"}"><div class="act-by">{t("act_by", d=az["scadenza"])}</div>'
               f'<div class="act-name">{az_nome(az)}<sup><a class="fn" href="#nota-{n_nota}">{n_nota}</a></sup></div><div class="act-eff">{az_eff(az)}</div>'
               f'<div class="act-state">{stato}</div></div>')
st.markdown(f'<div class="acts-head"><b>{t("acts_title")}</b><span>{t("acts_cap")}</span></div>'
            f'<div class="acts">{schede}</div>', unsafe_allow_html=True)


def dettaglio_costi(k):
    righe = []
    for voce, v in k["costo_voci"].items():
        if v["Meuro"] < 0.5:
            continue
        if voce == "azioni":
            righe.append(f'{t("cv_azioni")}: {fmt(v["Meuro"])} M€')
        else:
            righe.append(f'{t("cv_" + voce)}: {fmt(v["GWh"])} GWh × {fmt(v["prezzo"] * 1000)} €/MWh = {fmt(v["Meuro"])} M€')
    return "<br>".join(righe) + f'<br><b>= {fmt(k["costo_Meuro"])} M€</b>'


tip_costo = tip(t("tip_cost") + f'<br><br><b>{t("cmp_nothing")}</b><br>{dettaglio_costi(kpi_senza)}'
                f'<br><br><b>{t("cmp_act", x=preavviso)}</b><br>{dettaglio_costi(kpi_con)}')
righe_cmp = [("r1", "nec_non_serviti_GWh", "GWh", 0), ("r2", "ore_nec_scoperti", "h", 0),
             ("r9", "costo_Meuro", "M€", 0)]
corpo = ""
for chiave_t, k_, u, dec in righe_cmp:
    a_, b_ = kpi_senza[k_], kpi_con[k_]
    diff = b_ - a_
    cls = "good" if diff < -1e-9 else ("bad" if diff > 1e-9 else "")
    lbl_html = t(chiave_t) + (tip_costo if k_ == "costo_Meuro" else "")
    corpo += (f'<div class="c-lbl">{lbl_html}</div><div class="c-val">{fmt(a_, dec)} {u}</div>'
              f'<div class="c-val act">{fmt(b_, dec)} {u}</div><div class="c-diff {cls}">{"" if diff > 0 else "−" if diff < 0 else ""}{fmt(abs(diff), dec)} {u}</div>')
c_sx, c_dx = st.columns([2.6, 1])
with c_sx:
    st.markdown(f"""
<div class="cmp">
  <div></div>
  <div class="c-head"><b>{t("cmp_nothing")}</b><span>{t("cmp_nothing_sub")}</span></div>
  <div class="c-head act"><b>{t("cmp_act", x=preavviso)}</b><span>{t("cmp_act_sub", n=n_att) if piano else t("cmp_act_off")}</span></div>
  <div class="c-head"><b>{t("cmp_diff")}</b><span>&nbsp;</span></div>
  {corpo}
</div>""", unsafe_allow_html=True)
with c_dx:
    st.markdown(f"""<div class="hero-kpi"><div class="lbl">{t("hk_lbl")}</div>
      <div class="val">{fmt(protetti)} GWh</div>
      <div class="sub">{t("hk_sub", v=fmt(kpi_senza['nec_non_serviti_GWh']))}</div></div>""", unsafe_allow_html=True)

# ==============================================================================
# TABS
# ==============================================================================
st.write("")
ID_TAB = {"see": t("chain_see"), "measure": t("chain_measure"), "decide": t("chain_decide"), "ai": t("tab_ai")}
ordine = [tab_primo] + [k for k in ["see", "measure", "decide", "ai"] if k != tab_primo]
tabs = dict(zip(ordine, st.tabs([ID_TAB[k] for k in ordine])))
fin = (0, len(ts) - 1) if finestra_gg == "Tutto" else (max(0, t0 - finestra_gg * 24), min(len(ts) - 1, t1 + finestra_gg * 24))


def banda_evento(fig, testo=True):
    fig.add_vrect(x0=ts0, x1=ts1, fillcolor=P["evento"], opacity=0.10, line_width=0,
                  annotation_text="Dunkelflaute" if testo else None, annotation_position="top left",
                  annotation_font=dict(color=P["evento"]))


# ------------------------------------------------------------------ ① VEDERE
with tabs["see"]:
    sezione(t("see_title") + " · " + t("fc_title"), t("fc_cap", x=preavviso))
    roll_see = st.toggle(t("rolling"), value=False, key="roll_see")
    sel = (ens["tempo"] >= fin[0]) & (ens["tempo"] <= fin[1])
    liscia = (lambda v: pd.Series(np.asarray(v, float)).rolling(24, min_periods=1).mean().values) if roll_see else (lambda v: np.asarray(v, float))
    tempo = ens["tempo"][sel]
    tt = ts.iloc[tempo].values
    fig = go.Figure()
    fig.add_scatter(x=tt, y=liscia(ens["ren_p75"][sel]), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=tt, y=liscia(ens["ren_p25"][sel]), mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=rgba(P["eolico"], 0.20), name=t("fc_band"), hoverinfo="skip")
    fig.add_scatter(x=tt, y=liscia(ens["ren_p50"][sel]), mode="lines", line=dict(color=P["eolico"], width=2, dash="dot"),
                    name=t("fc_med"), hovertemplate="%{y:.1f} GW")
    d_true = sc_att["df"].iloc[tempo]
    fig.add_scatter(x=tt, y=liscia(d_true.potenza_generata_solare + d_true.potenza_generata_eolica), mode="lines",
                    line=dict(color=P["primario"], width=2.2), name=t("fc_sim"), hovertemplate="%{y:.1f} GW")
    d_plan = df_con.iloc[tempo]
    fig.add_scatter(x=tt, y=liscia(d_plan.DomEff if piano and preavviso > 0 else d_plan.Domanda_Tot), mode="lines",
                    line=dict(color=P["domanda_tot"], width=2.2), name=t("s_dt"), hovertemplate="%{y:.1f} GW")
    fig.add_scatter(x=tt, y=liscia(d_true.potenza_consumata_necessaria), mode="lines",
                    line=dict(color=P["domanda_nec"], width=1.8, dash="dash"), name=t("dem_nec"), hovertemplate="%{y:.1f} GW")
    banda_evento(fig)
    fig.update_layout(hovermode="x unified")
    with st.container(border=True):
        mostra(stile(fig, t("fc_chart") + (t("avg24") if roll_see else ""), 460))


# ------------------------------------------------------------------ ② MISURARE
with tabs["measure"]:
    sezione(t("tw_title"), t("tw_cap"))
    cm1, cm2, _ = st.columns([1, 1, 3])
    with cm1:
        rolling = st.toggle(t("rolling"), value=False, key="roll_meas")
    with cm2:
        mostra_mit = st.toggle(t("show_mit"), value=False, help=t("show_mit_help"))
    dc = df_con.copy()
    dc["ComunitaTagli"] = dc.Comunita.clip(lower=0) + dc.Risparmio + (dc.Taglio if (piano and preavviso >= 1) else 0)
    dc["Distacchi"] = 0.0 if (piano and preavviso >= 1) else dc.Taglio
    if rolling:
        num = [c for c in dc.columns if c not in ("tempo_ore", "timestamp", "in_evento")]
        dc[num] = dc[num].rolling(24, min_periods=1).mean()
    dc = dc.iloc[fin[0]:fin[1]]
    fig = go.Figure()
    serie = [(t("s_solar"), "Solare", P["solare"]), (t("s_wind"), "Eolico", P["eolico"])]
    if mostra_mit:
        serie += [(t("m_centrali"), "Fossile", P["fossile"]), (t("m_import"), "Import", P["import"]),
                  (t("m_accumuli"), "Accumulo", P["accumulo"]), (t("m_industria"), "Industria", P["riduzione"]),
                  (t("m_comunita"), "ComunitaTagli", P["taglio"])]
        if dc.Distacchi.max() > 0.01:
            serie.append((t("m_distacchi"), "Distacchi", "#B07AA1"))
    for nome, c, colr in serie:
        fig.add_scatter(x=dc.timestamp, y=dc[c], mode="lines", name=nome, stackgroup="uno",
                        line=dict(width=0.6, color=colr), fillcolor=rgba(colr, 0.8), hovertemplate="%{y:.1f} GW")
    if mostra_mit:
        fig.add_scatter(x=dc.timestamp, y=dc.NonServiti, mode="lines", name=t("s_ns"), fill="tozeroy",
                        fillcolor=rgba(P["non_serviti"], 0.40), line=dict(width=1.2, color=P["non_serviti"]), hovertemplate="%{y:.1f} GW")
    fig.add_scatter(x=dc.timestamp, y=dc.Domanda_Tot, mode="lines", name=t("s_dt"),
                    line=dict(width=2.2, color=P["domanda_tot"]), hovertemplate="%{y:.1f} GW")
    fig.add_scatter(x=dc.timestamp, y=dc.Domanda_Nec, mode="lines", name=t("s_dn"),
                    line=dict(width=2, color=P["domanda_nec"], dash="dash"), hovertemplate="%{y:.1f} GW")
    banda_evento(fig)
    fig.update_layout(hovermode="x unified")
    with st.container(border=True):
        mostra(stile(fig, t("bal_chart") + (t("avg24") if rolling else ""), 500))

    lab_con = t("with_plan", x=preavviso) if piano else t("current")
    st.markdown(f"**{t('cmp_title')}**")
    righe = [("r1", "nec_non_serviti_GWh", "GWh", 0), ("r2", "ore_nec_scoperti", "h", 0), ("r5", "fossile_GWh", "GWh", 0), ("r6", "import_GWh", "GWh", 0),
             ("r7", None, "GWh", 0), ("r8", "co2_Mt", "Mt", 2), ("r9", "costo_Meuro", "M€", 0)]

    def val(k, chiave, u, dec):
        v = k["riduzione_GWh"] + k["tagli_GWh"] if chiave is None else k[chiave]
        return f"{fmt(v, dec)} {u}"
    st.dataframe(pd.DataFrame({t("ind"): [t(r[0]) for r in righe],
                               lab_con: [val(kpi_con, *r[1:]) for r in righe],
                               t("surprise"): [val(kpi_senza, *r[1:]) for r in righe]}),
                 hide_index=True, use_container_width=True)
    with st.expander(t("hyp_title")):
        st.markdown(t("hyp_md"))
    with st.expander(t("inputs")):
        st.caption(f"`{fonte}` · {fmt(len(df_input))} {t('rows')} × {len(df_input.columns)} {t('cols')}")
        st.dataframe(df_input, use_container_width=True, height=380)

# ------------------------------------------------------------------ ③ DECIDERE
with tabs["decide"]:
    sezione(t("plan_title"), t("plan_cap", x=preavviso))
    att = {a["id"] for a in M.azioni_attive(preavviso, piano)}
    fg = go.Figure()
    for a in sorted(M.AZIONI, key=lambda a: a["scadenza"]):
        on = a["id"] in att
        fg.add_bar(y=[az_nome(a)], x=[a["scadenza"]], base=[-a["scadenza"]], orientation="h",
                   marker_color=P["ok"] if on else "#C9D2DC", showlegend=False,
                   hovertemplate=f"<b>{az_nome(a)}</b><br>{t('deadline_hover', d=a['scadenza'])}<br>{az_eff(a)}<extra></extra>",
                   text=[("✓ " if on else "✗ ") + az_eff(a)], textposition="inside", insidetextanchor="start",
                   textfont=dict(color="white" if on else P["testo_soft"], size=11))
    fg.add_vline(x=-preavviso, line=dict(color=P["primario"], width=2, dash="dash"))
    fg.add_annotation(x=-preavviso, y=1.06, yref="paper", text=f"<b>{t('today', x=preavviso)}</b>", showarrow=False,
                      font=dict(color=P["primario"]))
    fg.update_xaxes(range=[-14.5, 0.3], tickvals=list(range(-14, 1)), ticktext=[f"T{d}" if d else "T0" for d in range(-14, 1)],
                    title=t("tl_axis"))
    with st.container(border=True):
        mostra(stile(fg, t("tl_chart", a=len(att), b=len(M.AZIONI)), 460, ""))

    with st.container(border=True):
        fc = go.Figure()
        fc.add_scatter(x=curva.preavviso, y=curva.nec_non_serviti_GWh, mode="lines+markers", line=dict(color=P["non_serviti"], width=3),
                       marker=dict(size=7), fill="tozeroy", fillcolor=rgba(P["non_serviti"], 0.10), name=t("s_ns"),
                       hovertemplate="T−%{x}: %{y:,.0f} GWh<extra></extra>")
        y_x = float(curva.loc[curva.preavviso == preavviso, "nec_non_serviti_GWh"].iloc[0])
        fc.add_scatter(x=[preavviso], y=[y_x], mode="markers", marker=dict(size=15, color=P["primario"], line=dict(color="white", width=2)),
                       name=t("today_pt", x=preavviso), hovertemplate="%{y:,.0f} GWh<extra></extra>")
        fc.update_xaxes(title=t("curve_axis"), dtick=1, autorange="reversed")
        mostra(stile(fc, t("curve_chart"), 400, " GWh"))
        ns0 = float(curva.nec_non_serviti_GWh.iloc[0])
        if preavviso > 0 and ns0 > 0:
            st.markdown(t("curve_msg", v=fmt((ns0 - y_x) / preavviso), a=fmt(ns0), b=fmt(y_x), x=preavviso))

    # ---- comunità energetiche e cittadini
    sezione(t("comm_title"), t("comm_cap"))
    if "comunita" in att:
        c1, c2 = st.columns([1.6, 1])
        with c1:
            vigilia = ts0 - pd.Timedelta("1D")
            st.markdown(f"**📲 {t('comm_msg_lbl')}**")
            st.info(t("comm_msg", a=fgiorno(ts0), d=fmt((t1 - t0) / 24), b=t("comm_before", a=fgiorno(vigilia))))
        with c2:
            m1, m2 = st.columns(2)
            m1.metric(t("comm_k1"), f"{fmt(kpi_con['comunita_picco_GW'], 1)} GW")
            m2.metric(t("comm_k2"), f"{fmt(kpi_con['comunita_GWh'])} GWh")
            st.metric(t("comm_k3"), f"≈ {fmt(kpi_con['comunita_picco_GW'] / 0.5 * 1000 / 1e3, 0)} {t('mln')}", help=t("comm_k3_help"))
    else:
        st.warning(t("comm_off"), icon="⏳")


# ------------------------------------------------------------------ 🤖 APPROCCIO AI
with tabs["ai"]:
    sezione(t("ai_title"), t("ai_cap"))
    with st.container(border=True):
        st.markdown(f"**{t('ai_why')}**")
        st.markdown(t("ai_why_md"))
        st.markdown(t("how_gen_md"))
    if params:
        v = params["validazione"]
        o = params.get("fuori_campione", {})
        sezione(t("acc_title"), t("acc_cap"))
        sovrap = float(np.minimum(v["freq_durata_storico"], v["freq_durata_sintetico"]).sum())
        q = st.columns(3)
        q[0].metric(t("kpi_sim"), f"{sovrap*100:.0f}%", help=t("acc_overlap_h"))
        if o:
            pct = list(o["percentile_inverni_test"].values())
            n_ok = sum(0.05 <= p <= 0.95 for p in pct)
            rr = float(np.exp(o["loglik_giorno_hmm"] - o["loglik_giorno_ingenuo"]))
            q[1].metric(t("kpi_test"), f"{n_ok} {t('su')} {len(pct)}", help=t("acc_test_h"))
            q[2].metric(t("kpi_ll"), f"{fmt(rr, 1)}×", help=t("acc_ll_h", r=fmt(rr, 1)))

        def riga(nome, insieme, h, g, dec, unita=""):
            return {t("tm_metric"): nome, t("tm_set"): insieme,
                    t("tm_hist"): f"{fmt(h, dec)}{unita}", t("tm_gen"): f"{fmt(g, dec)}{unita}",
                    t("tm_var"): f"{(g / h - 1) * 100:+.0f}%"}
        gg = f" {t('dd')}"
        righe_m = [
            riga(t("ck1"), t("set_all"), v["eventi_per_inverno_storico"], v["eventi_per_inverno_sintetico"], 1),
            riga(t("ck2"), t("set_all"), v["durata_media_storico"], v["durata_media_sintetico"], 1, gg),
            riga(t("ck4"), t("set_all"), v["corr_produzione_temperatura_storico"], v["corr_produzione_temperatura_sintetico"], 2),
        ]
        if o:
            righe_m += [
                riga(t("ck1"), t("set_test"), o["test"]["eventi_per_inverno"], o["sintetico"]["eventi_per_inverno"], 1),
                riga(t("ck2"), t("set_test"), o["test"]["durata_media"], o["sintetico"]["durata_media"], 1, gg),
                riga(t("ck4"), t("set_test"), o["test"]["corr"], o["sintetico"]["corr"], 2),
            ]
        st.dataframe(pd.DataFrame(righe_m), hide_index=True, use_container_width=True)
        st.caption(t("tm_note"))

        # specchietto: perché la durata massima generata è più lunga, e il legame con la crescita della domanda
        bins_d = np.array(v["bins_durata_gg"]); fs = np.array(v["freq_durata_sintetico"])
        dh = int(v["durata_max_storico"])
        n_ev_sint = v["eventi_per_inverno_sintetico"] * v["n_inverni_sintetici"]
        n_lunghi = float(fs[bins_d > dh].sum()) * n_ev_sint
        ogni = v["n_inverni_sintetici"] / n_lunghi if n_lunghi > 0 else float("inf")
        sc_ds = M.applica_scenario(prep, scenari["dataset"])
        dd_ = sc_ds["df"]; a0, a1 = sc_ds["t0"], sc_ds["t1"]
        dom_ev = float(dd_.potenza_consumata.values[a0:a1].mean())
        ren_ev = float((dd_.potenza_generata_solare + dd_.potenza_generata_eolica).values[a0:a1].mean())
        CRESCITA = 1.125   # crescita media della domanda al 2030 usata nel dataset
        dopo = (dom_ev - ren_ev) * 24
        extra = dom_ev * (1 - 1 / CRESCITA) * 24
        prima = dopo - extra
        with st.container(border=True):
            st.markdown(f"**⏱ {t('dmax_title')}**")
            st.markdown(t("dmax_md", nh=v["n_inverni_storici"], ns=fmt(v["n_inverni_sintetici"]), dh=dh,
                          ogni=fmt(round(ogni, -1) if ogni < float("inf") else 0), sim=f"{sovrap*100:.0f}",
                          extra=fmt(extra), prima=fmt(prima), dopo=fmt(dopo), pct=f"{extra / prima * 100:.0f}",
                          e9=fmt(extra * dh / 1000, 1), dmax=int(v["durata_max_sintetico"]),
                          emax=fmt(extra * v["durata_max_sintetico"] / 1000, 1)))
        sezione(t("gen_title"), t("gen_cap", n=v["n_inverni_storici"], m=fmt(v["n_inverni_sintetici"])))
        c1, c2 = st.columns([1.6, 1])
        with c1, st.container(border=True):
            lab = [f"{b}" if b < 15 else "15+" for b in v["bins_durata_gg"]]
            fv = go.Figure()
            fv.add_bar(x=lab, y=np.array(v["freq_durata_storico"]) * 100, name=t("hist15"), marker_color=P["storico"], hovertemplate="%{y:.1f}%")
            fv.add_bar(x=lab, y=np.array(v["freq_durata_sintetico"]) * 100, name=t("synth"), marker_color=P["sintetico"], hovertemplate="%{y:.2f}%")
            fv.update_layout(barmode="group", bargap=0.25)
            fv.update_xaxes(title=t("dur_axis"))
            mostra(stile(fv, t("dur_chart"), 380, "%"))
        with c2, st.container(border=True):
            st.markdown(f"**{t('regimes')}**")
            st.dataframe(pd.DataFrame([{
                t("rg"): r.get("nome_en", r["nome"]) if EN else r["nome"], t("rg_out"): f"{r['r_medio']*100:.0f}%",
                t("rg_t"): f"{r['t_anom']:+.1f} °C", t("rg_d"): f"{fmt(r['durata_media_gg'], 1)} {t('dd')}"}
                for r in params["regimi"]]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame([{t("sc_tab"): etich(s), t("sc_dur"): f"{len(s['r'])} {t('dd')}",
                                    t("rg_out"): f"{np.mean(s['r'])*100:.0f}%"} for s in scenari.values()]),
                     hide_index=True, use_container_width=True)

    if params:
        sezione(t("need_title"), t("need_cap", x=preavviso))
        conf = [k for k in ["storico_peggiore", "sintetico_100", "sintetico_1000"] if k in scenari]
        fabb = {k: c_accumulo(df_input, PARAMS_PATH, k, ip_t, preavviso) for k in conf}
        nomi_b = {"storico_peggiore": t("need_hist"), "sintetico_100": t("need_100"), "sintetico_1000": t("need_1000")}
        col_b = {"storico_peggiore": P["storico"], "sintetico_100": P["sintetico"], "sintetico_1000": rgba(P["sintetico"], 0.45)}
        ys = [min(fabb[k], 30000) for k in conf]
        c1, c2 = st.columns([1.6, 1])
        with c1, st.container(border=True):
            fb = go.Figure(go.Bar(x=[nomi_b[k] for k in conf], y=ys, marker_color=[col_b[k] for k in conf],
                                  text=[f"{fmt(fabb[k])} GWh" for k in conf], textposition="outside",
                                  hovertemplate="%{y:,.0f} GWh<extra></extra>"))
            fb.add_hline(y=E_max, line=dict(color=P["accumulo"], dash="dash"), annotation_text=t("installed", v=fmt(E_max)),
                         annotation_position="top left")
            fb.update_yaxes(range=[0, max(ys + [E_max]) * 1.2])
            mostra(stile(fb, t("need_chart", x=preavviso), 400, " GWh"))
        with c2:
            if "storico_peggiore" in fabb and "sintetico_100" in fabb:
                st.markdown(t("need_msg", a=fmt(fabb["storico_peggiore"]), b=fmt(fabb["sintetico_100"])))

    sezione(t("ens_title"))
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown(t("ens_md"))
    with c2, st.container(border=True):
        pc = c_prob_curva(df_input, PARAMS_PATH, scen_key, ip_t, piano)
        fp = go.Figure()
        fp.add_scatter(x=pc.preavviso, y=pc.prob * 100, mode="lines+markers", line=dict(color=P["primario"], width=3),
                       marker=dict(size=7), hovertemplate="T−%{x}: %{y:.0f}%<extra></extra>", name=t("c_prob"))
        fp.add_scatter(x=[preavviso], y=[ens["prob_evento"] * 100], mode="markers", name=t("today_pt", x=preavviso),
                       marker=dict(size=15, color=P["solare"], line=dict(color="white", width=2)))
        fp.update_xaxes(title=t("curve_axis"), dtick=1, autorange="reversed")
        fp.update_yaxes(range=[0, 105])
        mostra(stile(fp, t("ens_chart"), 340, "%"))

    sezione(t("lim_title"))
    pv = (params or {}).get("fuori_campione", {}).get("previsione", {}).get("1", {}).get("modello", {})
    with st.container(border=True):
        st.markdown(t("lim_md", r1=f"{pv.get('richiamo', 0)*100:.0f}", p1=f"{pv.get('precisione', 0)*100:.0f}"))

note_html = "".join(
    f'<li id="nota-{NOTE_AZ[az["id"]]}"><b>{az_nome(az)}</b> ({t("act_by", d=az["scadenza"])}). {az["fonte_en"] if EN else az["fonte"]}</li>'
    for az in sorted(M.AZIONI, key=lambda x: -x["scadenza"]))
st.markdown(f'<div class="notes"><div class="notes-t">{t("notes_title")}</div><div class="notes-c">{t("notes_cap")}</div>'
            f'<ol>{note_html}</ol></div>', unsafe_allow_html=True)
st.markdown(f'<div class="footer-note">{t("footer")}</div>', unsafe_allow_html=True)
