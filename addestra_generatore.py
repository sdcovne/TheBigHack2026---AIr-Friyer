"""
Generatore AI di Dunkelflaute sintetiche — addestramento offline (v3).

Modello: Hidden Markov Model AUTOREGRESSIVO a 3 regimi meteo, appreso senza etichette con EM (Baum-Welch),
su due variabili giornaliere invernali (nov–gen):
  x1 = log(CF_comb giornaliero / CF_comb climatologico)   quanto producono eolico + FV rispetto all'atteso
  x2 = anomalia di temperatura (°C)                        accoppiamento freddo / calma
CF_comb = 0,65·CF_eolico + 0,35·CF_solare (stesse quote del dataset).

Novità v3
  * Validazione FUORI CAMPIONE: addestramento su 12 inverni (2010-11 … 2021-22), test su 3 mai visti
    (2022-23 … 2024-25, che contengono le Dunkelflaute reali di nov e dic 2024). Confronto con un modello ingenuo.
  * Esporta la climatologia oraria (produzione attesa) usata dall'app per misurare il calo.
  * Gli scenari estremi vengono generati dal modello riaddestrato su tutti i 15 inverni.

Uso:  python addestra_generatore.py hourly_country.csv generatore_params.json
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

SRC, OUT = sys.argv[1], sys.argv[2]
W_EOL, W_SOL = 0.65, 0.35
C_TOT = 439.0
C_EOL, C_SOL = C_TOT * W_EOL, C_TOT * W_SOL
K, SOGLIA, MIN_GG = 3, 0.35, 2
N_INVERNI_SINT = 10_000
ANNI_TEST = [2022, 2023, 2024]

# ---------------------------------------------------------------- dati
h = pd.read_csv(SRC, parse_dates=["time"])
h["cf"] = W_EOL * h.cf_wind + W_SOL * h.cf_solar
h["doy"], h["ora"] = h.time.dt.dayofyear, h.time.dt.hour


def dist_doy(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 365 - d)


# climatologia oraria (doy × ora) per l'app: CF eolico e solare attesi, finestra ±7 giorni
clim_eol = np.zeros((366, 24)); clim_sol = np.zeros((366, 24))
for o in range(24):
    ho = h[h.ora == o]
    for k in range(1, 367):
        m = dist_doy(ho.doy.values, k) <= 7
        clim_eol[k - 1, o] = ho.cf_wind.values[m].mean()
        clim_sol[k - 1, o] = ho.cf_solar.values[m].mean()

d = h.set_index("time")[["cf", "t2m"]].resample("D").mean()
d["doy"] = d.index.dayofyear
clim_cf = np.array([d.cf[dist_doy(d.doy.values, k) <= 7].mean() for k in range(1, 367)])
clim_t = np.array([d.t2m[dist_doy(d.doy.values, k) <= 7].mean() for k in range(1, 367)])
d["r"] = d.cf / clim_cf[d.doy - 1]
d["ta"] = d.t2m - clim_t[d.doy - 1]

inverni = {}
for y in range(2010, 2025):
    w = d[(d.index >= f"{y}-11-01") & (d.index < f"{y+1}-02-01")]
    if len(w) == 92:
        inverni[y] = w
to_seq = lambda w: np.column_stack([np.log(np.clip(w.r.values, 0.02, None)), w.ta.values])


# ---------------------------------------------------------------- AR-HMM: EM / Baum-Welch
def fit_arhmm(seqs, seed=0, max_it=400):
    Xp = np.vstack([x[:-1] for x in seqs]); Xc = np.vstack([x[1:] for x in seqs])
    km = KMeans(K, n_init=10, random_state=seed).fit(Xc)
    phi = np.full((K, 2), 0.5); c = km.cluster_centers_ * (1 - phi)
    cov = np.array([np.cov(Xc[km.labels_ == k].T) + 1e-3 * np.eye(2) for k in range(K)])
    A = np.full((K, K), 0.1 / (K - 1)); np.fill_diagonal(A, 0.9); pi = np.full(K, 1 / K)
    prev = -np.inf
    for it in range(max_it):
        G_all, xi, pi_acc, ll = [], np.zeros((K, K)), np.zeros(K), 0.0
        for x in seqs:
            le = log_emis(x, c, phi, cov); T = len(le); lA = np.log(A)
            la = np.zeros((T, K)); lb = np.zeros((T, K)); la[0] = np.log(pi) + le[0]
            for t in range(1, T):
                la[t] = logsumexp(la[t - 1][:, None] + lA, axis=0) + le[t]
            for t in range(T - 2, -1, -1):
                lb[t] = logsumexp(lA + le[t + 1] + lb[t + 1], axis=1)
            L = logsumexp(la[-1]); ll += L
            g = np.exp(la + lb - L); G_all.append(g); pi_acc += g[0]
            for t in range(T - 1):
                xi += np.exp(la[t][:, None] + lA + le[t + 1] + lb[t + 1] - L)
        G = np.vstack(G_all)
        pi = pi_acc / pi_acc.sum(); A = xi / xi.sum(1, keepdims=True)
        for k in range(K):
            w = G[:, k]
            for j in range(2):
                Z = np.column_stack([np.ones(len(Xp)), Xp[:, j]])
                beta = np.linalg.solve(Z.T @ (w[:, None] * Z), Z.T @ (w * Xc[:, j]))
                c[k, j], phi[k, j] = beta[0], np.clip(beta[1], -0.98, 0.98)
            res = Xc - c[k] - phi[k] * Xp
            cov[k] = (w[:, None, None] * np.einsum("ni,nj->nij", res, res)).sum(0) / w.sum() + 1e-4 * np.eye(2)
        if ll - prev < 1e-4:
            break
        prev = ll
    mu = c / (1 - phi)
    o = np.argsort(mu[:, 0])
    return dict(c=c[o], phi=phi[o], cov=cov[o], mu=mu[o], A=A[np.ix_(o, o)], pi=pi[o], it=it + 1)


def log_emis(x, c, phi, cov):
    xp, xc = x[:-1], x[1:]
    return np.column_stack([multivariate_normal(np.zeros(2), cov[k]).logpdf(xc - c[k] - phi[k] * xp) for k in range(K)])


def loglik_per_giorno(m, seqs):
    tot, n = 0.0, 0
    for x in seqs:
        le = log_emis(x, m["c"], m["phi"], m["cov"]); lA = np.log(m["A"])
        la = np.log(m["pi"]) + le[0]
        for t in range(1, len(le)):
            la = logsumexp(la[:, None] + lA, axis=0) + le[t]
        tot += logsumexp(la); n += len(le)
    return tot / n


def campiona(m, n_inv, T=92, seed=7):
    rng = np.random.default_rng(seed)
    out_r, out_t = np.empty((n_inv, T)), np.empty((n_inv, T))
    Lc = np.array([np.linalg.cholesky(cv) for cv in m["cov"]])
    s = rng.choice(K, size=n_inv, p=m["pi"])
    x = m["mu"][s] + np.einsum("nij,nj->ni", Lc[s], rng.standard_normal((n_inv, 2)))
    for t in range(T):
        out_r[:, t], out_t[:, t] = np.exp(x[:, 0]), x[:, 1]
        u = rng.random(n_inv)
        s = np.minimum((u[:, None] > np.cumsum(m["A"][s], axis=1)).sum(1), K - 1)
        x = m["c"][s] + m["phi"][s] * x + np.einsum("nij,nj->ni", Lc[s], rng.standard_normal((n_inv, 2)))
    return out_r, out_t


def eventi(r, ta=None):
    ev, i, n = [], 0, len(r)
    while i < n:
        if r[i] < SOGLIA:
            j = i
            while j < n and r[j] < SOGLIA:
                j += 1
            if j - i >= MIN_GG:
                seg = r[i:j]
                ev.append({"durata_gg": j - i, "r_medio": float(seg.mean()), "severita": float((1 - seg).sum()),
                           "r": [round(float(v), 4) for v in seg],
                           "ta_medio": float(ta[i:j].mean()) if ta is not None else None, "inizio_gg": i})
            i = j
        else:
            i += 1
    return ev


def statistiche(r_mat, ta_mat):
    ev = [e for i in range(len(r_mat)) for e in eventi(r_mat[i], ta_mat[i])]
    dur = np.array([e["durata_gg"] for e in ev]) if ev else np.array([0])
    max_sev = np.array([max([e["severita"] for e in eventi(r_mat[i])], default=0.0) for i in range(len(r_mat))])
    corr = float(np.corrcoef(np.log(np.clip(r_mat.ravel(), 0.02, None)), ta_mat.ravel())[0, 1])
    return dict(eventi_per_inverno=len(ev) / len(r_mat), durata_media=float(dur.mean()), durata_max=int(dur.max()),
                corr=corr, max_sev=max_sev, ev=ev, dur=dur)


# ---------------------------------------------------------------- 1) validazione fuori campione
train = [to_seq(inverni[y]) for y in inverni if y not in ANNI_TEST]
test_y = [y for y in ANNI_TEST if y in inverni]
test = [to_seq(inverni[y]) for y in test_y]
m_tr = fit_arhmm(train)
# modello ingenuo: giorni indipendenti, gaussiana unica (nessun regime, nessuna persistenza)
Xtr = np.vstack(train); mu0, cov0 = Xtr.mean(0), np.cov(Xtr.T)
ll_naive = float(np.mean([multivariate_normal(mu0, cov0).logpdf(x[1:]).mean() for x in test]))
ll_hmm = float(loglik_per_giorno(m_tr, test))
r_s, t_s = campiona(m_tr, N_INVERNI_SINT, seed=11)
st_sint = statistiche(r_s, t_s)
r_te = np.array([inverni[y].r.values for y in test_y]); t_te = np.array([inverni[y].ta.values for y in test_y])
st_test = statistiche(r_te, t_te)
percentili_test = {f"{y}-{str(y+1)[-2:]}": float((st_sint["max_sev"] <= s).mean())
                   for y, s in zip(test_y, st_test["max_sev"])}
oos = dict(
    inverni_addestramento=f"2010-11 … {max(y for y in inverni if y not in ANNI_TEST)}-{str(max(y for y in inverni if y not in ANNI_TEST)+1)[-2:]}",
    inverni_test=[f"{y}-{str(y+1)[-2:]}" for y in test_y],
    loglik_giorno_hmm=ll_hmm, loglik_giorno_ingenuo=ll_naive,
    test=dict(eventi_per_inverno=st_test["eventi_per_inverno"], durata_media=st_test["durata_media"],
              durata_max=st_test["durata_max"], corr=st_test["corr"]),
    sintetico=dict(eventi_per_inverno=st_sint["eventi_per_inverno"], durata_media=st_sint["durata_media"],
                   durata_max=st_sint["durata_max"], corr=st_sint["corr"]),
    percentile_inverni_test=percentili_test,
)


# ---- previsione a 1-3 giorni sugli inverni di test (modello addestrato senza vederli)
def prob_sotto_soglia(m, x, k, n_mc=400, seed=3):
    """P(r < soglia) per ogni giorno t+k, dati i giorni osservati fino a t (filtro in avanti + Monte Carlo)."""
    rng = np.random.default_rng(seed)
    le = log_emis(x, m["c"], m["phi"], m["cov"]); lA = np.log(m["A"])
    Lc = np.array([np.linalg.cholesky(cv) for cv in m["cov"]])
    la = np.log(m["pi"]) + le[0]
    out = {}
    for t in range(1, len(x)):              # x[t] e' l'ultimo giorno osservato
        if t > 1:
            la = logsumexp(la[:, None] + lA, axis=0) + le[t - 1]
        if t + k >= len(x):
            continue
        p = np.exp(la - logsumexp(la))
        s = rng.choice(K, size=n_mc, p=p)
        xx = np.repeat(x[t][None, :], n_mc, axis=0)
        for _ in range(k):
            u = rng.random(n_mc)
            s = np.minimum((u[:, None] > np.cumsum(m["A"][s], axis=1)).sum(1), K - 1)
            xx = m["c"][s] + m["phi"][s] * xx + np.einsum("nij,nj->ni", Lc[s], rng.standard_normal((n_mc, 2)))
        out[t + k] = float((np.exp(xx[:, 0]) < SOGLIA).mean())
    return out


def metriche(pred, vero):
    pred, vero = np.asarray(pred, bool), np.asarray(vero, bool)
    tp, fp, fn = (pred & vero).sum(), (pred & ~vero).sum(), (~pred & vero).sum()
    return dict(accuratezza=float((pred == vero).mean()), precisione=float(tp / max(tp + fp, 1)),
                richiamo=float(tp / max(tp + fn, 1)), giorni_evento=int(vero.sum()), giorni=int(len(vero)))


prev = {}
for k in (1, 2, 3):
    P, V, Pers = [], [], []
    for x in test:
        pr = prob_sotto_soglia(m_tr, x, k)
        r = np.exp(x[:, 0])
        for d, p in pr.items():
            P.append(p >= 0.5); V.append(r[d] < SOGLIA); Pers.append(r[d - k] < SOGLIA)
    prev[str(k)] = dict(modello=metriche(P, V), persistenza=metriche(Pers, V))
oos["previsione"] = prev
print("FUORI CAMPIONE:", json.dumps(oos, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- 2) modello finale su tutti i 15 inverni
tutti = [to_seq(w) for w in inverni.values()]
m = fit_arhmm(tutti)
print(f"Modello finale: EM in {m['it']} iterazioni")
r_s, t_s = campiona(m, N_INVERNI_SINT, seed=7)
st_s = statistiche(r_s, t_s)
r_h = np.array([w.r.values for w in inverni.values()]); t_h = np.array([w.ta.values for w in inverni.values()])
st_h = statistiche(r_h, t_h)

ev_storici = []
for y, w in inverni.items():
    for e in eventi(w.r.values, w.ta.values):
        e["inverno"] = f"{y}-{str(y+1)[-2:]}"
        ev_storici.append(e)
peggiore = max(ev_storici, key=lambda e: e["severita"])


def evento_rappresentativo(periodo):
    liv = float(np.quantile(st_s["max_sev"], 1 - 1 / periodo))
    cand = [e for e in st_s["ev"] if abs(e["severita"] - liv) < 0.15 * liv]
    e = min(cand, key=lambda e: abs(e["severita"] - liv))
    return {**e, "periodo_ritorno": periodo, "livello_severita": liv}


scenari = {
    "storico_peggiore": {**peggiore, "etichetta": f"Peggiore storico ({peggiore['inverno']})",
                         "etichetta_en": f"Worst on record ({peggiore['inverno']})"},
    "sintetico_10": {**evento_rappresentativo(10), "etichetta": "Sintetico · 1 inverno su 10", "etichetta_en": "Synthetic · 1-in-10 winters"},
    "sintetico_100": {**evento_rappresentativo(100), "etichetta": "Sintetico · 1 inverno su 100", "etichetta_en": "Synthetic · 1-in-100 winters"},
    "sintetico_1000": {**evento_rappresentativo(1000), "etichetta": "Sintetico · 1 inverno su 1000", "etichetta_en": "Synthetic · 1-in-1000 winters"},
}
bins = list(range(2, 16))


def hist(v):
    c = np.array([(v == b).sum() if b < 15 else (v >= 15).sum() for b in bins], float)
    return (c / c.sum()).round(4).tolist()


nomi = [("Anticiclone (calma e buio)", "Anticyclone (calm and dark)"), ("Variabile", "Mixed"), ("Perturbato (ventoso)", "Stormy (windy)")]
params = {
    "versione": 3,
    "modello": "HMM autoregressivo a 3 regimi (EM/Baum-Welch) su log(rapporto CF) e anomalia di temperatura giornalieri, nov-gen 2010-2025",
    "quote": {"eolico": W_EOL, "solare": W_SOL}, "capacita_GW": {"eolico": C_EOL, "solare": C_SOL},
    "soglia_rapporto": SOGLIA, "min_giorni": MIN_GG,
    "regimi": [{"nome": nomi[k][0], "nome_en": nomi[k][1], "r_medio": float(np.exp(m["mu"][k, 0])),
                "t_anom": float(m["mu"][k, 1]), "durata_media_gg": float(1 / (1 - m["A"][k, k])),
                "phi": m["phi"][k].round(3).tolist()} for k in range(K)],
    "transizioni": m["A"].round(4).tolist(),
    "scenari": scenari,
    "validazione": {
        "bins_durata_gg": bins, "freq_durata_storico": hist(st_h["dur"]), "freq_durata_sintetico": hist(st_s["dur"]),
        "eventi_per_inverno_storico": st_h["eventi_per_inverno"], "eventi_per_inverno_sintetico": st_s["eventi_per_inverno"],
        "durata_max_storico": st_h["durata_max"], "durata_max_sintetico": st_s["durata_max"],
        "durata_media_storico": st_h["durata_media"], "durata_media_sintetico": st_s["durata_media"],
        "corr_produzione_temperatura_storico": st_h["corr"], "corr_produzione_temperatura_sintetico": st_s["corr"],
        "n_inverni_storici": len(inverni), "n_inverni_sintetici": N_INVERNI_SINT,
    },
    "fuori_campione": oos,
    "climatologia": {"nota": "CF attesi per giorno dell'anno (1-366) e ora, media 2010-2025 ±7 gg",
                     "cf_eolico": clim_eol.round(4).tolist(), "cf_solare": clim_sol.round(4).tolist()},
}
json.dump(params, open(OUT, "w"), ensure_ascii=False)
for k, s in scenari.items():
    print(f"{s['etichetta']:35s} durata {s['durata_gg']} gg  r medio {s['r_medio']:.2f}  severità {s['severita']:.2f}")
