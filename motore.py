"""
Motore del Dunkelflaute Early Warning System (senza dipendenze da Streamlit: testabile da solo).

Catena del prodotto
  VEDERE    -> scenari di evento (dataset, peggiore storico, sintetici dal generatore AI) + ensemble di previsione
  MISURARE  -> digital twin orario: rinnovabili, accumulo, fossile, import, tagli -> GWh di consumi necessari non serviti
  DECIDERE  -> catalogo di azioni con scadenza (T-n giorni): il preavviso X decide quali sono ancora possibili
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

COLONNE_RICHIESTE = [
    "tempo_ore", "potenza_consumata", "potenza_consumata_necessaria", "potenza_consumata_non_essenziale",
    "potenza_generata_solare", "potenza_generata_eolica", "potenza_generata_non_rinnovabili", "scarto",
]

# ---------------------------------------------------------------------------------------------------------
# Ipotesi di sistema (valori di default coerenti con il dataset nov 2029 - gen 2030)
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Ipotesi:
    E_max: float = 80.0             # GWh di accumulo (batterie + pompaggi)
    P_acc: float = 20.0             # GW di potenza di carica/scarica
    eta: float = 0.90               # rendimento di carica
    F_max: float = 40.0             # GW programmabili di riserva (come nel dataset)
    scorta_base: float = 4000.0     # GWh di combustibile per l'evento senza azioni (~100 h a 40 GW)
    soc_senza_precarica: float = 0.5  # senza preavviso l'accumulo lavora sul mercato: a T0 al massimo metà pieno
    eff_taglio_programmato: float = 0.55  # quota dei non essenziali che si può tagliare se programmato
    eff_taglio_improvviso: float = 0.50   # ... se improvviso (distacchi a rotazione)
    # costi (M€ per GWh = migliaia di €/MWh)
    c_fossile: float = 0.12
    c_import: float = 0.25
    c_taglio_prog: float = 0.10
    c_taglio_impr: float = 0.50
    c_non_serviti: float = 5.0


# ---------------------------------------------------------------------------------------------------------
# Catalogo azioni (da azioni_mitigazione_preavviso.md). scadenza = giorni di preavviso minimi
# ---------------------------------------------------------------------------------------------------------
AZIONI = [
    dict(id="centrali", fonte='Ipotesi: +3 GW da centrali tenute in riserva e manutenzioni rinviate, +4.000 GWh di combustibile (circa 100 ore a 40 GW). Serve molto anticipo: la riserva di lignite tedesca doveva essere pronta entro 10 giorni dalla chiamata, e un carico di GNL spot richiede 10–15 giorni di viaggio. Fonti: oenergetice.cz (2023) sulla riserva di sicurezza; DIW Wochenbericht 49/2024 sulla riserva di capacità (circa 2 GW fuori mercato).', fonte_en="Assumption: +3 GW from plants kept in reserve and postponed maintenance, +4,000 GWh of fuel (about 100 hours at 40 GW). It needs long notice: Germany's lignite reserve had to be ready within 10 days of a call, and a spot LNG cargo takes 10–15 days to arrive. Sources: oenergetice.cz (2023) on the security reserve; DIW Wochenbericht 49/2024 on the capacity reserve (about 2 GW outside the market).", scadenza=10, costo_fisso=70.0, serie="Fossile",
         leve=[("scorta_extra", 4000.0), ("f_extra", 3.0)],
         nome="Centrali di riserva: combustibile, manutenzioni rinviate", nome_en="Reserve plants: fuel, postponed maintenance",
         effetto="+3 GW di centrali, +4.000 GWh di combustibile", effetto_en="+3 GW of plants, +4,000 GWh of fuel"),
    dict(id="import", fonte='Ipotesi prudente: +3 GW da contratti a termine con i paesi vicini (nucleare francese, idroelettrico nordico e alpino). Il valore è basso di proposito: le interconnessioni hanno un limite e una Dunkelflaute spesso colpisce anche i paesi vicini, che hanno meno energia da vendere. I contratti settimanali vanno chiusi con circa una settimana di anticipo.', fonte_en='Cautious assumption: +3 GW from forward contracts with neighbours (French nuclear, Nordic and Alpine hydro). The value is deliberately low: interconnectors have a limit and a Dunkelflaute often hits neighbouring countries too, leaving them less to sell. Weekly contracts must be closed about a week ahead.', scadenza=7, costo_fisso=20.0, serie="Import",
         leve=[("import", 3.0)],
         nome="Contratti di import dai paesi vicini", nome_en="Import contracts with neighbouring countries",
         effetto="+3 GW di import", effetto_en="+3 GW of imports"),
    dict(id="industria", fonte="Ipotesi: −3 GW di domanda industriale durante l'evento, spostando fermate e manutenzioni di impianti energivori (alluminio, cloro-soda, acciaio da forno elettrico), che richiedono 3–7 giorni di pianificazione. Riferimento: il programma tedesco dei carichi interrompibili (AbLaV, 2012–2022) contrattava fino a 1,5 GW distaccabili in 15 minuti o istantaneamente; con accordi pianificati si va oltre. Fonti: netztransparenz.de; Commissione europea, decisione sull'AbLaV.", fonte_en="Assumption: −3 GW of industrial demand during the event, by moving shutdowns and maintenance of energy-intensive plants (aluminium, chlor-alkali, electric-arc steel), which need 3–7 days of planning. Reference: Germany's interruptible-load scheme (AbLaV, 2012–2022) contracted up to 1.5 GW that could be shed within 15 minutes or instantly; planned agreements go further. Sources: netztransparenz.de; European Commission decision on AbLaV.", scadenza=5, costo_fisso=25.0, serie="Industria",
         leve=[("riduzione_gw", 3.0)],
         nome="Accordi con l'industria per ridurre la domanda", nome_en="Agreements with industry to cut demand",
         effetto="−3 GW di domanda industriale", effetto_en="−3 GW of industrial demand"),
    dict(id="accumuli", fonte="Ipotesi: accumuli di 80 GWh / 20 GW (batterie e pompaggi in Germania al 2030, valore regolabile). Senza allerta lavorano sul mercato e arrivano all'evento al massimo pieni a metà; con l'allerta si caricano del tutto nei giorni prima, sfruttando il surplus, e durante l'evento coprono solo la parte di domanda essenziale che i tagli non possono proteggere. Contesto: le batterie in Europa sono previste in crescita da 47 a 163 GW al 2030 (Wood Mackenzie, 2026).", fonte_en='Assumption: 80 GWh / 20 GW of storage (batteries and pumped hydro in Germany by 2030, adjustable). Without a warning it trades on the market and reaches the event at most half full; with a warning it is fully charged in the days before, using surplus, and during the event it only covers the essential demand that cuts cannot protect. Context: European battery capacity is expected to grow from 47 to 163 GW by 2030 (Wood Mackenzie, 2026).', scadenza=3, costo_fisso=5.0, serie="Accumulo",
         leve=[("precarica", 1.0), ("riserva_nec", 1.0)],
         nome="Caricare gli accumuli e riservarli agli essenziali", nome_en="Charge storage and reserve it for essentials",
         effetto="accumuli pieni, usati solo per la domanda essenziale", effetto_en="storage full, used only for essential demand"),
    dict(id="comunita", fonte="Ipotesi: 2 GW di flessibilità di giorno (circa 4 milioni di famiglie × 0,5 kW tra pompe di calore, auto elettriche ed elettrodomestici), metà spostata prima dell'evento; −1% di domanda dall'appello al risparmio, sul modello delle campagne EcoWatt in Francia; tagli ai non essenziali programmati, un po' più mirati di quelli improvvisi (55% contro 50% della domanda non essenziale). Basta un giorno di anticipo: è un messaggio a cittadini, comunità energetiche e aggregatori.", fonte_en="Assumption: 2 GW of daytime flexibility (about 4 million households × 0.5 kW across heat pumps, EVs and appliances), half of it brought forward before the event; −1% demand from a saving appeal, modelled on France's EcoWatt campaigns; planned non-essential cuts, slightly better targeted than sudden ones (55% vs 50% of non-essential demand). One day's notice is enough: it is a message to citizens, energy communities and aggregators.", scadenza=1, costo_fisso=5.0, serie="ComunitaTagli",
         leve=[("flex_gw", 2.0), ("risparmio_pct", 0.01), ("tagli_programmati", 1.0)],
         nome="Allerta a comunità e cittadini, tagli programmati", nome_en="Alert communities and citizens, planned cuts",
         effetto="−2 GW di giorno, −1% di domanda, tagli mirati", effetto_en="−2 GW in daytime, −1% demand, targeted cuts"),
]


@dataclass(frozen=True)
class Leve:
    scorta_extra: float = 0.0
    f_extra: float = 0.0
    import_gw: float = 0.0
    idro: float = 0.0
    riduzione_gw: float = 0.0
    riduzione_pct: float = 0.0
    flex_gw: float = 0.0
    precarica: bool = False
    riserva_nec: bool = False
    tagli_programmati: bool = False
    costo_fisso: float = 0.0


def azioni_attive(preavviso: int, piano: bool = True):
    return [a for a in AZIONI if piano and preavviso >= a["scadenza"]]


def leve_da_preavviso(preavviso: int, piano: bool = True) -> Leve:
    lv = dict(scorta_extra=0.0, f_extra=0.0, import_gw=0.0, idro=0.0, riduzione_gw=0.0, riduzione_pct=0.0, flex_gw=0.0,
              precarica=False, riserva_nec=False, tagli_programmati=False, costo_fisso=0.0)
    for a in azioni_attive(preavviso, piano):
        for k, v in a["leve"]:
            if k in ("precarica", "riserva_nec"):
                lv[k] = True
            elif k == "tagli_programmati":
                lv[k] = True
            elif k == "risparmio_pct":
                lv["riduzione_pct"] += v
            elif k == "import":
                lv["import_gw"] += v
            else:
                lv[k] += v
        lv["costo_fisso"] += a["costo_fisso"]
    return Leve(**lv)


# ---------------------------------------------------------------------------------------------------------
# Preparazione del dataset
# ---------------------------------------------------------------------------------------------------------
def prepara(df: pd.DataFrame, params: dict | None = None) -> dict:
    """Valida il CSV, individua l'evento e ricava la produzione rinnovabile 'attesa' ora per ora.
    Con i parametri del generatore l'attesa è la climatologia 2010-2025 (stessa definizione del generatore);
    senza, è la mediana delle ore fuori evento alla stessa ora del giorno (±10 giorni)."""
    mancanti = [c for c in COLONNE_RICHIESTE if c not in df.columns]
    if mancanti:
        raise ValueError(f"Mancano le colonne: {mancanti}")
    d = df.copy().reset_index(drop=True)
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"])
    else:
        d["timestamp"] = pd.Timestamp("2029-11-01") + pd.to_timedelta(d["tempo_ore"], unit="h")
    ren = (d.potenza_generata_solare + d.potenza_generata_eolica).values

    if "dunkelflaute" in d.columns and d["dunkelflaute"].sum() > 0:
        flag = d["dunkelflaute"].values == 1
    else:  # regola di riserva: rinnovabili sotto il 35% della domanda per almeno 48 h
        flag = ren < 0.35 * d.potenza_consumata.values
    # blocco più lungo
    best, run, best_end = 0, 0, 0
    for i, v in enumerate(flag):
        run = run + 1 if v else 0
        if run > best:
            best, best_end = run, i
    if best < 24:
        raise ValueError("Nel file non c'è un evento di carenza rinnovabile di almeno 24 ore.")
    t0, t1 = best_end - best + 1, best_end + 1

    ora = d.timestamp.dt.hour.values
    if params and "climatologia" in params:
        doy = d.timestamp.dt.dayofyear.values - 1
        ce, cs = np.asarray(params["climatologia"]["cf_eolico"]), np.asarray(params["climatologia"]["cf_solare"])
        base_eol = params["capacita_GW"]["eolico"] * ce[doy, ora]
        base_sol = params["capacita_GW"]["solare"] * cs[doy, ora]
        fonte_attesa = "climatologia"
    else:
        base_sol = base_eol = None
        fonte_attesa = "mediana"
    if base_sol is None:
        escl = np.zeros(len(d), bool); escl[max(0, t0 - 24):t1 + 24] = True
        base_sol, base_eol = np.empty(len(d)), np.empty(len(d))
        sol, eol = d.potenza_generata_solare.values, d.potenza_generata_eolica.values
        idx = np.arange(len(d))
        for i in range(len(d)):
            m = (~escl) & (ora == ora[i]) & (np.abs(idx - i) <= 240)
            base_sol[i], base_eol[i] = np.median(sol[m]), np.median(eol[m])
    # aumento della domanda durante l'evento (freddo) rispetto ai 3 giorni prima e dopo
    intorno = np.r_[max(0, t0 - 72):t0, t1:min(len(d), t1 + 72)]
    uplift = d.potenza_consumata.values[t0:t1].mean() / d.potenza_consumata.values[intorno].mean()
    rapporto = ren[t0:t1].sum() / (base_sol[t0:t1] + base_eol[t0:t1]).sum()
    return dict(df=d, t0=t0, t1=t1, base_sol=base_sol, base_eol=base_eol,
                uplift_domanda=float(max(1.0, uplift)), rapporto_dataset=float(rapporto), fonte_attesa=fonte_attesa)


def scenario_dataset(prep: dict) -> dict:
    d = prep["df"]
    n_gg = (prep["t1"] - prep["t0"]) / 24
    return dict(nome="dataset", etichetta=f"Evento del dataset ({n_gg:.0f} gg)", etichetta_en=f"Dataset event ({n_gg:.0f} days)",
                r=[prep["rapporto_dataset"]] * int(round(n_gg)), dataset=True)


def applica_scenario(prep: dict, scen: dict, shift_h: int = 0, rampa_h: int = 12) -> dict:
    """Sostituisce la finestra dell'evento con la sequenza giornaliera di rapporti r dello scenario."""
    d = prep["df"].copy()
    t0, t1 = prep["t0"], prep["t1"]
    if scen.get("dataset") and shift_h == 0:
        flag = np.zeros(len(d), bool); flag[t0:t1] = True
        d["in_evento"] = flag
        return dict(df=d, t0=t0, t1=t1)
    n = len(d)
    sol, eol, dom = d.potenza_generata_solare.values.copy(), d.potenza_generata_eolica.values.copy(), d.potenza_consumata.values.copy()
    nec_q = (d.potenza_consumata_necessaria / d.potenza_consumata).values
    # 1) ripristina la finestra originale alla produzione attesa e alla domanda senza effetto freddo
    sol[t0:t1], eol[t0:t1] = prep["base_sol"][t0:t1], prep["base_eol"][t0:t1]
    dom[t0:t1] = dom[t0:t1] / prep["uplift_domanda"]
    # 2) nuovo evento
    r_h = np.repeat(np.asarray(scen["r"], float), 24)
    s0 = int(np.clip(t0 + shift_h, 0, n - 1)); s1 = int(min(n, s0 + len(r_h)))
    fattore = np.ones(n)
    fattore[s0:s1] = r_h[: s1 - s0]
    for j in range(1, rampa_h + 1):  # rampe di ingresso e uscita
        a = j / (rampa_h + 1)
        if s0 - j >= 0:
            fattore[s0 - j] = a + (1 - a) * fattore[s0]
        if s1 - 1 + j < n:
            fattore[s1 - 1 + j] = a + (1 - a) * fattore[s1 - 1]
    zona = np.zeros(n, bool); zona[max(0, s0 - rampa_h):min(n, s1 + rampa_h)] = True
    sol = np.where(zona, prep["base_sol"] * fattore, sol)
    eol = np.where(zona, prep["base_eol"] * fattore, eol)
    dom[s0:s1] *= prep["uplift_domanda"]
    d["potenza_generata_solare"], d["potenza_generata_eolica"] = sol, eol
    d["potenza_consumata"] = dom
    d["potenza_consumata_necessaria"] = dom * nec_q
    d["potenza_consumata_non_essenziale"] = dom * (1 - nec_q)
    flag = np.zeros(n, bool); flag[s0:s1] = True
    d["in_evento"] = flag
    return dict(df=d, t0=s0, t1=s1)


# ---------------------------------------------------------------------------------------------------------
# Digital twin orario
# ---------------------------------------------------------------------------------------------------------
def simula(sc: dict, ip: Ipotesi, lv: Leve, dettaglio: bool = True, finestra: tuple | None = None, soc_iniz: float | None = None):
    d, t0, t1 = sc["df"], sc["t0"], sc["t1"]
    D = d.potenza_consumata.values; Dn = d.potenza_consumata_necessaria.values; Dnn = d.potenza_consumata_non_essenziale.values
    S = d.potenza_generata_solare.values; W = d.potenza_generata_eolica.values
    a, b = finestra if finestra else (0, len(d))
    P = ip.P_acc; E = ip.E_max
    soc = E * 0.5 if soc_iniz is None else soc_iniz
    scorta = ip.scorta_base + lv.scorta_extra
    idro, P_idro = lv.idro, 3.0
    eff_taglio = ip.eff_taglio_programmato if lv.tagli_programmati else ip.eff_taglio_improvviso

    cols = {k: np.zeros(b - a) for k in ["Accumulo", "Fossile", "Import", "Idro", "Riduzione", "Industria", "Risparmio", "Comunita",
                                          "Taglio", "NonServiti", "Scarto", "SoC", "Carica", "NonServitaFuori", "DomEff"]}
    ore = d.timestamp.dt.hour.values
    flex_ore = (ore >= 7) & (ore <= 22)
    # metà dell'energia spostata dalle comunità viene consumata nelle 24 h prima dell'evento (rimbalzo)
    rimbalzo = 0.5 * lv.flex_gw * flex_ore[t0:t1].sum() / 24 if lv.flex_gw > 0 else 0.0
    soc_t0 = None
    for i in range(a, b):
        k = i - a
        ren = S[i] + W[i]
        ev = t0 <= i < t1
        if i == t0:
            soc = E if lv.precarica else min(soc, E * ip.soc_senza_precarica)
            soc_t0 = soc
        dom, dn, dnn = D[i], Dn[i], Dnn[i]
        if t0 - 24 <= i < t0 and rimbalzo:
            dom += rimbalzo; dnn += rimbalzo
            cols["Comunita"][k] = -rimbalzo
        if ev:  # riduzioni programmate della domanda non essenziale
            fx = min(dnn, lv.flex_gw) if flex_ore[i] else 0.0
            dom -= fx; dnn -= fx
            cols["Comunita"][k] = fx
            ind = min(dnn, lv.riduzione_gw); dom -= ind; dnn -= ind
            rs = min(dnn, lv.riduzione_pct * dom); dom -= rs; dnn -= rs
            cols["Industria"][k], cols["Risparmio"][k] = ind, rs
            cols["Riduzione"][k] = ind + rs + fx
        imp = lv.import_gw if ev else 0.0
        offerta = ren + imp
        buco = dom - offerta
        if buco <= 0:
            surplus = -buco
            imp_usato = max(0.0, imp - surplus)  # l'import non serve se c'è surplus
            cols["Import"][k] = imp_usato
            ch = min(surplus, P, (E - soc) / ip.eta)
            soc += ip.eta * ch
            cols["Carica"][k] = ch
            cols["Scarto"][k] = max(0.0, surplus - ch)
        else:
            cols["Import"][k] = imp
            res = buco
            if ev and lv.riserva_nec:
                # strategia del piano: prima fossile e idro, l'accumulo solo per il buco sui necessari
                fo = min(res, ip.F_max, scorta); scorta -= fo; res -= fo
                fx = min(res, lv.f_extra); fo += fx; res -= fx      # centrali extra: combustibile proprio
                hy = min(res, P_idro, idro); idro -= hy; res -= hy
                buco_nec = max(0.0, res - dnn * eff_taglio)        # parte che i tagli non possono assorbire
                sc_ = min(buco_nec, P, soc); soc -= sc_; res -= sc_
            else:
                # comportamento "di mercato": l'accumulo copre subito il buco, poi il fossile
                sc_ = min(res, P, soc); soc -= sc_; res -= sc_
                if ev:
                    fo = min(res, ip.F_max, scorta); scorta -= fo; res -= fo
                    fx = min(res, lv.f_extra); fo += fx; res -= fx
                    hy = min(res, P_idro, idro); idro -= hy; res -= hy
                else:
                    fo = min(res, ip.F_max); res -= fo; hy = 0.0
            cols["Accumulo"][k], cols["Fossile"][k], cols["Idro"][k] = sc_, fo, hy
            if ev:
                tg = min(res, dnn * eff_taglio); res -= tg
                cols["Taglio"][k] = tg
                cols["NonServiti"][k] = max(0.0, res)
            else:
                cols["NonServitaFuori"][k] = max(0.0, res)
        cols["SoC"][k] = soc
        cols["DomEff"][k] = dom - cols["Taglio"][k]

    ev_m = (np.arange(a, b) >= t0) & (np.arange(a, b) < t1)
    ns = cols["NonServiti"]
    c_tg = ip.c_taglio_prog if lv.tagli_programmati else ip.c_taglio_impr
    voci = {  # voce: (GWh, M€ per GWh, M€)
        "centrali": (cols["Fossile"][ev_m].sum(), ip.c_fossile),
        "import": (cols["Import"][ev_m].sum(), ip.c_import),
        "riduzioni": (cols["Riduzione"][ev_m].sum(), ip.c_taglio_prog),
        "tagli": (cols["Taglio"][ev_m].sum(), c_tg),
        "non_serviti": (ns.sum(), ip.c_non_serviti),
    }
    costo_voci = {k: dict(GWh=float(g), prezzo=float(p), Meuro=float(g * p)) for k, (g, p) in voci.items()}
    costo_voci["azioni"] = dict(GWh=0.0, prezzo=0.0, Meuro=float(lv.costo_fisso))
    costo = sum(v["Meuro"] for v in costo_voci.values())
    buco_ev = np.maximum(0, D[t0:t1] - S[t0:t1] - W[t0:t1])
    kpi = dict(
        nec_non_serviti_GWh=float(ns.sum()),
        ore_nec_scoperti=int((ns > 0.1).sum()),
        picco_nec_scoperti_GW=float(ns.max()) if len(ns) else 0.0,
        buco_GWh=float(buco_ev.sum()), picco_buco_GW=float(buco_ev.max()) if len(buco_ev) else 0.0,
        accumulo_GWh=float(cols["Accumulo"][ev_m].sum()), soc_t0_GWh=float(soc_t0 or 0.0),
        fossile_GWh=float(cols["Fossile"][ev_m].sum()), import_GWh=float(cols["Import"][ev_m].sum()),
        idro_GWh=float(cols["Idro"][ev_m].sum()),
        riduzione_GWh=float(cols["Riduzione"][ev_m].sum()), tagli_GWh=float(cols["Taglio"][ev_m].sum()),
        comunita_GWh=float(np.clip(cols["Comunita"][ev_m], 0, None).sum()),
        comunita_picco_GW=float(cols["Comunita"][ev_m].max()) if ev_m.any() else 0.0,
        co2_Mt=float(cols["Fossile"][ev_m].sum() * 0.40 / 1000),
        costo_Meuro=float(costo), costo_voci=costo_voci, durata_h=int(t1 - t0),
        scarto_GWh=float(cols["Scarto"].sum()),
    )
    if not dettaglio:
        return None, kpi
    out = d.iloc[a:b][["tempo_ore", "timestamp", "potenza_consumata", "potenza_consumata_necessaria",
                       "potenza_generata_solare", "potenza_generata_eolica"]].copy()
    out.columns = ["tempo_ore", "timestamp", "Domanda_Tot", "Domanda_Nec", "Solare", "Eolico"]
    for k, v in cols.items():
        out[k] = v
    out["in_evento"] = ev_m
    return out, kpi


# ---------------------------------------------------------------------------------------------------------
# Ensemble di previsione: incertezza che cresce con l'anticipo
# ---------------------------------------------------------------------------------------------------------
def membri_ensemble(scen: dict, preavviso: int, n: int = 60, seed: int = 0, soglia: float = 0.35, min_gg: int = 2,
                    tau_giorni: float = 16.0):
    """Scenari di previsione a T-preavviso.
    Con probabilità skill = exp(-preavviso/tau) uno scenario 'vede' l'evento (perturbato in intensità, durata e arrivo,
    con incertezza crescente con l'anticipo); altrimenti è un inverno normale (la previsione ha perso informazione).
    Gli stessi numeri casuali per ogni preavviso rendono la probabilità coerente: sale man mano che l'evento si avvicina."""
    rng = np.random.default_rng(seed)
    r = np.asarray(scen["r"], float)
    skill = float(np.exp(-preavviso / tau_giorni))
    sig_b, sig_d, sig_L, sig_t = 0.08 * preavviso, 0.05 * preavviso, 0.04 * preavviso, 4.0 * preavviso
    out = []
    for m in range(n):
        u, z_b, z_L, z_t = rng.random(), rng.standard_normal(), rng.standard_normal(), rng.standard_normal()
        z_d = rng.standard_normal(200)
        z_c = rng.standard_normal(200)
        if u < skill:   # lo scenario vede l'evento
            L = max(1, int(round(len(r) * np.exp(sig_L * z_L))))
            base = np.interp(np.linspace(0, len(r) - 1, L), np.arange(len(r)), r)
            rm = np.clip(base * np.exp(sig_b * z_b + sig_d * z_d[:L]), 0.02, 2.0)
        else:           # inverno normale: produzione attorno all'attesa, giorni correlati
            L = len(r)
            x, lr = 0.0, []
            for k in range(L):
                x = 0.6 * x + 0.56 * z_c[k]
                lr.append(x)
            rm = np.clip(np.exp(np.array(lr)), 0.02, 2.0)
        run, best = 0, 0
        for v in rm < soglia:
            run = run + 1 if v else 0; best = max(best, run)
        out.append(dict(r=rm.tolist(), shift_h=int(round(sig_t * z_t)), evento=best >= min_gg, durata_gg=L))
    return out


def ensemble(prep, scen, ip, preavviso, n=60, seed=0, piano=True, margine_h=504):
    """Probabilità di Dunkelflaute e rischio sui necessari secondo la previsione a T−preavviso."""
    mem = membri_ensemble(scen, preavviso, n, seed)
    lv = leve_da_preavviso(preavviso, piano)
    sc_true = applica_scenario(prep, scen)
    a = max(0, prep["t0"] - margine_h)
    soc_a = None
    if a > 0:
        dd, _ = simula(sc_true, ip, lv, dettaglio=True, finestra=(0, a))
        soc_a = float(dd["SoC"].iloc[-1])
    ns, ren_tr = [], []
    for m in mem:
        sc = applica_scenario(prep, dict(r=m["r"]), shift_h=m["shift_h"])
        b = min(len(sc["df"]), sc["t1"] + margine_h)
        dd, k = simula(sc, ip, lv, dettaglio=True, finestra=(a, b), soc_iniz=soc_a)
        ns.append(k["nec_non_serviti_GWh"])
        ren_tr.append((dd["Solare"] + dd["Eolico"]).values)
    L = min(len(x) for x in ren_tr)
    R = np.array([x[:L] for x in ren_tr])
    return dict(
        prob_evento=float(np.mean([m["evento"] for m in mem])),
        prob_nec_scoperti=float(np.mean(np.array(ns) > 1.0)),
        ns_p10=float(np.percentile(ns, 10)), ns_p50=float(np.percentile(ns, 50)), ns_p90=float(np.percentile(ns, 90)),
        durata_mediana_gg=float(np.median([m["durata_gg"] for m in mem if m["evento"]] or [len(scen["r"])])),
        tempo=prep["df"].tempo_ore.values[a:a + L],
        ren_p25=np.percentile(R, 25, axis=0), ren_p50=np.percentile(R, 50, axis=0), ren_p75=np.percentile(R, 75, axis=0),
    )


# ---------------------------------------------------------------------------------------------------------
# Analisi per la decisione
# ---------------------------------------------------------------------------------------------------------
def curva_preavviso(prep, scen, ip, giorni=range(0, 15)):
    sc = applica_scenario(prep, scen)
    righe = []
    for x in giorni:
        _, k = simula(sc, ip, leve_da_preavviso(x), dettaglio=False)
        righe.append(dict(preavviso=x, nec_non_serviti_GWh=k["nec_non_serviti_GWh"],
                          ore_nec_scoperti=k["ore_nec_scoperti"], costo_Meuro=k["costo_Meuro"],
                          n_azioni=len(azioni_attive(x))))
    return pd.DataFrame(righe)


def accumulo_necessario(prep, scen, ip, preavviso=14, tol=1.0, e_max_lim=30000.0):
    """GWh minimi di accumulo (a pari durata E/P) per azzerare i necessari scoperti con il piano a T−preavviso."""
    sc = applica_scenario(prep, scen)
    lv = leve_da_preavviso(preavviso)
    durata = ip.E_max / ip.P_acc

    def ns(E):
        _, k = simula(sc, replace(ip, E_max=E, P_acc=max(E / durata, 1e-6)), lv, dettaglio=False)
        return k["nec_non_serviti_GWh"]

    if ns(0.0) <= tol:
        return 0.0
    if ns(e_max_lim) > tol:
        return float("inf")
    lo, hi = 0.0, e_max_lim
    while hi - lo > 10:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ns(mid) <= tol else (mid, hi)
    return hi


def scenari_da_parametri(params: dict) -> dict:
    out = {}
    for k, s in params["scenari"].items():
        out[k] = dict(nome=k, etichetta=s["etichetta"], etichetta_en=s.get("etichetta_en", s["etichetta"]), r=s["r"], dataset=False,
                      durata_gg=s["durata_gg"], r_medio=s["r_medio"])
    return out


def calo_produzione(prep: dict, sc: dict) -> float:
    """Calo di eolico + solare durante l'evento rispetto alla produzione attesa."""
    d, t0, t1 = sc["df"], sc["t0"], sc["t1"]
    ren = (d.potenza_generata_solare + d.potenza_generata_eolica).values[t0:t1].sum()
    att = (prep["base_sol"] + prep["base_eol"])[t0:t1].sum()
    return float(1 - ren / att)


CASI_VERIFICA = {
    "nov_2024": ("Novembre 2024", "November 2024"),
    "dic_2024": ("Dicembre 2024", "December 2024"),
    "natale_2024": ("Natale 2024", "Christmas 2024"),
}


def verifica_casi(df_ver: pd.DataFrame, params: dict | None, ip: Ipotesi, preavviso: int) -> pd.DataFrame:
    """Rigioca la rete 2030 sul meteo e la domanda reali del 2024: senza piano vs piano a T-preavviso."""
    righe = []
    for caso, g in df_ver.groupby("caso", sort=False):
        prep = prepara(g.drop(columns=["caso"]).reset_index(drop=True), params)
        sc = applica_scenario(prep, scenario_dataset(prep))
        _, k0 = simula(sc, ip, leve_da_preavviso(0, False), dettaglio=False)
        _, k1 = simula(sc, ip, leve_da_preavviso(preavviso, True), dettaglio=False)
        ts = prep["df"].timestamp
        righe.append(dict(caso=caso, inizio=ts.iloc[prep["t0"]], fine=ts.iloc[prep["t1"] - 1], durata_h=prep["t1"] - prep["t0"],
                          calo=calo_produzione(prep, sc), buco_GWh=k0["buco_GWh"],
                          ns_senza=k0["nec_non_serviti_GWh"], ns_con=k1["nec_non_serviti_GWh"],
                          ore_senza=k0["ore_nec_scoperti"], ore_con=k1["ore_nec_scoperti"],
                          costo_senza=k0["costo_Meuro"], costo_con=k1["costo_Meuro"], comunita_GWh=k1["comunita_GWh"]))
    return pd.DataFrame(righe)
