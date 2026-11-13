"""
ШАГ 7 (сезоны, риски перехода на версию 3). Четыре проверки сезонной модели (step7_seasons.py):
  comp   — летние частоты конкурентов на курортных парах: летом рейсы других авиакомпаний на паре с AER/MRV/KGD/MCX
           растут в 1 + доля x (resort_summer_ratio - 1) раз (доля 0,5 и 1; 1 + рейсы курортов в Москву);
  noise  — шум жадного алгоритма: каждый ключевой сценарий 1 раз обычным алгоритмом и NOISE_RUNS раз случайным
           (на каждом шаге случайный ход из тех, что не хуже лучшего на 10%, params: greedy_random);
  azimut — ответ «Азимута» (A4) в сезонной модели: добавляет столько же рейсов, сколько Red Wings, на парах, где летает,
           и/или снижает там цену на 15%; заранее ставит Минводы-Уфа ×11 (Red Wings раскладывает сеть заново);
           39 проверок step7_seasons_robust sens при +60% для сетей с ответом «Азимута»;
  ratios — чувствительность к отношению лето/зима Калининграда и Махачкалы (±30%, 1,0 и 2,0).

Запуск: python model/step7_seasons_checks.py comp|noise|noise_nomrv|azimut|ratios   (после step1 и step2; ~10-40 мин каждая на 2 ядрах)
Выход:  results/seasonal_check_competitors.csv, results/seasonal_check_noise.csv (+ _routes.csv),
        results/seasonal_check_noise_nomrv*.csv (прямые без Минвод-Уфы, для сравнения с гибридом без неё),
        results/seasonal_check_azimut.csv (+ _sens.csv), results/seasonal_check_ratios.csv
Параметры модели после расчёта возвращаются; при значениях по умолчанию расчёт совпадает со step7_seasons.py.
"""
import copy
import sys
import time
from multiprocessing import Pool

import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc
import step6_sensitivity as st6
import step7_seasons_robust as rb

N = 4
VARIANTS = {"прямые": ("прямые", False), "хаб в Уфе": ("хаб", False), "гибрид с Уфой": ("гибрид", False),
            "гибрид без Минвод-Уфы": ("гибрид", True), "прямые без Минвод-Уфы": ("прямые", True)}
NOISE_RUNS = 20
NOISE_TOL = 0.10
RIVAL = "A4"
CUT = 0.85                      # цена «Азимута» -15%
M = None


def with_params(ch):
    saved = {k: copy.deepcopy(P[k]) for k in ch}
    P.update(ch)
    return saved


def legs_str(net, Mx):
    return rb.st6_legs(net, Mx)


def run_job(job):
    """job = (проверка, метка, выручка, бортов, вариант, изменения параметров, сид или None)."""
    global M
    if M is None:
        M = sc.setup()
    kind, label, fare, n, variant, pch, seed = job
    t = time.time()
    mode, no_mrv = VARIANTS[variant]
    pch = dict(pch)
    if seed is not None:
        pch["greedy_random"] = {"seed": seed, "tol": NOISE_TOL}
    saved = with_params(pch)
    case = {**P["cases"]["базовый"], "fare": fare}
    Mx = dict(M)
    if no_mrv:
        Mx["direct_candidates"] = M["direct_candidates"] - {("MRV", "UFA")}
    nets, log = rb.greedy_net(Mx, mode, n, case, no_mrv=no_mrv)
    P["greedy_random"] = None
    prof, idle, bs = rb.season_result(nets, sc.base_network(Mx, "прямые"), Mx, case)
    for k, v in saved.items():
        P[k] = v
    row = {"проверка": kind, "условие": label, "выручка": fare, "бортов_дано": n, "вариант": variant,
           "сид": seed if seed is not None else "", "прибыль_млрд": round(prof, 3),
           "прибыль_минус_простой_млрд": round(prof - idle, 3), "бортов_лето": bs["лето"], "бортов_зима": bs["зима"],
           "рейсы_лето": legs_str(nets["лето"], Mx), "рейсы_зима": legs_str(nets["зима"], Mx)}
    print(f"[{time.time() - t:.0f} с] {kind} {label} {fare} {n} {variant} сид={seed}: {prof:+.3f} "
          f"борта {bs['лето']}/{bs['зима']}", flush=True)
    return row


def pool_run(jobs, out):
    global M
    M = sc.setup()
    jobs = sorted(jobs, key=lambda j: -j[3])            # долгие (8 бортов) первыми
    with Pool(2) as pool:
        rows = pool.map(run_job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / out, index=False, encoding="utf-8-sig")
    return df


# ---------------- comp ----------------
COMP = {"зимние частоты (как раньше)": {},
        "конкуренты летом ×(1+0,5(k-1))": {"summer_rival_share": 0.5},
        "конкуренты летом ×k": {"summer_rival_share": 1.0},
        "конкуренты и Москва летом ×k": {"summer_rival_share": 1.0, "summer_rival_moscow": True}}


def comp():
    jobs = [("конкуренты", lab, f, N, v, ch, None) for lab, ch in COMP.items()
            for f in (1.3, 1.45, 1.6, 1.8) for v in VARIANTS]
    jobs += [("конкуренты", lab, 1.6, 8, v, ch, None) for lab, ch in COMP.items() for v in VARIANTS if v != "хаб в Уфе"]
    pool_run(jobs, "seasonal_check_competitors.csv")


# ---------------- noise ----------------
NOISE_CASES = [(1.6, 4, v) for v in VARIANTS] + [(1.8, 4, v) for v in VARIANTS] + \
              [(1.45, 4, "прямые"), (1.6, 8, "прямые"), (1.6, 8, "гибрид с Уфой"), (1.6, 8, "гибрид без Минвод-Уфы")]


def pairs_of(s):
    return {x.split("×")[0] for x in s.split("; ") if x}


# вдогонку к azimut: прямые без Минвод-Уфы (сравнение с гибридом без неё на равных)
NOISE_CASES_NOMRV = [(1.6, 4, "прямые без Минвод-Уфы"), (1.8, 4, "прямые без Минвод-Уфы"), (1.6, 8, "прямые без Минвод-Уфы")]


def noise(cases=NOISE_CASES, tag=""):
    jobs = [("шум", "обычный" if seed is None else "случайный", f, n, v, {}, seed)
            for f, n, v in cases for seed in [None] + list(range(1, NOISE_RUNS + 1))]
    df = pool_run(jobs, f"seasonal_check_noise{tag}.csv")
    summ, routes = [], []
    for (f, n, v), g in df.groupby(["выручка", "бортов_дано", "вариант"], sort=False):
        det = g[g["условие"] == "обычный"].iloc[0]
        rnd = g[g["условие"] == "случайный"]
        best = g.loc[g["прибыль_млрд"].idxmax()]
        summ.append({"выручка": f, "бортов_дано": n, "вариант": v, "обычный_млрд": det["прибыль_млрд"],
                     "мин_млрд": rnd["прибыль_млрд"].min(), "медиана_млрд": round(rnd["прибыль_млрд"].median(), 3),
                     "макс_млрд": rnd["прибыль_млрд"].max(), "лучшее_найденное_млрд": best["прибыль_млрд"],
                     "размах_млрд": round(g["прибыль_млрд"].max() - g["прибыль_млрд"].min(), 3),
                     "лучшее_рейсы_лето": best["рейсы_лето"], "лучшее_рейсы_зима": best["рейсы_зима"]})
        for season, col in (("лето", "рейсы_лето"), ("зима", "рейсы_зима")):
            cnt = {}
            for s in g[col]:
                for p in pairs_of(s):
                    cnt[p] = cnt.get(p, 0) + 1
            for p, c in sorted(cnt.items(), key=lambda x: -x[1]):
                routes.append({"выручка": f, "бортов_дано": n, "вариант": v, "сезон": season, "пара": p,
                               "доля_прогонов": round(c / len(g), 2), "в_обычном": p in pairs_of(det[col])})
    pd.DataFrame(summ).to_csv(RESULTS / f"seasonal_check_noise{tag}_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(routes).to_csv(RESULTS / f"seasonal_check_noise{tag}_routes.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).drop(columns=["лучшее_рейсы_лето", "лучшее_рейсы_зима"]).to_string(index=False))


# ---------------- azimut ----------------
def respond(nets, Mx, freq=True, cut=False):
    """Ответ «Азимута» на готовую сеть: на парах, где Red Wings добавила рейсы и летает A4, A4 добавляет столько же
    рейсов (freq) и/или снижает цену на 15% (cut). Возвращает новые сети и изменения параметров цены."""
    s0 = M["s0_wz"]
    out, pairs = {}, set()
    for s, net in nets.items():
        n2 = copy.deepcopy(net)
        for p, c in net["direct"].items():
            extra = c.get("WZ", 0) - s0.get(p, 0)
            if extra > 0 and RIVAL in c and p in Mx["km"]:
                pairs.add(p)
                if freq:
                    n2["direct"][p][RIVAL] = c[RIVAL] + extra
        out[s] = n2
    return out, ({"rival_fare_factor": {p: CUT for p in pairs}} if cut else {}), pairs


def a4_pairs_note(nets, Mx):
    s0 = M["s0_wz"]
    return "; ".join(sorted({f"{p[0]}-{p[1]}" for net in nets.values() for p, c in net["direct"].items()
                             if c.get("WZ", 0) > s0.get(p, 0) and RIVAL in c and p in Mx["km"]}))


def azimut_greedy(job):
    """Заранее: A4 на Минводы-Уфа 4 -> 11 рейсов (+7) и -15% цены; Red Wings раскладывает сеть заново."""
    global M
    if M is None:
        M = sc.setup()
    fare, variant, pre_freq, pre_cut = job
    t = time.time()
    mode, no_mrv = VARIANTS[variant]
    saved = with_params({"rival_fare_factor": {("MRV", "UFA"): CUT} if pre_cut else {}})
    case = {**P["cases"]["базовый"], "fare": fare}
    Mx = dict(M)
    if no_mrv:
        Mx["direct_candidates"] = M["direct_candidates"] - {("MRV", "UFA")}
    orig = sc.base_network

    def base_with_a4(Mx_, mode_):
        net = orig(Mx_, mode_)
        net["direct"][("MRV", "UFA")][RIVAL] += pre_freq
        return net
    sc.base_network = base_with_a4
    try:
        nets, log = rb.greedy_net(Mx, mode, N, case, no_mrv=no_mrv)
        prof, idle, bs = rb.season_result(nets, sc.base_network(Mx, "прямые"), Mx, case)
        # потеря на текущей сети S0 из-за ответа (Red Wings на Минводы-Уфа сейчас не летает -> почти 0)
    finally:
        sc.base_network = orig
        for k, v in saved.items():
            P[k] = v
    cond = f"заранее: A4 Минводы-Уфа +{pre_freq}" + (" и -15% цены" if pre_cut else "")
    print(f"[{time.time() - t:.0f} с] {cond} {fare} {variant}: {prof:+.3f}", flush=True)
    return {"условие": cond, "выручка": fare, "вариант": variant, "прибыль_млрд": round(prof, 3),
            "бортов_лето": bs["лето"], "бортов_зима": bs["зима"],
            "рейсы_лето": legs_str(nets["лето"], Mx), "рейсы_зима": legs_str(nets["зима"], Mx), "пары_с_A4": ""}


def azimut():
    global M
    M = sc.setup()
    rows = []
    for fare in (1.6, 1.8):
        case = {**P["cases"]["базовый"], "fare": fare}
        for variant, (mode, no_mrv) in VARIANTS.items():
            Mx = dict(M)
            if no_mrv:
                Mx["direct_candidates"] = M["direct_candidates"] - {("MRV", "UFA")}
            nets, _ = rb.greedy_net(Mx, mode, N, case, no_mrv=no_mrv)
            s0 = sc.base_network(Mx, "прямые")
            for cond, freq, cut in (("без ответа", False, False), ("A4 +столько же рейсов", True, False),
                                    ("A4 -15% цены", False, True), ("A4 +рейсы и -15% цены", True, True)):
                n2, pch, pairs = respond(nets, Mx, freq, cut)
                saved = with_params(pch)
                prof, idle, bs = rb.season_result(n2, s0, Mx, case)
                for k, v in saved.items():
                    P[k] = v
                rows.append({"условие": cond, "выручка": fare, "вариант": variant, "прибыль_млрд": round(prof, 3),
                             "бортов_лето": bs["лето"], "бортов_зима": bs["зима"],
                             "рейсы_лето": legs_str(nets["лето"], Mx), "рейсы_зима": legs_str(nets["зима"], Mx),
                             "пары_с_A4": "; ".join(sorted(f"{p[0]}-{p[1]}" for p in pairs))})
                print(rows[-1]["условие"], fare, variant, rows[-1]["прибыль_млрд"], rows[-1]["пары_с_A4"], flush=True)
    jobs = [(f, v, 7, cut) for f in (1.6, 1.8) for v in ("прямые", "гибрид с Уфой") for cut in (False, True)]
    with Pool(2) as pool:
        rows += pool.map(azimut_greedy, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(RESULTS / "seasonal_check_azimut.csv", index=False, encoding="utf-8-sig")
    azimut_sens()


def azimut_sens():
    """39 проверок step7_seasons_robust sens при +60%, но сети — с ответом «Азимута» (+рейсы и -15% цены)."""
    global M
    out = []
    M = M or sc.setup()
    Mx, nets0, s0 = rb.setup_sens(1.6)
    for cond, freq, cut in (("без ответа", False, False), ("A4 +рейсы и -15% цены", True, True)):
        nets = copy.deepcopy(nets0)
        pch = {}
        if freq or cut:
            for m in list(nets):
                nets[m], ch, _ = respond(nets[m], Mx, freq, cut)
                for p, f in ch.get("rival_fare_factor", {}).items():
                    pch.setdefault("rival_fare_factor", {})[p] = f
        saved = with_params(pch)
        oat = st6.one_at_a_time(Mx, nets, s0)
        extra = []
        for name, ch in rb.EXTRA:
            sv = st6.with_params(ch)
            r = rb.profits_seasons(Mx, nets, s0, {**st6.BASE_CASE})
            st6.restore(sv)
            extra.append({"условие": name, **{k: round(v, 3) for k, v in r.items()}, "лучшая": st6.best_of(r)})
        for k, v in saved.items():
            P[k] = v
        oat = pd.concat([oat, pd.DataFrame(extra)], ignore_index=True)
        oat.insert(0, "ответ_Азимута", cond)
        out.append(oat)
        print(cond, oat["лучшая"].value_counts().to_dict(), flush=True)
    pd.concat(out).to_csv(RESULTS / "seasonal_check_azimut_sens.csv", index=False, encoding="utf-8-sig")


# ---------------- ratios ----------------
RATIO_CASES = [(f, v) for f in (1.45, 1.6, 1.8) for v in ("прямые", "гибрид с Уфой")] + [(1.8, "гибрид без Минвод-Уфы")]


def ratios():
    base = dict(P["resort_summer_ratio"])
    conds = {"база (KGD 1,7, MCX 1,3)": {}}
    for c in ("KGD", "MCX"):
        for v in sorted({round(base[c] * 0.7, 2), round(base[c] * 1.3, 2), 1.0, 2.0}):
            conds[f"{c} {v:g}"] = {c: v}
    conds["KGD 1,0 и MCX 1,0"] = {"KGD": 1.0, "MCX": 1.0}
    conds["KGD 2,0 и MCX 2,0"] = {"KGD": 2.0, "MCX": 2.0}
    jobs = [("KGD/MCX", lab, f, N, v, {"resort_summer_ratio": dict(base, **ch)}, None)
            for lab, ch in conds.items() for f, v in RATIO_CASES]
    pool_run(jobs, "seasonal_check_ratios.csv")


if __name__ == "__main__":
    {"comp": comp, "noise": noise, "noise_nomrv": lambda: noise(NOISE_CASES_NOMRV, "_nomrv"), "azimut": azimut,
     "ratios": ratios}[sys.argv[1]]()
