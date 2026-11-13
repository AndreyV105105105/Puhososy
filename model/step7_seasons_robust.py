"""
ШАГ 7 (сезоны, устойчивость). Те же проверки, что для версии 3, но на сезонной модели (лето и зима, step7_seasons.py):
  sens  — по одному параметру (как step6_sensitivity) + отношения лето/зима и плата за простой;
  mc    — Монте-Карло (как step6_sensitivity);
  thr   — пороги по выручке 1,0-1,8 (как step7_thresholds доходность), 4 борта;
  fleet — 2/4/8/14 бортов при выручке 1,6 и 1,8 (как step7_fleet).

Сети в sens/mc строятся сезонным жадным алгоритмом (greedy_seasons) при выручке SENS_FARE и потом не перестраиваются
(как в step6). В thr/fleet сеть заново раскладывается для каждого значения.
Простой: столбцы *_простой — та же сеть минус 8 млн руб. в месяц за борт, занятый летом и не занятый зимой
(проверка «что если», ставка не подтверждена).
Версию 3 (step6_sensitivity, step7_thresholds, step7_fleet) этот скрипт не меняет: параметры возвращаются после расчёта.

Запуск: python model/step7_seasons_robust.py sens|mc|thr|fleet [выручка для sens/mc, по умолчанию 1.45] [прогонов mc]
Выход:  results/seasonal_sensitivity_<выручка>.csv, results/seasonal_monte_carlo_<выручка>.csv,
        results/seasonal_thresholds_доходность.csv, results/seasonal_fleet.csv
"""
import sys
import time
from multiprocessing import Pool

import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc
import step6_sensitivity as st6

IDLE_CHECK = 8e6
N = P["new_aircraft"]["базовый"]
MODES6 = {"хаб": "H1 хаб", "прямые": "H2 прямые", "гибрид": "H3 гибрид", "гибрид_без_MRV": "H3 гибрид без Минвод-Уфы"}
M = None


def greedy_net(Mx, mode, n, case, hub=None, no_mrv=False):
    """Сезонная сеть варианта; no_mrv — без спицы и прямого Минводы-Уфа (ответ «Азимута»)."""
    saved = (P["hub_spoke_exclude"], sc.HUB, P["hub_city"])
    if hub:
        sc.HUB = P["hub_city"] = hub
    P["hub_spoke_exclude"] = {sc.HUB: ["MRV"]} if no_mrv else {}
    if no_mrv:
        Mx = dict(Mx)
        Mx["direct_candidates"] = Mx["direct_candidates"] - {("MRV", "UFA")}
    P["seasons"] = True
    nets, log, SM = sc.greedy_seasons(sc.base_network(Mx, mode), Mx, mode, n, case)
    P["seasons"] = False
    P["hub_spoke_exclude"], sc.HUB, P["hub_city"] = saved
    return nets, log


def season_result(nets, s0net, Mx, case):
    """Прибыль сверх S0 (млрд), плата за простой при IDLE_CHECK (млрд), борта лето/зима."""
    SM = sc.season_models(Mx)
    base, b0 = sc.evaluate_seasons({s: s0net for s, _, _ in SM}, SM, case)
    tot, res = sc.evaluate_seasons(nets, SM, case)
    saved = P["idle_cost_rub_month"]
    P["idle_cost_rub_month"] = IDLE_CHECK
    idle = sc.idle_cost(nets, Mx, sc.block_hours(s0net, Mx))
    P["idle_cost_rub_month"] = saved
    bs = {s: round((res[s]["блок_часы"] - b0[s]["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1) for s in res}
    return (tot["прибыль"] - base["прибыль"]) / 1e9, idle / 1e9, bs


# ---------------- sens / mc: подмена profits() в step6 на сезонную ----------------
def profits_seasons(Mx, nets, s0, case):
    M2, asc = st6.recalibrated(Mx, P["vot_rub_h"], P["b_freq"])
    saved_asc = (P["asc_train"], P["asc_car"])
    P["asc_train"], P["asc_car"] = asc
    SM = sc.season_models(M2)
    base, _ = sc.evaluate_seasons({s: s0 for s, _, _ in SM}, SM, case)
    bh0 = sc.block_hours(s0, M2)
    out = {}
    for m, n in nets.items():
        t, _ = sc.evaluate_seasons(n, SM, case)
        out[MODES6[m]] = (t["прибыль"] - base["прибыль"] - sc.idle_cost(n, M2, bh0)) / 1e9
    out["S0 не расширяться"] = 0.0
    P["asc_train"], P["asc_car"] = saved_asc
    return out


def setup_sens(fare):
    Mx = sc.setup()
    cal = pd.read_csv(RESULTS / "choice_calibration.csv", encoding="utf-8-sig")
    Mx["scale"] = float(cal.loc[cal["ошибка_лог"].idxmin(), "scale"])
    st6.SENS_FARE = fare
    st6.BASE_CASE = {**P["cases"]["базовый"], "fare": fare}
    st6.MODES = MODES6
    st6.profits = profits_seasons
    nets = {}
    for m in MODES6:
        mode = "гибрид" if m == "гибрид_без_MRV" else m
        nets[m], log = greedy_net(Mx, mode, N, st6.BASE_CASE, no_mrv=(m == "гибрид_без_MRV"))
        print(m, {s: st6_legs(n, Mx) for s, n in nets[m].items()}, flush=True)
    return Mx, nets, sc.base_network(Mx, "прямые")


def st6_legs(net, Mx):
    a, b = sc.rw_legs(net, Mx), sc.rw_legs(sc.base_network(Mx, "прямые"), Mx)
    return "; ".join(f"{p[0]}-{p[1]}×{f:g}" for p, f in sorted(a.items()) if f > b.get(p, 0))


EXTRA = [  # сезонные проверки: (название, изменения параметров)
    ("простой 8 млн/мес за борт", {"idle_cost_rub_month": IDLE_CHECK}),
    ("Сочи лето/зима 2,0", {"resort_summer_ratio": dict(P["resort_summer_ratio"], AER=2.0)}),
    ("Сочи лето/зима 3,5", {"resort_summer_ratio": dict(P["resort_summer_ratio"], AER=3.5)}),
    ("Калининград лето/зима 2,3", {"resort_summer_ratio": dict(P["resort_summer_ratio"], KGD=2.3)}),
    ("Махачкала лето/зима 1,0", {"resort_summer_ratio": dict(P["resort_summer_ratio"], MCX=1.0)}),
]


def sens(fare):
    Mx, nets, s0 = setup_sens(fare)
    oat = st6.one_at_a_time(Mx, nets, s0)
    rows = []
    for name, pch in EXTRA:
        saved = st6.with_params(pch)
        r = profits_seasons(Mx, nets, s0, {**st6.BASE_CASE})
        st6.restore(saved)
        rows.append({"условие": name, **{k: round(v, 3) for k, v in r.items()}, "лучшая": st6.best_of(r)})
    oat = pd.concat([oat, pd.DataFrame(rows)], ignore_index=True)
    oat.to_csv(RESULTS / f"seasonal_sensitivity_{round(fare * 100)}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(oat.to_string(index=False))


def mc(fare, n):
    Mx, nets, s0 = setup_sens(fare)
    t = time.time()
    st6.profits(Mx, nets, s0, st6.BASE_CASE)
    print(f"один набор допущений: {time.time() - t:.1f} с", flush=True)
    t = time.time()
    df = st6.monte_carlo(Mx, nets, s0, n=n)
    # плата за простой для той же выборки: зависит только от блок-часов сетей (через долю перегонов),
    # поэтому считаем её отдельно и вычитаем, не пересчитывая спрос
    P["idle_cost_rub_month"] = IDLE_CHECK
    for i, row in df.iterrows():
        saved = st6.with_params({"ferry_share": row["ferry_share"]})
        bh0 = sc.block_hours(s0, Mx)
        for m, net in nets.items():
            df.loc[i, MODES6[m] + "_простой"] = row[MODES6[m]] - sc.idle_cost(net, Mx, bh0) / 1e9
        st6.restore(saved)
    P["idle_cost_rub_month"] = 0
    df["S0 не расширяться_простой"] = 0.0
    df["лучшая_простой"] = [st6.best_of({k: r[k + "_простой"] for k in list(MODES6.values()) + ["S0 не расширяться"]})
                            for _, r in df.iterrows()]
    df.round(4).to_csv(RESULTS / f"seasonal_monte_carlo_{round(fare * 100)}.csv", index=False, encoding="utf-8-sig")
    print(f"{len(df)} прогонов за {time.time() - t:.0f} с")
    for col in ("лучшая", "лучшая_простой"):
        print(col, df[col].value_counts(normalize=True).round(3).to_dict())
    for h in MODES6.values():
        print(h, "медиана", round(df[h].median(), 3), "в плюсе", round((df[h] > 0).mean(), 3),
              "хуже прямых", round((df[h] < df["H2 прямые"] - 1e-6).mean(), 3))


# ---------------- thr / fleet: пул заданий ----------------
THR_FARES = [1.0, 1.15, 1.3, 1.45, 1.6, 1.8]
THR_MODES = {"хаб": "H1 хаб в Уфе", "прямые": "H2 прямые рейсы", "гибрид": "H3 гибрид"}
FLEET_VARIANTS = [("прямые", "UFA", False, "прямые"), ("хаб", "UFA", False, "хаб в Уфе"), ("хаб", "KZN", False, "хаб в Казани"),
                  ("гибрид", "UFA", False, "гибрид в Уфе"), ("гибрид", "UFA", True, "гибрид в Уфе без Минвод")]


def run_job(job):
    global M
    if M is None:
        M = sc.setup()
    kind, fare, n, mode, hub, no_mrv, name = job
    t = time.time()
    case = {**P["cases"]["базовый"], "fare": fare}
    Mx = dict(M)
    if no_mrv:
        Mx["direct_candidates"] = M["direct_candidates"] - {("MRV", "UFA")}
    saved = (sc.HUB, P["hub_city"])
    sc.HUB = P["hub_city"] = hub
    s0net = sc.base_network(Mx, "прямые")
    nets, log = greedy_net(Mx, mode, n, case, hub=hub, no_mrv=no_mrv)
    sc.HUB = P["hub_city"] = hub
    prof, idle, bs = season_result(nets, s0net, Mx, case)
    # та же задача, но сеть выбирается с учётом платы за простой
    P["idle_cost_rub_month"] = IDLE_CHECK
    nets_i, _ = greedy_net(Mx, mode, n, case, hub=hub, no_mrv=no_mrv)
    P["idle_cost_rub_month"] = 0
    sc.HUB = P["hub_city"] = hub
    prof_i, idle_i, bs_i = season_result(nets_i, s0net, Mx, case)
    sc.HUB, P["hub_city"] = saved
    spokes = sum(1 for x in log if x.get("принят") and x["ход"] == "спица")
    row = {"проверка": kind, "выручка": fare, "бортов_дано": n, "вариант": name,
           "прибыль_млрд": round(prof, 3), "бортов_лето": bs["лето"], "бортов_зима": bs["зима"],
           "простой_млрд": round(idle, 3), "прибыль_минус_простой_млрд": round(prof - idle, 3),
           "прибыль_сеть_с_учётом_простоя_млрд": round(prof_i - idle_i, 3),
           "бортов_лето_с_простоем": bs_i["лето"], "бортов_зима_с_простоем": bs_i["зима"], "спиц_хаба": spokes,
           "рейсы_лето": st6_legs(nets["лето"], Mx), "рейсы_зима": st6_legs(nets["зима"], Mx)}
    print(f"[{time.time() - t:.0f} с] {kind} {fare} {n} {name}: {row['прибыль_млрд']:+.3f} "
          f"(с простоем {row['прибыль_минус_простой_млрд']:+.3f} / {row['прибыль_сеть_с_учётом_простоя_млрд']:+.3f}) "
          f"борта {bs['лето']}/{bs['зима']}", flush=True)
    return row


def pool_run(jobs, out):
    global M
    M = sc.setup()
    with Pool(2) as pool:
        rows = pool.map(run_job, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(RESULTS / out, index=False, encoding="utf-8-sig")


def main(args):
    what = args[0]
    if what == "sens":
        sens(float(args[1]) if len(args) > 1 else 1.45)
    elif what == "mc":
        mc(float(args[1]) if len(args) > 1 else 1.45, int(args[2]) if len(args) > 2 else 500)
    elif what == "thr":
        jobs = [("пороги", f, N, m, "UFA", False, THR_MODES[m]) for f in THR_FARES for m in THR_MODES]
        pool_run(jobs, "seasonal_thresholds_доходность.csv")
    elif what == "fleet":
        jobs = [("флот", f, n, m, h, x, name) for n in (2, 4, 8, 14) for (m, h, x, name) in FLEET_VARIANTS
                for f in (1.6, 1.8)]
        jobs.sort(key=lambda j: -j[2])            # долгие (14 бортов) первыми
        pool_run(jobs, "seasonal_fleet.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
