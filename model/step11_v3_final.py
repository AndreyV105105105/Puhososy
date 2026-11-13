"""
ВЕРСИЯ 3, ИТОГОВЫЙ РАСЧЁТ (сезонная модель с уточнениями из проверок рисков, research/model_v3_risk_checks.md).

Настройки итогового расчёта (только в этом скрипте, params.py по умолчанию не меняется):
  - два сезона (лето 17 недель, зима 35), общий флот;
  - летом рейсы конкурентов на курортных парах растут как спрос (summer_rival_share = 1);
  - равные ходы жадного алгоритма выбираются по названию (greedy_tie_eps = 1e-9) — результат одинаков на любом компьютере;
  - лучший из обычного прогона и случайных (на каждом шаге случайный ход из почти лучших, допуск 10%):
    20 случайных при 4 бортах и выручке от +45%, 5 при меньшей выручке, у хаба и при 2 бортах, 3 при 8 бортах.

Основная сеть строится БЕЗ Минвод-Уфы (уязвима к ответу «Азимута» при любом варианте сети);
отдельно — те же варианты С Минводами-Уфой: «если „Азимут“ не ответит».

Запуск: python model/step11_v3_final.py main|fleet|all
Выход:  results/v3_final_scenarios.csv (выручка 1,0-1,8, 4 борта), results/v3_final_fleet.csv (2 и 8 бортов),
        results/v3_final_nets.pkl (сети лучших прогонов — для карт и дорожной карты).
"""
import pickle
import statistics
import sys
import time
from multiprocessing import Pool

import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc
import step7_seasons_robust as rb

SETTINGS = {"summer_rival_share": 1.0, "summer_rival_moscow": False, "greedy_tie_eps": 1e-9}
TOL = 0.10
FARES = [1.0, 1.15, 1.3, 1.45, 1.6, 1.8]
# вариант: (режим, без Минвод-Уфы, набор)
VARIANTS = {
    "прямые": ("прямые", True, "основная (без Минвод-Уфы)"),
    "гибрид с Уфой": ("гибрид", True, "основная (без Минвод-Уфы)"),
    "хаб в Уфе": ("хаб", True, "основная (без Минвод-Уфы)"),
    "прямые + Минводы-Уфа": ("прямые", False, "если «Азимут» не ответит"),
    "гибрид с Уфой + Минводы-Уфа": ("гибрид", False, "если «Азимут» не ответит"),
    "хаб в Уфе + Минводы-Уфа": ("хаб", False, "если «Азимут» не ответит"),
}
M = None


def runs_for(fare, n, mode):
    if mode == "хаб":
        return 5                      # у хаба шума не было (research/model_v3_risk_checks.md, п. 2)
    if n == 4:
        return 20 if fare >= 1.45 else 5
    return 3 if n >= 8 else 5


def one(Mx, mode, n, case, no_mrv, seed):
    P["greedy_random"] = None if seed is None else {"seed": seed, "tol": TOL}
    nets, _ = rb.greedy_net(Mx, mode, n, case, no_mrv=no_mrv)
    P["greedy_random"] = None
    return nets


def run(job):
    global M
    if M is None:
        M = sc.setup()
    P.update(SETTINGS)
    fare, n, variant = job
    mode, no_mrv, group = VARIANTS[variant]
    t0 = time.time()
    case = dict(P["cases"]["базовый"], fare=fare)
    Mx = dict(M)
    if no_mrv:
        Mx["direct_candidates"] = M["direct_candidates"] - {("MRV", "UFA")}
    s0net = sc.base_network(Mx, "прямые")
    P["seasons"] = True
    SM = sc.season_models(Mx)
    base, b0 = sc.evaluate_seasons({s: s0net for s, _, _ in SM}, SM, case)
    results = []
    for seed in [None] + list(range(1, runs_for(fare, n, mode) + 1)):
        P["seasons"] = True
        nets = one(Mx, mode, n, case, no_mrv, seed)
        P["seasons"] = True
        tot, res = sc.evaluate_seasons(nets, SM, case)
        results.append(((tot["прибыль"] - base["прибыль"]) / 1e9, seed, nets, tot, res))
    P["seasons"] = False
    profits = [r[0] for r in results]
    best = max(results, key=lambda r: (round(r[0], 9), r[1] is None))
    p, seed, nets, tot, res = best
    d = lambda k: tot[k] - base[k]
    bs = {s: round((res[s]["блок_часы"] - b0[s]["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1) for s in res}
    row = dict(набор=group, выручка=fare, бортов_дано=n, вариант=variant,
               прибыль_млрд=round(p, 3), обычный_прогон_млрд=round(profits[0], 3),
               медиана_млрд=round(statistics.median(profits), 3), минимум_млрд=round(min(profits), 3),
               прогонов=len(profits), лучший_сид=("обычный" if seed is None else seed),
               бортов_лето=bs["лето"], бортов_зима=bs["зима"],
               новых_пассажиров=round(d("пассажиров")), выручка_млрд=round(d("выручка") / 1e9, 2),
               субсидии_млрд=round(d("субсидии") / 1e9, 2),
               прирост_доли_пп=round(d("перевозок") / (P["market_total_mln"] * 1e6) * 100, 3),
               хаб_работает=any(nets[s].get("hub") and nets[s]["hub"].get("active") for s in nets),
               рейсы_лето=rb.st6_legs(nets["лето"], Mx), рейсы_зима=rb.st6_legs(nets["зима"], Mx))
    print(f"[{time.time()-t0:.0f} с] {fare} {n} {variant}: {p:+.3f} (обычный {profits[0]:+.3f}, "
          f"медиана {statistics.median(profits):+.3f}, прогонов {len(profits)})", flush=True)
    return row, (fare, n, variant), nets


def pool_run(jobs, out_csv, nets_store):
    jobs = sorted(jobs, key=lambda j: (-j[1], -j[0]))           # долгие первыми
    with Pool(2) as pool:
        parts = pool.map(run, jobs, chunksize=1)
    rows = [r for r, _, _ in parts]
    for _, k, nets in parts:
        nets_store[k] = nets
    df = pd.DataFrame(rows).sort_values(["набор", "бортов_дано", "выручка", "вариант"])
    df.to_csv(RESULTS / out_csv, index=False, encoding="utf-8-sig")
    return df


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    path = RESULTS / "v3_final_nets.pkl"
    store = pickle.load(open(path, "rb")) if path.exists() else {}
    if what in ("main", "all"):
        pool_run([(f, 4, v) for f in FARES for v in VARIANTS], "v3_final_scenarios.csv", store)
        pickle.dump(store, open(path, "wb"))
    if what in ("fleet", "all"):
        pool_run([(f, n, v) for f in (1.6, 1.8) for n in (2, 8) for v in VARIANTS if VARIANTS[v][0] != "хаб"],
                 "v3_final_fleet.csv", store)
        pickle.dump(store, open(path, "wb"))


if __name__ == "__main__":
    main()
