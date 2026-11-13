"""
ШАГ 7 (сезоны). Версия 3 с двумя сезонами: лето и зима для курортных направлений.

Зачем: ноябрьское табло — зимнее расписание, а курортные пары (Сочи, Минводы) летом летают в разы чаще.
В версии 3 спрос курортных пар — годовой (курортная поправка), а частота рейса одна на весь год.
Здесь частоты выбираются отдельно для лета и зимы, флот общий (в каждом сезоне не больше N бортов).

Спрос: годовой спрос пары сохраняется, летом недельный спрос курортной пары в resort_summer_ratio раз
выше зимнего (Сочи 2,5, Минводы 1,5; params.py). Базовая сеть S0 (табло ноября) — одинаковая в оба сезона.

Для каждой строки считаем и версию 3 «год» (один уровень частот), и сезонную модель — для сравнения.

Запуск: python model/step7_seasons.py   (после step1 и step2; ~5 мин на 2 ядрах)
Выход:  results/seasonal_scenarios.csv
        (столбец сеть_года_в_сезонах_млрд: сеть версии 3 с одинаковыми частотами круглый год,
         посчитанная с сезонным спросом, — проверка жадного алгоритма и цена круглогодичных курортных рейсов)
"""
import copy
import time
from multiprocessing import Pool

import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc

FARES = [1.3, 1.45, 1.6, 1.8]
VARIANTS = {
    # вариант: (режим жадного алгоритма, убрать спицу и прямой Минводы–Уфа)
    "прямые": ("прямые", False),
    "хаб в Уфе": ("хаб", False),
    "гибрид с Уфой": ("гибрид", False),
    "гибрид с Уфой без Минвод": ("гибрид", True),
}
AER_DEFAULT = P["resort_summer_ratio"]["AER"]
M = None


def jobs():
    out = []
    for fare in FARES:
        for v in VARIANTS:
            out.append((fare, 4, v, AER_DEFAULT))
    for v in VARIANTS:
        out.append((1.6, 8, v, AER_DEFAULT))
    for aer in (2.0, 3.5):
        for v in VARIANTS:
            out.append((1.6, 4, v, aer))
    return out


def new_legs(net, net0, Mx):
    """Что добавлено к сети S0: «пара×рейсов в неделю» для рейсов Red Wings, частота которых выросла."""
    a, b = sc.rw_legs(net, Mx), sc.rw_legs(net0, Mx)
    return "; ".join(f"{p[0]}-{p[1]}×{f:g}" for p, f in sorted(a.items()) if f > b.get(p, 0))


def run(job):
    global M
    if M is None:                                    # Windows (spawn): у процесса-работника свои данные
        M = sc.setup()
    fare, n, variant, aer = job
    t0 = time.time()
    mode, no_mrv = VARIANTS[variant]
    P["seasons"] = False
    P["hub_spoke_exclude"] = {"UFA": ["MRV"]} if no_mrv else {}
    P["resort_summer_ratio"] = dict(P["resort_summer_ratio"], AER=aer)
    Mx = dict(M)
    if no_mrv:
        Mx["direct_candidates"] = M["direct_candidates"] - {("MRV", "UFA")}
    case = dict(P["cases"]["базовый"], fare=fare)
    base = dict(выручка=fare, бортов_дано=n, вариант=variant, AER_лето_к_зиме=aer)
    rows = []

    # версия 3: один уровень частот на весь год (как step3/step7_thresholds)
    s0 = sc.evaluate(sc.base_network(Mx, "прямые"), Mx, case)
    net0 = sc.base_network(Mx, mode)
    net_year, _ = sc.greedy(net0, Mx, mode, n, case)
    if aer == AER_DEFAULT:
        r = sc.evaluate(net_year, Mx, case)
        b = round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1)
        legs = new_legs(net_year, net0, Mx)
        rows.append(dict(base, модель="версия 3 (год)", прибыль_млрд=round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         бортов_занято=b, бортов_лето=b, бортов_зима=b, рейсы_лето=legs, рейсы_зима=legs))

    # два сезона
    P["seasons"] = True
    SM = sc.season_models(Mx)
    s0net = sc.base_network(Mx, "прямые")
    s0, s0r = sc.evaluate_seasons({s: s0net for s, _, _ in SM}, SM, case)
    net0 = sc.base_network(Mx, mode)
    nets, log, SM = sc.greedy_seasons(net0, Mx, mode, n, case)
    tot, res = sc.evaluate_seasons(nets, SM, case)
    # проверка жадного алгоритма: сеть версии 3 (одинаковые частоты в оба сезона) — допустимое решение сезонной
    # задачи; если она даёт больше, чем сезонный жадный алгоритм, это близорукость алгоритма, а не модели
    yr, _ = sc.evaluate_seasons({s: net_year for s, _, _ in SM}, SM, case)
    bs = {s: round((res[s]["блок_часы"] - s0r[s]["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1) for s, _, _ in SM}
    rows.append(dict(base, модель="сезоны", прибыль_млрд=round((tot["прибыль"] - s0["прибыль"]) / 1e9, 3),
                     бортов_занято=max(bs.values()), бортов_лето=bs["лето"], бортов_зима=bs["зима"],
                     рейсы_лето=new_legs(nets["лето"], net0, Mx), рейсы_зима=new_legs(nets["зима"], net0, Mx),
                     сеть_года_в_сезонах_млрд=round((yr["прибыль"] - s0["прибыль"]) / 1e9, 3)))
    P["seasons"] = False
    print(f"[{time.time()-t0:.0f} с] {fare} {n} {variant} AER={aer}: "
          + " | ".join(f"{x['модель']}: {x['прибыль_млрд']:+.3f}" for x in rows), flush=True)
    return rows


def main():
    global M
    M = sc.setup()
    with Pool(2) as pool:
        parts = pool.map(run, jobs(), chunksize=1)
    out = pd.DataFrame([r for part in parts for r in part])
    cols = ["выручка", "бортов_дано", "вариант", "AER_лето_к_зиме", "модель", "прибыль_млрд", "бортов_занято",
            "бортов_лето", "бортов_зима", "рейсы_лето", "рейсы_зима", "сеть_года_в_сезонах_млрд"]
    out[cols].to_csv(RESULTS / "seasonal_scenarios.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 120)
    print(out[cols].to_string())


if __name__ == "__main__":
    main()
