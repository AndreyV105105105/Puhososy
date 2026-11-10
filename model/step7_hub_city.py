"""
Проверка выбора города для хаба расчётом, а не только рейтингом:
тот же хаб (H1), тот же алгоритм и флот — но в разных городах из топа рейтинга.
Считаем при доходности «Азимута» (1,15) и выше: +30%, +45%, +60% к минимальным тарифам.

Запуск: python model/step7_hub_city.py
Выход:  results/hub_city_compare.csv; с аргументом utair — results/hub_ufa_without_utair.csv
"""
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc

DEFAULT_HUB = P["hub_city"]

CITIES = ["KUF", "KZN", "UFA", "VOG", "OMS"]     # Самара + альтернативы из топа рейтинга (с известным пассажиропотоком)
N = P["new_aircraft"]["базовый"]


def hub_mix(r, hub):
    """Кто летит рейсами через город хаба: стыковочные (два плеча через хаб), «доехали до аэропорта»
    (летят не из своего города) и местные (летят из своего города в свой). Доли — до учёта мест в самолёте."""
    tr = gr = loc = 0.0
    for p, x in r["_pairs"].items():
        for pax, lg, _ in x["rw"]:
            if not any(hub in l for l in lg):
                continue
            if len(lg) == 2:
                tr += pax
            elif tuple(lg[0]) == tuple(p):
                loc += pax
            else:
                gr += pax
    tot = tr + gr + loc
    if not tot:
        return {"%_стыковочных": 0, "%_доехали_до_аэропорта": 0, "%_местных": 0}
    return {"%_стыковочных": round(tr / tot * 100), "%_доехали_до_аэропорта": round(gr / tot * 100),
            "%_местных": round(loc / tot * 100)}


def main():
    M = sc.setup()
    names = M["ap"]["город"].to_dict()
    rows = []
    for fare in (1.15, 1.3, 1.45, 1.6):   # сегодняшняя доходность («Азимут») и выше
        case = {**P["cases"]["базовый"], "fare": fare}
        s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
        for c in CITIES:
            sc.HUB = c
            P["hub_city"] = c
            net, log = sc.greedy(sc.base_network(M, "хаб"), M, "хаб", N, case)
            r = sc.evaluate(net, M, case)
            mix = hub_mix(r, c) if r["блок_часы"] > s0["блок_часы"] + 1 else \
                {"%_стыковочных": None, "%_доехали_до_аэропорта": None, "%_местных": None}   # хаб не открылся
            rows.append({"город_хаба": names[c], "доходность": fare, **mix,
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "новых_пассажиров": round(r["пассажиров"] - s0["пассажиров"]),
                         "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "спицы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(f"доходность {fare:.2f} | хаб {names[c]:<12} {rows[-1]['прибыль_млрд']:+.3f} млрд, "
                  f"{rows[-1]['самолётов']} борта | стык. {mix['%_стыковочных']}%, доехали {mix['%_доехали_до_аэропорта']}%, "
                  f"местн. {mix['%_местных']}% | {rows[-1]['спицы']}", flush=True)
    sc.HUB = P["hub_city"] = DEFAULT_HUB
    pd.DataFrame(rows).to_csv(RESULTS / "hub_city_compare.csv", index=False, encoding="utf-8-sig")


def utair_check():
    """Хаб в Уфе «без дублирования Utair»: запрещаем спицы на его маршруты из Уфы и пересчитываем."""
    M = sc.setup()
    rows = []
    P["hub_spoke_exclude"] = {"UFA": P["utair_ufa_routes"]}
    sc.HUB = P["hub_city"] = "UFA"
    for mode in ("хаб", "гибрид"):
        for fare in (1.3, 1.45, 1.6):
            case = {**P["cases"]["базовый"], "fare": fare}
            s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
            net, log = sc.greedy(sc.base_network(M, mode), M, mode, N, case)
            r = sc.evaluate(net, M, case)
            rows.append({"вариант": f"{mode} Уфа без маршрутов Utair", "доходность": fare,
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "рейсы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(rows[-1], flush=True)
    P["hub_spoke_exclude"] = {}
    sc.HUB = P["hub_city"] = DEFAULT_HUB
    pd.DataFrame(rows).to_csv(RESULTS / "hub_ufa_without_utair.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    import sys
    utair_check() if "utair" in sys.argv else main()
