"""
Проверка выбора города для хаба расчётом, а не только рейтингом:
тот же хаб (H1), тот же алгоритм и флот — но в разных городах из топа рейтинга.
Считаем при текущих допущениях и при доходности +20% (там, где хаб в Самаре начинает окупаться).

Запуск: python model/step7_hub_city.py
Выход:  results/hub_city_compare.csv
"""
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc

CITIES = ["KUF", "KZN", "UFA", "VOG", "OMS"]     # Самара + альтернативы из топа рейтинга (с известным пассажиропотоком)
N = P["new_aircraft"]["базовый"]


def main():
    M = sc.setup()
    names = M["ap"]["город"].to_dict()
    rows = []
    for fare in (1.0, 1.2):
        case = {**P["cases"]["базовый"], "fare": fare}
        s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
        for c in CITIES:
            sc.HUB = c
            P["hub_city"] = c
            net, log = sc.greedy(sc.base_network(M, "хаб"), M, "хаб", N, case)
            r = sc.evaluate(net, M, case)
            rows.append({"город_хаба": names[c], "доходность": fare,
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "новых_пассажиров": round(r["пассажиров"] - s0["пассажиров"]),
                         "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "спицы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(f"доходность {fare:.1f} | хаб {names[c]:<12} {rows[-1]['прибыль_млрд']:+.3f} млрд, "
                  f"{rows[-1]['самолётов']} борта | {rows[-1]['спицы']}", flush=True)
    sc.HUB = P["hub_city"] = "KUF"
    pd.DataFrame(rows).to_csv(RESULTS / "hub_city_compare.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
