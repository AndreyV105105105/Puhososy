"""
ML-блок, шаг 3. Меняется ли вывод, если поправку ×0,3 для пар без рейсов заменить вероятностью p из ML-блока
(results/ml_viability.csv: своя p у каждой пары, средняя 0,12, у 44 пар выше 0,5).

Сравниваем прямые рейсы, хаб в Самаре, Уфе и Казани и гибрид при выручке +30/+45/+60% к минимальным тарифам.
Особенно важно для хаба: он возит пассажиров как раз между парами без прямых рейсов.

Запуск: python model/check_ml.py [x0.3|ml]   →  results/check_ml.csv (оба режима, если без аргумента)
"""
import sys
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc
from step7_hub_city import hub_mix

DEFAULT_HUB = P["hub_city"]
VARIANTS = [("прямые", DEFAULT_HUB), ("хаб", "KUF"), ("хаб", "UFA"), ("хаб", "KZN"), ("гибрид", "UFA")]


def run(mode_unserved):
    P["unserved_mode"] = mode_unserved
    M = sc.setup()
    rows = []
    for mode, hub in VARIANTS:
        sc.HUB = P["hub_city"] = hub
        for fare in (1.3, 1.45, 1.6):
            case = {**P["cases"]["базовый"], "fare": fare}
            s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
            net, log = sc.greedy(sc.base_network(M, mode), M, mode, P["new_aircraft"]["базовый"], case)
            r = sc.evaluate(net, M, case)
            mix = hub_mix(r, hub) if net.get("hub") and net["hub"].get("active") else {}
            rows.append({"поправка_пар_без_рейсов": mode_unserved,
                         "вариант": mode if mode == "прямые" else f"{mode} в {P['hub_names'][hub]}",
                         "выручка_к_АТОР": fare, "%_стыковочных": mix.get("%_стыковочных"),
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "рейсы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(rows[-1], flush=True)
    sc.HUB = P["hub_city"] = DEFAULT_HUB
    P["unserved_mode"] = "x0.3"
    return rows


def main(modes=("x0.3", "ml")):
    rows = [r for m in modes for r in run(m)]
    out = RESULTS / ("check_ml.csv" if len(modes) == 2 else f"check_ml_{modes[0]}.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("x0.3", "ml"))
