"""
Проверка устойчивости: тариф на парах короче 550 км −25% (research/model_checks_10_11.md, Семёныч).
Короткие пары — спицы хаба в Уфе (Казань, Самара, Тюмень) и Екатеринбург–Уфа: остаются ли они в сети
и остаётся ли Уфа хабом. Хаб в Уфе — с долей стыковочных пассажиров.

Запуск: python model/check_short_fare.py   →  results/check_short_fare.csv
"""
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc
from step7_hub_city import hub_mix

DEFAULT_HUB = P["hub_city"]


def main(factor=0.75):
    P["short_fare_factor"] = factor
    M = sc.setup()
    rows = []
    for mode, hub in [("прямые", DEFAULT_HUB), ("хаб", "UFA"), ("хаб", "KZN"), ("гибрид", "UFA")]:
        sc.HUB = P["hub_city"] = hub
        for fare in (1.3, 1.45, 1.6):
            case = {**P["cases"]["базовый"], "fare": fare}
            s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
            net, log = sc.greedy(sc.base_network(M, mode), M, mode, P["new_aircraft"]["базовый"], case)
            r = sc.evaluate(net, M, case)
            mix = hub_mix(r, hub) if net.get("hub") and net["hub"].get("active") else {}   # только если хаб открыт
            rows.append({"вариант": mode if mode == "прямые" else f"{mode} в {P['hub_names'][hub]}", "выручка_к_АТОР": fare,
                         "%_стыковочных": mix.get("%_стыковочных"),
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "рейсы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(rows[-1], flush=True)
    P["short_fare_factor"] = 1.0
    sc.HUB = P["hub_city"] = DEFAULT_HUB
    pd.DataFrame(rows).to_csv(RESULTS / "check_short_fare.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
