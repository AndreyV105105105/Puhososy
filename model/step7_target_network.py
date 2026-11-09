"""
Целевая сеть 2028 (требование ТЗ: сеть 8–12 маршрутов и число самолётов).
Базовое решение даёт один шаг (Екатеринбург–Казань ×14). Целевая сеть — та, которую модель строит
при пороге доходности +45% к минимальным тарифам (≈ 2,8 борта), вместе с действующими рейсами Red Wings,
которые она использует. Плюс проверка: тот же порог БЕЗ субсидий (замечание Семёныча — пороги держатся на них).

Запуск: python model/step7_target_network.py
Выход:  results/target_network_2028.csv, results/map_цель2028.png, results/thresholds_без_субсидий.csv
"""
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc
import step2_choice as ch
import step8_map as mp

FARE = 1.45
N = P["new_aircraft"]["базовый"]


def main():
    M = sc.setup()
    case = {**P["cases"]["базовый"], "fare": FARE}
    s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
    net, log = sc.greedy(sc.base_network(M, "прямые"), M, "прямые", N, case)
    r = sc.evaluate(net, M, case, detail=True)
    s0net = ch.network_from_boards(M["ap"])["direct"]
    names = M["ap"]["город"].to_dict()
    rows = []
    for x in r["legs"]:
        old = s0net.get((x["i"], x["j"]), {}).get("WZ", 0)
        status = "новый" if old == 0 else ("больше рейсов" if x["рейсов_нед"] > old else "действующий")
        rows.append({"маршрут": f"{names[x['i']]} – {names[x['j']]}", "статус": status, "рейсов_нед_сейчас": old,
                     "рейсов_нед_2028": x["рейсов_нед"], "пассажиров_в_год": x["пасс_в_год"], "загрузка": x["загрузка"],
                     "км": x["км"], "i": x["i"], "j": x["j"]})
    df = pd.DataFrame(rows).sort_values(["статус", "пассажиров_в_год"], ascending=[True, False])
    df.to_csv(RESULTS / "target_network_2028.csv", index=False, encoding="utf-8-sig")
    print(f"Доходность {FARE}: прибыль +{(r['прибыль'] - s0['прибыль'])/1e9:.2f} млрд, бортов "
          f"{(r['блок_часы'] - s0['блок_часы'])/sc.BH_PER_AIRCRAFT:.1f}, новых пассажиров {r['пассажиров'] - s0['пассажиров']:.0f}")
    print(df[["маршрут", "статус", "рейсов_нед_сейчас", "рейсов_нед_2028", "пассажиров_в_год"]].to_string(index=False))
    # карта целевой сети
    pd.DataFrame(r["legs"]).to_csv(RESULTS / "network_цель2028_базовый.csv", index=False, encoding="utf-8-sig")
    mp.TITLES["цель2028"] = "Целевая сеть 2028 (доходность +45%)"
    mp.draw("цель2028", M["ap"], s0net)

    # те же пороги без субсидий
    out = []
    saved = P["subsidy_on"]
    P["subsidy_on"] = False
    for fare in (1.3, 1.45, 1.6, 1.75):
        case = {**P["cases"]["базовый"], "fare": fare}
        s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
        net, log = sc.greedy(sc.base_network(M, "прямые"), M, "прямые", N, case)
        r = sc.evaluate(net, M, case)
        out.append({"доходность": fare, "гипотеза": "H2 прямые рейсы, без субсидий",
                    "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                    "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                    "рейсы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
        print(out[-1], flush=True)
    P["subsidy_on"] = saved
    pd.DataFrame(out).to_csv(RESULTS / "thresholds_без_субсидий.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
