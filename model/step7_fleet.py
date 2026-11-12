"""
ШАГ 7. Сценарии по числу новых самолётов: от осторожного до максимума, который Red Wings может получить в теории.

Замечание экспертов на чекпоинте 12.11.2026: посчитать не только 2 и 4 борта, но и максимум.
Откуда числа (research/sj100_deliveries.md): в публичных планах около 42 SJ-100 до 2029 года на три авиакомпании
(Red Wings, «Азимут», «Аврора»), разбивки нет. Поровну — около 14 на авиакомпанию; «большую часть» обещают
Red Wings и «Азимуту» — тогда до 21. Сетка: 2 (осторожный), 4 (база), 8, 14 (треть поставок), 21 (половина).

Для каждого числа бортов и уровня выручки заново раскладываем самолёты жадным алгоритмом шага 3:
прямые рейсы, хаб в Уфе и Казани, гибрид. «Самолётов занято» показывает, упирается ли сеть во флот
(если занято меньше, чем дано, лишние борта не окупаются на внутренних рейсах между нашими городами).

Запуск: python model/step7_fleet.py [N]   →  results/fleet_scenarios.csv (results/fleet_N.csv для одного N)
"""
import sys
import time
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc
from step7_hub_city import hub_mix

FLEETS = [2, 4, 8, 14, 21]
FARES = [1.3, 1.45, 1.6, 1.8, 2.0]
DEFAULT_HUB = P["hub_city"]
# (вариант, город хаба, спицы, которые запрещены). Минводы–Уфа не выдерживает ответа «Азимута»
# (results/check_competitor.csv), поэтому гибрид считаем и без этой спицы.
VARIANTS = [("прямые", DEFAULT_HUB, []), ("хаб", "UFA", []), ("хаб", "KZN", []), ("гибрид", "UFA", []),
            ("гибрид", "UFA", ["MRV"])]


def run(n, M):
    rows = []
    saved_ex = P["hub_spoke_exclude"]
    for mode, hub, excl in VARIANTS:
        sc.HUB = P["hub_city"] = hub
        P["hub_spoke_exclude"] = {**saved_ex, hub: list(saved_ex.get(hub, [])) + excl}
        name = mode if mode == "прямые" else f"{mode} в {P['hub_names'][hub]}" + (" без Минвод" if excl else "")
        for fare in FARES:
            t = time.time()
            case = {**P["cases"]["базовый"], "fare": fare}
            s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
            net, log = sc.greedy(sc.base_network(M, mode), M, mode, n, case)
            r = sc.evaluate(net, M, case)
            mix = hub_mix(r, hub) if net.get("hub") and net["hub"].get("active") else {}
            rows.append({"бортов_дано": n, "выручка_к_АТОР": fare,
                         "вариант": name,
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "бортов_занято": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "новых_пассажиров_тыс": round((r["пассажиров"] - s0["пассажиров"]) / 1e3),
                         "прирост_доли_пп": round((r["перевозок"] - s0["перевозок"]) / (P["market_total_mln"] * 1e6) * 100, 3),
                         "%_стыковочных": mix.get("%_стыковочных"),
                         "рейсы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(f"[{time.time() - t:.0f} с] {rows[-1]}", flush=True)
    sc.HUB = P["hub_city"] = DEFAULT_HUB
    P["hub_spoke_exclude"] = saved_ex
    return rows


def main(fleets=FLEETS):
    M = sc.setup()
    rows = [r for n in fleets for r in run(n, M)]
    out = RESULTS / ("fleet_scenarios.csv" if len(fleets) > 1 else f"fleet_{fleets[0]}.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main([int(sys.argv[1])] if len(sys.argv) > 1 else FLEETS)
