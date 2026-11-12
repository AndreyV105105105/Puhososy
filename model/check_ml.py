"""
ML-блок, шаг 3. Как ML-фильтр новых пар (params: ml_gate, p ≥ 0,5) меняет выбор между хабом и прямыми рейсами.

Варианты фильтра:
  «фильтр p ≥ 0,5»          — база: новую пару без рейсов можно открыть, только если ML даёт p ≥ 0,5;
  «фильтр только прямые»    — спицы хаба без фильтра (хаб возит стыковочных, ему фильтр может мешать);
  «без фильтра»             — как в версии 2 модели;
  «p как множитель спроса»  — отвергнутый способ: спрос пары без рейсов × p вместо ×0,3.
Сравниваем прямые рейсы, хаб в Самаре, Уфе и Казани и гибрид при выручке +30/+45/+60% к минимальным тарифам.

Запуск: python model/check_ml.py [номер варианта]  →  results/check_ml.csv (все варианты, если без аргумента)
"""
import sys
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc
from step7_hub_city import hub_mix

DEFAULT_HUB = P["hub_city"]
VARIANTS = [("прямые", DEFAULT_HUB), ("хаб", "KUF"), ("хаб", "UFA"), ("хаб", "KZN"), ("гибрид", "UFA")]
FILTERS = [("фильтр p ≥ 0,5", {}), ("фильтр только прямые", {"ml_gate_hub": False}),
           ("без фильтра", {"ml_gate": False}), ("p как множитель спроса", {"ml_gate": False, "unserved_mode": "ml"})]


def run(label, change):
    saved = {k: P[k] for k in change}
    P.update(change)
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
            rows.append({"ML": label, "вариант": mode if mode == "прямые" else f"{mode} в {P['hub_names'][hub]}",
                         "выручка_к_АТОР": fare, "%_стыковочных": mix.get("%_стыковочных"),
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "рейсы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(rows[-1], flush=True)
    sc.HUB = P["hub_city"] = DEFAULT_HUB
    P.update(saved)
    return rows


def main(which=None):
    todo = FILTERS if which is None else [FILTERS[which]]
    rows = [r for label, ch in todo for r in run(label, ch)]
    out = RESULTS / ("check_ml.csv" if which is None else f"check_ml_{which}.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
