"""
Сетка расчётов для интерактивного калькулятора (docs/calculator.html).
Модель не запускается в браузере: заранее перебираем условия и сохраняем, какую сеть строит алгоритм.

Условия: доходность к минимальным тарифам АТОР × стоимость блок-часа × субсидии да/нет.
Варианты: прямые рейсы; хаб в Самаре, Казани, Уфе; гибрид (хаб в Уфе + прямые) — тот же флот, 4 новых борта.

Запуск: python model/step10_calculator_grid.py   (≈ 40–60 минут; можно параллельно на нескольких ядрах)
Выход:  results/calculator_grid.json
"""
import json
import sys
from itertools import product
from multiprocessing import Pool

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc

DEFAULT_HUB = P["hub_city"]
import step2_choice as ch

FARES = [1.0, 1.15, 1.3, 1.45, 1.6, 1.75]
COSTS = [450_000, 500_000, 550_000, 600_000, 650_000]
SUBSIDY = [True, False]
MODES = [("прямые", "KUF"), ("хаб", "KUF"), ("хаб", "KZN"), ("хаб", "UFA"), ("гибрид", "UFA")]
N = P["new_aircraft"]["базовый"]
M = None
S0NET = None


def init():
    global M, S0NET
    M = sc.setup()
    S0NET = ch.network_from_boards(M["ap"])["direct"]


def one(task):
    fare, cost, sub, (mode, hub) = task
    P["cost_bh_rub"], P["subsidy_on"] = cost, sub
    sc.HUB = P["hub_city"] = hub
    case = {**P["cases"]["базовый"], "fare": fare}
    s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
    net, log = sc.greedy(sc.base_network(M, mode), M, mode, N, case)
    r = sc.evaluate(net, M, case, detail=True)
    legs = []
    for x in r["legs"]:
        old = S0NET.get((x["i"], x["j"]), {}).get("WZ", 0)
        if x["рейсов_нед"] > old:
            legs.append({"i": x["i"], "j": x["j"], "f0": old, "f": x["рейсов_нед"], "pax": x["пасс_в_год"]})
    key = f"{fare}|{cost}|{int(sub)}|{mode}|{hub}"
    return key, {"profit": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                 "aircraft": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 2),
                 "pax": round(r["пассажиров"] - s0["пассажиров"]),
                 "share_pp": round((r["перевозок"] - s0["перевозок"]) / (P["market_total_mln"] * 1e6) * 100, 3),
                 "access_pct": round((r["доступность"] / s0["доступность"] - 1) * 100, 2),
                 "new_legs": legs}


def main():
    tasks = list(product(FARES, COSTS, SUBSIDY, MODES))
    procs = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
    out = {}
    old = RESULTS / "calculator_grid.json"
    if "--reuse" in sys.argv and old.exists():      # досчитать только новые варианты
        out = json.loads(old.read_text(encoding="utf-8"))["grid"]
        tasks = [t for t in tasks if f"{t[0]}|{t[1]}|{int(t[2])}|{t[3][0]}|{t[3][1]}" not in out]
    with Pool(procs, initializer=init) as pool:
        for k, (key, res) in enumerate(pool.imap_unordered(one, tasks), 1):
            out[key] = res
            print(f"[{k}/{len(tasks)}] {key}: {res['profit']:+.3f} млрд, {res['aircraft']} борта", flush=True)
    init()
    ap = M["ap"]
    meta = {"fares": FARES, "costs": COSTS, "modes": [f"{m}|{h}" for m, h in MODES], "aircraft": N,
            "base_fare": P["cases"]["базовый"]["fare"], "base_cost": 600_000,
            "airports": {c: {"name": ap.at[c, "город"], "lat": float(ap.at[c, "lat"]), "lon": float(ap.at[c, "lon"]),
                             "mass": float(ap.at[c, "масса"])} for c in M["cities"]},
            "rw_current": [{"i": p[0], "j": p[1], "f": f} for p, f in M["s0_wz"].items() if p in M["km"]]}
    (RESULTS / "calculator_grid.json").write_text(json.dumps({"meta": meta, "grid": out}, ensure_ascii=False),
                                                  encoding="utf-8")
    print("Готово:", RESULTS / "calculator_grid.json")


if __name__ == "__main__":
    main()
