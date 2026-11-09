"""
Проверка устойчивости выбора города хаба к сомнительной цифре: пассажиропоток Владикавказа
(0,94 млн за 2024 — один источник, помечено «сверить», замечание Семёныча).
Варианты: как в данных / 0,5 млн / без Владикавказа. Хаб в Волгограде, Уфе, Казани, Самаре.

Запуск: python model/check_ogz.py → results/hub_city_ogz_check.csv
"""
import sys
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import common
import step3_scenarios as sc

CITIES = ["VOG", "UFA", "KZN", "KUF"]
N = P["new_aircraft"]["базовый"]
_orig_load = common.load_airports


def run(variant):
    def patched():
        df = _orig_load()
        if variant == "0,5 млн":
            df.loc["OGZ", "масса"] = 0.5
        elif variant == "без Владикавказа":
            df.loc["OGZ", "масса"] = float("nan")
        return df
    sc.load_airports = patched
    import step2_choice as ch
    ch._GEO.clear()
    M = sc.setup()
    rows = []
    for fare in (1.45, 1.6):
        case = {**P["cases"]["базовый"], "fare": fare}
        s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
        for c in CITIES:
            sc.HUB = P["hub_city"] = c
            net, log = sc.greedy(sc.base_network(M, "хаб"), M, "хаб", N, case)
            r = sc.evaluate(net, M, case)
            rows.append({"Владикавказ": variant, "доходность": fare, "город_хаба": M["ap"].at[c, "город"],
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "спицы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(rows[-1], flush=True)
    sc.HUB = P["hub_city"] = "KUF"
    return rows


def main():
    variants = sys.argv[1:] or ["как в данных", "0,5 млн", "без Владикавказа"]
    rows = [r for v in variants for r in run(v)]
    out = RESULTS / "hub_city_ogz_check.csv"
    old = pd.read_csv(out, encoding="utf-8-sig") if out.exists() and sys.argv[1:] else None
    df = pd.DataFrame(rows) if old is None else pd.concat([old, pd.DataFrame(rows)])
    df.to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
