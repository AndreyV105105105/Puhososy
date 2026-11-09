"""
ШАГ 7. При каких условиях окупается хаб в Самаре и прямые рейсы — «пороги».

Это ответ на вопрос организаторов: «При каких условиях новый региональный авиахаб окажется
эффективнее прямых рейсов?». Вместо одного ответа «да/нет» показываем границы.

Как: меняем один рычаг (доходность пассажира, стоимость блок-часа, стоимость базы, закрытия Самары)
и для каждого значения ЗАНОВО раскладываем самолёты жадным алгоритмом шага 3 (сеть подстраивается
под условия). Порог — значение, при котором прибыль варианта становится больше нуля.

Запуск: python model/step7_thresholds.py [рычаг]   (рычаг: доходность | блок_час | база | закрытия | all)
Выход:  results/thresholds_<рычаг>.csv
"""
import sys
import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc

N = P["new_aircraft"]["базовый"]
LEVERS = {
    # рычаг: (что меняем, значения, какие гипотезы считать)
    "доходность": ("case:fare", [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], ("хаб", "прямые")),
    "блок_час":   ("param:cost_bh_rub", [450e3, 500e3, 550e3, 600e3, 650e3], ("хаб", "прямые")),
    "база":       ("param:hub_base_fixed_rub_year", [0, 45e6, 118e6, 231e6], ("хаб",)),
    "закрытия":   ("case:hub_closed", [0.0, 0.035, 0.10], ("хаб",)),
}
NAMES = {"хаб": "H1 хаб в Самаре", "прямые": "H2 прямые рейсы"}


def run_lever(M, lever):
    what, values, modes = LEVERS[lever]
    kind, name = what.split(":")
    rows = []
    for v in values:
        case = dict(P["cases"]["базовый"])
        saved = P.get(name)
        if kind == "case":
            case[name] = v
        else:
            P[name] = v
        s0 = sc.evaluate(sc.base_network(M, "прямые"), M, case)
        for mode in modes:
            net, log = sc.greedy(sc.base_network(M, mode), M, mode, N, case)
            r = sc.evaluate(net, M, case)
            rows.append({"рычаг": lever, "значение": v, "гипотеза": NAMES[mode],
                         "прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
                         "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
                         "спиц_Самары": sum(1 for x in log if x.get("принят") and x["ход"] == "спица"),
                         "рейсы": "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))})
            print(f"  {lever}={v:g} {NAMES[mode]}: {rows[-1]['прибыль_млрд']:+.3f} млрд, "
                  f"{rows[-1]['самолётов']} борта | {rows[-1]['рейсы']}", flush=True)
        if kind == "param":
            P[name] = saved
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"thresholds_{lever}.csv", index=False, encoding="utf-8-sig")
    return df


def main(args=None):
    args = args if args is not None else sys.argv[1:]
    levers = list(LEVERS) if not args or args[0] == "all" else args
    M = sc.setup()
    for lv in levers:
        print(f"\n== Рычаг: {lv} ==")
        run_lever(M, lv)


if __name__ == "__main__":
    main()
