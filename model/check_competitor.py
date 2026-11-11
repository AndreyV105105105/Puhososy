"""
Проверка устойчивости: что будет, если конкуренты ответят на рост Red Wings (план версии 2, шаг 3).
Модель сама ответ конкурентов не моделирует, поэтому проверяем три грубых сценария при том же флоте (4 борта):

  1) «отвечают тем же»: на каждой паре, где Red Wings добавила рейсы и уже летает другая авиакомпания,
     крупнейшая из них добавляет столько же рейсов. Сеть Red Wings та же, что без ответа.
  2) «новый перевозчик на Казань–Екатеринбург, 7 рейсов в неделю»: сейчас Red Wings там одна, и на этом
     маршруте держится первый шаг дорожной карты. Сеть Red Wings подбирается заново уже с конкурентом.
  3) то же, 14 рейсов в неделю.

Запуск: python model/check_competitor.py   →  results/check_competitor.csv
"""
import copy

import pandas as pd

from params import PARAMS as P
from common import RESULTS
import step3_scenarios as sc

N = P["new_aircraft"]["базовый"]
KEY = ("KZN", "SVX")          # маршрут первого шага; конкурентов на табло 10–16.11.2026 нет
ENTRANT = "XX"                # условный новый перевозчик


def delta(r, s0):
    return {"прибыль_млрд": round((r["прибыль"] - s0["прибыль"]) / 1e9, 3),
            "самолётов": round((r["блок_часы"] - s0["блок_часы"]) / sc.BH_PER_AIRCRAFT, 1),
            "новых_пассажиров": round(r["пассажиров"] - s0["пассажиров"])}


def added(log):
    return "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x.get("принят"))


def main():
    M = sc.setup()
    rows = []
    for fare in (1.3, 1.45, 1.6):
        case = {**P["cases"]["базовый"], "fare": fare}
        base = sc.base_network(M, "прямые")
        s0 = sc.evaluate(base, M, case)
        net, log = sc.greedy(sc.base_network(M, "прямые"), M, "прямые", N, case)
        rows.append({"сценарий": "без ответа конкурентов", "выручка_к_АТОР": fare, **delta(sc.evaluate(net, M, case), s0),
                     "кто_ответил": "", "рейсы": added(log)})

        # 1) конкуренты отвечают тем же числом рейсов на парах, где уже летают
        resp, who = copy.deepcopy(net), []
        for p, carriers in net["direct"].items():
            extra = carriers.get("WZ", 0) - M["s0_wz"].get(p, 0)
            rivals = {c: f for c, f in carriers.items() if c != "WZ"}
            if extra > 0 and rivals:
                c = max(rivals, key=rivals.get)
                resp["direct"][p][c] = rivals[c] + extra
                who.append(f"{p[0]}-{p[1]}: {c} +{extra:g}")
        rows.append({"сценарий": "конкуренты отвечают тем же числом рейсов", "выручка_к_АТОР": fare,
                     **delta(sc.evaluate(resp, M, case), s0), "кто_ответил": "; ".join(who), "рейсы": added(log)})

        # 2–3) новый перевозчик на маршруте первого шага; Red Wings подбирает сеть заново
        for f_new in (7, 14):
            def with_entrant():
                n = sc.base_network(M, "прямые")
                n["direct"].setdefault(KEY, {})[ENTRANT] = f_new
                return n
            s0e = sc.evaluate(with_entrant(), M, case)
            net_e, log_e = sc.greedy(with_entrant(), M, "прямые", N, case)
            d = delta(sc.evaluate(net_e, M, case), s0e)
            rows.append({"сценарий": f"новый перевозчик на Казань–Екатеринбург, {f_new} рейсов в неделю", "выручка_к_АТОР": fare, **d,
                         "кто_ответил": f"потеря Red Wings на текущей сети: {(s0e['прибыль'] - s0['прибыль']) / 1e9:+.3f} млрд",
                         "рейсы": added(log_e)})
        for r in rows[-4:]:
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "check_competitor.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
