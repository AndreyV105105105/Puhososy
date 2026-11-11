"""
ШАГ 6. Устойчивость вывода: не поменяется ли победитель, если наши допущения неверны.

Две проверки:
  A) По одному параметру: меняем одно допущение (спрос, тариф, стоимость часа, субсидии…),
     остальные оставляем — и смотрим, какая гипотеза лучше.
  B) Монте-Карло: 500 раз случайно выбираем ВСЕ допущения сразу в их диапазонах
     и считаем, в какой доле случаев каждая гипотеза лучшая. Это «вероятность», что вывод верен.

Сети маршрутов строим при доходности +45% (SENS_FARE — целевая сеть дорожной карты; при сегодняшней
доходности новых рейсов нет, проверять нечего) и потом не перестраиваем: проверяем, выдержит ли уже
выбранное решение другие условия.

Запуск: python model/step6_sensitivity.py   (после step3_scenarios.py)
Выход:  results/sensitivity.csv, results/monte_carlo.csv, results/routes_table.csv
"""
import copy
import numpy as np
import pandas as pd

from params import PARAMS as P
from common import RESULTS, DATA
import step2_choice as ch
import step3_scenarios as sc

MODES = {"хаб": "H1 хаб", "прямые": "H2 прямые", "гибрид": "H3 гибрид"}
SENS_FARE = 1.45
BASE_CASE = {**P["cases"]["базовый"], "fare": SENS_FARE}


def load_network(M, mode):
    """Сеть гипотезы из шага 3: текущая сеть + новые рейсы Red Wings."""
    net = sc.base_network(M, mode)
    legs = pd.read_csv(RESULTS / f"network_{mode}_доходность{round(SENS_FARE * 100)}.csv", encoding="utf-8-sig")
    for _, r in legs.iterrows():
        p = (r.i, r.j)
        net["direct"].setdefault(p, {})["WZ"] = r["рейсов_нед"]
        if net.get("hub") and sc.HUB in p:
            other = r.i if r.j == sc.HUB else r.j
            if r["рейсов_нед"] > net["hub"]["spokes"].get(other, 0):
                net["hub"]["active"] = True          # открыта новая спица → база в Самаре работает
            net["hub"]["spokes"][other] = r["рейсов_нед"]
    return net


def profits(M, nets, s0, case):
    """Прирост прибыли каждой гипотезы против текущей сети при заданных условиях, млрд ₽."""
    M, asc = recalibrated(M, P["vot_rub_h"], P["b_freq"])
    saved_asc = (P["asc_train"], P["asc_car"])
    P["asc_train"], P["asc_car"] = asc
    base = sc.evaluate(s0, M, case)["прибыль"]
    out = {MODES[m]: (sc.evaluate(n, M, case)["прибыль"] - base) / 1e9 for m, n in nets.items()}
    out["S0 не расширяться"] = 0.0          # честная альтернатива: если все в минусе, лучше не открывать рейсы
    P["asc_train"], P["asc_car"] = saved_asc
    return out


CAL_CACHE = {}
BASE_VOT, BASE_BFREQ = P["vot_rub_h"], P["b_freq"]


def recalibrated(M, vot, b_freq):
    """Стоимость часа и вес частоты участвуют в калибровке шага 2 (константа поезда и множитель рынка).
    Меняя их, надо заново откалибровать модель по табло — иначе сравнение нечестное.
    Калибруем на сетке значений один раз и берём ближайший узел."""
    if abs(vot - BASE_VOT) < 1e-9 and abs(b_freq - BASE_BFREQ) < 1e-9:
        return M, (P["asc_train"], P["asc_car"])     # базовые значения — калибровка шага 2 как есть
    gv = min((500, 625, 750, 875, 1000), key=lambda x: abs(x - vot))
    gf = min((1.0, 1.25, 1.35, 1.5), key=lambda x: abs(x - b_freq))
    if (gv, gf) not in CAL_CACHE:
        saved = {k: P[k] for k in ("vot_rub_h", "b_freq", "asc_train", "asc_car")}
        P.update(vot_rub_h=gv, b_freq=gf); ch.B_FARE = P["b_time"] / gv
        scale, best = ch.calibrate(ch.network_from_boards(M["ap"]), M["ap"], M["fc"], ch.demand_model(), quick=True)
        CAL_CACHE[(gv, gf)] = (scale, (P["asc_train"], P["asc_car"]))
        P.update(saved); ch.B_FARE = P["b_time"] / P["vot_rub_h"]
    scale, asc = CAL_CACHE[(gv, gf)]
    M2 = dict(M)
    M2["demand"] = {k: v / M["scale"] * scale for k, v in M["demand"].items()}
    return M2, asc


def with_params(changes):
    """Временно поменять параметры (и пересчитать вес цены, если меняется стоимость часа)."""
    saved = {k: copy.deepcopy(P[k]) for k in changes}
    P.update(changes)
    ch.B_FARE = P["b_time"] / P["vot_rub_h"]
    return saved


def restore(saved):
    P.update(saved)
    ch.B_FARE = P["b_time"] / P["vot_rub_h"]


def best_of(r):
    """Лучший вариант; при равенстве (например, все по нулям) — «не расширяться»: зачем что-то менять."""
    top = max(r.values())
    return "S0 не расширяться" if r["S0 не расширяться"] >= top - 1e-6 else max(r, key=r.get)


def one_at_a_time(M, nets, s0):
    tests = [  # (что меняем, изменения параметров, изменения сценария)
        ("база", {}, {}),
        ("спрос −20%", {}, {"demand": 0.8}), ("спрос +20%", {}, {"demand": 1.2}),
        ("доходность «Азимута» (+15%)", {}, {"fare": P["cases"]["базовый"]["fare"]}),
        ("тариф −15%", {}, {"fare": BASE_CASE["fare"] * 0.85}), ("тариф +10%", {}, {"fare": BASE_CASE["fare"] * 1.10}),
        ("блок-час 570 тыс.", {"cost_bh_rub": 570_000}, {}), ("блок-час 690 тыс.", {"cost_bh_rub": 690_000}, {}),
        ("стоимость часа пасс. 500", {"vot_rub_h": 500}, {}), ("стоимость часа пасс. 1000", {"vot_rub_h": 1000}, {}),
        ("вес частоты 1,0", {"b_freq": 1.0}, {}), ("вес частоты 1,5", {"b_freq": 1.5}, {}),
        ("билет через хаб 80% прямого", {"hub_fare_factor": 0.8}, {}), ("билет через хаб 100%", {"hub_fare_factor": 1.0}, {}),
        ("без субсидий", {"subsidy_on": False}, {}), ("субсидии −7,6%", {}, {"subsidy": 0.924}),
        ("перегонов нет", {"ferry_share": 0.0}, {}), ("перегон на каждый оборот", {"ferry_share": 1.0}, {}),
        ("база хаба 45 млн/год", {"hub_base_fixed_rub_year": 45e6}, {}), ("база хаба 231 млн/год", {"hub_base_fixed_rub_year": 231e6}, {}),
        ("базы ещё Пермь и Омск", {"rw_bases": ["SVX", "MOW", "PEE", "OMS"]}, {}),
        ("спрос на парах без рейсов ×0,5", {"unserved_demand_factor": 0.5}, {}),
        ("спрос на парах без рейсов ×1 (без поправки)", {"unserved_demand_factor": 1.0}, {}),
        ("тариф коротких пар ×0,5", {"short_fare_factor": 0.5}, {}), ("тариф коротких пар ×0,7", {"short_fare_factor": 0.7}, {}),
        ("тариф коротких пар ×1 (как в первой версии)", {"short_fare_factor": 1.0}, {}),
        ("до чужого аэропорта до 400 км", {"ground_max_km": 400}, {}),
        ("до чужого аэропорта не ездят", {"ground_max_km": 0}, {}),
        ("хаб закрыт 2%", {}, {"hub_closed": 0.02}), ("хаб закрыт 5%", {}, {"hub_closed": 0.05}),
        ("хаб закрыт 10%", {}, {"hub_closed": 0.10}),
    ]
    rows = []
    for name, pch, cch in tests:
        saved = with_params(pch)
        r = profits(M, nets, s0, {**BASE_CASE, **cch})
        restore(saved)
        rows.append({"условие": name, **{k: round(v, 3) for k, v in r.items()}, "лучшая": best_of(r)})
    return pd.DataFrame(rows)


def monte_carlo(M, nets, s0, n=500, seed=42):
    rng = np.random.default_rng(seed)
    rng_short = np.random.default_rng(seed + 1)   # отдельный генератор: остальные случайные допущения те же, что раньше
    rows = []
    for _ in range(n):
        pch = {"cost_bh_rub": rng.uniform(570_000, 690_000), "vot_rub_h": rng.uniform(500, 1000),
               "b_freq": rng.uniform(1.0, 1.5), "hub_fare_factor": rng.uniform(0.8, 1.0),
               "ferry_share": rng.uniform(0.0, 1.0), "hub_base_fixed_rub_year": rng.uniform(45e6, 231e6)}
        pch["short_fare_factor"] = rng_short.uniform(0.5, 0.7)   # тариф пар короче 550 км (research/fares_real_2026-11.md)
        cch = {"demand": rng.uniform(0.8, 1.2), "fare": BASE_CASE["fare"] * rng.uniform(0.85, 1.1),
               "subsidy": rng.uniform(0.9, 1.0), "hub_closed": rng.uniform(0.0, 0.10)}
        saved = with_params(pch)
        r = profits(M, nets, s0, {**BASE_CASE, **cch})
        restore(saved)
        rows.append({**pch, **cch, **r, "лучшая": best_of(r)})
    return pd.DataFrame(rows)


def routes_table(M, mode="гибрид"):
    """Таблица «маршрут · рейсов в неделю · пассажиров в год · источник» — требование ТЗ для каждого А→Б."""
    legs = pd.read_csv(RESULTS / f"network_{mode}_доходность{round(SENS_FARE * 100)}.csv", encoding="utf-8-sig")
    s0 = ch.network_from_boards(M["ap"])["direct"]
    routes = pd.read_csv(DATA / "routes.csv", encoding="utf-8-sig")
    url = {}
    for _, r in routes[routes["перевозчик"] == "Red Wings"].iterrows():
        url[sc.key(r["откуда"], r["куда"])] = r["источник_частоты_url"]
    out = []
    for _, r in legs.iterrows():
        p = (r.i, r.j)
        old = s0.get(p, {}).get("WZ", 0)
        out.append({"маршрут": f"{r.i} ↔ {r.j}", "км": r["км"], "рейсов_нед_сейчас": old, "рейсов_нед_в_решении": r["рейсов_нед"],
                    "статус": "действующий" if old >= r["рейсов_нед"] else ("новый" if old == 0 else "больше рейсов"),
                    "пассажиров_в_год": r["пасс_в_год"], "загрузка": r["загрузка"],
                    "источник_частоты": url.get(p, "Яндекс Расписания, табло 10–16.11.2026") if old else "решение модели (step3_scenarios.py)",
                    "источник_пассажиропотока": "оценка модели: гравитация по табло 40 аэропортов (step1) × логит выбора (step2)"})
    pd.DataFrame(out).to_csv(RESULTS / "routes_table.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(out)


def main():
    M = sc.setup()
    cal = pd.read_csv(RESULTS / "choice_calibration.csv", encoding="utf-8-sig")
    M["scale"] = float(cal.loc[cal["ошибка_лог"].idxmin(), "scale"])
    for m in MODES:                                  # сети при доходности SENS_FARE
        net, _ = sc.greedy(sc.base_network(M, m), M, m, P["new_aircraft"]["базовый"], BASE_CASE)
        r = sc.evaluate(net, M, BASE_CASE, detail=True)
        pd.DataFrame(r["legs"]).to_csv(RESULTS / f"network_{m}_доходность{round(SENS_FARE * 100)}.csv",
                                       index=False, encoding="utf-8-sig")
    nets = {m: load_network(M, m) for m in MODES}
    s0 = sc.base_network(M, "прямые")
    pd.set_option("display.width", 220)

    oat = one_at_a_time(M, nets, s0)
    oat.to_csv(RESULTS / "sensitivity.csv", index=False, encoding="utf-8-sig")
    print("A) По одному параметру — прирост прибыли, млрд ₽/год:")
    print(oat.to_string(index=False))

    mc = monte_carlo(M, nets, s0)
    mc.round(4).to_csv(RESULTS / "monte_carlo.csv", index=False, encoding="utf-8-sig")
    print(f"\nB) Монте-Карло, {len(mc)} случайных наборов допущений:")
    win = mc["лучшая"].value_counts(normalize=True)
    print(f"  S0 «не расширяться» лучше всех в {win.get('S0 не расширяться', 0):.0%} случаев")
    for h in MODES.values():
        q = mc[h].quantile([0.05, 0.5, 0.95]).round(2).tolist()
        print(f"  {h}: лучшая в {win.get(h, 0):.0%} случаев; прибыль 5% / медиана / 95%: {q} млрд; "
              f"в плюсе в {(mc[h] > 0).mean():.0%}")

    # порог надёжности хаба: при какой доле закрытий гибрид перестаёт быть лучше прямых
    rows = []
    for c in np.arange(0, 0.101, 0.005):
        r = profits(M, nets, s0, {**BASE_CASE, "hub_closed": c})
        rows.append((c, r["H3 гибрид"] - r["H2 прямые"]))
    th = next((c for c, d in rows if d < 0), None)
    if th is None:
        print("\nПорог надёжности: гибрид лучше прямых при любой доле закрытий хаба от 0 до 10%")
    elif th == 0:
        print("\nПорог надёжности: гибрид не лучше прямых даже при 0% закрытий хаба")
    else:
        print(f"\nПорог надёжности: гибрид лучше прямых, пока хаб закрыт меньше ~{th:.1%} времени (сейчас 10%)")

    winner = {v: k for k, v in MODES.items()}[oat.iloc[0]["лучшая"]] if oat.iloc[0]["лучшая"] in MODES.values() else "прямые"
    rt = routes_table(M, winner)
    print(f"\nТаблица маршрутов победителя ({winner}): {len(rt)} маршрутов → results/routes_table.csv")
    print(rt[["маршрут", "рейсов_нед_сейчас", "рейсов_нед_в_решении", "статус", "пассажиров_в_год"]].to_string(index=False))


if __name__ == "__main__":
    main()
