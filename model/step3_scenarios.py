"""
ШАГ 3. Сценарии: куда поставить новые самолёты — хаб, прямые рейсы или гибрид.

Сравнение честное: в каждом сценарии одинаковое число новых самолётов (N), и самолёты
распределяет один и тот же алгоритм. Отличается только то, что ему разрешено строить:
  H1 «хаб»     — рейсы из Самары («спицы»), пассажиры пересаживаются в Самаре волнами;
  H2 «прямые»  — прямые рейсы между любыми городами, без пересадок в Самаре;
  H3 «гибрид»  — и то и другое; алгоритм сам решает, что выгоднее.

Жадный алгоритм (как раскладывают самолёты на практике — «самое выгодное сначала»):
  пока есть свободные блок-часы новых самолётов:
     перебрать все ходы «открыть рейс / поднять частоту» (3 → 7 → 14 рейсов в неделю),
     для каждого пересчитать всю сеть (шаг 2) и прибыль Red Wings,
     выбрать ход с наибольшей прибылью на блок-час; если прибыльных ходов нет — остановиться.

Деньги:  прибыль = выручка (пассажиры × тариф) − блок-часы × стоимость часа + субсидии ПП 1242
Флот:    самолётов = блок-часы в год / (налёт 6,3 ч/сут × 365)
Доля:    прирост пассажиро-перевозок Red Wings / 108,9 млн (стыковочный пассажир = 2 перевозки, как в статистике)

Запуск: python model/step3_scenarios.py   (после step1 и step2)
Выход:  results/scenarios_summary.csv, results/network_<гипотеза>_<сценарий>.csv
"""
import copy
import math
import time
import numpy as np
import pandas as pd

from params import PARAMS as P
from common import load_airports, dist_km, block_h, RESULTS
import step2_choice as ch

HUB = P["hub_city"]
BH_PER_AIRCRAFT = P["util_h_day"] * 365     # блок-часов в год на один самолёт


# ---------------- подготовка ----------------
def setup():
    ap = load_airports()
    cal = pd.read_csv(RESULTS / "choice_calibration.csv", encoding="utf-8-sig")
    best = cal.loc[cal["ошибка_лог"].idxmin()]
    P["asc_train"] = float(best["asc_train"])          # константа поезда из калибровки шага 2
    scale = float(best["scale"])
    ln_k, a, theta, _ = ch.demand_model()
    cities = [c for c in ap.index if pd.notna(ap.loc[c, "пасс_2025_млн"])]
    coords = {c: (ap.loc[c, "lat"], ap.loc[c, "lon"]) for c in ap.index}
    km = {(i, j): dist_km(*coords[i], *coords[j]) for i in cities for j in cities if i < j}
    m = ap["пасс_2025_млн"]
    demand = {p: scale * math.exp(ln_k + a * math.log(m[p[0]] * m[p[1]]) - theta * math.log(k))
              for p, k in km.items()}
    return dict(ap=ap, fc=ch.fare_model(ap), cities=cities, coords=coords, km=km, demand=demand)


def key(i, j):
    return (i, j) if i < j else (j, i)


def subsidy_per_flight(k):
    if not P["subsidy_on"] or k > P["subsidy_max_km"]:
        return 0.0
    xs, ys = zip(*P["subsidy_points"])
    return float(np.interp(k, xs, ys))


def rw_legs(net, M):
    """Рейсы Red Wings внутри модели: {пара: рейсов в неделю в одну сторону}."""
    return {p: c["WZ"] for p, c in net["direct"].items() if "WZ" in c and p in M["km"]}


def block_hours(net, M):
    return sum(2 * f * 52 * block_h(M["km"][p]) for p, f in rw_legs(net, M).items())


# ---------------- оценка сети ----------------
def evaluate(net, M, case, detail=False):
    """Пересчитать всю сеть: пассажиры Red Wings, деньги, время пассажиров."""
    hub = net.get("hub")
    fare_k = case["fare"]
    pairs, load = [], {}
    hours_total = moscow = 0.0
    connected = 0
    for (i, j), T0 in M["demand"].items():
        T = T0 * case["demand"]
        _, opts = ch.options(i, j, M["ap"], net, M["fc"])
        d = T * sum(o["доля"] for o in opts if o["вариант"] == "прямой" and o["перевозчик"] == "WZ")
        h = T * sum(o["доля"] for o in opts if o["вариант"] == "через хаб")
        hours_total += T * sum(o["доля"] * o["часы"] for o in opts)
        moscow += T * sum(o["доля"] for o in opts if o["вариант"] == "через Москву")
        connected += any(o["вариант"] in ("прямой", "через хаб") for o in opts)
        pairs.append((i, j, d, h))
        if d:
            load[(i, j)] = load.get((i, j), 0) + d
        if h:
            for leg in (key(i, HUB), key(HUB, j)):
                load[leg] = load.get(leg, 0) + h

    # вместимость: если пассажиров больше, чем мест, лишние «не помещаются» (спилл)
    legs = rw_legs(net, M)
    fit = {}
    for p, f in legs.items():
        cap = 2 * f * 52 * P["seats_rw"] * P["lf_max"]
        if HUB in p and hub:
            cap *= 1 - case["hub_closed"]           # закрытия аэропорта хаба съедают часть рейсов
        fit[p] = min(1.0, cap / load[p]) if load.get(p) else 1.0

    rev = trips = segs = 0.0
    carried = {p: 0.0 for p in legs}
    for i, j, d, h in pairs:
        k = M["km"][(i, j)]
        if d:
            dc = d * fit[(i, j)]
            rev += dc * ch.fare(k, 1, M["fc"]) * fare_k
            trips += dc; segs += dc; carried[(i, j)] += dc
        if h:
            l1, l2 = key(i, HUB), key(HUB, j)
            hc = h * min(fit[l1], fit[l2])
            rev += hc * P["hub_fare_factor"] * ch.fare(k, 1, M["fc"]) * fare_k
            trips += hc; segs += 2 * hc
            carried[l1] += hc; carried[l2] += hc

    bh = block_hours(net, M)
    cost = bh * (P["cost_bh_rub"] + P["opportunity_rub_per_bh"])
    sub = sum(2 * f * 52 * subsidy_per_flight(M["km"][p]) for p, f in legs.items()) * case["subsidy"]
    res = dict(пассажиров=trips, перевозок=segs, выручка=rev, затраты=cost, субсидии=sub,
               прибыль=rev - cost + sub, блок_часы=bh, пасс_часы=hours_total, через_Москву=moscow,
               пар_без_Москвы=connected)
    if detail:
        res["legs"] = [dict(i=p[0], j=p[1], рейсов_нед=f, км=round(M["km"][p]), пасс_в_год=round(carried[p]),
                            загрузка=round(carried[p] / (2 * f * 52 * P["seats_rw"]), 2),
                            блок_часов_в_год=round(2 * f * 52 * block_h(M["km"][p])))
                       for p, f in legs.items()]
    return res


# ---------------- ходы алгоритма ----------------
def next_level(cur, levels=None):
    for lv in levels or P["freq_levels"]:
        if lv > cur:
            return lv
    return None


def candidate_moves(net, M, mode):
    moves = []
    if mode in ("хаб", "гибрид"):
        for x in M["cities"]:
            if x == HUB:
                continue
            nxt = next_level(net["hub"]["spokes"].get(x, 0), P["spoke_levels"])
            if nxt:
                moves.append(("спица", key(HUB, x), nxt))
    if mode in ("прямые", "гибрид"):
        for p in M["km"]:
            if mode == "гибрид" and HUB in p:
                continue                              # рейсы из Самары в гибриде — это спицы
            nxt = next_level(net["direct"].get(p, {}).get("WZ", 0))
            if nxt:
                moves.append(("прямой", p, nxt))
    return moves


def apply(net, move):
    kind, p, f = move
    new = copy.deepcopy(net)
    new["direct"].setdefault(p, {})["WZ"] = f
    if kind == "спица":
        other = p[0] if p[1] == HUB else p[1]
        new["hub"]["spokes"][other] = f
    return new


def greedy(net0, M, mode, n_aircraft, case):
    """Раскладываем N новых самолётов: каждый раз берём ход с наибольшей прибылью на блок-час.
    Не останавливаемся на первом убыточном ходе: у хаба отдача растёт с числом спиц (стыковки
    появляются, только когда спиц несколько). Идём до конца флота, а потом берём лучший момент остановки."""
    budget = n_aircraft * BH_PER_AIRCRAFT
    bh0 = block_hours(net0, M)
    net, cur = net0, evaluate(net0, M, case)
    history = [(net0, cur["прибыль"])]
    log = []
    while True:
        best = None
        bh_now = block_hours(net, M)
        for mv in candidate_moves(net, M, mode):
            cand = apply(net, mv)
            bh = block_hours(cand, M)
            if bh - bh0 > budget:
                continue
            r = evaluate(cand, M, case)
            gain = (r["прибыль"] - cur["прибыль"]) / max(bh - bh_now, 1)
            if best is None or gain > best[0]:
                best = (gain, mv, cand, r)
        if best is None:
            break
        gain, mv, net, cur = best
        history.append((net, cur["прибыль"]))
        log.append(dict(ход=mv[0], пара=f"{mv[1][0]}-{mv[1][1]}", рейсов_нед=mv[2],
                        прибыль_на_блок_час=round(gain), прибыль_сети_млрд=round(cur["прибыль"] / 1e9, 3),
                        самолётов_занято=round((block_hours(net, M) - bh0) / BH_PER_AIRCRAFT, 2)))
    k = int(np.argmax([h[1] for h in history]))      # лучший момент остановки
    for x in log[k:]:
        x["принят"] = False
    for x in log[:k]:
        x["принят"] = True
    return history[k][0], log


# ---------------- запуск ----------------
def base_network(M, mode):
    net = ch.network_from_boards(M["ap"])
    if mode in ("хаб", "гибрид"):
        # действующие рейсы Red Wings из Самары становятся первыми спицами хаба
        spokes = {(p[0] if p[1] == HUB else p[1]): c["WZ"] for p, c in net["direct"].items()
                  if HUB in p and "WZ" in c and p in M["km"]}
        net["hub"] = {"city": HUB, "spokes": spokes}
    return net


def main():
    t0 = time.time()
    M = setup()
    rows = []
    for case_name, case in P["cases"].items():
        n = P["new_aircraft"][case_name]
        s0 = evaluate(base_network(M, "прямые"), M, case)          # S0: текущая сеть, без хаба
        for mode in ("хаб", "прямые", "гибрид"):
            net, log = greedy(base_network(M, mode), M, mode, n, case)
            r = evaluate(net, M, case, detail=True)
            d = lambda k: r[k] - s0[k]
            rows.append({
                "гипотеза": {"хаб": "H1 хаб в Самаре", "прямые": "H2 прямые рейсы", "гибрид": "H3 гибрид"}[mode],
                "сценарий": case_name, "новых_самолётов_дано": n,
                "самолётов_занято": round((r["блок_часы"] - s0["блок_часы"]) / BH_PER_AIRCRAFT, 1),
                "новых_пассажиров_в_год": round(d("пассажиров")),
                "новых_перевозок_в_год": round(d("перевозок")),
                "выручка_млрд": round(d("выручка") / 1e9, 2),
                "прибыль_с_субсидиями_млрд": round(d("прибыль") / 1e9, 2),
                "субсидии_млрд": round(d("субсидии") / 1e9, 2),
                "прирост_доли_пп": round(d("перевозок") / (P["market_total_mln"] * 1e6) * 100, 3),
                "сэкономлено_пасс_часов_млн": round(-d("пасс_часы") / 1e6, 2),
                "доля_через_Москву": round(r["через_Москву"] / s0["через_Москву"] - 1, 3),
                "пар_без_Москвы_было_стало": f"{s0['пар_без_Москвы']} → {r['пар_без_Москвы']}",
            })
            tag = f"{mode}_{case_name}"
            pd.DataFrame(r["legs"]).assign(
                lat_i=lambda x: x.i.map(lambda c: M["coords"][c][0]), lon_i=lambda x: x.i.map(lambda c: M["coords"][c][1]),
                lat_j=lambda x: x.j.map(lambda c: M["coords"][c][0]), lon_j=lambda x: x.j.map(lambda c: M["coords"][c][1]),
            ).to_csv(RESULTS / f"network_{tag}.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(log).to_csv(RESULTS / f"moves_{tag}.csv", index=False, encoding="utf-8-sig")
            print(f"[{time.time()-t0:.0f} с] {tag}: принято {sum(x['принят'] for x in log)} из {len(log)} ходов: "
                  + "; ".join(f"{x['пара']}×{x['рейсов_нед']}" for x in log if x["принят"]))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "scenarios_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
