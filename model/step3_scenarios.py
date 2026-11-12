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
    P["asc_train"] = float(best["asc_train"])          # константы поезда и машины из калибровки шага 2
    P["asc_car"] = float(best["asc_car"])
    scale = float(best["scale"])
    ln_k, a, theta, _, vakhta_coef = ch.demand_model()
    cities = [c for c in ap.index if pd.notna(ap.loc[c, "масса"])]
    coords = {c: (ap.loc[c, "lat"], ap.loc[c, "lon"]) for c in ap.index}
    km = {(i, j): dist_km(*coords[i], *coords[j]) for i in cities for j in cities if i < j}
    m = ap["масса"]
    vk = lambda p: vakhta_coef if (p[0] in P["vakhta_cities"] or p[1] in P["vakhta_cities"]) else 0.0
    demand = {p: scale * math.exp(ln_k + a * math.log(m[p[0]] * m[p[1]]) - theta * math.log(k) + vk(p))
              for p, k in km.items()}
    top = sorted(demand, key=demand.get, reverse=True)[:P["direct_candidates_top"]]
    cand = set(top) | {p for p in km if P["existing_hub"] in p}
    s0 = ch.network_from_boards(ap)["direct"]
    s0_wz = {p: c["WZ"] for p, c in s0.items() if "WZ" in c}
    by_city = {}
    for p in km:
        by_city.setdefault(p[0], []).append(p)
        by_city.setdefault(p[1], []).append(p)
    return dict(ap=ap, fc=ch.fare_model(ap), cities=cities, coords=coords, km=km, demand=demand,
                direct_candidates=cand, pairs_by_city=by_city, s0_wz=s0_wz,
                served={p for p in s0 if p in km}, ml_p=ml_viability_p())


def ml_viability_p():
    """p жизнеспособности прямого рейса по парам из ML-блока (results/ml_viability.csv), если файл есть."""
    path = RESULTS / "ml_viability.csv"
    if not path.exists():
        return {}
    d = pd.read_csv(path, encoding="utf-8-sig")
    return {key(i, j): float(p) for i, j, p in zip(d["i"], d["j"], d["p"])}


def unserved_factor(p, M):
    """Поправка спроса для пары без рейсов сейчас: ×0,3 (по трём снятым рейсам) или p из ML-блока."""
    if p in M["served"]:
        return 1.0
    if P.get("unserved_mode") == "ml":
        if not M.get("ml_p"):
            raise RuntimeError("unserved_mode='ml': нет results/ml_viability.csv — запустите ml_dataset.py и ml_viability.py")
        return M["ml_p"][p]
    return P["unserved_demand_factor"]


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


def bases(net):
    """Где базируются самолёты: базы из параметров + Самара, если хаб уже запущен (открыта новая спица)."""
    b = list(P["rw_bases"])
    if net.get("hub") and net["hub"].get("active"):
        b.append(HUB)
    return b


def ferry_km(p, net, M):
    """Сколько лететь пустым до рейса: 0, если рейс касается базы, иначе от ближайшей базы до ближайшего конца."""
    bs = bases(net)
    if p[0] in bs or p[1] in bs:
        return 0.0
    pt = lambda c: P["moscow_point"] if c == "MOW" else M["coords"][c]
    return min(dist_km(*pt(b), *M["coords"][c]) for b in bs for c in p)


def block_hours(net, M, split=False):
    """Блок-часы Red Wings в год: рейсы с пассажирами + перегоны без пассажиров."""
    flown = ferry = 0.0
    for p, f in rw_legs(net, M).items():
        flown += 2 * f * 52 * block_h(M["km"][p])
        # перегоны считаем только для ДОБАВЛЕННЫХ рейсов: действующая сеть Red Wings уже как-то организована
        # (она летает сегодня), и «экономить» на её воображаемых перегонах нельзя
        extra = max(0.0, f - M["s0_wz"].get(p, 0.0))
        fk = ferry_km(p, net, M) if extra else 0.0
        if fk:
            ferry += P["ferry_share"] * extra * 52 * 2 * block_h(fk)   # подлёт и возврат на каждый оборот
    return (flown, ferry) if split else flown + ferry


# ---------------- оценка сети ----------------
def pair_eval(i, j, net, M, case):
    """Одна пара городов: где летит Red Wings, сколько пассажиров, время в пути."""
    T = M["demand"][(i, j)] * case["demand"]
    T *= unserved_factor((i, j), M)                # пары без рейсов сейчас: спрос не проверен фактом (×0,3 или ML)
    _, opts = ch.options(i, j, M["ap"], net, M["fc"])
    k = M["km"][(i, j)]
    rw = []
    for o in opts:
        if o["перевозчик"] != "WZ" or not o["плечи"]:
            continue
        if o["вариант"] in ("через хаб", "через Екатеринбург"):
            price = P["hub_fare_factor"] * ch.fare(k, 1, M["fc"])
        elif o["вариант"] == "доехать до аэропорта":
            price = o["тариф_рейса"]
        else:
            price = ch.fare(k, 1, M["fc"])
        rw.append((T * o["доля"], o["плечи"], price * case["fare"]))
    return dict(rw=rw,
                hours=T * sum(o["доля"] * o["часы"] for o in opts),
                moscow=T * sum(o["доля"] for o in opts if o["вариант"] == "через Москву"),
                conn=any(o["вариант"] in ("прямой", "через хаб", "через Екатеринбург") for o in opts),
                # «обобщённое время» из логсуммы — для индекса доступности (любой новый вариант его улучшает)
                gt=-opts[0]["logsum"] / abs(P["b_time"]))


def evaluate(net, M, case, detail=False, cache=None, changed=None):
    """Пересчитать всю сеть: пассажиры Red Wings, деньги, время пассажиров, доступность городов.
    Ускорение: если известен результат по парам для прошлой сети (cache) и изменился один рейс (changed),
    пересчитываем только пары, где участвует один из его городов, — остальные не меняются."""
    hub = net.get("hub")
    if cache is None or changed is None:
        pr = {p: pair_eval(p[0], p[1], net, M, case) for p in M["demand"]}
    else:
        pr = dict(cache)
        for p in set(M["pairs_by_city"].get(changed[0], [])) | set(M["pairs_by_city"].get(changed[1], [])):
            pr[p] = pair_eval(p[0], p[1], net, M, case)
    rw_opts, load = [], {}
    hours_total = moscow = 0.0
    connected = 0
    pair_time = {}
    for p, x in pr.items():
        for pax, lg, price in x["rw"]:
            rw_opts.append((pax, lg, price))
            for leg in lg:
                load[leg] = load.get(leg, 0) + pax
        hours_total += x["hours"]
        moscow += x["moscow"]
        connected += x["conn"]
        pair_time[p] = x["gt"]

    # вместимость: если пассажиров больше, чем мест, лишние «не помещаются» (спилл)
    legs = rw_legs(net, M)
    fit = {}
    for p, f in legs.items():
        cap = 2 * f * 52 * P["seats_rw"] * P["lf_max"]
        if HUB in p and hub and hub.get("active", True):
            closed = case["hub_closed"] if case["hub_closed"] is not None else P["hub_closed_fact"].get(HUB, 0.05)
            cap *= 1 - closed                         # закрытия аэропорта хаба съедают часть рейсов
        fit[p] = min(1.0, cap / load[p]) if load.get(p) else 1.0

    rev = trips = segs = 0.0
    carried = {p: 0.0 for p in legs}
    for pax, lg, price in rw_opts:
        if any(l not in fit for l in lg):
            continue                                 # рейс вне модели (город без пассажиропотока в данных)
        c = pax * min(fit[l] for l in lg)
        rev += c * price
        trips += c
        segs += c * len(lg)                          # стыковочный пассажир = 2 перевозки, как в статистике
        for l in lg:
            carried[l] += c

    flown, ferry = block_hours(net, M, split=True)
    bh = flown + ferry
    fixed = P["hub_base_fixed_rub_year"] if (hub and hub.get("active")) else 0.0
    cost = bh * (P["cost_bh_rub"] + P["opportunity_rub_per_bh"]) + fixed
    sub = sum(2 * f * P["subsidy_weeks"] * subsidy_per_flight(M["km"][p]) for p, f in legs.items()) * case["subsidy"]

    # индекс доступности Хансена (логсуммовая форма, Ben-Akiva): для каждого города сумма «масс» других
    # городов, взвешенных по обобщённому времени пути; итог — средний по городам, важно изменение в %
    m = M["ap"]["масса"].to_dict()   # обычный словарь: обращение к pandas в цикле по 780 парам было заметно медленнее
    acc = 0.0
    for (i, j), t in pair_time.items():
        w = math.exp(-P["accessibility_decay_h"] * t)
        acc += (m[j] + m[i]) * w
    res = dict(пассажиров=trips, перевозок=segs, выручка=rev, затраты=cost, субсидии=sub,
               прибыль=rev - cost + sub, блок_часы=bh, перегоны_бч=ferry, база_хаба=fixed,
               пасс_часы=hours_total, через_Москву=moscow, пар_без_Москвы=connected,
               доступность=acc / len(M["cities"]), _pairs=pr)
    if detail:
        res["legs"] = [dict(i=p[0], j=p[1], рейсов_нед=f, км=round(M["km"][p]), пасс_в_год=round(carried[p]),
                            загрузка=round(carried[p] / (2 * f * 52 * P["seats_rw"]), 2),
                            блок_часов_в_год=round(2 * f * 52 * block_h(M["km"][p])),
                            перегон_км=round(ferry_km(p, net, M)))
                       for p, f in legs.items()]
    return res


# ---------------- ходы алгоритма ----------------
def next_level(cur, levels=None):
    for lv in levels or P["freq_levels"]:
        if lv > cur:
            return lv
    return None


def capped(p, M, f):
    """Короткая пара без рейсов сейчас: частота не выше short_unserved_max_freq (история Челябинск–Уфа)."""
    return p not in M["served"] and M["km"][p] < P["short_unserved_km"] and f > P["short_unserved_max_freq"]


def ml_blocked(p, M, spoke=False):
    """ML-фильтр: пару без рейсов сейчас не открываем, если p жизнеспособности ниже порога (params: ml_gate*)."""
    if not P.get("ml_gate") or p in M["served"] or (spoke and not P.get("ml_gate_hub", True)):
        return False
    if not M.get("ml_p"):
        raise RuntimeError("ml_gate включён, но нет results/ml_viability.csv — запустите ml_dataset.py и ml_viability.py")
    return M["ml_p"][p] < P["ml_gate_p"]


def candidate_moves(net, M, mode):
    moves = []
    if mode in ("хаб", "гибрид"):
        for x in M["cities"]:
            if x == HUB:
                continue
            nxt = next_level(net["hub"]["spokes"].get(x, 0), P["spoke_levels"])
            if (nxt and not capped(key(HUB, x), M, nxt) and x not in P["hub_spoke_exclude"].get(HUB, [])
                    and not ml_blocked(key(HUB, x), M, spoke=True)):
                moves.append(("спица", key(HUB, x), nxt))
    if mode in ("прямые", "гибрид"):
        for p in M["direct_candidates"]:
            if mode == "гибрид" and HUB in p:
                continue                              # рейсы из Самары в гибриде — это спицы
            nxt = next_level(net["direct"].get(p, {}).get("WZ", 0))
            if nxt and not capped(p, M, nxt) and not ml_blocked(p, M):
                moves.append(("прямой", p, nxt))
    return moves


def apply(net, move):
    kind, p, f = move
    new = copy.deepcopy(net)
    new["direct"].setdefault(p, {})["WZ"] = f
    if kind == "спица":
        other = p[0] if p[1] == HUB else p[1]
        new["hub"]["spokes"][other] = f
        new["hub"]["active"] = True                  # первая новая спица = открываем базу в Самаре
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
            # открытие базы (первая спица) включает стыковки между ВСЕМИ спицами — тогда пересчитываем всю сеть
            opens_hub = mv[0] == "спица" and not net["hub"].get("active")
            r = evaluate(cand, M, case, cache=None if opens_hub else cur["_pairs"], changed=mv[1])
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
        net["hub"] = {"city": HUB, "spokes": spokes, "active": False}
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
                "гипотеза": {"хаб": f"H1 хаб в {P['hub_names'][P['hub_city']]}", "прямые": "H2 прямые рейсы", "гибрид": "H3 гибрид"}[mode],
                "сценарий": case_name, "новых_самолётов_дано": n,
                "самолётов_занято": round((r["блок_часы"] - s0["блок_часы"]) / BH_PER_AIRCRAFT, 1),
                "новых_пассажиров_в_год": round(d("пассажиров")),
                "новых_перевозок_в_год": round(d("перевозок")),
                "выручка_млрд": round(d("выручка") / 1e9, 2),
                "прибыль_с_субсидиями_млрд": round(d("прибыль") / 1e9, 2),
                "субсидии_млрд": round(d("субсидии") / 1e9, 2),
                "прирост_доли_пп": round(d("перевозок") / (P["market_total_mln"] * 1e6) * 100, 3),
                "сэкономлено_пасс_часов_млн": round(-d("пасс_часы") / 1e6, 2),
                "перегоны_самолётов": round(d("перегоны_бч") / BH_PER_AIRCRAFT, 2),
                "база_хаба_млн": round(r["база_хаба"] / 1e6),
                "доступность_прирост_%": round((r["доступность"] / s0["доступность"] - 1) * 100, 2),
                "через_Москву_изменение_%": round((r["через_Москву"] / s0["через_Москву"] - 1) * 100, 1),
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
