"""
Карточки-объяснения: почему модель рекомендует каждый маршрут — простым текстом для пользователя.

Для каждого маршрута, который модель добавляет при выручке +30/+45/+60% (прямые рейсы, 4 борта), собирает:
спрос пары, что выбирают пассажиры сейчас и после (прямой рейс Red Wings, другие авиакомпании, Москва,
поезд, машина), конкурентов на маршруте, прибыль на час полёта при добавлении, условие шага, субсидию и риски.
Ничего не пересчитывает заново, кроме тех же функций модели (step3_scenarios).

Запуск: python model/explain_routes.py   →  results/route_cards.md, results/route_cards.csv
"""
import pandas as pd

from params import PARAMS as P
from common import DATA, RESULTS
import step2_choice as ch
import step3_scenarios as sc

N = P["new_aircraft"]["базовый"]
STEPS = [(1.3, "Шаг 1", "выручка от +30% к минимальным тарифам (без субсидии от +45%) и фактическая загрузка нынешних рейсов от 85%"),
         (1.45, "Шаг 2", "выручка от +45% (без субсидии от +60%) и загрузка шага 1 от 70% два сезона"),
         (1.6, "Шаг 3", "выручка от +60% и субсидии сохранены")]
CARRIERS = {"WZ": "Red Wings", "UT": "Utair", "A4": "Азимут", "S7": "S7", "U6": "Уральские", "DP": "Победа",
            "FV": "Россия", "SU": "Аэрофлот", "N4": "Nordwind", "RT": "ЮВТ Аэро", "YC": "Ямал", "Y7": "NordStar",
            "5N": "Smartavia", "EO": "Икар", "IO": "ИрАэро", "KV": "КрасАвиа", "7R": "РусЛайн", "4G": "Газпром авиа"}


def groups(opts):
    """Доли выбора пассажиров по понятным группам."""
    g = {}
    for o in opts:
        v, c = o["вариант"], o["перевозчик"]
        if v == "прямой":
            k = "прямой рейс Red Wings" if c == "WZ" else "прямой рейс других авиакомпаний"
        elif v == "через Екатеринбург":
            k = "пересадка в Екатеринбурге"
        elif v == "доехать до аэропорта":
            k = "доехать до соседнего аэропорта"
        else:
            k = {"через Москву": "через Москву", "поезд": "поезд", "машина/автобус": "машина или автобус",
                 "через хаб": "через новый хаб"}.get(v, v)
        g[k] = g.get(k, 0) + o["доля"]
    return g


def pct(x):
    return f"{round(x * 100)}%"


_ML = None


def ml_reason(p, M):
    """Строка карточки из ML-блока: вероятность, что прямой рейс на такой паре держится, и главные факторы."""
    global _ML
    if _ML is None:
        path = RESULTS / "ml_viability.csv"
        _ML = {} if not path.exists() else {sc.key(r.i, r.j): r for r in
                                            pd.read_csv(path, encoding="utf-8-sig").itertuples()}
    r = _ML.get(p)
    if r is None:
        return "ML-оценка: нет (запустите model/ml_dataset.py и model/ml_viability.py)"
    f = f" (главное: {r.факторы_ML})" if isinstance(r.факторы_ML, str) and r.факторы_ML else ""
    return (f"ML-оценка жизнеспособности: на парах с такой географией и рынком прямой рейс держится "
            f"с вероятностью {r.p:.2f}{f}")


def main():
    M = sc.setup()
    names = M["ap"]["город"].to_dict()
    sub = pd.read_csv(DATA / "subsidized_rw_2026.csv", encoding="utf-8-sig").dropna(subset=["iata_1", "iata_2"])
    listed = {tuple(sorted(x)) for x in zip(sub["iata_1"], sub["iata_2"])}
    s0net = sc.base_network(M, "прямые")
    seen, cards = set(), []
    for fare, step, cond in STEPS:
        case = {**P["cases"]["базовый"], "fare": fare}
        net, log = sc.greedy(sc.base_network(M, "прямые"), M, "прямые", N, case)
        r = sc.evaluate(net, M, case, detail=True)
        conn = {}                                  # пассажиры рейса, которые пересаживаются (два плеча)
        for _, xx in r["_pairs"].items():
            for pax, lg, _ in xx["rw"]:
                for l in lg:
                    l = tuple(l)
                    c = conn.setdefault(l, [0.0, 0.0]); c[0] += pax; c[1] += pax if len(lg) == 2 else 0.0
        legs = {(x["i"], x["j"]): x for x in r["legs"]}
        last = {}
        for x in log:
            if x.get("принят"):
                a, b = x["пара"].split("-")
                last[sc.key(a, b)] = x
        for p, x in last.items():
            if p in seen:
                continue
            seen.add(p)
            i, j = p
            T = M["demand"][p] * case["demand"] * sc.unserved_factor(p, M)
            _, o0 = ch.options(i, j, M["ap"], s0net, M["fc"])
            _, o1 = ch.options(i, j, M["ap"], net, M["fc"])
            g0, g1 = groups(o0), groups(o1)
            f0 = s0net["direct"].get(p, {}).get("WZ", 0)
            f1 = net["direct"].get(p, {}).get("WZ", 0)
            rivals = {CARRIERS.get(c, c): f for c, f in s0net["direct"].get(p, {}).items() if c != "WZ"}
            leg = legs.get(p, {})
            risks = []
            if rivals:
                big = max(rivals.values())
                risks.append(f"на маршруте летают {', '.join(f'{k} ({v:g} в неделю)' for k, v in rivals.items())}"
                             + (": конкурент может ответить ростом частот" if big >= 3 else ""))
            else:
                risks.append("конкурентов нет; риск — вход нового перевозчика")
            if p not in M["served"]:
                risks.append("сейчас прямых рейсов нет: спрос — оценка модели с понижающей поправкой")
            if M["km"][p] < P["short_fare_km"]:
                risks.append("короткий маршрут: выручка чувствительна к цене билета")
            cs = conn.get(p, [0, 0]); tr = cs[1] / cs[0] if cs[0] else 0.0
            reasons = [f"спрос между городами — около {round(T / 1000)} тыс. поездок в год (все виды транспорта)",
                       "сейчас: " + ", ".join(f"{k} {pct(v)}" for k, v in sorted(g0.items(), key=lambda z: -z[1]) if v >= 0.03),
                       "после: " + ", ".join(f"{k} {pct(v)}" for k, v in sorted(g1.items(), key=lambda z: -z[1]) if v >= 0.03),
                       f"на этом маршруте самолёт приносит {x['прибыль_на_блок_час'] / 1000:.0f} тыс. ₽ прибыли на час полёта — больше, чем другие варианты в этот момент",
                       ml_reason(p, M),
                       f"{pct(tr)} пассажиров рейса летят с пересадкой в Екатеринбурге дальше по сети Red Wings" if tr >= 0.05 else "почти все пассажиры летят именно между этими двумя городами"]
            cards.append({"шаг": step, "условие": cond, "маршрут": f"{names[i]} – {names[j]}", "i": i, "j": j,
                          "км": round(M["km"][p]), "рейсов_нед_сейчас": f0, "рейсов_нед_после": f1,
                          "пассажиров_RW_в_год": leg.get("пасс_в_год"), "загрузка": leg.get("загрузка"),
                          "спрос_пары_в_год": round(T), "доля_RW_прямой_было": round(g0.get("прямой рейс Red Wings", 0), 3),
                          "доля_RW_прямой_стало": round(g1.get("прямой рейс Red Wings", 0), 3),
                          "доля_через_Москву_было": round(g0.get("через Москву", 0), 3),
                          "доля_через_Москву_стало": round(g1.get("через Москву", 0), 3),
                          "прибыль_на_блок_час_руб": round(x["прибыль_на_блок_час"]), "доля_пересадочных": round(tr, 3),
                          "в_перечне_субсидий_2026": "да" if p in listed else "нет",
                          "конкуренты": "; ".join(f"{k} {v:g}" for k, v in rivals.items()) or "нет",
                          "почему": " | ".join(reasons), "риски": " | ".join(risks)})
    df = pd.DataFrame(cards)
    df.to_csv(RESULTS / "route_cards.csv", index=False, encoding="utf-8-sig")
    out = ["# Почему модель рекомендует эти маршруты", "",
           "_Файл собирается автоматически: `python model/explain_routes.py`. Прямые рейсы, 4 новых борта, базовый сценарий._", ""]
    for _, c in df.iterrows():
        out += [f"## {c['шаг']}. {c['маршрут']}: {c['рейсов_нед_сейчас']:g} → {c['рейсов_нед_после']:g} рейсов в неделю",
                f"**Условие:** {c['условие']}. **Субсидия 2026:** {'есть' if c['в_перечне_субсидий_2026'] == 'да' else 'нет — без неё порог выше'}.", "",
                "**Почему:**"] + [f"- {s}" for s in c["почему"].split(" | ")] + [
                f"- Red Wings перевезёт на маршруте около {round((c['пассажиров_RW_в_год'] or 0) / 1000)} тыс. пассажиров в год, загрузка {pct(c['загрузка'] or 0)}", "",
                "**Риски:**"] + [f"- {s}" for s in c["риски"].split(" | ")] + [""]
    (RESULTS / "route_cards.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
