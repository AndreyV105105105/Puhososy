"""
Идея 1: сколько пассажиро-часов экономит прямая межрегиональная сеть Red Wings
по сравнению с поездкой через Москву (контрфактический сценарий).

Предварительный расчёт (S0 — текущая сеть). Все допущения вынесены в PARAMS.
Запуск: python time_savings.py  (нужен pip install airportsdata)
"""
import csv, math
from pathlib import Path
import airportsdata

DATA = Path(__file__).resolve().parent.parent / "data"

PARAMS = {
    "block_a_h": 0.75,      # допущение: постоянная часть блок-времени (руление, взлёт/посадка), ч
    "speed_kmh": 800,       # допущение: крейсерская скорость по ортодромии, км/ч
    "mct_moscow_h": 2.5,    # допущение: пересадка в Москве (мин. время ~1 ч + ожидание стыковки)
    "seats": 100,           # SSJ-100
    "load": 0.83,           # Red Wings 2025: 2 млн пасс. / 24 тыс. рейсов
    "weeks": 52,
    "vot_rub_h": [500, 1000, 1500],  # ПЛЕЙСХОЛДЕР: стоимость часа пассажира — заменить по ресерчу 3
}
MOSCOW = "SVO"  # точка пересадки в Москве

A = airportsdata.load("IATA")

def dist_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (A[a]["lat"], A[a]["lon"], A[b]["lat"], A[b]["lon"]))
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2 * 6371 * math.asin(math.sqrt(h))

def block_h(a, b):
    return PARAMS["block_a_h"] + dist_km(a, b) / PARAMS["speed_kmh"]

def main():
    rows = list(csv.DictReader(open(DATA / "routes.csv", encoding="utf-8-sig")))
    dom = [r for r in rows if r["перевозчик"] == "Red Wings" and r["сегмент"] == "ВВЛ" and int(r["рейсов_нед"]) > 0]
    out, tot_pax, tot_hours = [], 0, 0
    pairs = set()
    for r in dom:
        o, d = r["откуда"], r["куда"]
        t_dir = block_h(o, d)
        t_mow = block_h(o, MOSCOW) + PARAMS["mct_moscow_h"] + block_h(MOSCOW, d)
        pax = int(r["рейсов_нед"]) * PARAMS["weeks"] * PARAMS["seats"] * PARAMS["load"]
        saved = t_mow - t_dir
        out.append((o, d, round(dist_km(o, d)), round(t_dir, 2), round(t_mow, 2), round(saved, 2), round(pax)))
        tot_pax += pax; tot_hours += pax * saved
        pairs.add(tuple(sorted((o, d))))
    out.sort(key=lambda x: -x[5] * x[6])
    with open(Path(__file__).parent / "time_savings_S0.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["откуда", "куда", "км", "время_прямой_ч", "время_через_Москву_ч", "экономия_ч_на_пасс", "пасс_в_год"])
        w.writerows(out)
    print(f"Маршрутов: {len(dom)}, пар городов без Москвы: {len(pairs)}")
    print(f"Пассажиров в год (оценка): {tot_pax:,.0f}")
    print(f"Средняя экономия на пассажира: {tot_hours/tot_pax:.2f} ч")
    print(f"Экономия всего: {tot_hours/1e6:.2f} млн пассажиро-часов в год")
    for v in PARAMS["vot_rub_h"]:
        print(f"  при VOT {v} ₽/ч: {tot_hours*v/1e9:.2f} млрд ₽ в год")
    print("Топ-5 маршрутов по суммарной экономии:")
    for x in out[:5]:
        print("  ", x)

if __name__ == "__main__":
    main()
