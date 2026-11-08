"""
Сколько SSJ-100 фактически занято хабом Red Wings в Екатеринбурге (SVX).
Оценка по сверенному расписанию (routes.csv): блок-часы рейсов из/в SVX за неделю / налёт одного борта.
"""
import csv, math
from pathlib import Path
import airportsdata
A = airportsdata.load("IATA")
DATA = Path(__file__).resolve().parent.parent / "data"
BLOCK_A, SPEED = 0.75, 800          # допущения, как в time_savings.py
UTIL_H_DAY = [6.3, 8, 10]           # налёт SSJ-100, ч/сут: 6,3 — факт Red Wings 2024; 8 и 10 — для сравнения

def d(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (A[a]["lat"], A[a]["lon"], A[b]["lat"], A[b]["lon"]))
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*6371*math.asin(math.sqrt(h))

rows = [r for r in csv.DictReader(open(DATA/"routes.csv", encoding="utf-8-sig"))
        if r["перевозчик"] == "Red Wings" and "SVX" in (r["откуда"], r["куда"])]
freq = {}
for r in rows:
    other = r["куда"] if r["откуда"] == "SVX" else r["откуда"]
    key = (other, r["сегмент"])
    freq[key] = max(freq.get(key, 0), int(r["рейсов_нед"]))   # пары рейсов туда-обратно в неделю

tot = {"ВВЛ": 0.0, "МВЛ": 0.0}
for (o, seg), n in sorted(freq.items(), key=lambda x: -x[1]):
    bh = 2 * n * (BLOCK_A + d("SVX", o)/SPEED)
    tot[seg] += bh
all_h = sum(tot.values())
print(f"Направлений из SVX: {len(freq)} (ВВЛ {sum(1 for k in freq if k[1]=='ВВЛ')}, МВЛ {sum(1 for k in freq if k[1]=='МВЛ')})")
print(f"Рейсов в неделю (в обе стороны): {2*sum(freq.values())}")
print(f"Блок-часов в неделю: ВВЛ {tot['ВВЛ']:.0f}, МВЛ {tot['МВЛ']:.0f}, всего {all_h:.0f}")
for u in UTIL_H_DAY:
    print(f"  при налёте {u} ч/сут: ~{all_h/(7*u):.1f} SSJ-100 (только ВВЛ: {tot['ВВЛ']/(7*u):.1f})")
