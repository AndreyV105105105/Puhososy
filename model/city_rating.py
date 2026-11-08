"""
Рейтинг 16 городов-кандидатов (шаг 1 кейса): хаб / опорная база.
Критерии нормируются 0..1, веса — допущение; проверка устойчивости: каждый вес ±30%.
Надёжность (закрытия аэропортов) — временный фактор, считается отдельно (сценарий), в базовый рейтинг не входит.
"""
import csv, math, itertools
from pathlib import Path
import airportsdata
A = airportsdata.load("IATA")
DATA = Path(__file__).resolve().parent.parent / "data"

def dist(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (A[a]["lat"], A[a]["lon"], A[b]["lat"], A[b]["lon"]))
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*6371*math.asin(math.sqrt(h))

rows = list(csv.DictReader(open(DATA/"airports.csv", encoding="utf-8-sig")))
# конкуренция: 1 = свободен, 0.5 = есть сильный игрок, 0 = хаб/база доминирующего перевозчика (ресерч 2 + табло)
COMP = {"SVX":0.5,"KUF":1,"UFA":0.5,"KZN":0,"PEE":1,"OMS":0.5,"CEK":1,"TJM":0.5,"GOJ":1,"VOG":1,"OVB":0,"KJA":0,"IKT":0,"MRV":0,"AER":0,"KGD":0.5}
# присутствие RW: 1 = база/хаб, 0.5 = регулярные рейсы, 0 = почти нет
RW = {"SVX":1,"KUF":0.5,"UFA":0.5,"KZN":0.5,"PEE":1,"OMS":1,"CEK":0.5,"TJM":0.5,"GOJ":0.5,"VOG":0.5,"OVB":0,"KJA":0,"IKT":0,"MRV":0.5,"AER":0.5,"KGD":0}
# закрытия, % времени (samokatus, 90 дней) — для сценария
CLOSE = {"KUF":10.0,"KZN":9.5,"UFA":5.7,"PEE":4.3,"GOJ":3.7,"SVX":2.0,"OMS":0.9,"CEK":0.9,"OVB":0.0,"VOG":4.0,"AER":17.2,"MRV":0.0,"KGD":0.0}  # KJA, TJM, IKT — нет данных (страницы не загрузились), в сценарии = 2%

# --- критерии из ТЗ, добавлены 06.11.2026 ---
# возможности аэропорта: длина ВПП, м (en.wikipedia, инфобоксы, 06.11.2026). SSJ-100 нужно < 2 км — подходят все, поэтому вес малый
RWY = {"SVX":[3026,3004],"KUF":[2548,3001],"VOG":[2300,3280],"OMS":[2876,2500],"OVB":[3600,3600],"PEE":[3206],"KZN":[3724],"CEK":[3200],
       "UFA":[3761,2513],"GOJ":[2805,3000],"KJA":[3700],"AER":[2600,3000],"MRV":[3900],"KGD":[3350],"TJM":[3003,2704],"IKT":[3564]}
def airport(c):  # 0,5 за длину главной ВПП (2,5→3,5 км) + 0,5 за вторую полосу ≥ 2,5 км (резерв при ремонте/закрытии полосы)
    r = sorted(RWY[c], reverse=True); return 0.5*min(1, max(0, (r[0]-2500)/1000)) + 0.5*(len(r) > 1 and r[1] >= 2500)
# туристический спрос: турпоездки, млн, янв–окт 2025 (Росстат, топ-10 регионов; expert.ru 02.12.2025). Вне топ-10 = 0 (меньше 1,5 млн)
TOUR = {"AER":8.3,"KZN":2.4,"SVX":1.7,"MRV":1.5,"TJM":1.5}
# пересадки через Москву: гравитационная оценка спроса на пары городов без прямого рейса (табло 10–16.11.2026, data/boards_week.csv)
direct = set()
for r in csv.DictReader(open(DATA/"boards_week.csv", encoding="utf-8")):
    direct.add(frozenset((r["откуда"], r["куда"])))

def norm(v, lo, hi): return (v-lo)/(hi-lo) if hi > lo else 0
cand = [r for r in rows if r["iata"] != "SVX"]
pax = {r["iata"]: float(r["пасс_2025_млн"] or r["пасс_2024_млн"]) for r in rows}
pop = {r["iata"]: float(r["население_города_млн"] or 0.3) for r in rows}   # MRV: допущение 0,3 (нет данных)
d_svx = {r["iata"]: dist("SVX", r["iata"]) for r in rows}
d_mow = {r["iata"]: dist("SVO", r["iata"]) for r in rows}
# «тень Москвы»: ближе 600 км — пассажир уезжает на поезде/авто в московский хаб (Н. Новгород: «Ласточка» 3,5–4 ч)
def shadow(d): return 0 if d < 600 else min(1, (d-600)/300)
# удалённость от SVX: 0 при <400 км (каннибализация), 1 при 700–1500 км, дальше спад (хаб далеко от сети RW)
def sep(d): return 0 if d < 400 else min(1, (d-400)/300) if d < 1500 else max(0, 1-(d-1500)/1500)

crit = {
  "пассажиропоток": {c: norm(pax[c], 1, 6) for c in pax},       # 12,5 млн Сочи ограничен 1
  "население":      {c: norm(pop[c], 0.4, 1.6) for c in pop},
  "удалённость_от_SVX": {c: sep(d_svx[c]) for c in d_svx},
  "свободный_рынок": COMP,
  "вне_тени_Москвы": {c: shadow(d_mow[c]) for c in d_mow},
  "присутствие_RW": RW,
}
cities = list(pax)
def via_mow(h):
    # сколько пассажиров, летающих сейчас с пересадкой в Москве, хаб в городе h может перехватить:
    # Σ P_i·P_j / d_ij по парам (i, j) без прямого рейса (табло), дальше 600 км (ближе — поезд),
    # если крюк через h не больше 30%: (d_ih + d_hj) / d_ij ≤ 1,3. Гравитация без калибровки — относительная оценка.
    v = 0
    for i, j in itertools.combinations([c for c in cities if c != h], 2):
        d = dist(i, j)
        if frozenset((i, j)) in direct or d < 600: continue
        if (dist(i, h) + dist(h, j)) / d <= 1.3: v += pop[i]*pop[j]/d
    return v
vm = {c: via_mow(c) for c in cities}
crit["пересадки_через_Москву"] = {c: vm[c]/max(vm.values()) for c in cities}
crit["туризм"] = {c: TOUR.get(c, 0)/max(TOUR.values()) for c in cities}
crit["возможности_аэропорта"] = {c: airport(c) for c in cities}
W = {"пассажиропоток":0.10,"население":0.10,"удалённость_от_SVX":0.20,"свободный_рынок":0.15,"вне_тени_Москвы":0.15,"присутствие_RW":0.05,
     "пересадки_через_Москву":0.15,"туризм":0.05,"возможности_аэропорта":0.05}

def score(w):
    s = {r["iata"]: sum(w[k]*min(1, crit[k][r["iata"]]) for k in w) / sum(w.values()) for r in cand}
    return sorted(s.items(), key=lambda x: -x[1])

base = score(W)
print("Базовый рейтинг (без закрытий):")
for i, (c, s) in enumerate(base, 1):
    r = next(x for x in rows if x["iata"] == c)
    print(f"{i:2}. {c} {r['город']:<17} {s:.2f}  км от SVX {d_svx[c]:5.0f}  от Москвы {d_mow[c]:5.0f}  закрыт {CLOSE.get(c,'н/д')}%")
top3 = [c for c, _ in base[:3]]
# устойчивость: каждый вес ×0.7 и ×1.3
stable, tests = 0, 0
for k in W:
    for f in (0.7, 1.3):
        w = dict(W); w[k] *= f
        t = [c for c, _ in score(w)[:3]]; tests += 1
        stable += set(t) == set(top3)
        if set(t) != set(top3): print("  топ-3 меняется при", k, "x", f, "->", t)
print(f"Топ-3 {top3}: устойчив в {stable} из {tests} проверок; №1 {base[0][0]} остаётся первым во всех: ",
      all(score({**W, k: W[k]*f})[0][0] == base[0][0] for k in W for f in (0.7,1.3)))
# сценарий: штраф за закрытия (вес 0.2 к доле времени открыт, норм. 0..10%)
w2 = {**W}; crit["надёжность"] = {c: 1 - CLOSE.get(c, 2.0)/10 for c in pax}; w2["надёжность"] = 0.2
print("С учётом закрытий (сценарий):", [(c, round(s,2)) for c, s in score(w2)[:5]])
with open(Path(__file__).parent/"city_rating.csv","w",newline="",encoding="utf-8-sig") as f:
    wr = csv.writer(f); wr.writerow(["место","iata","город","балл","км_от_SVX"]+list(crit))
    for i,(c,s) in enumerate(base,1):
        r = next(x for x in rows if x["iata"]==c)
        wr.writerow([i,c,r["город"],round(s,3),round(d_svx[c])]+[round(min(1,crit[k][c]),2) for k in crit])
