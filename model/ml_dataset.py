"""
ML-блок, шаг 1. Датасет пар городов для модели жизнеспособности прямого рейса.

Одна строка = одна пара из 780 (40 городов модели). Метка «рейс_есть» = на паре сейчас есть регулярный прямой
рейс любой авиакомпании (табло вылета, ноябрь 2026 — то же, что M["served"] в шаге 3).

Признаки описывают только «географию и рынок» пары. Частоты, цены и перевозчики нынешних рейсов в признаки
НЕ берём: иначе модель «узнает» ответ из самих рейсов (утечка).

Запуск: python model/ml_dataset.py   (после step1 и step2)
Выход:  results/ml_pairs.csv
"""
import math
import pandas as pd

from params import PARAMS as P
from common import block_h, RESULTS
import step2_choice as ch
import step3_scenarios as sc

FEATURES = ["лог_км", "лог_масса_мин", "лог_масса_макс", "часы_земля", "выигрыш_самолёта_ч",
            "соседей_250км", "тень_соседа", "база_RW", "вахта"]
CLOSED = [("CEK", "UFA"), ("CEK", "KUF"), ("CEK", "TJM")]   # сняты в 2021–2023 (research/short_routes_history.md)


def ground_hours(i, j, km):
    """Самый быстрый путь по земле: поезд (факт РЖД, иначе формула шага 2) или машина до 800 км."""
    best = math.inf
    if km <= P["rail_max_km"]:
        fact = ch.rail_table().get(sc.key(i, j))
        best = fact[0] if fact else km * P["rail_detour"] / P["rail_speed_kmh"]
    if km <= P["road_max_km"]:
        rd = km * max(P["road_detour"], P["road_detour_city"].get(i, 0), P["road_detour_city"].get(j, 0))
        best = min(best, rd / P["ground_speed_kmh"])
    return min(best, 48.0)   # дальше 2 суток — всё равно «по земле не едут»


def shadow(c, other, ap, neigh):
    """Тень крупного соседа: во сколько раз (лог) самый большой аэропорт до 250 км по дороге больше своего.
    Челябинск в 207 км от Екатеринбурга: многие летят из Кольцово, своему аэропорту рейсы держать труднее."""
    ms = [ap.loc[a, "масса"] for a, _ in neigh.get(c, ()) if a != other]
    return math.log(max(ms) / ap.loc[c, "масса"]) if ms else -3.0


def build(M=None):
    M = M or sc.setup()
    ap = M["ap"]
    neigh = ch._neighbours(ap)
    base = set(ap.index[ap["red_wings_сейчас"].isin(["хаб RW", "база RW"])])
    rows = []
    for (i, j), km in M["km"].items():
        mi, mj = sorted([ap.loc[i, "масса"], ap.loc[j, "масса"]])
        g = ground_hours(i, j, km)
        rows.append(dict(
            i=i, j=j, пара=f"{ap.loc[i, 'город']} – {ap.loc[j, 'город']}", км=round(km),
            лог_км=math.log(km), лог_масса_мин=math.log(mi), лог_масса_макс=math.log(mj),
            часы_земля=round(g, 2),
            # самолёт: блок-время + 2,5 ч на дорогу в аэропорт и досмотр (как в шаге 2 по порядку величины)
            выигрыш_самолёта_ч=round(g - block_h(km) - 2.5, 2),
            соседей_250км=len(neigh.get(i, ())) + len(neigh.get(j, ())),
            тень_соседа=round(max(shadow(i, j, ap, neigh), shadow(j, i, ap, neigh)), 3),
            база_RW=int(i in base or j in base),
            вахта=int(i in P["vakhta_cities"] or j in P["vakhta_cities"]),
            гравитация=M["demand"][(i, j)],
            рейс_есть=int((i, j) in M["served"]),
            снят_2021_2023=int((i, j) in CLOSED or (j, i) in CLOSED),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv(RESULTS / "ml_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"{len(df)} пар, с прямым рейсом {df['рейс_есть'].sum()} ({df['рейс_есть'].mean():.0%})")
    print(df[FEATURES + ["рейс_есть"]].describe().round(2).to_string())
