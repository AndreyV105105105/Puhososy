"""Общие функции: загрузка данных и расстояния. Без внешних пакетов вроде airportsdata."""
import math
from pathlib import Path
import pandas as pd

from params import PARAMS

ROOT = Path(__file__).resolve().parent.parent   # корень репозитория
DATA = ROOT / "data"
import os
# Версия расчёта: по умолчанию все города; PUH_CORE=1 — только 16 крупных аэропортов (версия «14 городов»),
# результаты тогда пишутся в results_14/ (проверка, что вывод держится на обоих списках городов)
CORE_ONLY = os.environ.get("PUH_CORE") == "1"
RESULTS = ROOT / ("results_14" if CORE_ONLY else "results")
RESULTS.mkdir(exist_ok=True)


def load_airports() -> pd.DataFrame:
    """40 аэропортов (16 — в версии CORE_ONLY): iata как индекс, координаты, население, пассажиропоток."""
    df = pd.read_csv(DATA / "airports.csv", encoding="utf-8-sig").set_index("iata")
    # «масса» для модели: пассажиропоток 2025, если нет — 2024; ручные поправки — в PARAMS["mass_override"]
    df["масса"] = df["пасс_2025_млн"].fillna(df["пасс_2024_млн"])
    for k, v in PARAMS.get("mass_override", {}).items():
        if k in df.index:
            df.loc[k, "масса"] = v
    df.loc[df["масса"] <= 0, "масса"] = float("nan")
    if CORE_ONLY:
        core = set(pd.read_csv(DATA / "boards_week.csv", encoding="utf-8-sig")["откуда"])
        df.loc[~df.index.isin(core), "масса"] = float("nan")   # остальные города выпадают из расчёта
    return df


def dist_km(lat1, lon1, lat2, lon2) -> float:
    """Расстояние по дуге большого круга (формула гаверсинусов), км."""
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def block_h(km: float) -> float:
    """Блок-время рейса, ч: постоянная часть + полёт на крейсерской скорости."""
    return PARAMS["block_a_h"] + km / PARAMS["speed_kmh"]


def load_boards() -> pd.DataFrame:
    """Табло вылета: 16 исходных аэропортов (неделя 10–16.11.2026) + 23 новых города
    (data/boards_new_cities.csv, Семёныч: рейсы во вторник 10.11 × 7 — оценка недели)."""
    parts = [pd.read_csv(DATA / "boards_week.csv", encoding="utf-8-sig")]
    extra = DATA / "boards_new_cities.csv"
    if extra.exists():
        parts.append(pd.read_csv(extra, encoding="utf-8-sig"))
    return pd.concat(parts, ignore_index=True)
