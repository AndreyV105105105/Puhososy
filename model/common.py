"""Общие функции: загрузка данных и расстояния. Без внешних пакетов вроде airportsdata."""
import math
from pathlib import Path
import pandas as pd

from params import PARAMS

ROOT = Path(__file__).resolve().parent.parent   # корень репозитория
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_airports() -> pd.DataFrame:
    """16 аэропортов: iata как индекс, координаты, население, пассажиропоток."""
    df = pd.read_csv(DATA / "airports.csv", encoding="utf-8-sig").set_index("iata")
    # «масса» для модели: пассажиропоток 2025, если нет — 2024; ручные поправки — в PARAMS["mass_override"]
    df["масса"] = df["пасс_2025_млн"].fillna(df["пасс_2024_млн"])
    for k, v in PARAMS.get("mass_override", {}).items():
        if k in df.index:
            df.loc[k, "масса"] = v
    df.loc[df["масса"] <= 0, "масса"] = float("nan")
    return df


def dist_km(lat1, lon1, lat2, lon2) -> float:
    """Расстояние по дуге большого круга (формула гаверсинусов), км."""
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def block_h(km: float) -> float:
    """Блок-время рейса, ч: постоянная часть + полёт на крейсерской скорости."""
    return PARAMS["block_a_h"] + km / PARAMS["speed_kmh"]
