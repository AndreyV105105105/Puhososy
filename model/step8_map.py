"""
Карта сети маршрутов — рисуется автоматически после каждого пересчёта.

Серые линии — рейсы Red Wings сейчас (внутри модели), красные — новые рейсы или больше частот
в выбранном решении (толщина = рейсов в неделю). Размер точки — пассажиропоток аэропорта.

Запуск: python model/step8_map.py   (после step3)
Выход:  results/map_<гипотеза>_базовый.png
"""
import math
import matplotlib
matplotlib.use("Agg")                      # рисуем в файл, без окна
import matplotlib.pyplot as plt
import pandas as pd

from params import PARAMS as P
from common import load_airports, RESULTS
import step2_choice as ch

TITLES = {"прямые": "H2 прямые рейсы", "хаб": "H1 хаб в Самаре", "гибрид": "H3 гибрид"}


def xy(lat, lon):
    """Простая проекция: долготу сжимаем на cos(55°), чтобы карта не была растянута."""
    return lon * math.cos(math.radians(55)), lat


def draw(mode, airports, s0):
    path = RESULTS / f"network_{mode}_базовый.csv"
    if not path.exists():
        return None
    legs = pd.read_csv(path, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    pt = lambda c: xy(*P["moscow_point"]) if c == "MOW" else xy(airports.at[c, "lat"], airports.at[c, "lon"])

    for _, r in legs.iterrows():
        old = s0.get((r.i, r.j), {}).get("WZ", 0)
        (x1, y1), (x2, y2) = pt(r.i), pt(r.j)
        if r["рейсов_нед"] > old:
            ax.plot([x1, x2], [y1, y2], color="#d62728", lw=1 + r["рейсов_нед"] / 3, alpha=0.85, zorder=2)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{r['рейсов_нед']:g}/нед", fontsize=7, color="#d62728",
                    ha="center", va="bottom", zorder=4)
        else:
            ax.plot([x1, x2], [y1, y2], color="#999999", lw=1 + r["рейсов_нед"] / 4, alpha=0.6, zorder=1)

    for c in airports.index:
        pax = airports.at[c, "пасс_2025_млн"]
        x, y = pt(c)
        base = c in P["rw_bases"]
        hub = c == P["hub_city"]
        color = "#1f77b4" if base else ("#ff7f0e" if hub else "#333333")
        ax.scatter(x, y, s=20 + (0 if pd.isna(pax) else pax * 25), color=color, zorder=3, edgecolor="white")
        ax.text(x, y - 0.6, airports.at[c, "город"], fontsize=8, ha="center", va="top", zorder=4)
    mx, my = pt("MOW")
    ax.scatter(mx, my, marker="*", s=160, color="#7f7f7f", zorder=3)
    ax.text(mx, my - 0.6, "Москва", fontsize=8, ha="center", va="top", color="#7f7f7f")

    ax.plot([], [], color="#999999", lw=2, label="рейсы Red Wings сейчас")
    ax.plot([], [], color="#d62728", lw=3, label="новые рейсы / больше частот")
    ax.scatter([], [], color="#1f77b4", label="база Red Wings (Кольцово)")
    ax.scatter([], [], color="#ff7f0e", label="Самара — кандидат в хаб")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"{TITLES[mode]}, базовый сценарий: сеть Red Wings между городами модели", fontsize=11)
    ax.set_aspect("equal"); ax.axis("off")
    out = RESULTS / f"map_{mode}_базовый.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    airports = load_airports()
    s0 = ch.network_from_boards(airports)["direct"]
    for mode in TITLES:
        out = draw(mode, airports, s0)
        if out:
            print(f"Карта: {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
