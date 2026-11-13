"""
Карта сети маршрутов — рисуется автоматически после каждого пересчёта.

Серые линии — рейсы Red Wings сейчас (внутри модели), красные — новые рейсы или больше частот
в выбранном решении (толщина = рейсов в неделю). Размер точки — пассажиропоток аэропорта.
Подложка — контуры стран (data/geo_countries_50m.json, как у карты для слайда), проекция Ламберта.
Подписи городов и частот расставляются так, чтобы не накладываться друг на друга.

Запуск: python model/step8_map.py   (после step3)
Выход:  results/map_<гипотеза>_базовый.png
"""
import json
import math
import matplotlib
matplotlib.use("Agg")                      # рисуем в файл, без окна
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
import pandas as pd

from params import PARAMS as P
from common import load_airports, RESULTS, DATA
import step2_choice as ch

TITLES = {"прямые": "H2 прямые рейсы", "хаб": f"H1 хаб в {P['hub_names'][P['hub_city']]}", "гибрид": "H3 гибрид"}
SEA, RUS, OTHER, BORDER, RBORDER = "#EAF0F5", "#FFFFFF", "#F1F3F5", "#C9D2DB", "#AFBBC7"
INK, MUTED, GREY, RED, BLUE, ORANGE = "#16202B", "#5B6775", "#9AA6B4", "#C8102E", "#1F5FA8", "#F28C28"
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
SHORT = {"Набережные Челны": "Наб. Челны"}   # длинные названия в плотном центре карты

# --- проекция Ламберта (коническая равноугольная), как в step12_map_slide.py ---
_lon0, _lat0, _p1, _p2 = map(math.radians, (62, 55, 47, 63))
_n = math.log(math.cos(_p1) / math.cos(_p2)) / math.log(math.tan(math.pi / 4 + _p2 / 2) / math.tan(math.pi / 4 + _p1 / 2))
_F = math.cos(_p1) * math.tan(math.pi / 4 + _p1 / 2) ** _n / _n
_rho0 = _F / math.tan(math.pi / 4 + _lat0 / 2) ** _n


def xy(lat, lon):
    rho = _F / math.tan(math.pi / 4 + math.radians(lat) / 2) ** _n
    th = _n * (math.radians(lon) - _lon0)
    return rho * math.sin(th), _rho0 - rho * math.cos(th)


def countries(ax):
    path = DATA / "geo_countries_50m.json"
    if not path.exists():
        return
    for name, polys in json.loads(path.read_text(encoding="utf-8")).items():
        ru = name == "Russia"
        for ring in polys:
            ax.add_patch(Polygon([xy(y, x) for x, y in ring], closed=True, fc=RUS if ru else OTHER,
                                 ec=RBORDER if ru else BORDER, lw=0.6 if ru else 0.4, zorder=0, joinstyle="round"))


class Placer:
    """Жадная расстановка подписей: пробуем несколько позиций, берём первую без пересечений."""

    def __init__(self, fig, ax):
        self.fig, self.ax, self.boxes = fig, ax, []
        self.r = fig.canvas.get_renderer()

    def block(self, x, y, size_px):
        p = self.ax.transData.transform((x, y))
        self.boxes.append((p[0] - size_px, p[1] - size_px, p[0] + size_px, p[1] + size_px))

    def _free(self, b):
        return self._overlap(b) == 0

    def _overlap(self, b):
        return sum(max(0, min(b[2], o[2]) - max(b[0], o[0])) * max(0, min(b[3], o[3]) - max(b[1], o[1]))
                   for o in self.boxes)

    def put(self, x, y, text, offsets, must=True, near=8, **kw):
        """near — сколько первых позиций «рядом с точкой»; для дальних рисуем тонкую выноску."""
        best = None
        for k, (dx, dy, ha, va) in enumerate(offsets):
            arrow = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=2) if k >= near else None
            t = self.ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                 zorder=6, path_effects=HALO, arrowprops=arrow, **kw)
            bb = t.get_window_extent(self.r)
            b = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            ov = self._overlap(b)
            if ov == 0:
                self.boxes.append(b)
                return t
            if best is None or ov < best[0]:
                best = (ov, (dx, dy), ha, va, arrow)
            t.remove()
        if must and best:
            t = self.ax.annotate(text, (x, y), xytext=best[1], textcoords="offset points", ha=best[2], va=best[3],
                                 zorder=6, path_effects=HALO, arrowprops=best[4], **kw)
            bb = t.get_window_extent(self.r)
            self.boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
            return t
        return None


def city_offsets(r):
    d = r + 2
    near = [(d, 0, "left", "center"), (-d, 0, "right", "center"), (0, d, "center", "bottom"),
            (0, -d, "center", "top"), (d, d, "left", "bottom"), (-d, d, "right", "bottom"),
            (d, -d, "left", "top"), (-d, -d, "right", "top")]
    far = [(sx * D, sy * D, "left" if sx > 0 else ("right" if sx < 0 else "center"),
            "bottom" if sy > 0 else ("top" if sy < 0 else "center"))
           for D in (d + 14, d + 26, d + 40, d + 56) for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))]
    return near + far


def ru(x):
    return f"{x:g}".replace(".", ",")


def draw(mode, airports, s0, title=None, hub_note="кандидат в хаб (надстройка к прямым с выручки +80%)",
         path=None, out=None):
    path = path or RESULTS / f"network_{mode}_базовый.csv"
    if not path.exists():
        return None
    legs = pd.read_csv(path, encoding="utf-8-sig")
    pt = lambda c: xy(*P["moscow_point"]) if c == "MOW" else xy(airports.at[c, "lat"], airports.at[c, "lon"])
    cities = [c for c in airports.index if pd.notna(airports.at[c, "lat"])]

    fig, ax = plt.subplots(figsize=(16, 9.6), dpi=150)
    fig.patch.set_facecolor(SEA); ax.set_facecolor(SEA)
    xs, ys = zip(*[pt(c) for c in cities + ["MOW"]])
    padx, pady = (max(xs) - min(xs)) * 0.04, (max(ys) - min(ys)) * 0.08
    ax.set_xlim(min(xs) - padx, max(xs) + padx); ax.set_ylim(min(ys) - pady, max(ys) + pady)
    ax.set_aspect("equal"); ax.axis("off")
    countries(ax)

    new = []
    for _, r in legs.iterrows():
        old = s0.get((r.i, r.j), {}).get("WZ", 0)
        (x1, y1), (x2, y2) = pt(r.i), pt(r.j)
        if r["рейсов_нед"] > old:
            ax.plot([x1, x2], [y1, y2], color="white", lw=4 + r["рейсов_нед"] / 3, alpha=0.9, zorder=2,
                    solid_capstyle="round")
            ax.plot([x1, x2], [y1, y2], color=RED, lw=1.5 + r["рейсов_нед"] / 3, alpha=0.9, zorder=2,
                    solid_capstyle="round")
            # подпись частоты ставим ближе к дальнему от базы концу — так понятно, к какому рейсу она относится
            if r.i in P["rw_bases"]:
                x1, y1, x2, y2 = x2, y2, x1, y1
            new.append((x1, y1, x2, y2, old, r["рейсов_нед"]))
        else:
            ax.plot([x1, x2], [y1, y2], color=GREY, lw=0.8 + r["рейсов_нед"] / 4, alpha=0.7, zorder=1)

    fig.canvas.draw()
    pl = Placer(fig, ax)
    size = {c: 18 + (0 if pd.isna(airports.at[c, "масса"]) else airports.at[c, "масса"] * 22) for c in cities}
    for c in cities:
        x, y = pt(c)
        color = BLUE if c in P["rw_bases"] else (ORANGE if c == P["hub_city"] else "#2B3642")
        ax.scatter(x, y, s=size[c], color=color, zorder=4, edgecolor="white", linewidth=1)
        pl.block(x, y, math.sqrt(size[c]) / 2 * fig.dpi / 72)
    mx, my = pt("MOW")
    ax.scatter(mx, my, marker="*", s=200, color=MUTED, zorder=4, edgecolor="white", linewidth=0.8)
    pl.block(mx, my, 8)

    # сначала подписи базы и кандидата в хаб, потом частоты на новых рейсах, потом остальные города
    for c in [c for c in cities if c in P["rw_bases"] or c == P["hub_city"]]:
        x, y = pt(c)
        pl.put(x, y, airports.at[c, "город"], city_offsets(math.sqrt(size[c]) / 2), fontsize=10, color=INK,
               weight="bold")
    pl.put(mx, my, "Москва", city_offsets(7), fontsize=9, color=MUTED)
    for x1, y1, x2, y2, old, f in new:
        lab = f"{ru(old)} → {ru(f)}/нед" if old else f"новый, {ru(f)}/нед"
        for t in (0.3, 0.4, 0.2, 0.5, 0.6):
            x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
            if pl.put(x, y, lab, [(0, 7, "center", "bottom"), (0, -7, "center", "top")], must=False, fontsize=8.5,
                      color=RED, weight="bold"):
                break
        else:
            x, y = x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.3
            pl.put(x, y, lab, [(0, 7, "center", "bottom"), (0, -7, "center", "top")] + city_offsets(4)[8:],
                   near=2, fontsize=8.5, color=RED, weight="bold")

    # подписи городов: сначала крупные и те, что в сети Red Wings
    in_net = set(legs["i"]) | set(legs["j"])
    order = sorted([c for c in cities if c not in P["rw_bases"] and c != P["hub_city"]],
                   key=lambda c: (c not in in_net, -size[c]))
    for c in order:
        x, y = pt(c)
        name = SHORT.get(airports.at[c, "город"], airports.at[c, "город"])
        pl.put(x, y, name, city_offsets(math.sqrt(size[c]) / 2), must=True,
               fontsize=9 if c in in_net else 8, color=INK if c in in_net else MUTED)

    ax.plot([], [], color=GREY, lw=2, label="рейсы Red Wings сейчас")
    ax.plot([], [], color=RED, lw=3, label="новые рейсы / больше частот (подпись: рейсов в неделю)")
    ax.scatter([], [], color=BLUE, label="база Red Wings (Кольцово)")
    ax.scatter([], [], color=ORANGE, label=f"{airports.at[P['hub_city'], 'город']} — {hub_note}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)
    ax.set_title(title or f"{TITLES[mode]}, базовый сценарий: сеть Red Wings между городами модели",
                 fontsize=12.5, color=INK, loc="left")
    out = out or RESULTS / f"map_{mode}_базовый.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SEA); plt.close(fig)
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
