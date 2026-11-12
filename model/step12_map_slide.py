"""
Карта целевой сети 2028 для слайда (16:9): шаги дорожной карты, база в Екатеринбурге, действующие рейсы Red Wings.
Берёт results/network_цель2028_базовый.csv (после step7_target_network.py) и контуры стран data/geo_countries_50m.json.

Запуск: python model/step12_map_slide.py [light|dark]  →  results/map_slide_light.png и .svg
"""
import json, math, csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, FancyBboxPatch, Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patheffects as pe

from pathlib import Path as FPath
ROOT = FPath(__file__).resolve().parent.parent
U = str(ROOT)
G = ROOT / "data" / "geo_countries_50m.json"   # Natural Earth 1:50m (world-atlas 2.0.2), страны вокруг России, координаты до 0,01°
THEME = sys.argv[1] if len(sys.argv) > 1 else "light"
OUT = str(ROOT / "results" / f"map_slide_{THEME}.png")

for f in font_manager.findSystemFonts():
    if "Inter" in f: font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Inter"

if THEME == "light":
    C = dict(bg="#EAF0F5", land="#FFFFFF", russia="#FFFFFF", other="#F1F3F5", border="#C9D2DB", rborder="#AFBBC7",
             ink="#16202B", muted="#5B6775", exist="#9AA6B4", node="#2B3642", panel="#FFFFFF", panel_ec="#DCE3EA", halo="#FFFFFF")
else:
    C = dict(bg="#0E1621", land="#1A2533", russia="#1C2836", other="#151E29", border="#2B394A", rborder="#3A4B5F",
             ink="#EEF2F6", muted="#9AA8B8", exist="#5D6E82", node="#DCE4EC", panel="#141D28", panel_ec="#2B394A", halo="#0E1621")
RED = ["#B5122B", "#E3433B", "#F28C63"]   # шаги 1, 2, 3

# --- проекция Ламберта (коническая равноугольная) ---
lon0, lat0, p1, p2 = math.radians(62), math.radians(55), math.radians(47), math.radians(63)
n = math.log(math.cos(p1) / math.cos(p2)) / math.log(math.tan(math.pi/4 + p2/2) / math.tan(math.pi/4 + p1/2))
F = math.cos(p1) * math.tan(math.pi/4 + p1/2) ** n / n
rho0 = F / math.tan(math.pi/4 + lat0/2) ** n
def proj(lon, lat):
    la, lo = math.radians(lat), math.radians(lon)
    rho = F / math.tan(math.pi/4 + la/2) ** n
    th = n * (lo - lon0)
    return rho * math.sin(th), rho0 - rho * math.cos(th)

ap = {r["iata"]: r for r in csv.DictReader(open(f"{U}/data/airports.csv", encoding="utf-8-sig"))}
NAME = {k: v["город"] for k, v in ap.items()}
NAME.update({"MCX": "Махачкала", "NUX": "Новый Уренгой", "NJC": "Нижневартовск", "GSV": "Саратов", "TOF": "Томск",
             "BAX": "Барнаул", "MRV": "Минеральные Воды"})
def P(c):
    if c == "MOW": return proj(37.62, 55.75)
    return proj(float(ap[c]["lon"]), float(ap[c]["lat"]))

legs = list(csv.DictReader(open(f"{U}/results/network_цель2028_базовый.csv", encoding="utf-8-sig")))
STEP = {("KZN", "SVX"): (1, "7 → 14"), ("KUF", "SVX"): (2, "6,5 → 7"), ("OMS", "SVX"): (3, "2 → 7"), ("REN", "SVX"): (3, "2 → 3")}
OPTION = {("MRV", "UFA"): "0 → 3"}

fig = plt.figure(figsize=(16, 9), dpi=200)
fig.patch.set_facecolor(C["bg"])
ax = fig.add_axes([0, 0, 0.70, 1])
ax.set_facecolor(C["bg"]); ax.axis("off")

# экстент карты
_, ytop = proj(62, 68.4); _, ybot = proj(62, 41.2)
H = ytop - ybot; W = H * (16 * 0.70) / 9
xc = (proj(32.5, 56)[0] + proj(86, 56)[0]) / 2
ax.set_xlim(xc - W / 2, xc + W / 2); ax.set_ylim(ybot, ytop)
ax.set_aspect("equal")

geo = json.load(open(G))
for name, polys in geo.items():
    for ring in polys:
        pts = [proj(x, y) for x, y in ring]
        isr = name == "Russia"
        ax.add_patch(Polygon(pts, closed=True, fc=C["russia"] if isr else C["other"],
                             ec=C["rborder"] if isr else C["border"], lw=0.7 if isr else 0.45, zorder=1, joinstyle="round"))

def bez(p0, p1, p2, t):
    return ((1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t*t*p2[0], (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t*t*p2[1])
def arc(a, b, bend=0.14, t=0.5):
    (xa, ya), (xb, yb) = P(a), P(b)
    mx, my = (xa + xb) / 2, (ya + yb) / 2
    dx, dy = xb - xa, yb - ya
    L = math.hypot(dx, dy)
    cx, cy = mx - dy * bend, my + dx * bend
    return Path([(xa, ya), (cx, cy), (xb, yb)], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), bez((xa, ya), (cx, cy), (xb, yb), t)

# Москва — для контраста
mx, my = P("MOW")
ax.scatter([mx], [my], marker="*", s=260, color=C["muted"], zorder=6, edgecolor=C["halo"], linewidth=0.8)
ax.text(mx - 0.008, my - 0.008, "Москва", fontsize=10.5, color=C["muted"], ha="right", va="top", zorder=7,
        path_effects=[pe.withStroke(linewidth=3, foreground=C["halo"])])

cities = set()
# действующие рейсы
for r in legs:
    k = (r["i"], r["j"]); cities.update(k)
    if k in STEP or k in OPTION: continue
    f = float(r["рейсов_нед"])
    path, _ = arc(*k, bend=0.10)
    ax.add_patch(PathPatch(path, fc="none", ec=C["exist"], lw=0.7 + f * 0.22, alpha=0.75, zorder=3, capstyle="round"))
# опция
for k, lab in OPTION.items():
    path, mid = arc(*k, bend=-0.12)
    ax.add_patch(PathPatch(path, fc="none", ec=RED[1], lw=2.0, ls=(0, (3, 3)), alpha=0.9, zorder=4, capstyle="round"))
# шаги 1–3
for k, (s, lab) in sorted(STEP.items(), key=lambda x: -x[1][0]):
    f2 = float(next(r["рейсов_нед"] for r in legs if (r["i"], r["j"]) == k))
    bend = {("KZN","SVX"): 0.13, ("KUF","SVX"): -0.10, ("OMS","SVX"): 0.14, ("REN","SVX"): -0.24}[k]
    path, mid = arc(*k, bend=bend, t={("REN","SVX"): 0.5, ("KUF","SVX"): 0.3}.get(k, 0.5))
    ax.add_patch(PathPatch(path, fc="none", ec=C["halo"], lw=2.4 + f2 * 0.62 + 3, alpha=0.9, zorder=4, capstyle="round"))
    ax.add_patch(PathPatch(path, fc="none", ec=RED[s - 1], lw=2.4 + f2 * 0.62, zorder=5, capstyle="round"))
    # бейдж с номером шага
    ax.add_patch(Circle(mid, 0.0105, fc=RED[s - 1], ec=C["halo"], lw=1.8, zorder=8))
    ax.text(mid[0], mid[1] - 0.0004, str(s), fontsize=11.5, fontweight="bold", color="white", ha="center", va="center", zorder=9)

# узлы
for c in cities:
    x, y = P(c)
    if c == "SVX": continue
    big = c in ("KZN", "KUF", "OMS", "REN", "UFA", "MRV")
    ax.scatter([x], [y], s=46 if big else 22, color=C["node"], zorder=7, edgecolor=C["halo"], linewidth=1.2)
# Екатеринбург
ex, ey = P("SVX")
for rr, al in ((0.030, 0.10), (0.020, 0.18)):
    ax.add_patch(Circle((ex, ey), rr, fc=RED[0], ec="none", alpha=al, zorder=6))
ax.scatter([ex], [ey], s=210, color=RED[0], zorder=8, edgecolor=C["halo"], linewidth=2.2)

# подписи: (dx, dy, ha) в долях единиц проекции
OFF = {"SVX": (0.016, 0.010, "left"), "KZN": (-0.012, 0.010, "right"), "KUF": (-0.012, -0.004, "right"),
       "OMS": (0.010, -0.014, "left"), "REN": (0.000, -0.016, "center"), "UFA": (0.009, -0.008, "left"),
       "MRV": (0.012, -0.004, "left"), "AER": (-0.012, -0.004, "right"), "GOJ": (0.004, 0.012, "center"),
       "GSV": (-0.012, -0.002, "right"), "MCX": (0.012, -0.006, "left"), "NUX": (0.010, 0.006, "left"),
       "TJM": (0.010, 0.010, "left"), "NJC": (0.010, -0.010, "left"), "SGC": (-0.010, 0.010, "right"),
       "TOF": (0.010, 0.006, "left"), "BAX": (0.010, -0.010, "left")}
for c in cities:
    x, y = P(c)
    dx, dy, ha = OFF.get(c, (0.01, 0.006, "left"))
    main = c in ("SVX", "KZN", "KUF", "OMS", "REN")
    if c == "TJM": continue
    if c == "SVX":
        ax.text(x + dx, y + dy + 0.006, "Екатеринбург", fontsize=16, fontweight="bold", color=C["ink"], ha="left", va="bottom",
                zorder=10, path_effects=[pe.withStroke(linewidth=4, foreground=C["halo"])])
        ax.text(x + dx, y + dy + 0.004, "Кольцово — база и хаб Red Wings", fontsize=10.5, color=RED[0] if THEME == "light" else "#FF7A7A", ha="left", va="top",
                zorder=10, path_effects=[pe.withStroke(linewidth=4, foreground=C["halo"])])
        continue
    ax.text(x + dx, y + dy, NAME.get(c, c),
            fontsize=15 if c == "SVX" else (12.5 if main else 10), fontweight="bold" if main else "normal",
            color=C["ink"] if main or c == "SVX" else C["muted"], ha=ha, va="center", zorder=10, linespacing=1.25,
            path_effects=[pe.withStroke(linewidth=4, foreground=C["halo"])])

# --- правая панель ---
pan = fig.add_axes([0.70, 0, 0.30, 1]); pan.axis("off"); pan.set_xlim(0, 1); pan.set_ylim(0, 1)
pan.add_patch(FancyBboxPatch((0.04, 0.03), 0.92, 0.94, boxstyle="round,pad=0,rounding_size=0.03",
                             fc=C["panel"], ec=C["panel_ec"], lw=1.2))
X = 0.11
pan.text(X, 0.895, "Целевая сеть Red Wings 2028", fontsize=19, fontweight="bold", color=C["ink"], va="top")
pan.text(X, 0.835, "Новые SJ-100 — на прямые рейсы\nиз Екатеринбурга. Новый хаб не нужен.",
         fontsize=12.5, color=C["muted"], va="top", linespacing=1.35)
# KPI
k_y = 0.70
for i, (v, l) in enumerate([("1,5", "борта"), ("+0,32", "млрд ₽/год"), ("+118", "тыс. пасс./год")]):
    xx = X + i * 0.285
    pan.text(xx, k_y, v, fontsize=22, fontweight="bold", color=RED[0], va="bottom")
    pan.text(xx, k_y - 0.012, l, fontsize=10.5, color=C["muted"], va="top")
pan.plot([X, 0.89], [0.625, 0.625], color=C["panel_ec"], lw=1)

rows = [(1, "Шаг 1 · выручка ≥ +30%", "Казань – Екатеринбург", "7 → 14 рейсов/нед"),
        (2, "Шаг 2 · ≥ +45%, загрузка ≥ 70%", "Самара – Екатеринбург", "6,5 → 7 рейсов/нед"),
        (3, "Шаг 3 · ≥ +60%, субсидии сохранены", "Омск и Оренбург – Екатеринбург", "2 → 7 и 2 → 3 рейса/нед")]
y = 0.585
for s, cond, route, fr in rows:
    pan.add_patch(Circle((X + 0.025, y - 0.022), 0.024, fc=RED[s - 1], ec="none", transform=pan.transData))
    pan.text(X + 0.025, y - 0.0225, str(s), fontsize=12, fontweight="bold", color="white", ha="center", va="center")
    pan.text(X + 0.08, y, cond, fontsize=10.5, color=C["muted"], va="top")
    pan.text(X + 0.08, y - 0.032, route, fontsize=12.5, fontweight="bold", color=C["ink"], va="top")
    pan.text(X + 0.08, y - 0.066, fr, fontsize=11, color=C["ink"], va="top")
    y -= 0.112
# опция и легенда
pan.plot([X, X + 0.055], [y - 0.012, y - 0.012], color=RED[1], lw=2, ls=(0, (3, 2)))
pan.text(X + 0.08, y, "Опция: Уфа – Минеральные Воды, 3/нед —\nесли «Азимут» не ответит", fontsize=10.5, color=C["muted"], va="top", linespacing=1.3)
y -= 0.068
pan.plot([X, X + 0.055], [y - 0.008, y - 0.008], color=C["exist"], lw=2)
pan.text(X + 0.08, y, "Действующие рейсы Red Wings", fontsize=10.5, color=C["muted"], va="top")
y -= 0.045
pan.scatter([X + 0.027], [y - 0.008], s=110, color=RED[0], edgecolor=C["panel"], linewidth=1.5)
pan.text(X + 0.08, y, "Кольцово уже работает как хаб: там\nпересаживаются 55% пассажиров\nрейса из Омска, 38% — из Самары", fontsize=10.5, color=C["muted"], va="top", linespacing=1.3)
ax.text(0.012, 0.014, "Модель «Пухососы», версия 2 · выручка +60% к мин. тарифам АТОР · прибыль с субсидиями ПП № 1242",
        transform=ax.transAxes, fontsize=9, color=C["muted"], va="bottom", ha="left",
        path_effects=[pe.withStroke(linewidth=3, foreground=C["halo"])], zorder=11)

fig.savefig(OUT, facecolor=C["bg"])
fig.savefig(OUT.replace(".png", ".svg"), facecolor=C["bg"])
print("ok", OUT)
