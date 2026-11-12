"""
Дорожная карта: как распределяются 4 новых SJ-100 до 2028 года — по условиям, а не по датам.
Вариант — прямые рейсы (основа вывода: устойчивее хаба; в версии 2 хаб не открывается до выручки +60%).
Цифры берутся из расчёта порогов (results/thresholds_доходность.csv), поэтому после пересчёта
картинка обновляется сама. Шаг 4 — сценарий большого флота (results/fleet_scenarios.csv, step7_fleet.py):
если новых бортов 8 и больше, к сети из Екатеринбурга добавляется хаб в Уфе (гибрид).

Запуск: python model/step8_roadmap.py   (после step7_thresholds.py доходность и step7_fleet.py)
Выход:  results/roadmap.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

from params import PARAMS as P
from common import RESULTS

N = P["new_aircraft"]["базовый"]
RED, GREY, BLUE, INK, MUTED = "#c8102e", "#b9bec7", "#1f5fa8", "#1d232b", "#5b6573"
MODE = "H2 прямые рейсы"

NAMES = {"KZN": "Казань", "SVX": "Екатеринбург", "KUF": "Самара", "UFA": "Уфа", "CEK": "Челябинск",
         "OMS": "Омск", "MRV": "Минводы", "GOJ": "Н. Новгород", "TJM": "Тюмень", "VOG": "Волгоград",
         "STW": "Ставрополь", "ULV": "Ульяновск", "OGZ": "Владикавказ", "AER": "Сочи", "KRR": "Краснодар",
         "REN": "Оренбург", "KJA": "Красноярск", "HMA": "Ханты-Мансийск", "SGC": "Сургут", "TOF": "Томск", "BAX": "Барнаул", "GSV": "Саратов"}
HUB = P["hub_city"]


def routes(s):
    """'KZN-SVX×14; KUF-UFA×3; KUF-UFA×7' → {пара: итоговая частота}; город хаба ставим первым."""
    best = {}
    for part in [x.strip() for x in str(s).split(";") if x.strip() and "×" in x]:
        pair, f = part.split("×")
        a, b = pair.split("-")
        if b == HUB:
            a, b = b, a
        best[(a, b)] = max(best.get((a, b), 0), int(float(f)))
    return best


def wrap(s, width=42):
    """Перенос строки по словам для узких колонок."""
    out, cur = [], ""
    for wd in s.split(" "):
        if cur and len(cur) + 1 + len(wd) > width:
            out.append(cur); cur = "  " + wd
        else:
            cur = f"{cur} {wd}" if cur else wd
    return out + [cur]


def ru(x, nd=1):
    """Число по-русски: запятая вместо точки, округление «как в школе»."""
    return f"{round(x + 1e-9, nd):.{nd}f}".replace(".", ",")


BIG_FLEET, BIG_FARE = 8, 1.6


def big_fleet_step():
    """Шаг 4: гибрид с хабом в Уфе на 8 бортах при выручке +60% против прямых на тех же 8 бортах."""
    path = RESULTS / "fleet_scenarios.csv"
    if not path.exists():
        return None
    f = pd.read_csv(path, encoding="utf-8-sig")
    f = f[(f["бортов_дано"] == BIG_FLEET) & (f["выручка_к_АТОР"] == BIG_FARE)].set_index("вариант")
    if "гибрид в Уфе" not in f.index or "прямые" not in f.index:
        return None
    return f.loc["гибрид в Уфе"], f.loc["прямые"]


def main():
    t = pd.read_csv(RESULTS / "thresholds_доходность.csv", encoding="utf-8-sig")
    d = t[t["гипотеза"] == MODE].set_index("значение")
    big = big_fleet_step()
    hub = P["hub_names"][HUB]
    steps = [
        ("Шаг 1 · 2027", "выручка ≥ +30% (без субсидии +45%)\nи загрузка 7 рейсов ≥ 85%", 1.3),
        ("Шаг 2 · 2027–2028", "выручка ≥ +45% (без субсидии +60%)\nи загрузка шага 1 ≥ 70%", 1.45),
        ("Шаг 3 · 2028", "если выручка ≥ +60%\nи субсидии сохранены", 1.6),
    ]
    if big is not None:
        steps.append(("Шаг 4 · 2029+", f"если новых бортов ≥ {BIG_FLEET}\nи выручка ≥ +60%", "big"))
    fig, ax = plt.subplots(figsize=(18.8, 6.2), dpi=160)
    ax.set_xlim(0, 18.8); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(0.3, 6.85, f"Дорожная карта: куда идут новые SJ-100 Red Wings ({N} борта; шаг 4 — если их {BIG_FLEET} и больше)",
            fontsize=16, weight="bold", color=INK)
    ax.text(0.3, 6.45, "Переход к следующему шагу — по условию, а не по дате. Не выполнено — борта остаются на международных линиях.",
            fontsize=10.5, color=MUTED)
    x0, w, gap = 0.3, 4.2, 0.45
    prev = {}
    for k, (title, cond, fare) in enumerate(steps):
        n = N
        if fare == "big":
            row, alt = big
            row = row.rename({"бортов_занято": "самолётов"})
            n = BIG_FLEET
        else:
            row = d.loc[fare] if fare in d.index else d.iloc[(d.index.to_series() - fare).abs().argsort()[0]]
        dom = min(float(row["самолётов"]), n)
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 5.1, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#f6f7f9", ec="#d9dde3", lw=1))
        ax.text(x + 0.2, 5.75, title, fontsize=13, weight="bold", color=INK)
        ax.text(x + 0.2, 5.05, cond, fontsize=10, color=BLUE, va="center")
        bx, by, bw, bh = x + 0.2, 4.1, w - 0.4, 0.45
        ax.add_patch(plt.Rectangle((bx, by), bw, bh, fc=GREY, ec="none"))
        ax.add_patch(plt.Rectangle((bx, by), bw * dom / n, bh, fc=RED, ec="none"))
        for i in range(1, n):
            ax.plot([bx + bw * i / n] * 2, [by, by + bh], color="white", lw=2)
        ax.text(bx, by - 0.32, f"внутренние линии: {ru(dom)} из {n}", fontsize=9.5, color=RED, weight="bold")
        ax.text(bx, by - 0.62, f"остальные — международные: {ru(n - dom)}", fontsize=9.5, color=MUTED)
        cur = routes(row["рейсы"])
        add = {p: f for p, f in cur.items() if prev.get(p, 0) < f}
        if fare == "big":   # сначала спицы хаба, прямые одной строкой
            spokes = {p: f for p, f in add.items() if HUB in p}
            other = [NAMES.get(b if a == "SVX" else a, a) for (a, b) in add if HUB not in (a, b)]
            lines = [f"хаб {NAMES.get(HUB)}: " + ", ".join(f"{NAMES.get(b, b)} {f}" for (a, b), f in spokes.items())] \
                + ([f"+ прямые: {', '.join(dict.fromkeys(other))}"] if other else [])
            lines = [ln for s in lines for ln in wrap(s)]
        else:
            lines = [f"{NAMES.get(a, a)} – {NAMES.get(b, b)}: {f}/нед" for (a, b), f in add.items()] or ["—"]
        ax.text(x + 0.2, 3.05, "Новые рейсы и частоты:" if k == 0 else "Добавляется к прошлому шагу:",
                fontsize=9.5, color=MUTED)
        for j, ln in enumerate(lines[:5]):
            ax.text(x + 0.3, 2.7 - j * 0.32, ("  " + ln) if ln.startswith("  ") else "• " + ln, fontsize=10, color=INK)
        ax.text(x + 0.2, 1.2, f"прибыль +{ru(row['прибыль_млрд'], 2)} млрд ₽/год", fontsize=11, weight="bold", color=INK)
        if fare == "big":
            ax.text(x + 0.2, 1.6, f"только прямые на {n} бортах: +{ru(alt['прибыль_млрд'], 2)} млрд", fontsize=9.5, color=MUTED)
        if k < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 3.6), (x + w + gap - 0.02, 3.6), arrowstyle="-|>",
                                         mutation_scale=18, color=BLUE, lw=1.6))
        prev = cur
    ax.text(0.3, 0.6, "Сейчас (выручка на уровне «Азимута», +15%) новые внутренние рейсы не окупаются — борта на международных линиях. "
            "Хаб не открывается ни в одном проверенном городе до +60%; при 8+ бортах — только как надстройка к прямым.",
            fontsize=9.5, color=MUTED)
    ax.text(0.3, 0.25, "Выручка Red Wings на пассажира к мин. тарифам АТОР («Азимут» ≈ +15%, средняя по Red Wings с международными ≈ +70%). "
            "Прибыль — с субсидиями ПП № 1242.", fontsize=9.5, color=MUTED)
    ax.text(0.3, -0.1, "Казань и Самара из Екатеринбурга не в перечне субсидий 2026 — шаг 0: заявка на 2027; без субсидии пороги шагов 1–2 выше на 15 п.п. "
            "Минводы–Уфа не выдерживает ответа «Азимута» (results/check_competitor.csv).", fontsize=9.5, color=MUTED)
    out = RESULTS / "roadmap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"Дорожная карта: {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
