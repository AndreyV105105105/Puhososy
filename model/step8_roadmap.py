"""
Дорожная карта: как распределяются 4 новых SJ-100 до 2028 года — по условиям, а не по датам.
Вариант — прямые рейсы (основа вывода: устойчивее хаба; в версии 2 хаб не открывается до выручки +60%).
Цифры берутся из расчёта порогов (results/thresholds_доходность.csv), поэтому после пересчёта
картинка обновляется сама.

Запуск: python model/step8_roadmap.py   (после step7_thresholds.py доходность)
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
         "REN": "Оренбург", "SGC": "Сургут", "TOF": "Томск", "BAX": "Барнаул", "GSV": "Саратов"}
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


def ru(x, nd=1):
    """Число по-русски: запятая вместо точки, округление «как в школе»."""
    return f"{round(x + 1e-9, nd):.{nd}f}".replace(".", ",")


def main():
    t = pd.read_csv(RESULTS / "thresholds_доходность.csv", encoding="utf-8-sig")
    d = t[t["гипотеза"] == MODE].set_index("значение")
    hub = P["hub_names"][HUB]
    steps = [
        ("Шаг 1 · 2027", "если выручка ≥ +30%\nк мин. тарифам АТОР", 1.3),
        ("Шаг 2 · 2027–2028", "если выручка ≥ +45%\nи загрузка шага 1 ≥ 80%", 1.45),
        ("Шаг 3 · 2028", "если выручка ≥ +60%\nи субсидии сохранены", 1.6),
    ]
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=160)
    ax.set_xlim(0, 13.1); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.text(0.3, 6.85, "Дорожная карта: куда идут 4 новых SJ-100 Red Wings", fontsize=16, weight="bold", color=INK)
    ax.text(0.3, 6.45, "Переход к следующему шагу — по условию, а не по дате. Не выполнено — борта остаются на международных линиях.",
            fontsize=10.5, color=MUTED)
    x0, w, gap = 0.3, 3.95, 0.45
    prev = {}
    for k, (title, cond, fare) in enumerate(steps):
        row = d.loc[fare] if fare in d.index else d.iloc[(d.index.to_series() - fare).abs().argsort()[0]]
        dom = min(float(row["самолётов"]), N)
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 5.1, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#f6f7f9", ec="#d9dde3", lw=1))
        ax.text(x + 0.2, 5.75, title, fontsize=13, weight="bold", color=INK)
        ax.text(x + 0.2, 5.05, cond, fontsize=10, color=BLUE, va="center")
        bx, by, bw, bh = x + 0.2, 4.1, w - 0.4, 0.45
        ax.add_patch(plt.Rectangle((bx, by), bw, bh, fc=GREY, ec="none"))
        ax.add_patch(plt.Rectangle((bx, by), bw * dom / N, bh, fc=RED, ec="none"))
        for i in range(1, N):
            ax.plot([bx + bw * i / N] * 2, [by, by + bh], color="white", lw=2)
        ax.text(bx, by - 0.32, f"внутренние линии: {ru(dom)} борта", fontsize=9.5, color=RED, weight="bold")
        ax.text(bx, by - 0.62, f"международные линии: {ru(N - dom)} борта", fontsize=9.5, color=MUTED)
        cur = routes(row["рейсы"])
        add = {p: f for p, f in cur.items() if prev.get(p, 0) < f}
        lines = [f"{NAMES.get(a, a)} – {NAMES.get(b, b)}: {f}/нед" for (a, b), f in add.items()] or ["—"]
        ax.text(x + 0.2, 3.05, "Новые рейсы и частоты:" if k == 0 else "Добавляется к прошлому шагу:",
                fontsize=9.5, color=MUTED)
        for j, ln in enumerate(lines[:6]):
            ax.text(x + 0.3, 2.7 - j * 0.32, "• " + ln, fontsize=10, color=INK)
        ax.text(x + 0.2, 1.2, f"прибыль +{ru(row['прибыль_млрд'], 2)} млрд ₽/год", fontsize=11, weight="bold", color=INK)
        if k < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 3.6), (x + w + gap - 0.02, 3.6), arrowstyle="-|>",
                                         mutation_scale=18, color=BLUE, lw=1.6))
        prev = cur
    ax.text(0.3, 0.6, "Сейчас (выручка на уровне «Азимута», +15%) новые внутренние рейсы не окупаются — борта на международных линиях. "
            "Новый хаб не открывается ни в одном проверенном городе до выручки +60%.",
            fontsize=9.5, color=MUTED)
    ax.text(0.3, 0.25, "Выручка Red Wings на пассажира к мин. тарифам АТОР («Азимут» ≈ +15%, средняя по Red Wings с международными ≈ +70%). "
            "Прибыль — с субсидиями ПП № 1242.", fontsize=9.5, color=MUTED)
    ax.text(0.3, -0.1, "Казань и Самара из Екатеринбурга сейчас не в перечне субсидий: без субсидии шаг 1 окупается с +45%. "
            "Минводы–Уфа не выдерживает ответа «Азимута» (results/check_competitor.csv).", fontsize=9.5, color=MUTED)
    out = RESULTS / "roadmap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"Дорожная карта: {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
