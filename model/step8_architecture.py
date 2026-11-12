"""
Схема модели для слайда: гибрид «правила + модели, обученные на данных» + оптимизация + объяснение.
Запуск: python model/step8_architecture.py   →  results/architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from common import RESULTS

INK, MUTED, RED, BLUE, BG, EDGE = "#1d232b", "#5b6573", "#c8102e", "#1f5fa8", "#f6f7f9", "#d9dde3"
BLOCKS = [
    ("Данные", "факты", ["табло вылета 40 аэропортов", "пассажиропоток аэропортов", "тарифы АТОР, 55 реальных цен",
                         "поезда, субсидии, закрытия", "снятые рейсы 2021–2023"]),
    ("Правила", "ограничения отрасли", ["дальность ≤ 1500 км (ПП 1242)", "загрузка ≤ 85%", "частоты 3 / 7 / 14 в неделю",
                                       "до соседнего аэропорта ≤ 250 км", "новые короткие пары ≤ 3/нед"]),
    ("Модели на данных", "обучены на 179 рейсах", ["спрос: гравитационная модель", "выбор маршрута: логит —",
                                                  "оценка варианта по времени,", "пересадкам, частоте и цене",
                                                  "поправки по фактам ×0,3 и ×0,6"]),
    ("Оптимизация", "куда ставить борта", ["жадный алгоритм: прибыль", "на час полёта", "учёт перегонов, базы хаба,",
                                          "закрытий и субсидий", "равный флот у всех вариантов"]),
    ("Объяснение", "для пользователя", ["пороги: при какой выручке", "что окупается", "карточка каждого маршрута:",
                                       "почему, условие, риски", "калькулятор сети онлайн"]),
]


def main():
    fig, ax = plt.subplots(figsize=(16.5, 5.6), dpi=160)
    ax.set_xlim(0, 16.5); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.text(0.25, 5.2, "Как устроена модель: правила + модели на данных + оптимизация + объяснение",
            fontsize=15, weight="bold", color=INK)
    w, gap, x0 = 2.98, 0.33, 0.25
    for k, (title, sub, lines) in enumerate(BLOCKS):
        x = x0 + k * (w + gap)
        hl = title in ("Правила", "Модели на данных")
        ax.add_patch(FancyBboxPatch((x, 0.95), w, 3.85, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=BG, ec=RED if hl else EDGE, lw=1.8 if hl else 1))
        ax.text(x + 0.18, 4.4, title, fontsize=13, weight="bold", color=RED if hl else INK)
        ax.text(x + 0.18, 4.05, sub, fontsize=9.5, color=BLUE)
        for j, ln in enumerate(lines):
            ax.text(x + 0.16, 3.55 - j * 0.48, ln, fontsize=9, color=INK)
        if k < len(BLOCKS) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 2.85), (x + w + gap - 0.03, 2.85), arrowstyle="-|>",
                                         mutation_scale=16, color=BLUE, lw=1.5))
    ax.text(0.25, 0.55, "Гибрид — красные блоки: жёсткие правила отрасли ограничивают варианты, модели на данных оценивают спрос "
            "и качество каждого маршрута (время, пересадки, частота, цена).", fontsize=9.5, color=MUTED)
    ax.text(0.25, 0.2, "Каждая рекомендация объяснима: results/route_cards.md — почему маршрут, при каком условии, какие риски.",
            fontsize=9.5, color=MUTED)
    out = RESULTS / "architecture.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"Схема: {out.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
