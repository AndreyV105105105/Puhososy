"""
Сводка результатов для паспорта — собирается автоматически из results/ после каждого пересчёта.
Егору не нужно искать цифры по чатам: всё, что идёт в паспорт и слайды, лежит в одном файле.

Запуск: python model/step9_report.py
Выход:  docs/results_for_passport.md
"""
import numpy as np
import pandas as pd

from params import PARAMS as P
from common import RESULTS, ROOT, load_airports, CORE_ONLY
import step2_choice as ch

OUT = ROOT / "docs" / ("results_for_passport_14.md" if CORE_ONLY else "results_for_passport.md")


def md(df):
    """Таблица pandas → markdown (без внешних пакетов)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def validation():
    ap = load_airports()
    cal = pd.read_csv(RESULTS / "choice_calibration.csv", encoding="utf-8-sig")
    best = cal.loc[cal["ошибка_лог"].idxmin()]
    P["asc_train"], P["asc_car"] = float(best["asc_train"]), float(best["asc_car"])
    df = ch.run_network(ch.network_from_boards(ap), ap, ch.fare_model(ap), ch.demand_model(), float(best["scale"]))
    v = ch._legs_vs_obs(df)
    corr = np.corrcoef(np.log(v["модель_табло"]), np.log(v["T_obs"]))[0, 1]
    med = np.median(np.abs(v["модель_табло"] / v["T_obs"] - 1))
    air = df[["пасс_прямой", "пасс_доехать", "пасс_хаб", "пасс_через_SVX", "пасс_Москва"]].sum().sum()
    return len(v), corr, med, air, df["пасс_Москва"].sum() / air


def main():
    s = []
    s.append("# Результаты модели для паспорта\n")
    s.append("_Файл собирается автоматически (`model/step9_report.py`). Не редактировать руками — "
             "после пересчёта модели он обновится._\n")

    coefs = pd.read_csv(RESULTS / "demand_coefs.csv", encoding="utf-8-sig")
    fin = coefs[coefs["вариант"].str.startswith("ИТОГ")].iloc[0]
    n, corr, med, air, mow = validation()
    s.append("## Точность модели\n")
    s.append(f"- Спрос (гравитация): R² на отложенных парах **{fin['R2_отложенные']:.2f}**, пар в калибровке {int(fin['пар'])}; "
             f"масса города — пассажиропоток аэропорта (население: R² ≈ 0).")
    s.append(f"- Вся модель против табло: **{n} рейсов**, корреляция **{corr:.2f}**, медианная погрешность **{med:.0%}**.")
    s.append(f"- Текущая сеть (S0): {air/1e6:.1f} млн авиапассажиров в год между городами модели, "
             f"через Москву — **{mow:.0%}**.\n")

    sc = pd.read_csv(RESULTS / "scenarios_summary.csv", encoding="utf-8-sig")
    keep = ["гипотеза", "сценарий", "новых_самолётов_дано", "самолётов_занято", "новых_пассажиров_в_год",
            "выручка_млрд", "прибыль_с_субсидиями_млрд", "субсидии_млрд", "прирост_доли_пп",
            "сэкономлено_пасс_часов_млн", "доступность_прирост_%", "пар_без_Москвы_было_стало"]
    s.append("## Сравнение гипотез (таблица для раздела 7)\n")
    s.append(md(sc[[c for c in keep if c in sc.columns]]) + "\n")

    sens = pd.read_csv(RESULTS / "sensitivity.csv", encoding="utf-8-sig")
    wins = sens["лучшая"].value_counts()
    s.append("## Устойчивость вывода\n")
    s.append(f"По одному допущению ({len(sens)} проверок), сколько раз вариант лучший: "
             + ", ".join(f"{k} — {v}" for k, v in wins.items()) + ".\n")
    s.append(md(sens) + "\n")
    mc = pd.read_csv(RESULTS / "monte_carlo.csv", encoding="utf-8-sig")
    mw = mc["лучшая"].value_counts(normalize=True)
    s.append(f"Монте-Карло ({len(mc)} случайных наборов всех допущений): "
             + ", ".join(f"{k} — {v:.0%}" for k, v in mw.items()) + ".\n")

    th = sorted(RESULTS.glob("thresholds_*.csv"))
    if th:
        s.append("## Пороги: при каких условиях окупается хаб и прямые рейсы\n")
        for f in th:
            t = pd.read_csv(f, encoding="utf-8-sig")
            if "рычаг" not in t.columns:              # без субсидий — своя таблица
                s.append(f"**{f.stem}**\n")
                s.append(md(t) + "\n")
                continue
            s.append(f"**Рычаг: {t['рычаг'].iloc[0]}**\n")
            s.append(md(t[["значение", "гипотеза", "прибыль_млрд", "самолётов", "спиц_хаба", "рейсы"]]) + "\n")

    hc = RESULTS / "hub_city_compare.csv"
    if hc.exists():
        s.append("## Выбор города хаба расчётом (тот же флот и алгоритм)\n")
        s.append(md(pd.read_csv(hc, encoding="utf-8-sig")) + "\n")

    rt = RESULTS / "routes_table.csv"
    if rt.exists():
        s.append("## Маршруты выбранного решения (требование ТЗ: источник для каждого А→Б)\n")
        s.append(md(pd.read_csv(rt, encoding="utf-8-sig")) + "\n")
    maps = sorted(RESULTS.glob("map_*.png"))
    if maps:
        s.append("## Карты\n")
        s.extend(f"- `results/{m.name}`" for m in maps)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(s) + "\n", encoding="utf-8")
    print(f"Сводка для паспорта: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
