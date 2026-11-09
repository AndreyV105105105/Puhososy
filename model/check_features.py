"""
Проверка: улучшают ли модель спроса дополнительные признаки (туризм, один федеральный округ,
нелинейность по расстоянию). Каждый вариант проверяем на отложенных парах при трёх разных разбиениях.
Вывод (9.11.2026): улучшения в пределах шума разбиения → оставляем простую модель, чтобы
не подгонять 4+ параметра под 59 пар. Главный источник погрешности — сам «факт»
(кресла и загрузка по авиакомпаниям — допущения), а не формула.

Запуск: python model/check_features.py → results/check_features.csv
"""
import numpy as np
import pandas as pd
from step1_demand import load_airports, observed_flows, design, fit_ols, cross_validate, r2
from common import RESULTS

FD = {"SVX": "Урал", "CEK": "Урал", "TJM": "Урал", "KUF": "ПФО", "UFA": "ПФО", "KZN": "ПФО", "PEE": "ПФО",
      "GOJ": "ПФО", "OMS": "Сибирь", "OVB": "Сибирь", "KJA": "Сибирь", "IKT": "Сибирь", "VOG": "Юг",
      "AER": "Юг", "MRV": "СКФО", "KGD": "СЗ"}


def main():
    ap = load_airports()
    fl = observed_flows(ap)
    m = ap["пасс_2025_млн"]
    fl = fl[m.loc[fl.i].notna().values & m.loc[fl.j].notna().values].reset_index(drop=True)
    y = np.log(fl.T_obs.values)
    base = design(fl, ap, "пасс_2025_млн", ["OVB"])
    tour = (fl.i.isin(["AER", "MRV"]) | fl.j.isin(["AER", "MRV"])).astype(float).values
    same = np.array([FD[a] == FD[b] for a, b in zip(fl.i, fl.j)], float)
    lnd2 = np.log(fl.km.values) ** 2
    variants = {"база (итог шага 1)": base, "+ туризм (Сочи, Минводы)": np.column_stack([base, tour]),
                "+ один федеральный округ": np.column_stack([base, same]),
                "+ (ln d)²": np.column_stack([base, lnd2])}
    rows = []
    for name, X in variants.items():
        r, med = [], []
        for seed in (42, 7, 123):
            yc = cross_validate(X, y, 5, seed)
            r.append(r2(y, yc)); med.append(np.median(np.abs(np.exp(yc - y) - 1)))
        rows.append({"вариант": name, "параметров": X.shape[1], "R2_отложенные_среднее": round(np.mean(r), 3),
                     "медианная_ошибка_среднее": round(np.mean(med), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "check_features.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
