"""
ШАГ 1. Спрос между городами — гравитационная модель.

Идея: чем больше два города и чем ближе они друг к другу, тем больше людей летает между ними.
    T_ij = k * (M_i * M_j)^a / d_ij^theta
M — «масса» города (население или пассажиропоток аэропорта), d — расстояние, км.

Как подбираем k, a, theta:
  1) Факт: на табло 40 аэропортов (data/boards_week.csv) видно, сколько рейсов в неделю летает
     между каждой парой городов у ВСЕХ перевозчиков. Пассажиров в год = рейсы × кресла × загрузка × 52.
  2) Логарифмируем формулу — получается обычная линейная регрессия:
         ln T = ln k + a * ln(M_i * M_j) − theta * ln d
     и решаем её методом наименьших квадратов (МНК).
  3) Проверка на отложенных парах: делим пары на 5 групп, учим на 4, проверяем на 5-й, и так по кругу.
     Так видно, как модель угадывает пары, которых «не видела» — это и есть наши будущие новые маршруты.

Сравниваем две «массы»: население (как в паспорте) и пассажиропоток аэропорта. Берём ту, что точнее.

Ограничения (честно пишем в паспорте):
  - видим только пары, где уже есть прямые рейсы: спрос пар без рейсов модель экстраполирует;
  - на рейсах из хабов (OVB, SVX) часть пассажиров — транзитные, это завышает поток таких пар.

Запуск: python model/step1_demand.py
Выход:  results/demand_calibration.csv, results/demand_coefs.csv
"""
import numpy as np
import pandas as pd

from params import PARAMS
from common import load_airports, dist_km, RESULTS, DATA, load_boards

MASSES = {  # название варианта -> колонка в airports.csv
    "население": "население_города_млн",
    "пассажиропоток": "масса",
}


def observed_flows(airports: pd.DataFrame) -> pd.DataFrame:
    """Фактический поток по парам городов (обе стороны вместе), пассажиров в год."""
    b = load_boards()
    # оставляем только пары внутри наших 40 аэропортов: для них знаем население и координаты
    b = b[b["откуда"].isin(airports.index) & b["куда"].isin(airports.index)]
    # убираем дубли кодшеринга
    b = b[~b["перевозчик_IATA"].isin(PARAMS["codeshare_exclude"])]
    seats = b["перевозчик_IATA"].map(PARAMS["seats_by_carrier"]).fillna(PARAMS["seats_default"])
    lf = pd.Series(PARAMS["load_factor_market"], index=b.index)
    if PARAMS.get("use_carrier_lf"):
        t = pd.read_csv(DATA / "load_factors.csv", encoding="utf-8-sig")
        lf = b["перевозчик_IATA"].map(dict(zip(t["код"], t["загрузка"]))).fillna(PARAMS["load_factor_market"])
    # пассажиров в год = рейсы в неделю × кресла × загрузка × 52 недели × поправка на сезон (табло — ноябрь)
    b = b.assign(pax_year=b["рейсов_нед"] * seats * lf * PARAMS["weeks_per_year"] * PARAMS.get("season_factor", 1.0))
    # пара без направления: (KUF, SVX) и (SVX, KUF) — одна и та же пара
    b["i"] = b[["откуда", "куда"]].min(axis=1)
    b["j"] = b[["откуда", "куда"]].max(axis=1)
    flows = b.groupby(["i", "j"]).agg(T_obs=("pax_year", "sum"),
                                      рейсов_нед_обе_стороны=("рейсов_нед", "sum"),
                                      перевозчики=("перевозчик_IATA", lambda s: ",".join(sorted(set(s)))))
    flows = flows.reset_index()
    # табло есть только у 40 аэропортов: если второй город пары — не из них, видим лишь одно направление → ×2
    origins = set(load_boards()["откуда"])
    one_side = ~(flows.i.isin(origins) & flows.j.isin(origins))
    flows.loc[one_side, "T_obs"] *= 2
    flows["табло_с_обеих_сторон"] = ~one_side
    flows["km"] = [dist_km(*airports.loc[i, ["lat", "lon"]], *airports.loc[j, ["lat", "lon"]])
                   for i, j in zip(flows.i, flows.j)]
    return flows


def design(flows, airports, mass_col, hubs=(), vakhta=()):
    """Матрица признаков для регрессии: [1, ln(Mi*Mj), ln d, флажок «пара с транзитным хабом»]."""
    m = airports[mass_col]
    ln_mm = np.log(m.loc[flows.i].values * m.loc[flows.j].values)
    cols = [np.ones(len(flows)), ln_mm, np.log(flows.km.values)]
    if hubs:
        cols.append((flows.i.isin(hubs) | flows.j.isin(hubs)).astype(float).values)
    if vakhta:
        cols.append((flows.i.isin(vakhta) | flows.j.isin(vakhta)).astype(float).values)
    return np.column_stack(cols)


def fit_ols(X, y):
    """МНК: коэффициенты, при которых сумма квадратов ошибок минимальна."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def r2(y, y_hat):
    """Доля разброса, которую объясняет модель (1 — идеально, 0 — не лучше среднего)."""
    return 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def cross_validate(X, y, k, seed):
    """Проверка на отложенных парах: предсказание для каждой пары, когда она НЕ была в обучении."""
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(y)) % k
    y_hat = np.empty_like(y)
    for f in range(k):
        test = folds == f
        coef = fit_ols(X[~test], y[~test])
        y_hat[test] = X[test] @ coef
    return y_hat


def evaluate(name, flows, airports, mass_col, hubs=(), vakhta=()):
    """Обучить вариант модели и проверить его на отложенных парах."""
    X = design(flows, airports, mass_col, hubs, vakhta)
    y = np.log(flows.T_obs.values)
    coef = fit_ols(X, y)
    y_cv = cross_validate(X, y, PARAMS["holdout_folds"], PARAMS["random_seed"])
    row = {"вариант": name, "пар": len(y), "ln_k": coef[0], "a": coef[1], "theta": -coef[2],
           "хаб_поправка": coef[3] if hubs else 0.0,
           "вахта_поправка": coef[-1] if vakhta else 0.0,
           "R2_обучение": r2(y, X @ coef), "R2_отложенные": r2(y, y_cv),
           "медианная_ошибка_отложенные": np.median(np.abs(np.exp(y_cv - y) - 1))}
    return row, np.exp(X @ coef), np.exp(y_cv)


def has(airports, col, flows):
    """Есть ли значение колонки у обоих городов пары."""
    return airports[col].loc[flows.i].notna().values & airports[col].loc[flows.j].notna().values


def main():
    airports = load_airports()
    flows_all = observed_flows(airports)
    hubs = PARAMS["transfer_hubs"]
    pop, pax = MASSES["население"], MASSES["пассажиропоток"]

    # 1) Население или пассажиропоток? Сравниваем на одинаковом наборе пар (где известны оба).
    common = flows_all[has(airports, pop, flows_all) & has(airports, pax, flows_all)].reset_index(drop=True)
    table = [evaluate("население", common, airports, pop)[0],
             evaluate("пассажиропоток", common, airports, pax)[0]]

    # 2) Итоговая модель: пассажиропоток + поправка на транзитный хаб, на всех парах, где он известен.
    flows = flows_all[has(airports, pax, flows_all)].reset_index(drop=True)
    table.append(evaluate("пассажиропоток (все пары)", flows, airports, pax)[0])
    table.append(evaluate(f"пассажиропоток + хаб {','.join(hubs)}", flows, airports, pax, hubs)[0])
    final, t_fit, t_cv = evaluate(f"ИТОГ: пассажиропоток + хаб {','.join(hubs)} + вахта", flows, airports, pax, hubs,
                                  PARAMS["vakhta_cities"])
    table.append(final)
    coefs = pd.DataFrame(table)

    flows["T_модель"] = t_fit.round()
    flows["T_отложенные"] = t_cv.round()
    flows["ошибка_раз"] = (flows["T_отложенные"] / flows.T_obs).round(2)
    flows.to_csv(RESULTS / "demand_calibration.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv(RESULTS / "demand_coefs.csv", index=False, encoding="utf-8-sig")

    print(f"Пар с прямыми рейсами внутри 40 аэропортов: {len(flows_all)}, "
          f"с известным пассажиропотоком: {len(flows)}, с известными обеими массами: {len(common)}")
    print(f"Фактический поток на парах: {flows.T_obs.sum()/1e6:.2f} млн пасс./год\n")
    pd.set_option("display.width", 220)
    print(coefs.round(3).to_string(index=False))
    print("\nХудшие предсказания на отложенных парах (во сколько раз ошиблись):")
    w = flows.assign(m=np.abs(np.log(flows["ошибка_раз"]))).sort_values("m", ascending=False)
    print(w[["i", "j", "km", "T_obs", "T_отложенные", "ошибка_раз", "перевозчики"]].head(6).round({"km": 0, "T_obs": 0}).to_string(index=False))


if __name__ == "__main__":
    main()
