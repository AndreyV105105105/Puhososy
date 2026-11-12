"""
ML-блок, вариант Б. ML-поправка спроса: учимся на ошибке модели на 179 рейсах текущей сети.

Цель для рейса (пары с прямыми рейсами):  r = ln(T_факт / T_модель),
  T_факт   — пассажиров в год по табло (results/demand_calibration.csv, шаг 1),
  T_модель — поток на этом рейсе в модели шага 2 (S0, после калибровки), как его видно на табло.
Признаки — те же «география и рынок», что в ML-фильтре (results/ml_pairs.csv) + транзитный хаб на конце пары.
Частоты и цены рейсов в признаки не берём.

Модели: r̂ = 0 (как сейчас), r̂ = среднее, ридж-регрессия, неглубокий бустинг.
Проверка по городам (5 разбиений × 5 повторов), все метрики — только на прогнозах вне обучения.
Стоп-правило: если корреляция растёт меньше чем на 0,05 или ошибка падает меньше чем на 5% — в модель не встраиваем.

Запуск: python model/ml_demand.py   (после step1, step2, ml_dataset)
Выход:  results/ml_demand_metrics.csv, results/ml_demand_pairs.csv (r̂ по всем 780 парам)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from params import PARAMS as P
from common import load_airports, RESULTS
import step2_choice as ch
from ml_dataset import FEATURES

FEATS = FEATURES + ["транзитный_хаб"]
FOLDS, REPEATS = 5, 5


def legs_s0():
    """Поток на рейсах текущей сети в модели шага 2 и факт табло."""
    ap = load_airports()
    cal = pd.read_csv(RESULTS / "choice_calibration.csv", encoding="utf-8-sig")
    best = cal.loc[cal["ошибка_лог"].idxmin()]
    P["asc_train"], P["asc_car"] = float(best["asc_train"]), float(best["asc_car"])
    df = ch.run_network(ch.network_from_boards(ap), ap, ch.fare_model(ap), ch.demand_model(), float(best["scale"]))
    return ch._legs_vs_obs(df)


def table():
    v = legs_s0()
    pairs = pd.read_csv(RESULTS / "ml_pairs.csv", encoding="utf-8-sig")
    pairs["транзитный_хаб"] = (pairs["i"].isin(P["transfer_hubs"]) | pairs["j"].isin(P["transfer_hubs"])).astype(int)
    d = pairs.merge(v[["i", "j", "модель_табло", "T_obs"]], on=["i", "j"], how="left")
    d["r"] = np.log(d["T_obs"] / d["модель_табло"])
    return d


def make(name):
    if name == "ридж":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=15,
                                         l2_regularization=1.0, random_state=0)


def city_cv(d, name):
    cities = sorted(set(d["i"]) | set(d["j"]))
    X, y = d[FEATS].to_numpy(float), d["r"].to_numpy()
    s, n = np.zeros(len(d)), np.zeros(len(d))
    for rep in range(REPEATS):
        perm = np.random.default_rng(rep).permutation(cities)
        for f in range(FOLDS):
            test_c = set(perm[f::FOLDS])
            te = (d["i"].isin(test_c) | d["j"].isin(test_c)).to_numpy()
            if name == "среднее":
                s[te] += y[~te].mean()
            elif name == "r=0":
                pass
            else:
                s[te] += make(name).fit(X[~te], y[~te]).predict(X[te])
            n[te] += 1
    return s / n


def metrics(d, rhat):
    """Как в проверке шага 2: модель (с поправкой) против факта на рейсах."""
    m = np.log(d["модель_табло"].to_numpy()) + rhat
    f = np.log(d["T_obs"].to_numpy())
    e = m - f
    ratio = np.exp(e)
    return dict(корреляция_лог=np.corrcoef(m, f)[0, 1], ошибка_лог=np.sqrt((e ** 2).mean()),
                медианная_ошибка=np.median(np.abs(ratio - 1)), в_пределах_2_раз=((ratio >= 0.5) & (ratio <= 2)).mean())


def main():
    d_all = table()
    d = d_all.dropna(subset=["r"]).reset_index(drop=True)
    rows, R = [], {}
    for name in ("r=0", "среднее", "ридж", "бустинг"):
        R[name] = city_cv(d, name)
        rows.append({"модель": name, **{k: round(v, 3) for k, v in metrics(d, R[name]).items()}})
    met = pd.DataFrame(rows)
    print(f"Рейсов: {len(d)}. Проверка по городам (5×5, прогноз вне обучения):\n" + met.to_string(index=False))
    base = met.iloc[0]
    best = met.iloc[2:].sort_values("ошибка_лог").iloc[0]
    gain_c = best["корреляция_лог"] - base["корреляция_лог"]
    gain_e = 1 - best["ошибка_лог"] / base["ошибка_лог"]
    ok = gain_c >= 0.05 and gain_e >= 0.05
    print(f"\nЛучшая: {best['модель']}: корреляция {gain_c:+.3f}, ошибка {-gain_e:+.1%} → "
          + ("проходит стоп-правило" if ok else "НЕ проходит стоп-правило: остаётся проверкой"))

    # поправка по всем 780 парам: модель на всех рейсах
    full = make(best["модель"]).fit(d[FEATS].to_numpy(float), d["r"].to_numpy())
    d_all["r_hat"] = full.predict(d_all[FEATS].to_numpy(float))
    d_all["множитель"] = np.exp(d_all["r_hat"])
    oof = dict(zip(zip(d["i"], d["j"]), R[best["модель"]]))
    d_all["r_hat_вне_обучения"] = [oof.get((i, j)) for i, j in zip(d_all["i"], d_all["j"])]
    cl = d_all[d_all["снят_2021_2023"] == 1][["пара", "r_hat", "множитель"]]
    print("\nСнятые рейсы 2021–2023: множитель спроса из ML (ручная поправка модели ×0,3):\n" + cl.round(2).to_string(index=False))
    un = d_all[d_all["рейс_есть"] == 0]["множитель"]
    print(f"Пары без рейсов: множитель медиана {un.median():.2f}, среднее {un.mean():.2f}")
    sv = d_all[d_all["рейс_есть"] == 1]["множитель"]
    print(f"Пары с рейсами:  множитель медиана {sv.median():.2f}, среднее {sv.mean():.2f}")

    met["лучшая"] = met["модель"] == best["модель"]
    met["стоп_правило_пройдено"] = ok
    met.to_csv(RESULTS / "ml_demand_metrics.csv", index=False, encoding="utf-8-sig")
    d_all[["i", "j", "пара", "км", "рейс_есть", "снят_2021_2023", "модель_табло", "T_obs", "r", "r_hat",
           "r_hat_вне_обучения", "множитель"]].round(4).to_csv(RESULTS / "ml_demand_pairs.csv", index=False,
                                                              encoding="utf-8-sig")


if __name__ == "__main__":
    main()
