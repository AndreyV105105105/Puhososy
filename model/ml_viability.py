"""
ML-блок, шаг 2. Модель жизнеспособности прямого рейса: вероятность p, что на паре городов держится регулярный
прямой рейс, по «географии и рынку» пары (results/ml_pairs.csv, 780 пар, 179 с рейсами).

Модели:
  • база — логистическая регрессия только на гравитационном спросе (то, что модель уже умеет);
  • логистическая регрессия на всех признаках;
  • градиентный бустинг с монотонными ограничениями: p не падает с ростом аэропортов и с проигрышем земли.
Проверка: кросс-валидация с группами по городам — город из проверочной части не встречается в обучении
(иначе модель «помнит» города и точность завышена). 5 разбиений × 5 повторов, прогноз вне обучения (out-of-fold).
Контроль фактом: три снятых рейса 2021–2023 должны получить низкую p.

Запуск: python model/ml_viability.py   (после ml_dataset.py)
Выход:  results/ml_viability.csv (p по каждой паре), results/ml_metrics.csv, results/ml_importance.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS
from ml_dataset import FEATURES

MONO = {"лог_км": 0, "лог_масса_мин": 1, "лог_масса_макс": 1, "часы_земля": 1, "выигрыш_самолёта_ч": 1,
        "соседей_250км": 0, "тень_соседа": -1, "база_RW": 1, "вахта": 1}
FOLDS, REPEATS = 5, 5
LABEL = {"лог_км": "расстояние", "лог_масса_мин": "меньший аэропорт", "лог_масса_макс": "больший аэропорт",
         "часы_земля": "время по земле", "выигрыш_самолёта_ч": "выигрыш самолёта по времени",
         "соседей_250км": "соседние аэропорты", "тень_соседа": "крупный аэропорт рядом", "база_RW": "база Red Wings", "вахта": "вахтовый город"}


def X_of(df, name):
    if name == "гравитация":
        return np.log(df[["гравитация"]].to_numpy())
    X = df[FEATURES].to_numpy(dtype=float)
    if name == "логрегрессия":   # квадрат расстояния: рейсы держатся на средних дальностях
        X = np.column_stack([X, df["лог_км"].to_numpy() ** 2])
    return X


def make(name):
    if name == "бустинг":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                              min_samples_leaf=20, monotonic_cst=[MONO[f] for f in FEATURES],
                                              random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def city_cv(df, name):
    """Прогноз вне обучения: обучаемся на парах без городов проверочной части, прогнозируем пары с ними."""
    cities = sorted(set(df["i"]) | set(df["j"]))
    y = df["рейс_есть"].to_numpy()
    X = X_of(df, name)
    s, n = np.zeros(len(df)), np.zeros(len(df))
    for r in range(REPEATS):
        perm = np.random.default_rng(r).permutation(cities)
        for f in range(FOLDS):
            test_c = set(perm[f::FOLDS])
            te = (df["i"].isin(test_c) | df["j"].isin(test_c)).to_numpy()
            m = make(name).fit(X[~te], y[~te])
            s[te] += m.predict_proba(X[te])[:, 1]
            n[te] += 1
    return s / n


def metrics(y, p):
    return dict(ROC_AUC=roc_auc_score(y, p), Brier=brier_score_loss(y, p), logloss=log_loss(y, p))


def calibration(y, p, bins=5):
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"p_прогноз": p, "доля_факт": y}).groupby(q, observed=True).agg(
        пар=("p_прогноз", "size"), p_прогноз=("p_прогноз", "mean"), доля_факт=("доля_факт", "mean")).round(3)


def local_factors(model, df, name, top=3):
    """Объяснение по паре: насколько меняется p, если признак пары заменить на медиану по всем парам."""
    X = X_of(df, name)
    base = model.predict_proba(X)[:, 1]
    eff = {}
    for k, f in enumerate(FEATURES):
        Xm = X.copy()
        Xm[:, k] = np.median(X[:, k])
        if name == "логрегрессия" and f == "лог_км":
            Xm[:, -1] = Xm[:, k] ** 2
        eff[f] = base - model.predict_proba(Xm)[:, 1]
    out = []
    for r in range(len(df)):
        top_f = sorted(FEATURES, key=lambda f: -abs(eff[f][r]))[:top]
        out.append("; ".join(f"{LABEL[f]} {eff[f][r]:+.2f}" for f in top_f if abs(eff[f][r]) >= 0.01))
    return out


def main():
    df = pd.read_csv(RESULTS / "ml_pairs.csv", encoding="utf-8-sig")
    y = df["рейс_есть"].to_numpy()
    rows, P = [], {}
    for name in ("гравитация", "логрегрессия", "бустинг"):
        P[name] = city_cv(df, name)
        rows.append({"модель": name, **{k: round(v, 3) for k, v in metrics(y, P[name]).items()}})
    met = pd.DataFrame(rows)
    print("Проверка по городам (5×5, прогноз вне обучения):\n" + met.to_string(index=False))

    best = met.loc[met["ROC_AUC"].idxmax(), "модель"]
    p = P[best]
    print(f"\nКалибровка ({best}):\n" + calibration(y, p).to_string())

    df["p_логрегрессия"], df["p_бустинг"], df["p_гравитация"] = P["логрегрессия"], P["бустинг"], P["гравитация"]
    df["p"] = p
    closed = df[df["снят_2021_2023"] == 1][["пара", "км", "p", "p_логрегрессия", "p_бустинг"]]
    print("\nСнятые рейсы 2021–2023 (ML-фильтр модели пропускает пары с p ≥ 0,5):\n" + closed.round(3).to_string(index=False))
    un = df[df["рейс_есть"] == 0]["p"]
    print(f"\nПары без рейсов: средняя p {un.mean():.2f}, медиана {un.median():.2f}, p ≥ 0,5 у {int((un >= 0.5).sum())} пар")
    print(df[df["рейс_есть"] == 0].nlargest(10, "p")[["пара", "км", "p"]].round(2).to_string(index=False))

    # важность признаков: модель на всех данных, перестановки (насколько падает AUC, если признак перемешать)
    full = make(best).fit(X_of(df, best), y)
    imp = permutation_importance(full, X_of(df, best), y, scoring="roc_auc", n_repeats=20, random_state=0)
    names = FEATURES + (["лог_км²"] if best == "логрегрессия" else [])
    imp = pd.DataFrame({"признак": names, "падение_AUC": imp.importances_mean.round(3)}).sort_values(
        "падение_AUC", ascending=False)
    print("\nВажность признаков:\n" + imp.to_string(index=False))

    df["факторы_ML"] = local_factors(full, df, best)
    met["выбрана"] = met["модель"] == best
    met.to_csv(RESULTS / "ml_metrics.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(RESULTS / "ml_importance.csv", index=False, encoding="utf-8-sig")
    df.drop(columns=["гравитация"]).round(4).to_csv(RESULTS / "ml_viability.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
