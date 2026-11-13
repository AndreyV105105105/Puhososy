"""
ВЕРСИЯ 3: дорожная карта и карты сети по итоговому расчёту (после step11_v3_final.py).

Дорожная карта — по основной сети (без Минвод-Уфы): для каждого уровня выручки лучший вариант (прямые, если другой
вариант не лучше них больше чем на 0,1 млрд — шум подбора) и его рейсы
летом и зимой; рядом — прибавка, если «Азимут» не ответит (те же варианты с Минводами-Уфой).
Карты — лучший вариант основной сети при выручке +60% и +80%, отдельно лето и зима.

Запуск: python model/step12_v3_outputs.py
Выход:  results/v3_roadmap.csv, results/v3_compare.csv,
        results/network_v3_<вариант>_<выручка>_<сезон>.csv, results/map_v3_<вариант>_<выручка>_<сезон>.png
"""
import pickle

import pandas as pd

from params import PARAMS as P
from common import RESULTS, load_airports
import step2_choice as ch
import step3_scenarios as sc
import step8_map as mp
import step11_v3_final as fin

MAIN = "основная (без Минвод-Уфы)"
BONUS = {"прямые": "прямые + Минводы-Уфа", "гибрид с Уфой": "гибрид с Уфой + Минводы-Уфа",
         "хаб в Уфе": "хаб в Уфе + Минводы-Уфа"}
MAP_FARES = (1.6, 1.8)
NOISE = 0.1
SEASON_RU = {"лето": "лето (июнь–сентябрь)", "зима": "зима (октябрь–май)"}


def compare(df):
    t = df[df["бортов_дано"] == 4].pivot_table(index="выручка", columns="вариант", values="прибыль_млрд")
    t.to_csv(RESULTS / "v3_compare.csv", encoding="utf-8-sig")
    return t


def roadmap(df):
    d = df[(df["бортов_дано"] == 4)].set_index(["выручка", "вариант"])
    rows = []
    for fare in sorted(df["выручка"].unique()):
        main = {v: d.loc[(fare, v)] for v in BONUS if (fare, v) in d.index}
        # прямые — если другой вариант не лучше них больше чем на NOISE (шум подбора до 0,14 млрд,
        # research/model_v3_risk_checks.md): при равенстве проще сеть без базы хаба
        best_v = "прямые"
        for v in ("гибрид с Уфой", "хаб в Уфе"):
            if main[v]["прибыль_млрд"] > main[best_v]["прибыль_млрд"] + NOISE:
                best_v = v
        b = main[best_v]
        bonus = d.loc[(fare, BONUS[best_v])] if (fare, BONUS[best_v]) in d.index else None
        rows.append(dict(выручка=fare, лучший_вариант=best_v, прибыль_млрд=b["прибыль_млрд"],
                         прямые_млрд=main["прямые"]["прибыль_млрд"],
                         гибрид_млрд=main["гибрид с Уфой"]["прибыль_млрд"],
                         хаб_млрд=main["хаб в Уфе"]["прибыль_млрд"],
                         бортов_лето=b["бортов_лето"], бортов_зима=b["бортов_зима"],
                         новых_пассажиров=b["новых_пассажиров"], прирост_доли_пп=b["прирост_доли_пп"],
                         рейсы_лето=b["рейсы_лето"], рейсы_зима=b["рейсы_зима"],
                         если_Азимут_не_ответит_млрд=None if bonus is None else bonus["прибыль_млрд"],
                         если_Азимут_не_ответит_рейсы_лето=None if bonus is None else bonus["рейсы_лето"],
                         если_Азимут_не_ответит_рейсы_зима=None if bonus is None else bonus["рейсы_зима"]))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "v3_roadmap.csv", index=False, encoding="utf-8-sig")
    return out


def maps(rm, nets_store):
    M = sc.setup()
    P.update(fin.SETTINGS)
    airports = load_airports()
    s0 = ch.network_from_boards(airports)["direct"]
    made = []
    for fare in MAP_FARES:
        v = rm.set_index("выручка").loc[fare, "лучший_вариант"]
        key = (fare, 4, v)
        if key not in nets_store:
            continue
        mode, no_mrv, _ = fin.VARIANTS[v]
        Mx = dict(M)
        if no_mrv:
            Mx["direct_candidates"] = M["direct_candidates"] - {("MRV", "UFA")}
        P["seasons"] = True
        SM = sc.season_models(Mx)
        case = dict(P["cases"]["базовый"], fare=fare)
        for s, Ms, _ in SM:
            r = sc.evaluate(nets_store[key][s], Ms, case, detail=True)
            tag = f"{v.replace(' ', '_')}_{fare:g}_{s}"
            path = RESULTS / f"network_v3_{tag}.csv"
            pd.DataFrame(r["legs"]).assign(
                lat_i=lambda x: x.i.map(lambda c: M["coords"][c][0]), lon_i=lambda x: x.i.map(lambda c: M["coords"][c][1]),
                lat_j=lambda x: x.j.map(lambda c: M["coords"][c][0]), lon_j=lambda x: x.j.map(lambda c: M["coords"][c][1]),
            ).to_csv(path, index=False, encoding="utf-8-sig")
            pct = round((fare - 1) * 100)
            title = (f"Версия 3, {v}, выручка +{pct}%, {SEASON_RU[s]}: сеть Red Wings между городами модели "
                     f"(без Минвод–Уфы)")
            out = mp.draw(mode, airports, s0, title=title, path=path, out=RESULTS / f"map_v3_{tag}.png",
                          hub_note="кандидат в хаб (в версии 3 гибрид не лучше прямых больше чем на шум подбора)")
            made.append(out)
        P["seasons"] = False
    return made


def main():
    df = pd.read_csv(RESULTS / "v3_final_scenarios.csv", encoding="utf-8-sig")
    print(compare(df).round(3).to_string())
    rm = roadmap(df)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(rm.T.to_string())
    store = pickle.load(open(RESULTS / "v3_final_nets.pkl", "rb"))
    for m in maps(rm, store):
        print("Карта:", m.relative_to(RESULTS.parent))


if __name__ == "__main__":
    main()
