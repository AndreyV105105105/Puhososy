"""
«№2 в своём сегменте»: доля перевозчиков в межрегиональных рейсах в обход Москвы между 40 городами модели.
Считаем кресла в неделю по табло вылета (все перевозчики, без кодшеринга), сейчас и после целевой сети 2028.
Это кресла, а не пассажиры, и только наши 40 городов — оценка места Red Wings в сегменте, а не официальная статистика.

Запуск: python model/segment_share.py   (после step7_target_network.py)  →  results/segment_share.csv
"""
import pandas as pd

from params import PARAMS as P
from common import load_airports, RESULTS
import step2_choice as ch

NAMES = {"S7": "S7", "UT": "Utair", "FV": "Россия", "DP": "Победа", "A4": "Азимут", "N4": "Северный ветер",
         "U6": "Уральские авиалинии", "WZ": "Red Wings", "YC": "Ямал", "RT": "ЮВТ Аэро", "Y7": "NordStar",
         "5N": "Смартавиа", "EO": "Пегас Флай", "4G": "Газпром авиа"}


def seats(f, c):
    return 2 * f * P["seats_by_carrier"].get(c, P["seats_default"])   # туда и обратно, кресел в неделю


def main():
    net = ch.network_from_boards(load_airports())["direct"]
    now = {}
    for p, cs in net.items():
        for c, f in cs.items():
            now[c] = now.get(c, 0) + seats(f, c)
    t = pd.read_csv(RESULTS / "target_network_2028.csv", encoding="utf-8-sig")
    add = sum(seats(max(0, r["рейсов_нед_2028"] - r["рейсов_нед_сейчас"]), "WZ") for _, r in t.iterrows())
    after = dict(now)
    after["WZ"] = after.get("WZ", 0) + add
    df = pd.DataFrame({"перевозчик": [NAMES.get(c, c) for c in now], "кресел_нед_сейчас": list(now.values()),
                       "кресел_нед_2028": [after[c] for c in now]})
    df["доля_сейчас_%"] = (df["кресел_нед_сейчас"] / df["кресел_нед_сейчас"].sum() * 100).round(1)
    df["доля_2028_%"] = (df["кресел_нед_2028"] / df["кресел_нед_2028"].sum() * 100).round(1)
    df["место_сейчас"] = df["кресел_нед_сейчас"].rank(ascending=False, method="min").astype(int)
    df["место_2028"] = df["кресел_нед_2028"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("кресел_нед_2028", ascending=False)
    df.round(0).to_csv(RESULTS / "segment_share.csv", index=False, encoding="utf-8-sig")
    rw = df[df["перевозчик"] == "Red Wings"].iloc[0]
    second = df.sort_values("кресел_нед_2028", ascending=False).iloc[1]
    per_ac = add / max(1e-9, float(pd.read_csv(RESULTS / "thresholds_доходность.csv", encoding="utf-8-sig")
                                   .query("гипотеза == 'H2 прямые рейсы' and значение == 1.6")["самолётов"].iloc[0]))
    gap = second["кресел_нед_2028"] - rw["кресел_нед_2028"]
    print(df.head(8).to_string(index=False))
    print(f"\nRed Wings: место {rw['место_сейчас']} → {rw['место_2028']}; до №2 ({second['перевозчик']}) не хватает "
          f"{gap:.0f} кресел в неделю ≈ {gap / per_ac:.1f} борта сверх целевой сети ({per_ac:.0f} кресел/нед на борт)")


if __name__ == "__main__":
    main()
