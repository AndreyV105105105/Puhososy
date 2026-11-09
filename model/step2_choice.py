"""
ШАГ 2. Как полетит пассажир — логит-модель выбора (multinomial logit, McFadden).

Для каждой пары городов есть несколько вариантов: прямой рейс каждого перевозчика,
пересадка в Москве, пересадка в хабе (в сценариях с хабом), поезд.
У каждого варианта есть «полезность» U — чем больше, тем привлекательнее:
    U = b_time*часы + b_transfer*пересадки + b_freq*ln(рейсов в неделю) + b_fare*цена
Доля варианта: s_r = e^U_r / сумма e^U_k  (так устроена логит-модель).

Почему так:
  - b_freq*ln(f) даёт ту самую «S-кривую» из авиационной практики: доля перевозчика ~ f^1,35;
  - b_fare = b_time / стоимость часа: 750 ₽ дороже = на час дольше. Время и деньги согласованы.

Пассажиров варианта = спрос пары (шаг 1) × доля варианта.

Запуск: python model/step2_choice.py   (сначала step1_demand.py)
Выход:  results/choice_S0.csv — текущая сеть (сценарий S0) по всем парам городов
"""
import math
import numpy as np
import pandas as pd

from params import PARAMS
from common import load_airports, dist_km, block_h, RESULTS, DATA

P = PARAMS
B_FARE = P["b_time"] / P["vot_rub_h"]       # полезность за 1 ₽
EXTRA_POINTS = {"Москва": P["moscow_point"], "Санкт-Петербург": (59.8003, 30.2625)}  # LED, для тарифов


# ---------------- данные ----------------
def demand_model():
    """Коэффициенты итоговой гравитационной модели из шага 1."""
    c = pd.read_csv(RESULTS / "demand_coefs.csv", encoding="utf-8-sig")
    row = c[c["вариант"].str.startswith("ИТОГ")].iloc[0]
    return row["ln_k"], row["a"], row["theta"], row["хаб_поправка"], row.get("вахта_поправка", 0.0)


def fare_model(airports):
    """Тариф ≈ a + b*км + c*[в обход Москвы]. Подбираем по 24 парам АТОР (минимальные тарифы, август 2026)."""
    f = pd.read_csv(DATA / "fares_ator_2026-08-03.csv", encoding="utf-8-sig")
    coords = {r["город"]: (r["lat"], r["lon"]) for _, r in airports.reset_index().iterrows()}
    coords.update(EXTRA_POINTS)
    km = np.array([dist_km(*coords[o], *coords[d]) for o, d in zip(f["откуда"], f["куда"])])
    bypass = (f["в_обход_Москвы"] == "да").astype(float).values
    X = np.column_stack([np.ones(len(f)), km, bypass])
    coef, *_ = np.linalg.lstsq(X, f["тариф_без_багажа_руб"].values, rcond=None)
    return coef


def network_from_boards(airports):
    """Текущая сеть (S0): прямые рейсы между нашими городами и рейсы в Москву, рейсов в неделю в одну сторону."""
    b = pd.read_csv(DATA / "boards_week.csv", encoding="utf-8-sig")
    # в Москву: все перевозчики (тут Аэрофлот — настоящий, не кодшеринг)
    to_mow = b[b["куда"].isin(P["moscow_airports"])].groupby("откуда")["рейсов_нед"].sum().to_dict()
    # между нашими городами: без дублей кодшеринга
    r = b[b["откуда"].isin(airports.index) & b["куда"].isin(airports.index)
          & ~b["перевозчик_IATA"].isin(P["codeshare_exclude"])].copy()
    r["i"] = r[["откуда", "куда"]].min(axis=1)
    r["j"] = r[["откуда", "куда"]].max(axis=1)
    # табло с обоих концов → сумма по двум направлениям / 2 = рейсов в одну сторону;
    # если второй город не из 16 аэропортов с табло — видим одно направление, делить не нужно
    origins = set(b["откуда"])
    g = r.groupby(["i", "j", "перевозчик_IATA"])["рейсов_нед"].sum().reset_index()
    both = g.i.isin(origins) & g.j.isin(origins)
    g["рейсов_нед"] = g["рейсов_нед"].astype(float)
    g.loc[both, "рейсов_нед"] /= 2
    direct = {}
    for _, x in g.iterrows():
        direct.setdefault((x.i, x.j), {})[x["перевозчик_IATA"]] = x["рейсов_нед"]
    # рейсы в Москву для городов без табло: оценка по связи «частота ~ пассажиропоток» у 16 аэропортов
    m = airports["масса"]
    known = [c for c in to_mow if c in m.index and pd.notna(m[c]) and to_mow[c] > 0]
    if known:
        bx, ax_ = np.polyfit(np.log([m[c] for c in known]), np.log([to_mow[c] for c in known]), 1)
        for c in airports.index:
            if c not in origins and pd.notna(m[c]):
                to_mow[c] = float(np.exp(ax_ + bx * math.log(m[c])))
    return {"direct": direct, "to_moscow": to_mow, "hub": None}


# ---------------- модель выбора ----------------
def fare(km, bypass, fc):
    return fc[0] + fc[1] * km + fc[2] * bypass


def utility(hours, transfers, freq, price, asc=0.0):
    return (P["b_time"] * hours + P["b_transfer"] * transfers
            + P["b_freq"] * math.log(max(freq, 0.1)) + B_FARE * price + asc)


_GEO = {}
_RAIL = None


def rail_table():
    """Факт РЖД по 9 парам (Даня): {(i, j): (часы, поездов в неделю, цена купе или None)}."""
    global _RAIL
    if _RAIL is None:
        _RAIL = {}
        path = DATA / "rail.csv"
        if path.exists():
            r = pd.read_csv(path, encoding="utf-8-sig")
            for _, x in r.iterrows():
                k = (x.i, x.j) if x.i < x.j else (x.j, x.i)
                price = None if pd.isna(x["купе_руб"]) else float(x["купе_руб"])   # пустая цена ≠ 0
                _RAIL[k] = (x["быстрейший_мин"] / 60, float(x["поездов_нед"]), price)
    return _RAIL


def _geo(airports):
    """Кэш координат и расстояний: модель вызывает options() десятки тысяч раз, без кэша это медленно."""
    k = id(airports)
    if k not in _GEO:
        coords = {a: (float(airports.at[a, "lat"]), float(airports.at[a, "lon"])) for a in airports.index}
        coords["MOW"] = P["moscow_point"]
        _GEO[k] = (coords, {})
    return _GEO[k]


def _dist(airports, a, b):
    coords, cache = _geo(airports)
    if (a, b) not in cache:
        cache[(a, b)] = cache[(b, a)] = dist_km(*coords[a], *coords[b])
    return cache[(a, b)]


def options(i, j, airports, net, fc):
    """Все варианты добраться из i в j и их характеристики.
    У авиавариантов есть поле «плечи» — какие рейсы реально летят (нужно для загрузки и сверки с табло)."""
    key = lambda a, b: (a, b) if a < b else (b, a)
    D = lambda a, b: _dist(airports, a, b)
    km = D(i, j)
    opts = []
    # 1) прямые рейсы — отдельный вариант на каждого перевозчика
    for carrier, f in net["direct"].get(key(i, j), {}).items():
        opts.append(dict(вариант="прямой", перевозчик=carrier, часы=block_h(km), пересадок=0,
                         рейсов_нед=f, цена=fare(km, 1, fc), плечи=[key(i, j)]))
    # 2) через Москву — есть всегда, частота = узкое место из двух плеч
    k1, k2 = D(i, "MOW"), D("MOW", j)
    f_m = min(net["to_moscow"].get(i, 0), net["to_moscow"].get(j, 0))
    if f_m > 0:
        opts.append(dict(вариант="через Москву", перевозчик="разные",
                         часы=block_h(k1) + P["mct_moscow_h"] + block_h(k2), пересадок=1, рейсов_нед=f_m,
                         цена=P["through_fare_factor"] * (fare(k1, 0, fc) + fare(k2, 0, fc)), плечи=[]))
    # 3) через наш хаб — только в сценариях с хабом и если оба города — его «спицы»
    hub = net.get("hub")
    # волны стыковок в Самаре появляются только вместе с базой (первая новая спица), до этого сеть = S0
    if hub and hub.get("active", True) and i in hub["spokes"] and j in hub["spokes"] and hub["city"] not in (i, j):
        h = hub["city"]
        k1, k2 = D(i, h), D(h, j)
        opts.append(dict(вариант="через хаб", перевозчик="WZ",
                         часы=block_h(k1) + P["mct_hub_h"] + block_h(k2), пересадок=1,
                         рейсов_нед=min(hub["spokes"][i], hub["spokes"][j]),
                         цена=P["hub_fare_factor"] * fare(km, 1, fc), плечи=[key(i, h), key(h, j)]))
    # 3б) через действующий хаб Red Wings в Екатеринбурге: обе «спицы» — рейсы Red Wings из Кольцово
    e = P["existing_hub"]
    if e not in (i, j):
        f1 = net["direct"].get(key(i, e), {}).get("WZ", 0)
        f2 = net["direct"].get(key(e, j), {}).get("WZ", 0)
        if f1 and f2:
            k1, k2 = D(i, e), D(e, j)
            opts.append(dict(вариант="через Екатеринбург", перевозчик="WZ",
                             часы=block_h(k1) + P["mct_existing_hub_h"] + block_h(k2), пересадок=1,
                             рейсов_нед=min(f1, f2), цена=P["hub_fare_factor"] * fare(km, 1, fc),
                             плечи=[key(i, e), key(e, j)]))
    # 4) доехать на машине до соседнего аэропорта A (≤ 400 км) и улететь оттуда прямым рейсом
    for src, dst in ((i, j), (j, i)):
        for a in airports.index:
            if a in (i, j):
                continue
            kg = D(src, a)
            if kg > P["ground_max_km"]:
                continue
            ka = D(a, dst)
            for carrier, f in net["direct"].get(key(a, dst), {}).items():
                opts.append(dict(вариант="доехать до аэропорта", перевозчик=carrier,
                                 часы=kg / P["ground_speed_kmh"] + P["ground_buffer_h"] + block_h(ka), пересадок=1,
                                 рейсов_нед=f, цена=kg * P["ground_rub_per_km"] + fare(ka, 1, fc),
                                 плечи=[key(a, dst)], тариф_рейса=fare(ka, 1, fc)))
    # 5) поезд: факт РЖД, если пара есть в data/rail.csv, иначе формула
    if km <= P["rail_max_km"]:
        rk = km * P["rail_detour"]
        hours, freq, price = rk / P["rail_speed_kmh"], P["rail_freq_week"], rk * P["rail_rub_per_km"]
        fact = rail_table().get(key(i, j))
        if fact:
            hours, freq = fact[0], fact[1]
            price = fact[2] if fact[2] is not None else price
        if freq > 0:
            opts.append(dict(вариант="поезд", перевозчик="РЖД", часы=hours, пересадок=0,
                             рейсов_нед=freq, цена=price, asc=P["asc_train"], плечи=[]))
    # 6) машина или междугородний автобус
    if km <= P["road_max_km"]:
        rd = km * P["road_detour"]
        opts.append(dict(вариант="машина/автобус", перевозчик="авто", часы=rd / P["ground_speed_kmh"], пересадок=0,
                         рейсов_нед=P["car_freq_week"], цена=rd * P["ground_rub_per_km"], asc=P["asc_car"], плечи=[]))
    # страховка: если вариантов нет вовсе (город без табло и без оценки рейсов в Москву) — считаем «через Москву»
    # по оценке частоты, рейсы в Москву есть из всех аэропортов списка
    if not opts:
        opts.append(dict(вариант="через Москву", перевозчик="разные", часы=block_h(D(i, "MOW")) + P["mct_moscow_h"]
                         + block_h(D("MOW", j)), пересадок=1, рейсов_нед=7,
                         цена=P["through_fare_factor"] * (fare(D(i, "MOW"), 0, fc) + fare(D("MOW", j), 0, fc)), плечи=[]))
    # доли по логиту
    u = np.array([utility(o["часы"], o["пересадок"], o["рейсов_нед"], o["цена"], o.get("asc", 0)) for o in opts])
    s = np.exp(u - u.max()); s /= s.sum()          # вычитаем max для численной устойчивости
    logsum = float(u.max() + math.log(np.exp(u - u.max()).sum()))   # «ожидаемая полезность» всех вариантов
    for o, share in zip(opts, s):
        o["доля"] = share
        o["logsum"] = logsum
    return km, opts


def run_network(net, airports, fc, dem, scale=1.0):
    """Для всех пар городов: спрос × доли → пассажиры по вариантам.
    scale — поправочный множитель рынка (см. calibrate_scale).
    Дополнительно считаем поток на каждом рейсе (плече) — его и сверяем с табло."""
    ln_k, a, theta, hub_coef, vakhta_coef = dem
    cities = [c for c in airports.index if pd.notna(airports.loc[c, "масса"])]
    m = airports["масса"]
    rows, legs = [], {}
    for x in range(len(cities)):
        for y in range(x + 1, len(cities)):
            i, j = sorted((cities[x], cities[y]))
            km, opts = options(i, j, airports, net, fc)
            # спрос «откуда–куда» (флажок хаба = 0: транзитных пассажиров S7 в нём нет)
            vk = vakhta_coef if (i in P["vakhta_cities"] or j in P["vakhta_cities"]) else 0.0
            T = scale * math.exp(ln_k + a * math.log(m[i] * m[j]) - theta * math.log(km) + vk)
            for o in opts:
                if o["вариант"] in ("прямой", "доехать до аэропорта", "через Екатеринбург"):
                    for leg in o["плечи"]:
                        legs[leg] = legs.get(leg, 0) + T * o["доля"]
            sh = lambda cond: sum(o["доля"] for o in opts if cond(o))
            rows.append(dict(i=i, j=j, km=round(km), спрос=round(T),
                             доля_прямой=sh(lambda o: o["вариант"] == "прямой"),
                             доля_RW_прямой=sh(lambda o: o["вариант"] == "прямой" and o["перевозчик"] == "WZ"),
                             доля_доехать=sh(lambda o: o["вариант"] == "доехать до аэропорта"),
                             доля_хаб=sh(lambda o: o["вариант"] == "через хаб"),
                             доля_через_SVX=sh(lambda o: o["вариант"] == "через Екатеринбург"),
                             доля_Москва=sh(lambda o: o["вариант"] == "через Москву"),
                             доля_поезд=sh(lambda o: o["вариант"] == "поезд"),
                             доля_авто=sh(lambda o: o["вариант"] == "машина/автобус")))
    df = pd.DataFrame(rows)
    for col in ["прямой", "RW_прямой", "доехать", "хаб", "через_SVX", "Москва", "поезд", "авто"]:
        df[f"пасс_{col}"] = (df["спрос"] * df[f"доля_{col}"]).round()
    # поток на рейсах, как его видно на табло: из транзитного хаба летят ещё и пересаживающиеся
    lf = pd.DataFrame([dict(i=k[0], j=k[1], модель=v) for k, v in legs.items()])
    lf["множ_табло"] = [math.exp(hub_coef) if (i in P["transfer_hubs"] or j in P["transfer_hubs"]) else 1.0
                        for i, j in zip(lf.i, lf.j)]
    lf["модель_табло"] = lf["модель"] * lf["множ_табло"]
    df.attrs["legs"] = lf
    return df


def _legs_vs_obs(df):
    obs = pd.read_csv(RESULTS / "demand_calibration.csv", encoding="utf-8-sig")[["i", "j", "T_obs"]]
    return df.attrs["legs"].merge(obs, on=["i", "j"])


def calibrate_scale(df):
    """Шаг 1 подобран по рейсам, а здесь спрос делится ещё на Москву и поезд.
    Подбираем один множитель рынка так, чтобы пассажиров на рейсах в модели было столько же, сколько на табло.
    (Плюс он снимает систематическое занижение при переходе от логарифмов обратно к числам.)"""
    v = _legs_vs_obs(df)
    return v["T_obs"].sum() / v["модель_табло"].sum()


def log_error(df):
    """Ошибка модели на рейсах: корень из среднего квадрата ошибки в логарифмах."""
    v = _legs_vs_obs(df)
    e = np.log(v["модель_табло"] / v["T_obs"])
    return float(np.sqrt((e ** 2).mean()))


def calibrate(net, airports, fc, dem, quick=False):
    """Калибровка шага 2 по факту табло:
    1) константы поезда и машины (asc) — перебираем сетку и берём ту пару, где ошибка на рейсах минимальна
       (константа отражает всё, что не описано временем и ценой: дорогу до вокзала, комфорт, багаж);
    2) множитель рынка scale — чтобы пассажиров на рейсах в модели было столько же, сколько на табло."""
    rows = []
    # quick=True — для проверки устойчивости: поезд почти не влияет, перебираем только константу машины
    grid_t = [P["asc_train"]] if quick else np.arange(-6, 0.01, 1.0)
    grid_c = np.arange(-12, -3.99, 2.0) if quick else np.arange(-12, -3.99, 1.0)
    for asc_t in grid_t:
        for asc_c in grid_c:
            P["asc_train"], P["asc_car"] = float(asc_t), float(asc_c)
            scale = calibrate_scale(run_network(net, airports, fc, dem))
            err = log_error(run_network(net, airports, fc, dem, scale))
            rows.append(dict(asc_train=asc_t, asc_car=asc_c, scale=scale, ошибка_лог=err))
    grid = pd.DataFrame(rows)
    if not quick:
        grid.to_csv(RESULTS / "choice_calibration.csv", index=False, encoding="utf-8-sig")
    best = grid.loc[grid["ошибка_лог"].idxmin()]
    P["asc_train"], P["asc_car"] = float(best["asc_train"]), float(best["asc_car"])
    return float(best["scale"]), best


def validate(df):
    v = _legs_vs_obs(df)
    corr = np.corrcoef(np.log(v["модель_табло"]), np.log(v["T_obs"]))[0, 1]
    v["модель/факт"] = (v["модель_табло"] / v["T_obs"]).round(2)
    print(f"\nПроверка на {len(v)} рейсах (парах с прямыми рейсами): всего модель/факт = "
          f"{v['модель_табло'].sum()/v['T_obs'].sum():.2f}, корреляция (лог) = {corr:.2f}, "
          f"медианная ошибка = {np.median(np.abs(v['модель/факт'] - 1)):.0%}")
    sel = v[(v.i == "KUF") | (v.j == "KUF")]
    print("Пары с Самарой (модель/факт):", ", ".join(f"{a}-{b} {r}" for a, b, r in zip(sel.i, sel.j, sel["модель/факт"])))


def main():
    airports = load_airports()
    fc = fare_model(airports)
    dem = demand_model()
    net = network_from_boards(airports)
    scale, best = calibrate(net, airports, fc, dem)
    df = run_network(net, airports, fc, dem, scale)
    print(f"Калибровка: константа поезда = {best['asc_train']:.1f}, машины/автобуса = {best['asc_car']:.1f}, "
          f"множитель рынка = {scale:.2f}")
    df.round(3).to_csv(RESULTS / "choice_S0.csv", index=False, encoding="utf-8-sig")

    print(f"Тариф ≈ {fc[0]:.0f} ₽ + {fc[1]:.2f} ₽/км + {fc[2]:.0f} ₽ если в обход Москвы")
    air = df[["пасс_прямой", "пасс_доехать", "пасс_хаб", "пасс_через_SVX", "пасс_Москва"]].sum().sum()
    print(f"\nS0 — текущая сеть, {len(df)} пар городов (14 аэропортов):")
    print(f"  авиапассажиров между регионами: {air/1e6:.2f} млн/год")
    print(f"  из них через Москву: {df['пасс_Москва'].sum()/air:.0%}  ← «москвоцентричность»")
    print(f"  доехали до соседнего аэропорта: {df['пасс_доехать'].sum()/1e6:.2f} млн/год")
    print(f"  пересели в Екатеринбурге на рейсы Red Wings: {df['пасс_через_SVX'].sum()/1e3:.0f} тыс./год")
    print(f"  поездом: {df['пасс_поезд'].sum()/1e6:.2f} млн/год, машиной/автобусом: {df['пасс_авто'].sum()/1e6:.2f} млн/год")
    print(f"  Red Wings прямыми рейсами: {df['пасс_RW_прямой'].sum()/1e3:.0f} тыс./год")

    validate(df)
    print("\nПары с наибольшей долей через Москву (кандидаты для хаба или прямых рейсов):")
    top = df[df["пасс_прямой"] == 0].sort_values("пасс_Москва", ascending=False)
    print(top[["i", "j", "km", "спрос", "доля_Москва", "доля_поезд", "пасс_Москва"]].head(10).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
