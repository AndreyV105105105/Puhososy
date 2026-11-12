"""
Полный пересчёт одной командой: python model/run_full.py            (всё, кроме калькулятора)
                                 python model/run_full.py calc 4     (то же + сетка калькулятора на 4 процессах)

Чем отличается от run_all.py: считает ещё пороги, города хаба, целевую сеть, проверки и отчёт,
а независимые расчёты запускает одновременно — по одному на ядро процессора.
Порядок: шаги 1–3 по очереди (следующие шаги читают их результаты) → тяжёлые расчёты параллельно → карты и отчёт.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

FIRST = [["step1_demand.py"], ["step2_choice.py"], ["ml_dataset.py"], ["ml_viability.py"], ["step3_scenarios.py"]]
PARALLEL = [["step7_thresholds.py", "all"], ["step6_sensitivity.py"], ["step7_target_network.py"],
            ["check_short_fare.py"], ["step7_hub_city.py"], ["step7_hub_city.py", "utair"], ["check_competitor.py"], ["check_ml.py"], ["ml_demand.py"]]
LAST = [["segment_share.py"], ["step8_map.py"], ["step8_roadmap.py"], ["step8_architecture.py"], ["explain_routes.py"], ["step9_report.py"]]


def run(args):
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=HERE, env=ENV, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    sec = time.time() - t
    print(f"  {'ок ' if r.returncode == 0 else 'ОШИБКА'} {' '.join(args):<32} {sec:6.0f} с", flush=True)
    if r.returncode != 0:
        print((r.stdout[-1500:] + "\n" + r.stderr[-3000:]).strip(), flush=True)
    return r.returncode


def main():
    args = sys.argv[1:]
    t0 = time.time()
    failed = 0
    print("1. Спрос, выбор пассажира, сценарии", flush=True)
    for a in FIRST:
        failed += run(a) != 0
        if failed:
            sys.exit("Остановлено: без шагов 1–3 остальное считать бессмысленно.")
    workers = max(1, min(len(PARALLEL), os.cpu_count() or 1))
    print(f"2. Пороги, устойчивость, города хаба, целевая сеть — одновременно, процессов: {workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        failed += sum(c != 0 for c in ex.map(run, PARALLEL))
    print("3. Доля в сегменте, карты, дорожная карта, отчёт", flush=True)
    for a in LAST:
        failed += run(a) != 0
    if args[:1] == ["calc"]:
        procs = args[1] if len(args) > 1 else str(os.cpu_count() or 1)
        print(f"4. Сетка калькулятора, процессов: {procs}", flush=True)
        failed += run(["step10_calculator_grid.py", procs]) != 0
        failed += run(["step11_build_calculator.py"]) != 0
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин." + (f" Шагов с ошибкой: {failed}" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
