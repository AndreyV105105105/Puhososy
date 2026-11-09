"""Весь расчёт одной командой: python model/run_all.py  (≈1–2 минуты)
Пороги (step7_thresholds.py) считаются отдельно — это дольше: python model/step7_thresholds.py all"""
import step1_demand, step2_choice, step3_scenarios, step6_sensitivity, step8_map, step9_report

for step in (step1_demand, step2_choice, step3_scenarios, step6_sensitivity, step8_map, step9_report):
    print(f"\n========== {step.__name__} ==========")
    step.main()
