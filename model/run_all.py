"""Весь расчёт одной командой: python model/run_all.py  (≈2–3 минуты)"""
import step1_demand, step2_choice, step3_scenarios, step6_sensitivity

for step in (step1_demand, step2_choice, step3_scenarios, step6_sensitivity):
    print(f"\n========== {step.__name__} ==========")
    step.main()
