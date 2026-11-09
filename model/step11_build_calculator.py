"""
Сборка страницы калькулятора: шаблон docs/calculator_template.html + сетка results/calculator_grid.json
→ docs/calculator.html (один файл, открывается в браузере без сервера; годится для GitHub Pages).

Запуск: python model/step11_build_calculator.py   (после step10_calculator_grid.py)
"""
from common import ROOT, RESULTS

t = (ROOT / "docs" / "calculator_template.html").read_text(encoding="utf-8")
data = (RESULTS / "calculator_grid.json").read_text(encoding="utf-8")
body = t.replace("/*DATA*/null", data)
head = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
end = body.index("</style>") + len("</style>")
(ROOT / "docs" / "calculator.html").write_text(head + body[:end] + "\n</head>\n<body>\n" + body[end:] + "\n</body>\n</html>\n",
                                              encoding="utf-8")
print("Калькулятор: docs/calculator.html")
