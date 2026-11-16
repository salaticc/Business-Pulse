"""Readable view of the immutable published synthetic experiment."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.portfolio_ui import write_overview


def main():
    source=ROOT/'reports/v0.6.0/alert_report.json'
    report=json.loads(source.read_text(encoding='utf-8'))
    test=report['test_metrics'];val=report['validation_metrics']
    data={'title':'Business Pulse — прогноз и проверка предупреждений','subtitle':'Учебный ML-проект · синтетические данные · опубликованный эксперимент v0.6',
    'kpis':[{'label':'Найдено эпизодов на TEST','value':f"{test['incidents']['tp']} / {test['incidents']['total']}"},{'label':'Ложных предупреждений TEST','value':str(test['false_business_warnings'])},{'label':'Проверено политик на validation','value':str(len(report['threshold_comparison']))},{'label':'Дневной recall после подавления','value':f"{test['daily_emitted']['recall']*100:.1f}%"}],
    'charts':[{'title':'Число предупреждений и пропусков','unit':'дни / сообщения','rows':[{'label':'Верные предупреждения TEST','value':test['daily_emitted']['tp']},{'label':'Ложные предупреждения TEST','value':test['daily_emitted']['fp']},{'label':'Пропущенные дни TEST','value':test['daily_emitted']['fn']}]},{'title':'Стоимость ошибок по этапам','unit':'условные единицы, не деньги','rows':[{'label':'Validation','value':val['total_error_cost']},{'label':'TEST','value':test['total_error_cost']}]}],
    'insights':[{'fact':'7 из 8 эпизодов обнаружены, но дневной recall составляет 47,1%.','next':'Оценивать эпизоды и отдельные дни раздельно: подавление повторов меняет дневную confusion matrix.'},{'fact':'Порог 25% И 250 единиц выбран только на validation из 12 политик.','next':'Сохранить разделение выбора и оценки; опубликованный TEST служит проверкой воспроизводимости, а не новым независимым экспериментом.'},{'fact':'В прогнозировании Gradient Boosting сравнивался с простыми базовыми моделями.','next':'По временной CV выбран базовый прогноз, а не более сложная ML-модель. Проверять пользу на новых данных перед внедрением.'}],
    'scope':'Числа взяты из reports/v0.6.0/alert_report.json, а не из коммерческой эксплуатации. Все инциденты и стоимости синтетические. Нет доказанной прибыли, предотвращённого ущерба или внешней доставки уведомлений.'}
    write_overview(ROOT,data);print('Portfolio overview built from published v0.6 JSON; no model or policy changed.')


if __name__=='__main__':main()
