"""Build an offline recruiter overview from verified aggregate values only."""
import html
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_overview(root, data):
    root=Path(root);(root/'docs').mkdir(exist_ok=True);(root/'reports').mkdir(exist_ok=True)
    (root/'reports/overview_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    fig=plt.figure(figsize=(14,10),facecolor='#f6f8fc');fig.text(.055,.95,data['title'],fontsize=23,weight='bold',color='#142536')
    fig.text(.055,.919,data['subtitle'],fontsize=11,color='#586879')
    for i,k in enumerate(data['kpis'][:4]):
        x=.055+i*.235
        fig.text(x,.855,k['value'],fontsize=23,weight='bold',color='#117c78')
        fig.text(x,.824,k['label'],fontsize=11,color='#243445')
    palette=['#117c78','#4f71c4','#8699c7','#bc7b3c']
    for i,c in enumerate(data['charts'][:2]):
        ax=fig.add_axes([.07+i*.475,.43,.405,.31],facecolor='white')
        rows=c['rows'][:8]
        labels=[str(r['label']) for r in rows];vals=[r['value'] for r in rows]
        ax.barh(range(len(rows)),vals,color=palette[i],height=.55)
        ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set_title(c['title'],loc='left',pad=17,fontsize=12,weight='bold')
        ax.tick_params(axis='both',labelsize=9,length=0);ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True)
        if vals:
            maximum=max([abs(v) for v in vals]+[1]);ax.set_xlim(min([0]+vals)*1.15,maximum*1.24)
            for j,v in enumerate(vals):ax.text(v+maximum*.02,j,f'{v:,.2f}'.replace(',',' '),va='center',fontsize=9)
        ax.set_xlabel(c.get('unit',''),fontsize=9)
        for spine in ax.spines.values():spine.set_visible(False)
    y=.325
    for insight in data['insights'][:3]:
        for line in textwrap.wrap(insight['fact'],105):
            fig.text(.055,y,line,fontsize=11,weight='bold',color='#243445');y-=.022
        for line in textwrap.wrap(insight['next'],115):
            fig.text(.055,y,line,fontsize=10,color='#586879');y-=.02
        y-=.018
    fig.text(.055,.035,textwrap.fill(data['scope'],145),fontsize=8,color='#586879')
    fig.savefig(root/'reports/overview.png',dpi=140,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#162638;font:16px/1.55 system-ui,sans-serif}main{max-width:1140px;margin:auto;padding:44px 24px}h1{font-size:36px;line-height:1.15;margin:0 0 12px}h2{font-size:21px}p{max-width:90ch}.muted{color:#56687a}.scope{padding:16px 0;border-top:1px solid #dbe2eb}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:30px 0}.kpi{background:white;padding:20px;border-radius:12px;border:1px solid #e1e6ef}.value{font-size:30px;font-weight:750;color:#117c78}.panel{background:#fff;padding:24px;border:1px solid #e1e6ef;border-radius:12px;margin:24px 0}select,button{font:inherit;padding:8px 12px;border:1px solid #a9b6c6;border-radius:7px;background:white}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{text-align:left;border-bottom:1px solid #e4e9f0;padding:9px}td:last-child{text-align:right;font-variant-numeric:tabular-nums}.bar{height:9px;background:#117c78;border-radius:4px;min-width:1px}.insight{border-top:1px solid #e1e6ef;padding:14px 0}.tag{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:#117c78}.footer{font-size:13px;color:#586879}@media(max-width:700px){.kpis{grid-template-columns:repeat(2,1fr)}h1{font-size:28px}main{padding:24px 16px}.panel{padding:16px}}</style>
<main><div class="tag">Обзор аналитического проекта</div><h1 id="title"></h1><p id="subtitle" class="muted"></p><section id="kpis" class="kpis"></section>
<section class="panel"><h2>Исследовать показатели</h2><label for="view">Разрез графика </label><select id="view"></select><button id="csv">Скачать этот срез CSV</button><p class="muted">Переключатель меняет только срез ниже. KPI показывают весь описанный период.</p><h3 id="chartTitle"></h3><div id="bars"></div></section>
<section class="panel"><h2>Что означают результаты</h2><div id="insights"></div></section><p class="scope" id="scope"></p><p class="footer">Визуальный обзор рассчитанных показателей. Определения метрик и границы применения приведены в README проекта.</p></main>
<script>const d=__PAYLOAD__;const el=id=>document.getElementById(id);el('title').textContent=d.title;el('subtitle').textContent=d.subtitle;el('scope').textContent=d.scope;
for(const k of d.kpis){const box=document.createElement('div');box.className='kpi';const v=document.createElement('div');v.className='value';v.textContent=k.value;const label=document.createElement('div');label.textContent=k.label;box.append(v,label);el('kpis').append(box)}
d.charts.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.title;el('view').append(o)});
function show(){const c=d.charts[+el('view').value];el('chartTitle').textContent=c.title+' · '+(c.unit||'');el('bars').replaceChildren();const table=document.createElement('table');const max=Math.max(...c.rows.map(x=>Math.abs(x.value)),1);for(const r of c.rows){const tr=document.createElement('tr'),name=document.createElement('td'),visual=document.createElement('td'),num=document.createElement('td');name.textContent=r.label;num.textContent=r.value.toLocaleString('ru-RU',{maximumFractionDigits:2});const b=document.createElement('div');b.className='bar';b.style.width=Math.abs(r.value)/max*100+'%';b.style.background=r.value<0?'#bd4a51':'#117c78';visual.append(b);tr.append(name,visual,num);table.append(tr)}el('bars').append(table)}
el('view').onchange=show;show();for(const i of d.insights){const section=document.createElement('section');section.className='insight';const f=document.createElement('strong');f.textContent=i.fact;const p=document.createElement('p');p.textContent=i.next;section.append(f,p);el('insights').append(section)}
el('csv').onclick=()=>{const c=d.charts[+el('view').value];const quote=s=>'"'+String(s).replaceAll('"','""')+'"';const text='\\ufefflabel;value\\n'+c.rows.map(r=>quote(r.label)+';'+r.value).join('\\n');const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='overview-slice.csv';a.click();URL.revokeObjectURL(url)};</script></html>'''
    (root/'docs/index.html').write_text(template.replace('__TITLE__',html.escape(data['title'])).replace('__PAYLOAD__',payload),encoding='utf-8')
