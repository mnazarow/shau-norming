#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локальный веб-интерфейс нормирования ШАУ.

Разделы:
  • Прогноз     — загрузка спецификации (PDF/EPLAN-Excel/CSV) → трудоёмкость с разбивкой;
  • Обучение    — обучение ML-модели (CatBoost / numpy-fallback) с метриками и Anchor-тестом;
  • Аналитика   — графики по датасету (распределение часов, факт vs baseline, по IP/ПЛК);
  • Статистика  — сводные показатели и драйверы трудоёмкости.

Запуск:
    python3 app.py                 # http://127.0.0.1:8000
    python3 app.py --host 0.0.0.0 --port 8000

Зависимости: стандартная библиотека + parse_spec/norms_model/train_model
(+ pdfplumber/openpyxl для форматов, pandas/numpy для аналитики). Flask НЕ требуется.
"""
import os
import json
import base64
import tempfile
import argparse
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import parse_spec
import norms_model
import train_model

# ------------------------------------------------------------------ состояние
PREDICTOR = None        # функция features -> ML-поправка (после обучения / загрузки модели)
LAST_TRAIN = None       # последний отчёт обучения
DATA_CANDIDATES = ["training_data.csv", "synthetic_dataset.csv"]


def active_dataset():
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _load_saved_model():
    """Если рядом есть обученная CatBoost-модель — поднимаем предиктор при старте."""
    global PREDICTOR
    if os.path.exists("model_shau.cbm"):
        try:
            from catboost import CatBoostRegressor
            import pandas as pd
            m = CatBoostRegressor(); m.load_model("model_shau.cbm")
            PREDICTOR = lambda f: float(m.predict(
                pd.DataFrame([{k: f.get(k) for k in norms_model.ALL_FEATURES}])
                [norms_model.ALL_FEATURES])[0])
        except Exception:
            PREDICTOR = None


_load_saved_model()

EDITABLE_NUM = norms_model.NUM_FEATURES
LABELS = {
    "width_mm": "Ширина, мм", "height_mm": "Высота, мм", "depth_mm": "Глубина, мм",
    "mount_panel": "Монтажная панель (0/1)", "din_inclined": "Наклонная DIN (0/1)",
    "cable_entries": "Кабельные вводы", "breakers_le63": "Автоматы до 63А",
    "load_switches": "Выключатели нагрузки", "contactors": "Контакторы",
    "rcd": "УЗО/дифавтоматы", "freq_converters": "Преобразователи частоты",
    "soft_starters": "УПП", "power_supplies": "Блоки питания",
    "phase_relays": "Реле контроля фаз", "relays": "Промежуточные реле",
    "fans": "Вентиляторы", "thermostats": "Термостаты", "fuse_holders": "Держатели предохр.",
    "signal_lamps": "Сигнальные лампы", "analog_4_20mA": "Аналог. цепи 4-20мА",
    "interfaces_count": "Сетевые интерфейсы", "terminals": "Клеммы",
}


# ------------------------------------------------------------------ прогноз
def predict(features):
    bd = norms_model.estimate_hours(features, breakdown=True)
    baseline = bd["ИТОГО"]
    ml_delta, total = None, baseline
    if PREDICTOR is not None:
        try:
            ml_delta = PREDICTOR(features)
            total = baseline + ml_delta
        except Exception:
            ml_delta = None
    return {"breakdown": bd, "baseline": round(baseline, 1),
            "ml_delta": (round(ml_delta, 1) if ml_delta is not None else None),
            "total": round(total, 1)}


def features_from_upload(filename, content_bytes):
    ext = os.path.splitext(filename)[1].lower() or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
        tf.write(content_bytes); tmp = tf.name
    try:
        feats, _ = parse_spec.build_features(tmp)
    finally:
        os.unlink(tmp)
    for k in ("mount_panel", "din_inclined"):
        feats[k] = int(bool(feats.get(k)))
    feats.setdefault("interfaces_count",
                     sum(1 for v in feats.get("interfaces", {}).values() if v))
    return feats, None


# ------------------------------------------------------------------ аналитика/статистика
def dataset_report(path):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path)
    y = df["actual_hours"].astype(float)
    base = df["baseline_hours"].astype(float)
    # гистограмма часов
    counts, edges = np.histogram(y, bins=10)
    # рассеяние факт vs baseline (выборка до 300 точек)
    samp = df.sample(min(300, len(df)), random_state=1)
    scatter = [{"x": round(float(b), 1), "y": round(float(a), 1)}
               for a, b in zip(samp["actual_hours"], samp["baseline_hours"])]
    # средние по категориям
    def by_cat(col):
        g = df.groupby(col)["actual_hours"]
        return [{"label": str(k), "avg": round(float(v), 1), "count": int(g.size()[k])}
                for k, v in g.mean().items()]
    # корреляции числовых признаков с часами
    corr = []
    for f in norms_model.NUM_FEATURES:
        if df[f].nunique() > 1:
            c = float(np.corrcoef(df[f].astype(float), y)[0, 1])
            if c == c:  # не NaN
                corr.append({"feature": LABELS.get(f, f), "corr": round(c, 2)})
    corr.sort(key=lambda x: -abs(x["corr"]))
    mape_base = float((np.abs(y - base) / y).mean() * 100)
    return {
        "dataset": os.path.basename(path), "n": int(len(df)),
        "hours": {"mean": round(float(y.mean()), 1), "median": round(float(y.median()), 1),
                  "min": round(float(y.min()), 1), "max": round(float(y.max()), 1),
                  "std": round(float(y.std()), 1)},
        "baseline_mape": round(mape_base, 1),
        "hist": {"edges": [round(float(e), 1) for e in edges], "counts": [int(c) for c in counts]},
        "scatter": scatter,
        "by_ip": by_cat("ip_rating"), "by_plc": by_cat("plc_type"),
        "corr": corr[:8],
    }


# ------------------------------------------------------------------ HTML
HTML = r"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Нормирование ШАУ</title>
<style>
:root{--navy:#1F4E79;--blue:#2E75B6;--bg:#f4f6f9;--card:#fff;--line:#dde3ea;--ok:#2e7d32;--warn:#a05a00}
*{box-sizing:border-box}body{font-family:Arial,Helvetica,sans-serif;margin:0;background:var(--bg);color:#1d2733}
header{background:var(--navy);color:#fff;padding:16px 28px}
header h1{margin:0;font-size:19px}
nav{display:flex;gap:6px;background:#163a5c;padding:0 20px}
nav button{background:none;border:0;color:#cfe0f0;padding:13px 18px;font-size:14px;cursor:pointer;border-bottom:3px solid transparent}
nav button.on{color:#fff;border-bottom-color:#7fb0e0;font-weight:bold}
nav button:hover{color:#fff}
.wrap{max-width:1120px;margin:22px auto;padding:0 20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px}
.card h2{margin:0 0 14px;font-size:15px;color:var(--navy)}
#drop{border:2px dashed var(--blue);border-radius:10px;padding:30px;text-align:center;cursor:pointer;color:#456;background:#fafcff}
#drop.drag{background:#eaf2fb;border-color:var(--navy)}#drop b{color:var(--navy)}
.small{font-size:12px;color:#6b7785}
.total{font-size:44px;font-weight:bold;color:var(--navy);line-height:1}
.total span{font-size:17px;color:#6b7785;font-weight:normal}
.badge{display:inline-block;font-size:12px;padding:2px 9px;border-radius:20px;background:#eaf2fb;color:var(--navy);margin-left:6px}
.badge.ok{background:#e6f4ea;color:var(--ok)}.badge.no{background:#fdecea;color:#b3261e}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #eef1f5;text-align:left}
th{color:#6b7785;font-weight:normal;font-size:12px}
input,select{width:92px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;font-size:13px;text-align:right}
select{width:auto;text-align:left}
.bar{height:16px;border-radius:4px;background:var(--blue);min-width:2px}
.row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px}
.row .nm{width:170px;color:#3a4754;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .hh{width:62px;text-align:right;color:#1d2733}
button.act{background:var(--navy);color:#fff;border:0;border-radius:8px;padding:9px 16px;font-size:14px;cursor:pointer}
button.act:hover{background:#163a5c}button.act:disabled{opacity:.5;cursor:wait}
.note{font-size:12px;color:var(--warn);background:#fff6e8;border:1px solid #ffe1b0;border-radius:6px;padding:8px 10px;margin-top:10px}
.stat{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat .b{background:#f7fafd;border:1px solid var(--line);border-radius:8px;padding:14px}
.stat .b .v{font-size:26px;font-weight:bold;color:var(--navy)}
.stat .b .k{font-size:12px;color:#6b7785;margin-top:2px}
.hidden{display:none}.mt{margin-top:18px}
@media(max-width:860px){.grid2{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Нормирование трудоёмкости сборки ШАУ</h1></header>
<nav>
  <button data-t="predict" class="on">Прогноз</button>
  <button data-t="train">Обучение</button>
  <button data-t="analytics">Аналитика</button>
  <button data-t="stats">Статистика</button>
</nav>
<div class="wrap">

<!-- ============ ПРОГНОЗ ============ -->
<section id="t-predict"><div class="grid2">
  <div class="card">
    <h2>1. Спецификация</h2>
    <div id="drop">Перетащите файл сюда или <b>выберите</b><br>
      <span class="small">.pdf · .xlsx · .xls · .csv</span>
      <input id="file" type="file" accept=".pdf,.xlsx,.xls,.csv" class="hidden"></div>
    <div id="srcnote"></div>
    <h2 class="mt">2. Признаки <span class="small">(можно править)</span></h2>
    <table id="feat"><tbody></tbody></table>
    <div class="mt"><button class="act" id="recalc">Пересчитать</button></div>
  </div>
  <div class="card">
    <h2>Прогноз трудоёмкости</h2>
    <div class="total"><span id="ph">— загрузите спецификацию —</span>
      <span id="totwrap" class="hidden"><span id="tot"></span> <span>нормо-часов</span>
      <span id="mlbadge" class="badge hidden"></span></span></div>
    <div id="bars" class="mt"></div>
  </div>
</div></section>

<!-- ============ ОБУЧЕНИЕ ============ -->
<section id="t-train" class="hidden">
  <div class="card">
    <h2>Обучение ML-модели</h2>
    <div id="trainstatus" class="small">Загрузка статуса…</div>
    <div class="mt"><button class="act" id="btnTrain">Обучить модель</button>
      <span id="trainmsg" class="small" style="margin-left:10px"></span></div>
  </div>
  <div id="trainres" class="hidden">
    <div class="grid2 mt">
      <div class="card"><h2>Метрики на валидации</h2>
        <table><thead><tr><th>Модель</th><th>MAE, ч</th><th>MAPE</th><th>R²</th></tr></thead>
        <tbody id="metrows"></tbody></table>
        <div id="anchor" class="mt small"></div>
        <div id="crit" class="mt"></div>
      </div>
      <div class="card"><h2>Важность признаков (топ-10)</h2><div id="imp"></div></div>
    </div>
    <div class="card mt"><h2>Факт vs прогноз (валидация)</h2>
      <div class="small">Чем ближе точки к диагонали, тем точнее. Синий — baseline, зелёный — с ML.</div>
      <div id="trainscatter" class="mt"></div></div>
  </div>
</section>

<!-- ============ АНАЛИТИКА ============ -->
<section id="t-analytics" class="hidden">
  <div class="card"><div id="anote" class="small">Загрузка…</div></div>
  <div class="grid2 mt">
    <div class="card"><h2>Распределение трудоёмкости, ч</h2><div id="hist"></div></div>
    <div class="card"><h2>Факт vs параметрическая оценка</h2><div id="scatter"></div></div>
  </div>
  <div class="grid2 mt">
    <div class="card"><h2>Средняя трудоёмкость по IP</h2><div id="byip"></div></div>
    <div class="card"><h2>Средняя трудоёмкость по типу ПЛК</h2><div id="byplc"></div></div>
  </div>
  <div class="card mt"><h2>Драйверы трудоёмкости (корреляция с часами)</h2><div id="corr"></div></div>
</section>

<!-- ============ СТАТИСТИКА ============ -->
<section id="t-stats" class="hidden">
  <div class="card"><h2>Сводка по датасету</h2>
    <div id="snote" class="small">Загрузка…</div>
    <div id="statcards" class="stat mt"></div>
  </div>
  <div class="grid2 mt">
    <div class="card"><h2>Проекты по степени защиты IP</h2><table id="tip"></table></div>
    <div class="card"><h2>Проекты по типу ПЛК</h2><table id="tplc"></table></div>
  </div>
</section>

</div>
<script>
const NUM=__NUM__, LAB=__LAB__, IP=["IP21","IP54","IP65","IP66"];
let F={};
// ---- вкладки ----
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  ['predict','train','analytics','stats'].forEach(t=>
    document.getElementById('t-'+t).classList.toggle('hidden', t!==b.dataset.t));
  if(b.dataset.t==='train')loadTrainStatus();
  if(b.dataset.t==='analytics')loadAnalytics();
  if(b.dataset.t==='stats')loadStats();
});
// ---- утилиты графиков (чистый SVG) ----
function barRows(el,items,unit){
  const mx=Math.max(...items.map(i=>Math.abs(i.value)),1e-9);
  el.innerHTML=items.map(i=>`<div class="row"><div class="nm" title="${i.label}">${i.label}</div>
    <div class="bar" style="width:${Math.round(Math.abs(i.value)/mx*100)}%;background:${i.color||'#2E75B6'}"></div>
    <div class="hh">${i.disp!=null?i.disp:i.value}${unit||''}</div></div>`).join('');
}
function histSVG(counts,edges){
  const W=440,H=180,pl=34,pb=22,mx=Math.max(...counts,1),n=counts.length;
  const bw=(W-pl-6)/n;let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(let i=0;i<n;i++){const h=(H-pb-8)*counts[i]/mx,x=pl+i*bw+2,yv=H-pb-h;
    s+=`<rect x="${x}" y="${yv}" width="${bw-4}" height="${h}" fill="#2E75B6" rx="2"></rect>`;
    if(counts[i])s+=`<text x="${x+(bw-4)/2}" y="${yv-3}" font-size="9" fill="#456" text-anchor="middle">${counts[i]}</text>`;}
  s+=`<line x1="${pl}" y1="${H-pb}" x2="${W}" y2="${H-pb}" stroke="#ccc"></line>`;
  s+=`<text x="${pl}" y="${H-6}" font-size="10" fill="#789">${edges[0]} ч</text>`;
  s+=`<text x="${W-4}" y="${H-6}" font-size="10" fill="#789" text-anchor="end">${edges[edges.length-1]} ч</text>`;
  return s+'</svg>';
}
function scatterSVG(series){ // series=[{points,color}]
  const all=series.flatMap(s=>s.points);if(!all.length)return'';
  const W=440,H=300,p=34;
  const xs=all.map(d=>d.x),ys=all.map(d=>d.y);
  const lo=Math.min(...xs,...ys),hi=Math.max(...xs,...ys);
  const sx=v=>p+(v-lo)/(hi-lo||1)*(W-p-8),sy=v=>H-p-(v-lo)/(hi-lo||1)*(H-p-8);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  s+=`<line x1="${sx(lo)}" y1="${sy(lo)}" x2="${sx(hi)}" y2="${sy(hi)}" stroke="#bbb" stroke-dasharray="4 3"></line>`;
  series.forEach(se=>se.points.forEach(d=>{
    s+=`<circle cx="${sx(d.x)}" cy="${sy(d.y)}" r="2.6" fill="${se.color}" opacity="0.6"></circle>`;}));
  s+=`<line x1="${p}" y1="${H-p}" x2="${W}" y2="${H-p}" stroke="#ccc"></line>`;
  s+=`<line x1="${p}" y1="0" x2="${p}" y2="${H-p}" stroke="#ccc"></line>`;
  s+=`<text x="${W-4}" y="${H-p+16}" font-size="10" fill="#789" text-anchor="end">оценка, ч →</text>`;
  s+=`<text x="${p-26}" y="14" font-size="10" fill="#789">факт, ч</text>`;
  return s+'</svg>';
}
// ---- ПРОГНОЗ ----
const drop=document.getElementById('drop'),file=document.getElementById('file');
drop.onclick=()=>file.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')};
drop.ondragleave=()=>drop.classList.remove('drag');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');if(e.dataTransfer.files[0])upload(e.dataTransfer.files[0])};
file.onchange=e=>{if(e.target.files[0])upload(e.target.files[0])};
function upload(f){drop.innerHTML='Обработка <b>'+f.name+'</b>…';
  const r=new FileReader();r.onload=()=>{const b64=r.result.split(',')[1];
    fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({filename:f.name,content_b64:b64})}).then(x=>x.json()).then(d=>{
      if(d.error){drop.innerHTML='Ошибка: '+d.error;return;}
      drop.innerHTML='Файл: <b>'+f.name+'</b> — заменить';F=d.features;renderFeat();renderPred(d.prediction);
      document.getElementById('srcnote').innerHTML=d.source_note?'<div class="note">'+d.source_note+'</div>':'';
    }).catch(e=>drop.innerHTML='Ошибка сети: '+e);};r.readAsDataURL(f);}
function renderFeat(){let h='<tr><td>Степень защиты IP</td><td><select id="f_ip">'+
  IP.map(v=>'<option '+(F.ip_rating===v?'selected':'')+'>'+v+'</option>').join('')+'</select></td></tr>';
  NUM.forEach(k=>{h+='<tr><td>'+(LAB[k]||k)+'</td><td><input id="f_'+k+'" type="number" step="any" value="'+(F[k]??0)+'"></td></tr>';});
  document.querySelector('#feat tbody').innerHTML=h;}
function collect(){const o=Object.assign({},F);o.ip_rating=document.getElementById('f_ip').value;
  NUM.forEach(k=>o[k]=parseFloat(document.getElementById('f_'+k).value)||0);return o;}
document.getElementById('recalc').onclick=()=>fetch('/api/recompute',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({features:collect()})})
  .then(x=>x.json()).then(d=>{F=collect();renderPred(d.prediction);});
function renderPred(p){document.getElementById('ph').classList.add('hidden');
  document.getElementById('totwrap').classList.remove('hidden');
  document.getElementById('tot').textContent=(''+p.total).replace('.',',');
  const mb=document.getElementById('mlbadge');
  if(p.ml_delta!=null){mb.classList.remove('hidden');mb.textContent='baseline '+p.baseline+' + ML '+(p.ml_delta>=0?'+':'')+p.ml_delta;}
  else mb.classList.add('hidden');
  const bd=p.breakdown,order=["Мехобработка","Силовой монтаж","Слаботочка","Клеммы+маркировка","ПНР","IP66"];
  barRows(document.getElementById('bars'),order.map(k=>({label:k,value:bd[k]||0,disp:(bd[k]||0).toFixed(1)})),' ч');}
// ---- ОБУЧЕНИЕ ----
function loadTrainStatus(){fetch('/api/status').then(x=>x.json()).then(d=>{
  document.getElementById('trainstatus').innerHTML= d.dataset?
    ('Датасет: <b>'+d.dataset+'</b> — '+d.rows+' проектов. Движок: '+d.ml_engine+
     (d.model_loaded?'. Модель загружена.':'. Модель ещё не обучена.')):
    'Датасет не найден. Сгенерируйте synthetic_dataset.csv или соберите training_data.csv.';
  if(d.last_train)showTrain(d.last_train);});}
document.getElementById('btnTrain').onclick=()=>{const b=document.getElementById('btnTrain');
  b.disabled=true;document.getElementById('trainmsg').textContent='Обучение… (несколько секунд)';
  fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
   .then(x=>x.json()).then(d=>{b.disabled=false;document.getElementById('trainmsg').textContent='Готово.';
     if(d.error){document.getElementById('trainmsg').textContent='Ошибка: '+d.error;return;}showTrain(d);})
   .catch(e=>{b.disabled=false;document.getElementById('trainmsg').textContent='Ошибка сети: '+e;});};
function showTrain(d){document.getElementById('trainres').classList.remove('hidden');
  const mb=d.metrics_baseline,mm=d.metrics_ml;
  document.getElementById('metrows').innerHTML=
    `<tr><td>Параметрическая (Слой A)</td><td>${mb.MAE.toFixed(2)}</td><td>${mb.MAPE.toFixed(1)}%</td><td>${mb.R2.toFixed(3)}</td></tr>`+
    `<tr><td><b>Слой A + ML</b></td><td><b>${mm.MAE.toFixed(2)}</b></td><td><b>${mm.MAPE.toFixed(1)}%</b></td><td><b>${mm.R2.toFixed(3)}</b></td></tr>`;
  const a=d.anchor;document.getElementById('anchor').innerHTML=
    `Движок: <b>${d.engine}</b> · ${d.n_used} проектов (train ${d.n_train}/val ${d.n_val}, выбросов −${d.n_outliers})<br>`+
    `Anchor-тест: baseline ${a.baseline} + ML ${a.delta>=0?'+':''}${a.delta} = <b>${a.pred} ч</b> `+
    `<span class="badge ${a.ok?'ok':'no'}">${a.ok?'в диапазоне 18–22':'вне диапазона'}</span>`;
  document.getElementById('crit').innerHTML='Критерии ТЗ (MAPE≤15%, R²≥0,85): '+
    `<span class="badge ${d.criteria_ok?'ok':'no'}">${d.criteria_ok?'соблюдены':'не соблюдены'}</span>`;
  const imp=Object.entries(d.importances).sort((x,y)=>y[1]-x[1]).slice(0,10)
    .map(([k,v])=>({label:LAB[k]||k,value:v,disp:v.toFixed(3)}));
  barRows(document.getElementById('imp'),imp,'');
  document.getElementById('trainscatter').innerHTML=scatterSVG([
    {points:d.val.baseline.map((b,i)=>({x:b,y:d.val.actual[i]})),color:'#2E75B6'},
    {points:d.val.ml.map((m,i)=>({x:m,y:d.val.actual[i]})),color:'#2e9e54'}]);}
// ---- АНАЛИТИКА ----
function loadAnalytics(){fetch('/api/analytics').then(x=>x.json()).then(d=>{
  if(d.error){document.getElementById('anote').textContent='Ошибка: '+d.error;return;}
  document.getElementById('anote').innerHTML='Датасет: <b>'+d.dataset+'</b> — '+d.n+' проектов.';
  document.getElementById('hist').innerHTML=histSVG(d.hist.counts,d.hist.edges);
  document.getElementById('scatter').innerHTML=scatterSVG([{points:d.scatter,color:'#2E75B6'}]);
  barRows(document.getElementById('byip'),d.by_ip.map(r=>({label:r.label+' ('+r.count+')',value:r.avg,disp:r.avg+' ч'})),'');
  barRows(document.getElementById('byplc'),d.by_plc.map(r=>({label:r.label+' ('+r.count+')',value:r.avg,disp:r.avg+' ч'})),'');
  barRows(document.getElementById('corr'),d.corr.map(r=>({label:r.feature,value:r.corr,
    disp:r.corr,color:r.corr>=0?'#2E75B6':'#c2603a'})),'');});}
// ---- СТАТИСТИКА ----
function loadStats(){fetch('/api/analytics').then(x=>x.json()).then(d=>{
  if(d.error){document.getElementById('snote').textContent='Ошибка: '+d.error;return;}
  document.getElementById('snote').innerHTML='Датасет: <b>'+d.dataset+'</b>';
  const c=[['Проектов',d.n],['Медиана, ч',d.hours.median],['Среднее, ч',d.hours.mean],
    ['Мин, ч',d.hours.min],['Макс, ч',d.hours.max],['Разброс σ, ч',d.hours.std],
    ['MAPE baseline',d.baseline_mape+'%']];
  document.getElementById('statcards').innerHTML=c.map(([k,v])=>
    `<div class="b"><div class="v">${(''+v).replace('.',',')}</div><div class="k">${k}</div></div>`).join('');
  const tbl=(rows)=>'<thead><tr><th>Категория</th><th>Проектов</th><th>Сред. ч</th></tr></thead><tbody>'+
    rows.map(r=>`<tr><td>${r.label}</td><td>${r.count}</td><td>${r.avg}</td></tr>`).join('')+'</tbody>';
  document.getElementById('tip').innerHTML=tbl(d.by_ip);
  document.getElementById('tplc').innerHTML=tbl(d.by_plc);});}
</script></body></html>
"""
HTML = HTML.replace("__NUM__", json.dumps(EDITABLE_NUM)).replace("__LAB__", json.dumps(LABELS, ensure_ascii=False))


# ------------------------------------------------------------------ сервер
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        try:
            route = urlparse(self.path)
            if route.path in ("/", "/index.html"):
                return self._send(200, HTML, "text/html")
            if route.path == "/api/status":
                ds = active_dataset()
                rows = 0
                if ds:
                    try:
                        import pandas as pd
                        rows = int(len(pd.read_csv(ds)))
                    except Exception:
                        rows = 0
                eng = "CatBoost"
                try:
                    import catboost  # noqa
                except Exception:
                    eng = "numpy-fallback"
                return self._send(200, json.dumps({
                    "dataset": os.path.basename(ds) if ds else None, "rows": rows,
                    "ml_engine": eng, "model_loaded": PREDICTOR is not None,
                    "last_train": LAST_TRAIN}, ensure_ascii=False))
            if route.path in ("/api/analytics", "/api/stats"):
                ds = active_dataset()
                if not ds:
                    return self._send(200, json.dumps({"error": "датасет не найден"}, ensure_ascii=False))
                return self._send(200, json.dumps(dataset_report(ds), ensure_ascii=False))
            return self._send(404, json.dumps({"error": "not found"}))
        except BaseException as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, json.dumps({"error": "Некорректный запрос (JSON)"}, ensure_ascii=False))
            if self.path == "/api/predict":
                content = base64.b64decode(req["content_b64"])
                feats, _ = features_from_upload(req["filename"], content)
                note = ""
                if feats.get("_source") == "table" and not feats.get("cable_entries"):
                    note = ("Источник — табличный BOM: число кабельных вводов и аналоговых "
                            "цепей со схемы недоступно. Уточните вручную в признаках слева.")
                return self._send(200, json.dumps(
                    {"features": feats, "prediction": predict(feats), "source_note": note},
                    ensure_ascii=False))
            if self.path == "/api/recompute":
                return self._send(200, json.dumps({"prediction": predict(req["features"])}, ensure_ascii=False))
            if self.path == "/api/train":
                global PREDICTOR, LAST_TRAIN
                ds = req.get("dataset") or active_dataset()
                if not ds or not os.path.exists(ds):
                    return self._send(200, json.dumps({"error": "датасет не найден"}, ensure_ascii=False))
                report, predictor = train_model.run_training(ds)
                PREDICTOR, LAST_TRAIN = predictor, report
                return self._send(200, json.dumps(report, ensure_ascii=False))
            return self._send(404, json.dumps({"error": "not found"}))
        except BaseException as e:
            try:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print("Веб-интерфейс нормирования ШАУ — Прогноз / Обучение / Аналитика / Статистика")
    print(f"Откройте:  http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
