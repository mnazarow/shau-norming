#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локальный веб-интерфейс нормирования ШАУ.

Инженер загружает спецификацию (PDF / EPLAN-Excel / CSV) — система извлекает признаки
и выдаёт прогноз трудоёмкости с разбивкой по этапам. Признаки можно поправить вручную
и пересчитать.

Движок прогноза:
  * параметрическая модель (norms_model) — работает всегда;
  * если рядом есть обученная ML-модель (model_shau.cbm, CatBoost) — добавляется
    ML-поправка к baseline. Иначе используется только параметрическая оценка.

Запуск:
    python3 app.py            # затем открыть http://127.0.0.1:8000
    python3 app.py --port 8080

Зависимости: стандартная библиотека + parse_spec/norms_model (+ pdfplumber/openpyxl
для соответствующих форматов). Flask НЕ требуется.
"""
import os
import sys
import json
import base64
import tempfile
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import parse_spec
import norms_model

# Необязательная ML-поправка (если установлен CatBoost и есть сохранённая модель)
_CB = None
if os.path.exists("model_shau.cbm"):
    try:
        from catboost import CatBoostRegressor
        _CB = CatBoostRegressor(); _CB.load_model("model_shau.cbm")
    except Exception:
        _CB = None

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


def predict(features):
    """Возвращает разбивку по этапам, итог и (опц.) ML-поправку."""
    bd = norms_model.estimate_hours(features, breakdown=True)
    baseline = bd["ИТОГО"]
    ml_delta = None
    total = baseline
    if _CB is not None:
        try:
            import pandas as pd
            row = pd.DataFrame([{k: features.get(k) for k in norms_model.ALL_FEATURES}])
            ml_delta = float(_CB.predict(row)[0])
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
        feats, rows = parse_spec.build_features(tmp)
    finally:
        os.unlink(tmp)
    # нормализуем булевы в 0/1 для UI
    for k in ("mount_panel", "din_inclined"):
        feats[k] = int(bool(feats.get(k)))
    feats.setdefault("interfaces_count",
                     sum(1 for v in feats.get("interfaces", {}).values() if v))
    return feats, rows


HTML = r"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Нормирование ШАУ</title>
<style>
:root{--navy:#1F4E79;--blue:#2E75B6;--bg:#f4f6f9;--card:#fff;--line:#dde3ea}
*{box-sizing:border-box}
body{font-family:Arial,Helvetica,sans-serif;margin:0;background:var(--bg);color:#1d2733}
header{background:var(--navy);color:#fff;padding:18px 28px}
header h1{margin:0;font-size:20px}
header p{margin:4px 0 0;font-size:13px;opacity:.85}
.wrap{max-width:1080px;margin:24px auto;padding:0 20px;display:grid;
  grid-template-columns:1fr 1fr;gap:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px}
.card h2{margin:0 0 14px;font-size:15px;color:var(--navy)}
#drop{border:2px dashed var(--blue);border-radius:10px;padding:34px;text-align:center;
  cursor:pointer;color:#456;transition:.15s;background:#fafcff}
#drop.drag{background:#eaf2fb;border-color:var(--navy)}
#drop b{color:var(--navy)}
.small{font-size:12px;color:#6b7785}
.total{font-size:46px;font-weight:bold;color:var(--navy);line-height:1}
.total span{font-size:18px;color:#6b7785;font-weight:normal}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:20px;
  background:#eaf2fb;color:var(--navy);margin-left:6px}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:5px 6px;border-bottom:1px solid #eef1f5}
td.lab{color:#3a4754}
input,select{width:92px;padding:4px 6px;border:1px solid var(--line);border-radius:6px;
  font-size:13px;text-align:right}
select{width:auto;text-align:left}
.bar{height:18px;border-radius:4px;background:var(--blue);min-width:2px}
.row{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:13px}
.row .nm{width:150px;color:#3a4754}.row .hh{width:54px;text-align:right;color:#1d2733}
button{background:var(--navy);color:#fff;border:0;border-radius:8px;padding:9px 16px;
  font-size:14px;cursor:pointer}button:hover{background:#163a5c}
.note{font-size:12px;color:#a05a00;background:#fff6e8;border:1px solid #ffe1b0;
  border-radius:6px;padding:8px 10px;margin-top:10px}
.hidden{display:none}
@media(max-width:840px){.wrap{grid-template-columns:1fr}}
</style></head>
<body>
<header><h1>Нормирование трудоёмкости сборки ШАУ</h1>
<p>Загрузка спецификации (PDF / EPLAN-Excel / CSV) → прогноз в нормо-часах</p></header>
<div class="wrap">
  <div class="card">
    <h2>1. Спецификация</h2>
    <div id="drop">Перетащите файл сюда или <b>выберите</b><br>
      <span class="small">.pdf · .xlsx · .xls · .csv</span>
      <input id="file" type="file" accept=".pdf,.xlsx,.xls,.csv" class="hidden"></div>
    <div id="srcnote"></div>
    <h2 style="margin-top:20px">2. Признаки <span class="small">(можно править)</span></h2>
    <table id="feat"><tbody></tbody></table>
    <div style="margin-top:14px"><button id="recalc">Пересчитать</button></div>
  </div>
  <div class="card">
    <h2>Прогноз трудоёмкости</h2>
    <div class="total"><span id="placeholder">— загрузите спецификацию —</span>
      <span id="totwrap" class="hidden"><span id="tot"></span> <span>нормо-часов</span>
      <span id="mlbadge" class="badge hidden"></span></span></div>
    <div id="bars" style="margin-top:18px"></div>
    <div id="warn"></div>
  </div>
</div>
<script>
const IP=["IP21","IP54","IP65","IP66"];
let F={};
const drop=document.getElementById('drop'),file=document.getElementById('file');
drop.onclick=()=>file.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')};
drop.ondragleave=()=>drop.classList.remove('drag');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');if(e.dataTransfer.files[0])upload(e.dataTransfer.files[0])};
file.onchange=e=>{if(e.target.files[0])upload(e.target.files[0])};

function upload(f){
  drop.innerHTML='Обработка <b>'+f.name+'</b>…';
  const r=new FileReader();
  r.onload=()=>{
    const b64=r.result.split(',')[1];
    fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({filename:f.name,content_b64:b64})})
      .then(x=>x.json()).then(d=>{
        if(d.error){drop.innerHTML='Ошибка: '+d.error;return;}
        drop.innerHTML='Файл: <b>'+f.name+'</b> — заменить';
        F=d.features;renderFeat();renderPred(d.prediction);
        document.getElementById('srcnote').innerHTML = d.source_note?
          '<div class="note">'+d.source_note+'</div>':'';
      }).catch(e=>drop.innerHTML='Ошибка сети: '+e);
  };
  r.readAsDataURL(f);
}
const NUM=__NUM__, LAB=__LAB__;
function renderFeat(){
  let h='<tr><td class="lab">Степень защиты IP</td><td><select id="f_ip">'+
    IP.map(v=>'<option '+(F.ip_rating===v?'selected':'')+'>'+v+'</option>').join('')+
    '</select></td></tr>';
  NUM.forEach(k=>{h+='<tr><td class="lab">'+(LAB[k]||k)+'</td><td>'+
    '<input id="f_'+k+'" type="number" step="any" value="'+(F[k]??0)+'"></td></tr>';});
  document.querySelector('#feat tbody').innerHTML=h;
}
function collect(){
  const o=Object.assign({},F);o.ip_rating=document.getElementById('f_ip').value;
  NUM.forEach(k=>{o[k]=parseFloat(document.getElementById('f_'+k).value)||0;});
  return o;
}
document.getElementById('recalc').onclick=()=>{
  fetch('/api/recompute',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({features:collect()})})
    .then(x=>x.json()).then(d=>{F=collect();renderPred(d.prediction);});
};
function renderPred(p){
  document.getElementById('placeholder').classList.add('hidden');
  document.getElementById('totwrap').classList.remove('hidden');
  document.getElementById('tot').textContent=p.total.toString().replace('.',',');
  const mb=document.getElementById('mlbadge');
  if(p.ml_delta!==null){mb.classList.remove('hidden');
    mb.textContent='baseline '+p.baseline+' + ML '+(p.ml_delta>=0?'+':'')+p.ml_delta;}
  else mb.classList.add('hidden');
  const bd=p.breakdown,order=["Мехобработка","Силовой монтаж","Слаботочка","Клеммы+маркировка","ПНР","IP66"];
  const mx=Math.max(...order.map(k=>bd[k]||0));
  document.getElementById('bars').innerHTML=order.map(k=>{
    const v=bd[k]||0,w=mx?Math.round(v/mx*100):0;
    return '<div class="row"><div class="nm">'+k+'</div>'+
      '<div class="bar" style="width:'+w+'%"></div>'+
      '<div class="hh">'+v.toFixed(1)+' ч</div></div>';}).join('');
}
</script></body></html>
"""
HTML = HTML.replace("__NUM__", json.dumps(EDITABLE_NUM)).replace("__LAB__", json.dumps(LABELS, ensure_ascii=False))


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
        if self.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, json.dumps({"error": "bad json"}))
        try:
            if self.path == "/api/predict":
                content = base64.b64decode(req["content_b64"])
                feats, _ = features_from_upload(req["filename"], content)
                note = ""
                if feats.get("_source") == "table" and not feats.get("cable_entries"):
                    note = ("Источник — табличный BOM: число кабельных вводов и аналоговых "
                            "цепей со схемы недоступно. Уточните вручную в признаках слева.")
                out = {"features": feats, "prediction": predict(feats), "source_note": note}
                self._send(200, json.dumps(out, ensure_ascii=False))
            elif self.path == "/api/recompute":
                self._send(200, json.dumps({"prediction": predict(req["features"])},
                                           ensure_ascii=False))
            else:
                self._send(404, json.dumps({"error": "not found"}))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    eng = "параметрическая + ML (CatBoost)" if _CB else "параметрическая модель"
    print(f"Веб-интерфейс нормирования ШАУ. Движок: {eng}")
    print(f"Откройте:  http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
