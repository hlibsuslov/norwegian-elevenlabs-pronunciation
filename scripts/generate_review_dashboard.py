"""Generate a self-contained report and local review controls."""
import csv
import io
import json
from pathlib import Path

from .common import atomic_bytes, write_json

LABELS = {
    "click_or_pop":"Click or pop", "clipping":"Clipping", "dropout":"Dropout", "crackle":"Crackle",
    "possible_phonetic_interference":"Possible phonetic interference", "interword_pause":"Pause",
    "possible_stress_mismatch":"Possible stress difference", "unclear_speech":"Unclear speech",
    "possible_stretched_word":"Possible stretched word", "transient_outlier":"Short transient",
    "voice_spectral_anomaly":"Spectral change",
}


def timestamp(value, milliseconds=False):
    count = max(0,round(value*1000))
    sec, ms = divmod(count,1000)
    h, sec = divmod(sec,3600)
    m, sec = divmod(sec,60)
    clock=f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{clock},{ms:03d}" if milliseconds else clock


def contents(report):
    events=report["events"]
    names=("id","start","end","category","severity","status","timing","word","sentence","expected_language",
           "detected_language","pronunciation_language","confidence","recommended_repair")
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=names,extrasaction="ignore")
    writer.writeheader()
    # Text from the recording may begin with spreadsheet formula triggers.
    for row in events:
        writer.writerow({field:"'"+value if isinstance(value,str) and value[:1] in ("=","+","-","@","\t","\r")
                         else value for field,value in row.items() if field in names})
    subtitles=[]
    for i,row in enumerate(events,1):
        name=LABELS.get(row["category"], row["category"].replace("_"," "))
        subtitles.append(f"{i}\n{timestamp(row['start'],True)} --> {timestamp(max(row['end'],row['start']+.18),True)}\n[{name}] {row['status']}\n")
    return buffer.getvalue(), "\n".join(subtitles)


def dashboard(report, base=""):
    payload={
        "run":report["source"]["sha256"][:16], "filename":Path(report["source"]["path"]).name,
        "duration":report["duration_s"], "events":[{
            key:value for key,value in row.items() if key not in {"audio_file","clip_end"}
        } for row in report["events"]],
        "labels":LABELS, "base":base, "qa":report.get("limitations",[]),
    }
    embedded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    return HTML.replace("DATA_SLOT",embedded)


def export_report(output, report):
    output=Path(output)
    reports=output/"reports"
    write_json(reports/"defects.json",report)
    csv_text, srt=contents(report)
    atomic_bytes(reports/"defects.csv",csv_text.encode("utf-8-sig"))
    atomic_bytes(reports/"review_markers.srt",srt.encode())
    atomic_bytes(output/"review/index.html",dashboard(report).encode())
    atomic_bytes(reports/"quality_report.html",dashboard(report,"../review/").encode())


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio review</title><style>
:root{color-scheme:light;--ink:#152c2a;--muted:#46615e;--line:#cedbd6;--paper:#f7f8f3;--accent:#125a54}
*{box-sizing:border-box} [hidden]{display:none!important} body{margin:0;color:var(--ink);background:var(--paper);font:16px/1.5 "Segoe UI",sans-serif}
::selection{background:#b6ddd1;color:var(--ink)} :focus-visible{outline:2px solid #247c73;outline-offset:3px}
main{max-width:1320px;margin:auto;padding:32px clamp(18px,4vw,64px) 48px} h1,h2,h3,p{margin:0}
h1{font-size:clamp(32px,5vw,54px);line-height:1.03;letter-spacing:-.04em;font-weight:650} h2{font-size:26px;line-height:1.2}
header p{color:var(--muted);margin-top:10px;overflow-wrap:anywhere} header{padding:0 0 23px;border-bottom:1px solid var(--line)}
.controls{display:flex;flex-wrap:wrap;align-items:end;gap:13px;margin:18px 0 22px} label{display:grid;gap:3px;font-size:.88em;min-width:0}
select{max-width:100%}
button,select,input{font:inherit;color:var(--ink);background:#fff;border:1px solid #9fb5ae;border-radius:6px;padding:7px 11px}
button{cursor:pointer;text-align:left} button:hover{background:#eef4ec} .primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.primary:hover{background:#184742} button:disabled{color:var(--muted);cursor:default} .workspace{display:grid;grid-template-columns:minmax(240px,0.85fr) minmax(0,1.5fr);gap:clamp(18px,4vw,62px)}
nav{border-top:1px solid var(--line);max-height:70vh;overflow:auto;scrollbar-color:#72968d transparent} .row{display:grid;width:100%;padding:11px 8px;border:0;border-bottom:1px solid var(--line);border-radius:0;background:none;grid-template-columns:91px minmax(0,1fr);gap:8px;overflow-wrap:anywhere}
.row[aria-current=true]{background:#dcece4;font-weight:650} .time,.details time{font-variant-numeric:tabular-nums;color:var(--muted)}
.details{min-width:0;overflow-wrap:anywhere} .details>p{margin-top:10px;max-width:68ch} .media{display:grid;gap:6px;margin:15px 0}
audio{width:100%;min-width:0;height:44px} .charts{display:grid;gap:7px;margin:14px 0} img{width:100%;height:auto}
.actions{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0} .notice{color:var(--muted);margin-top:15px;max-width:72ch}
footer{margin-top:26px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px} a{color:var(--accent);text-underline-offset:3px}
@media(max-width:720px){main{padding-top:22px}.workspace{grid-template-columns:minmax(0,1fr)}nav{max-height:240px}.controls{align-items:stretch}.row{grid-template-columns:82px minmax(0,1fr)}}
</style></head><body><main><header><h1>Audio review</h1><p id="source"></p></header>
<section class="controls" aria-label="Filters"><label>Category<select id="category"></select></label>
<label>Review status<select id="status"><option value="">All statuses</option><option>Unreviewed</option><option>CONFIRMED</option><option>FALSE_POSITIVE</option><option>REGENERATE</option></select></label>
<label>Show routine pauses<select id="routine"><option value="hide">Hide</option><option value="show">Show</option></select></label>
<button id="export" type="button">Export decisions</button><span id="count" aria-live="polite"></span></section>
<div class="workspace"><nav id="events" aria-label="Candidate events"></nav><article class="details" id="details">
<h2 id="title">Select an event</h2><p id="caption"></p><p id="context"></p><p id="transcript"></p>
<div class="media" id="players"></div><div class="actions" id="actions" hidden>
<button class="primary" id="confirm" type="button">Confirm defect</button><button id="reject" type="button">Dismiss</button>
<button id="queue" type="button">Queue sentence repair</button><button id="clear" type="button">Clear decision</button></div>
<p id="message" aria-live="polite"></p><div class="charts" id="charts"></div><a id="evidence" hidden>Evidence JSON</a>
</article></div><footer id="foot"></footer></main>
<script type="application/json" id="data">DATA_SLOT</script><script>
const data=JSON.parse(document.querySelector('#data').textContent);
const at=id=>document.getElementById(id);
const stateKey='nelp-review:'+data.run;
let decisions={};
try{decisions=JSON.parse(localStorage.getItem(stateKey)||'{}')}catch{}
const patterns=new Map(), categories=new Map();
let active=null;
function clock(s){return new Date(Math.max(0,s)*1000).toISOString().slice(11,23)}
function name(category){return data.labels[category]||category.replaceAll('_',' ')}
function option(label,value){const node=document.createElement('option');node.textContent=label;node.value=value;return node}
function choose(row){
  const focused=document.activeElement;
  active=row; at('title').textContent=name(row.category);at('caption').textContent=clock(row.start)+'–'+clock(row.end)+' · '+row.status+' · '+row.timing.replaceAll('_',' ');
  at('context').textContent=row.sentence?(row.sentence_origin==='reverse_transcription'?'Reverse transcript context: ':'Manuscript: ')+row.sentence:'';
  at('transcript').textContent=row.asr_excerpt?'Heard by ASR: '+row.asr_excerpt:'';
  at('message').textContent=decisions[row.id]?.status ? 'Decision saved · '+decisions[row.id].status : '';
  at('actions').hidden=false;at('players').replaceChildren();at('charts').replaceChildren();
  const assets=row.assets||{};
  for(const [field,label] of [['context','Context'],['defect','Event'],['repaired','Repaired candidate']]){
    if(!assets[field])continue;
    const group=document.createElement('label');group.append(label);
    const audio=document.createElement('audio');audio.controls=true;audio.preload='metadata';audio.src=data.base+assets[field];
    group.append(audio);at('players').append(group)
  }
  for(const [field,label] of [['waveform','Signal waveform'],['spectrogram','Audio spectrogram']]){
    if(!assets[field])continue;
    const image=document.createElement('img');image.loading='lazy';image.alt=label;image.src=data.base+assets[field];at('charts').append(image)
  }
  at('evidence').hidden=!assets.evidence;
  if(assets.evidence)at('evidence').href=data.base+assets.evidence;
  at('queue').disabled=!row.repair_context;
  for(const [id,button] of patterns) button.setAttribute('aria-current',String(id===row.id));
  if(focused&&focused.closest('#events'))focused.focus()
}
function display(){
  const picked=at('category').value, filter=at('status').value;
  const rows=data.events.filter(row=>(at('routine').value==='show'||picked||row.severity!=='INFO')&&(!picked||row.category===picked)&&(!filter||
      (decisions[row.id]?.status||'Unreviewed')===filter));
  at('count').textContent=rows.length+' / '+data.events.length+' events';
  at('events').replaceChildren();patterns.clear();
  for(const row of rows){
    const button=document.createElement('button');button.type='button';button.className='row';
    const time=document.createElement('span');time.className='time';time.textContent=clock(row.start).slice(0,8);
    const label=document.createElement('span');label.textContent=name(row.category)+(row.word?' · '+row.word:'');
    button.append(time,label);button.onclick=()=>choose(row);patterns.set(row.id,button);at('events').append(button)
  }
  if(rows.length){choose(rows.find(row=>row.id===active?.id)||rows[0]);return}
  active=null;at('title').textContent='No matching events';for(const id of ['caption','context','transcript','message','players','charts'])at(id).replaceChildren();
  at('actions').hidden=true;at('evidence').hidden=true
}
function review(status){
  if(!active)return;
  if(status)decisions[active.id]={id:active.id,status,reviewed_at:new Date().toISOString()};else delete decisions[active.id];
  try{localStorage.setItem(stateKey,JSON.stringify(decisions))}catch{}
  if(status==='REGENERATE'&&location.protocol.startsWith('http')&&window.audioqaToken){
    fetch('/api/regenerate',{method:'POST',headers:{'Content-Type':'application/json','X-AudioQA-Token':window.audioqaToken},body:JSON.stringify(decisions[active.id])})
      .then(async response=>{const result=await response.json();if(!response.ok)throw Error(result.error);active.assets.repaired=result.audio;choose(active);at('message').textContent=result.message})
      .catch(error=>{at('message').textContent=error.message})
  }
  display()
}
at('source').textContent=data.filename+' · '+clock(data.duration).slice(0,8);
at('foot').textContent=data.qa.join(' ');
at('category').append(option('All categories',''));
for(const row of data.events)categories.set(row.category,(categories.get(row.category)||0)+1);
for(const [category,count] of [...categories].sort())at('category').append(option(name(category)+' ('+count+')',category));
for(const filter of ['category','status','routine'])at(filter).onchange=display;
for(const [id,status] of [['confirm','CONFIRMED'],['reject','FALSE_POSITIVE'],['queue','REGENERATE'],['clear','']])at(id).onclick=()=>review(status);
at('export').onclick=()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(Object.values(decisions),null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='review-decisions.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),500)
};display();
</script></body></html>"""
