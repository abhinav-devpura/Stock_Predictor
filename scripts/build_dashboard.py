"""Render the picks snapshot into a self-contained HTML dashboard.

Reads frontend/picks.json and writes:
  - frontend/dashboard.html      full standalone page (opens via file://, no server)
  - frontend/dashboard.body.html body-only fragment (for publishing as an Artifact)

Both embed the JSON inline, so they are fully self-contained (no external
requests) and render identically. Regenerate after `scripts/build_picks.py`.

Run: python scripts/build_dashboard.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FE = ROOT / "frontend"
DATA = json.loads((FE / "picks.json").read_text())

# ---- CSS (token-driven, both themes) ---------------------------------------
STYLE = """
:root{
  --bg:#f4f4f2; --surface-1:#fcfcfb; --card:#ffffff; --border:#e4e3de;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#86857f;
  --s-combined:#2a78d6; --s-value:#1baf7a; --s-text:#eda100;
  --accent:#2a78d6; --pos:#0f8a4f; --neg:#c53434; --grid:#ecebe6;
  --chip:#f0efeb; --bar:#e9e8e3;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#131312; --surface-1:#1a1a19; --card:#201f1e; --border:#33322f;
  --ink:#f7f7f4; --ink-2:#c3c2b7; --ink-3:#8f8e84;
  --s-combined:#3987e5; --s-value:#199e70; --s-text:#c98500;
  --accent:#3987e5; --pos:#3fbd77; --neg:#e66767; --grid:#2a2926;
  --chip:#2a2926; --bar:#2f2e2b;
}}
:root[data-theme="dark"]{
  --bg:#131312; --surface-1:#1a1a19; --card:#201f1e; --border:#33322f;
  --ink:#f7f7f4; --ink-2:#c3c2b7; --ink-3:#8f8e84;
  --s-combined:#3987e5; --s-value:#199e70; --s-text:#c98500;
  --accent:#3987e5; --pos:#3fbd77; --neg:#e66767; --grid:#2a2926;
  --chip:#2a2926; --bar:#2f2e2b;
}
:root[data-theme="light"]{
  --bg:#f4f4f2; --surface-1:#fcfcfb; --card:#ffffff; --border:#e4e3de;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#86857f;
  --s-combined:#2a78d6; --s-value:#1baf7a; --s-text:#eda100;
  --accent:#2a78d6; --pos:#0f8a4f; --neg:#c53434; --grid:#ecebe6;
  --chip:#f0efeb; --bar:#e9e8e3;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.5;-webkit-font-smoothing:antialiased;}
.num{font-variant-numeric:tabular-nums;
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 72px;}
header.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;}
h1{font-size:30px;font-weight:680;letter-spacing:-.02em;margin:0;text-wrap:balance;}
.sub{color:var(--ink-2);font-size:14px;margin-top:6px;}
.sub b{color:var(--ink);}
.eyebrow{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600;margin:44px 0 14px;}
.toggle{border:1px solid var(--border);background:var(--card);color:var(--ink-2);
  border-radius:8px;padding:7px 12px;font-size:13px;cursor:pointer;}
.toggle:hover{color:var(--ink);}
/* edge cards */
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 18px 16px;}
.card.lead{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset;}
.card .tag{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600;color:var(--ink-2);}
.dot{width:10px;height:10px;border-radius:3px;flex:none;}
.big{font-size:32px;font-weight:700;letter-spacing:-.02em;margin:10px 0 2px;}
.big.pos{color:var(--pos);} .big.neg{color:var(--neg);}
.gross{font-size:12px;color:var(--ink-3);}
.mini{display:flex;gap:18px;margin-top:14px;}
.mini div{font-size:12px;color:var(--ink-3);}
.mini b{display:block;font-size:16px;color:var(--ink);font-weight:640;margin-top:1px;}
.note{color:var(--ink-2);font-size:13.5px;margin-top:14px;max-width:70ch;}
/* chart */
.panel{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:6px;}
.legend span{display:inline-flex;align-items:center;gap:7px;font-size:13px;color:var(--ink-2);}
.chartbox{position:relative;width:100%;}
svg{display:block;width:100%;height:auto;overflow:visible;}
.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--border);
  border-radius:9px;padding:9px 11px;font-size:12px;box-shadow:0 6px 20px rgba(0,0,0,.14);
  opacity:0;transition:opacity .08s;min-width:150px;z-index:5;}
.tip .d{color:var(--ink-3);margin-bottom:5px;}
.tip .r{display:flex;justify-content:space-between;gap:14px;}
.tip .r b{color:var(--ink);}
/* table */
.tbl{width:100%;border-collapse:collapse;font-size:13.5px;}
.tbl thead th{text-align:right;font-weight:600;color:var(--ink-3);font-size:11.5px;
  letter-spacing:.05em;text-transform:uppercase;padding:0 10px 10px;border-bottom:1px solid var(--border);}
.tbl thead th.l{text-align:left;}
.tbl td{padding:10px;border-bottom:1px solid var(--grid);text-align:right;}
.tbl td.l{text-align:left;}
.tbl tr:hover td{background:var(--chip);}
.rank{color:var(--ink-3);width:34px;}
.tkr{font-weight:680;letter-spacing:-.01em;}
.chip{display:inline-block;background:var(--chip);color:var(--ink-2);border-radius:999px;
  padding:2px 9px;font-size:11.5px;white-space:nowrap;}
.scorecell{display:flex;align-items:center;justify-content:flex-end;gap:9px;}
.sbar{height:7px;border-radius:4px;background:var(--s-combined);}
.z{color:var(--ink-2);}
.foot{color:var(--ink-2);font-size:13.5px;margin-top:12px;}
.method{color:var(--ink-2);font-size:14px;max-width:78ch;}
.method ul{padding-left:18px;} .method li{margin:6px 0;}
.disc{color:var(--ink-3);font-size:12px;margin-top:26px;border-top:1px solid var(--border);padding-top:16px;}
/* nav */
.nav{display:flex;gap:6px;margin-bottom:26px;border-bottom:1px solid var(--border);}
.nav a{padding:9px 14px;font-size:14px;font-weight:600;color:var(--ink-3);text-decoration:none;
  border-bottom:2px solid transparent;margin-bottom:-1px;}
.nav a:hover{color:var(--ink);}
.nav a.on{color:var(--ink);border-bottom-color:var(--accent);}
/* prose / narrative page */
.kicker{font-size:12px;letter-spacing:.11em;text-transform:uppercase;color:var(--accent);font-weight:700;}
.lede{font-size:18px;line-height:1.55;color:var(--ink-2);max-width:60ch;margin-top:12px;text-wrap:pretty;}
.prose h2{font-size:22px;font-weight:680;letter-spacing:-.01em;margin:52px 0 8px;text-wrap:balance;}
.prose p{color:var(--ink-2);font-size:15px;line-height:1.62;max-width:68ch;}
.prose code{font-family:ui-monospace,Menlo,monospace;font-size:.86em;background:var(--chip);
  padding:1px 5px;border-radius:5px;color:var(--ink);}
.prose blockquote{margin:24px 0;padding:14px 20px;border-left:3px solid var(--accent);
  background:var(--card);border-radius:0 10px 10px 0;font-size:16px;line-height:1.5;color:var(--ink);max-width:64ch;}
.timeline{list-style:none;padding:0;margin:26px 0 0;position:relative;}
.timeline:before{content:"";position:absolute;left:15px;top:6px;bottom:10px;width:2px;background:var(--border);}
.timeline li{position:relative;padding:0 0 26px 52px;}
.timeline .node{position:absolute;left:6px;top:2px;width:20px;height:20px;border-radius:50%;
  background:var(--card);border:2px solid var(--accent);color:var(--accent);font-size:11px;font-weight:700;
  display:flex;align-items:center;justify-content:center;}
.timeline li.win .node{background:var(--accent);color:#fff;}
.timeline .ph{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);font-weight:600;}
.timeline h3{font-size:16.5px;font-weight:660;margin:2px 0 5px;}
.timeline p{margin:0;font-size:14.5px;}
.tbl2{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:12px;}
.tbl2 th{text-align:left;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600;padding:0 12px 9px;border-bottom:1px solid var(--border);}
.tbl2 td{padding:10px 12px;border-bottom:1px solid var(--grid);color:var(--ink-2);vertical-align:top;}
.tbl2 td b{color:var(--ink);}
.tag-pass{color:var(--pos);font-weight:600;} .tag-fail{color:var(--neg);font-weight:600;}
.takeaways{list-style:none;padding:0;margin:14px 0 0;display:grid;gap:10px;}
.takeaways li{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:13px 15px;
  font-size:14px;color:var(--ink-2);} .takeaways li b{color:var(--ink);}
@media (max-width:720px){.cards{grid-template-columns:1fr}.big{font-size:28px}
  .scroll{overflow-x:auto} h1{font-size:25px} .lede{font-size:16px}}
"""

BODY = """
<div class="wrap">
  __NAV__
  <header class="top">
    <div>
      <h1>Daily Best Picks</h1>
      <div class="sub" id="meta"></div>
    </div>
    <button class="toggle" id="themeBtn" type="button">Toggle theme</button>
  </header>

  <div class="eyebrow">The two edges, measured beta-neutral (long-short)</div>
  <div class="cards" id="cards"></div>
  <p class="note" id="corrNote"></p>

  <div class="eyebrow">Backtest — growth of $1 in each long-short book</div>
  <div class="panel">
    <div class="legend" id="legend"></div>
    <div class="chartbox" id="chartbox">
      <svg id="chart" viewBox="0 0 960 380" preserveAspectRatio="none" role="img"
           aria-label="Cumulative long-short equity curves"></svg>
      <div class="tip" id="tip"></div>
    </div>
    <p class="foot">Non-overlapping rebalances every ~<span id="hz"></span> trading days, top decile
      minus bottom decile. This is the market-neutral spread, not a total-return strategy.</p>
  </div>

  <div class="eyebrow">Where the edge comes from — the three legs</div>
  <div class="panel">
    <div class="legend" id="legend2"></div>
    <div class="chartbox" id="chartbox2">
      <svg id="chart2" viewBox="0 0 960 340" preserveAspectRatio="none" role="img"
           aria-label="Top decile, bottom decile and market cumulative returns"></svg>
      <div class="tip" id="tip2"></div>
    </div>
    <p class="foot">Growth of $1 held <b>long</b> in each book. The long-short curve above is exactly
      <b>top minus bottom</b>. The strategy <b>buys the top decile and shorts the bottom decile</b>,
      so the gap between the green and red lines — over and above the market (grey) — is the edge.</p>
  </div>

  <div class="eyebrow">Ranked picks — top 20 by combined score</div>
  <div class="panel scroll">
    <table class="tbl">
      <thead><tr>
        <th class="l">#</th><th class="l">Ticker</th><th class="l">Sector</th>
        <th>Score</th><th>Value z</th><th>Text z</th><th>Book/Mkt</th><th>Filing sim.</th>
      </tr></thead>
      <tbody id="picks"></tbody>
    </table>
    <p class="foot">Higher <b>Value z</b> = cheaper vs. its sector (higher book/market). Higher
      <b>Text z</b> = more stable 10-K/10-Q text vs. a year ago (stable filings outperform, per
      the "Lazy Prices" text-change signal).</p>
  </div>

  <div class="eyebrow">Bottom decile — the <span style="color:var(--neg)">shorted</span> names</div>
  <div class="panel scroll">
    <table class="tbl">
      <thead><tr>
        <th class="l">#</th><th class="l">Ticker</th><th class="l">Sector</th>
        <th>Score</th><th>Value z</th><th>Text z</th><th>Close</th>
      </tr></thead>
      <tbody id="shorts"></tbody>
    </table>
    <p class="foot">The lowest-scoring <span id="dn"></span> names on the as-of date — what the
      long-short would <b>sell short</b>. Mostly big filing-rewriters (very negative Text z) or
      expensive names (negative Value z). A short profits when these <b>underperform</b>.</p>
  </div>

  <div class="eyebrow">Methodology &amp; caveats</div>
  <div class="method">
    <p id="method"></p>
    <ul id="caveats"></ul>
  </div>
  <p class="disc">Research prototype for educational use — not investment advice.</p>
</div>

<script id="picksData" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('picksData').textContent);
const S = {combined:'--s-combined', value:'--s-value', text:'--s-text'};
const LBL = {combined:'Combined score', value:'Value (book/market)', text:'Text-change (Lazy Prices)'};
const cv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (x,d=1) => (x>=0?'+':'') + (100*x).toFixed(d) + '%';

// meta
const gen = new Date(D.generated_at).toISOString().slice(0,16).replace('T',' ');
document.getElementById('meta').innerHTML =
  `As of <b>${D.as_of_date}</b> · ${D.scored_names} scored names of ${D.universe_size} · `
  + `${D.horizon_days}-day horizon · built ${gen} UTC`;
document.getElementById('hz').textContent = D.horizon_days;

// edge cards
const order = ['combined','value','text'];
document.getElementById('cards').innerHTML = order.map(k=>{
  const e = D.edges[k]; if(!e) return '';
  const cls = e.ann_return_net>=0?'pos':'neg';
  return `<div class="card ${k==='combined'?'lead':''}">
    <div class="tag"><span class="dot" style="background:var(${S[k]})"></span>${LBL[k]}</div>
    <div class="big ${cls}">${pct(e.ann_return_net)}</div>
    <div class="gross">ann. return, net 10bps · gross ${pct(e.ann_return)}</div>
    <div class="mini">
      <div>Sharpe<b class="num">${e.sharpe.toFixed(2)}</b></div>
      <div>Hit rate<b class="num">${Math.round(e.hit_rate*100)}%</b></div>
      <div>Max DD<b class="num">${Math.round(e.max_drawdown*100)}%</b></div>
    </div></div>`;
}).join('');

if(D.value_text_correlation!=null){
  document.getElementById('corrNote').innerHTML =
    `Value vs. text-change long-short correlation <b>${(D.value_text_correlation>=0?'+':'')
    + D.value_text_correlation.toFixed(2)}</b> — near zero, so they diversify. The text edge is the `
    + `stronger of the two; combining trades a little return for a shallower drawdown.`;
}

// ---- reusable line chart (used for both the long-short books and the legs) ----
function makeChart(boxId, svgId, tipId, seriesArr){
  const box=document.getElementById(boxId), svgEl=document.getElementById(svgId), tip=document.getElementById(tipId);
  const W=960,H=340,P={t:16,r:118,b:26,l:44};
  const n=seriesArr[0].pts.length;
  const dates=seriesArr[0].pts.map(p=>p.date);
  let vmin=Infinity,vmax=-Infinity;
  seriesArr.forEach(s=>s.pts.forEach(p=>{vmin=Math.min(vmin,p.v);vmax=Math.max(vmax,p.v)}));
  const pad=(vmax-vmin)*0.06||0.1; vmin-=pad; vmax+=pad;
  const X=i=>P.l+(W-P.l-P.r)*(i/(n-1));
  const Y=v=>P.t+(H-P.t-P.b)*(1-(v-vmin)/(vmax-vmin));
  function draw(){
    const g=cv('--grid'), ink3=cv('--ink-3');
    let svg='';
    const ticks=4;
    for(let i=0;i<=ticks;i++){
      const v=vmin+(vmax-vmin)*i/ticks, y=Y(v);
      svg+=`<line x1="${P.l}" y1="${y}" x2="${W-P.r}" y2="${y}" stroke="${g}" stroke-width="1"/>`;
      svg+=`<text x="${P.l-8}" y="${y+4}" text-anchor="end" font-size="11" fill="${ink3}">$${v.toFixed(1)}</text>`;
    }
    if(1>=vmin&&1<=vmax){svg+=`<line x1="${P.l}" y1="${Y(1)}" x2="${W-P.r}" y2="${Y(1)}" stroke="${ink3}" stroke-width="1" stroke-dasharray="3 3" opacity=".5"/>`;}
    [0,Math.floor(n/2),n-1].forEach(i=>{
      svg+=`<text x="${X(i)}" y="${H-6}" text-anchor="middle" font-size="11" fill="${ink3}">${dates[i].slice(0,7)}</text>`;
    });
    // avoid overlapping end labels: nudge each to its slot in vertical order
    const ends=seriesArr.map(s=>({s, y:Y(s.pts[s.pts.length-1].v)}))
      .sort((a,b)=>a.y-b.y);
    let prevY=-1e9;
    ends.forEach(e=>{ e.ly=Math.max(e.y, prevY+14); prevY=e.ly; });
    seriesArr.forEach(s=>{
      const col=cv(s.cvar);
      const d=s.pts.map((p,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(p.v).toFixed(1)}`).join('');
      const dash=s.dash?` stroke-dasharray="6 4"`:'';
      svg+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"${dash}/>`;
      const last=s.pts[s.pts.length-1];
      svg+=`<circle cx="${X(n-1)}" cy="${Y(last.v)}" r="3.5" fill="${col}"/>`;
    });
    ends.forEach(e=>{
      const col=cv(e.s.cvar), last=e.s.pts[e.s.pts.length-1];
      svg+=`<text x="${X(n-1)+7}" y="${e.ly+4}" font-size="12" font-weight="600" fill="${col}">${e.s.short} $${last.v.toFixed(2)}</text>`;
    });
    svg+=`<line class="cross" x1="0" y1="${P.t}" x2="0" y2="${H-P.b}" stroke="${ink3}" stroke-width="1" opacity="0"/>`;
    svgEl.innerHTML=svg;
  }
  draw();
  box.addEventListener('pointermove', ev=>{
    const rect=svgEl.getBoundingClientRect();
    const px=(ev.clientX-rect.left)/rect.width*W;
    let i=Math.round(((px-P.l)/(W-P.l-P.r))*(n-1));
    i=Math.max(0,Math.min(n-1,i));
    const cross=svgEl.querySelector('.cross');
    if(cross){cross.setAttribute('x1',X(i));cross.setAttribute('x2',X(i));cross.setAttribute('opacity','.5');}
    const rows=seriesArr.map(s=>`<div class="r"><span style="color:${cv(s.cvar)}">${s.short}</span> <b class="num">$${s.pts[i].v.toFixed(2)}</b></div>`).join('');
    tip.innerHTML=`<div class="d">${dates[i]}</div>${rows}`;
    tip.style.opacity='1';
    const leftPx=X(i)/W*rect.width;
    tip.style.left=Math.min(rect.width-165,Math.max(0,leftPx+12))+'px';
    tip.style.top='8px';
  });
  box.addEventListener('pointerleave', ()=>{tip.style.opacity='0';
    const c=svgEl.querySelector('.cross'); if(c)c.setAttribute('opacity','0');});
  return draw;
}

// chart 1 — the three long-short books
const series1=order.map(k=>({cvar:S[k], short:LBL[k].split(' ')[0],
  pts:(D.curves[k]||[]).map(p=>({date:p.date,v:p.equity}))})).filter(s=>s.pts.length>1);
document.getElementById('legend').innerHTML=series1.map(s=>
  `<span><span class="dot" style="background:var(${s.cvar})"></span>${s.short}</span>`).join('');
const draw1=makeChart('chartbox','chart','tip', series1);

// chart 2 — the three legs the edge decomposes into
const DL=D.decile_curves||{};
const legDefs=[
  {k:'top', cvar:'--pos', short:'Top decile', label:'Top decile (bought)'},
  {k:'bottom', cvar:'--neg', short:'Bottom decile', label:'Bottom decile (shorted)', dash:true},
  {k:'universe', cvar:'--ink-3', short:'Market', label:'Market (universe avg)'},
];
const series2=legDefs.map(d=>({cvar:d.cvar, short:d.short, dash:d.dash,
  pts:(DL[d.k]||[]).map(p=>({date:p.date,v:p.equity}))})).filter(s=>s.pts.length>1);
let draw2=()=>{};
if(series2.length){
  document.getElementById('legend2').innerHTML=legDefs.map(d=>
    `<span><span class="dot" style="background:var(${d.cvar})"></span>${d.label}</span>`).join('');
  draw2=makeChart('chartbox2','chart2','tip2', series2);
}

// ---- top-20 table ---- (note: `top` is a reserved window global — do not shadow it)
const zf=v=>v==null?'—':(v>=0?'+':'')+v.toFixed(2);
const topPicks=D.picks.slice(0,20);
const maxScore=Math.max(...topPicks.map(p=>p.score||0), 0.01);
document.getElementById('picks').innerHTML=topPicks.map(p=>{
  const bw=Math.max(2, (Math.max(0,p.score)/maxScore)*70);
  return `<tr>
    <td class="l rank num">${p.rank}</td>
    <td class="l tkr">${p.ticker}</td>
    <td class="l">${p.sector?`<span class="chip">${p.sector}</span>`:'—'}</td>
    <td><div class="scorecell"><span class="num">${p.score.toFixed(2)}</span>
      <span class="sbar" style="width:${bw}px"></span></div></td>
    <td class="z num">${zf(p.value_z)}</td>
    <td class="z num">${zf(p.text_z)}</td>
    <td class="num">${p.book_to_market==null?'—':p.book_to_market.toFixed(2)}</td>
    <td class="num">${p.section_similarity==null?'—':p.section_similarity.toFixed(3)}</td>
  </tr>`;
}).join('');

// ---- bottom-decile (shorted) table ----
document.getElementById('dn').textContent=D.decile_size || (D.bottom_picks||[]).length;
document.getElementById('shorts').innerHTML=(D.bottom_picks||[]).map(p=>`<tr>
    <td class="l rank num">${p.rank}</td>
    <td class="l tkr">${p.ticker}</td>
    <td class="l">${p.sector?`<span class="chip">${p.sector}</span>`:'—'}</td>
    <td class="num">${p.score.toFixed(2)}</td>
    <td class="z num">${zf(p.value_z)}</td>
    <td class="z num">${zf(p.text_z)}</td>
    <td class="num">${p.close==null?'—':'$'+p.close.toFixed(2)}</td>
  </tr>`).join('');

// ---- methodology ----
document.getElementById('method').textContent=D.methodology;
document.getElementById('caveats').innerHTML=D.caveats.map(c=>`<li>${c}</li>`).join('');

// ---- theme toggle ----
function redraw(){draw1();draw2();}
document.getElementById('themeBtn').addEventListener('click',()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next = cur==='dark'?'light' : cur==='light'?'dark'
    : (matchMedia('(prefers-color-scheme:dark)').matches?'light':'dark');
  document.documentElement.setAttribute('data-theme',next);
  redraw();
});
matchMedia('(prefers-color-scheme:dark)').addEventListener?.('change',redraw);
</script>
"""

PROCESS = """
<div class="wrap prose">
  __NAV__
  <header class="top">
    <div>
      <div class="kicker">Behind the build</div>
      <h1>How it was built</h1>
      <p class="lede">This started as a naive Reddit sentiment bot for a single stock. It ended as a
      cross-sectional quant pipeline with <b>two</b> edges that survived a rigorous validation gate —
      and <b>five</b> factor families that didn't. The rejections are the point.</p>
    </div>
    <button class="toggle" id="themeBtn" type="button">Toggle theme</button>
  </header>

  <ol class="timeline">
    <li>
      <div class="node">0</div>
      <div class="ph">Where it started</div>
      <h3>A single-stock sentiment bot</h3>
      <p>v0 (<code>predict_stock.py</code>) scraped Reddit and scored headlines with TextBlob to
      predict one ticker. It ran — but had no way to tell whether the signal beat chance. The honest
      verdict was <em>not valuable</em>, and the question it raised — <em>how would I even know if this
      works?</em> — reframed the entire project.</p>
    </li>
    <li>
      <div class="node">1</div>
      <div class="ph">The foundation</div>
      <h3>Build the referee before the players</h3>
      <p>Before adding a single factor, I built an <b>event-study validation gate</b>: rank the
      universe, compare the top vs. bottom bucket's forward returns (Welch <em>t</em>-test), and
      require the effect to hold in <b>both</b> halves of the sample with the sign fixed <em>a
      priori</em>. Leakage was designed out — point-in-time as-of alignment on filing dates, a
      forward-return embargo, and non-overlapping sampling. The harness is self-tested on synthetic
      data (8/8) so I trust its verdicts.</p>
    </li>
    <li>
      <div class="node">2</div>
      <div class="ph">The graveyard</div>
      <h3>Five factor families, zero survivors</h3>
      <p>Calendar effects, momentum, post-earnings drift (PEAD), short-term reversal, and insider
      Form 4 buying — one by one, each was tested and <b>rejected</b>. The pattern was almost
      universal: strong in the early sample, faded or reversed in the late sample. Five unrelated
      signals decaying at the same split isn't coincidence — it was pointing at a regime shift, not
      alpha. The pattern <em>became</em> the finding.</p>
    </li>
    <li>
      <div class="node">3</div>
      <div class="ph">The turning point</div>
      <h3>Was it real decay, or just too little data?</h3>
      <p>I expanded the universe 3.4× (161 → ~565 names). If the fades were a power problem, <em>t</em>-stats
      should grow — and they did. But raw momentum and reversal <b>still flipped</b> out of sample even
      with the extra power, confirming genuine regime dependence rather than noise. Crucially, the
      expansion also pushed two <em>different</em> signals across the line for the first time.</p>
    </li>
    <li>
      <div class="node">4</div>
      <div class="ph">The trap I avoided</div>
      <h3>The beta mirage</h3>
      <p>An early top-decile portfolio looked like it minted money. It wasn't alpha — it was market
      <em>beta</em> in disguise. I re-measured every edge as a <b>beta-neutral long-short</b> (long the
      top decile, short the bottom), which cancels the market out and leaves only the true spread.
      That decomposition is the single most important guardrail in the project.</p>
    </li>
    <li class="win">
      <div class="node">✓</div>
      <div class="ph">Edge #1</div>
      <h3>Value — the first real signal</h3>
      <p>Sector-neutral <b>book-to-market</b> is the only factor ever to pass the <em>strict</em> gate
      (both halves individually significant). Its beta-neutral long-short earns <b>+8.9%/yr</b> net of
      costs (Sharpe 0.75) and — unlike everything before it — <em>strengthens</em> late instead of
      fading. Honest caveat, documented prominently: value is the factor <b>most inflated</b> by my
      universe's survivorship bias, so I treat the magnitude with heavy skepticism.</p>
    </li>
    <li class="win">
      <div class="node">✓</div>
      <div class="ph">Edge #2</div>
      <h3>Text-change — "Lazy Prices"</h3>
      <p>The idea: firms that heavily rewrite their 10-K/10-Q filings tend to underperform. On 40
      tickers it failed — but that was <em>underpowered</em>, so I scaled to 150 and it <b>passed</b>
      (<em>t</em>=+2.45): section-level filing similarity, beta-neutral long-short <b>+13.9%/yr</b> net,
      Sharpe 0.85, stronger late. Here survivorship works <em>in my favor</em> — the short side is
      big-rewriters, and the ones that went bankrupt delisted, so the premium is understated.</p>
    </li>
    <li>
      <div class="node">5</div>
      <div class="ph">Today</div>
      <h3>The screener &amp; dashboard</h3>
      <p>The two edges are genuinely uncorrelated (−0.13), so I combined them into one cross-sectional
      score — carefully weighting <b>walk-forward</b> (using only past data) to avoid the look-ahead
      trap that a naive in-sample fit would fall into. The result is the ranked board on the
      <a href="__DASH__">Dashboard</a>.</p>
    </li>
  </ol>

  <h2>The factor graveyard</h2>
  <p>Every factor had to clear the same gate. Most didn't — and documenting <em>why</em> is more
  honest than quietly dropping them.</p>
  <table class="tbl2">
    <thead><tr><th>Factor family</th><th>The idea</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr><td><b>Calendar effects</b></td><td>Turn-of-month / January / September seasonality</td><td><span class="tag-fail">rejected</span> — unstable, or significant in the wrong direction</td></tr>
      <tr><td><b>Momentum (12–1)</b></td><td>Recent winners keep winning</td><td><span class="tag-fail">rejected</span> — ≈0, flips across the regime split</td></tr>
      <tr><td><b>PEAD</b></td><td>Drift after earnings surprises</td><td><span class="tag-fail">rejected</span> — early pass, faded late; vanishes as a long-short</td></tr>
      <tr><td><b>Short-term reversal</b></td><td>5-day losers bounce back</td><td><span class="tag-fail">rejected</span> — huge early (<em>t</em>=7.4), gone late</td></tr>
      <tr><td><b>Insider buying (Form 4)</b></td><td>Insiders buy before good news</td><td><span class="tag-fail">rejected</span> — early-strong / late-fade</td></tr>
    </tbody>
  </table>

  <h2>The two that survived</h2>
  <table class="tbl2">
    <thead><tr><th>Edge</th><th>Signal</th><th>Gate</th><th>Beta-neutral LS</th><th>Survivorship</th></tr></thead>
    <tbody>
      <tr><td><b>Value</b></td><td>Cheap vs. sector (book/market)</td><td><span class="tag-pass">strict PASS</span></td><td><b>+8.9%/yr</b>, Sharpe 0.75</td><td>inflates it → read with caution</td></tr>
      <tr><td><b>Text-change</b></td><td>Stable 10-K/10-Q section text</td><td><span class="tag-pass">PASS</span></td><td><b>+13.9%/yr</b>, Sharpe 0.85</td><td>understates it → conservative</td></tr>
    </tbody>
  </table>

  <blockquote>Arriving at a mostly-null result — <em>without</em> shipping a market-beta portfolio
  dressed up as alpha — was the real accomplishment. The value delivered is the rigorous, reusable
  pipeline and the discipline that caught every trap along the way.</blockquote>

  <h2>What this project demonstrates</h2>
  <ul class="takeaways">
    <li><b>Statistical rigor.</b> An event-study gate, stability across time splits, non-overlapping
    sampling, and walk-forward weighting — applied uniformly and self-tested.</li>
    <li><b>Leakage discipline.</b> Point-in-time as-of alignment on SEC filing dates and a
    forward-return embargo, so no future information leaks into a signal.</li>
    <li><b>Data engineering.</b> A pipeline over SEC EDGAR XBRL fundamentals and full-text 10-K/10-Q
    filings, with rate-limited concurrent fetching and caching.</li>
    <li><b>Intellectual honesty.</b> Rejecting factors that didn't hold up, catching a beta mirage,
    and documenting survivorship bias in <em>both</em> directions rather than choosing the flattering
    read.</li>
  </ul>
  <p class="disc">Research prototype for educational use — not investment advice. Full methodology and
  every phase's results live in <code>NOTES.md</code> in the repository.</p>
</div>

<script>
document.getElementById('themeBtn').addEventListener('click',()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next = cur==='dark'?'light':cur==='light'?'dark':(matchMedia('(prefers-color-scheme:dark)').matches?'light':'dark');
  document.documentElement.setAttribute('data-theme',next);
});
</script>
"""


def _nav(dash_href, active):
    """Tab bar linking the dashboard and the narrative page. `dash_href` differs
    by output dir (index.html under docs/, dashboard.html under frontend/)."""
    d = ' class="on"' if active == "dash" else ""
    p = ' class="on"' if active == "proc" else ""
    return (f'<nav class="nav"><a href="{dash_href}"{d}>Dashboard</a>'
            f'<a href="process.html"{p}>How it was built</a></nav>')


def _page(body, title):
    inner = f"<style>{STYLE}</style>\n{body}"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title}</title>\n</head>\n<body>\n{inner}\n</body>\n</html>\n"
    )


def build():
    dash = BODY.replace("__DATA__", json.dumps(DATA))
    T_DASH, T_PROC = "Daily Best Picks", "How it was built — Stock Predictor"
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    # docs/ = GitHub Pages (dashboard is index.html); frontend/ = local file:// use
    # (dashboard is dashboard.html). Each gets a nav wired to its own filenames.
    for out, dash_href in [(docs, "index.html"), (FE, "dashboard.html")]:
        dash_name = "index.html" if out is docs else "dashboard.html"
        d_body = dash.replace("__NAV__", _nav(dash_href, "dash"))
        p_body = PROCESS.replace("__NAV__", _nav(dash_href, "proc")).replace("__DASH__", dash_href)
        (out / dash_name).write_text(_page(d_body, T_DASH))
        (out / "process.html").write_text(_page(p_body, T_PROC))

    # Artifact fragment: single self-contained page, no nav (links can't resolve
    # in the one-page Artifact host). Include a <title> so the tab is named.
    frag = dash.replace("__NAV__", "")
    (FE / "dashboard.body.html").write_text(
        f"<title>{T_DASH}</title>\n<style>{STYLE}</style>\n{frag}")

    print(f"Wrote dashboard + process pages to {docs}/ and {FE}/ "
          f"(as_of {DATA['as_of_date']}, {len(DATA['picks'])} picks)")


if __name__ == "__main__":
    build()
