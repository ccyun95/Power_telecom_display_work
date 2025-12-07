# -*- coding: utf-8 -*-
"""
1.py — KRX 수집/업데이트 → per-ticker JSON → 표 + 2개 차트(index.html)
- 표 아래에 Plotly 차트 2개(가격/지표, 수급/공매도)를 2.py 방식을 참고해 추가
- 차트 데이터는 섹션별 inline JSON으로 넣어 CORS 없이 file://에서도 동작
- CSV 값(특히 공매도잔고비중)을 그대로 JSON/차트에 반영하기 위해 컬럼명 trim 처리

CSV 스키마(열 순서와 명칭은 예시이며, 실제 CSV의 열이 그대로 JSON→차트에 사용됩니다):
일자,시가,고가,저가,종가,거래량,등락률,기관 합계,기타법인,개인,외국인 합계,전체,공매도,공매도비중,공매도잔고,공매도잔고비중
"""
import argparse
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz
import time
import pandas as pd
from pykrx import stock

# =========================
# 설정
# =========================
DATA_DIR = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "data"
DOCS_DIR = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "docs"
API_DIR = DOCS_DIR / "api"

OUTPUT_SUFFIX = "_stock_data.csv"
ENCODING = "utf-8-sig"
SLEEP_SEC = 0.3
WINDOW_DAYS_INIT = 370
SORT_DESC = True  # 최신→과거 저장

KST = tz.gettz("Asia/Seoul")

# =========================
# 유틸
# =========================
def kst_now():
    return datetime.now(tz=KST)

def kst_today_date():
    return kst_now().date()

def yyyymmdd(d):
    return d.strftime("%Y%m%d")

def empty_with_cols(cols):
    data = {}
    for c in cols:
        data[c] = pd.Series(dtype="object") if c == "일자" else pd.Series(dtype="float64")
    return pd.DataFrame(data)

def read_company_list(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"기업 리스트 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                name, ticker = [x.strip() for x in line.split(",", 1)]
            else:
                parts = line.replace("\t", " ").split()
                if len(parts) < 2:
                    logging.warning("기업 라인 파싱 불가: %s", line)
                    continue
                name, ticker = parts[0], parts[1]
            rows.append((name, ticker.zfill(6)))
    return rows

def csv_path_for(eng_name: str, _ticker: str) -> Path:
    return DATA_DIR / f"{eng_name}{OUTPUT_SUFFIX}"

def last_trading_day_by_ohlcv(ticker: str, today: datetime.date):
    start = today - timedelta(days=30)
    df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        start = today - timedelta(days=90)
        df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} : 최근 거래 자료가 없습니다.")
    return pd.to_datetime(df.index.max()).date()

def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_with_cols(["일자"])
    df = df.copy()
    if df.index.name is None:
        df.index.name = "일자"
    idx = pd.to_datetime(df.index, errors="coerce")
    df.index = idx
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "일자"}, inplace=True)
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def rename_investor_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "일자" not in df.columns:
        return empty_with_cols(["일자","기관 합계","기타법인","개인","외국인 합계","전체"])
    mapping = {
        "기관합계": "기관 합계",
        "외국인합계": "외국인 합계",
        "전체": "전체", "개인": "개인", "기타법인": "기타법인",
        "기관 합계": "기관 합계", "외국인 합계": "외국인 합계",
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    for need in ["기관 합계","기타법인","개인","외국인 합계","전체"]:
        if need not in df.columns:
            df[need] = 0
    keep = ["일자","기관 합계","기타법인","개인","외국인 합계","전체"]
    return df[keep]

def ensure_all_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 필요한 열이 비어있어도 CSV 스키마를 맞춘 후 숫자 변환
    req_cols = [
        "일자","시가","고가","저가","종가","거래량","등락률",
        "기관 합계","기타법인","개인","외국인 합계","전체",
        "공매도","공매도비중","공매도잔고","공매도잔고비중"
    ]
    for col in req_cols:
        if col not in df.columns:
            df[col] = 0
    return df[req_cols]

# =========================
# 수집/병합
# =========================
def fetch_block(ticker: str, start_d: datetime.date, end_d: datetime.date) -> pd.DataFrame:
    s, e = yyyymmdd(start_d), yyyymmdd(end_d)
    # OHLCV
    ohlcv = stock.get_market_ohlcv(s, e, ticker); df1 = normalize_date_index(ohlcv)
    # 투자자별 거래량
    inv   = stock.get_market_trading_volume_by_date(s, e, ticker); df2 = rename_investor_cols(normalize_date_index(inv))
    # 공매도/잔고 (열 이름은 있는 그대로 보존 → CSV 값 그대로 JSON/차트에 쓰기 위함)
    try: sv = stock.get_shorting_volume_by_date(s, e, ticker)
    except Exception: sv = pd.DataFrame()
    df3 = normalize_date_index(sv)
    try: sb = stock.get_shorting_balance_by_date(s, e, ticker)
    except Exception: sb = pd.DataFrame()
    df4 = normalize_date_index(sb)

    df = df1.merge(df2, on="일자", how="left").merge(df3, on="일자", how="left").merge(df4, on="일자", how="left")
    # 숫자형 강제 변환(없으면 0)
    for c in [c for c in df.columns if c != "일자"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # 스키마 보정
    df = ensure_all_cols(df)
    # 정렬
    df = df.sort_values("일자", ascending=not SORT_DESC)
    return df

def upsert_company(eng_name: str, ticker: str, run_on_holiday: bool):
    out_path = csv_path_for(eng_name, ticker)
    today = kst_today_date()
    end_date = last_trading_day_by_ohlcv(ticker, today)

    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        if base.empty:
            last_have = None
        else:
            base["일자"] = pd.to_datetime(base["일자"], errors="coerce").dt.date
            last_have = base["일자"].max()
        start_date = (last_have + timedelta(days=1)) if last_have else (end_date - timedelta(days=WINDOW_DAYS_INIT))
    else:
        start_date = end_date - timedelta(days=WINDOW_DAYS_INIT)

    if (end_date < today) and (not run_on_holiday) and (not out_path.exists()):
        logging.info("[%s] 휴장일(run_on_holiday=False) → 신규 생성 스킵", eng_name)
        return False
    if start_date > end_date:
        logging.info("[%s] 최신 상태 (추가 데이터 없음).", eng_name); return False

    logging.info("[%s] 수집 구간: %s ~ %s (티커 %s)", eng_name, start_date, end_date, ticker)
    df = fetch_block(ticker, start_date, end_date)

    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        merged = pd.concat([base, df], ignore_index=True)
        merged.drop_duplicates(subset=["일자"], keep="last", inplace=True)
        merged = merged.sort_values("일자", ascending=not SORT_DESC)
        merged.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\n")
        logging.info("[%s] 업데이트 완료 → %s", eng_name, out_path)
    else:
        df = df.sort_values("일자", ascending=not SORT_DESC)
        df.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\n")
        logging.info("[%s] 신규 생성 완료 → %s", eng_name, out_path)
    return True

# =========================
# 산출물(JSON + index.html)
# =========================
def emit_per_ticker_json(companies, rows_limit=None):
    API_DIR.mkdir(parents=True, exist_ok=True)
    cnt = 0
    for name, ticker in companies:
        csv_path = csv_path_for(name, ticker)
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, encoding=ENCODING)
        except Exception:
            df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # ✅ CSV 값을 그대로 쓰되, 컬럼명은 공백/숨은문자 제거
        df.columns = [str(c).strip() for c in df.columns]

        df_use = df.head(int(rows_limit)) if rows_limit else df
        payload = {
            "name": name,
            "ticker": str(ticker).zfill(6),
            "columns": list(df_use.columns),
            "rows": df_use.astype(object).where(pd.notna(df_use), "").values.tolist(),
            "generated_at": kst_now().strftime("%Y-%m-%d %H:%M:%S %Z")
        }
        (API_DIR / f"{name}_{str(ticker).zfill(6)}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        cnt += 1
    logging.info("per-ticker JSON 생성 완료: %d files → %s", cnt, API_DIR)

def emit_index_html(companies, rows_limit=None):
    import html as _html
    from string import Template

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sections = []
    generated = kst_now().strftime("%Y-%m-%d %H:%M:%S %Z")

    for name, ticker in companies:
        csv_path = csv_path_for(name, ticker)
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, encoding=ENCODING)
        except Exception:
            df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if rows_limit:
            df = df.head(int(rows_limit))

        # ✅ 컬럼명 trim
        df.columns = [str(c).strip() for c in df.columns]

        # 표 렌더용
        columns = [str(c) for c in df.columns]
        rows_for_table = df.astype(object).where(pd.notna(df), "").astype(str).values.tolist()

        thead = "".join(f"<th>{_html.escape(c)}</th>" for c in columns)
        tbody = "\n".join(
            "<tr>" + "".join(f"<td>{_html.escape(v)}</td>" for v in row) + "</tr>"
            for row in rows_for_table
        )

        # 섹션별 차트 데이터 (inline JSON)
        payload = {
            "name": name,
            "ticker": str(ticker).zfill(6),
            "columns": columns,
            "rows": rows_for_table,
        }
        json_raw = json.dumps(payload, ensure_ascii=False)
        json_safe = json_raw.replace("</", "<\\/")  # </script> 이스케이프

        sec_id = f"{name}_{str(ticker).zfill(6)}"
        sections.append(f"""
<section id="{_html.escape(sec_id)}">
  <h2>{_html.escape(name)} ({str(ticker).zfill(6)})</h2>

  <div class="scroll">
    <table>
      <thead><tr>{thead}</tr></thead>
      <tbody>
      {tbody}
      </tbody>
    </table>
  </div>
  <p class="meta">rows: {len(rows_for_table)} · source: data/{_html.escape(csv_path.name)} · json: (inline)</p>

  <!-- 차트: 표 아래 세로 스택으로 크게 2개 -->
  <div class="charts">
    <div id="chart-price-{_html.escape(sec_id)}" class="chart"></div>
    <div id="chart-flow-{_html.escape(sec_id)}" class="chart"></div>
  </div>

  <!-- 섹션별 데이터(JSON) -->
  <script id="data-{_html.escape(sec_id)}" type="application/json">{json_safe}</script>
</section>""")

    def _id_from(sec_html: str) -> str:
        try:
            return sec_html.split('id="', 1)[1].split('"', 1)[0]
        except Exception:
            return "section"

    nav = "".join(f'<a href="#{_id_from(s)}">{_id_from(s)}</a>' for s in sections)

    # 2.py의 차트 생성부를 반영한 템플릿
    html_template = Template("""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KRX 기업별 데이터 테이블</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; background-color: #f9fafb; }
  header { margin-bottom: 24px; border-bottom: 1px solid #e5e7eb; padding-bottom: 16px; }
  h1 { margin: 0 0 8px 0; font-size: 24px; color: #111827; }
  .meta-top { color:#6b7280; font-size:14px; }
  .nav { display:flex; flex-wrap:wrap; gap:8px 12px; margin-top:12px; }
  .nav a { font-size:14px; text-decoration:none; color:#2563eb; background: #eff6ff; padding: 4px 8px; border-radius: 4px; }
  .nav a:hover { background: #dbeafe; }

  section { margin: 40px 0; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  h2 { font-size: 20px; margin: 0 0 16px 0; border-left: 4px solid #2563eb; padding-left: 12px; color: #1f2937; }

  .scroll { overflow:auto; max-height: 420px; border:1px solid #e5e7eb; border-radius: 6px; background: #fff; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; }
  th, td { border-bottom: 1px solid #e5e7eb; padding: 8px 12px; text-align: right; }
  th { position: sticky; top:0; background:#f3f4f6; color: #374151; font-weight: 600; border-bottom: 2px solid #e5e7eb; }
  th:first-child, td:first-child { text-align: left; }
  tr:hover td { background-color: #f9fafb; }

  .meta { color:#9ca3af; font-size:12px; margin-top: 8px; text-align: right; }

  .charts { width: 100%; display: flex; flex-direction: column; gap: 24px; margin-top: 16px; }
  .chart { width: 100%; height: 500px; border:1px solid #e5e7eb; border-radius: 6px; background: #fff; }
</style>
</head>
<body>
<header>
  <h1>KRX 기업별 데이터 테이블</h1>
  <div class="meta-top">생성 시각: $generated · 타임존: Asia/Seoul</div>
  <nav class="nav">$nav</nav>
</header>

$sections

<script>
/* ===== 유틸(2.py 기반) ===== */
function SMA(arr,n){const o=Array(arr.length).fill(null);let s=0,q=[];for(let i=0;i<arr.length;i++){const v=+arr[i]||0;q.push(v);s+=v;if(q.length>n)s-=q.shift();if(q.length===n)o[i]=s/n}return o}
function EMA(arr,n){const o=Array(arr.length).fill(null);const k=2/(n+1);let p=null;for(let i=0;i<arr.length;i++){const v=+arr[i]||0;p=(p==null)?v:v*k+p*(1-k);o[i]=p}return o}
function STD(arr,n){const o=Array(arr.length).fill(null);let q=[];for(let i=0;i<arr.length;i++){const v=+arr[i]||0;q.push(v);if(q.length>n)q.shift();if(q.length===n){const m=q.reduce((a,b)=>a+b,0)/n;const s2=q.reduce((a,b)=>a+(b-m)*(b-m),0)/n;o[i]=Math.sqrt(s2)}}return o}
function RSI(close,n=14){const o=Array(close.length).fill(null);let g=0,l=0;for(let i=1;i<close.length;i++){const ch=close[i]-close[i-1],G=ch>0?ch:0,L=ch<0?-ch:0;if(i<=n){g+=G;l+=L;if(i===n){const rs=(g/n)/((l/n)||1e-9);o[i]=100-100/(1+rs)}}else{g=(g*(n-1)+G)/n;l=(l*(n-1)+L)/n;const rs=g/(l||1e-9);o[i]=100-100/(1+rs)}}return o}
function MACD(close,f=12,s=26,sg=9){const ef=EMA(close,f),es=EMA(close,s),m=ef.map((v,i)=>v!=null&&es[i]!=null?v-es[i]:null),signal=EMA(m.map(v=>v??0),sg),h=m.map((v,i)=>v!=null&&signal[i]!=null?v-signal[i]:null);return{macd:m,signal,hist:h}}
function bbBands(close,n=20,k=2){const ma=SMA(close,n),sd=STD(close,n),u=ma.map((m,i)=>m!=null&&sd[i]!=null?m+k*sd[i]:null),l=ma.map((m,i)=>m!=null&&sd[i]!=null?m-k*sd[i]:null);return{ma,upper:u,lower:l}}
function nnum(x){if(x==null)return 0;return +String(x).replace(/,/g,'').replace(/\\s+/g,'').replace(/%/g,'')||0}
const str = (x)=> (x==null ? '' : String(x));
const cumsum = (arr)=>{let s=0; return arr.map(v=>{s += (+v||0); return s;});};
const safeMax = (arr)=> Math.max( ...(arr.map(v=>+v||0).filter(v=>isFinite(v)&&!isNaN(v))), 0 );

function toAsc(date, ...series){
  const N = date.length;
  if (N < 2) return [date, ...series];
  if (date[0] <= date[N-1]) return [date, ...series];
  const rev = a => a.slice().reverse();
  return [rev(date), ...series.map(rev)];
}

function idxOf(cols, primary, alts=[]){
  const i=cols.indexOf(primary);
  if(i>-1) return i;
  for(const a of alts){ const j=cols.indexOf(a); if(j>-1) return j; }
  return -1;
}

function showError(secId,msg){
  for (const side of ['chart-price-','chart-flow-']){
    const el = document.getElementById(side+secId);
    if (el) el.innerHTML = '<div style="padding:12px;color:#b91c1c;font-size:13px">'+msg+'</div>';
  }
}

/* ===== 섹션 렌더 ===== */
function renderOne(secId){
  const tag=document.getElementById('data-'+secId);
  if(!tag){ showError(secId,'섹션 데이터가 없습니다.'); return; }
  let j=null; try{ j=JSON.parse(tag.textContent); }catch(e){ showError(secId,'섹션 데이터 파싱 실패: '+e); return; }

  // CSV→JSON에서 trim했지만 JS에서도 한 번 더 안전 처리
  const cols = (j.columns || []).map(c => String(c).trim());

  const iDate=idxOf(cols,'일자',['\\ufeff일자','DATE','date']),
        iOpen=idxOf(cols,'시가',['Open','open']),
        iHigh=idxOf(cols,'고가',['High','high']),
        iLow =idxOf(cols,'저가',['Low','low']),
        iClose=idxOf(cols,'종가',['Close','close']),
        iVol =idxOf(cols,'거래량',['Volume','volume']),
        iFor =idxOf(cols,'외국인 합계',['외국인합계','외인합계']),
        iInst=idxOf(cols,'기관 합계',['기관합계']),
        iShortR =idxOf(cols,'공매도비중',['공매도 비중','공매도 거래량 비중','비중','(공매도)비중']),
        iShortBR=idxOf(cols,'공매도잔고비중',[
          '공매도 잔고 비중','공매도잔고비중(%)','공매도잔고 비중(%)',
          '잔고비중','잔고 비중','공매도잔고비율','잔고비율'
        ]);

  if([iDate,iOpen,iHigh,iLow,iClose].some(i=>i<0)){ showError(secId,'필수 컬럼 누락'); return; }
  const rows=j.rows||[]; if(!rows.length){ showError(secId,'시계열 행이 없습니다.'); return; }

  let date   = rows.map(r=>str(r[iDate]));
  let open   = rows.map(r=>nnum(r[iOpen]));
  let high   = rows.map(r=>nnum(r[iHigh]));
  let low    = rows.map(r=>nnum(r[iLow]));
  let close  = rows.map(r=>nnum(r[iClose]));
  let vol    = (iVol>=0)? rows.map(r=>nnum(r[iVol])): rows.map(_=>0);
  let foreign= (iFor>=0)? rows.map(r=>nnum(r[iFor])): rows.map(_=>0);
  let inst   = (iInst>=0)? rows.map(r=>nnum(r[iInst])): rows.map(_=>0);
  let shortR = (iShortR>=0)? rows.map(r=>nnum(r[iShortR])): rows.map(_=>0);
  let shortBR= (iShortBR>=0)? rows.map(r=>nnum(r[iShortBR])): rows.map(_=>0);

  // 지표 계산 전 과거→현재 오름차순으로 정렬
  [date, open, high, low, close, vol, foreign, inst, shortR, shortBR] =
    toAsc(date, open, high, low, close, vol, foreign, inst, shortR, shortBR);

  // 보조지표
  const ma20=SMA(close,20), ma60=SMA(close,60), ma120=SMA(close,120);
  const bb=bbBands(close,20,2);
  const rsi=RSI(close,14);
  const {macd,signal,hist}=MACD(close,12,26,9);

  // 누적 수급
  const instCum    = cumsum(inst);
  const foreignCum = cumsum(foreign);

  // -------- 차트 1: 가격/지표 --------
  const layout1={
    grid:{rows:3,columns:1,pattern:'independent',roworder:'top to bottom'},
    xaxis:{domain:[0,1], rangeslider:{visible:false}, showspikes:true, spikemode:'across'},
    yaxis:{domain:[0.35,1.00], title:'주가 (원)', tickformat:',', showspikes:true},
    xaxis2:{anchor:'y2', showspikes:true},
    yaxis2:{domain:[0.18,0.30], title:'RSI', range:[0,100], tickvals:[30,70], showgrid:true},
    xaxis3:{anchor:'y3', showspikes:true},
    yaxis3:{domain:[0.00,0.15], title:'MACD'},
    legend:{orientation:'h', y:1.02, x:0.5, xanchor:'center'},
    margin:{t:40,l:60,r:40,b:30},
    hovermode:'x unified',
    plot_bgcolor:'#ffffff', paper_bgcolor:'#ffffff'
  };

  const traces1=[
    {type:'candlestick',x:date,open,high,low,close,name:'주가',
     increasing:{line:{color:'#ef4444'}}, decreasing:{line:{color:'#3b82f6'}} },
    {type:'scatter',mode:'lines',x:date,y:ma20,name:'MA20', line:{width:1.5}},
    {type:'scatter',mode:'lines',x:date,y:ma60,name:'MA60', line:{width:1.5}},
    {type:'scatter',mode:'lines',x:date,y:ma120,name:'MA120', line:{width:1.5}},
    {type:'scatter',mode:'lines',x:date,y:bb.upper,name:'BB상단', visible:'legendonly', line:{dash:'dot', width:1}},
    {type:'scatter',mode:'lines',x:date,y:bb.lower,name:'BB하단', visible:'legendonly', line:{dash:'dot', width:1}},
    {type:'scatter',mode:'lines',x:date,y:rsi,name:'RSI(14)',xaxis:'x2',yaxis:'y2'},
    {type:'bar',x:date,y:hist,name:'MACD Hist',xaxis:'x3',yaxis:'y3'},
    {type:'scatter',mode:'lines',x:date,y:macd,name:'MACD',xaxis:'x3',yaxis:'y3'},
    {type:'scatter',mode:'lines',x:date,y:signal,name:'Signal',xaxis:'x3',yaxis:'y3'},
  ];

  Plotly.newPlot('chart-price-'+secId, traces1, layout1, {responsive:true, displaylogo:false});

  // -------- 차트 2: 수급/공매도 --------
  const layout2={
    yaxis:{title:'누적 순매수', tickformat:',', showgrid:true},
    yaxis2:{title:'공매도 비율(%)', overlaying:'y', side:'right',
           range:[0, Math.max(1, safeMax(shortBR.concat(shortR))*1.2)]},
    margin:{t:40,l:60,r:50,b:30},
    hovermode:'x unified',
    legend:{orientation:'h', y:1.08, x:0.5, xanchor:'center'},
    plot_bgcolor:'#ffffff'
  };

  const traces2=[
    {type:'scatter',mode:'lines',x:date,y:instCum,   name:'기관 누적'},
    {type:'scatter',mode:'lines',x:date,y:foreignCum,name:'외국인 누적'},
    {type:'scatter',mode:'lines',x:date,y:shortR,    name:'공매도비중(%)',yaxis:'y2', line:{dash:'dot'}},
    {type:'scatter',mode:'lines',x:date,y:shortBR,   name:'공매도잔고비중(%)',yaxis:'y2'},
  ];

  Plotly.newPlot('chart-flow-'+secId, traces2, layout2, {responsive:true, displaylogo:false});
}

(function main(){
  const ids=Array.from(document.querySelectorAll('section[id]')).map(s=>s.id);
  for(const id of ids){ try{ renderOne(id); }catch(e){ showError(id,'렌더링 오류: '+e); } }
})();
</script>

<footer style="margin-top:60px; padding: 20px 0; border-top: 1px solid #e5e7eb; color:#6b7280; font-size:13px; text-align: center;">
  Published via GitHub Pages · Inline JSON rendering (CORS-free)
</footer>
</body>
</html>""")

    html_doc = html_template.substitute(
        generated=generated,
        nav=nav,
        sections="".join(sections),
    )

    (DOCS_DIR / "index.html").write_text(html_doc, encoding="utf-8")
    logging.info("index.html 생성 완료 → %s", DOCS_DIR / "index.html")

# =========================
# 엔트리포인트
# =========================
def main():
    parser = argparse.ArgumentParser(description="KRX 수집 & CSV 업데이트 & 웹문서 생성")
    parser.add_argument("--company-list", default=str(DATA_DIR / "company_list.txt"))
    parser.add_argument("--run-on-holiday", default="true",
                        help="휴장일에도 실행(전 영업일 데이터 사용) (true/false)")
    parser.add_argument("--rows-limit", default=None,
                        help="index.html/JSON에 포함할 최대 행 수(최신 상위). 기본: 전체")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_on_holiday = str(args.run_on_holiday).lower() in ("1","true","yes","y")
    rows_limit = int(args.rows_limit) if args.rows_limit else None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    API_DIR.mkdir(parents=True, exist_ok=True)

    try:
        companies = read_company_list(Path(args.company_list))
    except Exception as e:
        logging.exception("기업 리스트 로딩 실패: %s", e)
        return

    if not companies:
        logging.warning("수집 대상 기업이 없습니다.")
        return

    changed = False
    for name, ticker in companies:
        try:
            time.sleep(SLEEP_SEC)
            updated = upsert_company(name, ticker, run_on_holiday)
            changed = changed or updated
        except Exception as e:
            logging.exception("[%s,%s] 처리 중 오류: %s", name, ticker, e)

    # 기업별 JSON + index.html 생성
    emit_per_ticker_json(companies, rows_limit=rows_limit)
    emit_index_html(companies, rows_limit=rows_limit)

    if changed:
        logging.info("변경사항 존재 → 커밋 단계에서 반영됩니다.")
    else:
        logging.info("변경사항 없음.")

if __name__ == "__main__":
    main()
