# m2_reducer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import math
import re
import pandas as pd

# ---------- 데이터 모델 ----------
@dataclass
class PriceSummary:
    start_date: str
    end_date: str
    start_close: float
    end_close: float
    pct_change: float          # %
    max_drawdown: float        # % (음수)
    vol_annualized: Optional[float]  # % 연환산 변동성 (일수준 표준편차 * sqrt(252))
    avg_volume: Optional[float]

@dataclass
class NewsItem:
    title: str
    url: Optional[str]
    published_at: Optional[str]
    score: float

@dataclass
class FinancialKPI:
    period: str         # e.g., "2024Q4" or "2024"
    revenue: Optional[float]
    operating_income: Optional[float]
    net_income: Optional[float]
    eps: Optional[float]
    yoy_revenue: Optional[float]      # %
    yoy_net_income: Optional[float]   # %

@dataclass
class EssentialPack:
    price: Optional[PriceSummary]
    news: List[NewsItem]
    financials: List[FinancialKPI]
    pdf_summary: Optional[str] = None  

    def to_markdown(self) -> str:
        lines = []
        
        # 가격 요약
        if self.price:
            p = self.price
            lines.append("### 가격 요약")
            lines.append(
                f"- 기간: {p.start_date} → {p.end_date}\n"
                f"- 종가: {p.start_close:,.2f} → {p.end_close:,.2f} ({p.pct_change:+.2f}%)\n"
                f"- 최대낙폭(MDD): {p.max_drawdown:.2f}%\n"
                f"- 연환산 변동성(추정): {p.vol_annualized:.2f}%\n"
                f"- 평균 거래량: {p.avg_volume:,.0f}" if p.avg_volume is not None else ""
            )

        # 뉴스
        if self.news:
            lines.append("\n### 핵심 뉴스 Top")
            for i, n in enumerate(self.news, 1):
                when = f" ({n.published_at})" if n.published_at else ""
                url = f" — {n.url}" if n.url else ""
                lines.append(f"{i}. {n.title}{when}{url}")

        # 재무
        if self.financials:
            lines.append("\n### 핵심 재무")
            for f in self.financials[:4]:
                yr = f"- {f.period}: 매출 {num(f.revenue)}, 영업이익 {num(f.operating_income)}, 순이익 {num(f.net_income)}, EPS {num(f.eps)}"
                yoy = []
                if f.yoy_revenue is not None:
                    yoy.append(f"매출 YoY {f.yoy_revenue:+.1f}%")
                if f.yoy_net_income is not None:
                    yoy.append(f"순이익 YoY {f.yoy_net_income:+.1f}%")
                tail = (" (" + ", ".join(yoy) + ")") if yoy else ""
                lines.append(yr + tail)

        # ✅ PDF 요약
        if self.pdf_summary:
            lines.append("\n### PDF 요약")
            lines.append(self.pdf_summary)

        return "\n".join([s for s in lines if s])



def num(x: Optional[float]) -> str:
    return f"{x:,.0f}" if isinstance(x, (int, float)) and not math.isnan(x) else "-"


# --- 시그널 계산 유틸 ---
def compute_price_signals(summary: PriceSummary | None) -> dict:
    sig = {}
    if not summary:
        sig["price_empty"] = True
        return sig
    # 간단한 규칙 예시 (원하면 숫자 조정 가능)
    sig["price_uptrend"] = summary.pct_change > 0
    sig["mdd_gt_10"] = (summary.max_drawdown is not None) and (summary.max_drawdown < -10)
    if summary.vol_annualized is not None:
        sig["high_vol_gt_40"] = summary.vol_annualized > 40
    return sig

def compute_fin_signals(financials: list[FinancialKPI]) -> dict:
    sig = {}
    if not financials:
        sig["fin_empty"] = True
        return sig
    # 최신 항목 기준으로 YoY > 0 이면 플래그
    last = financials[-1]
    sig["yoy_rev_pos"] = (last.yoy_revenue is not None) and (last.yoy_revenue > 0)
    sig["yoy_net_income_pos"] = (last.yoy_net_income is not None) and (last.yoy_net_income > 0)
    # EPS 정보 있음 플래그
    sig["has_eps"] = (last.eps is not None)
    return sig

def compute_news_signals(news: list[NewsItem]) -> dict:
    sig = {}
    if not news:
        sig["news_empty"] = True
        return sig
    # 간단: Top1 점수 임계치
    sig["news_confident"] = (news[0].score >= 1.2)  # 키워드 매치 + 최신성 합산 기준
    return sig


# ---------- 파서들 ----------
def parse_prices(raw: Any) -> pd.DataFrame:
    """
    MCP의 get_historical_stock_prices 결과를 DataFrame으로 정규화.
    - 예상 구조: dict 안에 'data' 또는 'prices' 키 아래 리스트(각 항목: date, open, high, low, close, volume)
    - 혹은 바로 list/dict로 오는 경우도 커버
    """
    if raw is None:
        return pd.DataFrame()
    if isinstance(raw, dict):
        rows = raw.get("data") or raw.get("prices") or raw.get("result") or raw.get("items") or raw.get("rows")
        if rows is None and all(k in raw for k in ["date","close"]):  # 단일 row
            rows = [raw]
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = None
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 열 이름 표준화
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    for col in ["date","open","high","low","close","volume"]:
        if col not in df.columns:
            if col == "volume" and "vol" in df.columns:
                df["volume"] = df["vol"]
            else:
                # 누락 컬럼은 NaN으로
                df[col] = None
    # 날짜 파싱
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    except Exception:
        pass
    # 숫자 변환
    for c in ["open","high","low","close","volume"]:
        with pd.option_context('mode.chained_assignment', None):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df

def compute_price_summary(df: pd.DataFrame) -> Optional[PriceSummary]:
    if df.empty or df["close"].dropna().empty:
        return None
    start_close = float(df["close"].dropna().iloc[0])
    end_close = float(df["close"].dropna().iloc[-1])
    pct_change = (end_close / start_close - 1.0) * 100.0

    # MDD 계산
    closes = df["close"].ffill().values
    peak = -1e18
    mdd = 0.0
    for c in closes:
        if c > peak: peak = c
        draw = (c / peak - 1.0) * 100.0
        if draw < mdd: mdd = draw

    # 변동성(연환산) 추정
    df["ret"] = df["close"].pct_change()
    vol_daily = float(df["ret"].std()) if df["ret"].notna().any() else None
    vol_annual = vol_daily * math.sqrt(252) * 100.0 if vol_daily is not None else None

    # 평균 거래량
    avg_vol = float(df["volume"].mean()) if "volume" in df and df["volume"].notna().any() else None

    start_date = _fmt_date(df["date"].iloc[0])
    end_date   = _fmt_date(df["date"].iloc[-1])
    return PriceSummary(
        start_date=start_date, end_date=end_date,
        start_close=start_close, end_close=end_close,
        pct_change=pct_change, max_drawdown=mdd,
        vol_annualized=vol_annual, avg_volume=avg_vol
    )

def _fmt_date(x: Any) -> str:
    try:
        if isinstance(x, pd.Timestamp):
            return x.strftime("%Y-%m-%d")
        if isinstance(x, datetime):
            return x.strftime("%Y-%m-%d")
        return str(x)[:10]
    except Exception:
        return str(x)

def parse_news(raw: Any, query: str, top_k: int = 3) -> List[NewsItem]:
    """
    MCP get_yahoo_finance_news 예상 구조:
    - dict 안에 'data' 또는 'news' 또는 'items' 리스트: 각 항목 {title, link/url, published_at/date}
    간단한 스코어링: 키워드 일치(제목), 최신 가중.
    """
    # 후보 추출
    items = []
    if isinstance(raw, dict):
        items = raw.get("data") or raw.get("news") or raw.get("items") or []
    elif isinstance(raw, list):
        items = raw
    if not items:
        return []

    # 키워드 추출(질문에서 영문/숫자/한글 토큰)
    tokens = _extract_keywords(query)
    now = datetime.utcnow()

    scored: List[NewsItem] = []
    for it in items:
        title = it.get("title") if isinstance(it, dict) else str(it)
        if not title: continue
        url = it.get("url") or it.get("link")
        ts_raw = it.get("published_at") or it.get("date") or it.get("published") or ""
        when = None
        try:
            when = pd.to_datetime(ts_raw, utc=True).tz_localize(None).strftime("%Y-%m-%d")
            # recency 점수 (최근일수^-0.5)
            age_days = max((datetime.utcnow() - pd.to_datetime(ts_raw, utc=True).to_pydatetime()).days, 0)
            rec = 1.0 / max(math.sqrt(age_days + 1), 1.0)
        except Exception:
            rec = 0.6  # 날짜 없으면 중립

        # 키워드 일치 점수
        title_low = title.lower()
        match = sum(1 for t in tokens if t in title_low)
        score = match + rec
        scored.append(NewsItem(title=title, url=url, published_at=when, score=score))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]

def _extract_keywords(text: str) -> List[str]:
    s = text.lower()
    # 간단 토크나이저: 영문/숫자/한글
    toks = re.findall(r"[a-zA-Z0-9가-힣]+", s)
    # 너무 일반적인 단어 제거
    stop = {"the","a","an","of","for","and","to","in","on","with","최근","분기","분석","데이터"}
    return [t for t in toks if t not in stop and len(t) > 1]

def parse_financials(raw: Any) -> List[FinancialKPI]:
    """
    MCP get_financial_statement 예상 구조:
    - dict 안 'data' 리스트. 항목 예: {"period":"2024Q4", "totalRevenue":..., "operatingIncome":..., "netIncome":..., "eps":...}
    키 이름은 서버 구현에 따라 다를 수 있어 최대한 유연하게 처리.
    """
    rows = []
    if isinstance(raw, dict):
        rows = raw.get("data") or raw.get("items") or raw.get("statements") or []
    elif isinstance(raw, list):
        rows = raw
    if not rows:
        return []

    # 표준 컬럼 매핑
    def pick(d: Dict[str, Any], *names) -> Optional[float]:
        for n in names:
            if n in d and d[n] is not None: 
                try: return float(d[n])
                except: pass
        return None

    records = []
    for r in rows:
        period = r.get("period") or r.get("endDate") or r.get("asOfDate") or "-"
        revenue = pick(r, "totalRevenue", "revenue", "sales")
        op = pick(r, "operatingIncome", "operating_income")
        ni = pick(r, "netIncome", "net_income")
        eps = pick(r, "eps", "dilutedEPS", "basicEPS")
        records.append(FinancialKPI(period=str(period), revenue=revenue, operating_income=op, net_income=ni, eps=eps,
                                    yoy_revenue=None, yoy_net_income=None))

    # YoY 계산(같은 분기/연도 페어링 가정: period 문자열 끝 4자리/분기 키 기준)
    # 간단 규칙: 같은 타입(period 포맷이 유사한 것끼리 인접 비교)
    def _yoy(base: Optional[float], prev: Optional[float]) -> Optional[float]:
        if base is None or prev is None or prev == 0: return None
        return (base / prev - 1.0) * 100.0

    # period가 정렬되었다고 가정하고 인접 YoY
    for i in range(1, len(records)):
        cur, prev = records[i], records[i-1]
        cur.yoy_revenue = _yoy(cur.revenue, prev.revenue)
        cur.yoy_net_income = _yoy(cur.net_income, prev.net_income)

    return records

# ---------- 최종 리듀서 ----------
def reduce_all(
    user_request: str,
    prices_raw: Any = None,
    news_raw: Any = None,
    financials_raw: Any = None,
    pdf_rag_summary: str = None,
    news_top_k: int = 3,
):
    errors = {}
    # 가격 파싱
    df = parse_prices(prices_raw)
    price_summary = compute_price_summary(df) if not df.empty else None

    # 뉴스/재무 파싱
    news = parse_news(news_raw, user_request, top_k=news_top_k) if news_raw is not None else []
    fins = parse_financials(financials_raw) if financials_raw is not None else []

    # 시그널 합성
    sig = {}
    sig.update(compute_price_signals(price_summary))
    sig.update(compute_fin_signals(fins))
    sig.update(compute_news_signals(news))

    # 에러 힌트 기록
    def pull_err(x, key):
        if isinstance(x, dict) and "error" in x:
            errors[key] = str(x["error"])
    pull_err(prices_raw, "prices")
    pull_err(news_raw, "news")
    pull_err(financials_raw, "financials")

    #  pdf_summary는 EssentialPack의 별도 필드로 전달
    pack = EssentialPack(
        price=price_summary,
        news=news,
        financials=fins,
        pdf_summary=pdf_rag_summary
    )

    return pack, sig, errors



