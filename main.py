# main.py — Financial Analysis & Report Generator
# Pipeline: M1 (analysis) → M2 (search/reduce) → M3 (review)
# Outputs HWP via MCP (changes are comment-only; code behavior unchanged)
# Note: Prompts and strings may include Korean by design — they are NOT modified here.

import os
import re
import sys
import time
import json
import asyncio
import traceback
from datetime import datetime, timezone, timedelta
from typing import TypedDict, Optional, Any, Dict, List, Tuple

from langgraph.graph import StateGraph, START, END
import ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# === MCP client ===
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import platform, shlex
from contextlib import suppress
print("[DBG] using main.py:", __file__)
TIMEOUT = int(os.environ.get("MCP_TIMEOUT", "20"))

def _stdio_cmd(cmdline: str):
    """OS별 STDIO 서버 스폰용 셸 커맨드를 만든다."""
    if platform.system() == "Windows":
        return ("cmd", ["/c", cmdline])
    else:
        return ("bash", ["-lc", cmdline])

def _log(*args):
    print(*args, flush=True)

YF_DIR = os.environ.get("YF_MCP_DIR", r"E:\AI_proj\a2a_mcp_prj\yahoo-finance-mcp")

YF_PY  = os.environ.get("YF_PY",  r"E:\Anaconda\envs\yahoo_mcp_env\python.exe")
YF_SRV = os.environ.get("YF_SRV", os.path.join(YF_DIR, "server.py"))

YF_SERVER = StdioServerParameters(
    command=YF_PY,
    args=[YF_SRV],
)

# ---- HWP MCP (STDIO, explicit python) ----
HWP_DIR = os.environ.get("HWP_MCP_DIR", r"E:\AI_proj\a2a_mcp_prj\hwp-mcp")

HWP_PY  = os.environ.get("HWP_PY",  r"E:\Anaconda\envs\hwp-mcp\python.exe")
HWP_SRV = os.environ.get("HWP_SRV", os.path.join(HWP_DIR, "hwp_mcp_stdio_server.py"))

HWP_SERVER = StdioServerParameters(
    command=HWP_PY,
    args=[HWP_SRV],
)
try:
    from m2_reducer import reduce_all
except Exception:
    def reduce_all(**kwargs):
        class Pack:
            def to_markdown(self_inner): return "가격/뉴스/재무 요약(모의)"
        return Pack(), {"price_empty": False, "news_empty": False, "fin_empty": False}, {}

try:
    from rag_module import load_pdf_to_vectorstore
except Exception:
    def load_pdf_to_vectorstore(*args, **kwargs):
        return None

# =========================
# =========================
M1_MODEL = "qwen2.5:14b-instruct"
M2_MODEL = "qwen2.5:7b-instruct-q4_K_M"
M3_MODEL = "qwen2.5:14b-instruct"

# =========================
# =========================
class GraphState(TypedDict, total=False):
    user_request: str
    search_results: Optional[str]
    analysis_result: Optional[str]
    validation_result: Optional[str]
    ppt_file: Optional[str]
    picked_symbol: str | None
    retry_count: int
    raw_prices: Any
    raw_news: Any
    raw_financials: Any
    sig: Dict[str, Any]
    m2_tools: Dict[str, bool]
    errors: List[Dict[str, Any]]
    pdf_outputs: List[Dict[str, Any]]
    content_plan: Dict[str, Any]
    tool_plan: List[Dict[str, Any]]
    outline_iter: int
    outline_approved: bool
    ppt_asset_meta: Dict[str, Any]
    ui_pdf_path: Optional[str]
    ui_ticker: Optional[str]
    ui_period: Optional[str]
    ui_interval: Optional[str]
# =========================
# =========================
def _safe_filename(name: str | None, maxlen: int = 100) -> str:
    s = str(name or "report")
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = s.strip().strip(".")
    if not s:
        s = "report"
    return s[:maxlen]

def resolve_symbol(state: dict) -> str:
    """
    사용자의 요청/분석 텍스트에서 티커를 최대한 정확히 추출/추론한다.
    우선순위:
      1) UI에서 지정한 티커 (ui_ticker)
      2) 강한 정규식 매칭(점/하이픈/접미사 포함)
      3) 회사명 → 티커 탐색(MCP에 search_ticker가 있으면 호출)
      4) LLM로 회사명만 뽑아 MCP 뉴스/가격에서 역추론

    """
    import re, json, asyncio

    ui_ticker = (state.get("ui_ticker") or "").strip().upper()
    if ui_ticker:
        return ui_ticker

    req = (state.get("user_request") or "") + "\n" + (state.get("analysis_result") or "")
    req_u = req.upper()
    pat = r"\b([A-Z0-9]{1,6}(?:\.[A-Z]{1,4})?(?:-[A-Z]{1,4})?(?:\.[A-Z]{1,4})?)\b"
    m = re.search(pat, req_u)
    if m:
        tok = m.group(1)
        if tok not in {"JSON","OK","NEWS","PRICE","FIN","TRUE","FALSE","AND","OR"}:
            return tok

    try:
        import ollama
        prompt = f"""다음 사용자 요청에서 '회사명 또는 티커'를 1개만 JSON으로 추출해줘.
출력 예시: {{"name": "마이크로소프트"}}
요청: {state.get("user_request","")}"""
        r = ollama.chat(model=state.get("M2_MODEL") or "qwen2.5:7b-instruct-q4_K_M",
                        messages=[{"role":"user","content": prompt}])
        txt = r["message"]["content"].strip()
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.MULTILINE)
        name = json.loads(txt).get("name","").strip()
    except Exception:
        name = ""

    async def _mcp_symbol_search(q: str) -> str:
        if not q:
            return ""
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client
        try:
            async with stdio_client(YF_SERVER) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=6)
                    tools = await asyncio.wait_for(session.list_tools(), timeout=6)
                    names = {t.name for t in tools.tools}
                    for cand in ("search_ticker","find_ticker","symbol_search"):
                        if cand in names:
                            res = await asyncio.wait_for(
                                session.call_tool(cand, {"query": q, "limit": 3}),
                                timeout=8
                            )
                            if hasattr(res,"content") and isinstance(res.content,list) and len(res.content)>0 and hasattr(res.content[0],"text"):
                                import json as _json
                                try:
                                    arr = _json.loads(res.content[0].text)
                                    if isinstance(arr, list) and arr:
                                        sym = arr[0].get("symbol") or arr[0].get("ticker")
                                        return (sym or "").upper()
                                except Exception:
                                    pass
                    return ""
        except Exception:
            return ""

    try:
        sym = asyncio.run(_mcp_symbol_search(name or state.get("user_request","")))
        if sym:
            return sym
    except Exception:
        pass

    KNOWN = {
        "마이크로소프트":"MSFT", "애플":"AAPL", "엔비디아":"NVDA", "아마존":"AMZN", "메타":"META",
        "테슬라":"TSLA", "알파벳":"GOOGL", "삼성전자":"005930.KS", "현대차":"005380.KS", "도요타":"7203.T",
        "소니":"6758.T", "텐센트":"0700.HK", "버크셔 해서웨이":"BRK.B",
    }
    for k,v in KNOWN.items():
        if k in req:
            return v

    return "MSFT"

def parse_requery_needs(text: str):
    """Parse analysis text to detect which data axes (price/news/fin) need re-query."""
    t = (text or "").lower()
    needs = set()
    if "재검색" not in t:
        return needs
    if re.search(r"(가격|price|주가)", t): needs.add("price")
    if re.search(r"(뉴스|news|뉍스)", t): needs.add("news")
    if re.search(r"(재무|재무제표|financial|financials)", t): needs.add("fin")
    if not needs:
        needs = {"price","news","fin"}
    return needs

def is_empty_or_error(data):
    """Return True if a data object is empty or contains an error marker."""
    if data is None:
        return True
    if isinstance(data, dict):
        if "error" in data:
            return True
        for k in ("data", "news", "prices", "financials", "items", "rows"):
            if k in data and isinstance(data[k], list) and len(data[k]) == 0:
                return True
        if not data:
            return True
        return False
    if isinstance(data, (list, tuple)) and len(data) == 0:
        return True
    return False

def to_jsonable(obj):
    """Safely convert objects (including numpy scalars/containers) to JSON-serializable forms."""
    try:
        import numpy as np
        np_generic = (np.generic,)
    except Exception:
        np_generic = tuple()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if np_generic and isinstance(obj, np_generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return None

def _extract_list_from_raw(raw, keys=("prices","data","news","items")):
    """Extract a list payload from a possibly nested dict under common keys."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for k in keys:
            v = raw.get(k)
            if isinstance(v, list):
                return v
    return []

def _extract_first_json_block(s: str) -> str:
    """Extract the first JSON object/array block from a text snippet."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    start_obj = s.find("{")
    start_arr = s.find("[")
    starts = [p for p in (start_obj, start_arr) if p != -1]
    if not starts:
        return s
    start = min(starts)
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return s

def safe_json_loads(txt: str) -> dict:
    """Robust JSON loader that cleans common issues (quotes, comments, trailing commas)."""
    s = (txt or "").strip()

    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    s = _extract_first_json_block(s)

    try:
        return json.loads(s)
    except Exception:
        pass

    s2 = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s2 = s2.replace("\ufeff", "")                # BOM
    s2 = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s2)

    s2 = re.sub(r"//.*?$|/\*.*?\*/", "", s2, flags=re.MULTILINE | re.DOTALL)

    s2 = re.sub(r"\bTrue\b", "true", s2)
    s2 = re.sub(r"\bFalse\b", "false", s2)
    s2 = re.sub(r"\bNone\b", "null", s2)

    s2 = re.sub(r",\s*([}\]])", r"\1", s2)

    s2 = re.sub(r'(\{|,)\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1 "\2":', s2)

    if s2.count('"') == 0 and s2.count("'") > 0:
        s2 = s2.replace("'", '"')

    s2 = _extract_first_json_block(s2)
    try:
        return json.loads(s2)
    except Exception:
        low = (txt or "").lower()
        decision = "approved" if ("approved" in low or "ok" in low) else "refine"
        return {"decision": decision, "raw_excerpt": (txt or "")[:600]}

# =========================
# =========================
def run_adaptive_rag(vectorstore, query: str, top_k: int = 8) -> Optional[Dict[str, Any]]:
    """
    - LangChain Retriever에서 바로 top-k 패시지를 뽑아 원문/페이지 메타와 함께 반환
    - LLM(채움 단계)에서 직접 인용/표로 활용하도록 함
    """
    if vectorstore is None:
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    try:
        docs = retriever.get_relevant_documents(query)
    except Exception:
        docs = retriever.invoke(query)

    passages = []
    for rank, d in enumerate(docs or [], 1):
        meta = getattr(d, "metadata", {}) or {}
        passages.append({
            "rank": rank,
            "source": meta.get("source") or meta.get("file_path") or "pdf",
            "page": int(meta.get("page") or meta.get("page_number") or -1),
            "text": d.page_content,
        })
    return {"passages": passages}

# =========================
# =========================
def M1_analysis_node(state: GraphState):
    """Run M1 to decide whether data is sufficient or which axes require re-fetch."""
    print("\n[M1] 사용자 요청 분석 시작")
    prompt = """
너는 금융 분석 에이전트다. 아래 신호에 따라 'OK' 또는 '재검색: <부족항목, ...>' 중 하나만 출력해라.

신호(sig):
{json.dumps(sig, ensure_ascii=False)}

규칙(엄격):
- price_empty/news_empty/fin_empty 중 True가 있으면 '재검색: ' 뒤에 해당 항목(가격/뉴스/재무)을 한글로 콤마로 나열.
- 세 항목 모두 False이면 'OK'만 출력.
- 다른 말은 절대 쓰지 말 것.
    """
    res = ollama.chat(model=M1_MODEL, messages=[{"role": "user", "content": prompt}])
    answer = res['message']['content'].strip()
    print(f"[M1 응답] {answer}")
    state["analysis_result"] = answer
    return state

# =========================
# =========================
async def yahoo_mcp_call(symbol="MSFT", period="6mo", interval="1d",
                         fetch_price=True, fetch_news=True, fetch_fin=True):
    """Yahoo Finance MCP: STDIO + 단계별 디버그 로그 + 타임아웃"""
    import os, json, asyncio, time, traceback
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    TIMEOUT_INIT = 8
    TIMEOUT_CALL = 12

    def _unwrap_tool_result(result):
        try:
            if hasattr(result, 'content'):
                content = result.content
                if isinstance(content, list):
                    for item in content:
                        if hasattr(item, 'type') and hasattr(item, 'text'):
                            try:
                                return json.loads(item.text)
                            except Exception:
                                return {"text": item.text}
                elif isinstance(content, dict):
                    return content
            if isinstance(result, dict):
                return result
        except Exception as e:
            return {"error": str(e)}
        return {"error": "Unknown tool result format"}

    try:
        from pprint import pformat
        print("[YF][DBG] ENTER", time.time(), flush=True)
        print("[YF][DBG] PY :", YF_PY,  "exists=", os.path.exists(YF_PY),  flush=True)
        print("[YF][DBG] SRV:", YF_SRV, "exists=", os.path.exists(YF_SRV), flush=True)
        try:
            print("[YF][DBG] SERVER:", getattr(YF_SERVER, "command", None), getattr(YF_SERVER, "args", None), flush=True)
        except Exception:
            pass
    except Exception:
        pass

    print("[YF][DBG] stdio_client: ENTER", flush=True)
    try:
        async with stdio_client(YF_SERVER) as (read, write):
            print("[YF][DBG] stdio_client: OK (context opened)", flush=True)

            async with ClientSession(read, write) as session:
                print("[YF][DBG] init: start", flush=True)
                await asyncio.wait_for(session.initialize(), timeout=TIMEOUT_INIT)
                print("[YF][DBG] init: ok", flush=True)

                print("[YF][DBG] tools: start", flush=True)
                tools = await asyncio.wait_for(session.list_tools(), timeout=TIMEOUT_INIT)
                print("[YF][DBG] tools: ok", [t.name for t in tools.tools], flush=True)

                names = {t.name for t in tools.tools}
                def pick(*cands):
                    for c in cands:
                        if c in names:
                            return c
                    return None

                tool_price = pick(
                    "get_historical_stock_prices",
                    "historical_prices","get_prices",
                    "get_price","price",
                    "yahoo_get_price"
                    )

                tool_news  = pick(
                    "get_yahoo_finance_news",
                    "news",
                    "get_finance_news",
                    "get_news",
                    "yahoo_get_news"
                    )

                tool_fin   = pick(
                    "get_financial_statement",
                    "financials","get_financials",
                    "get_financials_statement",
                    "get_financials_v2",
                    "yahoo_get_financials"
                    )

                out = {"prices": None, "news": None, "financials": None}

                if fetch_price and tool_price:
                    try:
                        print("[YF][DBG] call: price", flush=True)
                        r = await asyncio.wait_for(
                            session.call_tool(tool_price, {"ticker": symbol, "period": period, "interval": interval}),
                            timeout=TIMEOUT_CALL
                        )
                        out["prices"] = _unwrap_tool_result(r)
                    except Exception as e:
                        out["prices"] = {"error": f"price: {e}"}

                if fetch_news and tool_news:
                    try:
                        print("[YF][DBG] call: news", flush=True)
                        r = await asyncio.wait_for(
                            session.call_tool(tool_news, {"ticker": symbol, "days": 180, "limit": 20}),
                            timeout=TIMEOUT_CALL
                        )
                        out["news"] = _unwrap_tool_result(r)
                    except Exception as e:
                        out["news"] = {"error": f"news: {e}"}

                if fetch_fin and tool_fin:
                    try:
                        print("[YF][DBG] call: fin", flush=True)
                        r = await asyncio.wait_for(
                            session.call_tool(tool_fin, {"ticker": symbol, "financial_type": "income", "limit": 8}),
                            timeout=TIMEOUT_CALL
                        )
                        out["financials"] = _unwrap_tool_result(r)
                    except Exception as e:
                        out["financials"] = {"error": f"financials: {e}"}

                print("[YF][DBG] done", flush=True)
                return out

    except Exception as e:
        print("[YF][DBG] EXC:", repr(e), "\n", traceback.format_exc(), flush=True)
        return {
            "prices": {"error": f"{e}"},
            "news": {"error": f"{e}"},
            "financials": {"error": f"{e}"},
        }

def M2_search_node(state: GraphState):
    """Plan fetch axes, call MCP, run reducer, and populate state with raw results."""
    import re, os, json, asyncio, concurrent.futures, traceback
    from typing import Optional, Dict, Any

    print("[M2] Yahoo Finance MCP 호출... (model:", M2_MODEL, ")")
    m1_msg = state.get("analysis_result", "")

    plan_prompt = f"""
너는 검색 오케스트레이터야. 다음 텍스트를 보고 어느 축을 조회할지 'price/news/fin' 중 True/False로 계획을 세워.
입력:
{m1_msg}

규칙:
- 첫 실행이면 price/news/fin 모두 True.
- '재검색'이면 부족한 축만 True (이미 충분한 축은 False).
- JSON 형식으로만 답해. 키: price, news, fin
예시: {{"price": true, "news": false, "fin": true}}
"""
    plan_fetch = {"price": False, "news": False, "fin": False}
    first_run = not any([state.get("raw_prices"), state.get("raw_news"), state.get("raw_financials")])

    try:
        plan_res = ollama.chat(model=M2_MODEL, messages=[{"role": "user", "content": plan_prompt}])
        plan_txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", plan_res["message"]["content"].strip(), flags=re.MULTILINE)
        plan_fetch = json.loads(plan_txt)
    except Exception:
        needs = parse_requery_needs(m1_msg)
        if first_run:
            plan_fetch = {"price": True, "news": True, "fin": True}
        else:
            plan_fetch = {
                "price": ("price" in needs) or is_empty_or_error(state.get("raw_prices")),
                "news":  ("news" in needs)  or is_empty_or_error(state.get("raw_news")),
                "fin":   ("fin" in needs)   or is_empty_or_error(state.get("raw_financials")),
            }
    if first_run:
        plan_fetch = {"price": True, "news": True, "fin": True}

    fetch_price = bool(plan_fetch.get("price"))
    fetch_news  = bool(plan_fetch.get("news"))
    fetch_fin   = bool(plan_fetch.get("fin"))

    req_all = (state.get("user_request", "") or "") + "\n" + (state.get("analysis_result", "") or "")
    req_clean = re.sub(r"```.*?```", "", req_all, flags=re.DOTALL)

    ui_tkr = (state.get("ui_ticker") or "").strip().upper()
    pat = r"\b([A-Z0-9]{1,6}(?:\.[A-Z]{1,4})?(?:-[A-Z]{1,4})?(?:\.[A-Z]{1,4})?)\b"
    ban = {
        "JSON","OK","NEWS","PRICE","FIN","TRUE","FALSE","NONE","AND","OR",
        "IMPORT","PRINT","DEF","CLASS","RETURN","IF","ELSE","TRY","EXCEPT","WITH","AS","FOR","WHILE"
    }
    m = re.search(pat, req_clean.upper())
    regex_pick = m.group(1) if (m and m.group(1) not in ban) else ""

    KNOWN = {
        "마이크로소프트":"MSFT","애플":"AAPL","엔비디아":"NVDA","아마존":"AMZN","메타":"META",
        "테슬라":"TSLA","알파벳":"GOOGL","삼성전자":"005930.KS","현대차":"005380.KS",
        "도요타":"7203.T","소니":"6758.T","텐센트":"0700.HK","버크셔 해서웨이":"BRK.B",
    }
    mapped = next((v for k,v in KNOWN.items() if k in req_all), "")

    symbol = ui_tkr or regex_pick or mapped or "MSFT"
    print("[M2] 사용할 심볼:", symbol)
    state["picked_symbol"] = symbol
    period = state.get("ui_period") or "6mo"
    interval = state.get("ui_interval") or "1d"

    print("[DBG] yahoo_mcp_call from:", yahoo_mcp_call.__module__, getattr(yahoo_mcp_call, "__code__", None).co_filename, getattr(yahoo_mcp_call, "__code__", None).co_firstlineno, flush=True)

    price_raw = None
    news_raw  = None
    fin_raw   = None

    try:
        data = asyncio.run(
            yahoo_mcp_call(
                symbol=symbol, period=period, interval=interval,
                fetch_price=fetch_price, fetch_news=fetch_news, fetch_fin=fetch_fin
            )
        )
        price_raw = data.get("prices")
        news_raw  = data.get("news")
        fin_raw   = data.get("financials")
    except Exception as e:
        prev_err = state.get("errors", [])
        if isinstance(prev_err, dict): prev_err = [prev_err]
        elif not isinstance(prev_err, list): prev_err = [str(prev_err)]
        state["errors"] = prev_err + [{"where": "M2/yahoo", "error": str(e)}]

        data = {"prices": {"error": str(e)}, "news": {"error": str(e)}, "financials": {"error": str(e)}}

    if fetch_price:
        state["raw_prices"] = {"data": [], "note": "가격 데이터 없음"} if is_empty_or_error(price_raw) else price_raw
    if fetch_news:
        if is_empty_or_error(news_raw):
            state["raw_news"] = {"news": [], "note": "관련 뉴스 없음"}
        else:
            state["raw_news"] = news_raw
            if isinstance(state["raw_news"], dict) and isinstance(state["raw_news"].get("news"), list) and len(state["raw_news"]["news"]) == 0:
                state["raw_news"] = {"news": [], "note": "관련 뉴스 없음"}
    if fetch_fin:
        if is_empty_or_error(fin_raw):
            state["raw_financials"] = {"data": [], "note": "재무 데이터 없음"}
        else:
            state["raw_financials"] = fin_raw
            if isinstance(state["raw_financials"], dict) and isinstance(state["raw_financials"].get("data"), list) and len(state["raw_financials"]["data"]) == 0:
                state["raw_financials"] = {"data": [], "note": "재무 데이터 없음"}

    pdf_rag_obj: Optional[Dict[str, Any]] = None
    pdf_rag_summary_text: Optional[str] = None
    pdf_path = state.get("ui_pdf_path") or "E:/AI_proj/a2a_mcp_prj/data/sample.pdf"
    if os.path.exists(pdf_path):
        try:
            vectorstore = load_pdf_to_vectorstore(pdf_path)
            pdf_rag_obj = run_adaptive_rag(vectorstore, query=state["user_request"], top_k=8)
            if pdf_rag_obj and isinstance(pdf_rag_obj.get("passages"), list):
                chunks = []
                for p in pdf_rag_obj["passages"][:8]:
                    pg = f"p.{p.get('page')}" if p.get('page', -1) != -1 else "p.-"
                    src = p.get("source", "pdf")
                    txt = (p.get("text") or "").strip()
                    if txt:
                        chunks.append(f"[{src} {pg}] {txt}")
                if chunks:
                    pdf_rag_summary_text = "\n\n---\n".join(chunks)[:4000]
        except Exception as e:
            print("[M2] PDF RAG 실패:", e)

    state["pdf_rag"] = pdf_rag_obj
    if pdf_rag_summary_text:
        state["pdf_rag_summary"] = pdf_rag_summary_text

    print("[M2] reduce_all 시작", flush=True)

    class _Pack:
        def __init__(self, md: str): self._md = md
        def to_markdown(self): return self._md

    def _do_reduce():
        return reduce_all(
            user_request=state["user_request"],
            prices_raw=state.get("raw_prices"),
            news_raw=state.get("raw_news"),
            financials_raw=state.get("raw_financials"),
            pdf_rag_summary=pdf_rag_summary_text,
            news_top_k=3,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_reduce)
            pack, sig, errors = fut.result(timeout=40)
    except concurrent.futures.TimeoutError:
        print("[M2] reduce_all 타임아웃 → 안전 폴백 사용", flush=True)
        pack  = _Pack("
        sig   = {"price_empty": False, "news_empty": False, "fin_empty": False}
        errors = [{"where":"M2/reduce_all","error":"timeout"}]
    except Exception as e:
        print("[M2] reduce_all 예외:", e, "\n", traceback.format_exc(), flush=True)
        pack  = _Pack(f"
        sig   = {"price_empty": False, "news_empty": False, "fin_empty": False}
        errors = [{"where":"M2/reduce_all","error":str(e)}]

    state["search_results"] = pack.to_markdown()
    state["sig"] = to_jsonable(sig)

    prev = state.get("errors", [])
    if isinstance(prev, dict):
        prev = [prev]
    elif not isinstance(prev, list):
        prev = [str(prev)]
    if errors is None:
        errors = []
    elif isinstance(errors, dict):
        errors = [errors]
    elif not isinstance(errors, list):
        errors = [str(errors)]
    state["errors"] = prev + errors

    state["m2_tools"] = {"price": fetch_price, "news": fetch_news, "fin": fetch_fin}

    state["picked_symbol"] = symbol

    price_raw = state.get("raw_prices")
    news_raw  = state.get("raw_news")
    fin_raw   = state.get("raw_financials")

    state["mcp_json"] = {
        "symbol":   symbol,
        "period":   period,
        "interval": interval,
        "price":    to_jsonable(price_raw),
        "news":     to_jsonable(news_raw),
        "fin":      to_jsonable(fin_raw),
        "pdf_rag":  to_jsonable(pdf_rag_obj),
        "sig":      to_jsonable(sig),
        "errors":   to_jsonable(state.get("errors", [])),
    }

    print("[M2] mcp_json keys:", list(state["mcp_json"].keys()))
    print("[M2] picked_symbol:", state.get("picked_symbol"))
    print("[M2] (plan) 축 선택:", state["m2_tools"])
    print("[M2] sig:", state["sig"])
    if errors:
        print("[M2] 에러들:", errors)
    print(f"[M2] 검색/요약 완료. (price={fetch_price}, news={fetch_news}, fin={fetch_fin}, pdf={pdf_rag_obj is not None})")

    return state

# =========================
# =========================
def M1_check_search_node(state: GraphState):
    """Second-pass check to decide if another search is needed based on signals."""
    print("[M1] 검색 결과 검토 중...")

    def is_real_missing(raw, sig_flag):
        if not sig_flag:
            return False
        if isinstance(raw, dict) and raw.get("note"):
            return False
        return True

    if (state.get("search_results") or "").strip():
        sig = state.get("sig", {}) or {}
        needs = []
        if is_real_missing(state.get("raw_prices"), sig.get("price_empty")):
            needs.append("가격")
        if is_real_missing(state.get("raw_news"), sig.get("news_empty")):
            needs.append("뉴스")
        if is_real_missing(state.get("raw_financials"), sig.get("fin_empty")):
            needs.append("재무")

        if needs:
            state["analysis_result"] = "재검색: " + ", ".join(needs)
            state["retry_count"] = state.get("retry_count", 0) + 1
            print("[M1] (판정) 부족 축 →", needs)
        else:
            state["analysis_result"] = "OK"
            print("[M1] (판정) 충분 → OK로 종료")

        print("[M1] (참고) 이번 M2 호출 축:", state.get("m2_tools"))
        return state

    if "재검색" in state.get("analysis_result", ""):
        state["retry_count"] = state.get("retry_count", 0) + 1
    print("[M1] (참고) 이번 M2 호출 축:", state.get("m2_tools"))
    return state

# =========================
# =========================
def M1_content_plan_node(state: GraphState):
    """Draft content plan (slides/blocks) based on available data profile."""
    print("[M1] (자유 설계) 콘텐츠 계획 생성.")
    m = re.search(r"[A-Z0-9]{1,6}(?:\.[A-Z]{1,4})?", (state.get("user_request","") or "").upper())
    ticker_guess = (
        (state.get("picked_symbol") or "").upper()
        or (state.get("ui_ticker") or "").upper()
        or (m.group(0) if m else "")
        or "MSFT"
    )

    prices = _extract_list_from_raw(state.get("raw_prices"), keys=("prices","data"))
    news   = _extract_list_from_raw(state.get("raw_news"),   keys=("news","items"))
    fins   = _extract_list_from_raw(state.get("raw_financials"), keys=("data",))

    data_profile = {
        "ticker": ticker_guess,
        "period": state.get("ui_period") or "6mo",
        "counts": {"prices": len(prices), "news": len(news), "financials": len(fins)},
        "signals": state.get("sig") or {},
        "has_pdf": bool(state.get("pdf_rag") or state.get("pdf_rag_summary"))
    }

    prompt = f"""
역할: 너는 "금융 애널리스트 + 문서 디자이너"다.
목표: 아래 데이터 프로파일을 보고 HWP 보고서 섹션을 **LLM이 스스로** 설계하라.
(코드 보강 없음. 네가 만든 설계가 그대로 최종 문서에 쓰인다.)

데이터 프로파일:
{json.dumps(to_jsonable(data_profile), ensure_ascii=False)}

[강제 규칙 — 커버(표지) 없이 시작]
- **cover/표지 섹션을 만들지 말라.** slides[*] 어디에도 "cover" 또는 "표지"라는 title/문단을 생성하지 말 것.
- 문서 제목 문자열(예: "{ticker_guess} 6개월 보고서")은 **문서 전체에서 정확히 1번만** 등장해야 한다.
  - 위치: **slides[0]의 첫 번째 paragraph.summary_points[0]** 한 줄로만 사용.
  - 다른 어떤 블록/슬라이드/제목에도 반복 금지.
- "작성일:" 같은 날짜 표시는 **생성하지 말라.** (날짜가 필요하면 본문 문맥 속에서 한 번만 자연스럽게 언급)

[블록/타이틀 규칙]
- slides[*].title은 모두 서로 달라야 하며, **문서 제목과 동일 금지**, 티커(예: "{ticker_guess}")만 단독으로 쓰는 제목 금지.
- **chart 블록 금지**(paragraph/table만 사용).
- slides[*]마다 paragraph 블록을 최소 1개 포함. paragraph는 summary_points(List[str])만 쓴다(빈 배열 금지).
- table은 필요한 곳에서만:
  - 뉴스면 ["날짜","제목","링크"]
  - 재무면 ["기간","매출","영업이익","순이익"]
  - PDF 인용용이면 ["근거","인용"]
- footnotes는 **실제 출처(URL/신뢰 출처명)** 있을 때만 포함. 없으면 키 생략.

[JSON 스키마]
{{
  "meta": {{
    "ticker":"{ticker_guess}",
    "period":"6mo",
    "lang":"ko-KR",
    "created_at":"{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}",
    "slide_budget": <int>,     // 4~10 권장
    "rationale": "<왜 그 수인지 한 줄>"
  }},
  "slides": [
    {{
      "id":"S0",
      "title":"<첫 섹션 제목>",      // 예: "시장 동향 요약" (문서 제목/티커 금지)
      "objective":"<섹션 목표 한 줄>",
      "plan_blocks":[
        {{ "type":"paragraph", "summary_points":["{ticker_guess} 6개월 보고서"] }},   // ← 문서 제목: 이 한 줄만 허용
        {{ "type":"paragraph", "summary_points":["핵심 1","핵심 2","핵심 3","핵심 4","핵심 5"] }}
      ]
      // footnotes: 있으면만 문자열
    }}
  ]
}}

[지침]
- 뉴스>=5 → 뉴스 타임라인/주제 요약 섹션 고려
- 재무>=4분기 → QoQ/YoY 섹션 1개, >=8분기 → 마진/비율 섹션 1개
- 변동성↑ → 리스크 섹션 1개
- 데이터 빈약 → 한계/주의 섹션 1개
- 마지막 결론/다음 액션 섹션 권고

[출력 규칙]
- 오직 **JSON**만 출력(설명/코드펜스 금지).
- paragraph/table만 사용(chart 금지).
- 문서 제목은 slides[0]의 첫 문단에서 **1회만** 등장해야 한다.
"""

    res = ollama.chat(model=M1_MODEL, messages=[{"role":"user","content": prompt}])
    content_plan = safe_json_loads(res["message"]["content"].strip())
    state["content_plan"] = content_plan
    state["outline_iter"] = state.get("outline_iter", 0) + 1
    state["outline_approved"] = False
    return state

def M1_fill_section_content_with_llm(state: GraphState):
    """
    목적: content_plan(섹션 틀)을 기반으로 LLM이 실제 본문을 '단일 패스'로 길고 구체적으로 채우게 한다.
    - 커버/표지 생성 금지, 제목 반복 금지(프롬프트에서 강제)
    - 매 불릿은 '구체 수치/날짜/사건/링크' 중 ≥1개를 반드시 포함(메타/자리표시 금지)
    - PDF RAG 패시지 있으면 직접 인용 4~8개 + 근거표 4~8행 '필수'
    - densify/로컬 후가공
    """
    print("[M1] 섹션 내용 LLM 채움…(구체화 강제 + PDF 반영)")

    plan = state.get("content_plan") or {}
    if not plan:
        print("[M1] content_plan 없음 → 패스")
        return state

    praw = state.get("raw_prices")
    nraw = state.get("raw_news")
    fraw = state.get("raw_financials")
    pdf_rag = state.get("pdf_rag") or state.get("pdf_rag_summary")

    data_bundle = {
        "prices": to_jsonable(praw),
        "news": to_jsonable(nraw),
        "financials": to_jsonable(fraw),
        "pdf_rag": to_jsonable(pdf_rag),
    }

    req = (state.get("user_request") or "")
    m = re.search(r"[A-Z0-9]{1,6}(?:\.[A-Z]{1,4})?", req.upper())

    ticker_guess = (
        (state.get("picked_symbol") or "").upper()
        or (state.get("ui_ticker") or "").upper()
        or (m.group(0) if m else "").upper()
        or (plan.get("meta", {}).get("ticker") or "").upper()
        or "REPORT"
    )

    plan.setdefault("meta", {})
    plan["meta"]["ticker"] = ticker_guess
    plan["meta"]["period"] = state.get("ui_period") or plan["meta"].get("period") or "6mo"

    doc_title = plan["meta"].get("title") or f"{ticker_guess} 6개월 보고서"
    plan["meta"]["title"] = doc_title

    print("[M1] doc_title:", doc_title, "meta.ticker:", plan["meta"]["ticker"])

    pdf_passages_block = ""
    try:
        if isinstance(pdf_rag, dict) and isinstance(pdf_rag.get("passages"), list):
            lines = []
            for p in pdf_rag["passages"][:12]:
                src = str(p.get("source") or "pdf")
                pg = p.get("page")
                pg_str = f"p.{pg}" if isinstance(pg, int) and pg >= 0 else "p.-"
                txt = (p.get("text") or "").strip()
                if txt:
                    lines.append(f"[{src} / {pg_str}] {txt[:900]}")
            if lines:
                pdf_passages_block = "\n\n".join(lines)
        elif isinstance(pdf_rag, str) and pdf_rag.strip():
            pdf_passages_block = pdf_rag.strip()[:9000]
    except Exception:
        pass

    prompt = f"""
역할: 너는 "금융 애널리스트 + 문서 작가"다.
아래 원천 데이터와 섹션 틀(content_plan)을 근거로 **본문을 반드시 구체적으로 채워라**.
(코드 보강/하드코딩 없음. 네가 만든 JSON이 곧 최종 결과다.)

[원천 데이터(JSON 요약)]
{json.dumps(data_bundle, ensure_ascii=False)[:12000]}

[PDF 원문 패시지(직접 인용 가능)]
{pdf_passages_block if pdf_passages_block else "(PDF 패시지 없음)"}

[현재 섹션 설계(틀)]
{json.dumps(to_jsonable(plan), ensure_ascii=False)[:12000]}

형식 규칙(엄격):
- 오직 **JSON**만 출력. 전체 content_plan(JSON 객체: meta, slides 포함)으로 반환(배열/부분/문자열 금지).
- **cover/표지 슬라이드 생성 금지.**
- 문서 제목 "{doc_title}"은 content_plan에 이미 존재하는 위치 외에는 **절대 반복 금지**.

본문 채움 규칙(핵심):
- slides[*]마다 **paragraph 블록 ≥1개**가 있어야 하며, 없으면 **네가 생성**해 채운다.
- paragraph는 **summary_points(List[str])만** 사용(빈 배열/빈 문자열 금지, text 키 금지).
- 각 summary_points는 **10~14개**, 각 **25~90자**, 중복/동어반복 금지.
- **중요: 다음과 같은 '자리표시/메타 불릿'은 금지** — 아래 예시와 유사한 불릿이 보이면 **데이터 근거가 있는 구체 문장으로 교체**할 것:
  - "시장 동향 요약", "주요 이벤트 및 뉴스 요약", "기술 지표 분석", "투자 전략 제안", "요약 및 결론", "핵심 1/2/3", "개요", "분석", "제안"
- **각 불릿에는 아래 중 하나 이상을 반드시 포함할 것**(원천 데이터 범위 내에서):
  - 날짜(YYYY-MM-DD), 수치(가격/비율/증감률), 사건명/뉴스 제목/링크, 재무 항목과 값(QoQ/YoY 방향 포함)
- table은 **table_spec가 있는 블록만** rows를 채운다. 데이터 부족하면 table은 생략하고 paragraph로 설명.
- **chart 블록 금지.**
- footnotes는 **실제 출처(URL/신뢰 출처명)**가 있을 때만 문자열로 포함(없으면 **키 생략**).

PDF 강제 반영(있을 때 필수):
- 위 [PDF 원문 패시지]에서 **4~8개를 직접 인용**해 본문 paragraph에 포함하라.
  - 인용 형식: "…원문 일부…" (p.<페이지> / <source>)
- 또한 표 1개를 추가하라(가능하면 PDF 관련 섹션에):
  - 헤더: ["근거","인용"]
  - 행: **4~8개**, 각 행은 (근거 요약, 짧은 직접 인용문)으로 작성.

검증:
- JSON 외 텍스트 금지.
- 어떤 슬라이드도 paragraph가 비어 있으면 안 된다(네가 생성해서 채울 것).
- "{doc_title}" 문자열이 기존 위치 외에는 등장하지 않아야 한다.
- 자리표시/메타 불릿이 발견되면 모두 **구체 불릿으로 대체**되어야 한다.
""".strip()

    def _call_and_parse(pmt: str):
        try:
            res = ollama.chat(
                model=M1_MODEL,
                messages=[{"role": "user", "content": pmt}],
                options={
                    "temperature": 0.2,
                    "num_ctx": 8192,
                    "num_predict": 7168,
                    "format": "json"
                }
            )
        except Exception as e:
            print("[M1] LLM 호출 에러:", e)
            return None
        raw = (res.get("message") or {}).get("content", "")
        if not raw or not raw.strip():
            print("[M1] LLM 응답 비어있음")
            return None
        try:
            return safe_json_loads(raw.strip())
        except Exception:
            print("[M1] JSON 파싱 실패. 앞부분:", raw[:400].replace("\n","\\n"))
            return None

    filled = _call_and_parse(prompt)

    if not (isinstance(filled, dict) and isinstance(filled.get("slides"), list)):
        prompt2 = (
            "재지시: 오직 JSON만, 전체 content_plan 객체로 반환. 커버/표지 금지. "
            f'"{doc_title}" 반복 금지. slides[*]는 paragraph.summary_points(10~14개)로 반드시 채울 것. '
            "자리표시/메타 불릿 금지 → 모두 구체 불릿으로 대체. table은 spec 있는 것만. "
            "PDF 인용 4~8개 + 근거표(4~8행) 필수.\n\n" + prompt
        )
        filled = _call_and_parse(prompt2)

    if not (isinstance(filled, dict) and isinstance(filled.get("slides"), list)):
        print("[M1] 경고: 채움 실패 → 원본 plan 유지")
        return state

    state["content_plan"] = filled
    print("[M1] 섹션 채움 완료(구체화 강제 + PDF 반영)")
    return state

# =========================
# =========================
def M3_review_content_plan_node(state: GraphState):
    """Review the content plan against hard rules and return structured feedback."""
    print("[M3] 콘텐츠 계획 리뷰...")
    plan = state.get("content_plan") or {}
    has_pdf = bool(state.get("pdf_rag") or state.get("pdf_rag_summary"))

    schema_hint = {
        "decision": "approved",     # or "refine"
        "reasons": ["..."],
        "must_fixes": [],
        "footnotes_ok": True,
        "violations": []
    }

    prompt = f"""
역할: 너는 "검증 에이전트(M3)"다. 아래 content_plan이 HWP 보고서 규칙을 만족하는지 **네가 직접 계수/판단**해서 결과를 JSON만으로 내라.

검증 대상 규칙(LLM이 직접 확인):
- COVER_3PARA: slides[0]는 표지(cover)이며 paragraph 3개(제목/작성일/대상·기간)만 포함, 표/차트 금지.
- NO_CHART: chart 블록 금지(어느 슬라이드에도 없어야 함). paragraph/table만 사용.
- TITLES_UNIQUE: slides[*].title은 서로 달라야 하고, 표지 제목("{(plan.get('slides') or [{}])[0].get('plan_blocks',[{},{},{}])[0].get('summary_points',[''])[0] if plan else ''}")과 같으면 안 됨.
- PARAGRAPH_MIN: slides[1..] 각 슬라이드에 paragraph ≥1, 그 paragraph.summary_points는 **최소 8개** 이상(비어있거나 text키만 있으면 안 됨).
- FOOTNOTES_VALID: footnotes는 **실제 출처(URL/신뢰 출처명)**가 있을 때만 포함(없으면 키 생략/빈 값 금지).
- PDF_SECTION: has_pdf=={str(has_pdf).lower()} 인 경우, **PDF 인사이트 섹션 ≥1개** 포함(제목/문단/필요시 근거표).

출력 형식(엄격):
- 오직 JSON만 출력. 설명/코드펜스 금지.
- 스키마는 아래 예시를 따름(키/형식 유지):
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}

판정 규칙:
- violations가 비어 있으면 decision="approved"
- 하나라도 있으면 decision="refine" + must_fixes에 바로 적용 가능한 수정 지시를 한국어로 구체적으로 적을 것(최대 8개).

content_plan:
{json.dumps(to_jsonable(plan), ensure_ascii=False)[:12000]}
""".strip()

    try:
        res = ollama.chat(
            model=M3_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": 768,
                "format": "json"
            }
        )
        raw = (res.get("message") or {}).get("content", "")
        fb = safe_json_loads(raw.strip())
    except Exception as e:
        print("[M3] 호출/파싱 실패:", e)
        fb = {
            "decision": "refine",
            "reasons": ["JSON 파싱 실패"],
            "must_fixes": ["M3 응답을 유효 JSON으로만 재생성하도록 수정"],
            "footnotes_ok": False,
            "violations": ["E_PARSE"]
        }

    state["m3_feedback"] = fb
    state["outline_approved"] = (str(fb.get("decision","refine")).lower() == "approved")
    return state

# =========================
# =========================
def M1_refine_content_plan_node(state: GraphState):
    """Refine content plan using review feedback; return updated plan."""
    print("[M1] 콘텐츠 계획 수정...")
    plan = state.get("content_plan") or {}
    fb = state.get("review_feedback") or {}

    prompt = f"""
역할: PPT 설계자.
아래 계획을 피드백에 맞게 수정해라. JSON만 반환.

기존 계획:
{json.dumps(to_jsonable(plan), ensure_ascii=False)[:8000]}

피드백:
{json.dumps(to_jsonable(fb), ensure_ascii=False)}
"""
    res = ollama.chat(model=M1_MODEL, messages=[{"role":"user","content": prompt}])
    revised = safe_json_loads(res["message"]["content"].strip())

    state["content_plan"] = revised
    state["outline_iter"] = state.get("outline_iter", 0) + 1
    state["outline_approved"] = False
    return state

# =========================
# =========================
def M1_translate_plan_to_tool_plan_node(state: GraphState):
    """Translate content plan to tool operations for document generation."""
    print("[M1] 계획 → MCP 툴 플랜 변환(ROBUST)...")
    plan = state.get("content_plan") or {}
    pid_placeholder = "__PRESENTATION_ID__"

    praw = state.get("raw_prices")
    nraw = state.get("raw_news")
    fraw = state.get("raw_financials")

    prices = _extract_list_from_raw(praw, keys=("prices","data"))
    news   = _extract_list_from_raw(nraw, keys=("news","items"))
    fins   = _extract_list_from_raw(fraw, keys=("data",))

    def last_120_prices(rows) -> Tuple[List[str], List[float]]:
        x, y = [], []
        for r in rows[-120:]:
            if not isinstance(r, dict):
                continue
            d = r.get("date") or r.get("Date") or r.get("timestamp") or r.get("Datetime")
            c = r.get("close") or r.get("Close") or r.get("adjClose") or r.get("Adj Close")
            if d is None:
                continue
            d_str = str(d)[:10] if len(str(d)) >= 10 else str(d)
            try:
                c_val = float(c)
            except Exception:
                continue
            x.append(d_str); y.append(c_val)
        return x, y

    def fin_series(rows, y_key="revenue", x_key="period"):
        xs, ys = [], []
        for r in rows[:12]:
            if not isinstance(r, dict):
                continue
            xv = r.get(x_key) or r.get("date") or "-"
            yv = r.get(y_key) or r.get("totalRevenue") or r.get("operatingIncome") or r.get("netIncome")
            try:
                yv = float(yv)
            except Exception:
                continue
            xs.append(str(xv)); ys.append(yv)
        return xs, ys

    news_rows = [["날짜","제목","링크"]]
    if news:
        for it in news[:10]:
            dt = str(it.get("published_at") or it.get("date") or "")[:10]
            news_rows.append([dt, it.get("title") or "-", it.get("url") or it.get("link") or ""])
    else:
        news_rows.append(["-","관련 뉴스 없음",""])

    fin_rows = [["기간","매출","영업이익","순이익"]]
    if fins:
        for f in fins[:8]:
            fin_rows.append([
                f.get("period") or f.get("date") or "-",
                f.get("revenue") or f.get("totalRevenue") or "-",
                f.get("operatingIncome") or f.get("ebit") or "-",
                f.get("netIncome") or "-"
            ])
    else:
        fin_rows.append(["-","-","-","-"])

    tool_plan: List[Dict[str, Any]] = []
    slides = plan.get("slides", []) or []

    for idx, s in enumerate(slides, start=1):
        layout = s.get("layout","title+content")
        if layout == "auto":
            layout = "title+content"

        tool_plan.append({
            "tool":"add_slide",
            "args":{"presentation_id": pid_placeholder, "layout": layout, "title": s.get("title","")}
        })

        for blk in s.get("plan_blocks", []) or []:
            typ = (blk.get("type") or "").lower()

            if typ not in {"bullets","paragraph","table","chart"}:
                typ = "bullets"

            if typ in {"bullets","paragraph"}:
                items = [it for it in (blk.get("summary_points") or blk.get("items") or []) if isinstance(it, str) and it.strip()]
                if not items:
                    items = ["데이터 요약: 현재 섹션에 삽입할 핵심 포인트가 부족합니다."]
                tool_plan.append({
                    "tool":"add_bullet_points",
                    "args":{"presentation_id": pid_placeholder, "items": items}
                })

            elif typ == "table":
                spec = blk.get("table_spec") or {}
                src  = (spec.get("source") or "custom").lower()
                rows_limit = int(spec.get("rows_limit") or 5)
                headers = spec.get("headers")

                if blk.get("rows"):
                    rows = blk["rows"]
                else:
                    if src == "news":
                        rows = news_rows[:rows_limit+1] if len(news_rows) > 1 else [["정보","없음"]]
                    elif src == "financials":
                        rows = fin_rows[:rows_limit+1] if len(fin_rows) > 1 else [["정보","없음"]]
                    else:
                        rows = [["정보","없음"]]

                    if headers and isinstance(headers, list):
                        need_header = True
                        if rows and isinstance(rows[0], list):
                            try:
                                need_header = [str(c) for c in rows[0]] != [str(h) for h in headers]
                            except Exception:
                                need_header = True
                        if need_header:
                            rows = [headers] + rows

                    if rows_limit and len(rows) > (rows_limit + 1):
                        rows = rows[:rows_limit + 1]

                if len(rows) == 1:
                    rows = rows + [["-","데이터 없음"]]

                tool_plan.append({
                    "tool":"add_table",
                    "args":{"presentation_id": pid_placeholder, "rows": rows}
                })

            elif typ == "chart":
                spec = blk.get("chart_spec") or {}
                binding = spec.get("binding") or {}
                tbl = [["항목","값"]] + [[str(k), str(v)] for k, v in binding.items()]
                tool_plan.append({
                    "tool":"add_table",
                    "args":{"presentation_id": pid_placeholder, "rows": tbl}
                })

    state["tool_plan"] = tool_plan
    state["_news_rows_for_hwp"] = news_rows
    state["_fin_rows_for_hwp"] = fin_rows
    print("[PLAN] tool_plan 생성 완료 (섹션 수:", len(slides), ")")
    return state

# =========================
# =========================
from datetime import datetime, timezone, timedelta

def _plan_to_hwp_sections(plan: Dict[str, Any], *, include_chart: bool = False, add_cover: bool = True) -> List[Dict[str, Any]]:
    """
    LLM이 채운 content_plan만 사용해 HWP 섹션 생성.
    - chart 블록은 기본적으로 스킵(include_chart=False).
    - add_cover=True면 제목/작성일/대상 정보를 담은 머릿말 섹션을 맨 앞에 추가.
    """
    sections: List[Dict[str, Any]] = []
    meta = (plan or {}).get("meta") or {}

    if add_cover:
        KST = timezone(timedelta(hours=9))
        created_kst = datetime.now(KST).strftime("%Y-%m-%d")
        title = meta.get("title") or f"{meta.get('ticker', 'Report')} 6개월 보고서"
        ticker = meta.get("ticker") or "-"
        period = meta.get("period") or "6mo"

        cover_blocks = [
            {"type": "paragraph", "text": f"{title}"},
            {"type": "paragraph", "text": f"작성일: {created_kst}"},
            {"type": "paragraph", "text": f"대상: {ticker}  |  기간: {period}"}
        ]
        sections.append({"title": title, "blocks": cover_blocks})

    slides = (plan or {}).get("slides", []) or []
    for s in slides:
        sec = {"title": s.get("title") or "", "blocks": []}
        for blk in (s.get("plan_blocks") or []):
            typ = (blk.get("type") or "").lower()

            if typ in {"bullets", "paragraph"}:
                pts = blk.get("summary_points") or blk.get("items") or []
                text = "\n".join(f"• {p.strip()}" for p in pts if isinstance(p, str) and p.strip())
                if text:
                    sec["blocks"].append({"type": "paragraph", "text": text})

            elif typ == "table":
                rows = blk.get("rows")
                if not rows:
                    spec = blk.get("table_spec") or {}
                    headers = spec.get("headers")
                    rows = [headers, [""] * len(headers)] if headers else [["", ""], ["", ""]]
                sec["blocks"].append({"type": "table", "rows": rows})

            elif typ == "chart":
                if include_chart:
                    spec = blk.get("chart_spec") or {}
                    binding = spec.get("binding") or {}
                    tbl = [["항목", "값"]] + [[str(k), str(v)] for k, v in binding.items()]
                    sec["blocks"].append({"type": "table", "rows": tbl})

        foot = s.get("footnotes")
        if isinstance(foot, str) and foot.strip():
            sec["blocks"].append({"type": "paragraph", "text": f"(출처) {foot.strip()}"})

        sections.append(sec)

    return sections

# =========================
# =========================
HWP_SERVER = StdioServerParameters(
    command="cmd",
    args=["/c", "cd /d E:\\AI_proj\\a2a_mcp_prj\\hwp-mcp && python hwp_mcp_stdio_server.py"]
)
CAND_CREATE  = ("hwp_create", "create_new_document", "create")
CAND_TEXT    = ("hwp_insert_text", "insert_text", "write_text", "type_text")
CAND_PARA    = ("hwp_insert_paragraph", "insert_paragraph")
CAND_TABLE   = ("hwp_insert_table", "insert_table", "create_table")
CAND_FILL    = ("hwp_fill_table_with_data", "fill_table_with_data")
CAND_SAVE    = ("hwp_save", "save_document", "save_as", "save")

def _pick(cands, names):
    for c in cands:
        if c in names:
            return c
    for c in cands:
        for n in names:
            if c in n:
                return n
    return None

async def _build_hwp_async(sections: List[Dict[str, Any]], save_path: str, title: str = "Auto Report"):
    import os, asyncio, time
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    print("[HWP][ASYNC] will save to:", save_path)
    print("[HWP][ASYNC] cwd:", os.getcwd())
    print("[HWP][ASYNC] HWP_PY:", os.environ.get("HWP_PY"))
    print("[HWP][ASYNC] HWP_SRV:", os.environ.get("HWP_SRV"))

    TIMEOUT = 30
    norm_save = os.path.normpath(save_path)

    async with stdio_client(HWP_SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=TIMEOUT)
            tools = await asyncio.wait_for(session.list_tools(), timeout=TIMEOUT)

            names = [t.name for t in tools.tools]
            schemas = {t.name: getattr(t, "inputSchema", None) for t in tools.tools}
            t_create = _pick(CAND_CREATE, names)
            t_text   = _pick(CAND_TEXT, names)
            t_para   = _pick(CAND_PARA, names)
            t_table  = _pick(CAND_TABLE, names)
            t_fill   = _pick(CAND_FILL, names)
            t_save   = _pick(CAND_SAVE, names)
            print("[HWP][ASYNC] save tool schema:", schemas.get(t_save))
            print("[HWP][ASYNC] chosen tools:", {"create": t_create, "text": t_text, "para": t_para, "table": t_table, "fill": t_fill, "save": t_save})
            for k, v in {"create": t_create, "text": t_text, "table": t_table, "fill": t_fill, "save": t_save}.items():
                if not v:
                    raise RuntimeError(f"[HWP] 필수 도구 탐색 실패: {k}")

            await asyncio.wait_for(session.call_tool(t_create, {}), timeout=TIMEOUT)
            await asyncio.wait_for(session.call_tool(t_text, {"text": f"{title}\n\n"}), timeout=TIMEOUT)

            for sec in sections:
                await asyncio.wait_for(session.call_tool(t_text, {"text": f"{sec.get('title', '')}\n"}), timeout=TIMEOUT)
                for b in sec.get("blocks", []):
                    if b.get("type") == "paragraph":
                        await asyncio.wait_for(session.call_tool(t_text, {"text": (b.get("text") or "") + "\n\n"}), timeout=TIMEOUT)
                    elif b.get("type") == "table":
                        rows = b.get("rows") or []
                        if len(rows) == 1:
                            rows = rows + [["-", "-"]]
                        cols = max(1, len(rows[0])) if rows else 2
                        rnum = max(2, len(rows))
                        await asyncio.wait_for(session.call_tool(t_table, {"rows": rnum, "cols": cols}), timeout=TIMEOUT)
                        await asyncio.wait_for(session.call_tool(t_fill, {"data": rows, "has_header": True}), timeout=TIMEOUT)
                        if t_para:
                            await asyncio.wait_for(session.call_tool(t_para, {}), timeout=TIMEOUT)

            print("[HWP][ASYNC] saving document...")
            os.makedirs(os.path.dirname(norm_save), exist_ok=True)

            def _with_ext(path: str, ext: str) -> str:
                import os as _os
                root, _ = _os.path.splitext(path)
                return root + ext

            preferred = _with_ext(norm_save, ".hwpx")
            fallback  = _with_ext(norm_save, ".hwp")

            async def _save_to(path: str):
                print(f"[HWP][ASYNC] try save with path={path}")
                try:
                    r = await asyncio.wait_for(session.call_tool(t_save, {"path": path}), timeout=TIMEOUT)
                    hint = None
                    try:
                        if hasattr(r, "content") and isinstance(r.content, list):
                            for it in r.content:
                                if hasattr(it, "text") and it.text:
                                    txt = it.text
                                    print("[HWP][ASYNC] save tool returned:", txt[:200])
                                    for cand in (path, preferred, fallback):
                                        if cand and os.path.basename(cand) in txt:
                                            hint = cand
                                            break
                    except Exception:
                        pass
                    return r, (hint or path)
                except Exception as e:
                    print(f"[HWP][ASYNC] save attempt failed (path):", e)
                    return None, None

            res, saved_hint = await _save_to(preferred)
            if not res:
                res, saved_hint = await _save_to(fallback)

            targets = [p for p in [saved_hint, preferred, fallback] if p]
            exist_path = None
            deadline = time.time() + 3.0
            while time.time() < deadline and not exist_path:
                for cand in targets:
                    if os.path.exists(cand) and os.path.getsize(cand) > 0:
                        exist_path = cand
                        break
                if not exist_path:
                    time.sleep(0.3)

            if exist_path:
                print("[HWP][ASYNC] file exists after save:", exist_path, os.path.getsize(exist_path), "bytes")
                return exist_path

            print("[HWP][ASYNC][WARN] file not found after save. last tried:", targets)
            return preferred

def build_hwp_via_mcp(sections, out_dir: str, meta_title: str = "Auto Report") -> str:
    import os, re, asyncio, time
    from datetime import datetime
    from pathlib import Path

    print("[HWP] build_hwp_via_mcp 시작")
    if not isinstance(sections, list) or not sections:
        raise RuntimeError("[HWP] sections가 비어있음")

    out_dir = os.path.abspath(out_dir or "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    safe_title = re.sub(r'[\\/:*?"<>|]+', "_", (meta_title or "Auto Report")).strip() or "Auto Report"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f"{safe_title}_{ts}.hwp")

    print(f"[HWP] target out_dir={out_dir}")
    print(f"[HWP] save_path={save_path} (exists_before={os.path.exists(save_path)})")
    print(f"[HWP] sections_count={len(sections)}")

    result_path = asyncio.run(_build_hwp_async(sections, save_path, title=meta_title))

    time.sleep(0.8)

    def _find_recent_in(dirpath: str):
        cands = []
        for pat in ("*.hwpx", "*.hwp"):
            cands += list(Path(dirpath).glob(pat))
        if not cands:
            return None
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        recent = cands[0]
        if time.time() - recent.stat().st_mtime < 180:
            print("[HWP] found recent file:", recent)
            return str(recent)
        return None

    if not result_path or not os.path.exists(result_path):
        alt = _find_recent_in(out_dir)
        if alt: result_path = alt

    if (not result_path) or (not os.path.exists(result_path)):
        alt = _find_recent_in(os.getcwd())
        if alt: result_path = alt

    if (not result_path) or (not os.path.exists(result_path)):
        docs = os.path.join(os.path.expanduser("~"), "Documents")
        scan_dirs = [docs] if os.path.isdir(docs) else []
        for sub in ("Hancom", "Hwp", "한글", "한컴", "HWP"):
            cand = os.path.join(docs, sub)
            if os.path.isdir(cand):
                scan_dirs.append(cand)
        for d in scan_dirs:
            alt = _find_recent_in(d)
            if alt:
                result_path = alt
                break

    if not result_path or not os.path.exists(result_path):
        print("[HWP][ERR] 파일이 안 보임. out_dir listing:")
        for p in Path(out_dir).glob("*"):
            try:
                mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
                print(f"  - {p.name} (mtime={mtime}, {p.stat().st_size} bytes)")
            except Exception:
                print(f"  - {p.name}")
        raise RuntimeError("[HWP] 저장 실패: 파일이 생성되지 않음")

    print("[HWP] 저장 완료 →", result_path)
    return result_path

# =========================
# =========================
def M1_build_hwp_report_node(state: GraphState):
    import os
    print("[M1] HWP 보고서 생성 시작… (차트 표 변환 제거 + 커버 추가)")
    plan = state.get("content_plan") or {}
    print("[M1] content_plan 유무:", bool(plan))

    print("[M1] _plan_to_hwp_sections 호출")
    sections = _plan_to_hwp_sections(plan, include_chart=False, add_cover=True)
    print("[M1] 섹션 수:", len(sections) if isinstance(sections, list) else "형식 이상")

    out_dir = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    print("[M1] out_dir:", out_dir, "exists=", os.path.isdir(out_dir))

    meta = plan.get("meta") or {}
    title = (
    state.get("report_title")
    or state.get("picked_symbol")
    or state.get("ui_ticker")
    or "Auto Report"
        )
    print("[M1] meta title:", title)

    print("[M1] build_hwp_via_mcp 호출")
    save = build_hwp_via_mcp(sections, out_dir=out_dir, meta_title=title)
    print("[M1] build_hwp_via_mcp 반환:", save)

    state["ppt_file"] = save
    state["hwp_saved_path"] = save
    print("[HWP] 저장:", save)
    print("[M1] HWP 보고서 생성 종료")
    return state

# =========================
# =========================
def M1_check_search_branch(state: GraphState):
    if "재검색" in state.get("analysis_result", "") and state.get("retry_count", 0) <= 1:
        return "retry_search"
    return "go_plan"

def content_plan_review_branch(state: GraphState):
    approved = bool(state.get("outline_approved"))
    iter_n = int(state.get("outline_iter", 0))
    if approved:
        return "approved"
    if iter_n >= 5:
        return "approved"
    return "refine"

# =========================
# =========================
workflow = StateGraph(GraphState)
workflow.add_node("M1_analysis", M1_analysis_node)
workflow.add_node("M2_search", M2_search_node)
workflow.add_node("M1_check_search", M1_check_search_node)

workflow.add_node("M1_content_plan", M1_content_plan_node)
workflow.add_node("M1_fill_section_content_with_llm", M1_fill_section_content_with_llm)
workflow.add_node("M3_review_content_plan", M3_review_content_plan_node)
workflow.add_node("M1_refine_content_plan", M1_refine_content_plan_node)
workflow.add_node("M1_translate_plan_to_tool_plan", M1_translate_plan_to_tool_plan_node)

workflow.add_node("M1_build_hwp_report", M1_build_hwp_report_node)

workflow.add_edge(START, "M1_analysis")
workflow.add_edge("M1_analysis", "M2_search")
workflow.add_edge("M2_search", "M1_check_search")
workflow.add_conditional_edges("M1_check_search", M1_check_search_branch,
    {"retry_search": "M2_search", "go_plan": "M1_content_plan"})
workflow.add_edge("M1_content_plan", "M3_review_content_plan")

workflow.add_conditional_edges("M3_review_content_plan", content_plan_review_branch,
    {"approved": "M1_fill_section_content_with_llm", "refine": "M1_refine_content_plan"})
workflow.add_edge("M1_fill_section_content_with_llm", "M1_translate_plan_to_tool_plan")

workflow.add_edge("M1_refine_content_plan", "M3_review_content_plan")

workflow.add_edge("M1_translate_plan_to_tool_plan", "M1_build_hwp_report")
workflow.add_edge("M1_build_hwp_report", END)

# =========================
# =========================
def run_from_ui(user_request: str,
                ui_pdf_path: Optional[str] = None,
                ui_ticker: Optional[str] = None,
                ui_period: str = "6mo",
                ui_interval: str = "1d") -> dict:
    """Streamlit 등 외부 UI에서 값 주입해 호출하는 진입점"""
    init_state: GraphState = {
        "user_request": user_request or "최근 6개월 가격/뉴스/재무를 요약해줘",
        "retry_count": 0,
        "pdf_outputs": [],
        "ui_pdf_path": ui_pdf_path,
        "ui_ticker": ui_ticker,
        "ui_period": ui_period,
        "ui_interval": ui_interval,
    }
    app = workflow.compile()
    final_state = app.invoke(init_state)
    return {
        "analysis_result": final_state.get("analysis_result"),
        "outline_iter": final_state.get("outline_iter"),
        "outline_approved": final_state.get("outline_approved"),
        "ppt_file": final_state.get("ppt_file"),
        "sig": final_state.get("sig"),
        "m2_tools": final_state.get("m2_tools"),
        "errors": final_state.get("errors"),
        "mcp_json": final_state.get("mcp_json"),
    }

# =========================
# main
# =========================
def main():
    pdf_path = "E:/AI_proj/a2a_mcp_prj/data/sample.pdf"
    if os.path.exists(pdf_path):
        try:
            _ = load_pdf_to_vectorstore(pdf_path)
            print("[MAIN] PDF vectorstore loaded.")
        except Exception as e:
            print(f"[MAIN] PDF load skipped: {e}")

    init_state: GraphState = {
        "user_request": "최근 6개월 가격/뉴스/재무를 요약해줘",
        "retry_count": 0,
        "pdf_outputs": []
    }

    app = workflow.compile()
    final_state = app.invoke(init_state)

    print("\n[MAIN] ==== RUN COMPLETE ====")
    print("[MAIN] analysis_result :", final_state.get("analysis_result"))
    print("[MAIN] outline_iter    :", final_state.get("outline_iter"))
    print("[MAIN] outline_approved:", final_state.get("outline_approved"))
    print("[MAIN] ppt_file        :", final_state.get("ppt_file"))
    if final_state.get("errors"):
        print("[MAIN] errors         :", final_state.get("errors"))

if __name__ == "__main__":
    main()