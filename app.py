"""
India Economic Intelligence – Streamlit Dashboard (v2)

This app provides:
- India-focused economic news feed with sentiment
- Personalised headlines using user interests + click history
- Market index snapshot (Nifty, Sensex, global indices) via yfinance
- Macro indicators (CPI, IIP, GDP, Unemployment) via data.gov.in or file upload
- Single-stock deep dive: charts, moving averages, corporate actions
- Auto-generated newsletter summary (editable + downloadable)

Run:
    pip install -r requirements.txt
    streamlit run news_dashboard.py
"""

from __future__ import annotations

import os
import time
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import feedparser
import requests_cache
import streamlit as st
import yfinance as yf
from textblob import TextBlob

# Optional imports
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh

    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


# ====================================================
#                GLOBAL CONFIG & PALETTE
# ====================================================

# Install cache for HTTP calls
requests_cache.install_cache("iecon_cache", expire_after=180)

PALETTE = {
    "bg_top": "#C8D9E6",
    "bg_bottom": "#F5EFEB",
    "navy": "#243447",
    "teal": "#4C8D99",
    "white": "#FFFFFF",
    "pos": "#00C49F",
    "neg": "#FF4C4C",
    "neu": "#F5B041",
}

INDEX_SYMBOLS = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "S&P 500": "^GSPC",
}

# API keys, resource IDs from env / secrets
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "").strip()
CPI_RESOURCE_ID = os.getenv("CPI_RESOURCE_ID", "").strip()
IIP_RESOURCE_ID = os.getenv("IIP_RESOURCE_ID", "").strip()
GDP_RESOURCE_ID = os.getenv("GDP_RESOURCE_ID", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT")) if os.getenv("SMTP_PORT") else None
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()

# cache TTLs for Streamlit
NEWS_TTL = 120
MARKET_TTL = 20
MACRO_TTL = 1800

# Basic Streamlit page config
st.set_page_config(
    page_title="India Economic Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Debug log holder
if "_log" not in st.session_state:
    st.session_state["_log"] = []


def log(msg: str) -> None:
    st.session_state["_log"].append(
        f"{datetime.utcnow().isoformat(timespec='seconds')} | {msg}"
    )


# Global CSS
st.markdown(
    f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(180deg, {PALETTE['bg_top']} 0%, {PALETTE['bg_bottom']} 100%);
}}
h1, h2, h3, h4 {{
    color: {PALETTE['navy']};
    font-weight: 700;
}}
.card {{
    background: {PALETTE['white']};
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}}
.small-muted {{
    color: {PALETTE['teal']};
    font-size: 0.9rem;
}}
.sent-badge {{
    display:inline-block;
    padding:3px 8px;
    border-radius:999px;
    color:white;
    font-weight:600;
    font-size:0.7rem;
}}
.skeleton {{
    height: 16px;
    border-radius: 999px;
    background: linear-gradient(90deg, #dce6f0, #f5f5f5, #dce6f0);
    background-size: 200% 100%;
    animation: shimmer 1.2s infinite linear;
}}
@keyframes shimmer {{
    from {{ background-position: -200% 0; }}
    to   {{ background-position:  200% 0; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ====================================================
#                   GENERIC HELPERS
# ====================================================

def safe_json_get(url: str, params: dict | None = None) -> Dict[str, Any] | None:
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log(f"safe_json_get error: {e} | {url}")
        return None


def format_price(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "N/A"


def fmt_dt(val) -> str:
    if not val:
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(val)


def load_uploaded_table(uploaded_file):
    """Return DataFrame or raw text for CSV/XLSX/PDF files."""
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        if name.endswith(".pdf") and PyPDF2:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for p in reader.pages:
                text += p.extract_text() or ""
            return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    st.warning("Unsupported file format. Use CSV / XLSX / PDF.")
    return None


# ====================================================
#                 SENTIMENT / PERSONALISATION
# ====================================================

def classify_sentiment(text: str) -> Tuple[str, float]:
    try:
        tb = TextBlob(text or "")
        score = round(tb.sentiment.polarity, 3)
        if score >= 0.05:
            return "positive", score
        if score <= -0.05:
            return "negative", score
        return "neutral", score
    except Exception as e:
        log(f"sentiment error: {e}")
        return "neutral", 0.0


def init_personalisation_state():
    if "user_interests" not in st.session_state:
        st.session_state["user_interests"] = ["inflation", "RBI"]
    if "click_counts" not in st.session_state:
        st.session_state["click_counts"] = defaultdict(int)


def register_click(key: str):
    st.session_state["click_counts"][key] += 1


def extract_trending_terms(headlines: List[str], top_n: int = 8) -> List[str]:
    stop_words = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "this",
        "that",
        "will",
        "have",
        "has",
        "into",
        "amid",
        "over",
        "india",
        "govt",
        "government",
    }
    tokens: List[str] = []
    for h in headlines:
        if not h:
            continue
        for w in h.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if len(w) > 3 and w not in stop_words:
                tokens.append(w)
    return [w for w, _ in Counter(tokens).most_common(top_n)]


def personalised_score(article: Dict[str, Any]) -> float:
    """Basic scoring based on interests + trending terms + click history."""
    interests = st.session_state.get("user_interests", [])
    trending = st.session_state.get("trending_terms", [])
    text = (article.get("title", "") + " " + (article.get("summary") or "")).lower()

    score = 0.0
    for it in interests:
        if it.lower() in text:
            score += 2.0
    for t in trending:
        if t in text:
            score += 1.0
    key = article.get("url") or article.get("title")
    score += st.session_state["click_counts"].get(key, 0) * 0.5
    return score


# ====================================================
#                        NEWS
# ====================================================

@st.cache_data(ttl=NEWS_TTL)
def fetch_newsapi(query: str, n: int = 10) -> List[Dict[str, Any]] | None:
    if not NEWSAPI_KEY:
        return None

    params = {
        "q": query,
        "pageSize": n,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    js = safe_json_get("https://newsapi.org/v2/everything", params=params)
    if not js or js.get("status") != "ok":
        return None

    out: List[Dict[str, Any]] = []
    for a in js.get("articles", [])[:n]:
        out.append(
            {
                "title": a.get("title"),
                "summary": a.get("description") or a.get("content") or "",
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
                "publishedAt_raw": a.get("publishedAt"),
            }
        )
    return out


@st.cache_data(ttl=NEWS_TTL)
def fetch_google_news(query: str, n: int = 10, country: str = "IN") -> List[Dict[str, Any]]:
    q = requests.utils.requote_uri(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-{country}&gl={country}&ceid={country}:en"
    feed = feedparser.parse(url)
    out: List[Dict[str, Any]] = []
    for entry in feed.entries[:n]:
        out.append(
            {
                "title": entry.get("title"),
                "summary": entry.get("summary") or "",
                "url": entry.get("link"),
                "source": (entry.get("source") or {}).get("title")
                if entry.get("source")
                else None,
                "publishedAt_raw": entry.get("published") or entry.get("updated") or "",
            }
        )
    return out


def _parse_utc(val):
    try:
        return pd.to_datetime(val, utc=True, errors="coerce")
    except Exception:
        return None


def load_news(query: str, n: int = 10, only_today: bool = True) -> List[Dict[str, Any]]:
    raw = fetch_newsapi(query, n=n) or fetch_google_news(query, n=n)
    if not raw:
        return []

    # unify structure + parse time to UTC
    cleaned: List[Dict[str, Any]] = []
    for a in raw:
        item = {
            "title": a.get("title") or "",
            "summary": a.get("summary") or "",
            "url": a.get("url") or "",
            "source": a.get("source"),
        }
        ts = _parse_utc(a.get("publishedAt_raw"))
        item["publishedAt"] = ts
        cleaned.append(item)

    if only_today:
        now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
        today = now_ist.date()
        filtered = []
        for c in cleaned:
            ts = c.get("publishedAt")
            if ts is None:
                continue
            if ts.tz_convert("Asia/Kolkata").date() == today:
                filtered.append(c)
        cleaned = filtered

    # sentiment + user score
    for c in cleaned:
        text = (c["title"] or "") + ". " + (c["summary"] or "")
        label, score = classify_sentiment(text)
        c["sent_label"] = label
        c["sent_score"] = score

    return cleaned


def render_news_page(headlines_count: int):
    st.subheader("📰 Economic & Policy News (India-focused)")

    query = st.text_input("Topic / search query", value="India economy")
    if "news_filter" not in st.session_state:
        st.session_state["news_filter"] = None

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        if st.button("Only positive"):
            st.session_state["news_filter"] = "positive"
    with filter_col2:
        if st.button("Only negative"):
            st.session_state["news_filter"] = "negative"
    with filter_col3:
        if st.button("Reset filter"):
            st.session_state["news_filter"] = None

    with st.spinner("Loading headlines..."):
        articles = load_news(query, n=headlines_count, only_today=True)

    if not articles:
        st.info("No headlines fetched. You may need NEWSAPI_KEY or a broader query.")
        return

    # trending terms + scores
    st.session_state["trending_terms"] = extract_trending_terms(
        [a["title"] for a in articles], top_n=8
    )
    for a in articles:
        a["_score"] = personalised_score(a)

    # sentiment meter
    avg_sent = float(np.mean([a["sent_score"] for a in articles]))
    if avg_sent >= 0.05:
        label = "😊 Overall mood: Positive"
        color = PALETTE["pos"]
    elif avg_sent <= -0.05:
        label = "😟 Overall mood: Negative"
        color = PALETTE["neg"]
    else:
        label = "😐 Overall mood: Neutral"
        color = PALETTE["neu"]

    st.markdown(
        f"""
        <div class='card' style="border-left: 6px solid {color};">
          <div class='small-muted'>Sentiment meter</div>
          <div style="font-size:0.95rem; font-weight:600;">{label} (avg score {avg_sent:+.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # filter and sort
    if st.session_state["news_filter"]:
        flt = st.session_state["news_filter"]
        articles = [a for a in articles if a["sent_label"] == flt]

    articles_sorted = sorted(
        articles,
        key=lambda x: (x["_score"], x.get("publishedAt") or pd.Timestamp.min),
        reverse=True,
    )

    # layout: left = list, right = trending + personalised picks
    left, right = st.columns([3, 1])

    with right:
        st.markdown("#### 🔥 Trending keywords")
        for t in st.session_state["trending_terms"]:
            st.markdown(f"- {t}")
        st.markdown("---")

        st.markdown("#### 🎯 For you")
        top_for_you = articles_sorted[:4]
        if not top_for_you:
            st.caption("No personalised picks yet.")
        else:
            for a in top_for_you:
                st.markdown(
                    f"- <a href='{a['url']}' target='_blank'>{a['title']}</a>",
                    unsafe_allow_html=True,
                )

    with left:
        if articles_sorted:
            # breaking banner
            top = articles_sorted[0]
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(90deg, {PALETTE['navy']} 0%, {PALETTE['teal']} 100%);
                    color:white; padding:6px 12px; border-radius:10px; margin-bottom:0.8rem;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:0.9rem;
                ">
                   <b>🔔 Breaking:</b>
                   <a href="{top['url']}" target="_blank" style="color:white; text-decoration:none;">
                     {top['title']}
                   </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for idx, a in enumerate(articles_sorted, start=1):
            title = a["title"]
            summary = a["summary"]
            label = a["sent_label"]
            sent_score = a["sent_score"]
            url = a["url"]
            src = a.get("source") or ""
            pub = a.get("publishedAt")

            badge_color = (
                PALETTE["pos"]
                if label == "positive"
                else PALETTE["neg"]
                if label == "negative"
                else PALETTE["neu"]
            )

            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex; justify-content:space-between; gap:0.8rem;">
                    <div style="flex:1;">
                      <a href="{url}" target="_blank" style="text-decoration:none; color:{PALETTE['navy']}; font-weight:600;">
                        {idx}. {title}
                      </a>
                      <div class="small-muted">{src} · {fmt_dt(pub)}</div>
                    </div>
                    <div style="text-align:right;">
                      <span class="sent-badge" style="background:{badge_color};">{label.upper()}</span>
                      <div class="small-muted" style="margin-top:4px;">score {sent_score:+.2f}</div>
                    </div>
                  </div>
                  <div style="margin-top:6px; font-size:0.9rem;">{summary}</div>
                  <div style="margin-top:6px;">
                    <a href="{url}" target="_blank">Read full article →</a>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Mark as read", key=f"read_{idx}"):
                register_click(url or title)
                st.experimental_rerun()

    # Newsletter section
    st.markdown("---")
    st.markdown("### 📝 Auto newsletter (editable)")

    top_for_newsletter = articles_sorted[:3]
    bullets = [
        textwrap.shorten(a["title"], width=140, placeholder="...") for a in top_for_newsletter
    ]
    if not bullets:
        bullets = ["No major headlines available."]

    newsletter_text = "Daily Economic Brief — India\n\n"
    for i, b in enumerate(bullets, start=1):
        newsletter_text += f"{i}. {b}\n"

    nl_edit = st.text_area("Newsletter body", value=newsletter_text, height=200)
    st.download_button(
        "Download as TXT",
        data=nl_edit.encode("utf-8"),
        file_name="economic_brief.txt",
    )


# ====================================================
#                        MARKETS
# ====================================================

@st.cache_data(ttl=MARKET_TTL)
def fetch_index_snapshot() -> Dict[str, Dict[str, float | None]]:
    out = {}
    for name, symbol in INDEX_SYMBOLS.items():
        try:
            df = yf.download(
                symbol, period="2d", interval="5m", progress=False, threads=False
            )
            if df is None or df.empty:
                out[name] = {"last": None, "change_pct": None}
                continue
            df = df[~df.index.duplicated(keep="last")]
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
            change_pct = (last - prev) / prev * 100 if prev else 0.0
            out[name] = {"last": last, "change_pct": change_pct}
        except Exception as e:
            log(f"index snapshot error {name}: {e}")
            out[name] = {"last": None, "change_pct": None}
    return out


def draw_index_cards():
    st.subheader("📈 Market indices snapshot")

    # skeleton placeholders
    cols = st.columns(len(INDEX_SYMBOLS))
    for c in cols:
        c.markdown('<div class="card"><div class="skeleton"></div></div>', unsafe_allow_html=True)

    snap = fetch_index_snapshot()

    cols = st.columns(len(INDEX_SYMBOLS))
    for col, (name, _) in zip(cols, INDEX_SYMBOLS.items()):
        idx_data = snap.get(name, {})
        price = idx_data.get("last")
        pct = idx_data.get("change_pct")

        if price is None or pct is None:
            body = "N/A"
            change_str = ""
            color = PALETTE["neu"]
        else:
            body = format_price(price)
            arrow = "▲" if pct >= 0 else "▼"
            color = PALETTE["pos"] if pct >= 0 else PALETTE["neg"]
            change_str = f"{arrow} {pct:+.2f}%"

        col.markdown(
            f"""
            <div class="card" style="text-align:center;">
              <div style="font-size:0.9rem; font-weight:600; color:{PALETTE['navy']};">{name}</div>
              <div style="font-size:1.3rem; margin-top:4px;">{body}</div>
              <div style="font-size:0.8rem; font-weight:600; color:{color}; margin-top:4px;">{change_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ====================================================
#                        MACRO
# ====================================================

@st.cache_data(ttl=MACRO_TTL)
def fetch_data_gov(resource_id: str, limit: int = 500) -> pd.DataFrame | None:
    if not resource_id or not DATA_GOV_API_KEY:
        return None
    try:
        js = safe_json_get(
            f"https://api.data.gov.in/resource/{resource_id}.json",
            params={"api-key": DATA_GOV_API_KEY, "limit": limit},
        )
        if not js or not js.get("records"):
            return None
        return pd.DataFrame(js["records"])
    except Exception as e:
        log(f"data.gov error {resource_id}: {e}")
        return None


def latest_value_from(df: pd.DataFrame | None) -> Tuple[Any, Any]:
    if df is None or df.empty:
        return None, None
    try:
        cols = list(df.columns)
        date_col = next(
            (c for c in cols if any(x in c.lower() for x in ["date", "month", "year", "quarter"])),
            cols[0],
        )
        val_col = next(
            (
                c
                for c in cols
                if any(
                    x in c.lower()
                    for x in ["value", "index", "cpi", "iip", "gdp", "rate", "percent", "%", "growth"]
                )
            ),
            cols[1] if len(cols) > 1 else cols[0],
        )
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        latest_row = tmp.sort_values(date_col).iloc[-1]
        return latest_row[val_col], latest_row[date_col]
    except Exception as e:
        log(f"latest_value_from error: {e}")
        return None, None


def render_macro_page():
    st.subheader("📊 Macro Indicators – CPI, IIP, GDP, Unemployment")

    with st.expander("Admin: upload fallback macro files (optional)"):
        c1, c2 = st.columns(2)
        with c1:
            cpi_upload = st.file_uploader("CPI file", type=["csv", "xlsx", "pdf"], key="up_cpi")
            iip_upload = st.file_uploader("IIP file", type=["csv", "xlsx", "pdf"], key="up_iip")
        with c2:
            gdp_upload = st.file_uploader("GDP file", type=["csv", "xlsx", "pdf"], key="up_gdp")
            unemp_upload = st.file_uploader("Unemployment file", type=["csv", "xlsx", "pdf"], key="up_unemp")

    # fetch remote data
    cpi_remote = fetch_data_gov(CPI_RESOURCE_ID) if CPI_RESOURCE_ID else None
    iip_remote = fetch_data_gov(IIP_RESOURCE_ID) if IIP_RESOURCE_ID else None
    gdp_remote = fetch_data_gov(GDP_RESOURCE_ID) if GDP_RESOURCE_ID else None

    # read uploads
    cpi_local = load_uploaded_table(cpi_upload) if cpi_upload else None
    iip_local = load_uploaded_table(iip_upload) if iip_upload else None
    gdp_local = load_uploaded_table(gdp_upload) if gdp_upload else None
    unemp_local = load_uploaded_table(unemp_upload) if unemp_upload else None

    # choose priority: remote > uploaded df
    cpi_df = cpi_remote if isinstance(cpi_remote, pd.DataFrame) else (
        cpi_local if isinstance(cpi_local, pd.DataFrame) else None
    )
    iip_df = iip_remote if isinstance(iip_remote, pd.DataFrame) else (
        iip_local if isinstance(iip_local, pd.DataFrame) else None
    )
    gdp_df = gdp_remote if isinstance(gdp_remote, pd.DataFrame) else (
        gdp_local if isinstance(gdp_local, pd.DataFrame) else None
    )

    cpi_val, cpi_date = latest_value_from(cpi_df)
    iip_val, iip_date = latest_value_from(iip_df)
    gdp_val, gdp_date = latest_value_from(gdp_df)

    card_data = [
        {
            "label": "Inflation (CPI, %)",
            "icon": "📊",
            "value": cpi_val,
            "date": cpi_date,
            "key": "CPI",
        },
        {
            "label": "Industrial Output (IIP, index)",
            "icon": "🏭",
            "value": iip_val,
            "date": iip_date,
            "key": "IIP",
        },
        {
            "label": "GDP Growth (Real, %)",
            "icon": "💹",
            "value": gdp_val,
            "date": gdp_date,
            "key": "GDP",
        },
        {
            "label": "Unemployment Rate (%, custom)",
            "icon": "👷",
            "value": None,
            "date": None,
            "key": "UNEMP",
        },
    ]

    st.markdown("#### Snapshot cards")

    cols = st.columns(4)
    for col, cd in zip(cols, card_data):
        val = cd["value"]
        try:
            txt = f"{float(val):.1f}%" if val is not None else "N/A"
        except Exception:
            txt = "N/A"
        date_str = (
            str(pd.to_datetime(cd["date"]).date()) if cd["date"] is not None else "Latest"
        )

        col.markdown(
            f"""
            <div class="card" style="text-align:center; background:linear-gradient(180deg,#082b57,#021426); color:white;">
              <div style="font-size:2rem;">{cd['icon']}</div>
              <div style="font-size:1.6rem; color:#F5B041; font-weight:700;">{txt}</div>
              <div style="font-size:0.9rem; font-weight:600; margin-top:4px;">{cd['label']}</div>
              <div style="font-size:0.75rem; color:#d0d6e0; margin-top:2px;">{date_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Detailed charts")

    tab_cpi, tab_iip, tab_gdp, tab_unemp = st.tabs(["CPI", "IIP", "GDP", "Unemployment"])

    def plot_time_series(df: pd.DataFrame, title_prefix: str):
        if df is None or df.empty:
            st.info("No data available.")
            return
        cols = list(df.columns)
        date_col = next(
            (c for c in cols if any(x in c.lower() for x in ["date", "month", "year", "quarter"])),
            cols[0],
        )
        val_col = next(
            (c for c in cols if any(
                x in c.lower()
                for x in ["value", "index", "rate", "percent", "%", "gdp", "cpi", "iip", "growth"]
            )),
            cols[1] if len(cols) > 1 else cols[0],
        )
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col, val_col]).sort_values(date_col)

        fig = px.line(
            tmp,
            x=date_col,
            y=val_col,
            markers=True,
            title=f"{title_prefix} – time series",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Detected date column: {date_col} | value column: {val_col}")

    with tab_cpi:
        st.write("CPI (Consumer Price Index)")
        plot_time_series(cpi_df, "CPI")

    with tab_iip:
        st.write("IIP (Index of Industrial Production)")
        plot_time_series(iip_df, "IIP")

    with tab_gdp:
        st.write("GDP growth")
        plot_time_series(gdp_df, "GDP")

    with tab_unemp:
        st.write("Unemployment (upload custom data)")
        if isinstance(unemp_local, pd.DataFrame):
            plot_time_series(unemp_local, "Unemployment")
        elif isinstance(unemp_local, str):
            st.text_area("Uploaded PDF preview", unemp_local[:2500])
        else:
            st.info("Upload unemployment data as CSV/XLSX/PDF in admin section.")


# ====================================================
#                    STOCK DEEP DIVE
# ====================================================

def render_stock_page(default_symbol: str):
    st.subheader("💹 Single-stock deep dive")

    symbol = st.text_input("Ticker symbol (e.g. RELIANCE.NS)", value=default_symbol)

    ranges = {
        "1D": ("1d", "5m"),
        "5D": ("5d", "15m"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "3Y": ("3y", "1wk"),
        "5Y": ("5y", "1wk"),
    }

    if "stock_range" not in st.session_state:
        st.session_state["stock_range"] = "1Y"

    st.caption("Choose time range:")
    tabs = st.tabs(list(ranges.keys()))
    for label, tab in zip(ranges.keys(), tabs):
        with tab:
            if st.button(label, key=f"btn_range_{label}"):
                st.session_state["stock_range"] = label
                st.experimental_rerun()

    selected_label = st.session_state["stock_range"]
    period, interval = ranges[selected_label]

    if not symbol:
        st.info("Enter a ticker symbol to view data.")
        return

    with st.spinner(f"Fetching {symbol} data ({selected_label})..."):
        df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty:
        st.error("No data found. Try another symbol or use `.NS` for Indian stocks.")
        return

    df = df.reset_index()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    price = float(latest["Close"])
    prev_price = float(prev["Close"])
    diff = price - prev_price
    pct = (diff / prev_price) * 100 if prev_price else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Price", f"₹{price:,.2f}", f"{diff:+.2f}")
    with col2:
        st.metric("Change (%)", f"{pct:+.2f}%")
    with col3:
        st.metric("Open", f"₹{float(latest['Open']):,.2f}")
    with col4:
        st.metric("High", f"₹{float(latest['High']):,.2f}")
    with col5:
        st.metric("Low", f"₹{float(latest['Low']):,.2f}")

    st.caption(f"Last updated: {latest['Date']}")

    # main price chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close",
        )
    )
    fig.update_layout(
        title=f"{symbol} – price ({selected_label})",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # MAs
    st.markdown("### Moving averages")
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    show20 = st.checkbox("Show 20-day MA", value=True)
    show50 = st.checkbox("Show 50-day MA", value=True)
    show200 = st.checkbox("Show 200-day MA", value=False)

    fig_ma = go.Figure()
    fig_ma.add_trace(
        go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close")
    )
    if show20:
        fig_ma.add_trace(
            go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20")
        )
    if show50:
        fig_ma.add_trace(
            go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50")
        )
    if show200:
        fig_ma.add_trace(
            go.Scatter(x=df["Date"], y=df["MA200"], mode="lines", name="MA200")
        )

    fig_ma.update_layout(
        title=f"{symbol} – moving averages",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    # Corporate actions
    st.markdown("### Corporate actions & news")
    ticker = yf.Ticker(symbol)
    try:
        divs = ticker.dividends
    except Exception:
        divs = None
    try:
        splits = ticker.splits
    except Exception:
        splits = None
    try:
        news = ticker.news or []
    except Exception:
        news = []

    if divs is not None and not divs.empty:
        st.subheader("Dividends")
        st.dataframe(divs.reset_index().tail(5))
    else:
        st.info("No dividend history available.")

    if splits is not None and not splits.empty:
        st.subheader("Splits")
        st.dataframe(splits.reset_index().tail(5))
    else:
        st.info("No stock split history available.")

    if news:
        st.subheader("Recent company news (Yahoo Finance)")
        for item in news[:6]:
            st.markdown(f"- [{item.get('title')}]({item.get('link')})")
    else:
        st.info("No company-specific news available.")


# ====================================================
#                       SIDEBAR
# ====================================================

def render_sidebar():
    init_personalisation_state()

    st.sidebar.title("Settings")

    # Headlines slider
    count = st.sidebar.slider("Headlines to show", 3, 20, value=6)

    # Auto-refresh
    st.sidebar.subheader("Auto-refresh")
    choice = st.sidebar.selectbox(
        "Interval",
        options=["Off", "30 sec", "1 min", "5 min"],
        index=2,
    )
    interval_map = {"Off": 0, "30 sec": 30, "1 min": 60, "5 min": 300}
    seconds = interval_map[choice]
    if HAS_AUTOREFRESH and seconds > 0:
        tick = st_autorefresh(interval=seconds * 1000, key="auto_refresh")
        st.sidebar.caption(f"Auto-refresh ticks: {tick}")
    elif seconds > 0:
        st.sidebar.info(
            "Install `streamlit-autorefresh` package to enable automatic page refresh."
        )

    st.sidebar.subheader("Personalisation")
    interests = st.sidebar.multiselect(
        "Select interests",
        [
            "RBI",
            "inflation",
            "banks",
            "GDP",
            "employment",
            "infrastructure",
            "startups",
            "markets",
        ],
        default=st.session_state.get("user_interests", ["inflation", "RBI"]),
    )
    if st.sidebar.button("Save interests"):
        st.session_state["user_interests"] = interests

    st.sidebar.subheader("Stock symbol")
    default_symbol = st.sidebar.text_input(
        "Default stock for deep dive", value="RELIANCE.NS"
    )

    if st.sidebar.button("Hard refresh (clear cache)"):
        requests_cache.clear()
        st.cache_data.clear()
        st.experimental_rerun()

    return count, default_symbol


# ====================================================
#                        MAIN
# ====================================================

def main():
    headlines_count, default_symbol = render_sidebar()

    st.markdown(
        "<h1>India Economic Intelligence</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small-muted'>News · Markets · Macro · Single-stock analytics</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    draw_index_cards()

    st.markdown("---")

    tabs = st.tabs(["🏠 Overview", "📰 News", "📊 Macro", "💹 Stock"])

    with tabs[0]:
        st.subheader("Overview")
        st.write(
            "- Live economic & market headlines\n"
            "- Key macro indicators for India (CPI, IIP, GDP, Unemployment)\n"
            "- Major stock index snapshot\n"
            "- Use other tabs for detailed views."
        )

    with tabs[1]:
        render_news_page(headlines_count)

    with tabs[2]:
        render_macro_page()

    with tabs[3]:
        render_stock_page(default_symbol)

    # Footer + debug
    st.markdown("---")
    st.markdown(
        f"<div class='small-muted'>Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Developer debug log"):
        for line in st.session_state["_log"][-200:]:
            st.text(line)


if __name__ == "__main__":
    main()
