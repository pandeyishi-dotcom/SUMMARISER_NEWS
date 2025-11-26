"""
India Economic Intelligence Dashboard
-------------------------------------

Full-featured Streamlit app for India-focused economic & market intelligence.

Main Features
-------------
- Modern UI: SkyBlue → Beige gradient, navy headings, teal subtext, white cards
- Shimmer loading skeletons while fetching data
- Auto-refresh (uses `streamlit-autorefresh` if installed, safe fallback text otherwise)
- News:
    - NewsAPI (if `NEWSAPI_KEY` is set) OR Google News RSS fallback
    - Inline sentiment labels via TextBlob
    - Personalized feed using interests + simple click history
- Markets:
    - Major indices snapshot via yfinance (animated tiles)
    - Single-stock deep dive (1D–5Y) with:
        - Price + change metrics (animated)
        - Trend + moving averages
        - Dividends / splits
        - Related news via Yahoo Finance
- Macro:
    - MOSPI / data.gov.in integration for CPI / IIP / GDP (auto micro-charts)
    - CSV / XLSX / PDF upload fallback for CPI / IIP / GDP / Unemployment
- Newsletter:
    - Auto-generated daily brief (3–4 bullets)
    - Editable text area + TXT download
    - Optional SMTP send (if SMTP_* env vars configured)
- Caching & Debug:
    - HTTP caching with requests_cache
    - st.cache_data TTLs for news/markets/macro
    - Debug log expander in the UI

How to run
----------
1. Install dependencies:

   pip install -r requirements.txt

   (Typical libs: streamlit, yfinance, textblob, requests, feedparser,
    requests-cache, python-dotenv, plotly, pandas, numpy, PyPDF2)

2. (Optional) Create a `.env` or Streamlit secrets with:
   - NEWSAPI_KEY
   - DATA_GOV_API_KEY
   - CPI_RESOURCE_ID
   - IIP_RESOURCE_ID
   - GDP_RESOURCE_ID
   - GNEWS_API_KEY (optional, if you want near-real-time news)
   - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS (optional for email)

3. Run:

   streamlit run news_dashboard.py
"""

import os
import time
import textwrap
from datetime import datetime
from collections import Counter, defaultdict
from io import BytesIO

import requests
import feedparser
import requests_cache
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

import streamlit as st

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


# ======================================================
#                PALETTE & GLOBAL CONFIG
# ======================================================

PALETTE = {
    "sky": "#C8D9E6",
    "beige": "#F5EFEB",
    "navy": "#2F4156",
    "teal": "#567C8D",
    "white": "#FFFFFF",
    "pos": "#00C49F",
    "neg": "#FF4C4C",
    "neu": "#F5B041",
}

INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "S&P 500": "^GSPC",
}

# API keys from env (or Streamlit secrets if you want)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "").strip()

CPI_RESOURCE_ID = os.getenv("CPI_RESOURCE_ID", "").strip()
IIP_RESOURCE_ID = os.getenv("IIP_RESOURCE_ID", "").strip()
GDP_RESOURCE_ID = os.getenv("GDP_RESOURCE_ID", "").strip()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "").strip()

# SMTP
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT")) if os.getenv("SMTP_PORT") else None
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()

# Cache for HTTP requests
requests_cache.install_cache("news_cache", expire_after=180)

# Streamlit cache TTLs
NEWS_TTL = 120
MARKET_TTL = 15
MACRO_TTL = 1800

# Debug log
if "_log" not in st.session_state:
    st.session_state["_log"] = []


def log(msg: str) -> None:
    """Append a debug message in session_state."""
    st.session_state["_log"].append(f"{datetime.utcnow().isoformat()} | {msg}")


def format_price(value):
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "N/A"


# ======================================================
#                    UI + CSS
# ======================================================

st.set_page_config(
    page_title="India Economic Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
:root {{
  --sky: {PALETTE['sky']};
  --beige: {PALETTE['beige']};
  --navy: {PALETTE['navy']};
  --teal: {PALETTE['teal']};
  --white: {PALETTE['white']};
  --pos: {PALETTE['pos']};
  --neg: {PALETTE['neg']};
  --neu: {PALETTE['neu']};
}}
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, var(--sky) 0%, var(--beige) 100%);
}}
h1, h2, h3, h4 {{
  color: var(--navy);
  font-weight: 700;
}}
.card {{
  background: var(--white);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}}
.small-muted {{
  color: var(--teal);
  font-size: 0.95em;
}}
.sent-badge {{
  display:inline-block;
  padding:4px 8px;
  border-radius:10px;
  color:white;
  font-weight:600;
  font-size:12px;
}}
.skel {{
  height:16px;
  border-radius:8px;
  background: linear-gradient(90deg, var(--sky), var(--white), var(--sky));
  background-size:200% 100%;
  animation: shimmer 1.2s infinite linear;
}}
@keyframes shimmer {{
  from {{ background-position:-200% 0; }}
  to   {{ background-position: 200% 0; }}
}}
.block-container {{
  padding-top:1rem;
  padding-bottom:1rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

# Extra DataFrame styling
st.markdown(
    """
<style>
.stDataFrame {
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}
</style>
""",
    unsafe_allow_html=True,
)


# ======================================================
#        GENERIC HELPERS (HTTP, FILE, SENTIMENT)
# ======================================================


def safe_json_get(url, params=None, headers=None, timeout=12):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"HTTP error: {e} | {url}")
        return None


def sentiment_label(text: str):
    """Return (label, score) using TextBlob polarity."""
    try:
        tb = TextBlob(text or "")
        score = round(tb.sentiment.polarity, 3)
        if score >= 0.05:
            return "positive", score
        elif score <= -0.05:
            return "negative", score
        else:
            return "neutral", score
    except Exception as e:
        log(f"sentiment error: {e}")
        return "neutral", 0.0


def fmt_dt(val):
    if not val:
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(val)


def load_uploaded_df(uploaded_file):
    """Handles CSV, XLSX, and PDF uploads and returns DataFrame or text."""
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
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    st.warning("Unsupported file format. Use CSV, XLSX or PDF.")
    return None


def read_pdf_bytes(uploaded_file):
    """Small helper to read PDF text from uploaded file."""
    if uploaded_file is None or not PyPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        log(f"PDF read error: {e}")
        return ""


# ======================================================
#                       NEWS
# ======================================================


@st.cache_data(ttl=NEWS_TTL)
def fetch_newsapi(query, n=10):
    """Fetch news via NewsAPI if key is available."""
    if not NEWSAPI_KEY:
        return None

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": n,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    js = safe_json_get(url, params=params)
    if not js or js.get("status") != "ok":
        return None

    out = []
    for a in js.get("articles", [])[:n]:
        out.append(
            {
                "title": a.get("title"),
                "summary": a.get("description") or a.get("content") or "",
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
                "publishedAt": a.get("publishedAt"),
            }
        )
    return out


@st.cache_data(ttl=NEWS_TTL)
def fetch_google_rss(query, n=10, country="IN"):
    """Fallback: Google News RSS search."""
    q = requests.utils.requote_uri(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-{country}&gl={country}&ceid={country}:en"
    feed = feedparser.parse(url)
    out = []
    for entry in feed.entries[:n]:
        out.append(
            {
                "title": entry.get("title"),
                "summary": entry.get("summary") or "",
                "url": entry.get("link"),
                "source": (entry.get("source") or {}).get("title")
                if entry.get("source")
                else None,
                "publishedAt": entry.get("published") or entry.get("published_parsed"),
            }
        )
    return out


def _parse_pub_to_utc(pub):
    try:
        ts = pd.to_datetime(pub, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def fetch_news(query, n=8, only_today=True):
    """
    Unified news fetch: prefers NewsAPI, falls back to Google RSS.
    If only_today=True, filters to IST-date = today.
    """
    raw = fetch_newsapi(query, n=n) if NEWSAPI_KEY else None
    if not raw:
        raw = fetch_google_rss(query, n=n)

    if not raw:
        return []

    cleaned = []
    for a in raw:
        item = {
            "title": a.get("title") or "",
            "summary": a.get("summary") or "",
            "url": a.get("url") or a.get("link") or "",
            "source": a.get("source"),
            "publishedAt_raw": a.get("publishedAt") or a.get("published") or "",
        }
        item["publishedAt"] = _parse_pub_to_utc(item["publishedAt_raw"])
        cleaned.append(item)

    if only_today:
        now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
        today_ist = now_ist.date()
        filtered = []
        for it in cleaned:
            ts = it.get("publishedAt")
            if ts is None:
                continue
            ts_ist = ts.tz_convert("Asia/Kolkata")
            if ts_ist.date() == today_ist:
                filtered.append(it)
        cleaned = filtered

    return cleaned


# Super-recent GNews (optional)
def fetch_latest_news(topic):
    """Fetch near real-time news via GNews.io (if key present in secrets/env)."""
    api_key = st.secrets.get("GNEWS_API_KEY", GNEWS_API_KEY) if hasattr(st, "secrets") else GNEWS_API_KEY
    if not api_key:
        return pd.DataFrame()

    url = (
        f"https://gnews.io/api/v4/search?q={topic}"
        f"&lang=en&country=in&max=10&sortby=publishedAt&token={api_key}"
    )
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data.get("articles", []))
        if df.empty:
            return pd.DataFrame()
        return df[["title", "publishedAt", "url"]]
    except Exception as e:
        log(f"GNews fetch error: {e}")
        return pd.DataFrame()


# ======================================================
#                  PERSONALISATION
# ======================================================


def init_personalization():
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = ["rbi", "infrastructure", "inflation"]
    if "click_counts" not in st.session_state:
        st.session_state["click_counts"] = defaultdict(int)


def record_click(aid):
    st.session_state["click_counts"][aid] += 1


def extract_trending_terms(headlines, top_n=8):
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "will",
        "have",
        "has",
        "india",
        "govt",
        "government",
    }
    words = []
    for h in headlines:
        if not h:
            continue
        for w in h.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if len(w) > 3 and w not in stop:
                words.append(w)
    return [w for w, _ in Counter(words).most_common(top_n)]


def score_for_user(article, interests, trending_terms):
    text = (article.get("title", "") + " " + (article.get("summary") or "")).lower()
    score = 0
    for it in interests:
        if it.lower() in text:
            score += 2
    for t in trending_terms:
        if t in text:
            score += 1
    aid = article.get("url") or article.get("title")
    score += st.session_state["click_counts"].get(aid, 0) * 0.5
    return score


# ======================================================
#                      MARKETS
# ======================================================


def build_index_card_html(name, price, pct):
    if price is None or pct is None:
        body = "N/A"
        change_str = ""
        color = PALETTE["neu"]
        arrow = ""
    else:
        color = PALETTE["pos"] if pct >= 0 else PALETTE["neg"]
        arrow = "▲" if pct >= 0 else "▼"
        body = format_price(price)
        change_str = f"{arrow} {pct:+.2f}%"

    return f"""
    <div class='card' style="text-align:left; padding:16px;">
        <div style="font-weight:700; font-size:14px; color:{PALETTE['navy']};">{name}</div>
        <div style="font-size:20px; margin-top:6px;">{body}</div>
        <div style="font-size:13px; font-weight:600; color:{color}; margin-top:4px;">
            {change_str}
        </div>
    </div>
    """


def animate_index_card(name, val, state_key):
    placeholder = st.empty()
    current = val.get("last")
    pct = val.get("pct")
    if current is None:
        placeholder.markdown(
            build_index_card_html(name, None, None), unsafe_allow_html=True
        )
        return

    current = float(current)
    prev = st.session_state.get(state_key, current)

    for p in np.linspace(prev, current, 20):
        html = build_index_card_html(name, p, pct)
        placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(0.02)

    st.session_state[state_key] = current


def animate_metric(label, value, delta, state_key):
    box = st.empty()
    if value is None:
        box.metric(label, "N/A", delta)
        return

    current = float(value)
    prev = st.session_state.get(state_key, current)

    for v in np.linspace(prev, current, 20):
        box.metric(label, f"₹{v:,.2f}", delta)
        time.sleep(0.02)

    st.session_state[state_key] = current


@st.cache_data(ttl=MARKET_TTL)
def fetch_index_snapshot():
    """
    Intraday snapshot of major indices from Yahoo Finance (5m candles).
    """
    out = {}
    for name, sym in INDICES.items():
        try:
            df = yf.download(
                sym,
                period="2d",
                interval="5m",
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                out[name] = {"last": None, "pct": None}
                continue
            df = df[~df.index.duplicated(keep="last")]
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
            pct = (last - prev) / prev * 100 if prev != 0 else 0.0
            out[name] = {"last": last, "pct": pct}
        except Exception as e:
            log(f"index fetch error {name}: {e}")
            out[name] = {"last": None, "pct": None}
    return out


@st.cache_data(ttl=MARKET_TTL)
def fetch_stock_actions(sym):
    try:
        t = yf.Ticker(sym)
        divs = t.dividends if hasattr(t, "dividends") else pd.Series(dtype=float)
        splits = t.splits if hasattr(t, "splits") else pd.Series(dtype=float)
        news = []
        try:
            raw = t.news
            if isinstance(raw, list):
                for item in raw[:8]:
                    news.append(
                        {"title": item.get("title"), "link": item.get("link")}
                    )
        except Exception:
            pass
        return {"dividends": divs, "splits": splits, "news": news}
    except Exception as e:
        log(f"stock actions error {sym}: {e}")
        return {"dividends": pd.Series(dtype=float), "splits": pd.Series(dtype=float), "news": []}


# ======================================================
#                     DATA.GOV.IN
# ======================================================


@st.cache_data(ttl=MACRO_TTL)
def fetch_data_gov_resource(resource_id, limit=1000, api_key=None):
    if not resource_id:
        return None
    key = api_key or DATA_GOV_API_KEY
    if not key:
        return None
    try:
        url = f"https://api.data.gov.in/resource/{resource_id}.json"
        params = {"api-key": key, "limit": limit}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"data.gov fetch error {resource_id}: {e}")
        return None


def latest_summary_from_df(df):
    if df is None or getattr(df, "empty", False):
        return None, None
    try:
        cols = list(df.columns)
        date_col = None
        value_col = None

        for c in cols:
            if any(x in c.lower() for x in ["date", "month", "year", "quarter"]):
                date_col = c
                break

        for c in cols:
            if any(
                x in c.lower()
                for x in [
                    "value",
                    "index",
                    "cpi",
                    "gdp",
                    "iip",
                    "growth",
                    "rate",
                    "percent",
                    "%",
                ]
            ):
                value_col = c
                break

        if date_col is None:
            date_col = cols[0]
        if value_col is None and len(cols) > 1:
            value_col = cols[1]

        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        latest = tmp.sort_values(date_col).iloc[-1]

        return latest[value_col], latest[date_col]
    except Exception as e:
        log(f"latest_summary_from_df error: {e}")
        return None, None


# ======================================================
#                    NEWSLETTER
# ======================================================


def build_newsletter(top_articles, macro_bullets=None):
    macro_bullets = macro_bullets or []
    bullets = []
    if macro_bullets:
        bullets.append(macro_bullets[0])
    for a in top_articles[:3]:
        bullets.append(
            textwrap.shorten(a.get("title", "") or "", width=140, placeholder="...")
        )
    if not bullets:
        bullets = ["No items available."]

    msg = "Daily Economic Brief — Auto-generated\n\n"
    msg += "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))
    return msg


# ======================================================
#                       SIDEBAR
# ======================================================

init_personalization()

st.sidebar.title("Controls & Settings")

headlines_count = st.sidebar.slider("Headlines to show", 3, 20, value=6)

auto_ref = st.sidebar.selectbox(
    "Auto-refresh",
    options=["Off", "1s", "30s", "1m", "5m"],
    index=3,
)

stock_input = st.sidebar.text_input("Single stock (symbol)", value="RELIANCE.NS")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Interests")

interests = st.sidebar.multiselect(
    "Pick interests",
    [
        "RBI",
        "infrastructure",
        "startups",
        "banks",
        "inflation",
        "GDP",
        "employment",
        "policy",
        "stock",
    ],
    default=["inflation", "RBI"],
)

if st.sidebar.button("Save interests"):
    st.session_state["prefs"] = interests

if st.sidebar.button("Manual refresh"):
    requests_cache.clear()
    st.cache_data.clear()
    st.experimental_rerun()

interval_map = {"Off": 0, "1s": 1, "30s": 30, "1m": 60, "5m": 300}
interval_seconds = interval_map.get(auto_ref, 0)

if HAS_AUTOREFRESH and interval_seconds > 0:
    tick = st_autorefresh(interval=interval_seconds * 1000, key="auto_refresh_tick")
    st.sidebar.caption(f"Auto-refresh ticks: {tick}")
elif interval_seconds > 0:
    st.sidebar.info(
        "Auto-refresh selected. Install `streamlit-autorefresh` for real auto reloads."
    )

# GDP animation speed slider (for later)
st.sidebar.markdown("---")
gdp_speed = st.sidebar.slider(
    "GDP animation speed (ms per frame)",
    min_value=100,
    max_value=1200,
    value=400,
    step=50,
)


# ======================================================
#                  HEADER & INDICES
# ======================================================

st.markdown(
    "<h1>📰 India Economic Intelligence — News & Markets</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='small-muted'>Live economic, policy, markets & corporate news — with macro indicators and single-stock deep dive.</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Small shimmer while fetching indices
skel_cols = st.columns(len(INDICES))
for c in skel_cols:
    c.markdown("<div class='card'><div class='skel'></div></div>", unsafe_allow_html=True)

with st.spinner("Fetching market snapshot..."):
    indices = fetch_index_snapshot()

cols_idx = st.columns(len(INDICES))
for i, (name, sym) in enumerate(INDICES.items()):
    val = indices.get(name, {"last": None, "pct": None})
    with cols_idx[i]:
        animate_index_card(name, val, state_key=f"idx_{name}")

st.markdown("---")

# ======================================================
#                   NEWS SECTION
# ======================================================

search_query = "India economy"

with st.spinner("Fetching latest economic news..."):
    news_raw = fetch_news(search_query, n=headlines_count, only_today=True)

if not news_raw:
    st.info("No news found for this query. Try setting NEWSAPI_KEY for better results.")
    news_raw = []

# sentiment + scoring
for a in news_raw:
    text = (a.get("title", "") or "") + ". " + (a.get("summary") or "")
    label, score = sentiment_label(text)
    a["sent_label"] = label
    a["sent_score"] = score

headlines_text = [a.get("title", "") for a in news_raw]
trending = extract_trending_terms(headlines_text, top_n=8)

for a in news_raw:
    a["_user_score"] = score_for_user(a, st.session_state.get("prefs", []), trending)


def parse_pub(a):
    p = a.get("publishedAt")
    if p is None:
        return pd.Timestamp.min
    return p


news_sorted = sorted(
    news_raw, key=lambda x: (x["_user_score"], parse_pub(x)), reverse=True
)

# Overall sentiment meter
if news_raw:
    avg_sent = float(np.mean([a.get("sent_score", 0.0) for a in news_raw]))
    if avg_sent >= 0.05:
        mood = "😊 Overall Mood: Positive"
        bar_color = PALETTE["pos"]
    elif avg_sent <= -0.05:
        mood = "😟 Overall Mood: Negative"
        bar_color = PALETTE["neg"]
    else:
        mood = "😐 Overall Mood: Neutral"
        bar_color = PALETTE["neu"]

    st.markdown(
        f"""
        <div class='card' style="margin-bottom:10px; border-left:6px solid {bar_color}; padding:10px;">
            <div style="font-size:13px; color:{PALETTE['teal']};">Sentiment Meter</div>
            <div style="font-size:15px; font-weight:600;">{mood} (avg score: {avg_sent:+.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Breaking banner
if news_sorted:
    top_headline = news_sorted[0]
    bt = top_headline.get("title", "")
    burl = top_headline.get("url", "")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, {PALETTE['navy']}, {PALETTE['teal']});
            color:white;
            padding:6px 12px;
            border-radius:8px;
            font-size:13px;
            margin-bottom:12px;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        ">
            <strong>🔔 Breaking:</strong>
            <a href="{burl}" target="_blank" style="color:white; text-decoration:none;">
                {bt}
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

main_col, side_col = st.columns([3, 1])

with side_col:
    st.markdown("<div style='font-weight:700; font-size:18px'>Trending</div>", unsafe_allow_html=True)
    if trending:
        for t in trending:
            st.markdown(f"- {t}")
    else:
        st.write("-")

    st.markdown("---")
    st.markdown("<div style='font-weight:700'>For you</div>", unsafe_allow_html=True)

    top_for_you = sorted(news_sorted, key=lambda x: x["_user_score"], reverse=True)[:4]
    if not top_for_you:
        st.write("No personalised picks.")
    else:
        for t in top_for_you:
            st.markdown(
                f"- <a href='{t.get('url')}' target='_blank'>{t.get('title')}</a>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("Quick filters")
    if st.button("Only Positive"):
        st.session_state["_filter"] = "positive"
        st.experimental_rerun()
    if st.button("Only Negative"):
        st.session_state["_filter"] = "negative"
        st.experimental_rerun()
    if st.button("Reset Filter"):
        st.session_state.pop("_filter", None)
        st.experimental_rerun()

flt = st.session_state.get("_filter")
if flt:
    news_sorted = [n for n in news_sorted if n.get("sent_label") == flt]

with main_col:
    st.markdown(
        "<div style='font-weight:700; font-size:20px'>Top headlines</div>",
        unsafe_allow_html=True,
    )
    if not news_sorted:
        st.write("No headlines.")
    for idx, a in enumerate(news_sorted, start=1):
        title = a.get("title", "")
        summary = a.get("summary", "")
        url = a.get("url", "")
        src = a.get("source", "")
        pub = a.get("publishedAt") or ""
        label = a.get("sent_label")
        sscore = a.get("sent_score")
        col = (
            PALETTE["pos"]
            if label == "positive"
            else PALETTE["neg"]
            if label == "negative"
            else PALETTE["neu"]
        )
        badge = f"<span class='sent-badge' style='background:{col}'>{label.upper()}</span>"
        st.markdown(
            f"""
            <div class='card'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='flex:1'>
                  <a href="{url}" target='_blank' style='text-decoration:none; color:{PALETTE['navy']}; font-weight:700'>{idx}. {title}</a>
                  <div class='small-muted'>{src} · {fmt_dt(pub)}</div>
                </div>
                <div style='text-align:right; margin-left:12px;'>
                  {badge}
                  <div style='color:{PALETTE['teal']}; font-size:12px; margin-top:6px;'>Score: {sscore:+.2f}</div>
                </div>
              </div>
              <div style='margin-top:8px; color:#222'>{summary}</div>
              <div style='margin-top:8px'><a href="{url}" target='_blank'>Read full article →</a></div>
            </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Mark read", key=f"mark_{idx}"):
            record_click(a.get("url") or title)
            st.experimental_rerun()

st.markdown("---")

# ======================================================
#               NEWSLETTER GENERATION
# ======================================================

st.markdown("### 📝 Auto Newsletter — Editable Brief")

macro_bullets = []
if CPI_RESOURCE_ID and DATA_GOV_API_KEY:
    cpi_js = fetch_data_gov_resource(CPI_RESOURCE_ID, limit=10)
    if cpi_js and cpi_js.get("records"):
        df_cpi_tmp = pd.DataFrame(cpi_js["records"])
        dv, dtv = latest_summary_from_df(df_cpi_tmp)
        if dv is not None and dtv is not None:
            macro_bullets.append(f"CPI latest index: {dv} (as of {dtv.date()})")

newsletter_seed = top_for_you if news_sorted else news_raw
nl_text = build_newsletter(newsletter_seed, macro_bullets)
nl_area = st.text_area("Newsletter (editable)", value=nl_text, height=220)

st.download_button(
    "Download newsletter (TXT)",
    data=nl_area.encode("utf-8"),
    file_name="economic_brief.txt",
)

if SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS:
    st.markdown("#### Send newsletter via SMTP (optional)")
    to_addr = st.text_input("To (comma separated)")
    if st.button("Send newsletter"):
        import smtplib
        import ssl
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "Daily Economic Brief"
        msg["From"] = SMTP_USER
        msg["To"] = [a.strip() for a in to_addr.split(",") if a.strip()]
        msg.set_content(nl_area)
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
            st.success("✅ Newsletter sent successfully")
        except Exception as e:
            st.error(f"Failed to send email: {e}")
else:
    st.info("SMTP not configured. Set SMTP_* env vars to enable email send.")

# ======================================================
#        MACRO CARDS + DETAILED DASHBOARD
# ======================================================

st.markdown("---")
st.markdown(
    "<h2>📊 India’s Macro Indicators</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='small-muted'>Click a card to open a detailed panel with charts and related news.</div>",
    unsafe_allow_html=True,
)

# Admin upload section
with st.expander("⚙️ Admin: Upload macro data files (CPI/IIP/GDP/Unemployment)"):
    st.info(
        "Upload CSV/XLSX/PDF files as fallback when data.gov.in is not configured or unavailable."
    )
    cpi_upload = st.file_uploader("CPI file", type=["csv", "xlsx", "pdf"])
    iip_upload = st.file_uploader("IIP file", type=["csv", "xlsx", "pdf"])
    gdp_upload = st.file_uploader("GDP file", type=["csv", "xlsx", "pdf"])
    unemp_upload = st.file_uploader("Unemployment file", type=["csv", "xlsx", "pdf"])

# Load uploaded dataframes
cpi_df_up = load_uploaded_df(cpi_upload) if cpi_upload is not None else None
iip_df_up = load_uploaded_df(iip_upload) if iip_upload is not None else None
gdp_df_up = load_uploaded_df(gdp_upload) if gdp_upload is not None else None
unemp_df_up = load_uploaded_df(unemp_upload) if unemp_upload is not None else None

# Fetch from data.gov
cpi_data_gov = None
iip_data_gov = None
gdp_data_gov = None

try:
    if CPI_RESOURCE_ID and DATA_GOV_API_KEY:
        js = fetch_data_gov_resource(CPI_RESOURCE_ID, limit=200)
        if js and js.get("records"):
            cpi_data_gov = pd.DataFrame(js["records"])
    if IIP_RESOURCE_ID and DATA_GOV_API_KEY:
        js = fetch_data_gov_resource(IIP_RESOURCE_ID, limit=200)
        if js and js.get("records"):
            iip_data_gov = pd.DataFrame(js["records"])
    if GDP_RESOURCE_ID and DATA_GOV_API_KEY:
        js = fetch_data_gov_resource(GDP_RESOURCE_ID, limit=200)
        if js and js.get("records"):
            gdp_data_gov = pd.DataFrame(js["records"])
except Exception as e:
    log(f"macro fetch error: {e}")

cpi_val, cpi_date = latest_summary_from_df(cpi_data_gov or (cpi_df_up if isinstance(cpi_df_up, pd.DataFrame) else None))
iip_val, iip_date = latest_summary_from_df(iip_data_gov or (iip_df_up if isinstance(iip_df_up, pd.DataFrame) else None))
gdp_val, gdp_date = latest_summary_from_df(gdp_data_gov or (gdp_df_up if isinstance(gdp_df_up, pd.DataFrame) else None))

cards = [
    {
        "label": "Index of Industrial Production",
        "short": "IIP",
        "icon": "🏭",
        "key": "iip",
        "val": iip_val or 4.0,
        "date": str(iip_date.date()) if iip_date is not None else "Latest",
    },
    {
        "label": "Inflation Rate (CPI based)",
        "short": "CPI",
        "icon": "📊",
        "key": "cpi",
        "val": cpi_val or 5.5,
        "date": str(cpi_date.date()) if cpi_date is not None else "Latest",
    },
    {
        "label": "GDP Growth (Real)",
        "short": "GDP",
        "icon": "💹",
        "key": "gdp",
        "val": gdp_val or 7.8,
        "date": str(gdp_date.date()) if gdp_date is not None else "Latest",
    },
    {
        "label": "Unemployment Rate",
        "short": "UNEMP",
        "icon": "👷",
        "key": "unemp",
        "val": 5.2,
        "date": "Latest",
    },
]

cols_cards = st.columns(4, gap="large")

for col, card in zip(cols_cards, cards):
    icon = card["icon"]
    label = card["label"]
    key = card["key"]
    val = card["val"]
    dt = card["date"]

    try:
        display_val = f"{float(val):.1f}%"
    except Exception:
        display_val = "N/A"

    html_card = f"""
    <div style='
        background: linear-gradient(180deg, #052e6f 0%, #021c47 100%);
        color: white;
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 0px 12px rgba(255,255,255,0.1);
        transition: transform 0.2s ease-in-out;
    ' onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1.00)'">
        <div style='font-size: 48px; margin-bottom: 8px;'>{icon}</div>
        <div style='font-size: 32px; color: #FFA500; font-weight: 700;'>{display_val}</div>
        <div style='font-size: 16px; font-weight: 600; margin-top: 5px;'>{label}</div>
        <div style='font-size: 13px; color: #ccc; margin-top: 3px;'>{dt}</div>
    </div>
    """
    col.markdown(html_card, unsafe_allow_html=True)

    if col.button(f"Open {card['short']} details", key=f"btn_{key}"):
        st.session_state["macro_panel"] = key

st.markdown("---")

if "macro_panel" not in st.session_state:
    st.session_state["macro_panel"] = None


def show_press_and_news(keyword, resource_id=None, uploaded=None, nnews=6):
    st.markdown("### ⚖️ Official / press releases")
    if resource_id and DATA_GOV_API_KEY:
        js = fetch_data_gov_resource(resource_id, limit=6)
        if js and js.get("records"):
            df = pd.DataFrame(js["records"])
            st.dataframe(df.head(6))
        else:
            st.info("No recent records from data.gov.in")
    elif isinstance(uploaded, pd.DataFrame):
        st.dataframe(uploaded.head(6))
    elif isinstance(uploaded, str):
        st.text_area("Uploaded PDF extract (preview)", uploaded[:2000])
    else:
        st.info("No official data. Use admin upload as fallback.")

    st.markdown("#### 🗞️ Related news")
    rel = fetch_news(keyword, n=nnews, only_today=False)
    if not rel:
        st.info("No related news found.")
        return
    for a in rel:
        t = a.get("title") or ""
        s = a.get("summary") or ""
        label, score = sentiment_label(t + " " + s)
        color = (
            PALETTE["pos"]
            if label == "positive"
            else PALETTE["neg"]
            if label == "negative"
            else PALETTE["neu"]
        )
        st.markdown(
            f"📰 **[{t}]({a.get('url')})** — "
            f"<span style='color:{color}; font-weight:700'>{label.upper()}</span> ({score:+.2f})",
            unsafe_allow_html=True,
        )
        if s and len(s) < 250:
            st.caption(s)


def render_macro_detail():
    panel = st.session_state.get("macro_panel")
    if not panel:
        return

    st.button(
        "← Back to overview",
        key="macro_back",
        on_click=lambda: st.session_state.update({"macro_panel": None}),
    )
    st.markdown(
        f"<h3>Detailed macro dashboard — {panel.upper()}</h3>",
        unsafe_allow_html=True,
    )

    sections = ["gdp", "cpi", "iip", "unemp"]

    for sec in sections:
        with st.expander(sec.upper(), expanded=(sec == panel)):
            left, right = st.columns([2, 1])

            if sec == "cpi":
                df = cpi_data_gov if cpi_data_gov is not None else (
                    cpi_df_up if isinstance(cpi_df_up, pd.DataFrame) else None
                )
                up_raw = cpi_df_up
                resource = CPI_RESOURCE_ID
                keyword = "CPI inflation India"
            elif sec == "iip":
                df = iip_data_gov if iip_data_gov is not None else (
                    iip_df_up if isinstance(iip_df_up, pd.DataFrame) else None
                )
                up_raw = iip_df_up
                resource = IIP_RESOURCE_ID
                keyword = "Index of Industrial Production India"
            elif sec == "gdp":
                df = gdp_data_gov if gdp_data_gov is not None else (
                    gdp_df_up if isinstance(gdp_df_up, pd.DataFrame) else None
                )
                up_raw = gdp_df_up
                resource = GDP_RESOURCE_ID
                keyword = "GDP India"
            else:
                df = unemp_df_up if isinstance(unemp_df_up, pd.DataFrame) else None
                up_raw = unemp_df_up
                resource = None
                keyword = "Unemployment India"

            with left:
                st.markdown("#### Trend over time")
                if df is not None and not df.empty:
                    try:
                        cols = list(df.columns)
                        date_col = next(
                            (c for c in cols if any(x in c.lower() for x in ["date", "month", "year", "quarter"])),
                            None,
                        )
                        value_col = next(
                            (c for c in cols if any(
                                x in c.lower()
                                for x in ["value", "index", "gdp", "cpi", "iip", "growth", "rate", "percent", "%"]
                            )),
                            None,
                        )
                        if date_col and value_col:
                            tmp = df.copy()
                            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                            tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)

                            if sec == "gdp":
                                # small animation for GDP
                                x = tmp[date_col].astype(str).tolist()
                                y = pd.to_numeric(
                                    tmp[value_col]
                                    .astype(str)
                                    .str.replace("%", "")
                                    .str.replace(",", ""),
                                    errors="coerce",
                                ).tolist()

                                fig = go.Figure(
                                    data=[go.Scatter(x=[], y=[], mode="lines+markers")]
                                )
                                frames = []
                                for k in range(1, len(x) + 1):
                                    frames.append(
                                        go.Frame(
                                            data=[go.Scatter(x=x[:k], y=y[:k], mode="lines+markers")],
                                            name=f"f{k}",
                                        )
                                    )
                                fig.frames = frames
                                fig.update_layout(
                                    title="Quarter-wise Real GDP Growth (%) — animated",
                                    xaxis_title="Period",
                                    yaxis_title="Growth (%)",
                                    updatemenus=[
                                        {
                                            "type": "buttons",
                                            "buttons": [
                                                {
                                                    "label": "Play",
                                                    "method": "animate",
                                                    "args": [
                                                        None,
                                                        {
                                                            "frame": {
                                                                "duration": gdp_speed,
                                                                "redraw": True,
                                                            },
                                                            "fromcurrent": True,
                                                        },
                                                    ],
                                                },
                                                {
                                                    "label": "Pause",
                                                    "method": "animate",
                                                    "args": [
                                                        [None],
                                                        {
                                                            "frame": {"duration": 0},
                                                            "mode": "immediate",
                                                        },
                                                    ],
                                                },
                                            ],
                                        }
                                    ],
                                    height=420,
                                    template="plotly_white",
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig_tr = px.line(
                                    tmp,
                                    x=date_col,
                                    y=value_col,
                                    title=f"{sec.upper()} trend",
                                    markers=True,
                                )
                                st.plotly_chart(fig_tr, use_container_width=True)
                        else:
                            st.info("Could not auto-detect date/value columns.")
                    except Exception as e:
                        st.warning(f"Trend plot error: {e}")
                else:
                    st.info("No data for this indicator yet. Upload a CSV/XLSX/PDF in admin section.")

            with right:
                show_press_and_news(keyword, resource_id=resource, uploaded=up_raw)


render_macro_detail()

# ======================================================
#            SINGLE STOCK DEEP DIVE SECTION
# ======================================================

st.markdown("---")
st.markdown("## 💹 Single Stock Deep Dive")

st.markdown("Enter ticker in sidebar (e.g. `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`).")

if stock_input:
    st.markdown("### 📅 Select Time Range")
    tab_labels = ["1D", "3M", "6M", "1Y", "2Y", "3Y", "5Y"]
    period_map = {
        "1D": ("1d", "5m"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "2Y": ("2y", "1wk"),
        "3Y": ("3y", "1wk"),
        "5Y": ("5y", "1wk"),
    }

    if "selected_period" not in st.session_state:
        st.session_state["selected_period"] = "1Y"

    tabs = st.tabs(tab_labels)
    for label, tab in zip(tab_labels, tabs):
        with tab:
            if st.button(f"Select {label}", key=f"stock_period_{label}"):
                st.session_state["selected_period"] = label
                st.experimental_rerun()

    selected_label = st.session_state["selected_period"]
    period, interval = period_map[selected_label]

    with st.spinner(f"Fetching {stock_input} ({selected_label})..."):
        data = yf.download(stock_input, period=period, interval=interval, progress=False)

    if data.empty:
        st.warning("No data found. Try another symbol or add `.NS` for Indian stocks.")
    else:
        data = data.reset_index()
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest

        current_price = float(latest["Close"])
        prev_price = float(prev["Close"])
        change_val = current_price - prev_price
        change_pct = (change_val / prev_price) * 100 if prev_price else 0.0

        open_price = float(latest["Open"])
        high_price = float(latest["High"])
        low_price = float(latest["Low"])
        volume = int(latest["Volume"])

        color = "green" if change_val > 0 else "red" if change_val < 0 else "gray"
        sentiment = (
            "Bullish 📈" if change_val > 0 else "Bearish 📉" if change_val < 0 else "Neutral ⚖️"
        )

        st.markdown(f"### {stock_input} — Current Snapshot")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            animate_metric(
                "Price",
                current_price,
                f"{change_val:+.2f}",
                state_key=f"stock_price_{stock_input}",
            )
        c2.metric("Change (%)", f"{change_pct:+.2f}%")
        c3.metric("Open", f"₹{open_price:,.2f}")
        c4.metric("High", f"₹{high_price:,.2f}")
        c5.metric("Low", f"₹{low_price:,.2f}")
        c6.metric("Volume", f"{volume:,}")

        st.caption(f"🕒 Last updated: {latest['Date']} | Sentiment: {sentiment}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Close"],
                mode="lines",
                name="Price",
                line=dict(color=color, width=2),
            )
        )
        fig.update_layout(
            title=f"{stock_input} — {selected_label} price trend",
            yaxis_title="Price (₹)",
            xaxis_title="Date",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Moving Averages")
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["MA200"] = data["Close"].rolling(window=200).mean()

        show_ma20 = st.checkbox("Show MA20 (short-term)", value=True)
        show_ma50 = st.checkbox("Show MA50 (medium-term)", value=True)
        show_ma200 = st.checkbox("Show MA200 (long-term)", value=False)

        fig_ma = go.Figure()
        fig_ma.add_trace(
            go.Scatter(
                x=data["Date"], y=data["Close"], mode="lines", name="Price", line=dict(width=2)
            )
        )
        if show_ma20:
            fig_ma.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["MA20"],
                    mode="lines",
                    name="MA20",
                    line=dict(width=1.8, dash="dot"),
                )
            )
        if show_ma50:
            fig_ma.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["MA50"],
                    mode="lines",
                    name="MA50",
                    line=dict(width=1.8, dash="dot"),
                )
            )
        if show_ma200:
            fig_ma.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["MA200"],
                    mode="lines",
                    name="MA200",
                    line=dict(width=1.8, dash="dot"),
                )
            )
        fig_ma.update_layout(
            title=f"{stock_input} — Moving averages",
            yaxis_title="Price (₹)",
            xaxis_title="Date",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_ma, use_container_width=True)

        st.markdown("### 🏢 Corporate Actions & Related News")
        sa = fetch_stock_actions(stock_input)
        divs = sa.get("dividends")
        splits = sa.get("splits")
        rel_news = sa.get("news", [])

        if not getattr(divs, "empty", True):
            st.subheader("Dividends")
            st.dataframe(divs.reset_index().tail(5))
        else:
            st.info("No dividend data available.")

        if not getattr(splits, "empty", True):
            st.subheader("Stock splits")
            st.dataframe(splits.reset_index().tail(5))
        else:
            st.info("No split data available.")

        if rel_news:
            st.subheader("Related company news")
            for n in rel_news:
                st.markdown(f"- [{n.get('title')}]({n.get('link')})")
        else:
            st.info("No related news available via Yahoo Finance.")

# ======================================================
#                     FOOTER
# ======================================================

st.markdown("---")
st.markdown(
    f"<div style='color:{PALETTE['teal']}'>Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>",
    unsafe_allow_html=True,
)

with st.expander("Show internal debug log"):
    for r in st.session_state["_log"][-200:]:
        st.text(r)

st.markdown(
    "_Tip: set NEWSAPI_KEY & DATA_GOV_API_KEY for richer data. Use admin upload if APIs are down._"
)
