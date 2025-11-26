"""
India Macro & Markets Monitor (Full Version)

Features:
- Dark neon dashboard for:
    1) 📰 News: Economic & policy feed
        - India / Global mode (NewsAPI or Google News RSS fallback)
        - Sentiment tags (Positive / Negative / Neutral)
        - Impact tags: Inflation, Policy, Markets, Growth, Employment, General
        - Topic chips: RBI, Inflation, Markets, Budget, Global
        - Bookmark system (session-based)
        - Keyword frequency bar chart
    2) 📊 Macro:
        - CPI / IIP / GDP / Unemployment
        - Data from data.gov.in (if env vars set) OR uploaded CSV/XLSX/PDF
        - Latest snapshot cards
        - Time-series chart
        - Change mode: Level / Month-over-Month / Year-over-Year (approx)
        - Indicator comparison: CPI vs IIP vs GDP (normalized to 100)
    3) 💹 Markets:
        - Live index snapshot (Nifty 50, Sensex, S&P 500, NASDAQ)
        - Watchlist: choose from Nifty & Sensex companies + custom tickers
        - Watchlist mini cards (1D change)
        - Correlation heatmap (6M daily returns)
        - Single-stock deep dive:
            - Range (1M–3Y), price chart
            - Moving averages
            - Risk panel: annualised volatility, max drawdown
            - Return distribution histogram

Run:
    pip install -r requirements.txt
    streamlit run news_dashboard.py
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple

import feedparser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import requests_cache
import streamlit as st
import yfinance as yf
from textblob import TextBlob

# Optional extras
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from streamlit_autorefresh import st_autorefresh

    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ------------------------------------------------------------------
#                     GLOBAL CONFIG & THEME
# ------------------------------------------------------------------

requests_cache.install_cache("macro_markets_cache", expire_after=180)

st.set_page_config(
    page_title="India Macro & Markets Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg_top": "#020617",
    "bg_bottom": "#020617",
    "card": "#020617",
    "card_inner": "#0B1220",
    "border": "#1D283A",
    "accent": "#F97316",       # orange
    "accent2": "#22D3EE",      # cyan
    "text": "#F9FAFB",
    "muted": "#9CA3AF",
    "pos": "#22C55E",
    "neg": "#EF4444",
    "neu": "#EAB308",
}

INDEX_SYMBOLS = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
}

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "").strip()
CPI_RESOURCE_ID = os.getenv("CPI_RESOURCE_ID", "").strip()
IIP_RESOURCE_ID = os.getenv("IIP_RESOURCE_ID", "").strip()
GDP_RESOURCE_ID = os.getenv("GDP_RESOURCE_ID", "").strip()

NEWS_TTL = 120
MARKET_TTL = 30
MACRO_TTL = 1800

if "_log" not in st.session_state:
    st.session_state["_log"] = []

if "bookmarks" not in st.session_state:
    st.session_state["bookmarks"] = []

if "news_query" not in st.session_state:
    st.session_state["news_query"] = "India economy"

# --- CSS ---
st.markdown(
    f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: radial-gradient(circle at top, #020617 0, #020617 40%, #020617 100%);
    color: {COLORS['text']};
}}
h1, h2, h3, h4 {{
    color: {COLORS['text']};
    font-weight: 700;
}}
.card {{
    background: {COLORS['card_inner']};
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.9rem;
    border: 1px solid {COLORS['border']};
    box-shadow: 0 16px 32px rgba(15,23,42,0.7);
}}
.small-muted {{
    color: {COLORS['muted']};
    font-size: 0.85rem;
}}
.chip {{
    display:inline-block;
    padding:2px 8px;
    border-radius:999px;
    border:1px solid {COLORS['border']};
    font-size:0.7rem;
    margin-right:4px;
    margin-bottom:4px;
}}
.sent-pill {{
    display:inline-block;
    padding:2px 10px;
    border-radius:999px;
    font-size:0.7rem;
    font-weight:600;
    color:black;
}}
.skeleton {{
    height: 16px;
    border-radius: 999px;
    background: linear-gradient(90deg, #1f2937, #111827, #1f2937);
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

# ------------------------------------------------------------------
#                     NIFTY / SENSEX TICKERS
# ------------------------------------------------------------------

PRESET_TICKERS_FALLBACK = {
    "Reliance Industries (NIFTY, SENSEX)": "RELIANCE.NS",
    "HDFC Bank (NIFTY, SENSEX)": "HDFCBANK.NS",
    "ICICI Bank (NIFTY, SENSEX)": "ICICIBANK.NS",
    "State Bank of India (NIFTY, SENSEX)": "SBIN.NS",
    "Infosys (NIFTY)": "INFY.NS",
    "TCS (NIFTY, SENSEX)": "TCS.NS",
    "Bharti Airtel (NIFTY, SENSEX)": "BHARTIARTL.NS",
    "ITC (NIFTY, SENSEX)": "ITC.NS",
    "Larsen & Toubro (NIFTY, SENSEX)": "LT.NS",
    "Axis Bank (NIFTY, SENSEX)": "AXISBANK.NS",
    "Kotak Mahindra Bank (NIFTY, SENSEX)": "KOTAKBANK.NS",
    "Hindustan Unilever (NIFTY, SENSEX)": "HINDUNILVR.NS",
    "Maruti Suzuki (NIFTY, SENSEX)": "MARUTI.NS",
    "Mahindra & Mahindra (NIFTY, SENSEX)": "M&M.NS",
    "UltraTech Cement (NIFTY, SENSEX)": "ULTRACEMCO.NS",
    "Sun Pharma (NIFTY, SENSEX)": "SUNPHARMA.NS",
    "Power Grid (NIFTY, SENSEX)": "POWERGRID.NS",
    "NTPC (NIFTY, SENSEX)": "NTPC.NS",
    "Asian Paints (NIFTY, SENSEX)": "ASIANPAINT.NS",
    "Bajaj Finance (NIFTY, SENSEX)": "BAJFINANCE.NS",
    "Titan (NIFTY, SENSEX)": "TITAN.NS",
    "Nestle India (NIFTY, SENSEX)": "NESTLEIND.NS",
    "JSW Steel (NIFTY, SENSEX)": "JSWSTEEL.NS",
    "Tata Motors (NIFTY)": "TATAMOTORS.NS",
    "Tata Steel (NIFTY)": "TATASTEEL.NS",
    "Wipro (NIFTY)": "WIPRO.NS",
    "HCL Tech (NIFTY)": "HCLTECH.NS",
    "Adani Ports (NIFTY)": "ADANIPORTS.NS",
    "Adani Enterprises (NIFTY)": "ADANIENT.NS",
    "Coal India (NIFTY)": "COALINDIA.NS",
    "ONGC (NIFTY)": "ONGC.NS",
    "Britannia (NIFTY)": "BRITANNIA.NS",
    "SBI Life (NIFTY)": "SBILIFE.NS",
    "HDFC Life (NIFTY)": "HDFCLIFE.NS",
    "Grasim (NIFTY)": "GRASIM.NS",
    "Tech Mahindra (NIFTY)": "TECHM.NS",
    "Tata Consumer (NIFTY)": "TATACONSUM.NS",
    "Bajaj Finserv (NIFTY)": "BAJAJFINSV.NS",
}

TICKERS_CSV_PATH = "indian_index_tickers.csv"


@st.cache_data
def load_preset_tickers() -> dict:
    """
    Load full Nifty/Sensex list from CSV if available, else fallback dict.

    CSV format:
        name,symbol,index
        Reliance Industries,RELIANCE.NS,NIFTY;SENSEX
        HDFC Bank,HDFCBANK.NS,NIFTY;SENSEX
        ...
    """
    if os.path.exists(TICKERS_CSV_PATH):
        try:
            df = pd.read_csv(TICKERS_CSV_PATH)
            df = df.dropna(subset=["name", "symbol"])
            mapping: Dict[str, str] = {}
            for _, row in df.iterrows():
                label = str(row["name"]).strip()
                idx = str(row.get("index", "")).strip()
                if idx:
                    label = f"{label} ({idx})"
                mapping[label] = str(row["symbol"]).strip()
            return mapping
        except Exception as e:
            st.warning(f"Could not read {TICKERS_CSV_PATH}, using fallback list. Error: {e}")
            return PRESET_TICKERS_FALLBACK
    return PRESET_TICKERS_FALLBACK


# ------------------------------------------------------------------
#                         GENERIC HELPERS
# ------------------------------------------------------------------

def log(msg: str) -> None:
    st.session_state["_log"].append(
        f"{datetime.utcnow().isoformat(timespec='seconds')} | {msg}"
    )


def safe_json_get(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log(f"HTTP error: {e} | {url}")
        return None


def fmt_ts(val) -> str:
    if not val:
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(val)


def read_table_or_text(uploaded):
    """Return DataFrame or string for CSV/XLSX/PDF; None otherwise."""
    if uploaded is None:
        return None

    name = uploaded.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith(".xlsx"):
            return pd.read_excel(uploaded)
        if name.endswith(".pdf") and PyPDF2:
            reader = PyPDF2.PdfReader(uploaded)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None

    st.warning("Unsupported file format. Use CSV / XLSX / PDF.")
    return None


# ------------------------------------------------------------------
#                     SENTIMENT & IMPACT TAGS
# ------------------------------------------------------------------

def get_sentiment(text: str) -> Tuple[str, float]:
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


IMPACT_KEYWORDS = {
    "Inflation": ["inflation", "cpi", "price rise", "wpi"],
    "Policy": ["rbi", "repo rate", "policy", "budget", "regulation"],
    "Markets": ["sensex", "nifty", "stocks", "equity", "markets"],
    "Growth": ["gdp", "growth", "expansion", "output"],
    "Employment": ["jobs", "unemployment", "employment"],
}


def detect_impact_tags(text: str) -> List[str]:
    text_low = (text or "").lower()
    tags = []
    for label, keys in IMPACT_KEYWORDS.items():
        if any(k in text_low for k in keys):
            tags.append(label)
    return tags or ["General"]


# ------------------------------------------------------------------
#                           NEWS
# ------------------------------------------------------------------

@st.cache_data(ttl=NEWS_TTL)
def fetch_newsapi(query: str, n: int, lang: str) -> List[Dict[str, Any]] | None:
    if not NEWSAPI_KEY:
        return None
    params = {
        "q": query,
        "language": lang,
        "pageSize": n,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    js = safe_json_get("https://newsapi.org/v2/everything", params)
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
                "published_raw": a.get("publishedAt"),
            }
        )
    return out


@st.cache_data(ttl=NEWS_TTL)
def fetch_google_news(query: str, n: int, country: str) -> List[Dict[str, Any]]:
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
                "published_raw": entry.get("published") or entry.get("updated") or "",
            }
        )
    return out


def _parse_utc(val):
    try:
        return pd.to_datetime(val, utc=True, errors="coerce")
    except Exception:
        return None


def load_headlines(
    query: str,
    limit: int,
    country_mode: str = "India",
) -> List[Dict[str, Any]]:
    """Get news from NewsAPI if available, else Google News RSS."""
    if country_mode == "India":
        lang = "en"
        country = "IN"
    else:
        lang = "en"
        country = "US"

    raw = fetch_newsapi(query, n=limit, lang=lang) or fetch_google_news(
        query, n=limit, country=country
    )
    if not raw:
        return []

    cleaned: List[Dict[str, Any]] = []
    for r in raw:
        ts = _parse_utc(r.get("published_raw"))
        title = r.get("title") or ""
        summary = r.get("summary") or ""
        sent_label, sent_score = get_sentiment(title + " " + summary)
        impact_tags = detect_impact_tags(title + " " + summary)

        cleaned.append(
            {
                "title": title,
                "summary": summary,
                "url": r.get("url") or "",
                "source": r.get("source"),
                "published": ts,
                "sent_label": sent_label,
                "sent_score": sent_score,
                "impact": impact_tags,
            }
        )
    return cleaned


def news_tab(headline_count: int):
    st.subheader("📰 Economic & policy news")

    # Quick topic chips
    st.markdown("**Quick topics:**", unsafe_allow_html=True)
    chip_cols = st.columns(5)
    topics = [
        ("RBI", "RBI monetary policy India"),
        ("Inflation", "India CPI inflation"),
        ("Markets", "Indian stock markets Nifty Sensex"),
        ("Budget", "India Union Budget"),
        ("Global", "global economy"),
    ]
    for (label, q), col in zip(topics, chip_cols):
        if col.button(label):
            st.session_state["news_query"] = q

    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input(
            "Search / topic",
            value=st.session_state["news_query"],
            key="news_query_input",
        )
        st.session_state["news_query"] = query
    with c2:
        country_mode = st.selectbox("Feed region", ["India", "Global"])

    sentiment_filter = st.radio(
        "Filter by sentiment",
        options=["All", "Positive", "Negative", "Neutral"],
        horizontal=True,
    )

    with st.spinner("Fetching headlines..."):
        headlines = load_headlines(query, headline_count, country_mode)

    if not headlines:
        st.info("No articles found. Try a broader keyword or set NEWSAPI_KEY.")
        return

    # Overall sentiment stats
    labels = [h["sent_label"] for h in headlines]
    pos_cnt = labels.count("positive")
    neg_cnt = labels.count("negative")
    neu_cnt = labels.count("neutral")

    st.markdown(
        f"""
        <div class="card" style="display:flex; gap:1.2rem; align-items:center;">
          <div>
            <div class="small-muted">Today’s sentiment mix</div>
            <div style="font-size:2.1rem; font-weight:700; color:{COLORS['accent2']};">
                {len(headlines)} articles
            </div>
          </div>
          <div>
            <div class="small-muted">Positive</div>
            <div style="color:{COLORS['pos']}; font-weight:600;">{pos_cnt}</div>
          </div>
          <div>
            <div class="small-muted">Negative</div>
            <div style="color:{COLORS['neg']}; font-weight:600;">{neg_cnt}</div>
          </div>
          <div>
            <div class="small-muted">Neutral</div>
            <div style="color:{COLORS['neu']}; font-weight:600;">{neu_cnt}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # sentiment filter
    if sentiment_filter != "All":
        label_map = {
            "Positive": "positive",
            "Negative": "negative",
            "Neutral": "neutral",
        }
        wanted = label_map[sentiment_filter]
        headlines = [h for h in headlines if h["sent_label"] == wanted]

    # trending keywords
    keywords = []
    for h in headlines:
        for w in (h["title"] or "").split():
            w = "".join(ch for ch in w.lower() if ch.isalpha())
            if len(w) > 4:
                keywords.append(w)
    common = Counter(keywords).most_common(10)

    left, right = st.columns([3, 2])

    # Right: keyword bar chart + bookmarks
    with right:
        st.markdown("#### 🔍 Frequent words")
        if common:
            kw_df = pd.DataFrame(common, columns=["word", "count"])
            fig_kw = px.bar(
                kw_df,
                x="word",
                y="count",
                title="Top headline words",
            )
            fig_kw.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=10, t=40, b=40),
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.caption("–")

        st.markdown("---")
        st.markdown("#### ⭐ Bookmarks (this session)")
        if not st.session_state["bookmarks"]:
            st.caption("No bookmarked articles yet.")
        else:
            for b in st.session_state["bookmarks"]:
                st.markdown(
                    f"- [{b['title']}]({b['url']})  \n"
                    f"  <span class='small-muted'>{b.get('source','')} · {fmt_ts(b.get('published'))}</span>",
                    unsafe_allow_html=True,
                )

    with left:
        # breaking banner
        top = sorted(
            headlines,
            key=lambda x: x.get("published") or pd.Timestamp.min,
            reverse=True,
        )[0]
        st.markdown(
            f"""
            <div class="card" style="border-left:3px solid {COLORS['accent2']}; margin-bottom:0.9rem;">
              <div class="small-muted">Breaking</div>
              <a href="{top['url']}" target="_blank" style="color:{COLORS['accent2']}; text-decoration:none; font-weight:600;">
                {top['title']}
              </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for i, art in enumerate(headlines):
            label = art["sent_label"]
            score = art["sent_score"]
            if label == "positive":
                bc = COLORS["pos"]
            elif label == "negative":
                bc = COLORS["neg"]
            else:
                bc = COLORS["neu"]

            sent_html = f"<span class='sent-pill' style='background:{bc};'>{label.upper()}</span>"
            tags_html = " ".join(
                f"<span class='chip'>{t}</span>" for t in art["impact"]
            )

            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex; justify-content:space-between; gap:0.8rem;">
                    <div style="flex:1;">
                      <a href="{art['url']}" target="_blank"
                         style="color:{COLORS['text']}; font-weight:600; text-decoration:none;">
                        {art['title']}
                      </a>
                      <div class="small-muted">
                        {(art.get('source') or '')} · {fmt_ts(art.get('published'))}
                      </div>
                    </div>
                    <div style="text-align:right;">
                      {sent_html}
                      <div class="small-muted" style="margin-top:4px;">score {score:+.2f}</div>
                    </div>
                  </div>
                  <div style="margin-top:6px; font-size:0.9rem;">{art['summary']}</div>
                  <div style="margin-top:6px;">{tags_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("⭐ Bookmark", key=f"bookmark_{i}"):
                st.session_state["bookmarks"].append(
                    {
                        "title": art["title"],
                        "url": art["url"],
                        "source": art.get("source"),
                        "published": art.get("published"),
                    }
                )


# ------------------------------------------------------------------
#                       MACRO INDICATORS
# ------------------------------------------------------------------

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


def detect_date_value_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    date_col = next(
        (c for c in cols if any(x in c.lower() for x in ["date", "month", "year", "quarter"])),
        cols[0],
    )
    value_col = next(
        (
            c
            for c in cols
            if any(
                x in c.lower()
                for x in [
                    "value",
                    "index",
                    "cpi",
                    "iip",
                    "gdp",
                    "rate",
                    "percent",
                    "%",
                    "growth",
                ]
            )
        ),
        cols[1] if len(cols) > 1 else cols[0],
    )
    return date_col, value_col


def latest_value(df: pd.DataFrame | None) -> Tuple[Any, Any]:
    if df is None or df.empty:
        return None, None
    try:
        date_col, val_col = detect_date_value_cols(df)
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        row = tmp.sort_values(date_col).iloc[-1]
        return row[val_col], row[date_col]
    except Exception as e:
        log(f"latest_value error: {e}")
        return None, None


def compute_change_series(
    df: pd.DataFrame, mode: str
) -> Tuple[pd.DataFrame, str, str]:
    """
    Return (df_with_change, x_col, y_col) for plotting, based on change mode:
    - 'Level' → raw value
    - 'MoM %' → month-over-month percentage change
    - 'YoY %' → year-over-year percentage change (approx 12-period lag)
    """
    date_col, val_col = detect_date_value_cols(df)
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[val_col] = pd.to_numeric(tmp[val_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, val_col]).sort_values(date_col)

    if mode == "Level":
        return tmp[[date_col, val_col]].copy(), date_col, val_col

    tmp = tmp.set_index(date_col).sort_index()
    if mode == "MoM %":
        ser = tmp[val_col].pct_change() * 100
        out = ser.dropna().reset_index()
        out.columns = [date_col, "MoM_change"]
        return out, date_col, "MoM_change"

    if mode == "YoY %":
        # assume monthly-ish frequency, 12-period lag
        ser = tmp[val_col].pct_change(periods=12) * 100
        out = ser.dropna().reset_index()
        out.columns = [date_col, "YoY_change"]
        return out, date_col, "YoY_change"

    # fallback
    return tmp[[date_col, val_col]].copy(), date_col, val_col


def macro_tab():
    st.subheader("📊 India macro indicators")

    # Admin uploads
    with st.expander("Upload fallback files (optional)"):
        c1, c2 = st.columns(2)
        with c1:
            up_cpi = st.file_uploader("CPI", type=["csv", "xlsx", "pdf"])
            up_iip = st.file_uploader("IIP", type=["csv", "xlsx", "pdf"])
        with c2:
            up_gdp = st.file_uploader("GDP", type=["csv", "xlsx", "pdf"])
            up_unemp = st.file_uploader("Unemployment", type=["csv", "xlsx", "pdf"])

    # Remote data
    cpi_remote = fetch_data_gov(CPI_RESOURCE_ID) if CPI_RESOURCE_ID else None
    iip_remote = fetch_data_gov(IIP_RESOURCE_ID) if IIP_RESOURCE_ID else None
    gdp_remote = fetch_data_gov(GDP_RESOURCE_ID) if GDP_RESOURCE_ID else None

    # Uploaded
    cpi_up = read_table_or_text(up_cpi) if up_cpi else None
    iip_up = read_table_or_text(up_iip) if up_iip else None
    gdp_up = read_table_or_text(up_gdp) if up_gdp else None
    unemp_up = read_table_or_text(up_unemp) if up_unemp else None

    # Use remote if DataFrame, else uploaded DataFrame
    cpi_df = cpi_remote if isinstance(cpi_remote, pd.DataFrame) else (
        cpi_up if isinstance(cpi_up, pd.DataFrame) else None
    )
    iip_df = iip_remote if isinstance(iip_remote, pd.DataFrame) else (
        iip_up if isinstance(iip_up, pd.DataFrame) else None
    )
    gdp_df = gdp_remote if isinstance(gdp_remote, pd.DataFrame) else (
        gdp_up if isinstance(gdp_up, pd.DataFrame) else None
    )

    # snapshot row
    cpi_val, cpi_dt = latest_value(cpi_df)
    iip_val, iip_dt = latest_value(iip_df)
    gdp_val, gdp_dt = latest_value(gdp_df)

    cards = st.columns(4)
    snap_items = [
        ("CPI (inflation)", "📊", cpi_val, cpi_dt),
        ("IIP (industrial output)", "🏭", iip_val, iip_dt),
        ("GDP (growth)", "💹", gdp_val, gdp_dt),
        ("Unemployment (uploaded)", "👷", None, None),
    ]
    for col, (label, icon, val, dt) in zip(cards, snap_items):
        if isinstance(val, (int, float, np.floating)):
            try:
                disp = f"{float(val):.1f}"
            except Exception:
                disp = str(val)
        else:
            disp = "N/A"
        date_txt = str(pd.to_datetime(dt).date()) if dt is not None else "Latest"
        col.markdown(
            f"""
            <div class="card" style="text-align:center;">
              <div style="font-size:2rem;">{icon}</div>
              <div style="font-size:1.8rem; color:{COLORS['accent']}; font-weight:700;">{disp}</div>
              <div style="font-size:0.9rem; font-weight:600;">{label}</div>
              <div class="small-muted">{date_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Detailed panel – indicator selector + change mode
    top_cols = st.columns([2, 2, 2])
    with top_cols[0]:
        indicator = st.selectbox(
            "Indicator",
            ["CPI", "IIP", "GDP", "Unemployment"],
        )
    with top_cols[1]:
        change_mode = st.selectbox(
            "Display mode",
            ["Level", "MoM %", "YoY %"],
            index=0,
        )
    with top_cols[2]:
        st.caption("Indicator comparison (normalized index)")
        compare_option = st.selectbox(
            "Compare",
            ["None", "CPI vs IIP", "CPI vs GDP", "IIP vs GDP"],
        )

    # Main indicator
    if indicator == "CPI":
        df = cpi_df
        desc = "Consumer Price Index (price-level changes)"
    elif indicator == "IIP":
        df = iip_df
        desc = "Index of Industrial Production (output of core sectors)"
    elif indicator == "GDP":
        df = gdp_df
        desc = "Gross Domestic Product – growth over time"
    else:
        # unemployment only from upload
        df = unemp_up if isinstance(unemp_up, pd.DataFrame) else None
        desc = "Unemployment rate (custom upload)"

    st.markdown(f"**{indicator}** – {desc}")

    if df is None or df.empty:
        if isinstance(unemp_up, str) and indicator == "Unemployment":
            st.text_area("Unemployment PDF preview", unemp_up[:2500])
        else:
            st.info("No structured data available for this indicator yet.")
    else:
        try:
            series_df, x_col, y_col = compute_change_series(df, change_mode)
            fig = px.line(
                series_df,
                x=x_col,
                y=y_col,
                markers=True,
                title=f"{indicator} – {change_mode}",
            )
            fig.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Detected columns → date: `{x_col}` · value: `{y_col}`")

            with st.expander("Raw data preview"):
                st.dataframe(series_df.tail(20))
        except Exception as e:
            st.warning(f"Could not plot time-series: {e}")

    st.markdown("---")

    # Comparison chart: normalized index = 100
    if compare_option != "None":
        mapping = {
            "CPI": cpi_df,
            "IIP": iip_df,
            "GDP": gdp_df,
        }
        if compare_option == "CPI vs IIP":
            a, b = "CPI", "IIP"
        elif compare_option == "CPI vs GDP":
            a, b = "CPI", "GDP"
        else:
            a, b = "IIP", "GDP"

        df_a = mapping.get(a)
        df_b = mapping.get(b)

        if df_a is None or df_b is None:
            st.info("Not enough data available for comparison.")
        else:
            try:
                da = df_a.copy()
                db = df_b.copy()
                ac, av = detect_date_value_cols(da)
                bc, bv = detect_date_value_cols(db)

                da[ac] = pd.to_datetime(da[ac], errors="coerce")
                da[av] = pd.to_numeric(da[av], errors="coerce")
                da = da.dropna(subset=[ac, av]).sort_values(ac)

                db[bc] = pd.to_datetime(db[bc], errors="coerce")
                db[bv] = pd.to_numeric(db[bv], errors="coerce")
                db = db.dropna(subset=[bc, bv]).sort_values(bc)

                da = da.set_index(ac)
                db = db.set_index(bc)

                joined = da[[av]].join(db[[bv]], how="inner", lsuffix=f"_{a}", rsuffix=f"_{b}")
                joined = joined.dropna()

                base_a = joined[av].iloc[0]
                base_b = joined[bv].iloc[0]
                joined[f"{a}_idx"] = joined[av] / base_a * 100
                joined[f"{b}_idx"] = joined[bv] / base_b * 100

                plot_df = joined[[f"{a}_idx", f"{b}_idx"]].reset_index().rename(columns={joined.index.name or "index": "date"})

                fig_cmp = px.line(
                    plot_df,
                    x="date",
                    y=[f"{a}_idx", f"{b}_idx"],
                    title=f"{a} vs {b} (normalized to 100)",
                )
                fig_cmp.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig_cmp, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute comparison chart: {e}")


# ------------------------------------------------------------------
#                          MARKETS
# ------------------------------------------------------------------

@st.cache_data(ttl=MARKET_TTL)
def fetch_index_snapshot() -> Dict[str, Dict[str, float | None]]:
    out: Dict[str, Dict[str, float | None]] = {}
    for name, ticker in INDEX_SYMBOLS.items():
        try:
            df = yf.download(
                ticker, period="2d", interval="5m", progress=False, threads=False
            )
            if df is None or df.empty:
                out[name] = {"last": None, "pct": None}
                continue
            df = df[~df.index.duplicated(keep="last")]
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
            pct = (last - prev) / prev * 100 if prev else 0.0
            out[name] = {"last": last, "pct": pct}
        except Exception as e:
            log(f"index error {name}: {e}")
            out[name] = {"last": None, "pct": None}
    return out


def markets_header():
    st.subheader("📈 Equity indices snapshot")

    cols = st.columns(len(INDEX_SYMBOLS))
    snap = fetch_index_snapshot()
    for col, (name, _) in zip(cols, INDEX_SYMBOLS.items()):
        d = snap.get(name, {})
        price = d.get("last")
        pct = d.get("pct")

        if price is None or pct is None:
            p_txt = "N/A"
            delta_txt = ""
            clr = COLORS["muted"]
        else:
            p_txt = f"{float(price):,.2f}"
            arrow = "▲" if pct >= 0 else "▼"
            clr = COLORS["pos"] if pct >= 0 else COLORS["neg"]
            delta_txt = f"{arrow} {pct:+.2f}%"

        col.markdown(
            f"""
            <div class="card" style="text-align:center;">
              <div class="small-muted">{name}</div>
              <div style="font-size:1.4rem; font-weight:700;">{p_txt}</div>
              <div style="font-size:0.85rem; font-weight:600; color:{clr}; margin-top:2px;">
                {delta_txt}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def stock_tab(default_symbol: str):
    markets_header()
    st.markdown("---")

    st.subheader("💹 Watchlist & single-stock view")

    preset_map = load_preset_tickers()

    # ------------- Nifty/Sensex picker + extra tickers -------------
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_labels = st.multiselect(
            "Select from Nifty & Sensex companies",
            options=sorted(preset_map.keys()),
            default=[
                "Reliance Industries (NIFTY, SENSEX)",
                "HDFC Bank (NIFTY, SENSEX)",
                "ICICI Bank (NIFTY, SENSEX)",
            ],
        )
        preset_symbols = [preset_map[l] for l in selected_labels] if selected_labels else []
    with c2:
        extra_raw = st.text_input(
            "Any extra tickers? (comma-separated, e.g. INFY.NS, AAPL, TSLA)",
            value=default_symbol,
        )
        extra_syms = [s.strip() for s in extra_raw.split(",") if s.strip()]

    watchlist_symbols = sorted(set(preset_symbols + extra_syms))

    if not watchlist_symbols:
        st.info("Select at least one company from the list or add a manual ticker above.")
        return

    # ------------- Mini watchlist cards (1D change) -------------
    st.markdown("#### 🔭 Watchlist snapshot")
    cols = st.columns(min(len(watchlist_symbols), 4))

    for i, sym in enumerate(watchlist_symbols):
        col = cols[i % len(cols)]
        with col:
            try:
                df = yf.download(sym, period="5d", interval="1d", progress=False)
                if df.empty:
                    st.markdown(
                        f"<div class='card'><div class='small-muted'>{sym}</div><div>no data</div></div>",
                        unsafe_allow_html=True,
                    )
                    continue
                df = df.tail(5)
                last = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2] if len(df) > 1 else last
                pct = (last - prev) / prev * 100 if prev else 0.0
                clr = COLORS["pos"] if pct >= 0 else COLORS["neg"]
                st.markdown(
                    f"""
                    <div class='card'>
                      <div class='small-muted'>{sym}</div>
                      <div style='font-size:1.3rem; font-weight:600;'>{last:,.2f}</div>
                      <div style='color:{clr}; font-size:0.8rem; font-weight:600;'>{pct:+.2f}% (1D)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                log(f"watchlist error {sym}: {e}")

    st.markdown("---")

    # ------------- Correlation heatmap on watchlist -------------
    if len(watchlist_symbols) >= 2:
        st.markdown("#### 🔗 Correlation heatmap (daily returns)")
        try:
            price_df = yf.download(
                watchlist_symbols,
                period="6mo",
                interval="1d",
                progress=False,
            )["Close"]
            price_df = price_df.dropna(how="all")
            returns = price_df.pct_change().dropna()
            corr = returns.corr()

            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation of daily returns (6M)",
            )
            fig_corr.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute correlation heatmap: {e}")

    st.markdown("---")

    # ------------- Single stock deep dive (use one from watchlist) -------------
    st.subheader("🔍 Single-stock analysis")

    default_choice = watchlist_symbols[0] if watchlist_symbols else default_symbol
    symbol = st.selectbox(
        "Choose symbol for detailed view",
        options=watchlist_symbols,
        index=watchlist_symbols.index(default_choice) if default_choice in watchlist_symbols else 0,
    )

    range_map = {
        "1M": ("1mo", "1d"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "3Y": ("3y", "1wk"),
    }

    if "stock_range" not in st.session_state:
        st.session_state["stock_range"] = "6M"

    st.caption("Time range:")
    tabs = st.tabs(list(range_map.keys()))
    for label, tab in zip(range_map.keys(), tabs):
        with tab:
            if st.button(label, key=f"range_{label}"):
                st.session_state["stock_range"] = label
                st.experimental_rerun()

    selected = st.session_state["stock_range"]
    period, interval = range_map[selected]

    with st.spinner(f"Fetching {symbol}..."):
        data = yf.download(symbol, period=period, interval=interval, progress=False)

    if data.empty:
        st.error("No price data found. Try another symbol or add `.NS` for Indian stocks.")
        return

    data = data.reset_index()
    last = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else last

    price = float(last["Close"])
    prev_price = float(prev["Close"])
    diff = price - prev_price
    pct = (diff / prev_price) * 100 if prev_price else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last", f"{price:,.2f}", f"{diff:+.2f}")
    c2.metric("Change %", f"{pct:+.2f}%")
    c3.metric("Open", f"{float(last['Open']):,.2f}")
    c4.metric("High", f"{float(last['High']):,.2f}")
    c5.metric("Low", f"{float(last['Low']):,.2f}")

    st.caption(f"Last bar: {last['Date']} | Range: {selected}")

    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close"))
    fig.update_layout(
        title=f"{symbol} – price ({selected})",
        template="plotly_dark",
        height=420,
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Moving averages + basic risk metrics
    st.markdown("### Moving averages & risk")
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    show20 = st.checkbox("Show 20-period MA", value=True)
    show50 = st.checkbox("Show 50-period MA", value=False)

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close"))
    if show20:
        fig_ma.add_trace(go.Scatter(x=data["Date"], y=data["MA20"], mode="lines", name="MA20"))
    if show50:
        fig_ma.add_trace(go.Scatter(x=data["Date"], y=data["MA50"], mode="lines", name="MA50"))
    fig_ma.update_layout(
        title=f"{symbol} – moving averages",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    # Risk: daily volatility + max drawdown + return histogram
    st.markdown("### Risk profile (based on selected range)")
    data["ret"] = data["Close"].pct_change()
    ret = data["ret"].dropna()
    if not ret.empty:
        vol = ret.std() * np.sqrt(252) * 100  # annualised approx
        cum = (1 + ret).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1).min() * 100

        r1, r2 = st.columns(2)
        r1.metric("Annualised volatility", f"{vol:.2f}%")
        r2.metric("Max drawdown", f"{dd:.2f}%")

        fig_hist = px.histogram(
            ret * 100,
            nbins=40,
            title="Distribution of daily returns (%)",
        )
        fig_hist.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Not enough data points to compute risk metrics.")


# ------------------------------------------------------------------
#                         SIDEBAR & MAIN
# ------------------------------------------------------------------

def sidebar_controls() -> Tuple[int, str]:
    st.sidebar.title("Controls")

    # auto-refresh
    choice = st.sidebar.selectbox(
        "Auto-refresh",
        ["Off", "30s", "60s", "5m"],
        index=1,
    )
    interval_map = {"Off": 0, "30s": 30, "60s": 60, "5m": 300}
    seconds = interval_map[choice]
    if HAS_AUTOREFRESH and seconds > 0:
        tick = st_autorefresh(interval=seconds * 1000, key="auto_refresh")
        st.sidebar.caption(f"Tick: {tick}")
    elif seconds > 0:
        st.sidebar.info("Install `streamlit-autorefresh` for automatic refresh.")

    headline_count = st.sidebar.slider("News items", 4, 20, value=8)

    default_symbol = st.sidebar.text_input(
        "Default stock symbol", value="RELIANCE.NS"
    )

    if st.sidebar.button("Clear caches and rerun"):
        requests_cache.clear()
        st.cache_data.clear()
        st.experimental_rerun()

    return headline_count, default_symbol


def main():
    headline_count, default_symbol = sidebar_controls()

    st.markdown(
        f"""
        <h1>India Macro & Markets Monitor</h1>
        <div class="small-muted">
          Dark dashboard for economic news, macro indicators and equity markets.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tabs = st.tabs(["📰 News", "📊 Macro", "💹 Markets"])

    with tabs[0]:
        news_tab(headline_count)

    with tabs[1]:
        macro_tab()

    with tabs[2]:
        stock_tab(default_symbol)

    st.markdown("---")
    st.markdown(
        f"<div class='small-muted'>Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Debug log (developer)"):
        for line in st.session_state["_log"][-200:]:
            st.text(line)


if __name__ == "__main__":
    main()
