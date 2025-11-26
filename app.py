# ---------- INDIA MACRO INDICATORS (World Bank API, fully functional) ----------

# World Bank indicator codes:
# CPI inflation: FP.CPI.TOTL.ZG
# GDP growth (real): NY.GDP.MKTP.KD.ZG
# Manufacturing growth (proxy for IIP): NV.IND.MANF.KD.ZG
# Unemployment rate: SL.UEM.TOTL.ZS

@st.cache_data(ttl=MACRO_TTL)
def fetch_wb_indicator(code: str, country: str = "IND", max_years: int = 40) -> pd.DataFrame:
    """
    Fetch a single World Bank indicator for India.
    Returns a DataFrame with columns: ['year', 'value'] sorted ascending.
    """
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{code}"
    params = {"format": "json", "per_page": max_years}
    js = safe_json_get(url, params=params)
    if not js or len(js) < 2 or js[1] is None:
        return pd.DataFrame(columns=["year", "value"])

    rows = []
    for item in js[1]:
        rows.append(
            {
                "year": item.get("date"),
                "value": item.get("value"),
            }
        )
    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).sort_values("year")
    return df


def latest_from_df(df: pd.DataFrame):
    """Return (latest_value, latest_year) from a WB-style DataFrame."""
    if df is None or df.empty:
        return None, None
    tmp = df.dropna(subset=["value"]).sort_values("year")
    row = tmp.iloc[-1]
    return float(row["value"]), int(row["year"])


def macro_tab():
    st.markdown("---")
    st.markdown("## 📊 India macro indicators")
    st.markdown(
        "<div class='small-muted'>Powered by World Bank World Development Indicators (auto-fetched)</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading macro data from the World Bank..."):
        cpi_df = fetch_wb_indicator("FP.CPI.TOTL.ZG")        # Inflation, consumer prices (annual %)
        gdp_df = fetch_wb_indicator("NY.GDP.MKTP.KD.ZG")     # GDP growth (annual %)
        iip_df = fetch_wb_indicator("NV.IND.MANF.KD.ZG")     # Manufacturing value added, annual % growth (proxy for IIP)
        unemp_df = fetch_wb_indicator("SL.UEM.TOTL.ZS")      # Unemployment, total (% of labour force)

    cpi_val, cpi_year = latest_from_df(cpi_df)
    gdp_val, gdp_year = latest_from_df(gdp_df)
    iip_val, iip_year = latest_from_df(iip_df)
    u_val, u_year = latest_from_df(unemp_df)

    cards = st.columns(4)

    def macro_card(col, emoji, label, value, year):
        if isinstance(value, (int, float, np.floating)):
            disp = f"{value:.1f}"
        else:
            disp = "N/A"
        yr = str(year) if year is not None else "latest"
        col.markdown(
            f"""
            <div class="card" style="text-align:center;">
              <div style="font-size:2rem;">{emoji}</div>
              <div style="font-size:1.8rem; color:{COLORS['accent']}; font-weight:700;">{disp}</div>
              <div style="font-size:0.9rem; font-weight:600;">{label}</div>
              <div class="small-muted">{yr}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    macro_card(cards[0], "📊", "Inflation (CPI, %)", cpi_val, cpi_year)
    macro_card(cards[1], "🏭", "Manufacturing growth (%, proxy for IIP)", iip_val, iip_year)
    macro_card(cards[2], "💹", "GDP growth (%, real)", gdp_val, gdp_year)
    macro_card(cards[3], "👷", "Unemployment (% of labour force)", u_val, u_year)

    st.markdown("---")

    indicator_map = {
        "Inflation (CPI)": {
            "code": "FP.CPI.TOTL.ZG",
            "df": cpi_df,
            "desc": "Inflation, consumer prices (annual % – World Bank code FP.CPI.TOTL.ZG)",
        },
        "Manufacturing growth": {
            "code": "NV.IND.MANF.KD.ZG",
            "df": iip_df,
            "desc": "Manufacturing, value added (annual % growth – proxy for industrial production, code NV.IND.MANF.KD.ZG)",
        },
        "GDP growth": {
            "code": "NY.GDP.MKTP.KD.ZG",
            "df": gdp_df,
            "desc": "GDP growth (annual %, constant prices – code NY.GDP.MKTP.KD.ZG)",
        },
        "Unemployment rate": {
            "code": "SL.UEM.TOTL.ZS",
            "df": unemp_df,
            "desc": "Unemployment, total (% of labour force) – code SL.UEM.TOTL.ZS",
        },
    }

    left, right = st.columns([3, 1])

    with left:
        selected_label = st.selectbox("Choose indicator to explore", list(indicator_map.keys()))
        info = indicator_map[selected_label]
        df = info["df"]
        st.caption(info["desc"])

        if df is None or df.empty:
            st.info("No data available for this indicator (World Bank API returned empty).")
        else:
            mode = st.radio(
                "Display",
                ["Level (value)", "Change vs previous year (percentage points)"],
                horizontal=True,
            )

            plot_df = df.copy()
            if mode == "Level (value)":
                ycol = "value"
                title_suffix = ""
            else:
                plot_df["change"] = plot_df["value"].diff()
                ycol = "change"
                title_suffix = " – change vs previous year"

            fig = px.line(
                plot_df,
                x="year",
                y=ycol,
                markers=True,
                title=f"{selected_label}{title_suffix}",
            )
            fig.update_layout(
                template="plotly_dark",
                height=420,
                xaxis_title="Year",
                yaxis_title="Value (%)",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw data (tail)"):
                st.dataframe(plot_df.tail(15))

    with right:
        st.markdown("#### ⬇️ Download CSVs")
        for label, info in indicator_map.items():
            df = info["df"]
            if df is None or df.empty:
                continue
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"{label}",
                data=csv,
                file_name=f"india_{label.replace(' ', '_').lower()}_worldbank.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.markdown("#### ℹ️ Data source")
        st.caption(
            "World Bank — World Development Indicators (no API key needed).\n\n"
            "- CPI: FP.CPI.TOTL.ZG\n"
            "- Manufacturing: NV.IND.MANF.KD.ZG\n"
            "- GDP: NY.GDP.MKTP.KD.ZG\n"
            "- Unemployment: SL.UEM.TOTL.ZS"
        )
