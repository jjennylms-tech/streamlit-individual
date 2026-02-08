# app.py
# MSIN0231 ML4B Individual Assignment - Streamlit Market Research Assistant
# Implements:
#   Q1: Validate industry input
#   Q2: Retrieve and show 5 most relevant Wikipedia URLs
#   Q3: Generate <500-word industry report based on those 5 pages
#
# Notes on cost control:
# - Only call the LLM after we have the 5 pages.
# - We truncate each page's content (so we don't send huge Wikipedia articles).
# - We optionally do 2-step summarise-then-synthesise to reduce tokens.

import os
import re
import textwrap
from typing import List, Dict, Tuple

import streamlit as st

# --- Optional LangChain Wikipedia retriever (recommended by brief) ---
# If you don't have these packages installed, see requirements.txt below.
USE_LANGCHAIN = True
try:
    from langchain_community.retrievers import WikipediaRetriever
except Exception:
    USE_LANGCHAIN = False

# --- Optional OpenAI LLM via LangChain ---
# You can swap this for Gemini/DeepSeek providers if you prefer.
USE_OPENAI = True
try:
    from langchain_openai import ChatOpenAI
except Exception:
    USE_OPENAI = False


# =============================================================================
# Helpers
# =============================================================================

def clean_industry(s: str) -> str:
    """Basic cleaning to reduce accidental empty inputs."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def enforce_word_limit(text: str, max_words: int = 500) -> str:
    """Hard-cap the report to max_words (assignment requires <500 words)."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " …"


def extract_wikipedia_urls(docs) -> List[str]:
    """Extract URLs from LangChain Wikipedia docs."""
    urls = []
    for d in docs:
        # metadata often contains 'source' for wikipedia retriever
        src = None
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            src = d.metadata.get("source") or d.metadata.get("url")
        if not src and hasattr(d, "page_content"):
            # fallback (rare)
            src = None
        if src:
            urls.append(src)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def truncate_text_for_cost(text: str, max_chars: int) -> str:
    """Cost control: reduce how much Wikipedia content we send into the LLM."""
    if not text:
        return ""
    return text[:max_chars]


@st.cache_data(show_spinner=False)
def retrieve_wikipedia(industry: str, k: int = 5):
    """
    Retrieve top-k Wikipedia documents for the industry using WikipediaRetriever.
    Cached to avoid repeated calls when you rerun the app.
    """
    if not USE_LANGCHAIN:
        raise RuntimeError(
            "LangChain WikipediaRetriever not available. "
            "Install packages from requirements.txt."
        )

    retriever = WikipediaRetriever(top_k_results=k, lang="en")
    docs = retriever.get_relevant_documents(industry)
    return docs


def build_llm(model_name: str, temperature: float):
    """
    Create an LLM client using OpenAI via LangChain.
    Works both locally (env var) and on Streamlit Cloud (Secrets).
    """
    if not USE_OPENAI:
        raise RuntimeError(
            "langchain_openai not available. Install it or implement another provider."
        )

    # 1) Streamlit Cloud: Secrets
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    # 2) Local: environment variable
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to Streamlit Secrets (OPENAI_API_KEY) "
            "or set it as an environment variable before running."
        )

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )


def llm_summarise_pages(llm, pages: List[Dict], max_words_each: int = 110) -> List[str]:
    """
    Summarise each Wikipedia page separately.
    This is a cost-control technique: short summaries are cheaper than full pages.
    """
    summaries = []
    for i, p in enumerate(pages, start=1):
        prompt = f"""
You are a market research assistant for a business analyst.
Summarise the Wikipedia page below into at most {max_words_each} words.
Focus on facts useful for understanding the industry: definition, scope, key segments, stakeholders,
economics/market aspects, technology, and trends. Avoid fluff.

PAGE TITLE: {p['title']}
PAGE URL: {p['url']}
PAGE TEXT:
{p['text']}
"""
        resp = llm.invoke(prompt)
        s = resp.content.strip()
        summaries.append(enforce_word_limit(s, max_words_each))
    return summaries


def llm_write_report(llm, industry: str, page_summaries: List[str], urls: List[str]) -> str:
    """
    Produce a single <500-word industry report grounded in the five pages.
    """
    joined = "\n\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(page_summaries)])

    prompt = f"""
You are a market research assistant. Write a concise industry report for a business analyst.

INDUSTRY: {industry}

Constraints:
- Less than 500 words total.
- Base the report ONLY on the five Wikipedia pages (summaries) provided below.
- Use clear business language (structure + headings is welcome).
- Include a short "Sources" line listing the five URLs at the end.

Write a report with these sections:
1) Industry definition & scope
2) Value chain / key segments
3) Major drivers & trends
4) Risks / challenges
5) What a large corporation should watch next (2-3 bullets)

Wikipedia summaries (ground truth):
{joined}

Five URLs:
{chr(10).join(urls)}
"""
    resp = llm.invoke(prompt)
    report = resp.content.strip()

    # Enforce hard word limit
    report = enforce_word_limit(report, 500)
    return report


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="ML4B Market Research Assistant", layout="wide")
st.title("📊 Market Research Assistant (Wikipedia-based)")
st.caption("MSIN0231 ML4B Individual Assignment – Q1 to Q3")

with st.sidebar:
    st.header("Settings (Cost & Quality)")
    st.write(
        "Tip: cheaper models + shorter context = lower cost.\n"
        "This app truncates Wikipedia text and summarises pages before writing the report."
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    # You can rename these to the exact models you are allowed to use.
    # If you only want one, keep one option.
    model_name = st.selectbox(
        "LLM model (example: OpenAI)",
        options=[
            "gpt-5-mini"     # if available in your account
        ],
        index=0
    )

    wiki_chars_per_page = st.slider(
        "Max Wikipedia characters per page (cost control)",
        min_value=1200,
        max_value=8000,
        value=3500,
        step=200
    )

st.divider()
st.header("Step 1 (Q1): Provide an industry")

industry_raw = st.text_input("Enter an industry (e.g., 'Electric vehicles', 'Fast fashion', 'Cloud computing'):")
industry = clean_industry(industry_raw)

# Q1 requirement: if no industry, ask for update (do not proceed)
if not industry:
    st.info("Please enter an industry to continue.")
    st.stop()

st.success(f"Industry received: **{industry}**")

st.divider()
st.header("Step 2 (Q2): Retrieve the 5 most relevant Wikipedia URLs")

colA, colB = st.columns([1, 1])

with colA:
    run_retrieval = st.button("Find Wikipedia pages", type="primary")

# ---- Run retrieval ONLY when button is clicked ----
if run_retrieval:
    with st.spinner("Searching Wikipedia…"):
        try:
            docs = retrieve_wikipedia(industry, k=5)

            # Extract and keep 5 unique URLs
            urls = extract_wikipedia_urls(docs)
            urls = urls[:5]

            if len(urls) < 5:
                st.warning(
                    f"Only found {len(urls)} Wikipedia URL(s). "
                    "Please try a broader or more standard industry term (e.g., 'retail banking' instead of a niche phrase)."
                )
                st.stop()

            # Build pages aligned with retrieved docs (truncate for cost control)
            pages = []
            for d in docs:
                title = (d.metadata.get("title") if hasattr(d, "metadata") else None) or "Wikipedia page"
                url = (d.metadata.get("source") if hasattr(d, "metadata") else None) or ""
                text = getattr(d, "page_content", "") or ""

                pages.append({
                    "title": title,
                    "url": url,
                    "text": truncate_text_for_cost(text, wiki_chars_per_page)
                })

            # Keep only the pages whose URL is in our top-5 list
            pages = [p for p in pages if p["url"] in urls]

            # Safety: ensure exactly 5 pages stored
            pages = pages[:5]

            st.session_state["wiki_urls"] = urls
            st.session_state["wiki_pages"] = pages

        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

# If user hasn't clicked retrieval yet, stop before moving on
if "wiki_urls" not in st.session_state or not st.session_state["wiki_urls"]:
    st.warning("Click **Find Wikipedia pages** to retrieve and display the 5 URLs.")
    st.stop()

# Display Q2 output (URLs)
if "wiki_urls" in st.session_state and st.session_state["wiki_urls"]:
    st.subheader("Q2 Output: Five most relevant Wikipedia URLs")
    for i, u in enumerate(st.session_state["wiki_urls"], start=1):
        st.write(f"{i}. {u}")
else:
    st.warning("Click **Find Wikipedia pages** to retrieve and display the 5 URLs.")
    st.stop()

st.divider()
st.header("Step 3 (Q3): Generate an industry report (<500 words)")

col1, col2 = st.columns([1, 1])

with col1:
    generate = st.button("Generate report", type="primary")

with col2:
    st.write("Optional: you can preview the truncated Wikipedia text used for grounding below.")

if generate:
    with st.spinner("Generating report…"):
        try:
            llm = build_llm(model_name=model_name, temperature=temperature)

            pages = st.session_state["wiki_pages"]
            urls = st.session_state["wiki_urls"]

            # Summarise each page to reduce token usage
            summaries = llm_summarise_pages(llm, pages, max_words_each=110)

            # Write final report from summaries
            report = llm_write_report(llm, industry=industry, page_summaries=summaries, urls=urls)

            st.session_state["report"] = report
            st.session_state["summaries"] = summaries

        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.stop()

if "report" in st.session_state:
    st.subheader("Q3 Output: Industry report (≤ 500 words)")
    st.write(st.session_state["report"])

    # Helpful: show word count for compliance
    wc = len(st.session_state["report"].split())
    st.caption(f"Word count: {wc} (must be < 500)")

    with st.expander("Show page summaries used (for transparency)"):
        for i, s in enumerate(st.session_state.get("summaries", []), start=1):
            st.markdown(f"**Summary {i}:** {s}")

    with st.expander("Show truncated Wikipedia text used (cost control)"):
        for p in st.session_state["wiki_pages"]:
            st.markdown(f"### {p['title']}")
            st.write(p["url"])
            st.text(p["text"][:4000] + ("…" if len(p["text"]) > 4000 else ""))

else:
    st.info("Click **Generate report** to produce the <500-word industry report.")
