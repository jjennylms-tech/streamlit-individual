# app.py
# MSIN0231 ML4B Individual Assignment - Streamlit Market Research Assistant
# Q1: Validate industry input + (NEW) LLM self-justification in Step 1
# Q2: Retrieve and show 5 most relevant Wikipedia URLs
# Q3: Generate <500-word industry report based on those 5 pages

import os
import re
import json
from typing import List, Dict

import streamlit as st

# --- LangChain Wikipedia retriever ---
USE_LANGCHAIN = True
try:
    from langchain_community.retrievers import WikipediaRetriever
except Exception:
    USE_LANGCHAIN = False

# --- OpenAI LLM via LangChain ---
USE_OPENAI = True
try:
    from langchain_openai import ChatOpenAI
except Exception:
    USE_OPENAI = False


# =============================================================================
# Helpers
# =============================================================================

def clean_industry(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def enforce_word_limit(text: str, max_words: int = 500) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " …"


def extract_wikipedia_urls(docs) -> List[str]:
    urls = []
    for d in docs:
        src = None
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            src = d.metadata.get("source") or d.metadata.get("url")
        if src:
            urls.append(src)

    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def truncate_text_for_cost(text: str, max_chars: int) -> str:
    return (text or "")[:max_chars]


@st.cache_data(show_spinner=False)
def retrieve_wikipedia(industry: str, k: int = 5):
    if not USE_LANGCHAIN:
        raise RuntimeError(
            "LangChain WikipediaRetriever not available. "
            "Install langchain-community and restart the app."
        )
    retriever = WikipediaRetriever(top_k_results=k, lang="en")
    if hasattr(retriever, "invoke"):
        return retriever.invoke(industry)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(industry)
    raise RuntimeError("WikipediaRetriever API changed: no supported method found.")


def build_llm(model_name: str, temperature: float, api_key_override: str | None = None):
    if not USE_OPENAI:
        raise RuntimeError("langchain_openai not available. Install langchain-openai.")

    api_key = (api_key_override or "").strip() or None
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Please enter it in the sidebar, "
            "or add it to Streamlit Secrets (OPENAI_API_KEY), "
            "or set it as an environment variable before running."
        )

    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


def llm_is_industry_check(llm, user_input: str) -> Dict:
    """
    Step 1 self-justification:
    - assistant decides if input is an industry/sector/market category
    - returns YES/NO + short reason + 3 suggestions
    """
    prompt = f"""
You are a strict classifier for a market research assistant.

Decide whether the user input is an INDUSTRY / SECTOR / MARKET CATEGORY.

Rules:
- If it is a greeting, random word, person, place, or unclear: NOT an industry.
- Accept short valid sectors like "AI", "IT", "VR", "AR".
- Accept phrases with numbers like "3D printing", "5G infrastructure", "Industry 4.0".

Return ONLY valid JSON:
{{
  "is_industry": true/false,
  "reason": "one short sentence",
  "suggestions": ["industry 1", "industry 2", "industry 3"]
}}

User input: "{user_input}"
"""
    resp = llm.invoke(prompt)
    raw = (resp.content or "").strip()
    try:
        data = json.loads(raw)
        suggestions = list(data.get("suggestions", []))[:3]
        if not suggestions:
            suggestions = ["retail banking", "cloud computing", "fast fashion"]
        return {
            "is_industry": bool(data.get("is_industry", False)),
            "reason": str(data.get("reason", "")).strip() or "No reason provided.",
            "suggestions": suggestions,
        }
    except Exception:
        return {
            "is_industry": False,
            "reason": "I couldn’t confidently identify this as an industry/sector input.",
            "suggestions": ["retail banking", "cloud computing", "fast fashion"],
        }


def llm_summarise_pages(llm, pages: List[Dict], max_words_each: int = 110) -> List[str]:
    summaries = []
    for p in pages:
        prompt = f"""
You are a market research assistant for a business analyst.
Summarise the Wikipedia page below into at most {max_words_each} words.
Focus on definition, scope, segments, stakeholders, economics/market aspects, technology, and trends.

PAGE TITLE: {p['title']}
PAGE URL: {p['url']}
PAGE TEXT:
{p['text']}
"""
        resp = llm.invoke(prompt)
        s = (resp.content or "").strip()
        summaries.append(enforce_word_limit(s, max_words_each))
    return summaries


def llm_write_report(llm, industry: str, page_summaries: List[str], urls: List[str]) -> str:
    joined = "\n\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(page_summaries)])
    prompt = f"""
You are a market research assistant. Write a concise industry report for a business analyst.

INDUSTRY: {industry}

Constraints:
- Less than 500 words total.
- Base the report ONLY on the five Wikipedia page summaries below.
- Include a short "Sources" line listing the five URLs at the end.

Sections:
1) Industry definition & scope
2) Value chain / key segments
3) Major drivers & trends
4) Risks / challenges
5) What a large corporation should watch next (2-3 bullets)

Wikipedia summaries:
{joined}

Five URLs:
{chr(10).join(urls)}
"""
    resp = llm.invoke(prompt)
    return enforce_word_limit((resp.content or "").strip(), 500)


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(
    page_title="ML4B Market Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Market Research Assistant (Wikipedia-based)")
st.caption("MSIN0231 ML4B Individual Assignment – Q1 to Q3")

with st.sidebar:
    st.header("Settings (Cost & Quality)")

    # API key FIRST (masked)
    api_key_input = st.text_input("Please enter your OpenAI API key", type="password")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    model_name = st.selectbox("LLM model (OpenAI)", options=["gpt-5-mini"], index=0)

    wiki_chars_per_page = st.slider(
        "Max Wikipedia characters per page (cost control)",
        min_value=1200,
        max_value=8000,
        value=3500,
        step=200
    )

# Gate app until API key exists (sidebar/secrets/env)
api_key_available = bool((api_key_input or "").strip()) or bool(os.getenv("OPENAI_API_KEY"))
try:
    api_key_available = api_key_available or bool(st.secrets.get("OPENAI_API_KEY"))
except Exception:
    pass

if not api_key_available:
    st.warning("Please enter your OpenAI API key in the sidebar to start using the app.")
    st.stop()

st.divider()
st.header("Step 1 (Q1): Provide an industry (Assistant self-check)")

industry_raw = st.text_input(
    "Enter an industry (e.g., 'Electric vehicles', 'Fast fashion', 'Cloud computing'):"
)
industry = clean_industry(industry_raw)

if not industry:
    st.info("Please enter an industry to continue.")
    st.stop()

# --- NEW: Step 1 justification happens HERE ---
# We run it only when the user clicks this button (so it doesn't call the LLM on every rerun).
justify_btn = st.button("Check if this is an industry", type="primary")

if justify_btn:
    with st.spinner("Assistant is validating your input…"):
        try:
            llm_check = build_llm(model_name=model_name, temperature=0.0, api_key_override=api_key_input)
            verdict = llm_is_industry_check(llm_check, industry)
            st.session_state["industry_verdict"] = verdict
        except Exception as e:
            st.error(f"Validation failed: {e}")
            st.stop()

# If we have a verdict, show it
verdict = st.session_state.get("industry_verdict")

if not verdict:
    st.info("Click **Check if this is an industry** to proceed. Step 2 will appear only if it is a valid industry.")
    st.stop()

if not verdict["is_industry"]:
    st.warning(f"Not an industry: {verdict['reason']}")
    st.write("Try an industry term like:")
    for s in verdict["suggestions"]:
        st.write(f"- {s}")
    # IMPORTANT: Stop here — do NOT move to Step 2
    st.stop()

st.success(f"Valid industry confirmed: **{industry}** — {verdict['reason']}")

# =============================================================================
# Step 2 (only appears if Step 1 passed)
# =============================================================================

st.divider()
st.header("Step 2 (Q2): Retrieve the 5 most relevant Wikipedia URLs")

run_retrieval = st.button("Find Wikipedia pages")

if run_retrieval:
    with st.spinner("Searching Wikipedia…"):
        try:
            docs = retrieve_wikipedia(industry, k=5)
            urls = extract_wikipedia_urls(docs)[:5]

            if len(urls) < 5:
                st.warning(
                    f"Only found {len(urls)} Wikipedia URL(s). "
                    "Please try a broader or more standard industry term."
                )
                st.stop()

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

            pages = [p for p in pages if p["url"] in urls][:5]

            st.session_state["wiki_urls"] = urls
            st.session_state["wiki_pages"] = pages

        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

if "wiki_urls" not in st.session_state:
    st.info("Click **Find Wikipedia pages** to retrieve and display the 5 URLs.")
    st.stop()

st.subheader("Q2 Output: Five most relevant Wikipedia URLs")
for i, u in enumerate(st.session_state["wiki_urls"], start=1):
    st.write(f"{i}. {u}")

# =============================================================================
# Step 3
# =============================================================================

st.divider()
st.header("Step 3 (Q3): Generate an industry report (<500 words)")

generate = st.button("Generate report", type="primary")

if generate:
    with st.spinner("Generating report…"):
        try:
            llm = build_llm(model_name=model_name, temperature=temperature, api_key_override=api_key_input)
            pages = st.session_state["wiki_pages"]
            urls = st.session_state["wiki_urls"]

            summaries = llm_summarise_pages(llm, pages, max_words_each=110)
            report = llm_write_report(llm, industry=industry, page_summaries=summaries, urls=urls)

            st.session_state["report"] = report
            st.session_state["summaries"] = summaries

        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.stop()

if "report" in st.session_state:
    st.subheader("Q3 Output: Industry report (≤ 500 words)")
    st.write(st.session_state["report"])
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
