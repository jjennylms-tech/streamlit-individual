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


@st.cache_data(show_spinner=False, ttl=1)
def retrieve_wikipedia(industry: str, k: int = 10):
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
    temperature = 0.0 (deterministic classifier)
    """
    prompt = f"""
You are a strict classifier for a market research assistant.

Decide whether the user input is an INDUSTRY / SECTOR / MARKET CATEGORY.

Rules:
- The input must be an INDUSTRY or SECTOR name, not a product, object, or item.
- REJECT single generic products or objects (e.g. "shoes", "car", "phone", "bag") — these are products, not industries.
- ACCEPT industry/sector terms (e.g. "footwear industry", "automotive industry", "consumer electronics").
- ACCEPT short valid sector abbreviations like "AI", "IT", "VR", "AR".
- ACCEPT phrases with numbers like "3D printing", "5G infrastructure", "Industry 4.0".
- If it is a greeting, random word, person, or place: NOT an industry.
- NEVER accept abbreviations or informal shorthand as valid — ALWAYS expand them to full formal industry names.
- Examples that MUST be expanded: "pharma" → "Pharmaceutical industry", "auto" → "Automotive industry", "tech" → "Technology industry", "fintech" → "Financial technology industry", "biotech" → "Biotechnology industry", "telco" → "Telecommunications industry"
- If the input is an abbreviation or shorthand, set "is_industry" to true BUT set "corrected" to the full formal name so the app forces the user to confirm.

Return ONLY valid JSON. For AMBIGUOUS inputs return exactly 5 suggestions. For MISSPELLED inputs return exactly 2 suggestions:
{{
  "is_industry": true/false,
  "reason": "one short sentence",
  "description": "1 sentence only: what this industry is and what it covers, no numbers or market size figures",
  "corrected": "the full formal industry name, never abbreviations or informal shorthand",
  "suggestions": [
    {{"label": "emoji + industry name 1", "description": "1 sentence about what this specific sub-industry covers and its key activities, no numbers or market size figures"}},
    {{"label": "emoji + industry name 2", "description": "1 sentence about what this specific sub-industry covers and its key activities, no numbers or market size figures"}},
    {{"label": "emoji + industry name 3", "description": "1 sentence about what this specific sub-industry covers and its key activities, no numbers or market size figures"}},
    {{"label": "emoji + industry name 4", "description": "1 sentence about what this specific sub-industry covers and its key activities, no numbers or market size figures"}},
  ]
}}

Rules for suggestions:
- "corrected" must be the single best corrected/interpreted industry name for the input.
- For AMBIGUOUS inputs (e.g. "energy", "tech", "health"), you MUST return EXACTLY 4 specific sub-industry options plus "✏️ Other (please type your industry)" as the 5th and final option. Never return fewer than 5 suggestions for ambiguous inputs.
- For MISSPELLED or INFORMAL inputs (e.g. "pharma", "automatives"), suggestions must contain ONLY 2 alternative related industries, different from "corrected".
- NEVER include the original broad/ambiguous/misspelled input as a suggestion.
- NEVER repeat the corrected name inside suggestions.
- All suggestion names must be REAL INDUSTRY NAMES only.
- Each suggestion (except "Other") must start with a relevant emoji that fits that specific industry.
- Do NOT suggest unrelated industries.
- Example for "energy": suggestions = ["☀️ Renewable energy industry", "🛢️ Oil & gas industry", "⚡ Electric power industry", "☢️ Nuclear energy industry", "✏️ Other (please type your industry)"]
- Example for "pharma": corrected = "Pharmaceutical industry", suggestions = ["💉 Biotechnology industry", "🏥 Healthcare & life sciences industry"]


User input: "{user_input}"
"""
    resp = llm.invoke(prompt)
    raw = (resp.content or "").strip()
    try:
        data = json.loads(raw)
        raw_suggestions = list(data.get("suggestions", []))[:5]
        # Handle both old string format and new dict format
        suggestions = []
        for s in raw_suggestions:
            if isinstance(s, dict):
                suggestions.append({
                    "label": s.get("label", ""),
                    "description": s.get("description", "")
                })
            else:
                suggestions.append({"label": str(s), "description": ""})

        if not suggestions:
            suggestions = [{"label": "retail banking", "description": ""},
                          {"label": "cloud computing", "description": ""},
                          {"label": "fast fashion", "description": ""}]
        return {
            "is_industry": bool(data.get("is_industry", False)),
            "reason": str(data.get("reason", "")).strip() or "No reason provided.",
            "description": str(data.get("description", "")).strip(),
            "corrected": str(data.get("corrected", user_input)).strip(),
            "suggestions": suggestions,
        }
    except Exception:
        return {
            "is_industry": False,
            "reason": "I couldn't confidently identify this as an industry/sector input.",
            "description": "",
            "corrected": user_input,
            "suggestions": ["retail banking", "cloud computing", "fast fashion"],
        }


def llm_filter_pages(llm, industry: str, pages: List[Dict]) -> List[Dict]:
    """
    Step 2 LLM relevance filter:
    - Retrieves 10 pages, LLM filters to 5 most relevant
    - temperature = 0.0 (deterministic selection)
    """
    titles_list = "\n".join([f"{i+1}. {p['title']}" for i, p in enumerate(pages)])
    prompt = f"""
You are a market research assistant.
The user is researching the industry: "{industry}"

Below are Wikipedia page titles that were retrieved.
Select EXACTLY 5 page numbers that are most relevant to this industry.

Prioritise in this order:
1. Global industry overview pages (e.g. "Pharmaceutical industry", "Cosmetic industry")
2. Major market segment pages (e.g. "Shampoo", "Hair coloring", "Skincare", "Generic drug")
3. Key technology or enabling pages (e.g. "Biotechnology", "Cloud computing architecture")
4. Well-known global companies ONLY if fewer than 5 broader pages are available

NEVER select:
- Country-specific pages unless fewer than 5 global pages are available
- Single company, brand or celebrity brand pages (e.g. "Olaplex", "CeraVe", "Huda Beauty")
- Individual people, historical figures, biographical or list-of-people pages
- Trade magazines, naming standards, chemical compounds or ingredient-specific pages
- Legal frameworks, education/academic programs or regulatory body pages
- Specific retail stores, salons or niche service channel pages

ALWAYS prefer:
- Global industry overview over country-specific versions
- Core product category pages over support function pages
- Segment pages (e.g. "Shampoo", "Hair coloring", "Lipstick", "Tourism", "Food service") over peripheral topics
- Fill remaining slots with segment or technology pages before selecting company pages

Return ONLY a JSON list of 5 numbers, e.g. [1, 3, 4, 7, 9]. No explanation.

Pages:
{titles_list}
"""
    resp = llm.invoke(prompt)
    raw = (resp.content or "").strip()
    try:
        relevant_indices = json.loads(raw)
        filtered = [pages[i-1] for i in relevant_indices if 1 <= i <= len(pages)][:5]
        if len(filtered) < 5:
            # fallback: fill remaining slots from original pages not already selected
            selected_titles = {p["title"] for p in filtered}
            for p in pages:
                if len(filtered) >= 5:
                    break
                if p["title"] not in selected_titles:
                    filtered.append(p)
                    selected_titles.add(p["title"])
        return filtered
    except Exception:
        return pages[:5]  # fallback: return first 5 if parsing fails


def llm_summarise_pages(llm, pages: List[Dict], max_words_each: int = 110) -> List[str]:
    """
    Step 3a: Summarise each Wikipedia page
    temperature = 0.1
    """
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
    """
    Step 3b: Write the final industry report
    temperature = 0.1
    """
    joined = "\n\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(page_summaries)])
    prompt = f"""
You are a market research assistant helping a business analyst at a large corporation.

INDUSTRY: {industry}

STRICT CONSTRAINTS:
- Total response must be UNDER 500 words (excluding Sources).
- Base the report ONLY on the five Wikipedia summaries below — do not add outside knowledge.
- Avoid filler phrases like "it is worth noting" or "in conclusion".
- If you see corrupted currency formatting like "150Bin2009" or "US150B", rewrite it as clean currency figures like "$150B in 2009".
- Do NOT number the sections. Use only the section title in bold markdown (e.g. **INDUSTRY DEFINITION & SCOPE**).
- Each section title must be followed by a blank line, then the content.
- Use bullet points for VALUE CHAIN & KEY PLAYERS, MARKET DRIVERS & TRENDS, RISKS & CHALLENGES, and STRATEGIC IMPLICATIONS.
- Use a short paragraph (no bullets) for INDUSTRY OVERVIEW.
- STRATEGIC IMPLICATIONS should have exactly 2-3 bullets.

Write the report in this exact structure:

**INDUSTRY OVERVIEW**
[short paragraph]

**VALUE CHAIN & KEY PLAYERS**
[bullet points]

**MARKET DRIVERS & TRENDS**
[bullet points]

**RISKS & CHALLENGES**
[bullet points]

**STRATEGIC IMPLICATIONS**
[2-3 bullet points]

**Sources:**
{chr(10).join(f"- {u}" for u in urls)}

Wikipedia summaries:
{joined}

Five URLs:
{chr(10).join(urls)}
"""
    resp = llm.invoke(prompt)
    return enforce_word_limit((resp.content or "").strip(), 500)


def format_report(report_text: str) -> str:
    """Clean and format report: remove numbering, promote ALL-CAPS titles to ## headers."""
    # Remove leading numbers like "1." "2." "1)" from section titles
    report_text = re.sub(r'^\s*\d+[\.\)]\s+(?=[A-Z])', '', report_text, flags=re.MULTILINE)
    # Make ALL-CAPS section titles into ## markdown headers
    report_text = re.sub(r'^(\*\*)?([A-Z][A-Z &\/\-]+)(\*\*)?$', r'## \2', report_text, flags=re.MULTILINE)
    return report_text


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(
    page_title="ML4B Market Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Market Research Assistant (Wikipedia-based)")

with st.sidebar:
    st.header("Settings")

    # API key (masked)
    api_key_input = st.text_input("Please enter your OpenAI API key", type="password")

    # Fixed settings (not shown to user as sliders)
    temperature = 0.2          # Step 3: report generation
    model_name = st.selectbox("LLM model (OpenAI)", options=["gpt-5-mini"], index=0)
    wiki_chars_per_page = 3500  # fixed cost control

# Gate app until API key exists
api_key_available = bool((api_key_input or "").strip()) or bool(os.getenv("OPENAI_API_KEY"))
try:
    api_key_available = api_key_available or bool(st.secrets.get("OPENAI_API_KEY"))
except Exception:
    pass

if not api_key_available:
    st.warning("Please enter your OpenAI API key in the sidebar to start using the app.")
    st.stop()

if st.button("🔄 Start New Search"):
    for key in ["industry_verdict", "wiki_urls", "wiki_pages", "report", 
                "summaries", "selected_industry", "selected_industry_display",
                "show_alternatives", "industry_confirmed", "selected_description", "other_confirmed"]:
        st.session_state.pop(key, None)
    st.rerun()

# =============================================================================
# Step 1 (Q1): Industry validation
# =============================================================================

st.divider()
st.header("Step 1: Provide an industry")

industry_raw = st.text_input(
    "Enter an industry (e.g., 'Electric vehicles', 'Fast fashion', 'Cloud computing'):"
)
industry = clean_industry(industry_raw)

if not industry:
    st.info("Please enter an industry to continue.")
    st.stop()

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

verdict = st.session_state.get("industry_verdict")

if not verdict:
    st.info("Click **Check if this is an industry** to proceed. Step 2 will appear only if it is a valid industry.")
    st.stop()

if not verdict["is_industry"]:
    st.warning(f"Not an industry: {verdict['reason']}")
    st.write("Try an industry term like:")
    for s in verdict["suggestions"]:
        label = s.get("label", "") if isinstance(s, dict) else s
        clean_label = re.sub(r'[^\w\s\&\-\(\)\/\,\.]', '', label).strip()
        st.write(f"- {clean_label}")
    st.stop()

corrected = verdict.get("corrected", industry)
description = verdict.get("description", "")
corrected = verdict.get("corrected", industry)
is_misspelled_or_informal = corrected.lower().strip() != industry.lower().strip()
is_ambiguous = any(word in industry.lower() for word in 
                   ["energy", "tech", "media", "transport", "finance", "food", "health", 
                    "beauty", "fashion", "retail", "sport", "entertainment", "travel"])

if is_misspelled_or_informal:
    # Case 1: Informal/misspelled — ask Yes/No confirmation
    st.info(f"🤔 Did you mean **'{corrected}'**?")
    if description:
        st.caption(f"📋 {description}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, proceed with this", use_container_width=True):
            st.session_state["selected_industry"] = corrected
            st.session_state["selected_industry_display"] = corrected
            st.session_state["industry_confirmed"] = True
            st.rerun()
    with col2:
        if st.button("❌ No, show me other options", use_container_width=True):
            st.session_state["show_alternatives"] = True
            st.rerun()

    if st.session_state.get("show_alternatives"):
        st.write("Select the industry you meant:")
        for i, suggestion in enumerate(verdict.get("suggestions", [])):
            label = suggestion.get("label", "") if isinstance(suggestion, dict) else suggestion
            desc = suggestion.get("description", "") if isinstance(suggestion, dict) else ""
            clean = re.sub(r'[^\w\s\&\-\(\)\/\,\.]', '', label).strip()
            selected = st.session_state.get("selected_industry", "")
            is_selected = selected and clean.lower() == selected.lower()
            if is_selected:
                st.markdown(
                    f"<div style='background-color:#e0e0e0; padding:10px; border-radius:8px; "
                    f"text-align:center; color:#555; margin-bottom:8px;'>{label} ✓</div>",
                    unsafe_allow_html=True
                )
                if desc:
                    st.caption(f"📋 {desc}")
            else:
                if st.button(f"{label}", key=f"alt_{i}", use_container_width=True):
                    st.session_state["selected_industry"] = clean
                    st.session_state["selected_industry_display"] = label
                    st.session_state["selected_description"] = desc
                    st.session_state["industry_confirmed"] = True
                    st.rerun()

elif is_ambiguous:
    st.warning(f"⚠️ **'{industry}'** is quite broad. Please select the specific sector you mean:")
    for i, suggestion in enumerate(verdict.get("suggestions", [])):
        label = suggestion.get("label", "") if isinstance(suggestion, dict) else suggestion
        desc = suggestion.get("description", "") if isinstance(suggestion, dict) else ""
        clean = re.sub(r'[^\w\s\&\-\(\)\/\,\.]', '', label).strip()
        selected = st.session_state.get("selected_industry", "")
        is_selected = selected and clean.lower() == selected.lower()
        if is_selected:
            st.markdown(
                f"<div style='background-color:#e0e0e0; padding:10px; border-radius:8px; "
                f"text-align:center; color:#555; margin-bottom:8px;'>{label} ✓</div>",
                unsafe_allow_html=True
            )
            if desc:
                st.caption(f"📋 {desc}")
        else:
            if st.button(f"{label}", key=f"suggest_{i}", use_container_width=True):
                st.session_state["selected_industry"] = clean
                st.session_state["selected_industry_display"] = label
                st.session_state["selected_description"] = desc
                st.session_state["industry_confirmed"] = True
                st.rerun()

else:
    # Case 3: Correct and specific — just confirm and show description
    st.success(f"✅ Valid industry confirmed: **{corrected}** — {verdict['reason']}")
    if description:
        st.info(f"{description}")
    st.session_state["selected_industry"] = corrected
    st.session_state["industry_confirmed"] = True

# Show final selected confirmation
# Update industry variable for downstream steps
if st.session_state.get("industry_confirmed"):
    industry = st.session_state.get("selected_industry", industry)

# Handle "Other" custom input
# Handle "Other" custom input
# Handle "Other" custom input
if "other" in (st.session_state.get("selected_industry") or "").lower() and "please type" in (st.session_state.get("selected_industry") or "").lower():
    custom = st.text_input("Please type your industry:", key="other_industry_input")
    if custom:
        clean_custom = clean_industry(custom)
        if st.button("✅ Confirm", key="confirm_custom"):
            st.session_state["selected_industry"] = clean_custom
            st.session_state["selected_industry_display"] = clean_custom
            st.session_state["selected_description"] = ""
            st.session_state["industry_confirmed"] = True
            st.session_state["other_confirmed"] = True
            st.rerun()

# Show confirmed other industry
if st.session_state.get("other_confirmed"):
    confirmed_industry = st.session_state.get("selected_industry", "")
    st.success(f"✅ Confirmed: **{confirmed_industry}**")
    industry = confirmed_industry

# =============================================================================
# Step 2 (Q2): Retrieve & filter Wikipedia URLs
# =============================================================================

st.divider()
st.header("Step 2: Retrieve the 5 most relevant Wikipedia URLs")

run_retrieval = st.button("Find Wikipedia pages")

if run_retrieval:
    with st.spinner("Searching Wikipedia and filtering for relevance…"):
        try:
            # Retrieve 10 pages
            docs = retrieve_wikipedia(industry, k=20)

            pages = []
            for d in docs:
                title = (d.metadata.get("title") if hasattr(d, "metadata") else None) or "Wikipedia page"
                url = (d.metadata.get("source") if hasattr(d, "metadata") else None) or ""
                text = getattr(d, "page_content", "") or ""
                if url:
                    pages.append({
                        "title": title,
                        "url": url,
                        "text": truncate_text_for_cost(text, wiki_chars_per_page)
                    })

            if len(pages) < 5:
                st.warning(
                    f"Only found {len(pages)} Wikipedia page(s). "
                    "Please try a broader or more standard industry term."
                )
                st.stop()

            # LLM filters 10 → 5 most relevant (temperature=0.0)
            llm_filter = build_llm(model_name=model_name, temperature=0.0, api_key_override=api_key_input)
            filtered_pages = llm_filter_pages(llm_filter, industry, pages)

            if len(filtered_pages) < 5:
                st.warning("Could not find 5 relevant Wikipedia pages. Try a broader industry term.")
                st.stop()

            filtered_urls = [p["url"] for p in filtered_pages]

            st.session_state["wiki_urls"] = filtered_urls
            st.session_state["wiki_pages"] = filtered_pages

        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            st.stop()

if "wiki_urls" not in st.session_state:
    st.info("Click **Find Wikipedia pages** to retrieve and display the 5 URLs.")
    st.stop()

st.subheader("Output: Five most relevant Wikipedia URLs")
for i, p in enumerate(st.session_state["wiki_pages"], start=1):
    st.write(f"{i}. **{p['title']}**: {p['url']}")

# =============================================================================
# Step 3 (Q3): Generate industry report
# =============================================================================

st.divider()
st.header("Step 3: Generate an industry report (<500 words)")

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
    st.subheader("Output: Industry report (≤ 500 words)")

    # Format report: remove numbers, make section titles large and bold
    formatted = format_report(st.session_state["report"])
    st.markdown(formatted)

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
