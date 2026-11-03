"""ATS Resume Checker - Streamlit + Groq.

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get
an estimated ATS score with concrete improvements.
"""

import io
import json
import os
import re

import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# Model IDs from https://console.groq.com/docs/models (check there for updates).
MODEL_OPTIONS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B (production, best quality)",
    "openai/gpt-oss-20b": "GPT-OSS 20B (production, fastest/cheapest)",
    "qwen/qwen3.8-27b": "Qwen3.8 27B (preview)",
}
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_RESUME_CHARS = 15000
MAX_JD_CHARS = 6000

# Weights used to compute the overall score in Python (so it is consistent).
WEIGHTS = {
    "keywords_match": 0.30,
    "formatting_parsability": 0.20,
    "content_impact": 0.25,
    "structure_sections": 0.15,
    "readability_length": 0.10,
}
LABELS = {
    "keywords_match": "Keywords & role match",
    "formatting_parsability": "Formatting & ATS parsability",
    "content_impact": "Content & measurable impact",
    "structure_sections": "Structure & sections",
    "readability_length": "Readability & length",
}

SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and \
senior technical recruiter. You evaluate resumes strictly and honestly.
Respond with ONE valid JSON object only. No markdown, no commentary."""

USER_PROMPT_TEMPLATE = """Evaluate the resume below for ATS compatibility and quality.
{jd_block}
Score each category from 0 to 100 (be strict; 90+ is rare):
- keywords_match: relevant skills/keywords {jd_hint}
- formatting_parsability: clean text, standard headings, no signs of tables/columns/graphics garbling the text, consistent dates, contact info present
- content_impact: action verbs, quantified achievements, clarity of results
- structure_sections: presence/order of Summary, Experience, Education, Skills, Projects etc.
- readability_length: concise bullets, no fluff, appropriate length, no typos

Return JSON with exactly this schema:
{{
  "category_scores": {{
    "keywords_match": <int>,
    "formatting_parsability": <int>,
    "content_impact": <int>,
    "structure_sections": <int>,
    "readability_length": <int>
  }},
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<short point>", "..."],
  "improvements": [
    {{"priority": "High|Medium|Low", "issue": "<what is wrong>", "fix": "<specific action>"}}
  ],
  "missing_keywords": ["<keyword>", "..."],
  "rewrite_examples": [
    {{"before": "<weak line copied from resume>", "after": "<stronger rewrite>"}}
  ]
}}
Rules: 4-8 strengths max, 5-10 improvements ordered by priority, up to 15 missing keywords, \
2-4 rewrite examples. Only use facts present in the resume; never invent metrics in rewrites \
(use placeholders like [X%] where a number is needed).

RESUME:
\"\"\"
{resume}
\"\"\"
"""


# ----------------------------------------------------------------------------
# Core logic (no Streamlit calls, so it is easy to test)
# ----------------------------------------------------------------------------
def extract_text(file_name: str, data: bytes) -> str:
    """Extract plain text from PDF, DOCX or TXT bytes."""
    name = file_name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()
    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:  # resumes sometimes put content in tables
            for row in table.rows:
                parts.extend(cell.text for cell in row.cells)
        return "\n".join(p for p in parts if p.strip()).strip()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()
    raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")


def build_messages(resume_text: str, job_description: str = "") -> list:
    resume_text = resume_text[:MAX_RESUME_CHARS]
    job_description = (job_description or "").strip()[:MAX_JD_CHARS]
    if job_description:
        jd_block = f'\nTarget job description:\n"""\n{job_description}\n"""\n'
        jd_hint = "compared with the target job description"
    else:
        jd_block = "\nNo job description was given; judge against general industry expectations for the role the resume targets.\n"
        jd_hint = "for the role this resume appears to target"
    user = USER_PROMPT_TEMPLATE.format(
        jd_block=jd_block, jd_hint=jd_hint, resume=resume_text
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_json_response(raw: str) -> dict:
    """Parse model output into a dict, tolerating code fences or extra text."""
    if not raw:
        raise ValueError("The model returned an empty response.")
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("Could not parse the model response as JSON.")


def _clamp(value) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def normalize_result(data: dict) -> dict:
    """Validate the model output and compute the overall score in Python."""
    raw_scores = data.get("category_scores") or {}
    scores = {k: _clamp(raw_scores.get(k, 0)) for k in WEIGHTS}
    overall = round(sum(scores[k] * w for k, w in WEIGHTS.items()))

    def as_list(key):
        v = data.get(key)
        return v if isinstance(v, list) else []

    improvements = []
    for item in as_list("improvements"):
        if isinstance(item, dict):
            improvements.append(
                {
                    "priority": str(item.get("priority", "Medium")).title(),
                    "issue": str(item.get("issue", "")).strip(),
                    "fix": str(item.get("fix", "")).strip(),
                }
            )
    order = {"High": 0, "Medium": 1, "Low": 2}
    improvements.sort(key=lambda i: order.get(i["priority"], 1))

    rewrites = [
        {"before": str(r.get("before", "")), "after": str(r.get("after", ""))}
        for r in as_list("rewrite_examples")
        if isinstance(r, dict)
    ]
    return {
        "overall": overall,
        "category_scores": scores,
        "summary": str(data.get("summary", "")).strip(),
        "strengths": [str(s) for s in as_list("strengths")],
        "improvements": improvements,
        "missing_keywords": [str(k) for k in as_list("missing_keywords")],
        "rewrite_examples": rewrites,
    }


def score_band(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Decent - needs work"
    return "Needs significant improvement"


def analyze_resume(client, model: str, resume_text: str, job_description: str = "") -> dict:
    """Call Groq and return a normalized result dict."""
    messages = build_messages(resume_text, job_description)
    kwargs = dict(model=model, messages=messages, temperature=0.2, max_tokens=4000)
    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except Exception as exc:  # some models reject JSON mode; retry without it
        if "response_format" in str(exc) or "json" in str(exc).lower():
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    raw = resp.choices[0].message.content
    return normalize_result(parse_json_response(raw))


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file configured
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


def render_results(result: dict) -> None:
    overall = result["overall"]
    st.subheader("Your ATS score")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall (estimate)", f"{overall}/100")
        st.caption(score_band(overall))
    with c2:
        st.progress(overall / 100)
        if result["summary"]:
            st.write(result["summary"])

    st.markdown("#### Score breakdown")
    for key, label in LABELS.items():
        val = result["category_scores"][key]
        st.write(f"**{label}** - {val}/100")
        st.progress(val / 100)

    if result["strengths"]:
        st.markdown("#### Strengths")
        for s in result["strengths"]:
            st.markdown(f"- {s}")

    if result["improvements"]:
        st.markdown("#### Suggested improvements")
        icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
        for imp in result["improvements"]:
            icon = icons.get(imp["priority"], "🟠")
            with st.expander(f"{icon} {imp['priority']}: {imp['issue']}"):
                st.write(imp["fix"])

    if result["missing_keywords"]:
        st.markdown("#### Missing or weak keywords")
        st.write(", ".join(f"`{k}`" for k in result["missing_keywords"]))

    if result["rewrite_examples"]:
        st.markdown("#### Example rewrites")
        for ex in result["rewrite_examples"]:
            st.markdown(f"**Before:** {ex['before']}")
            st.markdown(f"**After:** {ex['after']}")
            st.divider()

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(result, indent=2),
        file_name="ats_report.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
    st.title("📄 ATS Resume Checker")
    st.write(
        "Upload your resume to get an estimated ATS score and specific ways to improve it."
    )

    with st.sidebar:
        st.header("Settings")
        api_key = get_api_key()
        if not api_key:
            api_key = st.text_input("Groq API key", type="password", help="Get one at console.groq.com")
        else:
            st.success("API key loaded from secrets.")
        model = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL),
            format_func=lambda m: MODEL_OPTIONS[m],
        )
        custom = st.text_input("Or custom Groq model ID (optional)")
        if custom.strip():
            model = custom.strip()
        st.caption(
            "Scores are an AI-based estimate, not the output of a real ATS. "
            "Use them as guidance."
        )

    uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    job_description = st.text_area(
        "Job description (optional, improves keyword matching)", height=150
    )

    if st.button("Analyze resume", type="primary"):
        if not api_key:
            st.error("Please add your Groq API key in the sidebar or in Streamlit secrets.")
            return
        if uploaded is None:
            st.error("Please upload a resume first.")
            return
        try:
            text = extract_text(uploaded.name, uploaded.getvalue())
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
        if len(text) < 100:
            st.error(
                "Almost no text could be extracted. If your PDF is a scan or image, "
                "an ATS can't read it either - export a text-based PDF or DOCX."
            )
            return
        try:
            client = Groq(api_key=api_key)
            with st.spinner("Analyzing your resume..."):
                result = analyze_resume(client, model, text, job_description)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return
        render_results(result)


if __name__ == "__main__":
    main()
