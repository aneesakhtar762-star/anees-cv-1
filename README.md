# 📄 ATS Resume Checker

A Streamlit app that estimates how ATS-friendly a resume is and gives specific improvements.
Powered by the [Groq API](https://console.groq.com).

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for better keyword matching
- Overall ATS score (0-100) plus a 5-part breakdown: keywords, formatting, content impact, structure, readability
- Strengths, prioritized improvements, missing keywords and before/after rewrite examples
- Download the report as JSON

> The score is an AI-based estimate, not the output of a real ATS. Use it as guidance.

## Files
| File | Purpose |
|------|---------|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies (Streamlit Cloud needs this exact name) |
| `README.md` | This file |

## Run locally
1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key at https://console.groq.com/keys
4. Provide the key (either way works):
   - Environment variable: `export GROQ_API_KEY="gsk_..."` (Windows PowerShell: `$env:GROQ_API_KEY="gsk_..."`)
   - Or create `.streamlit/secrets.toml` containing:
     ```toml
     GROQ_API_KEY = "gsk_..."
     ```
   - Or just paste it into the sidebar when the app is running.
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Models
Default is `openai/gpt-oss-120b`. You can switch models in the sidebar or type any model ID from
https://console.groq.com/docs/models. Model availability changes, so update the `MODEL_OPTIONS`
list in `app.py` if a model is retired.

## Deploy on Streamlit Community Cloud
1. Push this project to a GitHub repository (files at the repo root).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**.

**Never commit your API key to GitHub.** Keep it only in Streamlit secrets or environment variables.

## Notes
- Scanned or image-only PDFs have no readable text; the app will ask for a text-based PDF or DOCX.
- Very long resumes are truncated to the first ~15,000 characters.
- Resume text is sent to Groq for analysis; don't upload documents you aren't comfortable sharing.
