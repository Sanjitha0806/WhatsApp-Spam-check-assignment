import csv
import re
from collections import Counter
from pathlib import Path
import streamlit as st

DATA_FILE = Path(__file__).with_name("spam.csv")

def preprocess_message(message, lowercase=True, replace_urls=True, replace_emails=True,
                       remove_numbers=False, remove_punctuation=False):
    text = str(message)
    if lowercase:
        text = text.lower()
    if replace_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    if replace_emails:
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " EMAIL ", text)
    if remove_numbers:
        text = re.sub(r"\d+", " NUMBER ", text)
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_messages():
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    cleaned = []
    for row in rows:
        label = (row.get("label") or row.get("v1") or "").strip().lower()
        message = row.get("message") or row.get("v2") or ""
        if label and message:
            cleaned.append({"label": label, "message": message})
    return cleaned

st.set_page_config(page_title="WhatsApp Spam Text Preprocessing", layout="wide")
st.title("WhatsApp / SMS Spam — Text Preprocessing")
st.caption("Dataset exploration and text preprocessing only — no ML model or TF-IDF.")
try:
    rows = load_messages()
except Exception as exc:
    st.error(f"Could not read spam.csv. Keep it beside app.py. Details: {exc}")
    st.stop()

with st.sidebar:
    st.header("Preprocessing options")
    lowercase = st.checkbox("Lowercase text", value=True)
    replace_urls = st.checkbox("Replace URLs with URL", value=True)
    replace_emails = st.checkbox("Replace emails with EMAIL", value=True)
    remove_numbers = st.checkbox("Remove numbers", value=False)
    remove_punctuation = st.checkbox("Remove punctuation", value=False)

for row in rows:
    row["processed"] = preprocess_message(row["message"], lowercase, replace_urls,
                                           replace_emails, remove_numbers, remove_punctuation)
counts = Counter(row["label"] for row in rows)
a, b, c = st.columns(3)
a.metric("Messages", len(rows))
b.metric("Ham", counts.get("ham", 0))
c.metric("Spam", counts.get("spam", 0))

st.subheader("Class balance")
total = max(len(rows), 1)
for name in ("ham", "spam"):
    amount = counts.get(name, 0)
    pct = amount / total * 100
    st.markdown(f"**{name.upper()} — {amount} ({pct:.1f}%)**")
    st.markdown(f"<div style='background:#e9eef5;border-radius:6px;height:18px;width:100%'><div style='background:{'#2471c8' if name == 'ham' else '#e07835'};width:{pct:.1f}%;height:18px;border-radius:6px'></div></div>", unsafe_allow_html=True)

st.subheader("Raw message and preprocessed result")
choice = st.selectbox("Show messages", ["All", "ham", "spam"])
shown = [r for r in rows if choice == "All" or r["label"] == choice][:20]
for i, row in enumerate(shown, start=1):
    with st.expander(f"{row['label'].upper()} message {i}"):
        st.write("**Raw:**", row["message"])
        st.write("**Preprocessed:**", row["processed"])

st.subheader("Try preprocessing your own message")
example = st.text_area("Enter a message", "WIN £1000 now!!! Click https://example.com")
if example:
    st.write("**Preprocessed text:**")
    st.code(preprocess_message(example, lowercase, replace_urls, replace_emails,
                               remove_numbers, remove_punctuation))
