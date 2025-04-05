import gradio as gr
import os
import requests
import pdfplumber
import docx
import spacy
import nltk
from textstat import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

# Ensure required NLTK packages
nltk.download('punkt')

# Load spaCy model (make sure it's installed in requirements.txt)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Disable grammar tool if Java is missing
try:
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')
    grammar_check_enabled = True
except Exception:
    grammar_check_enabled = False

# Setup GROQ API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
RECOMMENDED_GROQ_MODEL = "llama3-70b-8192"

# FILE READER
def read_file(file):
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".txt":
        return file.read().decode("utf-8")
    elif ext == ".pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif ext == ".docx":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode("utf-8", errors='ignore')

# SMART SUMMARY
def smart_summarize(text, summary_words):
    if not GROQ_API_KEY:
        return "GROQ API key not set. Please set the GROQ_API_KEY environment variable."

    prompt = f"""
Please summarize the following content in approximately {summary_words} words:
Text:
{text[:16000]}
"""
    try:
        response = groq_client.chat.completions.create(
            model=RECOMMENDED_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(8192, summary_words * 3)
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error using GROQ model: {e}"

# ANALYSIS FUNCTION
def analyze_text(text, summary_words):
    if not text:
        return "No text provided.", "", "", ""

    summary = smart_summarize(text, summary_words)

    tfidf = TfidfVectorizer(stop_words='english', max_features=5)
    tfidf.fit([text])
    keywords = ", ".join(tfidf.get_feature_names_out())

    if grammar_check_enabled:
        matches = tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        grammar_feedback = f"{len(matches)} issues found.\n\nCorrected Version:\n{corrected[:500]}..."
    else:
        grammar_feedback = "Grammar checking disabled (Java not available)."

    readability_score = textstat.flesch_reading_ease(text)
    doc = nlp(text)
    word_count = len([token for token in doc if token.is_alpha])

    return summary, f"Words: {word_count}\nReadability: {readability_score:.2f}", f"Key Topics: {keywords}", grammar_feedback

# HANDLERS
def handle_upload(file, summary_words):
    text = read_file(file)
    return analyze_text(text, summary_words)

def handle_cloud_url(url, summary_words):
    response = requests.get(url)
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        f.write(response.content)
    with open(filename, "rb") as file:
        result = handle_upload(file, summary_words)
    os.remove(filename)
    return result

def handle_editor_input(text, summary_words):
    return analyze_text(text, summary_words)

# GRADIO UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 Text Analyzer Tool
    Analyze, summarize, and extract key insights from any text:
    - ✅ AI Summarization (GROQ LLaMA3)
    - ✅ Key Topics Extraction
    - ✅ Grammar & Readability Score
    - ✅ Word Count Analyzer
    """)

    with gr.Tab("📁 Upload from Local"):
        file_input = gr.File(label="Upload File")
        summary_length_input = gr.Number(value=150, label="Summary Length (words)")
        btn1 = gr.Button("🔍 Analyze")
        output1 = gr.Textbox(label="📄 Summary", lines=10)
        output2 = gr.Textbox(label="📊 Word Count & Readability", lines=3)
        output3 = gr.Textbox(label="🧠 Key Topics", lines=3)
        output4 = gr.Textbox(label="🔧 Grammar Suggestions", lines=8)
        btn1.click(fn=handle_upload, inputs=[file_input, summary_length_input], outputs=[output1, output2, output3, output4])

    with gr.Tab("🌐 Upload from URL"):
        url_input = gr.Textbox(label="Paste URL")
        url_summary_length = gr.Number(value=150, label="Summary Length (words)")
        btn2 = gr.Button("📥 Analyze from URL")
        url_output1 = gr.Textbox(label="📄 Summary", lines=10)
        url_output2 = gr.Textbox(label="📊 Word Count & Readability", lines=3)
        url_output3 = gr.Textbox(label="🧠 Key Topics", lines=3)
        url_output4 = gr.Textbox(label="🔧 Grammar Suggestions", lines=8)
        btn2.click(fn=handle_cloud_url, inputs=[url_input, url_summary_length], outputs=[url_output1, url_output2, url_output3, url_output4])

    with gr.Tab("✍️ Paste Text"):
        text_editor = gr.Textbox(label="Paste text here", lines=15)
        editor_summary_length = gr.Number(value=150, label="Summary Length (words)")
        btn3 = gr.Button("🔍 Analyze Text")
        editor_output1 = gr.Textbox(label="📄 Summary", lines=10)
        editor_output2 = gr.Textbox(label="📊 Word Count & Readability", lines=3)
        editor_output3 = gr.Textbox(label="🧠 Key Topics", lines=3)
        editor_output4 = gr.Textbox(label="🔧 Grammar Suggestions", lines=8)
        btn3.click(fn=handle_editor_input, inputs=[text_editor, editor_summary_length], outputs=[editor_output1, editor_output2, editor_output3, editor_output4])

demo.launch()
