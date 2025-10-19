import os
import re
import json
import zipfile
from flask import Flask, request, jsonify, send_file, render_template
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers  import StrOutputParser
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from dotenv import load_dotenv
from io import BytesIO

# -----------------------------
# INITIAL SETUP
# -----------------------------
app = Flask(__name__)
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "gemini-2.5-pro"
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# LANGCHAIN PROMPT TEMPLATE
# -----------------------------
prompt_template = PromptTemplate(
    input_variables=["resume_text"],
    template="""
    You are a resume formatter assistant.

Return **only valid JSON** (no markdown, no explanation) in the format:

{{
  "name": "...",  
  "Professional_Summary": "...",
  "Technical_Skill_Sets": [...],
  "Work_Experience": [
     {{
        "Company": "...",
        "Role": "...",
        "Duration": "...",
        "Key_Responsibilities": [...]
     }}
  ],
  "Academic_Background": [
    {{
      "Qualification": "...",
      "Institute": "...",
      "percentage"...."
    }}
  ],
  "Certifications": [...],
  "Extra_Curricular_Activities": [...]
}}

RESUME TEXT:
{resume_text}
"""
)

# -----------------------------
# LOAD RESUME (Supports PDF & DOCX)
# -----------------------------
def load_resume(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type.")
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])
    return full_text

# -----------------------------
# LLM PROCESS
# -----------------------------
def convert_to_json(resume_text: str):
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"resume_text": resume_text})

    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("⚠️ Failed JSON parse. Returning fallback.")
        return {"error": "Invalid JSON from LLM", "raw": response}

# -----------------------------
# GENERATE COMPANY PDF
# -----------------------------
def generate_company_resume(data_json, name="Candidate", contact="example@email.com", filename="resume"):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("company_resume_exact_template.html")

    html_content = template.render(name=name, contact=contact, data=data_json)
    pdf_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_formatted.pdf")

    HTML(string=html_content).write_pdf(pdf_path)
    return pdf_path

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_resume():
    files = request.files.getlist("resume")
    name = request.form.get("name", "Candidate")
    contact = request.form.get("contact", "example@email.com")

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    pdf_paths = []
    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # Process each file
        text = load_resume(path)
        resume_json = convert_to_json(text)
        if "error" in resume_json:
            continue

        pdf_path = generate_company_resume(resume_json, name, contact, file.filename)
        pdf_paths.append(pdf_path)

    # Create ZIP
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for pdf in pdf_paths:
            zipf.write(pdf, os.path.basename(pdf))
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name="Formatted_Resumes.zip",
        mimetype="application/zip"
    )

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)
