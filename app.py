import os, io, time
from flask import Flask, render_template, request
from PIL import Image
import PyPDF2
from google import genai

# --------- CONFIG ----------


API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Run: set GEMINI_API_KEY=YOUR_KEY")

client = genai.Client(api_key=API_KEY)

# Use a current model (per Google docs)
TEXT_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-flash"  # multimodal

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB


def extract_pdf_text(pdf_file) -> str:
    """Typed/Selectable-text PDFs only (no OCR)."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        parts = []
        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            if t:
                parts.append(t)
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def build_prompt(mode: str, question: str, notes_text: str) -> str:
    base = (
        "You are FocusStudy AI, an exam-focused tutor.\n"
        "Use ONLY the provided notes. If notes do not contain the answer, say:\n"
        "\"I can't find this in the provided notes.\" Then give a short general hint (2-3 lines max).\n"
        "Format cleanly using headings and bullet points.\n"
    )

    mode_line = {
        "explain": "Mode: EXPLAIN. Explain simply and give one small example.\n",
        "exam": "Mode: EXAM READY. Write a structured exam answer with headings + key points.\n",
        "revision": "Mode: REVISION. Give crisp revision bullets.\n",
    }.get(mode, "Mode: EXPLAIN.\n")

    notes_block = f"\nNOTES:\n{notes_text}\n" if notes_text.strip() else "\nNOTES: (none)\n"
    return f"{base}{mode_line}{notes_block}\nQUESTION:\n{question}\n\nANSWER:\n"


def format_html(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = text.replace("\n\n", "<br><br>")
    text = text.replace("\n", "<br>")
    return text


def gen_with_retry(model: str, contents, tries: int = 3):
    last_err = None
    for i in range(tries):
        try:
            resp = client.models.generate_content(model=model, contents=contents)
            return getattr(resp, "text", "") or ""
        except Exception as e:
            last_err = e
            time.sleep(2 * (i + 1))
    raise last_err


def ocr_image_with_gemini(image_file) -> str:
    img_bytes = image_file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    ocr_prompt = (
        "Extract the handwritten text clearly and accurately. "
        "Return ONLY the extracted text. No extra commentary."
    )
    return (gen_with_retry(VISION_MODEL, [ocr_prompt, pil_img]) or "").strip()


@app.route("/", methods=["GET", "POST"])
def home():
    answer_html = ""
    error = ""

    if request.method == "POST":
        mode = request.form.get("mode", "explain")
        question = (request.form.get("question") or "").strip()

        pdf = request.files.get("pdf")
        image = request.files.get("image")

        if not question:
            error = "Please type your question."
            return render_template("index.html", answer=answer_html, error=error)

        notes_text = ""

        if pdf and pdf.filename:
            pdf_text = extract_pdf_text(pdf)
            if not pdf_text:
                error = "This PDF looks scanned/handwritten. Please upload clear images instead."
                return render_template("index.html", answer=answer_html, error=error)
            notes_text = pdf_text

        if image and image.filename:
            try:
                extracted = ocr_image_with_gemini(image)
                if extracted:
                    notes_text = (notes_text + "\n\n" + extracted).strip() if notes_text else extracted
            except Exception as e:
                error = f"Image OCR failed: {str(e)}"
                return render_template("index.html", answer=answer_html, error=error)

        prompt = build_prompt(mode, question, notes_text)

        try:
            final_text = gen_with_retry(TEXT_MODEL, prompt)
            answer_html = format_html(final_text)
        except Exception as e:
            error = f"Gemini error: {str(e)}"

    return render_template("index.html", answer=answer_html, error=error)


if __name__ == "__main__":
    app.run(debug=True)

