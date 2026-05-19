import os
import re
import io
import tempfile
from datetime import date

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MSDS Chemical Intelligence Chatbot",
    page_icon="⚗️",
    layout="wide"
)

# ── Lazy imports (heavy libs loaded after page config) ────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── API Key ───────────────────────────────────────────────────────────────────
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

# ── Embeddings (cached so model loads only once) ──────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db1" not in st.session_state:
    st.session_state.db1 = None
if "db2" not in st.session_state:
    st.session_state.db2 = None
if "chem1_name" not in st.session_state:
    st.session_state.chem1_name = None
if "chem2_name" not in st.session_state:
    st.session_state.chem2_name = None
if "chem1_loaded" not in st.session_state:
    st.session_state.chem1_loaded = False
if "chem2_loaded" not in st.session_state:
    st.session_state.chem2_loaded = False
if "message_count" not in st.session_state:
    st.session_state.message_count = 0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "input_key" not in st.session_state:          # ← NEW: for input-clear trick
    st.session_state.input_key = 0

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e0e1a; color: #e0e0e0; }
    .stSidebar { background-color: #16162a; }
    .chat-bubble {
        border-radius: 12px; padding: 12px 16px; margin: 8px 0;
        max-width: 85%; line-height: 1.6; font-size: 0.95rem;
    }
    .user-bubble   { background-color: #1e1e2e; margin-left: auto; text-align: right; }
    .chem1-bubble  { background-color: #0d3b6e; }
    .chem2-bubble  { background-color: #1a4731; }
    .comparison-bubble { background-color: #3b1f5e; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    .stTextInput>div>input {
        background-color: #1e1e2e; color: white; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# BACKEND HELPERS
# ═════════════════════════════════════════════════════════════════════════════

GHS_SECTIONS = {
    1: "Identification",
    2: "Hazard Identification",
    3: "Composition / Information on Ingredients",
    4: "First Aid Measures",
    5: "Fire Fighting Measures",
    6: "Accidental Release Measures",
    7: "Handling and Storage",
    8: "Exposure Controls / Personal Protection",
    9: "Physical and Chemical Properties",
    10: "Stability and Reactivity",
    11: "Toxicological Information",
    12: "Ecological Information",
    13: "Disposal Considerations",
    14: "Transport Information",
    15: "Regulatory Information",
    16: "Other Information",
}

SECTION_RE = re.compile(
    r"(?:SECTION|Section)\s+(\d+)|^(\d+)\.\s+[A-Z][a-zA-Z]",
    re.MULTILINE
)


def tag_section(chunk):
    """Add section_number and section_name metadata to a chunk."""
    match = SECTION_RE.search(chunk.page_content)
    if match:
        num = int(match.group(1) or match.group(2))
        chunk.metadata["section_number"] = num
        chunk.metadata["section_name"] = GHS_SECTIONS.get(num, f"Section {num}")
    return chunk


def infer_chemical_name(pages, fallback: str) -> str:
    """Try to extract product/chemical name from first 3 pages."""
    patterns = [
        r"Product\s+Name\s*[:\-]\s*(.+)",
        r"Chemical\s+Name\s*[:\-]\s*(.+)",
        r"Substance\s*[:\-]\s*(.+)",
        r"Trade\s+Name\s*[:\-]\s*(.+)",
    ]
    text = " ".join(p.page_content for p in pages[:3])
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().split("\n")[0].strip()
            if 2 < len(name) < 80:
                return name
    return fallback


def load_pdf_to_db(uploaded_file, embeddings):
    """Load PDF, chunk, tag sections, build FAISS. Returns (db, chem_name)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        chunks = [tag_section(c) for c in chunks]
        db = FAISS.from_documents(chunks, embeddings)
        fallback = os.path.splitext(uploaded_file.name)[0]
        chem_name = infer_chemical_name(pages, fallback)
        return db, chem_name
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Hazard highlighting ───────────────────────────────────────────────────────

TIER1 = ["FATAL", "POISON", "CARCINOGEN", "EXPLOSIVE", "RADIOACTIVE", "LD50", "LC50"]
TIER2 = ["TOXIC", "FLAMMABLE", "CORROSIVE", "OXIDIZER", "MUTAGEN", "DANGER"]
TIER3 = ["IRRITANT", "HARMFUL", "WARNING", "SENSITIZER"]


def highlight_hazards(text: str) -> str:
    for kw in TIER1:
        text = re.sub(
            rf"\b({kw})\b",
            r'<span style="color:red;font-weight:bold">\1</span>',
            text, flags=re.IGNORECASE
        )
    for kw in TIER2:
        text = re.sub(
            rf"\b({kw})\b",
            r'<span style="color:orange;font-weight:bold">\1</span>',
            text, flags=re.IGNORECASE
        )
    for kw in TIER3:
        text = re.sub(
            rf"\b({kw})\b",
            r'<span style="color:#FFD700;font-weight:bold">\1</span>',
            text, flags=re.IGNORECASE
        )
    return text


# ── Query routing ─────────────────────────────────────────────────────────────

COMPARISON_SIGNALS = [
    "compare", "contrast", "difference", "both", r"\bvs\b", "versus",
    "which is more", "which is less", "side by side", "side-by-side"
]


def detect_query_type(query: str, chem1_name, chem2_name) -> str:
    q = query.lower()
    for sig in COMPARISON_SIGNALS:
        if re.search(sig, q, re.IGNORECASE):
            return "comparison"
    if chem1_name and (chem1_name.lower() in q or
                       any(x in q for x in ["chemical 1", "chem 1", "first one", "first chemical"])):
        return "chem1"
    if chem2_name and (chem2_name.lower() in q or
                       any(x in q for x in ["chemical 2", "chem 2", "second one", "second chemical"])):
        return "chem2"
    loaded1 = st.session_state.chem1_loaded
    loaded2 = st.session_state.chem2_loaded
    if loaded1 and not loaded2:
        return "chem1"
    if loaded2 and not loaded1:
        return "chem2"
    if loaded1 and loaded2:
        return "ambiguous"
    return "ambiguous"


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_RULES = """
You are an MSDS document reader. You answer ONLY based on the text provided to you from the
MSDS document. You do NOT use any external chemistry knowledge. You do NOT infer, approximate,
or guess any values.

Rules you must follow without exception:
1. If a piece of information is not explicitly present in the document text provided, respond
   with "Not specified in document" for that field — never infer or estimate.
2. All numerical values (LD50, flash point, boiling point, exposure limits, concentrations)
   must be quoted VERBATIM from the source text — do not paraphrase numbers.
3. Every answer must cite its source as: [Section X — Section Name, Page Y]
4. Do not use general chemistry knowledge to fill gaps.
5. For comparison queries: present each chemical's data in clearly labeled side-by-side format.
   If one chemical has a field and the other does not, explicitly state
   "[Chemical Name]: Not specified in document" for the missing one.
6. For exhaustive queries ("tell me everything", "full summary"): walk through all 16 GHS
   sections in order. For any section not found in the document, write
   "Section X — [Name]: Not found in document."
"""

single_prompt = PromptTemplate(
    input_variables=["context", "chemical_name", "question", "history"],
    template=SYSTEM_RULES + """

Chemical being queried: {chemical_name}

Conversation history (last 6 turns):
{history}

Relevant document content:
{context}

Question: {question}

Answer (cite section and page for every fact):"""
)

comparison_prompt = PromptTemplate(
    input_variables=["context1", "chem1_name", "context2", "chem2_name", "question", "history"],
    template=SYSTEM_RULES + """

You are comparing TWO chemicals. Present answers side-by-side, clearly labeled.

Chemical 1: {chem1_name}
Chemical 2: {chem2_name}

Conversation history (last 6 turns):
{history}

Document content for {chem1_name}:
{context1}

Document content for {chem2_name}:
{context2}

Question: {question}

Comparison answer (label every fact with chemical name + section + page):"""
)


def build_history_string() -> str:
    turns = st.session_state.chat_history[-6:]
    lines = []
    for msg in turns:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def get_answer(query: str, query_type: str) -> str:
    api_key = get_groq_key()
    if not api_key:
        return "⚠️ GROQ_API_KEY not set. Please add it to Streamlit secrets."

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=api_key)
    history = build_history_string()

    try:
        if query_type == "chem1":
            docs = st.session_state.db1.similarity_search(query, k=6)
            context = "\n\n".join(d.page_content for d in docs)
            chain = single_prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "chemical_name": st.session_state.chem1_name,
                "question": query,
                "history": history,
            })

        elif query_type == "chem2":
            docs = st.session_state.db2.similarity_search(query, k=6)
            context = "\n\n".join(d.page_content for d in docs)
            chain = single_prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "chemical_name": st.session_state.chem2_name,
                "question": query,
                "history": history,
            })

        elif query_type == "comparison":
            docs1 = st.session_state.db1.similarity_search(query, k=6)
            docs2 = st.session_state.db2.similarity_search(query, k=6)
            context1 = "\n\n".join(d.page_content for d in docs1)
            context2 = "\n\n".join(d.page_content for d in docs2)
            chain = comparison_prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context1": context1,
                "chem1_name": st.session_state.chem1_name,
                "context2": context2,
                "chem2_name": st.session_state.chem2_name,
                "question": query,
                "history": history,
            })

        else:  # ambiguous
            names = []
            if st.session_state.chem1_name:
                names.append(st.session_state.chem1_name)
            if st.session_state.chem2_name:
                names.append(st.session_state.chem2_name)
            answer = (
                f"Please specify which chemical you mean: **{' or '.join(names)}**, "
                f"or say **'both'** to compare them."
            )

    except Exception as e:
        answer = f"⚠️ LLM Error: {e}. Please check your Groq API key in Streamlit secrets."

    # Append to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "source": query_type
    })
    st.session_state.message_count += 1
    return answer


# ── Chat rendering ────────────────────────────────────────────────────────────

def render_chat():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble user-bubble"><b>You</b><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            source = msg.get("source", "chem1")
            content = highlight_hazards(msg["content"])
            if source == "chem1":
                label = f'⚗️ <span style="color:#5ba3f5">{st.session_state.chem1_name or "Chemical 1"}:</span>'
                css_class = "chem1-bubble"
            elif source == "chem2":
                label = f'⚗️ <span style="color:#5fdb8a">{st.session_state.chem2_name or "Chemical 2"}:</span>'
                css_class = "chem2-bubble"
            elif source == "comparison":
                label = '⚗️ <span style="color:#c084fc">Comparison:</span>'
                css_class = "comparison-bubble"
            else:
                label = "⚗️ Assistant:"
                css_class = "chem1-bubble"
            st.markdown(
                f'<div class="chat-bubble {css_class}">{label}<br>{content}</div>',
                unsafe_allow_html=True
            )


# ── Export helpers ────────────────────────────────────────────────────────────

def export_pdf(messages: list, title: str) -> bytes:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            rightMargin=60, leftMargin=60,
                            topMargin=60, bottomMargin=60)
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle("heading", parent=styles["Heading2"],
                                   textColor=colors.HexColor("#1B3A6B"))
    body_style = styles["BodyText"]
    body_style.leading = 16

    story = []
    story.append(Paragraph("MSDS Chemical Intelligence Report", styles["Title"]))
    story.append(Paragraph(title, styles["Heading3"]))
    story.append(Paragraph(f"Generated: {date.today()}", styles["Normal"]))
    story.append(Spacer(1, 20))

    for msg in messages:
        role = "You" if msg["role"] == "user" else f"Assistant ({msg.get('source','').upper()})"
        # Strip HTML tags for PDF
        clean = re.sub(r"<[^>]+>", "", msg["content"])
        story.append(Paragraph(role, heading_style))
        story.append(Paragraph(clean, body_style))
        story.append(Spacer(1, 10))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def export_docx(messages: list, title: str) -> bytes:
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = DocxDocument()
    doc.add_heading("MSDS Chemical Intelligence Report", 0)
    doc.add_paragraph(title)
    p = doc.add_paragraph(f"Generated: {date.today()}")
    p.runs[0].italic = True

    for msg in messages:
        role = "You" if msg["role"] == "user" else f"Assistant ({msg.get('source','').upper()})"
        clean = re.sub(r"<[^>]+>", "", msg["content"])
        h = doc.add_heading(role, level=2)
        h.runs[0].font.color.rgb = RGBColor(0x1B, 0x3A, 0x6B)
        doc.add_paragraph(clean)
        doc.add_paragraph("")

    footer_section = doc.sections[0]
    footer = footer_section.footer
    footer.paragraphs[0].text = f"Generated by MSDS Chatbot | {date.today()}"

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def filter_messages_for_export(export_type: str) -> tuple[list, str]:
    msgs = st.session_state.chat_history
    names = " & ".join(filter(None, [st.session_state.chem1_name, st.session_state.chem2_name]))

    if export_type == "Full Chat Transcript":
        return msgs, names
    elif export_type == "Last Summary Only":
        # Find the last assistant message that looks like a summary
        for msg in reversed(msgs):
            if msg["role"] == "assistant" and "SECTION" in msg["content"].upper():
                return [msg], names
        return msgs[-2:] if len(msgs) >= 2 else msgs, names
    elif export_type == "Comparison Report":
        filtered = [m for m in msgs if m.get("source") == "comparison"]
        return filtered or msgs, names
    return msgs, names


# ═════════════════════════════════════════════════════════════════════════════
# PRESET QUERY DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

PRESETS = {
    "⚠️ Hazards":      "What are all the hazard identifications for this chemical?",
    "🧤 PPE":           "What personal protective equipment (PPE) is required for this chemical?",
    "🚑 First Aid":     "What are the first aid measures for this chemical?",
    "🧪 Properties":    "What are the physical and chemical properties of this chemical?",
    "🏭 Storage":       "What are the storage and handling requirements for this chemical?",
    "⚡ Reactivity":    "What is the reactivity and stability information for this chemical?",
    "☠️ Toxicology":   "What is the full toxicological information for this chemical?",
    "📋 Regulatory":    "What regulatory information applies to this chemical?",
    "📞 Emergency":     "What are the emergency contact details and emergency response procedures?",
}


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

embeddings = load_embeddings()

with st.sidebar:
    st.title(" MSDS Magic")
    st.caption("Chemical Intelligence Platform")
    st.markdown("---")

    # ── Chemical 1 loader ────────────────────────────────────────────────────
    st.subheader("📂 Chemical 1")
    file1 = st.file_uploader("Load Chemical 1 (PDF)", type=["pdf"], key="upload1",
                              label_visibility="collapsed")
    if file1:
        with st.spinner(f"Processing {file1.name}…"):
            try:
                old_name = st.session_state.chem1_name
                db, name = load_pdf_to_db(file1, embeddings)
                st.session_state.db1 = db
                st.session_state.chem1_name = name
                st.session_state.chem1_loaded = True
                if old_name and old_name != name:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"ℹ️ **{old_name}** has been replaced with **{name}** in Slot 1. Previous conversation history preserved.",
                        "source": "system"
                    })
                st.success(f"✅ {name} loaded successfully")
            except Exception as e:
                st.error(f"Failed to load PDF: {e}")

    if st.session_state.chem1_loaded:
        st.success(f"✅ {st.session_state.chem1_name} — Ready")
    else:
        st.info("⬜ No chemical loaded")

    st.markdown("---")

    # ── Chemical 2 loader ────────────────────────────────────────────────────
    st.subheader("📂 Chemical 2")
    file2 = st.file_uploader("Load Chemical 2 (PDF)", type=["pdf"], key="upload2",
                              label_visibility="collapsed")
    if file2:
        with st.spinner(f"Processing {file2.name}…"):
            try:
                old_name = st.session_state.chem2_name
                db, name = load_pdf_to_db(file2, embeddings)
                st.session_state.db2 = db
                st.session_state.chem2_name = name
                st.session_state.chem2_loaded = True
                if old_name and old_name != name:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"ℹ️ **{old_name}** has been replaced with **{name}** in Slot 2. Previous conversation history preserved.",
                        "source": "system"
                    })
                st.success(f"✅ {name} loaded successfully")
            except Exception as e:
                st.error(f"Failed to load PDF: {e}")

    if st.session_state.chem2_loaded:
        st.success(f"✅ {st.session_state.chem2_name} — Ready")
    else:
        st.info("⬜ No chemical loaded")

    st.markdown("---")

    # ── Full Summary ──────────────────────────────────────────────────────────
    st.subheader("📋 Full MSDS Summary")
    summary_options = []
    if st.session_state.chem1_loaded:
        summary_options.append("Chemical 1")
    if st.session_state.chem2_loaded:
        summary_options.append("Chemical 2")
    if st.session_state.chem1_loaded and st.session_state.chem2_loaded:
        summary_options.append("Both")

    if summary_options:
        summary_target = st.radio("Generate for:", summary_options)
        if st.button("Generate Full Summary"):
            exhaustive_q = (
                "Provide an exhaustive summary of all 16 GHS sections of this MSDS in order. "
                "For each section, extract every piece of data present. "
                "If a section is not found in the document, state 'Not found in document'."
            )
            with st.spinner("Generating exhaustive MSDS summary…"):
                if summary_target == "Both":
                    qtype = "comparison"
                elif summary_target == "Chemical 2":
                    qtype = "chem2"
                else:
                    qtype = "chem1"
                chem_label = (
                    f"{st.session_state.chem1_name} & {st.session_state.chem2_name}"
                    if qtype == "comparison"
                    else (st.session_state.chem1_name if qtype == "chem1" else st.session_state.chem2_name)
                )
                header = f"📋 **FULL MSDS SUMMARY — {chem_label} — {date.today()}**\n\n"
                answer = get_answer(exhaustive_q, qtype)
                # Prepend header to last assistant message
                if st.session_state.chat_history:
                    last = st.session_state.chat_history[-1]
                    if last["role"] == "assistant":
                        last["content"] = header + last["content"]
            st.rerun()
    else:
        st.info("Load at least one chemical first.")

    st.markdown("---")

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("📤 Export")
    export_format = st.selectbox("Format", ["PDF", "Word (.docx)"])
    export_type = st.selectbox("Export", [
        "Full Chat Transcript", "Last Summary Only", "Comparison Report"
    ])
    if st.button("Export Now"):
        if not st.session_state.chat_history:
            st.warning("No chat history to export yet.")
        else:
            try:
                msgs, title = filter_messages_for_export(export_type)
                if export_format == "PDF":
                    data = export_pdf(msgs, title)
                    st.download_button(
                        "⬇️ Download PDF", data=data,
                        file_name=f"msds_report_{date.today()}.pdf",
                        mime="application/pdf"
                    )
                else:
                    data = export_docx(msgs, title)
                    st.download_button(
                        "⬇️ Download Word", data=data,
                        file_name=f"msds_report_{date.today()}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            except Exception as e:
                st.error(f"Export failed: {e}")

    st.markdown("---")

    # ── Session info + reset ──────────────────────────────────────────────────
    st.caption(f"💬 {st.session_state.message_count} messages this session")
    if st.button("🗑️ New Chat"):
        for key in ["chat_history", "db1", "db2", "chem1_name", "chem2_name",
                    "chem1_loaded", "chem2_loaded", "message_count", "pending_query",
                    "input_key", "_last_input"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═════════════════════════════════════════════════════════════════════════════

st.title("⚗️ MSDS Chemical Intelligence Chatbot")

if not st.session_state.chem1_loaded and not st.session_state.chem2_loaded:
    st.info("👈 Load one or two MSDS PDFs from the sidebar to get started.")

# ── Chat display ──────────────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    render_chat()

# ── Preset buttons ────────────────────────────────────────────────────────────
if st.session_state.chem1_loaded or st.session_state.chem2_loaded:
    st.markdown("**Quick Queries:**")
    preset_cols = st.columns(len(PRESETS))
    both_loaded = st.session_state.chem1_loaded and st.session_state.chem2_loaded

    for col, (label, query_text) in zip(preset_cols, PRESETS.items()):
        with col:
            if st.button(label, key=f"preset_{label}"):
                if both_loaded:
                    st.session_state[f"show_target_{label}"] = True
                else:
                    qtype = "chem1" if st.session_state.chem1_loaded else "chem2"
                    with st.spinner("Thinking…"):
                        get_answer(query_text, qtype)
                    st.rerun()

    # Show target selector if both loaded
    for label, query_text in PRESETS.items():
        if st.session_state.get(f"show_target_{label}"):
            target = st.selectbox(
                f"Ask about — {label}",
                ["Chemical 1", "Chemical 2", "Both"],
                key=f"target_sel_{label}"
            )
            if st.button(f"Ask ↑ ({label})", key=f"ask_{label}"):
                qtype = "chem1" if target == "Chemical 1" else ("chem2" if target == "Chemical 2" else "comparison")
                with st.spinner("Thinking…"):
                    get_answer(query_text, qtype)
                del st.session_state[f"show_target_{label}"]
                st.rerun()

    st.markdown("---")

# ── Input box ─────────────────────────────────────────────────────────────────
col_input, col_send = st.columns([6, 1])
with col_input:
    user_input = st.text_input(
        "",
        placeholder="Ask anything about the loaded chemical(s)…",
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_key}"   # ← dynamic key clears field on increment
    )
with col_send:
    send_btn = st.button("Send ↑", use_container_width=True)

# ── Handle send ───────────────────────────────────────────────────────────────
if send_btn or (user_input and st.session_state.get("_last_input") != user_input):
    if not st.session_state.chem1_loaded and not st.session_state.chem2_loaded:
        st.warning("Please load at least one MSDS PDF first.")
    elif not user_input.strip():
        pass
    else:
        st.session_state["_last_input"] = user_input
        qtype = detect_query_type(
            user_input,
            st.session_state.chem1_name,
            st.session_state.chem2_name
        )
        # Guard: comparison needs both loaded
        if qtype == "comparison" and not (st.session_state.chem1_loaded and st.session_state.chem2_loaded):
            qtype = "chem1" if st.session_state.chem1_loaded else "chem2"

        with st.spinner("Thinking…"):
            get_answer(user_input, qtype)

        st.session_state.input_key += 1   # ← increment key → widget re-renders blank
        st.rerun()
