# MSDS Chemical Intelligence Chatbot

**Live App:** [Add your Streamlit link here]

A Streamlit app for querying and comparing chemical safety data sheets (MSDS/SDS) through a chatbot that is deliberately designed to refuse to guess. Safety data sheets are the kind of document where a wrong or made-up number is actually dangerous, so this project's core design principle is strict grounding: every answer must trace back to the actual document text, and if the information isn't there, the app says so instead of filling the gap.

## Features

- **Dual-document support** — load up to two MSDS PDFs simultaneously and query either one individually or run a structured side-by-side comparison
- **GHS-structured retrieval** — document chunks are automatically tagged against the 16 standard GHS (Globally Harmonized System) sections that real safety data sheets are organized around — Hazard Identification, First Aid Measures, Toxicological Information, and so on — so the app's retrieval and full-summary features respect the actual structure of the document type, not just generic text chunks
- **Strict anti-hallucination design:**
  - The model is instructed to use only the retrieved document text — no outside chemistry knowledge
  - Numerical values (LD50, flash point, exposure limits, etc.) must be quoted exactly as written, never paraphrased or rounded
  - Every fact is cited with its section and page number
  - Missing information is explicitly reported as "Not specified in document" rather than inferred
- **Intent routing** — automatically detects whether a question is about chemical 1, chemical 2, or a comparison of both, based on keywords and which documents are loaded, and asks for clarification when it's genuinely ambiguous
- **Hazard highlighting** — hazard-related terms are color-coded by severity tier directly in the response text
- **Export** — chat transcripts, the last summary, or a comparison-only report can be exported to PDF or Word

## Tech Stack, and why each piece is here

- **LangChain** — handles the pipeline of loading the PDF, splitting it into chunks, and assembling the final prompt sent to the model.
- **PyPDFLoader** — extracts raw text from the uploaded PDF, page by page, which matters because page numbers need to be preserved for citation.
- **FAISS** — the vector search index, with a separate one built for each loaded chemical, so chemical 1's content is never accidentally mixed into a chemical 2 answer.
- **HuggingFace `all-MiniLM-L6-v2`** — converts each document chunk into a numeric vector representing its meaning, which is what makes "find the most relevant part of this document" possible.
- **Groq (Llama 3.3 70B)** — the language model that reads the retrieved chunks and writes the final answer, constrained by the strict grounding rules in the system prompt.
- **reportlab / python-docx** — generate the exportable PDF and Word reports.
- **Streamlit** — builds the chat interface, file upload sidebar, and export controls.

## How It Works, step by step

1. **Loading a document.** When you upload an MSDS PDF, its text is extracted page by page.
2. **Chunking with section awareness.** The text is split into chunks (~800 characters, with 100-character overlap), and each chunk is scanned with a regex pattern to detect which of the 16 GHS sections it belongs to (e.g., "SECTION 7" or "7. Handling and Storage"). That section number and name are attached to the chunk as metadata.
3. **Building the searchable index.** Each chunk is embedded and stored in its own FAISS index — one index for chemical 1, a separate one for chemical 2 if a second document is loaded.
4. **Figuring out what you're asking.** When you type a question, a query classifier checks for comparison language ("compare," "vs," "both"), mentions of a specific chemical's name, or which document(s) are currently loaded, to decide whether to answer from chemical 1's index, chemical 2's index, or both.
5. **Retrieving and answering.** The top 6 most relevant chunks are pulled from the appropriate index (or both, for a comparison) and inserted into a prompt template that lays out the strict rules: quote numbers exactly, cite section and page for every fact, and say "Not specified in document" for anything missing. This prompt goes to the Groq-hosted model, and the response comes back already following that structure.
6. **Display and export.** Hazard-related keywords in the response are highlighted by severity, and the full conversation (or just the comparison portions, or just the latest summary) can be exported as a formatted PDF or Word document.

## Setup

```bash
pip install -r requirements.txt
```

Add your Groq API key to Streamlit secrets:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "your-key-here"
```

## Usage

```bash
streamlit run app.py
```

Load one or two MSDS PDFs from the sidebar, then use the preset quick-query buttons or type a free-form question. Use "Generate Full Summary" for a structured walkthrough of all 16 GHS sections.
