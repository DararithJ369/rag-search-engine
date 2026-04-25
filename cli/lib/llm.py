import os, json
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from .search_utils import PROMPTS_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

model = "gemini-2.5-flash"
client = genai.Client(api_key=api_key) if api_key else None

def _format_docs(documents) -> str:
    # Default: numbered style for general use
    if documents is None:
        return ""
    if isinstance(documents, str):
        return documents
    if isinstance(documents, dict):
        documents = [documents]
    if not documents:
        return ""
    if not isinstance(documents, (list, tuple)):
        return str(documents)
    lines = []
    for idx, doc in enumerate(documents, 1):
        title = doc.get("title", "")
        description = doc.get("description", "")
        lines.append(f"[{idx}] {title}: {description}")
    return "\n".join(lines)

def generate_content(prompt: str, query: str, **kwargs) -> str:
    if not client:
        raise RuntimeError("Gemini API client not available. Set GEMINI_API_KEY environment variable.")
    
    try:
        prompt = prompt.format(query=query, **kwargs)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text
    except genai_errors.ClientError as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise RuntimeError(
                "Gemini API quota exceeded. Please try again later."
            ) from e
        raise RuntimeError(f"Gemini API error: {error_msg}") from e

def augment_prompt(query: str, method: str) -> str:
    with open(PROMPTS_PATH / f'{method}.md', 'r') as f:
        prompt = f.read()
    return generate_content(prompt, query)

def correct_spelling(query: str) -> str:
    return augment_prompt(query, "spelling")
def rewrite_query(query: str) -> str:
    return augment_prompt(query, "rewrite")
def expand_query(query: str) -> str:
    return augment_prompt(query, "expand")

def llm_judge(query: str, formatted_results: str) -> str:
    with open(PROMPTS_PATH / 'llm_judge.md', 'r') as f:
        prompt = f.read()
    response = generate_content(prompt, query, formatted_results=formatted_results)
    results = json.loads(response)
    return results

def _rag(query: str, documents: list[dict], prompt_fname: str) -> str:
    with open(PROMPTS_PATH / prompt_fname, 'r') as f:
        prompt = f.read()
    if prompt_fname == 'answer_with_citations.md':
        # [1], [2], etc. style
        docs = "\n".join([
            f"[{idx}] {doc.get('title', '')}: {doc.get('description', '')}"
            for idx, doc in enumerate(documents, 1)
        ])
    elif prompt_fname == 'answer_question_detailed.md':
        # doc_id style
        docs = "\n".join([
            f"doc_id: {doc.get('doc_id', '')} | {doc.get('title', '')}: {doc.get('description', '')}"
            for doc in documents
        ])
    else:
        docs = _format_docs(documents)
    return generate_content(prompt, query, docs=docs)

def answer_question(query: str, documents: list[dict]) -> str:
    return _rag(query, documents, 'answer_question.md')

def summarize_documents(query: str, documents: list[dict]) -> str:
    return _rag(query, documents, 'summarization.md')

def citations_documents(query: str, documents: list[dict]) -> str:
    return _rag(query, documents, 'answer_with_citations.md')

def detailed_question_answering(query: str, documents: list[dict]) -> str:
    return _rag(query, documents, 'answer_question_detailed.md')