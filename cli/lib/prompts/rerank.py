import os, json
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from ..search_utils import PROMPTS_PATH
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

model = "gemini-2.5-flash"
client = genai.Client(api_key=api_key) if api_key else None

def individual_rank(query: str, documents: dict) -> list[dict]:
    if not client:
        raise RuntimeError("Gemini API client not available. Set GEMINI_API_KEY environment variable.")
    
    with open(PROMPTS_PATH / 'individual_rerank.md', 'r') as f:
        prompt = f.read()
    results = []
    for doc in documents:
        try:
            _prompt = prompt.format(
                query=query, 
                title=doc['title'], 
                description=doc['description']
            )
            response = client.models.generate_content(model=model, contents=_prompt)
            clean_response_text = (response.text or "").strip()
            try: 
                clean_response_text = int(clean_response_text)
            except:
                print(f"Failed to cast {clean_response_text} to int for {doc['title']}, defaulting to 0")
                clean_response_text = 0
            results.append({**doc, 'rerank_response': clean_response_text})
        except genai_errors.ClientError as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise RuntimeError(
                    "Gemini API quota exceeded. Please try again later."
                ) from e
            raise RuntimeError(f"Gemini API error: {error_msg}") from e
    results = sorted(results, key=lambda x: x['rerank_response'], reverse=True)
    return results

def batch_rank(query: str, documents: dict) -> list[dict]:
    if not client:
        raise RuntimeError("Gemini API client not available. Set GEMINI_API_KEY environment variable.")

    with open(PROMPTS_PATH / 'batch_rerank.md', 'r') as f:
        prompt = f.read()
    _mtemp = '''<movie title={title}>\n{desc}\n</movie>\n'''
    doc_list_str = ""
    for doc in documents:
        doc_list_str += _mtemp.format(title=doc['title'], desc=doc['description'])
        
    _prompt = prompt.format(
        query=query,
        doc_list_str=doc_list_str
    )
    response = client.models.generate_content(model=model, contents=_prompt)
    print(response.text) 
    response_parsed = json.loads(response.text)
    results = []
    for idx, doc in enumerate(documents):
        results.append({**doc, 'rerank_score': int(response_parsed[idx])})
    results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    return results
        
def cross_encoder_rerank(query: str, documents: dict) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    # scores is a list of numbers, one for each pair
    scores = cross_encoder.predict(pairs)
    results = []
    for idx, doc in enumerate(documents):
        results.append({**doc, 'cross_encoder_score': scores[idx]})
        
    results = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)
    return results