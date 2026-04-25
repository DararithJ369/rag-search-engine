from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies

class MultimodalSearch:
    def __init__(self, documents: list[str], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def embed_image(self, image_fpath: str) -> list[float]:
        img = Image.open(image_fpath)
        return self.model.encode([img])[0]
    
    def search_with_image(self, image_fpath: str, limit: int) -> list[dict]:
        img_emb = self.embed_image(image_fpath)
        
        similarities = []
        for idx, text_emb in enumerate(self.text_embeddings):
            _similarity = cosine_similarity(img_emb, text_emb)
            similarities.append((idx, _similarity))
            
        sorted_sims = sorted(similarities, key=lambda x: x[1], reverse=True) 
        sorted_sims = sorted_sims[:limit]
        results = []
        for idx, score in sorted_sims:
            _doc = self.documents[idx]
            results.append({
                'title': _doc['title'],
                'description': _doc['description'],
                'doc_id': idx,
                'score': score
            })
        return results
    
def image_search_command(image_fpath: str, limit: int = 5):
    movies = load_movies()
    ms = MultimodalSearch(movies)
    res = ms.search_with_image(image_fpath, limit)
    for idx, r in enumerate(res, 1):
        print(f"{idx}. {r['title']} (similarity: {r['score']:.4f})")
        print(f"        {r['description'][:100]}...\n")

def verify_image_embedding(image_fpath: str):
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_fpath)
    print(f"Embedding shape: {embedding.shape[0]} dimensions ")