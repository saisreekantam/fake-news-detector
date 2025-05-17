from sentence_transformers import SentenceTransformer, util
import requests
from PIL import Image
import imagehash
from io import BytesIO
import logging

# Load the small model once globally (efficient and reusable)
model = SentenceTransformer('all-MiniLM-L6-v2')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_text_score(text1: str, text2: str) -> float:
    
    text1, text2 = text1.strip(), text2.strip()

    if not text1 or not text2:
        logger.warning("One or both input texts are empty.")
        return 100.0

    try:
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()  # Range: [-1, 1]
        distance = (1 - similarity) * 100  # Convert to percentage difference
        score = round(min(max(distance, 0.0), 100.0), 2)

        if score < 20:
            logger.info(f"High text similarity detected ({100 - score:.2f}%)")

        return score
    except Exception as e:
        logger.error(f"Text similarity error: {e}")
        return 100.0

def image_similarity_score(url1: str, url2: str) -> float:
    
    try:
        response1 = requests.get(url1, timeout=5)
        response2 = requests.get(url2, timeout=5)
        response1.raise_for_status()
        response2.raise_for_status()

        img1 = Image.open(BytesIO(response1.content)).convert("RGB")
        img2 = Image.open(BytesIO(response2.content)).convert("RGB")

        # Use multiple hashes for robustness
        hashers = [imagehash.average_hash, imagehash.phash, imagehash.dhash]
        distances = []

        for hasher in hashers:
            h1 = hasher(img1)
            h2 = hasher(img2)
            distances.append(h1 - h2)

        avg_distance = sum(distances) / len(distances)
        percent_diff = (avg_distance / 64.0) * 100
        score = round(min(max(percent_diff, 0.0), 100.0), 2)

        if score < 15:
            logger.info(f"Similar images detected ({100 - score:.2f}% match)")

        return score

    except Exception as e:
        logger.warning(f"[Image Similarity Error] {e}")
        return 100.0  
    