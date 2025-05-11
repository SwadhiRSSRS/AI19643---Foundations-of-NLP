import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


# === Define fallacies with their simplified explanations
FALLACY_TEMPLATES = {
    "Ad Hominem": "attacks the character of the opponent rather than the argument",
    "Strawman": "misrepresents an argument to make it easier to attack",
    "Appeal to Emotion": "manipulates emotional responses instead of presenting valid arguments",
    "False Cause": "assumes correlation implies causation",
    "Hasty Generalization": "draws a conclusion based on insufficient evidence"
}

# === Load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")
fallacy_embeddings = model.encode(list(FALLACY_TEMPLATES.values()), convert_to_tensor=True)


def detect_fallacies(segments, threshold=0.6):
    """
    Detects potential logical fallacies in argument segments using semantic similarity.
    
    Parameters:
    - segments: list of {sentence, label}
    - threshold: cosine similarity threshold to flag fallacy

    Returns:
    - detection_results: list of dictionaries with fallacy type, explanation, etc.
    """
    results = []
    for segment in segments:
        sentence = segment["sentence"]
        sentence_embedding = model.encode(sentence, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(sentence_embedding, fallacy_embeddings)[0]
        best_idx = int(np.argmax(similarities))
        max_score = float(similarities[best_idx])

        if max_score >= threshold:
            fallacy = list(FALLACY_TEMPLATES.keys())[best_idx]
            explanation = FALLACY_TEMPLATES[fallacy]
        else:
            fallacy = "None"
            explanation = "No significant fallacy detected."

        results.append({
            "label": segment["label"],
            "sentence": sentence,
            "fallacy_detected": fallacy,
            "explanation": explanation,
            "similarity_score": round(max_score, 3)
        })

    return results


def run_fallacy_detection(input_json_path="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\classified_argument_segments.json", output_json_path="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\logical_fallacy_detection_results.json", threshold=0.6):
    with open(input_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    detection_results = detect_fallacies(segments, threshold)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(detection_results, f, indent=2)

    print(f"✅ Fallacy detection complete. Results saved to '{output_json_path}'.")


# === Entry point for testing
if __name__ == "__main__":
    input_path = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\classified_argument_segments.json"
    output_path = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\logical_fallacy_detection_results.json"
    run_fallacy_detection(input_path, output_path)
