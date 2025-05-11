import json
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nltk.download("punkt", quiet=True)

# Load emotion classification model
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-go-emotion",
    top_k=None  
)


def analyze_emotions(text):
    """
    Analyzes sentence-wise emotions in a given text using the GoEmotion model.
    
    Returns: list of dicts with sentence, emotion label, and confidence score.
    """
    sentences = sent_tokenize(text)
    results = []

    for sentence in sentences:
        prediction_scores = emotion_classifier(sentence)[0]
        top = max(prediction_scores, key=lambda x: x["score"])
        results.append({
            "sentence": sentence,
            "emotion": top["label"],
            "confidence": round(top["score"], 2)
        })

    return results


def summarize_emotion_feedback(emotion_results):
    """
    Creates a summary of emotional tone, transitions, and dominant emotion.
    """
    emotions = [r["emotion"] for r in emotion_results]
    most_common = Counter(emotions).most_common(1)[0][0]

    transitions = []
    for i in range(1, len(emotions)):
        if emotions[i] != emotions[i - 1]:
            transitions.append({"from": emotions[i - 1], "to": emotions[i]})

    return {
        "overall_sentiment": most_common if len(set(emotions)) == 1 else "mixed",
        "emotions_detected": emotion_results,
        "emotion_transitions": transitions
    }


def run_emotion_analysis(input_path="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\raw_text.txt", output_path="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\emotion_feedback.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    emotions = analyze_emotions(text)
    feedback = summarize_emotion_feedback(emotions)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2)

    print(f"✅ Emotion feedback saved to '{output_path}'.")


# === Run module directly (for testing)
if __name__ == "__main__":
    input_file = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\raw_text.txt"
    output_file = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\emotion_feedback.json"
    run_emotion_analysis(input_file, output_file)
