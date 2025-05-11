import os
import re
import json
import subprocess
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# === File Paths
BASE_DIR = "D:\\swadhi\\6 th sem\\NLP_Project\\modules"
RAW_TEXT_PATH = os.path.join(BASE_DIR, "raw_text.txt")
EMOTION_PATH = os.path.join(BASE_DIR, "emotion_feedback.json")
FALLACY_PATH = os.path.join(BASE_DIR, "logical_fallacy_detection_results.json")
FEEDBACK_SUMMARY_PATH = os.path.join(BASE_DIR, "feedback_summary.txt")
REPHRASED_PATH = os.path.join(BASE_DIR, "rephrased_output.txt")
SUMMARY_OUTPUT_PATH = os.path.join(BASE_DIR, "summary_minified.txt")

# === Tone Mapping
tone_map = {
    "admiration": "[respectful tone]", "amusement": "[playful tone]", "anger": "[calm tone]",
    "annoyance": "[neutralizing tone]", "approval": "[encouraging tone]", "caring": "[compassionate tone]",
    "confusion": "[clarifying tone]", "curiosity": "[inquisitive tone]", "desire": "[passionate tone]",
    "disappointment": "[sympathetic tone]", "disapproval": "[firm tone]", "disgust": "[neutral tone]",
    "embarrassment": "[gentle tone]", "excitement": "[enthusiastic tone]", "fear": "[reassuring tone]",
    "gratitude": "[grateful tone]", "grief": "[soft tone]", "joy": "[cheerful tone]",
    "love": "[affectionate tone]", "nervousness": "[soothing tone]", "optimism": "[uplifting tone]",
    "pride": "[confident tone]", "realization": "[insightful tone]", "relief": "[relaxed tone]",
    "remorse": "[apologetic tone]", "sadness": "[empathetic tone]", "surprise": "[expressive tone]",
    "neutral": "[normal tone]"
}


# === Step 1: Feedback Generation
def generate_feedback_summary():
    with open(RAW_TEXT_PATH, "r", encoding='utf-8') as f:
        raw_text = f.read()

    with open(EMOTION_PATH, "r", encoding='utf-8') as f:
        emotion_data = json.load(f)

    with open(FALLACY_PATH, "r", encoding='utf-8') as f:
        fallacy_data = json.load(f)

    sentences = sent_tokenize(raw_text)
    feedback_lines = []

    for idx, sentence in enumerate(sentences):
        emotion = emotion_data["emotions_detected"][idx]
        fallacy = fallacy_data[idx]
        tone = tone_map.get(emotion["emotion"].lower(), "[neutral tone]")

        feedback = f" Sentence {idx+1}:\n"
        feedback += f"Original: {sentence}\n"
        feedback += f"- Emotion: {emotion['emotion']} {tone}\n"
        feedback += f"- Fallacy: {fallacy['fallacy_detected']}\n"

        if fallacy["fallacy_detected"] != "None":
            feedback += "⚠️ Consider revising due to logical fallacy.\n"

        if emotion["emotion"].lower() in ["anger", "disgust"]:
            feedback += "💡 Try using softer language to sound more persuasive.\n"

        feedback_lines.append(feedback)

    with open(FEEDBACK_SUMMARY_PATH, "w", encoding='utf-8') as f:
        f.write("\n".join(feedback_lines))

    print("✅ Feedback summary saved.")


# === Step 2: Rephrasing with Gemma
def clean_response(text):
    cleaned = re.sub(r"^(Sure,? )?(here (is|are|was) .*?:\s*)?", "", text, flags=re.IGNORECASE).strip()
    return cleaned.replace("*", "").strip()


def prompt_ollama_rephrase(sentence, emotion, fallacy):
    prompt = f"""
You are an expert AI communication and debate coach to improve the debating skills of a human.

Rephrase the sentence for better tone, clarity, and logical strength.
Apply appropriate tone modulation tags like [calm tone], [cheerful tone], etc., based on emotion and logic.

Only return the clean rephrased sentence.

Sentence: {sentence}
Emotion: {emotion}
Fallacy: {fallacy}
"""

    result = subprocess.run(["ollama", "run", "gemma:2b"],
                            input=prompt.encode("utf-8"),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    return clean_response(result.stdout.decode("utf-8"))


def rephrase_text():
    with open(RAW_TEXT_PATH, "r", encoding='utf-8') as f:
        raw_text = f.read()
    sentences = sent_tokenize(raw_text)

    with open(EMOTION_PATH, "r", encoding='utf-8') as f:
        emotion_data = json.load(f)
    with open(FALLACY_PATH, "r", encoding='utf-8') as f:
        fallacy_data = json.load(f)

    rephrased_sentences = []

    for idx in range(len(sentences)):
        sentence = sentences[idx]
        emotion = emotion_data["emotions_detected"][idx]
        fallacy = fallacy_data[idx]["fallacy_detected"]
        print(f"🔁 Rephrasing Sentence {idx+1}...")

        rephrased = prompt_ollama_rephrase(sentence, emotion["emotion"], fallacy)
        rephrased_sentences.append(rephrased)

    with open(REPHRASED_PATH, "w", encoding='utf-8') as f:
        f.write(" ".join(rephrased_sentences))

    print("✅ Rephrased text saved.")


# === Optional: Feedback Summary Summarization
def summarize_feedback_summary(input_path=FEEDBACK_SUMMARY_PATH, output_path=SUMMARY_OUTPUT_PATH):
    with open(input_path, "r", encoding='utf-8') as f:
        full_summary = f.read()

    prompt = f"""
You are an expert communication coach.
Analyze the following input. Identify key weaknesses or improvement areas. Then, provide a short paragraph (3–5 lines) suggesting what the speaker/writer can improve — such as tone, clarity, word choice, sentence structure, or logic.
Be direct and practical. Summarize from the provided content. 

Text:
\"\"\"
{full_summary}
\"\"\"

Return only the cleaned paragraph
"""

    result = subprocess.run(["ollama", "run", "gemma:2b"],
                            input=prompt.encode("utf-8"),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    summary = clean_response(result.stdout.decode("utf-8"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print("📝 Minified feedback summary saved.")


def run_feedback_generation():
    print("🎯 Stage 1: Generating Feedback Summary...")
    generate_feedback_summary()

    print("\n✍️ Stage 2: Rephrasing Text...")
    rephrase_text()

    summarize_feedback_summary()
# === MAIN EXECUTION
if __name__ == "__main__":
    run_feedback_generation()