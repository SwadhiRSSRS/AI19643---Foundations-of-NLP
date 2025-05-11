# modules/text_to_speech_emotion.py

from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import speedup
import os
import json

# 🎭 Emotion-to-Tone Speed Mapping
tone_map = {
    "admiration": 1.15,
    "amusement": 1.2,
    "anger": 0.95,
    "annoyance": 1.05,
    "approval": 1.1,
    "caring": 0.9,
    "confusion": 1.0,
    "curiosity": 1.1,
    "desire": 1.15,
    "disappointment": 0.85,
    "disapproval": 1.0,
    "disgust": 0.95,
    "embarrassment": 0.9,
    "excitement": 1.25,
    "fear": 0.9,
    "gratitude": 1.1,
    "grief": 0.8,
    "joy": 1.2,
    "love": 1.15,
    "nervousness": 0.95,
    "optimism": 1.1,
    "pride": 1.1,
    "realization": 0.95,
    "relief": 1.0,
    "remorse": 0.85,
    "sadness": 0.8,
    "surprise": 1.3,
    "neutral": 1.0
}
feedback_file = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\summary_minified.txt"
rephrase_file = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\rephrased_output.txt"
emotion_file = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\emotion_feedback.json"

def speak_feedback_summary_with_emotion(feedback_file="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\summary_minified.txt", rephrase_file="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\rephrased_output.txt", emotion_file="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\emotion_feedback.json", output_audio= "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\combined_feedback.mp3"):
    """
    Speaks feedback summary and rephrased text using overall emotion for tone modulation.

    Args:
        feedback_file (str): Path to feedback_summary.txt
        rephrase_file (str): Path to rephrased_output.txt
        emotion_file (str): Path to emotion_feedback.json
        output_audio (str): Output mp3 file path
    """

    # Load feedback content
    with open(feedback_file, "r", encoding="utf-8") as f:
        summary_text = f.read().strip()
    with open(rephrase_file, "r", encoding="utf-8") as f:
        rephrased_text = f.read().strip()

    # Load emotion data
    with open(emotion_file, "r", encoding="utf-8") as f:
        emotion_data = json.load(f)
        overall_emotion = emotion_data.get("overall_sentiment", "neutral").lower()
        if emotion_data.get("overall_sentiment") == "mixed":
            # fallback to most common emotion
            from collections import Counter
            emotions = [e["emotion"] for e in emotion_data["emotions_detected"]]
            most_common = Counter(emotions).most_common(1)[0][0].lower()
            overall_emotion = most_common
        else:
            overall_emotion = emotion_data.get("overall_sentiment", "neutral").lower()

        speech_speed = tone_map.get(overall_emotion, 1.0)

    # Combine feedback text
    combined_text = f"Here is your feedback summary. {summary_text}. Now, here's an improved version: {rephrased_text}"

    # Generate TTS
    tts = gTTS(text=combined_text, lang="en")
    temp_file = "temp_feedback.mp3"
    tts.save(temp_file)

    # Apply speed modulation
    segment = AudioSegment.from_mp3(temp_file)
    modulated_segment = speedup(segment, playback_speed=speech_speed)

    # Export final result
    modulated_segment.export(output_audio, format="mp3")
    print(f"✅ Combined feedback with emotion modulation saved to: {output_audio}")

    play(segment)
    os.remove(temp_file)



if __name__ == "__main__":
    feedback_file = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\summary_minified.txt"
    rephrase_file = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\rephrased_output.txt"
    emotion_file = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\emotion_feedback.json"
    speak_feedback_summary_with_emotion(feedback_file, rephrase_file, emotion_file)