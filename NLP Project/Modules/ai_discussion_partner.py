import subprocess
import re
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import speedup

from modules.speech_to_text import transcribe_file, record_audio  # 🎤 Whisper functions


# === Clean Text ===
def clean_text_for_tts(text):
    text = re.sub(r"[\*\_`]", "", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


# === Get Discussion Response from Gemma ===
def get_discussion_response(input_text):
    prompt = f"""
You are a friendly AI discussion partner. Engage with the following user input by offering a thoughtful, respectful, and fact-based response.

- Use logical reasoning or simple facts to support your view
- Encourage further thinking without being confrontational
- Keep the reply within 3–5 concise sentences

User says: "{input_text}"
"""
    print("💬 Thinking with Gemma (2B)...\n\n")
    result = subprocess.run(
        ["ollama", "run", "gemma:2b"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    response = result.stdout.decode("utf-8").strip()
    if response.lower().startswith("sure"):
        response = response.split(":", 1)[-1].strip()

    return clean_text_for_tts(response)


# === Speak Text with gTTS ===
def speak_response(text, save_path="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\discussion.mp3", speed=1.1):
    print("🔊 Generating speech with gTTS...\n")
    tts = gTTS(text=text, lang='en')
    tts.save(save_path)
    audio = AudioSegment.from_mp3(save_path)
    audio = speedup(audio, playback_speed=speed)
    play(audio)


# === Main Response Function ===
def generate_discussion_and_speak(input_text, output_file="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\discussion_response.txt", speed=1.1):
    response = get_discussion_response(input_text)
    print("🤖 Gemma:", response)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)
    speak_response(response, save_path="D:\\swadhi\\6 th sem\\NLP_Project\\modules\\discussion.mp3", speed=speed)


# === Text Chat Loop ===
def chatbot_loop(speed=1.1):
    print("\n🤖 Gemma Chatbot Mode — Say 'bye' or 'done' to exit.\n\n")
    while True:
        user_input = input("🗣️ You: ").strip()
        print()
        if any(x in user_input.lower() for x in ["bye", "done", "exit"]):
            confirm = input("❓ Do you want to end the conversation? [y/n]: ").strip().lower()
            if confirm.startswith("y"):
                print("🤖 Gemma: That was a great discussion! Talk to you soon.")
                break
            else:
                continue
        response = get_discussion_response(user_input)
        print("🤖 Gemma:", response)
        speak_response(response, speed=speed)


# === Mic-based Chat Loop ===
def mic_chat_loop(speed=1.1):
    print("\n🎤 Gemma Mic Chat Mode — Say 'bye' or 'done' to exit.")
    while True:
        record_audio()  # saves to mic_recording.wav
        user_input = transcribe_file("modules/mic_recording.wav").strip()
        print("🗣️ You:", user_input)

        if any(x in user_input.lower() for x in ["bye", "done", "exit"]):
            confirm = input("❓ Do you want to end the conversation? [y/n]: ").strip().lower()
            if confirm.startswith("y"):
                print("🤖 Gemma: That was a great discussion! Talk to you soon.")
                break
            else:
                continue

        response = get_discussion_response(user_input)
        print("🤖 Gemma:", response)
        speak_response(response, speed=speed)


# === Full Flow ===
def discussion_partner_flow():
    print("\n🎯  Choose input method:\n\n")
    print("1️⃣  Text File\n")
    print("2️⃣  Direct Text (Chatbot)\n")
    print("3️⃣  Audio Input (Microphone or Audio File)\n")
    choice = input("Enter choice [1/2/3]: ").strip()

    speed_input = input("⚙️ Speech speed (1.0 = normal, 1.2 = faster, 0.9 = slower): ").strip()
    try:
        speed = float(speed_input)
    except ValueError:
        speed = 1.1  # Default

    if choice == "1":
        file_path = input("📂 Enter path to input text file: ").strip()
        with open(file_path, "r", encoding="utf-8") as f:
            input_text = f.read()
        generate_discussion_and_speak(input_text, speed=speed)

    elif choice == "2":
        chatbot_loop(speed=speed)

    elif choice == "3":
        mic_or_file = input("🎤 Use Microphone? [y/n]: ").strip().lower()
        if mic_or_file == 'y':
            mic_chat_loop(speed=speed)
        else:
            file_path = input("🎧 Enter path to audio file (.mp3 or .wav): ").strip()
            text = transcribe_file(file_path)
            generate_discussion_and_speak(text, speed=speed)
    else:
        print("❌ Invalid choice.")


# === MAIN RUN ===
if __name__ == "__main__":
    discussion_partner_flow()
