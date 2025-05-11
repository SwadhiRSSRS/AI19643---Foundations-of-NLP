import pandas as pd
import numpy as np
import nltk
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # type: ignore

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["argument"], inplace=True)

    sentences, labels = [], []
    for arg in df['argument']:
        sents = sent_tokenize(arg)
        for i, sent in enumerate(sents):
            sentences.append(sent)
            if i == 0:
                labels.append("Claim")
            elif i == len(sents) - 1:
                labels.append("Conclusion")
            else:
                labels.append("Premise")

    data = pd.DataFrame({'sentence': sentences, 'label': labels})
    return data


def balance_data(data):
    le = LabelEncoder()
    data['label_enc'] = le.fit_transform(data['label'])
    grouped = data.groupby('label_enc')
    avg_count = int(grouped.size().mean())

    balanced_dfs = []
    for label, group in grouped:
        if len(group) > avg_count:
            downsampled = resample(group, replace=False, n_samples=avg_count, random_state=42)
            balanced_dfs.append(downsampled)
        else:
            upsampled = resample(group, replace=True, n_samples=avg_count, random_state=42)
            balanced_dfs.append(upsampled)

    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df, le


def tokenize_and_prepare(balanced_df, le):
    sentences, labels = [], []

    for _, row in balanced_df.iterrows():
        for sent in sent_tokenize(row['sentence']):
            sentences.append(sent)
            labels.append(row['label'])

    encoded_labels = le.transform(labels)
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
    y = to_categorical(encoded_labels)

    X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, tokenizer, le


def build_lstm_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, 1000))
    return model


def segment_and_classify(text, model, tokenizer, le):
    segments = sent_tokenize(text)
    seg_seq = tokenizer.texts_to_sequences(segments)
    seg_pad = pad_sequences(seg_seq, maxlen=50, padding='post', truncating='post')
    predictions = model.predict(seg_pad)
    predicted_labels = le.inverse_transform(np.argmax(predictions, axis=1))
    return [{"sentence": s, "label": l} for s, l in zip(segments, predicted_labels)]


def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def save_results_to_json(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def classify_argument_segments():
    # === Paths ===
    csv_path = "D:\\swadhi\\6 th sem\\NLP_Project\\Datasets\\arg_quality_rank_30k.csv"
    input_text_path = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\raw_text.txt"
    output_json_path = "D:\\swadhi\\6 th sem\\NLP_Project\\modules\\classified_argument_segments.json"

    # === Data Prep ===
    data = load_and_prepare_data(csv_path)
    balanced_df, le = balance_data(data)
    X_train, X_test, y_train, y_test, tokenizer, le = tokenize_and_prepare(balanced_df, le)

    # === Model ===
    model = build_lstm_model()
    model.summary()
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=128)

    # === Evaluation ===
    evaluate_model(model, X_test, y_test, le)

    # === Inference ===
    with open(input_text_path, "r", encoding='utf-8') as f:
        input_text = f.read()

    results = segment_and_classify(input_text, model, tokenizer, le)
    save_results_to_json(results, output_json_path)

    # === Display Results ===
    for item in results:
        print(f"[{item['label']}] {item['sentence']}")


if __name__ == "__main__":
    classify_argument_segments()
