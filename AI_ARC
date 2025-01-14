import openai
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# OpenAI API Key
openai.api_key = "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"

# Sample Dataset for Emotion Classification
emotion_data = {
    "texts": [
        "I am so happy with your service!",
        "This is the worst support I have experienced!",
        "Thank you for helping me resolve my issue so quickly!",
        "I am really frustrated with the delay.",
        "Your team is amazing and very supportive.",
        "Why does it take so long to get a response?",
        "I feel sad because no one is assisting me.",
        "The service was fantastic, keep it up!",
        "I'm so disappointed in the experience.",
        "I am thrilled with the prompt service!",
        "My problem was not resolved properly.",
        "The staff was courteous and helpful.",
        "I can't believe how quickly you responded!",
        "This has been a very unpleasant experience.",
        "I appreciate your assistance with my issue."
    ],
    "labels": ["happy", "angry", "grateful", "frustrated", "happy", "frustrated", "sad", "happy", "angry",
               "happy", "angry", "grateful", "happy", "angry", "grateful"]
}

# Map emotions to integers for the model
emotion_label_map = {"happy": 0, "angry": 1, "grateful": 2, "frustrated": 3, "sad": 4}
emotion_data["labels"] = [emotion_label_map[label] for label in emotion_data["labels"]]

# Tokenization and Padding
vocab_size = 5000
max_length = 20
embedding_dim = 128

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(emotion_data["texts"])
sequences = tokenizer.texts_to_sequences(emotion_data["texts"])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Prepare training data
X_emotion = np.array(padded_sequences)
y_emotion = np.array(emotion_data["labels"])

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)

# Build Emotion Recognition Model
emotion_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(len(emotion_label_map), activation='softmax')
])

# Compile the model
emotion_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
emotion_model.fit(X_train, y_train, epochs=10, batch_size=2, verbose=1, validation_data=(X_val, y_val))

# Function for Emotion Prediction
def predict_emotion(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = emotion_model.predict(padded)
    emotion = list(emotion_label_map.keys())[np.argmax(prediction)]
    return emotion

# Function to handle GPT-based responses
def generate_gpt_response(user_input):
    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and knowledgeable customer support assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        return gpt_response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "I'm sorry, but I'm unable to process your request right now. Please try again later."

# Combined Chatbot Logic
def chatbot_response(user_input):
    # Step 1: Predict Emotion
    try:
        detected_emotion = predict_emotion(user_input)
    except Exception:
        detected_emotion = None

    # Predefined responses for emotions
    response_map = {
        "happy": ["I'm glad to hear that! Is there anything else I can assist you with?",
                  "That's wonderful news! How can I further help you today?",
                  "I'm thrilled you're happy with our service! Anything more I can do for you?"],
        "angry": ["I'm sorry to hear that you're upset. Let me know how I can help resolve this issue.",
                  "I apologize for the inconvenience. How can I assist you further?",
                  "I'm here to help fix this. Please tell me more about the problem."],
        "grateful": ["You're welcome! We are always here to help.",
                     "It's my pleasure to assist you. Anything else you need?",
                     "I'm glad I could help. Do you have any other questions?"],
        "frustrated": ["I understand your frustration. Can you provide more details so I can assist you better?",
                       "I apologize for the delay. How can I make things right?",
                       "I'm here to help. Please let me know what I can do to assist you."],
        "sad": ["I'm here to help. Please let me know what I can do to assist you.",
                "I'm sorry you're feeling this way. How can I support you?",
                "Let me help you with this issue. What can I do for you?"]
    }

    # If emotion is recognized, use predefined responses
    if detected_emotion in response_map:
        return np.random.choice(response_map[detected_emotion])

    # If emotion is not recognized, delegate to GPT
    return generate_gpt_response(user_input)

# Run the Chatbot
print("Customer Support Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Thank you for reaching out. Have a great day!")
        break

    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
