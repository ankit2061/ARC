import sqlite3
import random
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Initialize database connection
conn = sqlite3.connect("customer_support.db")
cursor = conn.cursor()

# Database setup
cursor.execute('''CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    user_id TEXT,
    status TEXT,
    details TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    user_id TEXT,
    issue TEXT,
    status TEXT,
    resolution TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS predefined_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent TEXT,
    response TEXT
)''')

conn.commit()

# Populate predefined responses with 170 entries
predefined_responses = [
    ("greeting", "Hello! How can I assist you today?"),
    ("order_status", "Please provide your order ID so I can check the status."),
    ("create_ticket", "Sure, let me create a ticket for you. Please describe your issue."),
    ("human_agent", "I am connecting you to a human agent. Please hold on."),
    ("farewell", "Thank you for reaching out. Have a great day!"),
    ("greeting", "Hellloooo! What brings you here today?"),
    *[(f"custom_intent_{i}", f"This is a custom predefined response {i}.") for i in range(6, 171)]
]

cursor.executemany("INSERT OR IGNORE INTO predefined_responses (intent, response) VALUES (?, ?)", predefined_responses)
conn.commit()

# Load GPT-2 model for fallback responses
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Emotion detection pipeline using Hugging Face
emotion_classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

# Helper functions
def get_predefined_response(intent):
    cursor.execute("SELECT response FROM predefined_responses WHERE intent = ?", (intent,))
    result = cursor.fetchone()
    return result[0] if result else None

def create_ticket(user_id, issue):
    ticket_id = f"TKT-{random.randint(1000, 9999)}"
    cursor.execute("INSERT INTO tickets (ticket_id, user_id, issue, status, resolution) VALUES (?, ?, ?, ?, ?)",
                   (ticket_id, user_id, issue, "open", ""))
    conn.commit()
    return ticket_id

def get_order_details(order_id):
    cursor.execute("SELECT order_id, status, details FROM orders WHERE order_id = ?", (order_id,))
    result = cursor.fetchone()
    if result:
        return {
            "order_id": result[0],
            "status": result[1],
            "details": result[2]
        }
    return None

def get_emotion(text):
    result = emotion_classifier(text)[0]
    return result['label'], result['score']

def generate_gpt2_response(prompt):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Chatbot logic
def handle_order_related_queries(user_input, user_id):
    """
    Handle queries related to orders, including issues, refunds, and tracking.
    """
    # Attempt to extract order ID from the input
    words = user_input.split()
    for word in words:
        if word.startswith("ORD-"):  # Assuming order IDs have a specific prefix like "ORD-"
            order_id = word
            break
    else:
        return "Could you please provide your order ID so I can assist you further?"

    # Fetch order details
    order_details = get_order_details(order_id)
    if not order_details:
        return f"I couldn't find any details for order ID {order_id}. Please check the ID and try again."

    # Respond based on query type
    if "issue" in user_input.lower():
        return (f"Order ID: {order_id}\n"
                f"Status: {order_details['status']}\n"
                f"Details: {order_details['details']}\n"
                "Would you like me to create a ticket for this issue?")
    elif "refund" in user_input.lower():
        return (f"Order ID: {order_id}\n"
                f"Status: {order_details['status']}\n"
                f"Details: {order_details['details']}\n"
                "If the order is eligible for a refund, I can assist you in initiating the process.")
    elif "track" in user_input.lower():
        return (f"Order ID: {order_id}\n"
                f"Status: {order_details['status']}\n"
                f"Details: {order_details['details']}\n"
                "You can track your order using the tracking link provided in your email.")

    return "I'm not sure how to handle this order-related query. Could you provide more details?"

def generate_response(user_input, user_id):
    if "hello" in user_input.lower():
        return get_predefined_response("greeting")

    elif any(keyword in user_input.lower() for keyword in ["issue with my order", "refund", "track my order"]):
        return handle_order_related_queries(user_input, user_id)

    elif "create ticket" in user_input.lower():
        issue = " ".join(user_input.split()[2:])  # Extract issue description
        ticket_id = create_ticket(user_id, issue)
        return f"Ticket created successfully. Your ticket ID is {ticket_id}."

    elif "human agent" in user_input.lower():
        return get_predefined_response("human_agent")

    elif "bye" in user_input.lower():
        return get_predefined_response("farewell")

    else:
        label, score = get_emotion(user_input)
        if label and score > 0.8:
            return f"I detected that you are feeling {label} ({score:.2f}). How can I assist you further?"
        else:
            return generate_gpt2_response(user_input)

# Main interaction loop
print("Welcome to the AI Support System! Type 'exit' to quit.")
while True:
    user_id = "USER-1234"  # Example static user ID
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Thank you for reaching out. Have a great day!")
        break

    response = generate_response(user_input, user_id)
    print(f"Chatbot: {response}")
