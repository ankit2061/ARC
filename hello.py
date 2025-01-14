import spacy
from langdetect import detect
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Data class to store processed query information"""
    original_query: str
    tokens: List[str]
    language: str
    sentiment: Dict[str, float]
    sentiment_label: str
    intent: str
    similarity_score: float
    entities: List[Dict[str, str]]
    requires_human: bool


class CustomerServiceBot:
    def __init__(self, model_name: str = "en_core_web_md"):
        """Initialize the chatbot with necessary models and resources"""
        try:
            self.nlp = spacy.load(model_name)
            download("vader_lexicon", quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

        # Define intents with examples and responses
        self.intent_patterns = {
            "order_tracking": {
                "examples": [
                    "Where is my order?",
                    "Track my order",
                    "Order status",
                    "When will my order arrive?"
                ],
                "response": "I'll help you track your order. Could you please provide your order number?"
            },
            "cancellation": {
                "examples": [
                    "Cancel my order",
                    "Stop my subscription",
                    "I want to cancel",
                    "How do I cancel?"
                ],
                "response": "I understand you want to cancel. I'll guide you through the cancellation process."
            },
            "account_help": {
                "examples": [
                    "Can't login",
                    "Reset password",
                    "Account issues",
                    "Update account details"
                ],
                "response": "I'll help you with your account. What specific issue are you experiencing?"
            }
        }

        # Convert intent examples to spaCy docs
        self.intent_docs = {
            intent: [self.nlp(example) for example in patterns["examples"]]
            for intent, patterns in self.intent_patterns.items()
        }

    def process_query(self, user_query: str) -> ProcessedQuery:
        """Process user query and return structured information"""
        try:
            # Tokenization and entity recognition
            doc = self.nlp(user_query)
            tokens = [token.text for token in doc]
            entities = [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ]

            # Language detection
            language = detect(user_query)

            # Sentiment analysis
            sentiment = self.sia.polarity_scores(user_query)
            sentiment_label = (
                "positive" if sentiment["compound"] > 0.05
                else "negative" if sentiment["compound"] < -0.05
                else "neutral"
            )

            # Intent recognition
            intent, score = self._recognize_intent(doc)

            # Determine if human escalation is needed
            requires_human = self._check_human_escalation(
                sentiment_label, score, len(user_query.split())
            )

            return ProcessedQuery(
                original_query=user_query,
                tokens=tokens,
                language=language,
                sentiment=sentiment,
                sentiment_label=sentiment_label,
                intent=intent,
                similarity_score=score,
                entities=entities,
                requires_human=requires_human
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _recognize_intent(self, doc) -> Tuple[str, float]:
        """Recognize intent using similarity matching"""
        best_score = 0
        best_intent = "unknown"

        for intent, examples in self.intent_docs.items():
            # Calculate similarity with all examples
            similarities = [doc.similarity(example) for example in examples]
            max_similarity = max(similarities)

            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent

        return best_intent, best_score

    def _check_human_escalation(
            self,
            sentiment: str,
            intent_score: float,
            query_length: int
    ) -> bool:
        """Determine if the query should be escalated to a human agent"""
        return any([
            sentiment == "negative" and intent_score < 0.6,  # Angry customer with unclear intent
            intent_score < 0.4,  # Very unclear intent
            query_length > 50,  # Complex query
        ])

    def generate_response(self, processed: ProcessedQuery) -> str:
        """Generate an appropriate response based on processed query"""
        if processed.requires_human:
            return self._generate_escalation_response(processed)

        if processed.intent in self.intent_patterns:
            base_response = self.intent_patterns[processed.intent]["response"]
        else:
            base_response = "I'm not quite sure what you're asking. Could you please rephrase that?"

        # Add empathy for negative sentiment
        if processed.sentiment_label == "negative":
            base_response = f"I understand your frustration. {base_response}"

        # Add language-specific handling
        if processed.language != "en":
            base_response += "\nWould you prefer to continue in your preferred language?"

        return base_response

    def _generate_escalation_response(self, processed: ProcessedQuery) -> str:
        """Generate response for cases requiring human escalation"""
        if processed.sentiment_label == "negative":
            return ("I apologize for any frustration. Let me connect you with a human agent "
                    "who can better assist you with this matter. Please hold on.")
        return ("I want to ensure you get the best possible help. Let me transfer you "
                "to one of our customer service representatives.")


if __name__ == "__main__":
    # Initialize bot
    bot = CustomerServiceBot()

    # Example usage
    test_queries = [
        "Where is my order #12345?",
        "Je veux annuler ma commande",  # French: I want to cancel my order
        "This is the worst service ever! Nothing works!",
        "How do I update my shipping address?",
    ]

    for query in test_queries:
        try:
            processed = bot.process_query(query)
            response = bot.generate_response(processed)

            print(f"\nQuery: {query}")
            print(f"Language: {processed.language}")
            print(f"Sentiment: {processed.sentiment_label}")
            print(f"Intent: {processed.intent} (score: {processed.similarity_score:.2f})")
            print(f"Requires Human: {processed.requires_human}")
            print(f"Response: {response}")

        except Exception as e:
            logger.error(f"Error processing test query '{query}': {str(e)}")