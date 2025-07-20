import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
from typing import List, Optional
import tensorflow as tf
from pydantic import BaseModel
import pandas as pd

class TopicPredictionRequest(BaseModel):
    content_text: Optional[str] = None
    topic_title: Optional[str] = None
    topic_description: Optional[str] = None


class TopicPredictor:
    def __init__(self, model_path: str = "models/curriculum_recommender"):
        """
        Initializes the predictor by loading the saved TensorFlow model index.
        """
        try:
            self.model = tf.saved_model.load(model_path)
        except OSError:
            print(f"Error: Model not found at path '{model_path}'.")
            print("Please ensure you have trained and saved the model first.")
            self.model = None

    def _preprocess_text(self, text: str) -> str:
        """
        Applies the same text cleaning logic used during model training.
        """
        cleaned_text = text.lower().replace('>', '')
        filtered_words = [word for word in cleaned_text.split() if len(word) > 3]
        return " ".join(filtered_words)

    def predict(self, request: TopicPredictionRequest, top_k: int = 4) -> List[str]:
        """
        Takes in a request and uses the topic fields to predict related content.
        """
        if not self.model:
            print("Model is not loaded. Cannot make predictions.")
            return []

        if not request.topic_title:
            print("Warning: `topic_title` is missing. Predictions may be poor.")
            return []

        raw_query_text = (request.topic_title or "")
        processed_query_text = self._preprocess_text(raw_query_text)

        _, recommended_content_ids = self.model(
            queries=tf.constant([processed_query_text])
        )

        predictions = [cid.decode('utf-8') for cid in recommended_content_ids[0].numpy()]

        return predictions[:int(top_k)]


def parse_arguments():
    """
    Sets up and parses command-line arguments for the script.

    This function creates an argument parser that allows users to specify
    the topic they want predictions for, making the script more flexible
    and user-friendly.
    """
    parser = argparse.ArgumentParser(
        description='Generate content recommendations for a given topic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows default values in help
    )

    parser.add_argument(
        '--topic',
        type=str,
        default='Calculus',
        help='The topic title to generate recommendations for'
    )

    parser.add_argument(
        '--neighbors',
        type=str,
        default='5',
        help='The K number of recommendations'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    predictor = TopicPredictor()

    if predictor.model:
        sample_request = TopicPredictionRequest(
            topic_title=args.topic,  # Now uses the command-line argument

        )

        recommendations = predictor.predict(sample_request, top_k=args.neighbors)

        print(f"✅ Top 10 recommendations for topic '{sample_request.topic_title}':")
        print(recommendations)
        print(f"Number of recommendations: {len(recommendations)}")

        content_df = pd.read_csv("data/content.csv")
        recommended_titles = content_df[content_df['id'].isin(recommendations)]

        print("✅ Found Titles for Recommended Content:")
        print(recommended_titles[['id', 'title']])