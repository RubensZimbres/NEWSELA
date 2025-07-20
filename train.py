import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #0

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text
from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    topics_df = pd.read_csv("data/topics.csv").fillna("")
    content_df = pd.read_csv("data/content.csv").fillna("")
    correlations_df = pd.read_csv("data/correlations.csv")
except FileNotFoundError:
    print("Please make sure topics.csv, content.csv, and correlations.csv are in a 'data/' subfolder.")
    exit()

def get_full_topic_path_text(topic_id, p_map, t_map):
    breadcrumbs = []
    current_id = topic_id
    visited = set()
    while pd.notna(current_id) and current_id and current_id not in visited:
        visited.add(current_id)
        title = t_map.get(current_id, "")
        if title:
            breadcrumbs.append(title)
        current_id = p_map.get(current_id)
    return " >> ".join(reversed(breadcrumbs))

parent_map = topics_df.set_index('id')['parent'].to_dict()
title_map = topics_df.set_index('id')['title'].to_dict()
topics_df['ancestor_path'] = topics_df['id'].apply(lambda tid: get_full_topic_path_text(tid, parent_map, title_map))
topics_df['topic_full_text'] = (
    topics_df['title'] + " " + topics_df['description'] + " " + topics_df['ancestor_path']
).apply(lambda text: ' '.join([word for word in text.lower().replace('>', '').split() if len(word) > 3]))
print("✅ Text features for topics created.")

correlations_df['content_ids'] = correlations_df['content_ids'].str.split()
training_df_exploded = correlations_df.explode('content_ids').rename(columns={'content_ids': 'content_id'})

training_df = pd.merge(
    training_df_exploded,
    topics_df[['id', 'topic_full_text']],
    left_on='topic_id',
    right_on='id'
)

final_df = training_df[['topic_full_text', 'content_id']].astype(str)

ratings_dataset = tf.data.Dataset.from_tensor_slices(dict(final_df))

print(f"✅ Topic-to-Content data prepared. Created {len(final_df)} training pairs.")

tf.random.set_seed(42)

# Vocabulary for Topics
topic_text_data = tf.data.Dataset.from_tensor_slices(topics_df['topic_full_text'].unique())
topic_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
topic_vocabulary.adapt(topic_text_data.batch(1024).map(lambda x: tf.strings.split(x)))
print(f"Topic vocabulary size: {topic_vocabulary.vocabulary_size()}")


# Vocabulary for Content
content_id_data = tf.data.Dataset.from_tensor_slices(content_df['id'].unique())
content_id_vocabulary = tf.keras.layers.StringLookup(mask_token=None, num_oov_indices=0)
content_id_vocabulary.adapt(content_id_data)
print(f"Content ID vocabulary size: {content_id_vocabulary.vocabulary_size()}")

# Candidate dataset of all content IDs for the model's task layer
content_ids_dataset = tf.data.Dataset.from_tensor_slices(content_df['id'].unique())


class CurriculumModel(tfrs.Model):

    def __init__(self, topic_vocabulary, content_id_vocabulary, embedding_dim=768):
        super().__init__()

        class TopicTower(tf.keras.Model):
            def __init__(self, vocabulary_layer, embedding_dim):
                super().__init__()
                self.vocabulary_layer = vocabulary_layer
                self.embedding_layer = tf.keras.layers.Embedding(
                    vocabulary_layer.vocabulary_size(),
                    embedding_dim
                )

            def call(self, inputs):
                split_text = tf.strings.split(inputs)
                token_indices = self.vocabulary_layer(split_text)
                embeddings = self.embedding_layer(token_indices)
                return tf.reduce_sum(embeddings, axis=1)

        # -- Left Tower: Processes Topics (Query) --
        self.topic_model = TopicTower(topic_vocabulary, embedding_dim)

        # -- Right Tower: Processes Content (Candidate) --
        self.content_model = tf.keras.Sequential([
            content_id_vocabulary,
            tf.keras.layers.Embedding(content_id_vocabulary.vocabulary_size(), embedding_dim)
        ])

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=content_ids_dataset.batch(768).map(self.content_model)
            )
        )

    def compute_loss(self, data: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        topic_embedding = self.topic_model(data["topic_full_text"])
        content_embedding = self.content_model(data["content_id"])

        return self.task(topic_embedding, content_embedding)


# --- 6. Train the Model ---
shuffled_dataset = ratings_dataset.shuffle(len(final_df), seed=42, reshuffle_each_iteration=False)
train_size = int(0.8 * len(final_df))
train = shuffled_dataset.take(train_size)
test = shuffled_dataset.skip(train_size)
cached_train = train.batch(4096).cache()
cached_test = test.batch(4096).cache()

embedding_dimension = 768
model = CurriculumModel(topic_vocabulary, content_id_vocabulary, embedding_dimension)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.077))



print("\n🚀 Starting model training...")
model.fit(cached_train, epochs=5, validation_data=cached_test)
print("✅ Model training complete.")


print("\n🧪 Evaluating model and generating predictions for F2 score calculation...")
model.evaluate(cached_test, return_dict=True)

# The index uses the query tower (topic_model) to find nearest neighbors.
index = tfrs.layers.factorized_top_k.BruteForce(model.topic_model)

# It's populated with all possible candidates (content_ids) and their embeddings
# from the candidate tower (content_model).
index.index_from_dataset(
  tf.data.Dataset.from_tensor_slices(content_df['id'].unique()).batch(128).map(
      lambda content_id: (content_id, model.content_model(content_id))
  )
)

# Create a dictionary mapping each topic_id to its true list of content_ids.
correlations_df['content_ids_list'] = correlations_df['content_ids'].apply(lambda x: x if isinstance(x, list) else str(x).split())
ground_truth_map = correlations_df.set_index('topic_id')['content_ids_list'].to_dict()

# Create a test set of unique topics to generate predictions for.
unique_topics = training_df[['topic_id', 'topic_full_text']].drop_duplicates()
_, test_topics = train_test_split(unique_topics, test_size=0.2, random_state=42)
print(f"\nTotal topics for testing: {len(test_topics)}")

# Generate predictions for the test topics ---
K = 4
predictions = {}


print(f"Generating top {K} recommendations for {len(test_topics)} test topics...")
for _, row in tqdm(test_topics.iterrows(), total=len(test_topics)):
    topic_id = row['topic_id']
    topic_text = row['topic_full_text']

    _, recommended_content_ids = index(tf.constant([topic_text]), k=K)

    predictions[topic_id] = [cid.decode('utf-8') for cid in recommended_content_ids[0].numpy()]




# Calculate and display the Mean F2 Score ---
def calculate_mean_f2_score(predictions, ground_truth, k):
    """Calculates the mean F2 score for a set of predictions."""
    f2_scores = []
    for topic_id, predicted_ids in predictions.items():
        true_ids = ground_truth.get(topic_id)
        if not true_ids or not isinstance(true_ids, list):
            continue

        predicted_set = set(predicted_ids)
        true_set = set(filter(None, true_ids))

        if not true_set:
            continue

        true_positives = len(predicted_set.intersection(true_set))
        if true_positives == 0:
            f2_scores.append(0.0)
            continue

        precision = true_positives / len(predicted_set)
        recall = true_positives / len(true_set)

        f2 = (5 * precision * recall) / ((4 * precision + recall) + 1e-9)
        f2_scores.append(f2)

    return np.mean(f2_scores) if f2_scores else 0.0

mean_f2 = calculate_mean_f2_score(predictions, ground_truth_map, K)

print("\n" + "="*40)
print(f"📊 Mean F2 Score @{K}: {mean_f2:.4f}")
print("="*40)

save_path = "models/curriculum_recommender"
if not os.path.exists(save_path):
    os.makedirs(save_path)

tf.saved_model.save(index, save_path)
print(f"\n✅ Model index saved to: {save_path}")


print("\n🚀 Creating submission file...")

submission_df = pd.DataFrame.from_dict(predictions, orient='index')
submission_df['content_ids'] = submission_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
submission_df = submission_df.reset_index()
submission_df = submission_df.rename(columns={'index': 'topic_id'})
final_submission = submission_df[['topic_id', 'content_ids']]
final_submission.to_csv('submission.csv', index=False)

print("✅ Submission file 'submission.csv' created successfully.")
print(final_submission.head())