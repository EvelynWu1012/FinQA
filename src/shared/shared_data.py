"""
This files are dedicated to storing shared data
"""
# Contains processed data for each question
processed_dataset = {}
# Will contain the list of questions extracted from the dataset
questions = []
# Maps each question to its cluster label
question_to_cluster_label = {}
# Maps each cluster index to list of (question, embedding)
cluster_idx_to_questions = {}
# Store embeddings for reuse if needed
question_embeddings = None
