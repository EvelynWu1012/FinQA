"""
select most relevant examples for prompt
"""
import heapq

# prompt_example_selector.py
import shared_data
import importlib
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Reload shared_data to make sure it's the latest version
importlib.reload(shared_data)

# global variables
NUM_CLUSTERS = 10
RANDOM_STATE = 42


# =============================================================================
# Step 1: Prepare Data i.e. questions
# =============================================================================

def prepare_questions():
    questions = shared_data.questions  # List of questions
    print(f"Questions in shared_data: {len(questions)}")
    return questions


# =============================================================================
# Step 2: Generate Embeddings
# =============================================================================
def generate_embeddings(questions):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embeddings = model.encode(questions, convert_to_tensor=True)

    return question_embeddings


# =============================================================================
# Step 3: Cluster the Embeddings
# =============================================================================

def generate_clusters(question_embeddings, num_clusters, random_state):
    num_clusters = NUM_CLUSTERS
    random_state = RANDOM_STATE
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(question_embeddings)
    return clusters


# =============================================================================
# Step 4: Store Clustered Questions
# =============================================================================
def get_question_to_label(questions, clusters):
    """Returns a mapping from question to its cluster label."""
    question_to_cluster_label = \
        {question: label for question, label in zip(questions, clusters)}
    return question_to_cluster_label


def get_clustered_questions(questions, question_embeddings, clusters):
    """Returns a mapping from label to list of (question, embedding)
    from the same cluster."""
    cluster_idx_to_questions = defaultdict(list)
    for q, emb, label in zip(questions, question_embeddings, clusters):
        cluster_idx_to_questions[label].append((q, emb))
    return cluster_idx_to_questions


def get_top_3_similar_questions(input_question, question_to_cluster_label,
                                cluster_idx_to_questions):
    """Returns top 3 similar questions in the same cluster as
    input_question."""
    label = question_to_cluster_label.get(input_question)
    if label is None:
        return []

    cluster_items = cluster_idx_to_questions[label]

    # Step 1: Find embedding of the input question
    input_embedding = None
    for q, emb in cluster_items:
        if q == input_question:
            input_embedding = emb
            break
    if input_embedding is None:
        return []

    # Step 2: Use heap to track top 3 similar questions
    heap = []
    for q, emb in cluster_items:
        if q == input_question:
            continue
        sim = cosine_similarity([input_embedding], [emb])[0][0]
        heapq.heappush(heap, (sim, q))

    top_3 = heapq.nlargest(3, heap)
    return [q for sim, q in top_3]


def initialize_prompt_example_selector():
    """
    Should be called once during setup to compute and store clustering-related
    variables that are reused during inference.
    """
    questions = prepare_questions()
    embeddings = generate_embeddings(questions)
    clusters = generate_clusters(embeddings, NUM_CLUSTERS, RANDOM_STATE)

    shared_data.question_embeddings = embeddings
    shared_data.question_to_cluster_label = get_question_to_label(questions,
                                                                  clusters)
    shared_data.cluster_idx_to_questions = get_clustered_questions(questions,
                                                                   embeddings,
                                                                   clusters)


def prompt_example_generator(user_question):
    """
    Given a user question, return top 3 most similar questions
    from the same cluster using semantic similarity.
    Assumes initialization has been done already.
    """
    if not shared_data.question_to_cluster_label:
        raise ValueError(
            "You must run initialize_prompt_example_selector() before calling prompt_example_generator.")

    return get_top_3_similar_questions(
        user_question,
        shared_data.question_to_cluster_label,
        shared_data.cluster_idx_to_questions
    )