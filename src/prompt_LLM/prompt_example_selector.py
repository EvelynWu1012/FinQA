"""
select most relevant examples for prompt
"""
import heapq
from src.shared import shared_data
import importlib
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Reload shared_data to make sure it's the latest version
importlib.reload(shared_data)

# global variables
NUM_CLUSTERS = 10
RANDOM_STATE = 42
BATCH_SIZE = 512  # For MiniBatchKMeans configuration


# =============================================================================
# Step 1: Prepare Data i.e. questions
# =============================================================================

def prepare_questions():
    """
    Retrieves and returns the list of questions stored in shared_data.
    This function provides access to the questions for further processing.
    """

    # Retrieve the list of questions from shared_data
    questions = shared_data.questions  # List of questions stored in
    # shared_data

    return questions  # Return the list of questions for further use


# =============================================================================
# Step 2: Generate Embeddings
# =============================================================================
def generate_embeddings(questions):
    """
    Generates embeddings for a list of questions using the
    SentenceTransformer model.
    Args:
        questions (list of str): A list of questions to be encoded into
        embeddings.
    Returns:
        question_embeddings (Tensor): A tensor containing the embeddings of
        the input questions.
    """

    # Load the pre-trained SentenceTransformer model
    # Using a pre-trained model for generating embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the provided list of questions
    # Convert questions to tensor embeddings
    question_embeddings = model.encode(questions,
                                       convert_to_tensor=True)

    # Return the embeddings as a tensor
    return question_embeddings


# =============================================================================
# Step 3: Cluster the Embeddings
# =============================================================================

def generate_clusters(question_embeddings, num_clusters=NUM_CLUSTERS,
                      random_state=RANDOM_STATE):
    """
    Performs clustering on the given question embeddings using MiniBatchKMeans.

    Args:
        question_embeddings (Tensor or np.ndarray): Embeddings of the
        questions to be clustered.
        num_clusters (int, optional): The number of clusters to form.
        Default is NUM_CLUSTERS.
        random_state (int, optional): The seed for random number generation.
        Default is RANDOM_STATE.

    Returns:
        clusters (np.ndarray): Cluster labels for each input question
        embedding.
    """

    # Initialize the MiniBatchKMeans model with the given parameters
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,  # Number of clusters to create
        random_state=random_state,
        # Ensure reproducibility with a fixed random state
        batch_size=BATCH_SIZE,  # Batch size for each MiniBatch step
        max_iter=100,  # Maximum iterations for convergence
        n_init=3,  # Number of different initializations to run
        compute_labels=True,  # Whether to compute the labels of the clusters
        verbose=0  # Turn off verbosity for clean output
    )

    # Convert embeddings to a numpy array if they're in tensor format
    if hasattr(question_embeddings,
               'numpy'):  # Check if the embeddings are in tensor format
        question_embeddings = question_embeddings.numpy()

    # Perform clustering and obtain the cluster labels
    clusters = kmeans.fit_predict(
        question_embeddings)  # Fit the model and predict cluster labels

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


def get_top_similar_questions(input_question, question_to_cluster_label,
                              cluster_idx_to_questions, top_num):
    """
    Returns the top `top_num` most similar questions from the same cluster
    as the input question.

    Args:
        input_question (str): The question for which similar questions need
        to be found.
        question_to_cluster_label (dict): A mapping from questions to their
        cluster labels.
        cluster_idx_to_questions (dict): A mapping from cluster labels to
        questions and their embeddings.
        top_num (int): The number of top similar questions to return.

    Returns:
        list: List of the top `top_num` most similar questions in the same
        cluster as input_question.
    """

    # Step 1: Retrieve the cluster label for the input question
    label = question_to_cluster_label.get(
        input_question)  # Get the cluster label for the input question
    if label is None:
        return []  # Return empty if the question doesn't have a label

    # Step 2: Retrieve the list of questions and their embeddings in the
    # same cluster
    cluster_items = cluster_idx_to_questions[label]

    # Step 3: Find the embedding of the input question in the cluster
    input_embedding = None
    for q, emb in cluster_items:
        if q == input_question:
            input_embedding = emb  # Store the embedding of the input question
            break
    if input_embedding is None:
        return []

    # Step 4: Use a heap to efficiently track the top similar questions
    # based on cosine similarity
    heap = []
    for q, emb in cluster_items:
        if q == input_question:
            continue  # Skip the input question itself

        # Compute cosine similarity between the input question embedding and
        # the current question embedding
        sim = cosine_similarity([input_embedding], [emb])[0][0]
        heapq.heappush(heap, (sim, q))

    # Step 5: Get the top `top_num` most similar questions using the heap
    top_examples = heapq.nlargest(top_num, heap)
    return [q for sim, q in top_examples]


def initialize_question_clusters():
    """
    Initializes clustering-related variables for question embeddings.
    This function should be called once during setup to compute and store
    the necessary clustering data that is reused during inference.
    """

    # Step 1: Prepare the list of questions for clustering
    questions = prepare_questions()  # Retrieve the list of questions

    # Step 2: Generate embeddings for the questions using a pre-trained model
    embeddings = generate_embeddings(questions)

    # Step 3: Perform clustering on the embeddings to group similar questions
    clusters = generate_clusters(embeddings, NUM_CLUSTERS, RANDOM_STATE)

    # Step 4: Store embeddings, cluster labels, and cluster contents in shared data
    shared_data.question_embeddings = embeddings
    shared_data.question_to_cluster_label = (
        get_question_to_label(questions, clusters))
    shared_data.cluster_idx_to_questions = (
        get_clustered_questions(questions, embeddings, clusters))


def prompt_example_generator(user_question, top_num):
    """
    Given a user question, return the top most similar questions from the same cluster
    using semantic similarity. Assumes that question clusters have already been initialized.
    """

    # Step 1: Ensure that clustering has been initialized before generating examples
    if not shared_data.question_to_cluster_label:
        raise ValueError(
            "You must run initialize_question_clusters() before calling "
            "prompt_example_generator.")

    # Step 2: Retrieve the top similar questions from the same cluster as the user question
    return get_top_similar_questions(
        user_question,
        shared_data.question_to_cluster_label,
        shared_data.cluster_idx_to_questions,
        top_num
    )
