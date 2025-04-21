import faiss
import numpy as np
from src.shared import shared_data
from sentence_transformers import SentenceTransformer


# global variables
BATCH_SIZE = 512  # For FAISS batch processing, but not strictly needed
MODEL = SentenceTransformer('all-MiniLM-L6-v2')


# =============================================================================
# Step 1: Prepare Data i.e. questions
# =============================================================================

def prepare_questions():
    """
    Retrieves and returns the list of questions stored in shared_data.
    This function provides access to the questions for further processing.
    """
    # Retrieve the list of questions from shared_data
    questions = shared_data.questions

    return questions


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
        question_embeddings (np.ndarray): A numpy array containing the
        embeddings of the input questions.
    """

    # Generate embeddings for the provided list of questions
    question_embeddings = MODEL.encode(questions)

    question_embeddings = np.array(question_embeddings, dtype=np.float32)

    # Return the embeddings as a numpy array
    return question_embeddings


# =============================================================================
# Step 3: Build FAISS Index for Fast Retrieval
# =============================================================================

def build_faiss_index(question_embeddings):
    """
    Build a FAISS index for fast nearest neighbor retrieval.

    Args:
        question_embeddings (np.ndarray): Embeddings of the questions.

    Returns:
        faiss.IndexFlatIP: A FAISS index built for fast similarity search.
    """
    # Ensure float32 dtype
    if question_embeddings.dtype != np.float32:
        question_embeddings = question_embeddings.astype(np.float32)

    # Create a FAISS index for inner product similarity
    # (dot product = cosine similarity if vectors are normalized)
    # Get the embedding dimension
    dimension = question_embeddings.shape[1]
    # Flat index for inner product
    index = faiss.IndexFlatIP(dimension)
    # Add embeddings to the index
    index.add(question_embeddings)

    return index


# =============================================================================
# Step 4: Retrieve Top Similar Questions Using FAISS
# =============================================================================

def get_top_similar_questions_faiss(input_question, index, top_num):
    """
    Returns the top `top_num` most similar questions
    using FAISS similarity search.

    Args:
        input_question (str):
        The question for which similar questions need to be found.
        index (faiss.IndexFlatIP): The FAISS index for fast similarity search.
        top_num (int): The number of top similar questions to return.

    Returns:
        list: List of the top `top_num` most similar questions.
    """
    if index is None:
        raise ValueError("FAISS index is not initialized.")
    # Get the embedding for the input question
    input_embedding = MODEL.encode([input_question], normalize_embeddings=True)

    # ensure dtype is float32
    input_embedding = np.array(input_embedding, dtype=np.float32)

    # Perform a nearest neighbor search using FAISS
    # d = distances (similarities), idx = indices of closest neighbors
    d, idx = index.search(input_embedding, top_num)

    # Retrieve the top similar questions based on indices
    top_similar_questions = [shared_data.questions[i] for i in idx[0]]

    return top_similar_questions


def initialize_faiss_index():
    """
    Initializes the FAISS index for question embeddings.
    This function should be called once during setup to compute and store
    the necessary index data that is reused during inference.
    """
    # Step 1: Prepare the list of questions for indexing
    questions = prepare_questions()  # Retrieve the list of questions

    # Step 2: Generate embeddings for the questions using a pre-trained model
    embeddings = generate_embeddings(questions)

    # Step 3: Build the FAISS index for fast similarity search
    index = build_faiss_index(embeddings)

    # Step 4: Store the FAISS index and embeddings in shared data
    shared_data.faiss_index = index
    shared_data.question_embeddings = embeddings

    return index, embeddings


def prompt_example_generator(user_question, top_num):
    """
    Given a user question, return the top most similar questions using FAISS.
    Assumes that the FAISS index has already been initialized.
    """
    # Step 1: Ensure that the FAISS index has been initialized
    if not hasattr(shared_data, 'faiss_index'):
        raise ValueError(
            "You must run initialize_faiss_index() before calling "
            "prompt_example_generator.")

    # Step 2: Retrieve the top similar questions using FAISS
    return get_top_similar_questions_faiss(
        user_question,
        shared_data.faiss_index,
        top_num
    )
