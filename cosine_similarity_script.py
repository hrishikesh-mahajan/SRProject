from numpy import ndarray
from scipy.sparse._matrix import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stringContainer: list[str] = ["", ""]


def store_string(inputStringOne: str) -> bool:
    stringContainer[0] = inputStringOne
    return True


def add_compare_string(inputStringTwo: str) -> bool:
    stringContainer[1] = inputStringTwo
    return True


def preprocess(text: str) -> list[str]:
    return text.lower().split()  # Simple tokenization by splitting on spaces


def calculate_cosine_similarity(corpus: list[str]) -> float:
    # TF-IDF vectorization
    tfidf_vectorizer: object = TfidfVectorizer(tokenizer=preprocess)
    tfidf_matrix: spmatrix = tfidf_vectorizer.fit_transform(corpus)

    # Calculate cosine similarity
    cosine_sim: ndarray[Any, Any] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])  # type: ignore
    return cosine_sim[0][0]


def calculate_similarity_score() -> float:
    corpus: list[str] = [stringContainer[0], stringContainer[1]]
    # Calculate cosine similarity
    similarity_score: float = calculate_cosine_similarity(corpus)
    return similarity_score * 100
