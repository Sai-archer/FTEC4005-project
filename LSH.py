import numpy as np
import itertools
import random
import time
import re

import hashlib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def clean_text(text):

    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower().strip()
    words = text.split()
    words = [word for word in words if len(word) > 1 and not word.isdigit()]

    return ' '.join(words)


def preprocess_articles(file_path):
    """
    Preprocess articles: load them, split t_id and content, and clean the content.
    """
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                t_id, content = line.split(" ", 1)
                cleaned_content = clean_text(content) 
                articles.append((t_id, cleaned_content))
    return articles

def hash_shingle(shingle):
    return hashlib.md5(shingle.encode()).hexdigest()[:20]  # Take the first 20 chars


def generate_shingles(doc, shingle_size):

    words = doc.split()
    shingles = [' '.join(words[i:i + shingle_size]) for i in range(len(words) - shingle_size + 1)]
    #print(shingles)
    hashed_shingles = [hash_shingle(shingle) for shingle in shingles]
    #print(hashed_shingles)
    return hashed_shingles


def compute_minhash_matrix(shingles_in_all_docs, shingles_dict, n_hashes, seed=42):
    """
    Compute the MinHash matrix for all documents using optimized NumPy operations.
    """

    np.random.seed(42)

    start_time = time.time()
    num_docs = len(shingles_in_all_docs)
    num_shingles = len(shingles_dict)
    
    # Generate random coefficients for hash functions
    a = np.random.randint(1, num_shingles, size=n_hashes)
    b = np.random.randint(0, num_shingles, size=n_hashes)
    p = np.full(n_hashes, 2**31 - 1)  # Large prime number

    # MinHash matrix initialization
    minhash_matrix = np.full((n_hashes, num_docs), np.inf)

    # Shingle to index mapping
    shingles_indices = {shingle: idx for idx, shingle in enumerate(shingles_dict)}

    for doc_idx, shingles in enumerate(shingles_in_all_docs):
        indices = np.array([shingles_indices[shingle] for shingle in shingles if shingle in shingles_indices])
        if indices.size > 0:
            # Compute hash values for the current document
            hash_values = (np.outer(a, indices) + b[:, None]) % p[:, None]
            minhash_matrix[:, doc_idx] = hash_values.min(axis=1)
    print(f"Document-term matrix shape: {minhash_matrix.shape}")
    print(f"Optimized MinHash computation time: {time.time() - start_time:.2f} seconds")
    return minhash_matrix


def get_band_hashes(minhash_matrix, band_size):
    """
    Generate hash buckets for LSH bands.
    """
    start_time = time.time()
    np.random.seed(43)
    n_hashes, n_docs = minhash_matrix.shape
    num_bands = n_hashes // band_size

    band_hashes = {}
    for band_idx in range(num_bands):
        band_start = band_idx * band_size
        band_end = band_start + band_size
        band_slice = minhash_matrix[band_start:band_end, :].astype(int)
        band_keys = [tuple(band_slice[:, doc_idx]) for doc_idx in range(n_docs)]

        for doc_idx, band_key in enumerate(band_keys):
            if band_idx not in band_hashes:
                band_hashes[band_idx] = {}
            if band_key not in band_hashes[band_idx]:
                band_hashes[band_idx][band_key] = []
            band_hashes[band_idx][band_key].append(doc_idx)

    print(f"Total get_band_hashes time: {time.time() - start_time:.2f} seconds")
    return band_hashes


def filter_similar_pairs(docs, candidate_pairs, threshold=0.8):
    """
    Optimized filtering of similar pairs based on cosine similarity.
    """
    np.random.seed(49)
    start_time = time.time()

    # Prepare document-term matrix (sparse representation)
    doc_contents = [doc[1] for doc in docs]
    vectorizer = CountVectorizer()
    doc_matrix = vectorizer.fit_transform(doc_contents)

    # Convert candidate pairs to array for vectorized processing
    candidate_pairs = np.array(list(candidate_pairs))
    indices_1, indices_2 = candidate_pairs[:, 0], candidate_pairs[:, 1]

    # Compute cosine similarities for all candidate pairs in a single batch
    sims = cosine_similarity(doc_matrix[indices_1], doc_matrix[indices_2]).diagonal()

    # Filter pairs with similarity above the threshold
    filtered_pairs = candidate_pairs[sims >= threshold]

    print(f"Total filter_similar_pairs time: {time.time() - start_time:.2f} seconds")
    return set(map(tuple, filtered_pairs))


def get_similar_docs(docs, n_hashes, band_size, shingle_size):
    """
    Optimized method to find similar documents using LSH and MinHash.
    """
    from collections import defaultdict
    start_time = time.time()
    doc_ids = [doc[0] for doc in docs]
    doc_contents = [doc[1] for doc in docs]

    # Generate shingles and create a dictionary for unique shingles
    shingles_in_all_docs = [set(generate_shingles(content, shingle_size)) for content in doc_contents]
    total_shingles = list(set.union(*shingles_in_all_docs))
    shingles_dict = {shingle: idx for idx, shingle in enumerate(total_shingles)}

    print(f"Total unique shingles: {len(shingles_dict)}")

    # Compute MinHash matrix
    minhash_matrix = compute_minhash_matrix(shingles_in_all_docs, shingles_dict, n_hashes, seed=42)

    # Compute band hashes
    band_hashes = get_band_hashes(minhash_matrix, band_size)

    # Find candidate pairs
    candidate_pairs = set()
    for band_idx, band in band_hashes.items():
        for band_key, doc_indices in band.items():
            if len(doc_indices) > 1:
                candidate_pairs.update(itertools.combinations(doc_indices, 2))
    
    print(f"Number of candidate pairs: {len(candidate_pairs)}")

    # Filter candidate pairs using cosine similarity
    similar_pairs = filter_similar_pairs(docs, candidate_pairs)

    # Map to document IDs
    similar_pairs_id = [(doc_ids[doc1], doc_ids[doc2]) for doc1, doc2 in similar_pairs]

    print(f"Total get_similar_docs time: {time.time() - start_time:.2f} seconds")
    return similar_pairs_id


def main():
    
    np.random.seed(100)
    random.seed(100)
    input_file = 'C:/Users/User/Downloads/FTEC4005-project/Task-1 Find Similar Articles with LSH/all_articles.txt'
    output_file = 'C:/Users/User/Downloads/FTEC4005-project/Task-1 Find Similar Articles with LSH/result.txt'

    n_hashes = 100
    band_size = 20
    shingle_size = 2
    cosine_similarity_threshold = 0.8

    start_time = time.time()
    articles = preprocess_articles(input_file)
    similar_docs = get_similar_docs(articles, n_hashes, band_size, shingle_size)

    with open(output_file, 'w') as f:
        for doc1_id, doc2_id in similar_docs:
            f.write(f"{doc1_id} {doc2_id}\n")

    print(f"Results written to {output_file}.")
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()

    