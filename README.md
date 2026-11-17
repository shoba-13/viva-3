# Book Recommendation Engine (K-Nearest Neighbors)

A book recommendation system built with **scikit-learn** that suggests five books similar to a given title. Completed as part of the freeCodeCamp *Machine Learning with Python* certification.

## Overview

Given a book title from the dataset, the `get_recommends` function finds the five most similar books using **K-Nearest Neighbors** on user rating patterns, and returns each one with its distance from the original book.

```python
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```

```python
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.79],
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.74],
    ['Interview with the Vampire', 0.73],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.54],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.52]
  ]
]
```

A smaller distance means a more similar book. Results are listed from farthest to closest.

## Dataset

**Book-Crossings**: about 1.1 million ratings (scale 1-10) of 270,000 books by 90,000 users. It is already loaded in the notebook, so no download is needed.

## Approach

**1. Data cleaning** – to avoid sparse, unreliable data, I kept only:
- users with at least **200** ratings
- books with at least **100** ratings

Duplicate (user, title) pairs were removed.

**2. Feature matrix** – ratings are pivoted into a table where each row is a book, each column is a user, and each value is a rating (0 if unrated). It is converted to a sparse matrix (`csr_matrix`) for efficiency.

**3. Model** – `NearestNeighbors` from `sklearn.neighbors` with:
- metric: **cosine** distance
- algorithm: **brute** force

**4. Recommendation function** – `get_recommends` looks up the book's row, finds its 6 nearest neighbors (the first is the book itself, so it is skipped), and returns the remaining 5 with their distances.

## Tech Stack

Python 3 · pandas · NumPy · SciPy · scikit-learn · Google Colaboratory

## How to Run

1. Open the notebook in Google Colab and save a copy.
2. Run the cells in order. The dataset downloads in the second cell.
3. Run the cleaning, model and function cells.
4. Call `get_recommends("<book title>")`.
5. Run the last cell to check that the challenge passes.

## Possible Improvements

- Try other metrics or tune the rating-count thresholds
- Add author or genre information for hybrid recommendations
- Use matrix factorization (e.g. SVD) to handle sparsity
- Build a small web interface for searching titles
