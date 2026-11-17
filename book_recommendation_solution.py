# ===== Paste into the matching Colab cells =====
# Cell 1 (imports) and cell 2 (data download / df_books, df_ratings) are given.
# Make sure these imports exist in cell 1:
#   import numpy as np, pandas as pd
#   from scipy.sparse import csr_matrix
#   from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------- data cleaning cell
# Keep only users with >= 200 ratings and books with >= 100 ratings
# (otherwise the data is too sparse for good neighbours and the test will not match)
user_counts = df_ratings['user'].value_counts()
df_ratings = df_ratings[df_ratings['user'].isin(user_counts[user_counts >= 200].index)]

book_counts = df_ratings['isbn'].value_counts()
df_ratings = df_ratings[df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]

# Attach titles; drop duplicate (user, title) pairs so the pivot works
df = df_ratings.merge(df_books, on='isbn').drop_duplicates(['user', 'title'])

# Rows = book titles, columns = users, values = rating (0 when unrated)
pivot = df.pivot(index='title', columns='user', values='rating').fillna(0)
matrix = csr_matrix(pivot.values)

# ---------------------------------------------------------------- model cell
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(matrix)

# ---------------------------------------------------------------- function cell
def get_recommends(book=""):
    idx = pivot.index.get_loc(book)
    distances, indices = model.kneighbors(
        pivot.iloc[idx].values.reshape(1, -1), n_neighbors=6)

    # [0] is the book itself, so skip it; the test expects farthest first
    recs = [[pivot.index[i], float(d)]
            for d, i in zip(distances[0][1:], indices[0][1:])]
    recs = recs[::-1]

    return [book, recs]

# ---------------------------------------------------------------- try it
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)
