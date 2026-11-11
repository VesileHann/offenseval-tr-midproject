"""Majority and TF-IDF + logistic-regression baselines."""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import tr_lower


def tfidf_lr(train_text, train_y):
    """Fit word 1-2 + char_wb 2-5 TF-IDF and a balanced logistic regression."""
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           preprocessor=tr_lower)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, preprocessor=tr_lower)
    X = hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(X, train_y)
    return lambda texts: clf.predict(hstack([word.transform(texts), char.transform(texts)]).tocsr())


def majority(train_y):
    label = "OFF" if (np.asarray(train_y) == "OFF").mean() > 0.5 else "NOT"
    return lambda texts: np.full(len(texts), label)
