import operator
import pandas as pd
import numpy as np


class TopicModelHolder:
    '''
    Holds a data frame relating topic models to its score and number of topics used in the model

    Attributes
    ----------
        * metric_func - the metric function used to evaluate score. Tested with SciKit-Learn's silhouette_score,
                        calinski_harabasz_score, and davies_boulding_score.
        * score_function - the relational operator (direction) for the scoring function. W.r.t. SciKit-Learn:
                            * silhouette_score = '>'
                            * calinski_harabasz_score = '>'
                            * davies_boulding_score = '<'
        * model_df - the data frame relating topic models to its score and number of topics used in the model
    '''
    def __init__(self, metric_func, score_relation_direction=">"):
        self.metric_func = metric_func
        self.score_function = {">": operator.gt, "<": operator.lt}[score_relation_direction]
        self.model_df = pd.DataFrame(columns=["score", "model"])

    def update(self, X, model, n, labels):
        self.model_df = pd.concat(
            [self.model_df, pd.DataFrame({"score": self.metric_func(X, labels), "model": model}, index=[n])])

    def get_best_model(self):
        index_ = {operator.gt: 0, operator.lt: -1}[self.score_function]
        return self.model_df.sort_values("score", ascending=False).iloc[index_].model


def get_best_labels(model, corp):
    return [np.array([tuple_[1] for tuple_ in model.get_document_topics(doc)]).argmax() for doc in corp]