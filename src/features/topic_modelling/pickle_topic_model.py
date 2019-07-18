from src.data.make_dataset import *
from src.features.text_cleaning import *
from src.features.topic_modelling.lda_topic_model_sklearn import *

if __name__ == "__main__":
    articles, watchlist = load()

    lda_model = LDATopicModel(min_topics=3, max_topics=10)

    lda_model.fit(articles, watchlist)

    save_pickle(lda_model, "lda_model_sklearn")
    print("File size of {}: ".format("lda_model_sklearn"), os.path.getsize(get_path("lda_model_sklearn")))

    LDA_holder = open_pickle("lda_model_sklearn")

    dict_ = {"date": 50,
             "title": "ACADIA Pharmaceuticals Initiates Phase 3 CLARITY Program with Pimavanserin as Adjunctive Treatment for Major Depressive Disorder"}
    a = pd.DataFrame(dict_, index=[0])

    dict_ = {"date": 50, "title": "ACADIA Pharmaceuticals Reports First Quarter 2019 Financial Results"}
    b = pd.DataFrame(dict_, index=[0])

    dict_ = {"date": 50,
             "title": "ACADIA Pharmaceuticals Initiates Phase 3 CLARITY Program with Pimavanserin as Adjunctive Treatment for Major Depressive Disorder"}
    c = pd.DataFrame(dict_, index=[0])

    dict_ = {"date": 50,
             "title": "ACADIA Pharmaceuticals Initiates Phase 3 CLARITY Program with Pimavanserin as Adjunctive Treatment for Major Depressive Disorder"}
    d = pd.DataFrame(dict_, index=[0])

    for data in [a, b, c, d]:
        print(LDA_holder.get_topic(data, watchlist, 6))

    print(LDA_holder.get_topic(articles.sample(5), watchlist, 6))