# Summary

This is the User Guide for GraphLab Create

* [Getting started](install.md)
* [Working with data](sframe/introduction.md)
    * [Tabular data](sframe/tabular-data.md)
    * [Graph data](sgraph/sgraph.md)
    * [Visualization](sframe/visualization.md)
    * [Feature Engineering](feature-engineering/introduction.md)
      * Numeric Features
        * [Quadratic Features](feature-engineering/quadratic_features.md)
        * [Feature Binning](feature-engineering/feature_binner.md)
        * [Numeric Imputer](feature-engineering/numeric_imputer.md) 
      * Categorical Features
        * [One Hot Encoder](feature-engineering/one_hot_encoder.md)
        * [Count Thresholder](feature-engineering/count_thresholder.md)
        * [Categorical Imputer](feature-engineering/categorical_imputer.md)
      * Text Features
        * [TF-IDF](feature-engineering/tfidf.md)
        * [Tokenizer](feature-engineering/tokenizer.md)
        * [BM25](feature-engineering/bm25.md)
      * Image Features
        * [Deep Feature Extractor](feature-engineering/deep_feature_extractor.md)
      * Misc.
        * [Hasher](feature-engineering/feature_hasher.md)
        * [Transformer Chain](feature-engineering/transformer_chain.md)
        * [Custom Transformer](feature-engineering/custom_transformer.md)
* [Modeling data](modeling-data/intro.md)
    * [Graph analytics](graph_analytics/intro.md)
        * [Examples](graph_analytics/graph_analytics.md)
    * [Regression](supervised-learning/regression.md)
        * [Linear Regression](supervised-learning/linear-regression.md)
        * [Boosted Trees Regression](supervised-learning/boosted_trees_regression.md)
    * [Classification](supervised-learning/classifier.md)
        * [Logistic Regression](supervised-learning/logistic-regression.md)
        * [Nearest Neighbor Classifier](supervised-learning/knn_classifier.md)
        * [SVM](supervised-learning/svm.md)
        * [Boosted Trees Classifier](supervised-learning/boosted_trees_classifier.md)
        * [Neuralnet Classifier](supervised-learning/neuralnet-classifier.md)
    * [Clustering](clustering/kmeans.md)
    * [Nearest Neighbors](nearest_neighbors/nearest_neighbors.md)
    * [Text analysis](text/intro.md)
        * [Processing text](text/analysis.md)
        * [Topic models](text/topic-models.md)
    * [Recommender systems](recommender/introduction.md)
        * [Choosing a model](recommender/choosing-a-model.md)
        * [Making recommendations](recommender/making-recommendations.md)
        * [Finding similar items](recommender/finding-similar-items.md)
    * [Data matching](data_matching/introduction.md)
        * [Autotagger](data_matching/autotagger.md)
        * [Deduplication](data_matching/deduplication.md)
        * [Similarity Search](data_matching/similarity_search.md)
    * [Model parameter search](model_parameter_search/introduction.md)
        * [Models](model_parameter_search/models.md)
        * [Choosing a search space](model_parameter_search/search.md)
        * [Evaluation functions](model_parameter_search/evaluation.md)
        * [Distributed execution](model_parameter_search/distributing.md)
* [Deployment](deployment/introduction.md)
    * [Job Execution](deployment/pipeline-introduction.md)
        * [Launch Asynchronous Job](deployment/pipeline-launch.md)
        * [EC2 & Hadoop](deployment/pipeline-ec2-hadoop.md)
        * [End-to-End Example](deployment/pipeline-example.md)
        * [Distributed Job Execution](deployment/pipeline-distributed.md)
        * [Monitoring Jobs](deployment/pipeline-monitoring-jobs.md)
        * [Session Management](deployment/pipeline-keeping-track-of-jobs-tasks-and-environments.md)
        * [Dependencies](deployment/pipeline-dependencies.md)
    * [Predictive Services](deployment/pred-intro.md)
        * [Getting Started](deployment/pred-getting-started.md)
        * [Predictive Objects](deployment/pred-working-with-objects.md)
        * [Querying](deployment/pred-querying.md)
        * [Operations](deployment/pred-operating.md)
* [Data Formats and Sources](data_formats_and_sources/intro.md)
    * [Spark RDDs](data_formats_and_sources/spark_integration.md)
    * [SQL Databases](data_formats_and_sources/odbc_integration.md)
* [Conclusion](conclusion.md)
* Exercises
    * [Tabular data](sframe/exercises.md)
    * [Graph data](sgraph/exercises.md)
    * [Graph analytics](graph_analytics/exercises.md)
    * [Classification](supervised-learning/exercises.md)
    * [Text analysis](text/exercises.md)
    * [Recommender systems](recommender/exercises.md)
* [FAQ/Common Problems](faq.md)
