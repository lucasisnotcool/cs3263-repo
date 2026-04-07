# Ablation Report for Helpfulness and Sentiment Models

## 1. Overview

This report summarizes the model ablation results for the two supervised components in the eWOM pipeline:

- helpfulness classification
- sentiment classification

For both tasks, the same three candidate classifiers were benchmarked:

- Logistic Regression (LR)
- Multinomial Naive Bayes (MNB)
- Complement Naive Bayes (CNB)

The goal of the ablation was to identify which classifier works best with the current sparse TF-IDF feature space and to explain why the selected model is more suitable than the alternatives.

The analysis in this report is based on the saved model summaries:

- `models/helpfulness/amazon_helpfulness_final_summary.json`
- `models/sentiment/amazon_polarity_full_benchmark_summary.json`


## 2. Experimental Setup

### 2.1 Helpfulness task

The helpfulness model uses:

- TF-IDF text features with unigrams and bigrams
- numeric metadata features:
  - rating
  - verified purchase
  - review length in words
  - title length in characters
  - text length in characters

Feature configuration:

- max features: 50,000
- minimum document frequency: 5
- maximum document frequency: 0.95
- n-gram range: 1 to 2

Candidate models:

- LR with `class_weight="balanced"`
- MNB
- CNB

Model selection rule:

- primary metric: validation macro-F1
- tie-breakers: average precision, ROC-AUC, balanced accuracy

For LR, the classification threshold was tuned on the validation set to maximize macro-F1.

### 2.2 Sentiment task

The sentiment model uses:

- TF-IDF text features only

Feature configuration:

- max features: 50,000
- minimum document frequency: 5
- maximum document frequency: 0.95
- n-gram range: 1 to 2

Candidate models:

- LR
- MNB
- CNB

Model selection rule:

- primary metric: validation macro-F1


## 3. Dataset Summary

### 3.1 Helpfulness dataset

The helpfulness dataset is strongly imbalanced.

| Split | Rows | Not Helpful | Helpful | Helpful Rate |
| --- | ---: | ---: | ---: | ---: |
| Train | 6,400,000 | 5,952,964 | 447,036 | 6.98% |
| Validation | 800,000 | 744,120 | 55,880 | 6.98% |
| Test | 800,000 | 744,120 | 55,880 | 6.98% |

This imbalance is important for interpretation. A model can obtain high raw accuracy simply by predicting the majority class, so macro-F1, minority-class precision, and minority-class recall are more meaningful than accuracy alone.

The final helpfulness run used the already-prepared split files under `data/helpfulness`. Those files retain the existing prepared-label distribution, so the statistics and model metrics below reflect that saved split.

### 3.2 Sentiment dataset

The sentiment task uses Amazon Polarity and is nearly perfectly balanced.

| Split | Rows | Negative | Positive |
| --- | ---: | ---: | ---: |
| Original train | 3,599,994 | 1,799,996 | 1,799,998 |
| Train subset after validation split | 3,239,994 | 1,619,996 | 1,619,998 |
| Validation | 360,000 | 180,000 | 180,000 |
| Test | 400,000 | 200,000 | 200,000 |

Because the sentiment dataset is balanced, accuracy and macro-F1 are expected to be very similar.


## 4. Helpfulness Ablation Results

### 4.1 Validation comparison

The helpfulness benchmark compared all three candidate models using their best validation thresholds.

| Model | Threshold | Val Macro-F1 | Val Accuracy | Helpful Precision | Helpful Recall | Helpful F1 | ROC-AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 0.7967 | 0.6693 | 0.9085 | 0.3641 | 0.4152 | 0.3880 | 0.8423 | 0.3481 |
| MNB | 0.4130 | 0.6569 | 0.9052 | 0.3430 | 0.3900 | 0.3650 | 0.8188 | 0.3156 |
| CNB | 0.9036 | 0.6569 | 0.9052 | 0.3430 | 0.3900 | 0.3650 | 0.8188 | 0.3156 |

### 4.2 Test performance of the selected model

The selected helpfulness model was Logistic Regression.

| Metric | Test Value |
| --- | ---: |
| Threshold | 0.7967 |
| Accuracy | 0.9087 |
| Balanced Accuracy | 0.6830 |
| Macro-F1 | 0.6711 |
| Helpful Precision | 0.3663 |
| Helpful Recall | 0.4206 |
| Helpful F1 | 0.3916 |
| ROC-AUC | 0.8434 |
| Average Precision | 0.3528 |

### 4.3 Why Logistic Regression is best for helpfulness

Logistic Regression outperformed both Naive Bayes variants on every important ranking and classification metric used for model selection. On validation, LR improved macro-F1 from `0.6569` to `0.6693`, improved helpful-class F1 from `0.3650` to `0.3880`, improved ROC-AUC from `0.8188` to `0.8423`, and improved average precision from `0.3156` to `0.3481`. These gains are meaningful because the helpfulness task is highly imbalanced and the main challenge is identifying the minority helpful class without flooding the output with false positives.

There are several technical reasons why LR is better than MNB and CNB in this setting.

First, the feature space is not a pure bag-of-words count model. The helpfulness model combines sparse TF-IDF text features with scaled numeric metadata features such as review length, title length, rating, and verified purchase. Logistic Regression handles this mixed feature space naturally because it learns a discriminative weight for each feature. By contrast, Naive Bayes is built around stronger distributional assumptions and is less flexible when continuous metadata signals are combined with correlated textual features.

Second, the TF-IDF representation includes unigrams and bigrams. These features are highly correlated. For example, single words and phrase-level bigrams often co-occur and partially duplicate information. Naive Bayes assumes conditional independence among features given the class label, and that assumption is clearly violated in this representation. Logistic Regression does not require that assumption and can learn to down-weight redundant features more effectively.

Third, the helpfulness task is heavily skewed toward the negative class. The LR model uses `class_weight="balanced"` and then further tunes the decision threshold on the validation set. That combination is especially important here. At the default threshold of `0.5`, LR produced very high helpful recall (`0.7470`) but poor helpful precision (`0.2049`), meaning it predicted too many reviews as helpful. After threshold tuning to `0.7967`, the predicted positive rate dropped from `25.46%` to `7.96%`, much closer to the true helpful rate of `6.98%`. This substantially improved minority-class precision while keeping enough recall to maximize macro-F1. MNB and CNB also benefited from threshold tuning, but even after tuning they still lagged behind LR.

Fourth, CNB provided no practical advantage over MNB in this experiment. Their selected-threshold validation metrics are effectively identical. CNB is often useful for imbalanced text classification, but here its complement-based correction was not enough to overcome the broader limitations of Naive Bayes on correlated TF-IDF features and mixed metadata inputs.

Finally, the selected LR model generalizes well. The train macro-F1 was `0.6785`, the validation macro-F1 was `0.6693`, and the test macro-F1 was `0.6711`. This small gap suggests that the LR model is not merely fitting the training data more aggressively; it is learning a decision boundary that transfers well to held-out data.


## 5. Sentiment Ablation Results

### 5.1 Validation comparison

| Model | Val Macro-F1 | Val Accuracy | Negative Precision | Negative Recall | Positive Precision | Positive Recall | ROC-AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 0.9392 | 0.9392 | 0.9390 | 0.9395 | 0.9395 | 0.9390 | 0.9835 | 0.9825 |
| MNB | 0.8934 | 0.8934 | 0.8905 | 0.8972 | 0.8964 | 0.8897 | 0.9589 | 0.9592 |
| CNB | 0.8934 | 0.8934 | 0.8905 | 0.8972 | 0.8964 | 0.8897 | 0.9589 | 0.9592 |

### 5.2 Test comparison

| Model | Test Macro-F1 | Test Accuracy | ROC-AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: |
| LR | 0.9391 | 0.9391 | 0.9832 | 0.9820 |
| MNB | 0.8920 | 0.8920 | 0.9583 | 0.9583 |
| CNB | 0.8920 | 0.8920 | 0.9583 | 0.9583 |

### 5.3 Test performance of the selected model

The selected sentiment model was Logistic Regression.

| Metric | Test Value |
| --- | ---: |
| Accuracy | 0.9391 |
| Macro-F1 | 0.9391 |
| ROC-AUC | 0.9832 |
| Average Precision | 0.9820 |
| Negative Precision | 0.9397 |
| Negative Recall | 0.9383 |
| Positive Precision | 0.9384 |
| Positive Recall | 0.9398 |

### 5.4 Why Logistic Regression is best for sentiment

The sentiment results are much clearer than the helpfulness results. Logistic Regression outperformed both Naive Bayes baselines by a very large margin. On validation, LR achieved a macro-F1 of `0.9392`, while MNB and CNB were both around `0.8934`. On the test set, LR achieved `0.9391`, while both Naive Bayes variants remained at `0.8920`. This is an absolute improvement of roughly `4.7` percentage points in macro-F1, which is substantial for a mature text-classification baseline.

This result is consistent with the structure of the sentiment task. The dataset is very large and balanced, which strongly favors discriminative linear models. Logistic Regression can estimate reliable feature weights from millions of examples and exploit both positive and negative evidence in the TF-IDF space. In sentiment classification, this matters because sentiment is often signaled by combinations of correlated words and phrases rather than independent token counts.

MNB and CNB are more restrictive. They model the relationship between words and class labels using stronger probabilistic assumptions that work well as simple baselines, but they tend to underfit when the feature space is rich and highly correlated. With unigrams and bigrams, many features overlap semantically and statistically. Logistic Regression can directly optimize the separating boundary in this high-dimensional space, while Naive Bayes tends to be less precise about where that boundary should lie.

The near-identical performance of MNB and CNB is also informative. CNB is often introduced as a stronger variant for imbalanced text datasets, but sentiment here is balanced almost perfectly across classes. That removes much of the usual advantage CNB can offer. As a result, both Naive Bayes models behave almost the same, and neither can match LR.

Generalization is again strong for LR. The train macro-F1 was `0.9434`, validation macro-F1 was `0.9392`, and test macro-F1 was `0.9391`. The gap is small, which indicates that the selected LR sentiment model is stable and not overfitting the training set.


## 6. Cross-Task Discussion

The two ablation studies reveal a consistent pattern: Logistic Regression is the strongest baseline for both helpfulness and sentiment when the feature representation is TF-IDF based.

However, the margin of improvement differs across tasks.

- In helpfulness, LR is better than Naive Bayes by a moderate but meaningful margin.
- In sentiment, LR is better by a much larger margin.

This difference is expected.

The helpfulness task is harder because the positive class is rare and the label is more subjective. Many reviews that look linguistically similar may differ in helpfulness depending on structure, specificity, or usefulness to the reader. That makes the class boundary noisier and lowers absolute performance for every model.

The sentiment task is easier in comparison because the classes are balanced and sentiment cues are more directly reflected in lexical content. In that setting, the discriminative advantage of LR becomes much more visible.

Another common pattern is that MNB and CNB remain very close to each other on both tasks. This suggests that the main performance limit is not which Naive Bayes variant is used, but the broader modeling assumptions shared by both.


## 7. Conclusion

The ablation results support choosing Logistic Regression as the default classifier for both helpfulness and sentiment.

For helpfulness, LR is the best choice because it:

- achieves the highest validation and test macro-F1
- improves minority helpful-class precision, recall, and F1 over both Naive Bayes baselines
- achieves stronger ranking quality through higher ROC-AUC and average precision
- works better with mixed TF-IDF plus metadata features
- benefits effectively from class weighting and validation-based threshold tuning

For sentiment, LR is the best choice because it:

- clearly outperforms both Naive Bayes variants on validation and test
- exploits the large balanced dataset effectively
- handles correlated unigram and bigram TF-IDF features better than Naive Bayes
- generalizes stably from training to validation and test data

Overall, the ablation study shows that while MNB and CNB remain reasonable lightweight baselines, Logistic Regression is the most reliable and best-performing model in the current eWOM pipeline.
