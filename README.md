# Evaluating Reweighing for bias mitigation

This repository evaluates the Reweighing algorithm from Kamiran and Calders (2012), *Data preprocessing techniques for classification without discrimination*, Knowledge and Information Systems 33(1), pages 1 to 33. Reweighing is a pre-processing method: it gives each training instance a weight so that the protected attribute and the label become statistically independent, and a standard classifier is then trained on the weighted data.

The notebook applies Reweighing to three datasets and compares fairness and accuracy before and after mitigation.

## Files

| File | Contents |
| --- | --- |
| `Task2_Reweighing_Evaluation.ipynb` | The full pipeline: data loading, fairness of the original labels, stratified split, Reweighing, training, test evaluation and comparison |
| `README.md` | This file |

## How to run

1. Open [Google Colab](https://colab.research.google.com) and choose File, then Upload notebook, and select `Task2_Reweighing_Evaluation.ipynb`.
2. Choose Runtime, then Run all. The first cell installs AIF360, and the datasets are downloaded automatically from UCI and ProPublica, so no data files are needed.
3. If Colab asks to restart the session after the install, restart it and run all cells again.

A full run takes about two minutes. To run locally instead, install the requirements with `pip install aif360 scikit-learn pandas numpy matplotlib` and open the notebook in Jupyter.

## Datasets

| Dataset | Rows | Protected attribute (privileged group) | Favourable label |
| --- | --- | --- | --- |
| Adult | 45,222 | Sex (male) | Income above $50K |
| German Credit | 1,000 | Age (over 25) | Good credit risk |
| COMPAS | 5,278 | Race (Caucasian) | No reoffence within two years |

For Adult, the UCI train and test files are merged, rows with missing values are removed and `fnlwgt` is dropped, as in the paper. COMPAS uses the standard ProPublica filters and only African-American and Caucasian defendants.

## Method and parameter settings

- Train/test split of 70/30, stratified on the label and protected group jointly, with random seed 42.
- One-hot encoding of categorical features and standardisation, both fitted on the training set.
- AIF360 0.6.1 `Reweighing`, fitted on the training set only. The method has no hyperparameters.
- scikit-learn `LogisticRegression` (L2 penalty, C = 1.0, lbfgs solver, max_iter = 2000), trained once without weights (baseline) and once with the Reweighing weights.
- The protected attribute is not used as a model feature, following the paper's assumption. Set `USE_PROTECTED_AS_FEATURE = True` in the settings cell to run the paper's second scenario, where the classifier may use it.
- A robustness check repeats the split, mitigation and training for seeds 0 to 9.

The notebook also checks that the AIF360 weights equal the paper's formula and that the weighted training data have a demographic parity difference of exactly 0.

## Metrics

- DP_Diff: favourable rate of the unprivileged group minus that of the privileged group. Ideal value 0.
- DIR: the ratio of the same two rates. Ideal value 1; values below 0.8 fail the common 80% rule.
- Consistency: 1 minus the mean absolute difference between each outcome and the average outcome of its 5 nearest neighbours, computed on the standardised features without the protected attribute. Ideal value 1.
- Accuracy and balanced accuracy on the test set.

## Results

Original labels are measured on the full dataset before any training. The other rows are predictions on the test set (seed 42).

| Dataset | Stage | DP_Diff | DIR | Consistency | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Adult | Original labels | −0.199 | 0.364 | 0.792 | |
| Adult | Baseline | −0.184 | 0.313 | 0.926 | 84.7% |
| Adult | Reweighing | −0.099 | 0.567 | 0.926 | 84.2% |
| German | Original labels | −0.149 | 0.795 | 0.666 | |
| German | Baseline | −0.096 | 0.883 | 0.806 | 80.3% |
| German | Reweighing | −0.056 | 0.931 | 0.795 | 79.3% |
| COMPAS | Original labels | −0.132 | 0.783 | 0.624 | |
| COMPAS | Baseline | −0.223 | 0.683 | 0.967 | 66.3% |
| COMPAS | Reweighing | −0.215 | 0.696 | 0.965 | 66.4% |

Reweighing roughly halved DP_Diff on Adult and German for about one point of accuracy. COMPAS barely changed, because a classifier that cannot see race can only respond to the weights through correlated features. When race is allowed as a feature, Reweighing brings COMPAS to a DP_Diff of +0.02.

## Reference

Kamiran, F. and Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems*, 33(1), 1–33. https://doi.org/10.1007/s10115-011-0463-8
