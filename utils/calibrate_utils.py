import re
import os
import umap
import torch
import random
import warnings
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import HDBSCAN
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from torchmetrics.classification import BinaryCalibrationError


# Suppress Warnings and prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def get_char_spans(text, tokenizer):
    """
    Tokenises the generated code revision string and returns character-level offset mappings.
    Args:
        text (str): The generated code revision to tokenize.
        tokenizer: A HuggingFace-style tokeniser.
    Returns:
        list[tuple[int, int]]: A list of (start_char, end_char) tuples representing 
                               the character span of each token.
    """
    # Re-tokenize to get offsets
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    return enc['offset_mapping']


def get_deletion_spans(text):
    """
    Identifies character spans in generated code revision string that should be removed.
    This includes fences, markers, and end tokens.
    Args:
        text (str): The generated code revision string.
    Returns:
        list[tuple[int, int]]: A list of (start, end) character index tuples representing spans to delete.
    """
    spans = []
    
    # Regexes that may delete some substrings
    regexes = [
        r'^```[a-zA-Z0-9_+-]*\n?', r'```\w*\s*', r'```\s*',
        r'\[_BEGIN_FIXED_CODE\]\s*', r'\[_END_FIXED_CODE\]\s*',
        r'<\|im_end\|>\s*', r'<\|endoftext\|>\s*', r'</s>\s*',
        r'<\|EOT\|>\s*', r'<｜end▁of▁sentence｜>\s*'
    ]
    
    for regex in regexes:
        for match in re.finditer(regex, text, flags=re.MULTILINE):
            spans.append((match.start(), match.end()))

    # Truncates at custom end token
    split_marker = "[_END_FIXED_CODE]"
    if split_marker in text:
        cutoff = text.index(split_marker)
        spans.append((cutoff, len(text)))
    
    return spans


def apply_filter(generated, token_ids, softmax_probabilities, tokenizer):
    """
    Removes tokens whose character spans fall entirely within deletion spans.
    Args:
        generated (str): The generated code revision string.
        token_ids (list[int]): The list of token IDs corresponding to the generated code revision string.
        softmax_probabilities (list[float]): The list of softmax probabilities aligned with each token.
        tokenizer: A HuggingFace-style tokeniser.
    Returns:
        tuple[str, list[int], list[float]]: A tuple containing the filtered code revision string, 
                                            list of token IDs, and list of softmax probabilities.
    """
    deletion_spans = get_deletion_spans(generated)
    char_spans = get_char_spans(generated, tokenizer)

    # Keep tokens that do NOT fall fully inside any deletion span
    keep_indices = []
    for i, (start, end) in enumerate(char_spans):
        if not any(del_start <= start and end <= del_end for (del_start, del_end) in deletion_spans):
            keep_indices.append(i)

    generated_f = tokenizer.decode([token_ids[i] for i in keep_indices if i < len(token_ids)])
    token_ids_f = [token_ids[i] for i in keep_indices if i < len(token_ids)]
    softmax_probabilities_f = [softmax_probabilities[i] for i in keep_indices if i < len(softmax_probabilities)]

    return generated_f, token_ids_f, softmax_probabilities_f


def filter_sequence(generated, tokenizer):
    """
    Applies fence, marker, and end token filtering to each set of generated code revisions.
    Args:
        generated (pd.DataFrame): A set of generated code revision strings, tokens IDs, and softmax probabilities.
        tokenizer: A HuggingFace-style tokeniser.
    Returns:
        pd.DataFrame: A new set of generated code revision strings, tokens IDs, and softmax probabilities 
                      that have been filtered.
    """
    generated_sequence_f_set = []
    token_ids_f_set = []
    softmax_probabilities_f_set = []

    for n in range(len(generated)):
        e = generated.iloc[n]
        generated_sequence_f, token_ids_f, softmax_probabilities_f = apply_filter(e.generated_sequence, e.token_ids, e.softmax_probabilities, tokenizer)
        generated_sequence_f_set.append(generated_sequence_f)
        token_ids_f_set.append(token_ids_f)
        softmax_probabilities_f_set.append(softmax_probabilities_f)

    generated['generated_sequence_f'] = generated_sequence_f_set
    generated['token_ids_f'] = token_ids_f_set
    generated['softmax_probabilities_f'] = softmax_probabilities_f_set

    return generated


def normalised_sequence_likelihood(softmax_probabilities):
    """
    Computes the length-normalised likelihood of a token sequence.
    This calculates the geometric mean of token-level probabilities.
    Args:
        softmax_probabilities (list[float]): A sequence of per-token softmax probabilities.
    Returns:
        float: The geometric mean of the softmax probabilities. Returns 0 if the input sequence is empty.
    """
    product = np.prod(softmax_probabilities)
    if len(softmax_probabilities) == 0:
        return 0
    return product ** (1/len(softmax_probabilities))


def average_token_probability(softmax_probabilities):
    """
    Computes the average probability of tokens in a sequence.
    This calculates the arithmetic mean of token-level probabilities.
    Args:
        softmax_probabilities (list[float]): A sequence of per-token softmax probabilities.
    Returns:
        float: The arithmetic mean of the softmax probabilities. Returns 0 if the input sequence is empty.
    """
    if len(softmax_probabilities) == 0:
        return 0
    return np.average(softmax_probabilities)


def minimum_token_probability(softmax_probabilities):
    """
    Computes the minimum probability among tokens in a sequence.
    This identifies the least confident token in the sequence.
    Args:
        softmax_probabilities (list[float]): A sequence of per-token softmax probabilities.
    Returns:
        float: The smallest token probability. Returns 0 if the input sequence is empty.
    """
    if len(softmax_probabilities) == 0:
        return 0
    return np.min(softmax_probabilities)


def lowest_k_token(softmax_probabilities):
    """
    Computes the average of the lowest token-level softmax probabilities based on the 'knee' point.
    This method identifies the point in the sorted token-level softmax  probabilities where
    the slope changes most sharply (the "knee") and averages all tokens below that point. 
    Args:
        softmax_probabilities (list[float]): A sequence of per-token softmax probabilities.
    Returns:
        float: The mean of the lowest probability tokens up to the knee point.
               Returns 0 if the input sequence is empty. 
               Returns the mean of all softmax probabilities if the knee is at index 0.
    """
    if len(softmax_probabilities) == 0:
        return 0

    sorted_probs = np.array(sorted(softmax_probabilities))
    x_axis = np.arange(len(sorted_probs))

    kneedle = KneeLocator(x_axis, sorted_probs, S=0, curve="concave", direction="increasing", online=False)
    knee_index = kneedle.elbow

    if knee_index == 0:
        return np.mean(sorted_probs)

    return np.mean(sorted_probs[:knee_index])


def attention_weighted_uncertainty(softmax_probabilities, attention):
    """
    Computes an attention-weighted uncertainty score based on token-level softmax probabilities and rolled out attention values.
    The function scales the attention values to [0, 1], computes uncertainty as `1 - softmax probability`, weights it by the 
    scaled attention values, sorts the weighted uncertainties, and identifies the point in the weighted uncertainties 
    where the slope changes most sharply (the "knee") and averages all tokens above that point. 
    Args:
        softmax_probabilities (list[float]): A sequence of per-token softmax probabilities.
        attention (list): A list of rolled out attention values corresponding to each token.
    Returns:
        float: The mean of the highest attention-weighted uncertainties up to the knee point.
               Returns 1 if the input sequence is empty. 
               Returns the mean of all attention-weighted uncertainties if the knee is at index 0.
    """
    if len(softmax_probabilities) == 0:
        return 1
    
    epsilon = 1e-8
    min_val = min(attention)
    max_val = max(attention)
    attention_scaled = [(v - min_val) / (max_val - min_val + epsilon) for v in attention]

    uncertainty = [1 - prob for prob in softmax_probabilities]
    w_uncertainty = [u * (1 + a) for u, a in zip(uncertainty, attention_scaled)]

    sorted_w_uncertainty =  np.array(sorted(w_uncertainty, reverse=True))
    x_axis = np.arange(len(sorted_w_uncertainty))
    
    kneedle = KneeLocator(x_axis, sorted_w_uncertainty, S=0, curve="convex", direction="decreasing", online=False)
    knee_index = kneedle.elbow

    if knee_index == 0:
        return np.mean(sorted_w_uncertainty)

    return np.mean(sorted_w_uncertainty[:knee_index])


def ece(confidence, accuracy, bins, print=True):
    """
    Computes Expected Calibration Error (ECE) for binary predictions.
    Args:
        confidence (list[float]): The predicted probabilities.
        accuracy (list[bool]): The binary correctness labels.
        bins (int): The number of bins for calibration.
        print (bool, optional): Whether to print the ECE. Defaults to True.
    Returns:
        float: ECE value rounded to two decimals.
    """
    metric = BinaryCalibrationError(n_bins=bins, norm='l1')
    ece = round(metric(torch.tensor(confidence, dtype=torch.float32), torch.tensor(accuracy, dtype=torch.float32)).item(), 2)
    if print == True:
        print("ECE:", ece)
    return ece


def brier(confidence, accuracy, print=True):
    """
    Computes the Brier score for binary predictions.
    Args:
        confidence (list[float]): The predicted probabilities.
        accuracy (list[bool]): The binary correctness labels.
        print (bool, optional): Whether to print the Brier score. Defaults to True.
    Returns:
        float: Brier score rounded to two decimals.
    """
    brier = round(brier_score_loss(accuracy, confidence), 2)
    if print == True:
        print("Brier:", brier)
    return brier


def bin_coverage(confidences, n_bins=10, eps=1e-12, print=True):
    """
    Counts how many confidence bins contain at least 30 samples.
    Args:
        confidence (list[float]): The predicted probabilities.
        n_bins (int, optional): The number of equal-width bins over [0, 1]. Defaults to 10.
        eps (float, optional): A small value for numerical clipping of bin probabilities. Defaults to 1e-12.
        print (bool, optional): Whether to print the bin coverage. Defaults to True.
    Returns:
        int: Number of bins with at least 30 samples.
    """
    confidences = np.asarray(confidences)
    counts, _ = np.histogram(confidences, bins=n_bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = np.clip(probs, eps, 1.0)
    covered_bin_count = np.sum(counts >= 30) 

    if print == True:
        print("Bins covered:", covered_bin_count)
    return covered_bin_count


def calculate_confidence(result, benchmark, generated):
    """
    Computes all confidence scores for a set of generated code revisions.
    Args:
        result (str): Output file name (without extension).
        benchmark (str): Directory path for saving the JSONL file.
        generated (pd.Dataframe): A set of filtered softmax probabilities and rolled out attention values.
    Returns:
        generated (pd.Dataframe): The set of confidence scores.
    """
    normalised_sequence_likelihood_set = []
    average_token_probability_set = []
    minimum_token_probability_set = []
    lowest_k_token_set = []
    attention_weighted_uncertainty_set = []

    for n in range(len(generated)):
        e = generated.iloc[n].softmax_probabilities_f
        a = generated.iloc[n].attention_mass_key
        normalised_sequence_likelihood_set.append(normalised_sequence_likelihood(e))
        average_token_probability_set.append(average_token_probability(e))
        minimum_token_probability_set.append(minimum_token_probability(e))
        lowest_k_token_set.append(lowest_k_token(e))
        attention_weighted_uncertainty_set.append(attention_weighted_uncertainty(e,a))

    generated['normalised_sequence_likelihood'] = normalised_sequence_likelihood_set
    generated['average_token_probability'] = average_token_probability_set
    generated['minimum_token_probability'] = minimum_token_probability_set
    generated['lowest_k_token_probability'] = lowest_k_token_set
    generated['attention_weighted_uncertainty'] = attention_weighted_uncertainty_set

    generated.to_json(benchmark + "/" + result + ".jsonl", orient='records', lines=True)
    return generated


def global_platt_scale(train_generated, test_generated, correctness, confidence):
    """
    Applies global Platt-scaling to calibrate confidence scores using logistic regression.
    Fits a logistic regression on confidence scores from the training set to predict correctness, 
    then applies it to confidence scores from the test set to obtain calibrated probabilities. 
    Calibration metrics are computed on the test set.
    Args:
        train_generated (pd.DataFrame): Training data containing confidence scores and ground-truth correctness labels.
        test_generated (pd.DataFrame): Test data containing confidence scores and ground-truth correctness labels.
        correctness (str): The correctness metric targeted for calibration.
        confidence (str): The confidence score targeted for calibration
    """
    clf = LogisticRegression()
    y = np.array(train_generated[correctness])
    x = np.array(train_generated[confidence])
    x = x.reshape(-1, 1)
    clf.fit(x,y)

    x = np.array(test_generated[confidence])
    x = x.reshape(-1, 1)
    calibrated_probs = clf.predict_proba(x)[:, 1]

    ece(calibrated_probs, test_generated[correctness], 10)
    brier(calibrated_probs, test_generated[correctness])
    bin_coverage(calibrated_probs, n_bins=10, eps=1e-12)


class LocalPlattScaler:
    def __init__(self, 
                 min_cluster_size, 
                 min_samples,
                 backoff_strategy,
                 umap_components=20, 
                 umap_metric='cosine', 
                 cluster_metric='euclidean'):
        """
        Performs input-aware calibration by fitting local Platt-scalers within embedding-based clusters.
        Args:
            min_cluster_size (int): Minimum cluster size hyperparameter for HDBSCAN.
            min_samples (int): Minimum samples hyperparameter for HDBSCAN density estimation.
            backoff_strategy (str): Fallback method when no local Platt-scaler exists.
            umap_components (int, optional): Number of dimensions for UMAP reduction. Defaults to 20.
            umap_metric (str, optional): Distance metric used by UMAP. Defaults to cosine.
            cluster_metric (str, optional): Distance metric used for clustering and NN lookup. Defaults to euclidean.
        Returns:
            LocalPlattScaler: Fitted scaler instance after calling `fit()`.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_components = umap_components
        self.umap_metric = umap_metric
        self.cluster_metric = cluster_metric
        self.backoff_strategy = backoff_strategy
        self.score_scaler = StandardScaler()

    def fit(self, embeddings, scores, labels):
        self.scores_train = np.array(scores).reshape(-1, 1)
        self.y_train = np.array(labels)
        
        # Fit Global Platt-scaler as a fallback
        self.global_scaler = LogisticRegression()
        self.global_scaler.fit(self.scores_train, self.y_train)
        
        # Fit UMAP to embeddings in the training set
        self.reducer = umap.UMAP(
            n_components=self.umap_components,
            metric=self.umap_metric,
            random_state=42 
        )
        
        self.embedding_reduced = self.reducer.fit_transform(embeddings)
        
        # Calculate Scaling Factor
        umap_std_per_dim = np.std(self.embedding_reduced, axis=0)
        self.umap_avg_std = np.mean(umap_std_per_dim)
        
        # Scale confidence scores to match UMAP variance
        scores_z = self.score_scaler.fit_transform(self.scores_train)
        self.scores_for_clustering = scores_z * self.umap_avg_std * self.score_weight
        
        # Append selected confidence scores to UMAP dimensions
        self.train_features = np.hstack([self.embedding_reduced, self.scores_for_clustering])
        
        # Fit HDBSCAN
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.cluster_metric
        )
        self.cluster_labels = self.clusterer.fit_predict(self.train_features)
        
        # Use NN Classifier at inference time for fast lookup
        self.nn_classifier = KNeighborsClassifier(
            n_neighbors=1, metric=self.cluster_metric, algorithm='brute'
        )
        self.nn_classifier.fit(self.train_features, self.cluster_labels)

        # Fit Local Platt-scalers
        self.fit_local_scalers() 

        return self
    
    def fit_local_scalers(self):
        self.cluster_scalers = {}
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1: continue
            
            mask = self.cluster_labels == cluster_id
            
            # Only fit if we have mixed labels in this specific cluster
            if len(np.unique(self.y_train[mask])) >= 2:
                scaler = LogisticRegression()
                scaler.fit(self.scores_train[mask], self.y_train[mask])
                self.cluster_scalers[cluster_id] = scaler

    def predict(self, test_embeddings, test_scores):
        test_scores = np.array(test_scores).reshape(-1, 1)
        
        # Applies UMAP to transform embeddings in test set
        test_emb_raw = self.reducer.transform(test_embeddings)
        
        # Scale confidence scores in test set
        test_scores_z = self.score_scaler.transform(test_scores)
        test_scores_clustering = test_scores_z * self.umap_avg_std 
        
        # Append scaled confidence scores in test set to UMAP dimensions
        test_features = np.hstack([test_emb_raw, test_scores_clustering])
        
        # Predict cluster assignment
        assigned_clusters = self.nn_classifier.predict(test_features)
        
        calibrated_preds = []
        for i, cluster_id in enumerate(assigned_clusters):
            original_score = test_scores[i][0]
            
            # Case A: valid local scaler exists for this cluster
            if cluster_id in self.cluster_scalers:
                scaler = self.cluster_scalers[cluster_id]
                prob = scaler.predict_proba([[original_score]])[0, 1]
                
            # Case B: No local scaler exists for this cluster (noise point, or pure cluster)
            else:
                if self.backoff_strategy == 'global':
                    # Fallback to global Platt-scaler
                    prob = self.global_scaler.predict_proba([[original_score]])[0, 1]
                else:
                    # Fallback to the raw confidence score
                    prob = original_score

            calibrated_preds.append(np.clip(prob, 0, 1))
            
        return np.array(calibrated_preds)


def apply_local_platt_scale(train_generated, test_generated, correctness, confidence, min_cluster_size, min_samples, backoff_strategy):
    """
    Applies local Platt-scaling for a set of generated code revisions.
    Args:
        train_generated (pd.DataFrame): A training set with embeddings, confidence scores, and correctness labels.
        test_generated (pd.DataFrame): A test set with embeddings, confidence scores, and correctness labels.
        correctness (str): The correctness metric targeted for calibration.
        confidence (str): The confidence score targeted for calibration
        min_cluster_size (int): Minimum cluster size hyperparameter for HDBSCAN.
        min_samples (int): Minimum samples hyperparameter for HDBSCAN density estimation.
        backoff_strategy (str): Fallback method when no local Platt-scaler exists.
    Returns:
        tuple: (ECE, Brier score, Bin coverage) calibration metrics on test set.
    """
    clf = LocalPlattScaler(min_cluster_size=min_cluster_size, min_samples=min_samples, backoff_strategy=backoff_strategy)
    y_train = np.array(train_generated[correctness])
    x_train_embed = list(train_generated['embeddings'])
    x_train = np.array(train_generated[confidence])
    x_train = x_train.reshape(-1, 1)
    clf.fit(x_train_embed, x_train, y_train)

    x_test_embed = list(test_generated['embeddings'])
    x_test = np.array(test_generated[confidence])
    x_test = x_test.reshape(-1, 1)
    calibrated_probs = clf.predict(x_test_embed, x_test)

    ece_ = ece(calibrated_probs, test_generated[correctness], 10, print=False)
    brier_ = brier(calibrated_probs, test_generated[correctness], print=False)
    bc_ = bin_coverage(calibrated_probs, n_bins=10, eps=1e-12, print=False)

    return ece_, brier_, bc_


def grid_search_local_platt_scale(train_generated, test_generated, correctness, confidence):
    """
    Performs a grid search over Local Platt-scaling hyperparameters for the best calibration results.
    Args:
        train_generated (pd.DataFrame): A training set with embeddings, confidence scores, and correctness labels.
        test_generated (pd.DataFrame): A test set with embeddings, confidence scores, and correctness labels.
        confidence (str): The confidence score targeted for calibration
        correctness (str): The correctness metric targeted for calibration.
    """
    min_cluster_size = [50, 75, 100, 125, 150]
    min_samples = [5, 20, 35, 50, 65, 80]
    backoff = ['raw', 'global']

    best_stats = {"ece": 1, "brier": 1, "bc": 1}
    for mcs in min_cluster_size:
        for ms in min_samples:
            for bo in backoff:
                if ms > mcs:
                    continue

                ece, brier, bc = apply_local_platt_scale(train_generated, test_generated, correctness, confidence, mcs, ms, bo)
                if (ece < best_stats["ece"]) or ((ece == best_stats["ece"]) & (brier < best_stats["brier"])) or ((ece == best_stats["ece"]) & (brier == best_stats["brier"]) & (bc > best_stats["bc"])):
                    if (bc > 1):
                        best_stats["min_cluster_size"] = mcs
                        best_stats["min_samples"] = ms
                        best_stats["backoff"] = bo
                        best_stats["ece"] = ece
                        best_stats["brier"] = brier
                        best_stats["bc"] = bc

    print(str(best_stats))