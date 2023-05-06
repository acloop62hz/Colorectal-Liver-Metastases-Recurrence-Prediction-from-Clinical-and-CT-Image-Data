import random
import pandas as pd
import numpy as np

def entropy(p):
    if p == 0:
        return 0
    elif p == 1:
        return 0
    else:
        return - (p * np.log2(p) + (1 - p) * np.log2(1-p))

def information_gain(left_child, right_child):
    parent = left_child + right_child
    p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
    p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
    p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
    IG_p = entropy(p_parent)
    IG_l = entropy(p_left)
    IG_r = entropy(p_right)
    return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r

def draw_bootstrap(X_train, y_train):
    bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
    oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
    X_bootstrap = X_train.iloc[bootstrap_indices].values
    y_bootstrap = y_train[bootstrap_indices]
    X_oob = X_train.iloc[oob_indices].values
    y_oob = y_train[oob_indices]
    return X_bootstrap, y_bootstrap, X_oob, y_oob

def oob_score(tree, X_test, y_test):
    mis_label = 0
    for i in range(len(X_test)):
        pred = predict_tree(tree, X_test[i])
        if pred != y_test[i]:
            mis_label += 1
    return mis_label / len(X_test)

def find_split_point(X_bootstrap, y_bootstrap, max_features):
    feature_ls = list()
    num_features = len(X_bootstrap[0])

    while len(feature_ls) <= max_features:
        feature_idx = random.sample(range(num_features), 1)
        if feature_idx not in feature_ls:
            feature_ls.extend(feature_idx)

    best_info_gain = -999
    node = None
    for feature_idx in feature_ls:
        for split_point in X_bootstrap[:,feature_idx]:
            left_child = {'X_bootstrap': [], 'y_bootstrap': []}
            right_child = {'X_bootstrap': [], 'y_bootstrap': []}

        # split children for continuous variables
        if type(split_point) in [int, float]:
            for i, value in enumerate(X_bootstrap[:,feature_idx]):
                if value <= split_point:
                    left_child['X_bootstrap'].append(X_bootstrap[i])
                    left_child['y_bootstrap'].append(y_bootstrap[i])
                else:
                    right_child['X_bootstrap'].append(X_bootstrap[i])
                    right_child['y_bootstrap'].append(y_bootstrap[i])
        # split children for categoric variables
        else:
            for i, value in enumerate(X_bootstrap[:,feature_idx]):
                if value == split_point:
                    left_child['X_bootstrap'].append(X_bootstrap[i])
                    left_child['y_bootstrap'].append(y_bootstrap[i])
                else:
                    right_child['X_bootstrap'].append(X_bootstrap[i])
                    right_child['y_bootstrap'].append(y_bootstrap[i])

        split_info_gain = information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
        if split_info_gain > best_info_gain:
            best_info_gain = split_info_gain
            left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
            right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
            node = {'information_gain': split_info_gain,
                    'left_child': left_child,
                    'right_child': right_child,
                    'split_point': split_point,
                    'feature_idx': feature_idx}

    return node

def terminal_node(node):
    y_bootstrap = node['y_bootstrap']
    pred = max(y_bootstrap, key = y_bootstrap.count)
    return pred


def split_node(node, max_features, min_samples_split, max_depth, depth):
    left_child = node['left_child']
    right_child = node['right_child']

    del(node['left_child'])
    del(node['right_child'])

    if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
        empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
        node['left_split'] = terminal_node(empty_child)
        node['right_split'] = terminal_node(empty_child)
        return

    if depth >= max_depth:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return node

    if len(left_child['X_bootstrap']) <= min_samples_split:
        node['left_split'] = node['right_split'] = terminal_node(left_child)
    else:
        node['left_split'] = find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
        split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
    if len(right_child['X_bootstrap']) <= min_samples_split:
        node['right_split'] = node['left_split'] = terminal_node(right_child)
    else:
        node['right_split'] = find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
        split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

def build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
    root_node = find_split_point(X_bootstrap, y_bootstrap, max_features)
    split_node(root_node, max_features, min_samples_split, max_depth, 1)
    return root_node

def random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    tree_ls = list()
    oob_ls = list()
    for i in range(n_estimators):
        X_bootstrap, y_bootstrap, X_oob, y_oob = draw_bootstrap(X_train, y_train)
        tree = build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
        tree_ls.append(tree)
        oob_error = oob_score(tree, X_oob, y_oob)
        oob_ls.append(oob_error)
    #print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls

def predict_tree(tree, X_test):
    feature_idx = tree['feature_idx']

    if X_test[feature_idx] <= tree['split_point']:
        if type(tree['left_split']) == dict:
            return predict_tree(tree['left_split'], X_test)
        else:
            value = tree['left_split']
            return value
    else:
        if type(tree['right_split']) == dict:
            return predict_tree(tree['right_split'], X_test)
        else:
            return tree['right_split']
        
def predict_rf(tree_ls, X_test):
    pred_ls = list()
    for i in range(len(X_test)):
        ensemble_preds = [predict_tree(tree, X_test.values[i]) for tree in tree_ls]
        final_pred = max(ensemble_preds, key = ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)

def feature_selection(X, y, k):
    # Calculate the correlation coefficients between each feature and the y variable
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    
    # Get the absolute values of the correlation coefficients
    abs_correlations = np.abs(correlations)
    
    # Select the top 'k' features with the highest absolute correlation coefficients
    top_k_indices = np.argsort(abs_correlations)[-k:]
    x_selected = X[:, top_k_indices]
    
    return x_selected, top_k_indices

def feature_selection_value(X, y, feature_names = 'original_column_names'):
    # Calculate the correlation coefficients between each feature and the y variable
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    
    # Get the absolute values of the correlation coefficients
    abs_correlations = np.abs(correlations)
    
    # Sort the features by ascending order of absolute correlation coefficients
    sorted_indices = np.argsort(abs_correlations)
    sorted_correlations = abs_correlations[sorted_indices]
    sorted_features = [feature_names[i] for i in sorted_indices]

    return sorted_features, sorted_correlations

def cross_validation_ConfusionMatrix(X, y, k):

    """
    Performs k-fold cross-validation on the input model using the input data.
    
    Args:
        model: a function or callable class that takes in X_train and y_train as arguments and returns a trained model.
        X: a numpy array of input features.
        y: a numpy array of target labels.
        k: an integer indicating the number of folds for cross-validation.
    
    Returns:
        avg_score: a float representing the average score across all folds.
        confusion_matrix: a 2*2 numpy array
    """
        # Define the number of classes
    num_classes = 2

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Split the data into k folds
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    fold_accuracies = []

    # Loop through each fold
    for i in range(k):
        # Set up training and testing data for this fold
        X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
        X_test = X_folds[i]
        y_test = y_folds[i]
        
        if isinstance(X_train, np.ndarray):
            # Train the model on the training data
            model = random_forest(pd.DataFrame(X_train), y_train, n_estimators=100, max_features=5, max_depth=10, min_samples_split=1)
            preds = predict_rf(model, pd.DataFrame(X_test)).flatten()
            
        
        # Calculate the accuracy for this fold
        fold_accuracy = sum(preds == y_test) / len(y_test)
        fold_accuracies.append(fold_accuracy)
        
        # Iterate over each example
        for j in range(len(y_test)):
            true_class = y_test[j]
            pred_class = preds[j]
            confusion_matrix[true_class, pred_class] += 1
    
    # Calculate the mean accuracy as the final accuracy
    mean_accuracy = np.mean(fold_accuracies)

    return mean_accuracy, confusion_matrix


def train_test_split(X, y, test_size=0.25, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : numpy array-like, shape (n_samples, n_features)
        The input data.

    y : numpy array-like, shape (n_samples,)
        The target values.

    test_size : float, optional (default=0.25)
        The proportion of the dataset to include in the test split.

    random_state : int or RandomState, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator.

    Returns
    -------
    X_train : numpy array-like, shape (n_train_samples, n_features)
        The training input samples.

    X_test : numpy array-like, shape (n_test_samples, n_features)
        The test input samples.

    y_train :  numpy array-like, shape (n_train_samples,)
        The training target values.

    y_test : numpy array-like, shape (n_test_samples,)
        The test target values.
    """

    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)

    # Get number of samples
    n_samples = X.shape[0]

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    # Calculate number of test samples
    n_test_samples = int(test_size * n_samples)

    # Get test indices
    test_indices = indices[:n_test_samples]

    # Get train indices
    train_indices = indices[n_test_samples:]

    # Split data into train and test subsets
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def calculate_feature_importance(X, y, model, column_names):
    """
    Calculate the feature importance for a random forest model without using sklearn.
    
    Parameters:
    X (numpy.ndarray): The input feature matrix.
    y (numpy.ndarray): The target variable array.
    model (numpy.ndarray): The trained random forest model.
    column_names (list): The names of columns
    
    Returns:
    pandas.DataFrame: A dataframe containing the feature importance values sorted in descending order.
    """
    
    # Get the number of features
    n_features = X.shape[1]
    
    # Initialize an empty array to hold the importance scores
    importance = np.zeros(n_features)
    
    # Calculate the importance of each feature
    for i in range(n_features):
        # Get the predictions for the original data
        y_pred = predict_rf(model, pd.DataFrame(X))

        # Shuffle the values of the ith feature
        X_shuffled = X.copy()
        np.random.shuffle(X_shuffled[:, i])

        # Get the predictions for the shuffled data
        y_pred_shuffled = predict_rf(model, pd.DataFrame(X_shuffled))

        # Calculate the importance score as the difference in accuracy
        importance[i] = np.mean((y_pred - y_pred_shuffled) ** 2)

    # Normalize the importance scores
    importance = importance / np.sum(importance)

    # Create a pandas dataframe with the importance scores and feature names
    feature_names = [column_names[i] for i in range(n_features)]
    importance_df = pd.DataFrame({"importance": importance, "feature": feature_names})

    # Sort the dataframe in descending order of importance
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df

def cross_validate(model, X, y, k = 3):
    """
    Performs k-fold cross-validation on the input model using the input data.
    
    Args:
        model: a function or callable class that takes in X_train and y_train as arguments and returns a trained model.
        X: a numpy array of input features.
        y: a numpy array of target labels.
        k: an integer indicating the number of folds for cross-validation.
    
    Returns:
        avg_score: a float representing the average score across all folds.
    """

    # Split the data into k folds
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    scores = []
    # Perform k-fold cross-validation
    for i in range(k):
        # Set up training and testing data for this fold
        X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
        X_test = X_folds[i]
        y_test = y_folds[i]
        
        X_train = pd.DataFrame(X_train)

        # Train the model on the training data
        model = random_forest(X_train, y_train, n_estimators=100, max_features=5, max_depth=10, min_samples_split=2)
        
        preds = predict_rf(model, pd.DataFrame(X_test)).flatten()
        acc_score = sum(preds == y_test) / len(y_test)
        # Evaluate the model on the testing data
        scores.append(acc_score)
    
    # Calculate the average score across all folds
    avg_score = sum(scores) / k
    
    return avg_score