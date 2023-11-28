import numpy as np

def calculate_precision_at_k(answer_df, pred_df, k):
    """
    Calculates the precision at k for a given set of answer and prediction dataframes.

    Args:
        answer_df (pd.DataFrame): DataFrame containing ground truth answers.
        pred_df (pd.DataFrame): DataFrame containing model predictions.
        k (int): Number of top predictions to consider for calculating precision.

    Returns:
        float: Average precision at k for the given data.
    """

    primary_col = answer_df.columns[0]  # Obtain the first column name for primary grouping
    secondary_col = answer_df.columns[1]  # Obtain the second column name for secondary values

    # Group both answer and prediction dataframes by the primary column.
    answer_dict = answer_df.groupby(primary_col)[secondary_col].apply(list).to_dict()
    pred_dict = pred_df.groupby(primary_col)[secondary_col].apply(list).to_dict()

    # Check for duplicates in the predicted secondary_col for each primary_col
    duplicated_preds = pred_df.groupby(primary_col).apply(lambda x: x[secondary_col].duplicated().any())
    if duplicated_preds.any():
        raise ValueError(f"Predicted {secondary_col} contains duplicates for some {primary_col}.")

    total_precision = 0
    total_count = 0

    for primary_key in answer_dict:
        if primary_key in pred_dict:
            true_positives = len(set(answer_dict[primary_key]).intersection(pred_dict[primary_key][:k]))
            # Calculate precision as the ratio of true positives to k.
            precision = true_positives / k
            total_precision += precision
            total_count += 1

    if total_count == 0:
        return 0  # Avoid division by zero

    return total_precision / total_count

def calculate_precision_at_k_dacon(answer_df, pred_df, k):
    """
    Calculates the precision at k for a given set of answer and prediction dataframes.

    Args:
        answer_df (pd.DataFrame): DataFrame containing ground truth answers.
        pred_df (pd.DataFrame): DataFrame containing model predictions.
        k (int): Number of top predictions to consider for calculating precision.

    Returns:
        float: Average precision at k for the given data.
    """

    primary_col = answer_df.columns[0]  # Obtain the first column name for primary grouping
    secondary_col = answer_df.columns[1]  # Obtain the second column name for secondary values

    # Group both answer and prediction dataframes by the primary column.
    answer_dict = answer_df.groupby(primary_col)[secondary_col].apply(list).to_dict()
    pred_dict = pred_df.groupby(primary_col)[secondary_col].apply(list).to_dict()

    # Check for duplicates in the predicted secondary_col for each primary_col
    duplicated_preds = pred_df.groupby(primary_col).apply(lambda x: x[secondary_col].duplicated().any())
    if duplicated_preds.any():
        raise ValueError(f"Predicted {secondary_col} contains duplicates for some {primary_col}.")

    total_precision = 0
    total_count = 0

    for primary_key in answer_dict:
        if primary_key in pred_dict:
            true_positives = len(set(answer_dict[primary_key]).intersection(pred_dict[primary_key][:k]))
            # Calculate precision as the ratio of true positives to k.
            precision = true_positives / min(len(answer_dict[primary_key]), k)
            total_precision += precision
            total_count += 1

    if total_count == 0:
        return 0  # Avoid division by zero

    return total_precision / total_count


def recall5_dacon(answer_df, submission_df):
    """
    Calculate recall@5 for given dataframes.
    
    Parameters:
    - answer_df: DataFrame containing the ground truth
    - submission_df: DataFrame containing the predictions
    
    Returns:
    - recall: Recall@5 value
    """
    
    primary_col = answer_df.columns[0]
    secondary_col = answer_df.columns[1]
    
    # Check if each primary_col entry has exactly 5 secondary_col predictions
    prediction_counts = submission_df.groupby(primary_col).size()
    if not all(prediction_counts == 5):
        raise ValueError(f"Each {primary_col} should have exactly 5 {secondary_col} predictions.")


    # Check for NULL values in the predicted secondary_col
    if submission_df[secondary_col].isnull().any():
        raise ValueError(f"Predicted {secondary_col} contains NULL values.")
    
    # Check for duplicates in the predicted secondary_col for each primary_col
    duplicated_preds = submission_df.groupby(primary_col).apply(lambda x: x[secondary_col].duplicated().any())
    if duplicated_preds.any():
        raise ValueError(f"Predicted {secondary_col} contains duplicates for some {primary_col}.")


    # Filter the submission dataframe based on the primary_col present in the answer dataframe
    submission_df = submission_df[submission_df[primary_col].isin(answer_df[primary_col])]
    
    # For each primary_col, get the top 5 predicted secondary_col values
    top_5_preds = submission_df.groupby(primary_col).apply(lambda x: x[secondary_col].head(5).tolist()).to_dict()
    
    # Convert the answer_df to a dictionary for easier lookup
    true_dict = answer_df.groupby(primary_col).apply(lambda x: x[secondary_col].tolist()).to_dict()
    
    
    individual_recalls = []
    for key, val in true_dict.items():
        if key in top_5_preds:
            correct_matches = len(set(true_dict[key]) & set(top_5_preds[key]))
            individual_recall = correct_matches / min(len(val), 5) # 공정한 평가를 가능하게 위하여 분모(k)를 'min(len(val), 5)' 로 설정함 
            individual_recalls.append(individual_recall)


    recall = np.mean(individual_recalls)
    
    return recall

def recall5(answer_df, submission_df):
    """
    Calculate recall@5 for given dataframes.
    
    Parameters:
    - answer_df: DataFrame containing the ground truth
    - submission_df: DataFrame containing the predictions
    
    Returns:
    - recall: Recall@5 value
    """
    
    primary_col = answer_df.columns[0]
    secondary_col = answer_df.columns[1]
    
    # Check if each primary_col entry has exactly 5 secondary_col predictions
    prediction_counts = submission_df.groupby(primary_col).size()
    if not all(prediction_counts == 5):
        raise ValueError(f"Each {primary_col} should have exactly 5 {secondary_col} predictions.")


    # Check for NULL values in the predicted secondary_col
    if submission_df[secondary_col].isnull().any():
        raise ValueError(f"Predicted {secondary_col} contains NULL values.")
    
    # Check for duplicates in the predicted secondary_col for each primary_col
    duplicated_preds = submission_df.groupby(primary_col).apply(lambda x: x[secondary_col].duplicated().any())
    if duplicated_preds.any():
        raise ValueError(f"Predicted {secondary_col} contains duplicates for some {primary_col}.")


    # Filter the submission dataframe based on the primary_col present in the answer dataframe
    submission_df = submission_df[submission_df[primary_col].isin(answer_df[primary_col])]
    
    # For each primary_col, get the top 5 predicted secondary_col values
    top_5_preds = submission_df.groupby(primary_col).apply(lambda x: x[secondary_col].head(5).tolist()).to_dict()
    
    # Convert the answer_df to a dictionary for easier lookup
    true_dict = answer_df.groupby(primary_col).apply(lambda x: x[secondary_col].tolist()).to_dict()
    
    individual_recalls = []
    for key, val in true_dict.items():
        if key in top_5_preds:
            correct_matches = len(set(true_dict[key]) & set(top_5_preds[key]))
            individual_recall = correct_matches / len(val)
            individual_recalls.append(individual_recall)

    recall = np.mean(individual_recalls)
    
    return recall