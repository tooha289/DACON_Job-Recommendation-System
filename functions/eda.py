import pandas as pd
from scipy.stats import chi2_contingency

def create_count_dataframe(data_frame, column_name, hue_column):
    """
    Creates a count dataframe based on specified columns in a given DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): The input DataFrame.
    - column_name (str): The name of the column for which counts will be calculated.
    - hue_column (str): The name of the column used for grouping.

    Returns:
    - pd.DataFrame: A count dataframe with counts of occurrences based on the specified columns.
    """
    # Make a copy of the original DataFrame to avoid modifying the input
    df = data_frame.copy()
    
    # Add a temporary column with a constant value for counting
    df['value'] = 1

    # Create a pivot table to get counts based on the specified columns
    count_df = pd.pivot_table(df, values='value', index=column_name,
                              columns=hue_column, aggfunc='sum',
                              margins=True, margins_name='Total')
    
    return count_df

def calculate_ratio_df_with_column(data_frame, target_column, group_column, limit_ratio=0.):
    """
    Calculates two ratio dataframes based on specified column, utilizing the value counts of the specified target_column in a given DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the column for which ratios will be calculated.
    - group_column (str): The name of the column used for grouping.
    - limit_ratio (float, optional): The minimum ratio threshold to include columns in the result. Default is 0.0.

    Returns:
    - pd.DataFrame, pd.DataFrame: Two ratio dataframes representing ratios in different ways.

    **Ratio DataFrame 1:**
    - This DataFrame represents the ratio of each value in `target_column` relative to the total count of its corresponding value in `group_column`.
    - Each row represents a unique value in `target_column`.
    - Each column (except the last one) represents a unique value in `group_column`.
    - The last column represents the total count for each value in `target_column`.

    **Ratio DataFrame 2:**
    - This DataFrame represents the ratio of each value in `target_column` relative to the ratio of its corresponding value in `group_column`.
    - Each row represents a unique value in `target_column`.
    - Each column (except the last one) represents a unique value in `group_column`.
    - The last column is not used in this DataFrame.
    """

    # Create count dataframe using the provided function
    count_df = create_count_dataframe(data_frame, target_column, group_column)

    # Copy count dataframe for ratio calculations
    ratio_df = count_df.copy()
    
    total_cnt = ratio_df.iloc[-1, -1]

    ratio_df = ratio_df[(ratio_df.iloc[:, -1] / total_cnt) > limit_ratio]
    ratio_df = ratio_df.loc[:,(ratio_df.iloc[-1, :] / total_cnt) > limit_ratio]
    
    # Calculate ratios 1
    columns = ratio_df.columns
    all_count = ratio_df.iloc[:, -1]
    for i in range(0, len(columns)-1):
        ratio_df.iloc[:, i] = ratio_df.iloc[:, i] / all_count

    # Calculate ratios 2
    ratio_df_ratio = ratio_df.copy()
    columns = ratio_df_ratio.columns
    all_ratio = list(ratio_df_ratio.iloc[-1, :])
    for i in range(0, len(columns)-1):
        ratio_df_ratio.iloc[:-1, i] = ratio_df_ratio.iloc[:-1, i] / all_ratio[i]
        
    return ratio_df, ratio_df_ratio

def calculate_ratio_df_with_hue(data_frame, target_column, group_column, limit_ratio=0.):
    """
    Calculates two ratio dataframes based on specified column, utilizing the value counts of the specified group_column in a given DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the column for which ratios will be calculated.
    - group_column (str): The name of the column used for grouping based on its value counts.
    - limit_ratio (float, optional): The minimum ratio threshold to include columns in the result. Default is 0.0.

    Returns:
    - pd.DataFrame, pd.DataFrame: Two ratio dataframes representing ratios in different ways.
    The ratios are calculated based on the value counts of the specified group_column.

    **Ratio DataFrame 1:**
    - This DataFrame represents the ratio of each value in `target_column` relative to the total count of its corresponding value in `group_column`.
    - Each row represents a unique value in `target_column`.
    - Each column (except the last one) represents a unique value in `group_column`.
    - The last column represents the total count for each value in `target_column` across all values in `group_column`.

    **Ratio DataFrame 2:**
    - This DataFrame represents the ratio of each value in `target_column` relative to the ratio of its corresponding value in `group_column`.
    - Each row represents a unique value in `target_column`.
    - Each column (except the last one) represents a unique value in `group_column`.
    - The last column is not used in this DataFrame.
    """
    
    # Create count dataframe using the provided function
    count_df = create_count_dataframe(data_frame, target_column, group_column)

    # Copy count dataframe for ratio calculations
    ratio_df = count_df.copy()

    total_cnt = ratio_df.iloc[-1, -1]

    ratio_df = ratio_df[(ratio_df.iloc[:, -1] / total_cnt) > limit_ratio]
    ratio_df = ratio_df.loc[:,(ratio_df.iloc[-1, :] / total_cnt) > limit_ratio]
    
    # Calculate ratios 1
    columns = ratio_df.columns
    all_count = list(ratio_df.iloc[-1, :])
    for i in range(0, len(columns)):
        ratio_df.iloc[:-1, i] = ratio_df.iloc[:-1, i] / all_count[i]

    # Calculate ratios 2
    ratio_df_ratio = ratio_df.copy()
    all_ratio = ratio_df_ratio.iloc[:-1, -1]
    for i in range(0, len(columns)-1):
        ratio_df_ratio.iloc[:-1, i] = ratio_df_ratio.iloc[:-1, i] / all_ratio
        
    return ratio_df, ratio_df_ratio

def analyze_chi_square(count_df):
    """
    Perform the chi-square test on a contingency table derived from the given count dataframe.

    Parameters:
    - count_df (pd.DataFrame): The count dataframe used for the chi-square test.

    Returns:
    - tuple: A tuple containing the chi-square statistic, p-value, degrees of freedom, and expected frequencies.
    """
    # Create a contingency table
    contingency_table = count_df.copy().iloc[:-1, :-1]
    
    # Perform the chi-square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    
    # Print and display the results
    print(f"Chi-square Statistic: {chi2_stat}")
    print(f"P-value: {p_val}")
    print(f"Degrees of Freedom: {dof}")
    print("Expected Frequencies:")
    print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

    return chi2_stat, p_val, dof, expected