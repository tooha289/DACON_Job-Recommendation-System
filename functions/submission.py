import pandas as pd

def create_submission(model_result, resume_decode, recruitment_decode, top_k=5):
    submisson = model_result.copy()

    primary_col = submisson.columns[0]
    submisson = submisson.iloc[:,:top_k+1]
    submisson = submisson.rename(columns={primary_col:"resume_idx"})

    submisson = submisson.melt(id_vars='resume_idx', var_name='rating_seq', value_name='recruitment_idx')
    submisson = submisson.sort_values(['resume_idx', 'rating_seq'])

    submisson['resume_seq'] = submisson['resume_idx'].apply(lambda x: resume_decode[x])
    submisson['recruitment_seq'] = submisson['recruitment_idx'].apply(lambda x: recruitment_decode[x])
    submisson = submisson.iloc[:, -2:]

    return submisson

def create_score_dataframe(proba_df, rec_result_df, model_name=""):
    num_rows, num_cols = proba_df.shape
    score_name = model_name + "_" if model_name != "" else ""

    data = []
    for r in range(num_rows):
        row = {}
        for c in range(num_cols):
            if c==0:
                row['resume_idx'] = proba_df.iloc[r,c]
            else:
                row['recruitment_idx'] = rec_result_df.iloc[r,c]
                row[f'{score_name}score'] = proba_df.iloc[r,c]
                data.append(row.copy())
    score_df = pd.DataFrame(data)
    return score_df

def create_ensemble_submission_2col(score_df, column1, column2, start=0, end=101, step=10):
    submission_dfs = {}
    
    for i in range(start,end,step):
        left = i / 100
        right = (100 - i) / 100
        print(f"{left:.2f} : {right:.2f}")
        
        submission_score = score_df.copy()
        submission_score['sum'] = submission_score[column1]*left +\
                                submission_score[column2]*right 
        submission_score = submission_score[['resume_seq', 'recruitment_seq', 'sum']]
        submission_score_5 = submission_score.groupby(['resume_seq']).apply(lambda group: group.nlargest(5, 'sum'))
        submission_score_5 = submission_score_5[['resume_seq', 'recruitment_seq']].reset_index(drop=True)
        submission_dfs[f"{left:.2f}_{right:.2f}"] = submission_score_5
    return submission_dfs