import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split


def get_label(x, label_cols):
    for col in label_cols:
        if str(x[col]) != 'nan':
            return int(col.replace('label_Question ','').replace('A','').replace('B','')) - 1
    return 'Error'


def filter_company(row, companies):
    return row["Company Name"] in companies

def remove_first_sentence(x):
    return "".join(x.split('.')[1:])

def get_first_sentence(x):
    return "".join(x.split('.')[0]).strip()

combined = pd.DataFrame()
for file in glob.glob('/home/rjalota/climabench_data/all_data/ClimateInsurance/raw/ClimateRiskData*.csv'):
    df = pd.read_csv(file, encoding='unicode_escape')
    renamed_cols = []
    print(f"before: {df.columns}")
    for col in df.columns:
        if 'Question' in col:
            if 'Yes No' in col:
                renamed_cols.append(f'label_{col.replace(" Yes No","")}')
            else:
                renamed_cols.append('text')
        else:
            renamed_cols.append(col)
    df.columns = renamed_cols
    print(f"after: {df.columns}")
    combined = pd.concat([df.reset_index(drop=True), combined]).reset_index(drop=True)

combined = combined[combined['text'] != 'test data']
label_cols = [col for col in combined.columns if 'label' in col]
print(f"label_cols: {label_cols}")
print(f"text: {len(combined['text'].unique())}")
combined['label'] = combined.apply(get_label, args=(label_cols,), axis=1)
print(f"combined['label_Question 8'].unique(): {combined['label_Question 1'].unique()}")
combined = combined[['Company Name', 'text', 'label']]
combined = combined.drop_duplicates('text').dropna()
combined['question_text'] = combined['text'].apply(get_first_sentence)
combined['text'] = combined['text'].apply(remove_first_sentence)
combined = combined[combined['text'].apply(len) > 0]

print(f"combined['question_text']: {combined['question_text'].unique()}")

train_co = combined[combined.apply(filter_company, args=(combined['Company Name'].value_counts()[:830].keys(),), axis=1)]

val_test_others = combined[combined.apply(filter_company, args=(combined['Company Name'].value_counts()[830:].keys(),), axis=1)]
val_co, test_co = train_test_split(val_test_others, test_size=0.5, random_state=42)


total = train_co.shape[0] + val_co.shape[0] + test_co.shape[0]

# print(f"Split: {train_co.shape[0]/total, val_co.shape[0]/total, test_co.shape[0]/total}")
# print("-------------------------------------------")
# print(f'Train:\n{train_co["label"].value_counts()}')
# print("-------------------------------------------")
# print(f'Val:\n{val_co["label"].value_counts()}')
# print("-------------------------------------------")
# print(f'Test:\n{test_co["label"].value_counts()}')
# print("-------------------------------------------")

# if not os.path.isdir('ClimateInsuranceMulti'):
#     os.mkdir('ClimateInsuranceMulti')

print(train_co.head(5))
print(train_co.columns)
# train_co.to_csv(f'home/rjalota/climabench_data/all_data/ClimateInsuranceMulti/train.csv', index=False)
# val_co.to_csv(f'ClimateInsuranceMulti/val.csv', index=False)
# test_co.to_csv(f'ClimateInsuranceMulti/test.csv', index=False)