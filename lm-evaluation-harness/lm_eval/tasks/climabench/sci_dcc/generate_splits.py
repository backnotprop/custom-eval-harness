import pandas as pd 

if __name__ == "__main__":
    df = pd.read_csv("/nas/data/LLM/llm-evaluation-datasets/tasks/climabench/SciDCC/SciDCC.csv")
    train = df.sample(frac=0.8, random_state=23)
    rem = df.loc[~df.index.isin(train.index)]
    test = rem.sample(frac=0.5, random_state=23)
    dev = rem.loc[~rem.index.isin(test.index)]
    print(df['Category'].unique())
    print(len(train))
    print(len(test))
    print(len(dev))
    dev.to_csv("/nas/data/LLM/llm-evaluation-datasets/tasks/climabench/SciDCC/dev.csv",index=False)
    train.to_csv("/nas/data/LLM/llm-evaluation-datasets/tasks/climabench/SciDCC/train.csv",index=False)
    test.to_csv("/nas/data/LLM/llm-evaluation-datasets/tasks/climabench/SciDCC/test.csv",index=False)



