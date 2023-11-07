import pandas as pd 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='run splitter')
    parser.add_argument("--path", default="/nas/data/LLM/llm-evaluation-datasets/tasks/climabench", help="path to climabench data. ") 
    parser.add_argument("--out", default="/nas/data/LLM/llm-evaluation-datasets//tasks/climabench", help="output directory path, usually same as path")  
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(f"{args.path}/SciDCC/SciDCC.csv")
    train = df.sample(frac=0.8, random_state=23)
    rem = df.loc[~df.index.isin(train.index)]
    test = rem.sample(frac=0.5, random_state=23)
    dev = rem.loc[~rem.index.isin(test.index)]
    print(df['Category'].unique())
    print(len(train))
    print(len(test))
    print(len(dev))
    dev.to_csv(f"{args.path}/SciDCC/dev.csv",index=False)
    train.to_csv(f"{args.path}/SciDCC/train.csv",index=False)
    test.to_csv(f"{args.path}/SciDCC/test.csv",index=False)



