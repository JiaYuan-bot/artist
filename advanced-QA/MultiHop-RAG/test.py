from datasets import load_dataset

ds = load_dataset("yixuantt/MultiHopRAG", "corpus")

# # 保存为 JSON 文件
# ds['train'].to_json("./dataset/corpus.json",
#                     orient='records',
#                     lines=False,
#                     force_ascii=False)

# ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")

# 转换为 pandas 并保存
df = ds['train'].to_pandas()
df.to_json("./dataset/corpus.json",
           orient='records',
           force_ascii=False,
           indent=2)

# # 1. 查看数据集结构
# print(ds)

# # 2. 查看各个 split
# print(ds.keys())  # 例如: dict_keys(['train', 'test'])

# # 3. 查看某个 split 的信息
# print(ds['train'])

# # 4. 查看列名 (schema)
# print(ds['train'].column_names)

# # 5. 查看 features (详细 schema)
# print(ds['train'].features)

# import pandas as pd

# df = ds['train'].to_pandas()
# print(df.head(1))
# print(df.iloc[0])

# from datasets import load_dataset

# ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
# import pandas as pd

# df = ds['train'].to_pandas()
# print(df.head(1))
# print(df.iloc[0])
