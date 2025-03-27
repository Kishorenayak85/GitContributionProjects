from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate
from datasets import Dataset
from tqdm import tqdm
import pandas as pd


def create_ragas_dataset(eval_dataset, retriever, auto_merging_engine, qa_prompt):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    retrieved_nodes = retriever.retrieve(row["question"])
    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
    qa_prompt.format(query_str=row["question"], context_str=context_str)
    answer = str(auto_merging_engine.query(row["question"]))
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer,
         "contexts" : [(retriever.retrieve(row["question"])[i].get_text()) for i in range(len((retriever.retrieve(row["question"]))))],
         "ground_truth" : row["ground_truth"]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_correctness
    ],
  )
  return result