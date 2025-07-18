import os
from typing import List
import datasets
import faiss
from openai import OpenAI
import numpy as np

from preprocess.utils import simple_preprocess


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Build index for the dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to index")
    parser.add_argument("--model_api_base_url", type=str, default="https://api.openai.com/v1", help="Base URL for the OpenAI API")
    parser.add_argument("--api_key", type=str, default='EMPTY', help="API key for OpenAI")

    args = parser.parse_args()
    dataset_name = args.dataset_name

    if dataset_name == 'wiki23':
        data_path = "data/wiki23"
        # check if the dataset is already processed
        if not os.path.exists(data_path):
            data = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
            data = data.map(lambda x: {
                'id': x['id'],
                'contents': simple_preprocess(f"{x['title']}\n{x['text']}")
            }, remove_columns=['title', 'text'], num_proc=32)
    elif dataset_name == 'wiki18':
        data_path = "data/wiki18"
        if not os.path.exists(data_path):
            # Load the dataset from jsonl file
            data = datasets.load_dataset("json", data_files="data/wiki18_100w.jsonl", split="train")
            data = data.map(lambda x: {'contents': simple_preprocess(x['contents']),}, num_proc=32)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Initialize OpenAI client
    openai_client = OpenAI(base_url=args.model_api_base_url, api_key=args.api_key)
    models = openai_client.models.list()
    model = models.data[0].id

    def get_embeddings(batch):
        texts = batch['contents']
        embeddings = []
        try:
            response = openai_client.embeddings.create(
                input=texts,
                model=model
            )
            for item in response.data:
                embeddings.append(item.embedding)
        except:
            embeddings = [None] * len(texts)
        return {'embedding': embeddings}
    
    # Generate embeddings for the dataset
    data = data.map(
        get_embeddings,
        batched=True,
        batch_size=256, 
        num_proc=2,
        cache_file_name=None,   
    )
    # Save the embeddings to disk
    embedding_path = f"data/{dataset_name}_embeddings"
    data.save_to_disk(embedding_path)
    # Filter out any entries that failed to generate embeddings
    failed_embeddings = data.filter(lambda x: x['embedding'] is None, num_proc=32)
    # Try to regenerate failed embeddings
    if len(failed_embeddings) > 0:
        print(f"Regenerating {len(failed_embeddings)} failed embeddings...")
        data = data.map(
            get_embeddings,
            batched=True,
            batch_size=1,
            num_proc=1,
            cache_file_name=None,
            desc="Regenerating failed embeddings"
        )
        # Save the embeddings again
        data = datasets.concatenate_datasets([data, failed_embeddings])
    data = data.filter(lambda x: x['embedding'] is not None, num_proc=32)

    all_embeddings = data['embedding']
    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(all_embeddings)

    dim = all_embeddings.shape[-1]
    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if not faiss_index.is_trained:
        faiss_index.train(all_embeddings)
    faiss_index.add(all_embeddings)
    
    # Save the index to disk
    index_path = f"data/{dataset_name}.index"
    faiss.write_index(faiss_index, index_path)
    print(f"Index saved to {index_path}")

    # Save the corresponding dataset
    data = data.remove_columns(['embedding'])
    data.save_to_disk(f"data/{dataset_name}")
    print(f"Dataset saved to data/{dataset_name}")




