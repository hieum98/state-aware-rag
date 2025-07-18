import datasets
import ast
import os

# Collect all data
datanames = [
             '2wikiqa', 
             'nq', 
             'hotpotqa', 
             'musique',
             'simpleqa', 
             'multihop-rag'
             ]
all_data = []
for dataname in datanames:
    if dataname == 'nq':
        data = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
        split_names = data.keys()
        data = datasets.concatenate_datasets([data[split] for split in split_names])
        # Randomly select 2000 instances
        data = data.shuffle(seed=42).select(range(2000))
        data = data.map(lambda x: {'id': f"nq-{x['id']}", 'metadata': {'source': 'nq', 'url': None, 'title': None}})
        all_data.append(data)
    elif dataname == '2wikiqa':
        data = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets','2wikimultihopqa')
        split_names = data.keys()
        data = datasets.concatenate_datasets([data[split] for split in split_names])
        data = data.map(lambda x: {'id': f"2wikiqa-{x['id']}"})
        data = data.map(lambda x: {
            'metadata': {
                'title': x.get('metadata', {}).get('supporting_facts', {}).get('title', None),
                'url': x.get('metadata', {}).get('supporting_facts', {}).get('url', None),
                'source': '2wikiqa'
            },
            'type': x.get('metadata', {}).get('type', None)
        }, num_proc=32)
        # Randomly select 5000 instances, evenly distributed across types
        all_types = set(data['type'])
        selected_data = []
        for t in all_types:
            type_data = data.filter(lambda x: x['type'] == t, num_proc=32)
            selected_data.append(type_data.shuffle(seed=42).select(range(5000 // len(all_types))))
        data = datasets.concatenate_datasets(selected_data)
        # Remove the 'type' column
        data = data.remove_columns(['type'])
        all_data.append(data)
    elif dataname == 'hotpotqa':
        data = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'hotpotqa')
        split_names = data.keys()
        data = datasets.concatenate_datasets([data[split] for split in split_names])
        data = data.map(lambda x: {'id': f"hotpotqa-{x['id']}"})
        data = data.map(lambda x: {
            'metadata': {
                'title': x.get('metadata', {}).get('supporting_facts', {}).get('title', None),
                'url': x.get('metadata', {}).get('supporting_facts', {}).get('url', None),
                'source': 'hotpotqa',
            },
            'type': x.get('metadata', {}).get('type', None),
            'level': x.get('metadata', {}).get('level', 'easy')
        }, num_proc=32)
        all_types = set(data['type'])
        all_levels = set(data['level'])
        selected_data = []
        for lv in all_levels:
            if lv == 'easy':
                # Only randomly select 200 instances for easy level, evenly distributed across types
                for t in all_types:
                    type_data = data.filter(lambda x: x['type'] == t and x['level'] == lv, num_proc=32)
                    selected_data.append(type_data.shuffle(seed=42).select(range(200 // len(all_types))))
            elif lv == 'medium':
                # Randomly select 1000 instances for medium level, evenly distributed across types
                for t in all_types:
                    type_data = data.filter(lambda x: x['type'] == t and x['level'] == lv, num_proc=32)
                    selected_data.append(type_data.shuffle(seed=42).select(range(2000 // len(all_types))))
            else:
                # Randomly select 5000 instances for hard level, evenly distributed across types
                for t in all_types:
                    type_data = data.filter(lambda x: x['type'] == t and x['level'] == lv, num_proc=32)
                    selected_data.append(type_data.shuffle(seed=42).select(range(5000 // len(all_types))))
        data = datasets.concatenate_datasets(selected_data)
        # Remove the 'type' and 'level' columns
        data = data.remove_columns(['type', 'level'])
        all_data.append(data)
    elif dataname == 'musique':
        data = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'musique')
        split_names = data.keys()
        _data = []
        for split in split_names:
            # Randomly select 2500 instances from each split
            if len(data[split]) > 2500:
                split_data = data[split].shuffle(seed=42).select(range(2500))
            else:
                split_data = data[split]
            _data.append(split_data)
        data = datasets.concatenate_datasets(_data)
        data = data.map(lambda x: {'id': f"musique-{x['id']}"})
        data = data.map(lambda x: {
            'question_decomposition': x.get('metadata', {}).get('question_decomposition', [])
        }, num_proc=32)
        data = data.map(lambda x: {
            'metadata': {
                'title': [item.get('support_paragraph', {}).get('title', '') for item in x.get('question_decomposition', [])],
                'url': [item.get('support_paragraph', {}).get('url', '') for item in x.get('question_decomposition', [])],
                'source': 'musique'
            }
        }, num_proc=32, remove_columns=['question_decomposition'])
        # remove the empty title 
        data = data.map(lambda x: {
            'metadata': {
                'title': [title for title in x['metadata']['title'] if title],
                'url': [url for url in x['metadata']['url'] if url],
                'source': x['metadata']['source']
            }
        }, num_proc=32)
        data = data.map(lambda x: {
            'metadata': {
                'title': x['metadata']['title'] if x['metadata']['title'] else None,
                'url': x['metadata']['url'] if x['metadata']['url'] else None,
                'source': x['metadata']['source']
            }
        }, num_proc=32)
        all_data.append(data)
    elif dataname == 'simpleqa':
        data = datasets.load_dataset('basicv8vc/SimpleQA')
        split_names = data.keys()
        data = datasets.concatenate_datasets([data[split] for split in split_names])
        data = data.map(lambda x, idx: {'id': f"simpleqa-{idx}"}, with_indices=True)
        # convet metadata to dictionary from string
        data = data.map(lambda x: {'metadata': ast.literal_eval(x.get('metadata', '{}'))}, num_proc=32)
        data = data.map(lambda x: {
            'metadata': {
                'title': x.get('metadata', {}).get('title', None),
                'url': x.get('metadata', {}).get('urls', None),
                'source': 'simpleqa'
            }
        }, num_proc=32)
        all_data.append(data)
    elif dataname == 'multihop-rag':
        data = datasets.load_dataset('yixuantt/MultiHopRAG', 'MultiHopRAG')
        split_names = data.keys()
        data = datasets.concatenate_datasets([data[split] for split in split_names])
        data = data.map(lambda x, idx: {'id': f"multihop-rag-{idx}"}, with_indices=True)
        data = data.map(lambda x: {
            'metadata': {
                'title': [item.get('title', '') for item in x.get('evidence_list', [])],
                'url': [item.get('url', '') for item in x.get('evidence_list', [])],
            }
        }, num_proc=32)
        # remove the empty title
        data = data.map(lambda x: {
            'metadata': {
                'title': [title for title in x['metadata']['title'] if title],
                'url': [url for url in x['metadata']['url'] if url],
                'source': 'multihop-rag'
            }
        }, num_proc=32)
        data = data.map(lambda x: {
            'metadata': {
                'title': x['metadata']['title'] if x['metadata']['title'] else None,
                'url': x['metadata']['url'] if x['metadata']['url'] else None,
                'source': x['metadata']['source']
            }
        }, num_proc=32)
        all_data.append(data)

# Concatenate all datasets
all_data = datasets.concatenate_datasets(all_data)
keep_columns = ['id', 'question', 'metadata', 'golden_answers']
to_remove = [col for col in all_data.column_names if col not in keep_columns]
all_data = all_data.remove_columns(to_remove)

# Save the combined dataset
data_path = 'data/train_data'
os.makedirs(data_path, exist_ok=True)
all_data.save_to_disk(data_path)