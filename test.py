import datasets
from preprocess.utils import normalize_text
from collections import Counter


def find_the_source(example, full_data):
    question = example['question']
    normalized_question = normalize_text(question)
    # Find the matching instance in the full dataset
    matching_instance = full_data.filter(lambda x: normalized_question in x['normalized_question'] or x['normalized_question'] in normalized_question,
                                         num_proc=32)
    support_urls = []
    support_titles = []
    source_id = []
    if len(matching_instance) > 0:
        for item in matching_instance:
            support_urls.extend(item['metadata']['url'] if item['metadata']['url'] else [])
            support_titles.extend(item['metadata']['title'] if item['metadata']['title'] else [])
            source_id.append(item['metadata']['source'])
    # Remove duplicates, empty titles and urls
    support_urls = list(set([url for url in support_urls if url]))
    support_titles = list(set([title for title in support_titles if title]))
    return {
        'support_urls': support_urls,
        'support_titles': support_titles,
        'source_id': source_id
    }


small_data = datasets.load_dataset('RUC-AIBOX/0.8k-data-SimpleDeepSearcher', split='train')
# Find the source of each small data's  instance in the combined dataset
full_data = datasets.load_from_disk('data/train_data')
# Remove empty, None, or whitespace-only questions
full_data = full_data.filter(lambda x: x['question'] and x['question'].strip(), num_proc=32)
# Normalize the questions in the full dataset
full_data = full_data.map(lambda x: { 'normalized_question': normalize_text(x['question'])}, num_proc=32) 
small_data = small_data.map(lambda x: find_the_source(x, full_data))

# Get statistics of the small data
# Count the number of instances with support URLs or support titles is greater than 0
support_urls_count = small_data.filter(lambda x: len(x['support_urls']) > 0, num_proc=32).num_rows
support_titles_count = small_data.filter(lambda x: len(x['support_titles']) > 0, num_proc=32).num_rows
# Count the number of instances within each source_id
source_id_counter = Counter()
for item in small_data:
    for source in item['source_id']:
        source_id_counter[source] += 1
# Print the statistics
print(f"Total instances in small data: {small_data.num_rows}")
print(f"Instances with support URLs: {support_urls_count} ({support_urls_count / small_data.num_rows * 100:.2f}%)")
print(f"Instances with support titles: {support_titles_count} ({support_titles_count / small_data.num_rows * 100:.2f}%)")
print("Source ID counts:")
for source, count in source_id_counter.items():
    print(f"{source}: {count} instances")

# Save the small data with support URLs and titles
small_data.save_to_disk('data/small_data_with_support')

