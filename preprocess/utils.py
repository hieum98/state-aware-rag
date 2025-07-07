import re
import unicodedata


def simple_preprocess(text: str) -> str:
    """
    Simple preprocess function to:
    - replace multiple spaces with a single space
    - replace multiple new lines with a new line
    - replace urls with the string "<url>"
    - replace emails with the string "<email>"
    - replace tags (e.g. @user) with the string "<tag>"
    - remove special characters
    Args:
        text: text to be preprocessed
    Returns:
        text: preprocessed text
    """
    # Normalize text e.g., é -> e, ñ -> n, etc.
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii')
    # Replace multilple \r \t \f and spaces in to single character
    text = re.sub(r'[\r\t\f ]+', ' ', text)
    text = re.sub(r'&gt|&lt', ' ', text)
    # replace multiple new lines with a new line
    text = re.sub(r'\n+', '\n', text)
    # replace urls with the string "<url>"
    # text = re.sub(r'http://\S+|https://\S+', '[URL]', text, flags=re.MULTILINE)
    # replace emails with the string "<email>"
    # text = re.sub(r'\S+@\S+', '[Email]', text, flags=re.MULTILINE)
    # replace file paths with the string "<file>"
    # text = re.sub(r'([a-zA-Z]:\\|\\\\|\/)', '<file>', text, flags=re.MULTILINE)
    # replace tags (e.g. @user) with the string "<tag>"
    # text = re.sub(r'@\w+', '[Tag]', text, flags=re.MULTILINE)
    # replace multiple tags with a single tag
    # text = re.sub(r'(\[Tag\] )+', '[Tag] ', text)
    # text = re.sub(r'(\[Email\] )+', '[Email] ', text)
    # text = re.sub(r'(\[URL\] )+', '[URL] ', text)
    # Truncating too long text
    if len(text.split(' ')) > 2048:
        text = ' '.join(text.split(' ')[:2048])
    # remove special characters
    # text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()\-\[\]]', '', text)
    return text

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return simple_preprocess(white_space_fix(remove_articles(remove_punc(lower(s)))))

    

