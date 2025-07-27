def tokenize(text):
    import re
    NORMALIZATION_MAP = {"ሀ": "ሐ","ሃ": "ሓ","ሓ": "ሐ"}
    text = re.sub(r'https?://\S+|@\w+', '', text)
    text = re.sub(r'[^\w\s\u1200-\u137F\.\d]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    for char, replacement in NORMALIZATION_MAP.items():
      text = text.replace(char, replacement)
    tokens = re.findall(r'[\u1200-\u137F]+|[a-zA-Z]+|\d+(?:\.\d+)?', text)
    return tokens



def remove_stopwords(tokens):
  from nltk.corpus import stopwords
  amharic_stopwords = set{"ነው", "ያለ", "እና", "የ", 'ነበር', 'አልነበረም', 'ነበሩ', 'እስከ', 'በተጨማሪ', 'እንግዲኛ', 'እንግዲኛም', "እየ"}
  english_stopwords = set(stopwords.words('english'))
  all_stopwords = english_stopwords.union(amharic_stopwords)
  return [t for t in tokens if t not in stopwords]  