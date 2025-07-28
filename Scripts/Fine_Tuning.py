def load_conll(filepath):
    sentences = []
    tokens, labels = [], []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": labels})
                    tokens, labels = [], []
            else:
                splits = line.split()
                if len(splits) == 2:
                    token, label = splits
                    tokens.append(token)
                    labels.append(label)

    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": labels})

    return sentences




def conll_to_dataframe(conll_data):
    import pandas as pd
    rows = []
    for sent_id, sentence in enumerate(conll_data):
        tokens = sentence["tokens"]
        labels = sentence["ner_tags"]
        for token, label in zip(tokens, labels):
            rows.append({"sentence_id": sent_id, "token": token, "ner_tag": label})
    return pd.DataFrame(rows)



def tokenize_and_align_labels(examples,tag2id, id2tag):
    from transformers import AutoTokenizer
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        padding="max_length",
        truncation=True,
        max_length=128,
        is_split_into_words=True,)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[label[word_idx]])
            else:
                # For tokens inside a word, either label or ignore (-100)
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs 




