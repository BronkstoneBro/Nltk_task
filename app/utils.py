import nltk
from typing import List, Tuple, Dict

nltk.download('popular')


def tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text)


def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    return nltk.pos_tag(tokens)


def named_entity_recognition(pos_tags: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    chunked = nltk.ne_chunk(pos_tags)
    named_entities = []
    for tree in chunked:
        if hasattr(tree, 'label'):
            entity = " ".join([child[0] for child in tree.leaves()])
            entity_type = tree.label()
            named_entities.append({"entity": entity, "type": entity_type})
    return named_entities
