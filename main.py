import nltk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

nltk.download('popular')

app = FastAPI()


class Text(BaseModel):
    text: str


@app.post("/tokenize/")
async def tokenize(text: Text):
    """
    Endpoint for Tokenization.
    Accepts text in JSON format and returns a list of tokens.
    """
    try:
        tokens = nltk.word_tokenize(text.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"tokens": tokens}


@app.post("/pos_tag/")
async def pos_tag(text: Text):
    """
    Endpoint for Part-of-Speech Tagging.
    Accepts text in JSON format and returns a list of (token, tag) pairs.
    """

    try:
        tokens = nltk.word_tokenize(text.text)
        pos_tags = nltk.pos_tag(tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"pos_tags": pos_tags}


@app.post("/ner/")
async def named_entity_recognition(text: Text):
    """
    Endpoint for Named Entity Recognition.
    Accepts text in JSON format and returns a list of entities and their types.
    """
    try:
        tokens = nltk.word_tokenize(text.text)
        pos_tags = nltk.pos_tag(tokens)
        chunked = nltk.ne_chunk(pos_tags)
        named_entities = []
        for tree in chunked:
            if hasattr(tree, 'label'):
                entity = " ".join([child[0] for child in tree.leaves()])
                entity_type = tree.label()
                named_entities.append({"entity": entity, "type": entity_type})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ner": named_entities}


if __name__ == "__main__":
    uvicorn.run(app, host="127,0,0,1", port=8000)
