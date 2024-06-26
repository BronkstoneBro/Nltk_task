from fastapi import FastAPI, HTTPException
from .models import Text
from .utils import tokenize, pos_tag, named_entity_recognition
import uvicorn

app = FastAPI()


@app.post("/tokenize/")
async def tokenize_endpoint(text: Text):
    try:
        tokens = tokenize(text.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"tokens": tokens}


@app.post("/pos_tag/")
async def pos_tag_endpoint(text: Text):
    try:
        tokens = tokenize(text.text)
        pos_tags = pos_tag(tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"pos_tags": pos_tags}


@app.post("/ner/")
async def ner_endpoint(text: Text):
    try:
        tokens = tokenize(text.text)
        pos_tags = pos_tag(tokens)
        named_entities = named_entity_recognition(pos_tags)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ner": named_entities}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
