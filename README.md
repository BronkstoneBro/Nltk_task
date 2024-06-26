### ⚙️ 1. Create a virtual environment

```bash
python -m venv env
source env/bin/activate 

```

## 🛠️ 2. Install dependencies:

```bash

pip install fastapi uvicorn pydantic nltk numpy

```

### ▶️ 3. Run the application:


```bash
uvicorn app.main:app --reload

```
### 🧪 4. Test the endpoint:

Send a POST request to http://127.0.0.1:8000/tokenize, http://127.0.0.1:8000/pos_tag, http://127.0.0.1:8000/ner with a JSON body containing the text to be summarized.

Example request:

```json
{
    "text": "Quentin Tarantino: The Greatest Actor of All Time"
}
```
Excepted response:

```json /tokenize
{
    "tokens": [
        "Quentin",
        "Tarantino",
        ":",
        "The",
        "Greatest",
        "Actor",
        "of",
        "All",
        "Time"
    ]
}
```

```json /pos_tag
{
    "pos_tags": [
        [
            "Quentin",
            "NNP"
        ],
        [
            "Tarantino",
            "NN"
        ],
        [
            ":",
            ":"
        ],
        [
            "The",
            "DT"
        ],
        [
            "Greatest",
            "NNP"
        ],
        [
            "Actor",
            "NNP"
        ],
        [
            "of",
            "IN"
        ],
        [
            "All",
            "NNP"
        ],
        [
            "Time",
            "NNP"
        ]
    ]
}
```


```json /ner
{
    "ner": [
        {
            "entity": "Quentin",
            "type": "GPE"
        },
        {
            "entity": "Greatest",
            "type": "ORGANIZATION"
        },
        {
            "entity": "All",
            "type": "ORGANIZATION"
        }
    ]
}
```