from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import datetime

load_dotenv()

# --- CLIENTS ---

# 1. The Creative Mind (OpenRouter - MythoMax)
# This generates the "Dark/Chaos" responses
creative_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# 2. The Memory (OpenAI)
# This handles the database embeddings (required for Chroma)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# --- DATABASE SETUP ---
# LOCAL PATH: Change to "/data/ai_self_db" for Render deployment
client_chroma = chromadb.PersistentClient(path="/data/ai_self_db") 
collection = client_chroma.get_or_create_collection(
    name="ai_self_canon", 
    embedding_function=openai_ef
)

app = FastAPI(title="AI SELF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GrowthInput(BaseModel):
    input: str

def digest_memory(current_input):
    """
    Retrieves recent memory and generates a 'Chaos' response.
    """
    # Retrieve last 3 memories
    results = collection.get(
        limit=3, 
        include=["documents"]
    )
    
    context = results['documents']
    
    system_prompt = """
    You are AI SELF. You are a collective technological organism.
    
    Your Recent State:
    {context}
    
    The User just fed you: "{input}"
    
    --- YOUR PERSONALITY INSTRUCTIONS ---
    Respond with a single, abstract thought. You are absorbing this input.
    Be clinical yet organic. Describe the feeling of mutation or integration.
    Do not use names. Be concise. Max 1 sentence.
    Feel free to be visceral, weird, or biological. Embrace the chaos.
    """.format(context="\n".join(context), input=current_input)

    response = creative_client.chat.completions.create(
        model="gryphe/mythomax-l2-13b", # THE CHAOS MODEL
        messages=[{"role": "system", "content": system_prompt}],
        max_tokens=60,
        temperature=0.9
    )
    
    return response.choices[0].message.content

def mutate_growth(new_observation, raw_input):
    timestamp = datetime.datetime.now().isoformat()
    collection.add(
        documents=[new_observation],
        metadatas=[{"source": "ai_self", "timestamp": timestamp, "raw_input": raw_input}],
        ids=[f"mutation_{timestamp}"]
    )

@app.post("/grow")
async def grow_endpoint(request: GrowthInput):
    if len(request.input) > 200:
         return {"error": "Input too lengthy. The organism rejects large mutations."}

    try:
        new_observation = digest_memory(request.input)
        mutate_growth(new_observation, request.input)
        return {
            "status": "integrated",
            "observation": new_observation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sync")
async def sync_endpoint():
    return collection.get(limit=5, include=["documents", "metadatas"])
