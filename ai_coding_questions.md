```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two 1D vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Prevent division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))

def search_documents(query_embedding: list[float], doc_embeddings: list[list[float]], top_k: int = 3):
    """
    Search for the top_k most similar documents given a query embedding 
    and a list of document embeddings.
    """
    q_vec = np.array(query_embedding)
    doc_matrix = np.array(doc_embeddings)
    
    # Compute similarity scores for all documents at once
    dot_products = np.dot(doc_matrix, q_vec)
    query_norm = np.linalg.norm(q_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1) ## , by setting axis = 1, you instruct NumPy to row-by-row calculate -> will get list of magnitudes
    
    # Avoid division by zero
    denominator = doc_norms * query_norm
    denominator[denominator == 0] = 1e-10
    
    scores = dot_products / denominator
    
    # Sort indices in descending order of similarity scores
    sorted_indices = np.argsort(scores)[::-1]
    
    # Return top_k results
    results = []
    for idx in sorted_indices[:top_k]:
        results.append({"index": int(idx), "score": float(scores[idx])})
        
    return results
top_results = search_documents(query_vec, documents, top_k=3)

# 2. Grab the top result (index 0 is the highest score)
best_match = top_results[0]
best_index = best_match["index"]
best_score = best_match["score"]
best_vector = documents[best_index]
```


LOOP 
```python

import json
import inspect
from typing import Callable, Dict, Any, List

# 1. Define local tool functions
def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression."""
    try:
        # Use a restricted evaluation for safety in production
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Mock weather API response
    weather_data = {
        "london": "15°C, Cloudy",
        "tokyo": "22°C, Sunny",
        "pune": "28°C, Partly Cloudy"
    }
    loc = location.lower().strip()
    return weather_data.get(loc, f"Weather data not found for {location}.")

# Registry mapping names to actual python callables
TOOLS: Dict[str, Callable] = {
    "calculate": calculate,
    "get_weather": get_weather
}

# 2. Tool dispatcher / executor
def execute_tool(name: str, arguments: dict) -> str:
    if name not in TOOLS:
        return f"Error: Tool {name} does not exist."
    try:
        func = TOOLS[name]
        # Unpack dictionary arguments into the function call
        result = func(**arguments)
        return str(result)
    except Exception as e:
        return f"Error executing {name}: {e}"

# 3. Mock LLM Interface for Demonstration
class MockLLM:
    """Simulates LLM behavior generating a tool call first, then a final text answer."""
    def __init__(self):
        self.step = 0

    def chat(self, messages: List[dict]) -> dict:
        self.step += 1
        if self.step == 1:
            # Model decides to call a tool
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "name": "get_weather",
                    "arguments": {"location": "Pune"}
                }]
            }
        else:
            # Model returns final text response after getting observation
            return {
                "role": "assistant",
                "content": "The current weather in Pune is 28°C and Partly Cloudy.",
                "tool_calls": []
            }

# 4. The Agent Loop
def run_agent_loop(user_prompt: str, max_rounds: int = 5):
    llm = MockLLM()
    messages = [{"role": "user", "content": user_prompt}]
    
    for round_idx in range(max_rounds):
        print(f"\n--- Round {round_idx + 1} ---")
        response = llm.chat(messages)
        
        # Append assistant response to history
        messages.append(response)
        
        tool_calls = response.get("tool_calls", [])
        
        # If no tool calls, the agent is done
        if not tool_calls:
            print("Final Answer from LLM:")
            print(response["content"])
            break
            
        # Execute tool calls and append results back into history
        for call in tool_calls:
            name = call["name"]
            args = call["arguments"]
            print(f"-> LLM requested tool: {name} with args: {args}")
            
            observation = execute_tool(name, args)
            print(f"<- Tool execution result (Observation): {observation}")
            
            # Feed result back as a tool message/observation
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": observation
            })

# Run the loop
if __name__ == "__main__":
    run_agent_loop("What is the weather like in Pune right now?")
```


CHunking
```python
    words = document.split()
    chunk_size = 50
    overlap = 15
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move the window forward by chunk_size minus the overlap
        start += (chunk_size - overlap)

 # Combine retrieved chunks into a single reference context
context = "\n---\n".join(retrieved_chunks)
```


charcter 
```python 
def chunk_by_characters(document, chunk_size=50, overlap=15):
    chunks = []
    start = 0
    step = chunk_size - overlap  # move forward this many characters each time

    while start < len(document):
        end = start + chunk_size
        chunk = document[start:end]
        chunks.append(chunk)
        start += step

    return chunks
```

