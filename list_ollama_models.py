import ollama

def list_local_models():
    print("Checking local Ollama instance for installed models...\n")
    try:
        response = ollama.list()
        models = response.get('models', [])
        
        if not models:
            print("No models found. You haven't downloaded any models yet.")
            return

        print(f"{'MODEL NAME':<30} | {'SIZE (GB)':<10}")
        print("-" * 45)
        
        for model in models:
            # Convert bytes to Gigabytes for easier reading
            size_gb = model.get('size', 0) / (1024**3)
            print(f"{model.get('name', 'Unknown'):<30} | {size_gb:.2f} GB")
            
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure the Ollama background app is running.")

if __name__ == "__main__":
    list_local_models()