import litellm
import importlib
        
try:
    # Try to reload litellm module
    litellm = importlib.reload(litellm)
    print("litellm module reloaded successfully.")
except Exception as reload_error:
    print(f"Error reloading litellm module: {reload_error}")
        
print(f"litellm version: {litellm.__version__}")
        
try:
    from litellm import model_info
    print("litellm.model_info imported successfully")
except ImportError as e:
    print(f"Error importing litellm.model_info: {e}")