import litellm

print(f"litellm version: {litellm.__version__}")

try:
    from litellm import model_info
    print("litellm.model_info imported successfully")
except ImportError as e:
    print(f"Error importing litellm.model_info: {e}")
