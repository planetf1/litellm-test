import litellm
import json

# Define the list of models to test (same as in example2.py)
models_to_test = [
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
]

print("Starting JSON Output Tests for Multiple Models...\n")

for model_name in models_to_test:
    print(f"--- Testing JSON Output for model: {model_name} ---")
    try:
        json_prompt_response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": "What is the capital of France? Please provide your answer in a simple JSON structure with the key 'capital' and the city name as the value. Answer only with a JSON object and no additional text"}],
            response_format={"type": "json_object"},
            tool_choice=None # ADDED: Explicitly set tool_choice=None - to handle potential "toolConfig.toolChoice.tool field" error
        )

        output_content = json_prompt_response.choices[0].message.content.strip()

        try:
            json_output = json.loads(output_content)
            print("\n-- Simple JSON Output Test --")
            print("Status: JSON Output Received and Parsed Successfully")
            print("JSON Response:")
            print(json.dumps(json_output, indent=4))  # Print nicely formatted JSON

        except json.JSONDecodeError as e:
            print("\n-- Simple JSON Output Test --")
            print("Status: JSON Output Error - Could not parse JSON")
            print(f"JSON Decode Error: {e}")
            print("Raw Response Content (Not JSON):")
            print(output_content) # Print raw content if not JSON

    except Exception as e:
        print(f"Error during simple JSON output test for model {model_name}: {e}")
    print("\n--- Finished JSON Output Test for model: {model_name} ---\n")

print("\nJSON Output Tests for All Models Completed.")