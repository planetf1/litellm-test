import litellm
import json
import time
import traceback
from datetime import datetime

# Define the list of models to test
models_to_test = [
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
]

# Ensure error logs are saved to a file
error_log_file = "litellm_bedrock_errors.log"

def weather_tool(location):
    """Simulated weather tool."""
    if location.lower() == "brighton":
        return {"answer": "The weather in Brighton is currently cloudy with a temperature of 10 degrees Celsius."}
    else:
        return {"answer": f"Weather information for {location} is not available in this simulation."}

def test_model(model_name):
    """Tests a given model for various functionalities."""
    results = {
        "model": model_name,
        "capabilities_summary": "",
        "simple_prompt_result": "",
        "json_output_result": "",
        "tool_use_result": "",
        "execution_time": None,
        "tokens": None,
        "cost": None,
        "error": None,
    }
    start_time = time.time()

    try:
        # 1. Summarize Capabilities
        # Corrected line: using get_model_info instead of model_info
        capabilities = litellm.get_model_info(model=model_name)
        results["capabilities_summary"] = f"Model Capabilities: {capabilities}"

        # 2. Simple Prompt
        simple_prompt_response = litellm.completion(model=model_name, messages=[{"role": "user", "content": "What is the capital of France?"}])
        results["simple_prompt_result"] = f"Simple Prompt Response: {simple_prompt_response.choices[0].message.content.strip()}"

        # 3. Structured JSON Output
        # Check if model is a Llama 3 model, if so use response_format, otherwise try to parse
        if "llama3" in model_name:
            json_prompt_response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": "What is the capital of Germany? Please provide your answer in JSON format with a field called 'answer'."}],
                response_format={"type": "json_object"} # Known issue: https://github.com/BerriAI/litellm/issues/1787 - response_format not fully respected by all models, especially in Bedrock.
            )
            try:
                json_output = json.loads(json_prompt_response.choices[0].message.content)
                results["json_output_result"] = f"JSON Output Response: {json_output}"
            except json.JSONDecodeError as e:
                results["json_output_result"] = f"JSON Output Response: Error decoding JSON: {e}, Raw response: {json_prompt_response.choices[0].message.content.strip()}"

        else: # Attempt to parse JSON from other models - might not be structured properly
            json_prompt_response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": "What is the capital of Germany? Please provide your answer in JSON format with a field called 'answer'."}]
            )
            try:
                json_output = json.loads(json_prompt_response.choices[0].message.content)
                results["json_output_result"] = f"JSON Output Response (Parsed): {json_output}"
            except json.JSONDecodeError as e:
                results["json_output_result"] = f"JSON Output Response (Parsed): Could not parse JSON, raw response: {json_prompt_response.choices[0].message.content.strip()}"


        # 4. Tool Use (Simulated Weather)
        # Define tools schema - simple weather tool
        tools = [{
            "type": "function",
            "function": {
                "name": "weather_tool",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
        try: # Wrap tool use in a try-except block
            tool_prompt_response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": "What is the weather like in Brighton today? Please use the weather tool if available."}],
                tools=tools, # Known issue: https://github.com/BerriAI/litellm/issues/1788 - Tool calls might not be fully supported or reliable in Bedrock through LiteLLM.
                tool_choice="auto", # Let the model decide if tool is needed
                drop_params=True # ADDED: Drop unsupported parameters for tool use
            )

            tool_response_content = tool_prompt_response.choices[0].message.content
            tool_calls = tool_prompt_response.choices[0].message.tool_calls

            if tool_calls:
                results["tool_use_result"] = "Tool Call Initiated"
                # For simplicity, assuming only one tool call and function is weather_tool
                function_name = tool_calls[0].function.name
                function_args = json.loads(tool_calls[0].function.arguments)

                if function_name == "weather_tool":
                    weather_result = weather_tool(location=function_args["location"])
                    results["tool_use_result"] = f"Tool Use Response: {weather_result}"
                else:
                    results["tool_use_result"] = "Tool Use Error: Unknown tool function called."

            else:
                results["tool_use_result"] = f"Tool Use Response: No tool call made. Model response: {tool_response_content}"

        except litellm.UnsupportedParamsError as e: # Catch UnsupportedParamsError specifically
            results["tool_use_result"] = f"Tool Use Skipped: {e}" # Indicate tool use was skipped due to error
        except litellm.BadRequestError as e: # Catch BadRequestError for models that explicitly reject tool use
            results["tool_use_result"] = f"Tool Use Not Supported: {e}" # Indicate tool use is not supported by the model


        results["execution_time"] = time.time() - start_time
        results["tokens"] = simple_prompt_response.usage.total_tokens # Using simple prompt tokens as an example - might need to sum across all calls for a full picture
        # Cost calculation might require more detailed token breakdown and pricing info - skipping for now as LiteLLM cost might not be accurate for Bedrock

    except Exception as e:
        results["error"] = f"Error: {e}"
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        with open(error_log_file, "a") as log_file:
            json.dump(error_details, log_file, indent=4)
            log_file.write("\n")

    return results

if __name__ == "__main__":
    all_results = []
    error_summary = {}

    print("Starting LiteLLM Bedrock Model Tests...\n")

    for model in models_to_test:
        print(f"Testing model: {model}")
        model_results = test_model(model)
        all_results.append(model_results)
        if model_results["error"]:
            error_summary[model] = model_results["error"]
        print(f"Finished testing model: {model}\n")

    print("\n--- Test Results Summary ---")
    for result in all_results:
        print(f"\nModel: {result['model']}")
        if result["error"]:
            print(f"Status: Error - {result['error']}")
        else:
            print("Status: Success")
            print(f"Execution Time: {result['execution_time']:.2f} seconds")
            print(f"Tokens Used: {result['tokens']}")
            # print(f"Cost: {result['cost']}") # Cost reporting might not be accurate for Bedrock via LiteLLM

        print(f"Capabilities Summary: {result['capabilities_summary']}")
        print(f"Simple Prompt Result: {result['simple_prompt_result']}")
        print(f"JSON Output Result: {result['json_output_result']}")
        print(f"Tool Use Result: {result['tool_use_result']}")


    if error_summary:
        print("\n--- Error Summary ---")
        print("The following models encountered errors. See error log file 'litellm_bedrock_errors.log' for details.")
        for model, error in error_summary.items():
            print(f"- {model}: {error}")
    else:
        print("\n--- All models tested successfully without errors! ---")

    print("\n--- Test Completed ---")