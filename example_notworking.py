import os
import json
import time
import traceback
from typing import List, Dict, Any
from litellm import completion, get_model_capabilities, AuthenticationError, OpenAIError, BadRequestError, RateLimitError
from tabulate import tabulate
from datetime import datetime

# Models to test
MODELS = [
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
]

# Prompts
CAPABILITIES_PROMPT = "Summarize the capabilities of this model in a few sentences."
CAPITAL_PROMPT = "What is the capital of France?"
STRUCTURED_PROMPT = "Provide the answer to the question 'What is the capital of France?' as a JSON object with a single field named 'answer'."
TOOL_PROMPT = "What is the current weather in London?"  # Using a simple tool request for demonstration


# Error handling and fallback strategy
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class Tool:
    """
    Simple tool implementation for weather information.
    In a real scenario, this would interact with an external API.
    """
    def get_weather(self, location: str) -> str:
        if location.lower() == "london":
            return "The current weather in London is sunny with a temperature of 20 degrees Celsius."
        else:
            return f"Weather information for {location} is not available."

def execute_with_retry(func, *args, **kwargs):
    """
    Executes a function with retry logic for common LLM-related errors.
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return func(*args, **kwargs)
        except (AuthenticationError, BadRequestError, RateLimitError) as e:
            print(f"Error occurred: {e}")
            if "rate_limit" in str(e).lower() or "too many request" in str(e).lower():
                print(f"Rate limit or too many requests error, retrying in {RETRY_DELAY} seconds...")
                retries += 1
                time.sleep(RETRY_DELAY)
            else:
                print(f"Non-retriable error")
                raise
        except OpenAIError as e:
            print(f"OpenAIError occurred. Error details: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            raise
    raise Exception(f"Failed after {MAX_RETRIES} retries.")

def get_model_capabilities_summary(model: str) -> str:
    """
    Retrieves and summarizes the capabilities of a model using LiteLLM.
    """
    try:
        capabilities = execute_with_retry(get_model_capabilities, model=model)
        return f"Capabilities for {model}:\n{capabilities}"
    except Exception as e:
        return f"Failed to get capabilities for {model}: {e}"


def run_completion(model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Runs a completion request with the specified model and prompt.
    """
    try:
        response = execute_with_retry(completion, model=model, messages=messages)
        return response
    except Exception as e:
        raise Exception(f"Error during completion for {model}: {e}") from e

def get_json_response(response):
    """
    Extract the json response from the LLM
    """
    try:
        content = response.choices[0].message.content
        json_data = json.loads(content)
        return json_data
    except json.JSONDecodeError:
        print("Response is not valid JSON")
        raise


def run_tool(tool: Tool, prompt: str) -> str:
    """
    Executes a tool based on the prompt.
    """
    if "weather" in prompt.lower():
        location = prompt.split("in")[-1].strip()
        return tool.get_weather(location)
    else:
        return "Tool not available for this prompt."


def main(models_to_run: List[str] = MODELS):
    """
    Main function to run the tests.
    """

    #ensure the results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    tool = Tool()
    results: List[Dict[str, Any]] = []
    errors_log = []

    for model in models_to_run:
        print(f"\n--- Testing model: {model} ---")
        model_results: Dict[str, Any] = {"model": model}

        try:
            # 1. Summarize capabilities
            print("  - Summarizing capabilities...")
            capabilities_summary = get_model_capabilities_summary(model)
            print(f"    {capabilities_summary}")
            model_results["capabilities"] = capabilities_summary

            # 2. Simple LLM prompt
            print("  - Executing simple prompt...")
            messages = [{"role": "user", "content": CAPITAL_PROMPT}]
            capital_response = run_completion(model, messages)
            print(f"    Response: {capital_response.choices[0].message.content}")
            model_results["simple_response"] = capital_response.choices[0].message.content
            model_results["simple_latency"] = capital_response.usage.total_tokens / (capital_response.get("response_ms",0)/1000.0) if capital_response.get("response_ms",0) != 0 else 0
            model_results["simple_tokens"] = capital_response.usage.total_tokens

            # 3. Structured output prompt
            print("  - Executing structured output prompt...")
            structured_messages = [{"role": "user", "content": STRUCTURED_PROMPT}]
            structured_response = run_completion(model, structured_messages)
            print(f"    Response: {structured_response.choices[0].message.content}")
            json_response = get_json_response(structured_response)
            model_results["structured_response"] = json_response
            model_results["structured_latency"] = structured_response.usage.total_tokens / (structured_response.get("response_ms",0)/1000.0) if structured_response.get("response_ms",0) != 0 else 0
            model_results["structured_tokens"] = structured_response.usage.total_tokens

            # 4. Tool-based prompt
            print("  - Executing tool-based prompt...")
            tool_response = run_tool(tool, TOOL_PROMPT)
            print(f"    Tool Response: {tool_response}")

            #Use the tool response to prompt the LLM with the result
            tool_prompt_messages = [{"role": "user", "content": f"Based on the following information: '{tool_response}' - please explain if the weather is good or bad."}]
            tool_response = run_completion(model, tool_prompt_messages)
            print(f"    LLM Response: {tool_response.choices[0].message.content}")
            model_results["tool_response"] = tool_response.choices[0].message.content
            model_results["tool_latency"] = tool_response.usage.total_tokens / (tool_response.get("response_ms",0)/1000.0) if tool_response.get("response_ms",0) != 0 else 0
            model_results["tool_tokens"] = tool_response.usage.total_tokens

        except Exception as e:
            error_message = f"Error testing model {model}: {e}"
            print(error_message)
            traceback.print_exc()
            errors_log.append({
                "model": model,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            model_results["error"] = error_message

        results.append(model_results)

    # Save error logs to a file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    errors_filename = f"{results_dir}/errors_{timestamp}.json"
    with open(errors_filename, "w") as f:
        json.dump(errors_log, f, indent=4)
    print(f"\nError log saved to {errors_filename}")

    # Create the results table
    table_data = []
    headers = ["Model", "Capabilities", "Simple Response", "Simple Latency(tokens/sec)", "Simple Tokens", "Structured Response", "Structured Latency(tokens/sec)","Structured Tokens", "Tool Response", "Tool Latency(tokens/sec)", "Tool Tokens", "Error"]

    for result in results:
      table_data.append([
          result.get("model", ""),
          result.get("capabilities", "")[:100] + "..." if "capabilities" in result and len(result.get("capabilities","")) > 100 else result.get("capabilities",""),
          result.get("simple_response", "")[:100] + "..." if "simple_response" in result and len(result.get("simple_response","")) > 100 else result.get("simple_response",""),
          result.get("simple_latency",""),
          result.get("simple_tokens",""),
          result.get("structured_response", "") if result.get("structured_response", "") is None else str(result.get("structured_response", ""))[0:100]+"...",
          result.get("structured_latency", ""),
          result.get("structured_tokens",""),
          result.get("tool_response","")[:100] + "..." if "tool_response" in result and len(result.get("tool_response","")) > 100 else result.get("tool_response",""),
          result.get("tool_latency",""),
          result.get("tool_tokens",""),
          result.get("error", "")
        ])
        
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print("\n--- Results Summary ---")
    print(table)
    
    #save to a file
    results_filename = f"{results_dir}/results_{timestamp}.txt"
    with open(results_filename, "w") as f:
        f.write(table)
    print(f"\nResults saved to {results_filename}")


if __name__ == "__main__":
    # Allow running only a specific model (e.g., for debugging)
    if "MODEL" in os.environ:
        specific_model = os.environ["MODEL"]
        if specific_model in MODELS:
            main([specific_model])
        else:
            print(f"Model {specific_model} not found in the list of models.")
    else:
        main()
