#!/usr/bin/env python3
"""
OpenAI API Compatibility Test Script

This script tests whether your OpenAI-compatible API supports all the parameters
and features used in locobench/agents/openai_agent.py

Usage:
    python test_openai_api.py --api-key YOUR_API_KEY --model gpt-4o
    python test_openai_api.py --api-key YOUR_API_KEY --base-url https://your-gateway.com/v1 --model gpt-4o
"""

import asyncio
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Error: OpenAI package not installed. Run: pip install openai")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAICompatibilityTester:
    """Tests OpenAI API compatibility with locobench agent requirements"""

    # Same model configs from openai_agent.py
    MODEL_CONFIGS = {
        "gpt-5.2-1211-global": {
            "max_tokens": 200000,
            "supports_functions": True,
            "use_max_completion_tokens": True,
            "fixed_temperature": 1.0,
        },
        "qwen3-max": {
            "max_tokens": 100000,
            "supports_functions": True,
            "use_max_completion_tokens": True,
            "fixed_temperature": 1.0,
        },
        "gemini-3-flash-preview": {
            "max_tokens": 100000,
            "supports_functions": True,
            "use_max_completion_tokens": True,
            "fixed_temperature": 1.0,
        },
    }

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o"):
        self.model = model
        print("model: ", model)
        self.model_config = self.MODEL_CONFIGS.get(model)

        # Initialize OpenAI client (same logic as openai_agent.py)
        client_kwargs = {}

        if base_url:
            # Gateway mode - use X-Api-Key header
            client_kwargs["base_url"] = base_url
            client_kwargs["api_key"] = api_key  # Required by OpenAI SDK
            logger.info(f"Using Gateway mode: {base_url}")
            self.auth_mode = "gateway"
        else:
            # Direct OpenAI API mode
            client_kwargs["api_key"] = api_key
            logger.info("Using direct OpenAI API mode")
            self.auth_mode = "direct"

        self.client = AsyncOpenAI(**client_kwargs)

        # Test results storage
        self.test_results = {}

    def get_test_tools(self) -> List[Dict[str, Any]]:
        """Get simple test tools for function calling"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculator_add",
                    "description": "Add two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "First number"
                            },
                            "b": {
                                "type": "number",
                                "description": "Second number"
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_system_read",
                    "description": "Read a file from the filesystem",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]

    def execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute a tool call (mock implementation)"""
        function_name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in arguments"}

        # Mock implementations
        if function_name == "calculator_add":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            result = a + b
            return {"result": f"{a} + {b} = {result}"}

        elif function_name == "get_current_time":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return {"result": f"Current time: {current_time}"}

        elif function_name == "file_system_read":
            file_path = arguments.get("file_path", "")
            return {"result": f"Mock file content for: {file_path}"}

        else:
            return {"error": f"Unknown function: {function_name}"}

    async def test_basic_chat(self) -> Dict[str, Any]:
        """Test basic chat completion"""
        logger.info("Testing basic chat completion...")

        try:
            # Prepare request parameters (same logic as openai_agent.py)
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Hello! Please respond with 'Hello World'"}
                ]
            }

            # Handle temperature parameter
            if self.model_config.get("omit_temperature", False):
                # o-series: omit temperature
                pass
            elif "fixed_temperature" in self.model_config:
                # GPT-5 series: fixed temperature
                kwargs["temperature"] = self.model_config["fixed_temperature"]
            else:
                # Normal models: custom temperature
                kwargs["temperature"] = 0.1

            # Handle max tokens parameter
            if self.model_config.get("use_max_completion_tokens", False):
                kwargs["max_completion_tokens"] = 100
            else:
                kwargs["max_tokens"] = 100

            print("kwargs: ", dict(kwargs))
            response = await self.client.chat.completions.create(**kwargs)

            # Check response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                usage = response.usage if hasattr(response, 'usage') else None

                return {
                    "success": True,
                    "content": content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else None,
                        "completion_tokens": usage.completion_tokens if usage else None,
                        "total_tokens": usage.total_tokens if usage else None
                    },
                    "parameters_sent": kwargs
                }
            else:
                return {"success": False, "error": "No response choices"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_function_calling(self) -> Dict[str, Any]:
        """Test function calling capability"""
        logger.info("Testing function calling...")

        if not self.model_config.get("supports_functions", False):
            return {"success": False, "error": "Model does not support function calling"}

        try:
            tools = self.get_test_tools()

            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Please add 15 and 25 together using the calculator tool."}
                ],
                "tools": tools,
                "tool_choice": "auto"
            }

            # Handle temperature
            if self.model_config.get("omit_temperature", False):
                pass
            elif "fixed_temperature" in self.model_config:
                kwargs["temperature"] = self.model_config["fixed_temperature"]
            else:
                kwargs["temperature"] = 0.1

            # Handle max tokens
            if self.model_config.get("use_max_completion_tokens", False):
                kwargs["max_completion_tokens"] = 500
            else:
                kwargs["max_tokens"] = 500

            response = await self.client.chat.completions.create(**kwargs)

            # Check if model made tool calls
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                
                result = {
                    "success": True,
                    "content": message.content,
                    "has_tool_calls": hasattr(message, 'tool_calls') and message.tool_calls is not None,
                    "tool_calls": [],
                    "parameters_sent": kwargs
                }

                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_result = self.execute_tool_call(tool_call)
                        result["tool_calls"].append({
                            "id": tool_call.id,
                            "function_name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                            "result": tool_result
                        })

                return result
            else:
                return {"success": False, "error": "No response choices"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_multi_turn_with_tools(self) -> Dict[str, Any]:
        logger.info("Testing multi-turn conversation with tools...")

        if not self.model_config.get("supports_functions", False):
            return {"success": False, "error": "Model does not support function calling"}

        tools = self.get_test_tools()

        messages = [
            {"role": "user", "content": "Add 10 and 20, then tell me the current time."}
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        if not self.model_config.get("omit_temperature", False):
            kwargs["temperature"] = self.model_config.get("fixed_temperature", 0.1)

        if self.model_config.get("use_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = 1000
        else:
            kwargs["max_tokens"] = 1000

        response1 = await self.client.chat.completions.create(**kwargs)

        conversation_log = []

        if not (response1.choices and len(response1.choices) > 0):
            return {"success": False, "error": "No response choices in turn 1"}

        message1 = response1.choices[0].message

        # 固定住 assistant 消息对象，后面不要再用 messages[-1] 去写它
        assistant_msg = {
            "role": "assistant",
            "content": message1.content,
        }
        messages.append(assistant_msg)

        tool_calls = getattr(message1, "tool_calls", None)
        if tool_calls:
            assistant_msg["tool_calls"] = []

            for tool_call in tool_calls:
                assistant_msg["tool_calls"].append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                })

                tool_result = self.execute_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                })

                conversation_log.append({
                    "type": "tool_call",
                    "function": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                    "result": tool_result,
                })

        # 第二轮
        messages.append({
            "role": "user",
            "content": "Great! Now multiply the sum by 2."
        })

        response2 = await self.client.chat.completions.create(**kwargs)

        if response2.choices and len(response2.choices) > 0:
            message2 = response2.choices[0].message
            return {
                "success": True,
                "turns": 2,
                "final_content": message2.content,
                "conversation_log": conversation_log,
                "message_count": len(messages),
            }

        return {"success": False, "error": "Failed to complete multi-turn conversation"}



    async def test_parameter_variations(self) -> Dict[str, Any]:
        """Test various parameter combinations"""
        logger.info("Testing parameter variations...")

        results = {}

        # Test different temperatures (if supported)
        if not self.model_config.get("omit_temperature", False):
            temperatures = [0.0, 0.5, 1.0]
            if not self.model_config.get("fixed_temperature"):
                # Only test different temps if not fixed
                for temp in temperatures:
                    try:
                        kwargs = {
                            "model": self.model,
                            "messages": [{"role": "user", "content": f"Say 'Temperature test {temp}'"}],
                            "temperature": temp
                        }

                        if self.model_config.get("use_max_completion_tokens", False):
                            kwargs["max_completion_tokens"] = 50
                        else:
                            kwargs["max_tokens"] = 50

                        response = await self.client.chat.completions.create(**kwargs)
                        results[f"temperature_{temp}"] = {
                            "success": True,
                            "content": response.choices[0].message.content if response.choices else None
                        }
                    except Exception as e:
                        results[f"temperature_{temp}"] = {"success": False, "error": str(e)}

        # Test max_tokens vs max_completion_tokens
        token_limits = [50, 100, 200]
        for limit in token_limits:
            # Test max_tokens
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Count from 1 to 10."}],
                    "max_tokens": limit
                }

                if not self.model_config.get("omit_temperature", False):
                    if self.model_config.get("fixed_temperature"):
                        kwargs["temperature"] = self.model_config["fixed_temperature"]
                    else:
                        kwargs["temperature"] = 0.1

                response = await self.client.chat.completions.create(**kwargs)
                results[f"max_tokens_{limit}"] = {
                    "success": True,
                    "content": response.choices[0].message.content if response.choices else None
                }
            except Exception as e:
                results[f"max_tokens_{limit}"] = {"success": False, "error": str(e)}

            # Test max_completion_tokens (if model supports it)
            if self.model_config.get("use_max_completion_tokens", False):
                try:
                    kwargs = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Count from 1 to 10."}],
                        "max_completion_tokens": limit
                    }

                    if not self.model_config.get("omit_temperature", False):
                        if self.model_config.get("fixed_temperature"):
                            kwargs["temperature"] = self.model_config["fixed_temperature"]
                        else:
                            kwargs["temperature"] = 0.1

                    response = await self.client.chat.completions.create(**kwargs)
                    results[f"max_completion_tokens_{limit}"] = {
                        "success": True,
                        "content": response.choices[0].message.content if response.choices else None
                    }
                except Exception as e:
                    results[f"max_completion_tokens_{limit}"] = {"success": False, "error": str(e)}

        return results

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all compatibility tests"""
        logger.info(f"Starting compatibility tests for model: {self.model}")
        logger.info(f"Authentication mode: {self.auth_mode}")
        logger.info(f"Model config: {self.model_config}")

        results = {
            "model": self.model,
            "auth_mode": self.auth_mode,
            "model_config": self.model_config,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }

        # Test basic chat
        results["tests"]["basic_chat"] = await self.test_basic_chat()

        # Test function calling
        results["tests"]["function_calling"] = await self.test_function_calling()

        # Test multi-turn with tools
        results["tests"]["multi_turn_tools"] = await self.test_multi_turn_with_tools()

        # Test parameter variations
        results["tests"]["parameter_variations"] = await self.test_parameter_variations()

        return results


def print_results(results: Dict[str, Any]):
    """Print test results in a readable format"""
    print("\n" + "="*80)
    print("OPENAI API COMPATIBILITY TEST RESULTS")
    print("="*80)

    print(f"Model: {results['model']}")
    print(f"Auth Mode: {results['auth_mode']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Supports Functions: {results['model_config'].get('supports_functions', False)}")
    print(f"Uses max_completion_tokens: {results['model_config'].get('use_max_completion_tokens', False)}")
    print(f"Omits Temperature: {results['model_config'].get('omit_temperature', False)}")

    print("\n" + "-"*60)
    print("TEST RESULTS:")
    print("-"*60)

    for test_name, test_result in results["tests"].items():
        print(f"\n📋 {test_name.upper().replace('_', ' ')}")

        if isinstance(test_result, dict) and "success" in test_result:
            status = "✅ PASS" if test_result["success"] else "❌ FAIL"
            print(f"   Status: {status}")

            if not test_result["success"] and "error" in test_result:
                print(f"   Error: {test_result['error']}")

            if "content" in test_result and test_result["content"]:
                content = test_result["content"][:100] + "..." if len(test_result["content"]) > 100 else test_result["content"]
                print(f"   Response: {content}")

            if "tool_calls" in test_result:
                print(f"   Tool Calls Made: {len(test_result['tool_calls'])}")
                for tool_call in test_result["tool_calls"]:
                    print(f"     - {tool_call['function_name']}: {tool_call.get('result', {})}")

            if "usage" in test_result and test_result["usage"]:
                usage = test_result["usage"]
                if usage.get("total_tokens"):
                    print(f"   Token Usage: {usage['total_tokens']} total ({usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion)")

        elif isinstance(test_result, dict):
            # Parameter variations
            for param_name, param_result in test_result.items():
                status = "✅ PASS" if param_result.get("success", False) else "❌ FAIL"
                print(f"   {param_name}: {status}")
                if not param_result.get("success", False):
                    print(f"     Error: {param_result.get('error', 'Unknown error')}")

    print("\n" + "="*80)
    print("SUMMARY:")

    # Count passed/failed tests
    total_tests = 0
    passed_tests = 0

    for test_name, test_result in results["tests"].items():
        if isinstance(test_result, dict) and "success" in test_result:
            total_tests += 1
            if test_result["success"]:
                passed_tests += 1
        elif isinstance(test_result, dict):
            # Parameter variations
            for param_result in test_result.values():
                total_tests += 1
                if param_result.get("success", False):
                    passed_tests += 1

    print(f"Tests Passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Your API is fully compatible.")
    elif passed_tests > 0:
        print("⚠️  Some tests passed. Your API has partial compatibility.")
    else:
        print("💥 All tests failed. Your API may not be compatible.")

    print("="*80)


async def main():
    parser = argparse.ArgumentParser(description="Test OpenAI API compatibility")
    parser.add_argument("--api-key", default="", help="Your API key")
    parser.add_argument("--base-url", default="" ,help="Base URL for gateway APIs (optional)")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model name to test (default: gpt-4o)")
    parser.add_argument("--output", help="./result")

    args = parser.parse_args()

    # Run tests
    tester = OpenAICompatibilityTester(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model
    )

    results = await tester.run_all_tests()

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())