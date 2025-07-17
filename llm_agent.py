"""
LLM Agent demonstration using OpenAI's reasoning model with tool usage.
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from weather_tool import get_weather, WEATHER_TOOL_DEFINITION
from prompt_templates import SYSTEM_PROMPT_TEMPLATE, format_prompt


class LLMAgent:
    """
    A simple LLM agent that uses OpenAI's reasoning model and has access to weather tools.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM agent.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"  # Using OpenAI's reasoning model
        self.tools = [WEATHER_TOOL_DEFINITION]
        self.conversation_history = []
    
    def _execute_tool_call(self, tool_call) -> str:
        """
        Execute a tool call and return the result.
        
        Args:
            tool_call: The tool call object from OpenAI
            
        Returns:
            str: The result of the tool execution
        """
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "get_weather":
            return get_weather(function_args["location"])
        else:
            return f"Unknown function: {function_name}"
    
    def chat(self, user_input: str, use_reasoning: bool = True) -> str:
        """
        Process a user input and return the agent's response.
        
        Args:
            user_input (str): The user's query
            use_reasoning (bool): Whether to use reasoning in the response
            
        Returns:
            str: The agent's response
        """
        # Format the system prompt with the user query
        system_prompt = format_prompt(SYSTEM_PROMPT_TEMPLATE, user_input)
        
        # Build the conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Add conversation history if it exists
        if self.conversation_history:
            messages = [messages[0]] + self.conversation_history + [messages[1]]
        
        print(f"🤖 Processing: '{user_input}'")
        
        try:
            # Make the initial request with tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message
            
            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                print(f"🔧 Using tools: {[tc.function.name for tc in assistant_message.tool_calls]}")
                
                # Add the assistant's message to the conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_result = self._execute_tool_call(tool_call)
                    print(f"🌡️  Tool result for {tool_call.function.name}: {json.loads(tool_result)['location']}")
                    
                    # Add the tool result to the conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Get the final response after tool execution
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )
                
                final_message = final_response.choices[0].message.content
            else:
                final_message = assistant_message.content
            
            # Update conversation history (keep last 6 messages to avoid token limits)
            self.conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": final_message}
            ])
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
            
            return final_message
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        print("🔄 Conversation history cleared")


def demonstrate_agent():
    """
    Demonstrate the LLM agent with various queries.
    """
    print("=" * 60)
    print("🚀 LLM Agent Demo - OpenAI with Weather Tools")
    print("=" * 60)
    
    agent = LLMAgent()
    
    # Example queries to demonstrate different capabilities
    demo_queries = [
        "What's the weather like in San Francisco?",
        "I'm planning a trip to London tomorrow. How's the weather?",
        "Compare the weather between New York and Tokyo",
        "What should I wear today if I'm in London?",
        "What's 2 + 2?",  # Non-weather query to show general reasoning
    ]
    
    for query in demo_queries:
        print(f"\n💬 User: {query}")
        response = agent.chat(query)
        print(f"🤖 Assistant: {response}")
        print("-" * 50)
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    demonstrate_agent() 