"""
LLM Agent demonstration using OpenAI's reasoning model with tool usage.
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from weather_tool import get_weather, WEATHER_TOOL_DEFINITION
from prompt_templates import SYSTEM_PROMPT_TEMPLATE, format_prompt

"""
OpenInference Semantic Conventions for LLM Tracing
==================================================

This module demonstrates how to properly instrument LLM applications using OpenInference 
semantic conventions. These conventions provide a standardized way to trace AI/ML workflows
for observability and debugging.

SPAN KINDS:
-----------
1. AGENT: Top-level spans representing an AI agent's complete interaction
   - Use for: Agent sessions, chat interactions, multi-step reasoning
   - Attributes: input.value, output.value, session.id

2. LLM: Spans for Language Model API calls  
   - Use for: OpenAI completions, chat API calls, model inference
   - Attributes: 
     * llm.model_name: Model identifier (e.g., "gpt-4o")
     * llm.provider: Provider name (e.g., "openai")
     * llm.system: AI system (e.g., "openai")
     * llm.prompt_template.template: Prompt template as f-string
     * llm.prompt_template.variables: Template variables as JSON
     * llm.tools: Available tools as JSON array
     * llm.input_messages.{i}.message.role: Message role for input
     * llm.input_messages.{i}.message.content: Message content for input
     * llm.output_messages.{i}.message.role: Message role for output
     * llm.output_messages.{i}.message.content: Message content for output
     * llm.output_messages.{i}.message.tool_calls: Tool calls in output

3. TOOL: Spans for tool/function executions
   - Use for: Function calls, API calls, external tool usage
   - Attributes:
     * tool.name: Function name
     * tool.description: Tool purpose/description  
     * tool.parameters: Function parameters as JSON
     * message.tool_calls.{i}.tool_call.function.name: Tool function name
     * message.tool_calls.{i}.tool_call.function.arguments: Tool arguments
     * message.tool_calls.{i}.tool_call.function.output: Tool result

COMMON ATTRIBUTES:
------------------
- input.value: Input to the span (query, parameters, etc.)
- output.value: Output from the span (response, result, etc.)
- session.id: Session identifier for grouping related interactions
- user.id: User identifier for user-specific tracking

MESSAGE FLATTENING:
-------------------
Instead of storing messages as JSON blobs, flatten them using indexed attributes:
✅ Good: llm.input_messages.0.message.role = "user"
❌ Bad: llm.input_messages = '[{"role": "user", "content": "..."}]'

This provides better searchability and filtering in observability tools.

TOOL CALL PATTERN:
------------------
When LLMs make tool calls, create this span hierarchy:
1. Agent span (top-level)
   ├── LLM span (initial call with tools)
   ├── Tool span (for each tool execution)
   └── LLM span (final response after tools)

EXAMPLE USAGE:
--------------
```python
with tracer.start_as_current_span(
    "Agent Chat",
    attributes={
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
        SpanAttributes.INPUT_VALUE: user_input,
        SpanAttributes.SESSION_ID: "session_123"
    }
) as agent_span:
    # ... agent logic
    agent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, response)
```

For more details, see: https://arize.com/docs/ax/observe/tracing/how-to-tracing-manual/
"""




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

        #TODO - add tool call instrumentation here

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
        #TODO - add agent chat instrumentation here
        
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
            #TODO - add initial llm request instrumentation here
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

                #TODO - add final llm request instrumentation here
                
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

    #TODO - add agent instrumentation here
    
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