"""
LLM Agent demonstration using OpenAI's reasoning model with tool usage.
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from weather_tool import get_weather, WEATHER_TOOL_DEFINITION
from prompt_templates import SYSTEM_PROMPT_TEMPLATE, format_prompt

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues, MessageAttributes


def _set_message_attributes(span, messages_list, attr_prefix):
    """
    Helper function to set flattened message attributes on a span.
    
    Args:
        span: The span to set attributes on
        messages_list: List of message dictionaries
        attr_prefix: Either SpanAttributes.LLM_INPUT_MESSAGES or SpanAttributes.LLM_OUTPUT_MESSAGES
    """
    for i, message in enumerate(messages_list):
        span.set_attribute(f"{attr_prefix}.{i}.{MessageAttributes.MESSAGE_ROLE}", message.get("role", ""))
        span.set_attribute(f"{attr_prefix}.{i}.{MessageAttributes.MESSAGE_CONTENT}", message.get("content", ""))
        
        # Handle tool calls if present
        if "tool_calls" in message and message["tool_calls"]:
            span.set_attribute(f"{attr_prefix}.{i}.{MessageAttributes.MESSAGE_TOOL_CALLS}", json.dumps(message["tool_calls"]))
        
        # Handle tool call id if present (for tool messages)
        if "tool_call_id" in message:
            span.set_attribute(f"{attr_prefix}.{i}.{MessageAttributes.MESSAGE_TOOL_CALL_ID}", message["tool_call_id"])


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
        self.tracer = trace.get_tracer(__name__)
    
    def _execute_tool_call(self, tool_call) -> str:
        """
        Execute a tool call and return the result.
        
        Args:
            tool_call: The tool call object from OpenAI
            
        Returns:
            str: The result of the tool execution
        """
        with self.tracer.start_as_current_span(
            name=f"Tool: {tool_call.function.name}",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.TOOL_NAME: tool_call.function.name,
                SpanAttributes.TOOL_DESCRIPTION: "Get weather information for a location",
                SpanAttributes.TOOL_PARAMETERS: tool_call.function.arguments,
                SpanAttributes.INPUT_VALUE: tool_call.function.arguments,
            }
        ) as tool_span:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "get_weather":
                result = get_weather(function_args["location"])
                tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, result)
                return result
            else:
                error_msg = f"Unknown function: {function_name}"
                tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, error_msg)
                return error_msg
    
    def chat(self, user_input: str, use_reasoning: bool = True) -> str:
        """
        Process a user input and return the agent's response.
        
        Args:
            user_input (str): The user's query
            use_reasoning (bool): Whether to use reasoning in the response
            
        Returns:
            str: The agent's response
        """
        with self.tracer.start_as_current_span(
            name="Agent Chat",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                SpanAttributes.INPUT_VALUE: user_input,
            }
        ) as agent_span:
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
            
            # try:
            # Make the initial request with tools
            with self.tracer.start_as_current_span(
                name="LLM Call - Initial",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                    SpanAttributes.LLM_MODEL_NAME: self.model,
                    SpanAttributes.LLM_PROVIDER: "openai",
                    SpanAttributes.LLM_SYSTEM: "openai",
                    SpanAttributes.LLM_PROMPT_TEMPLATE: SYSTEM_PROMPT_TEMPLATE,
                    SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES: json.dumps({"user_query": user_input}),
                    SpanAttributes.LLM_TOOLS: json.dumps(self.tools),
                }
            ) as llm_span:
                # Set input messages with flattened attributes
                _set_message_attributes(llm_span, messages, SpanAttributes.LLM_INPUT_MESSAGES)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.7
                )
                
                assistant_message = response.choices[0].message
                
                # Create output message dict and set flattened attributes
                output_message = {
                    "role": "assistant",
                    "content": assistant_message.content,
                }
                if assistant_message.tool_calls:
                    output_message["tool_calls"] = [tc.to_dict() if hasattr(tc, 'to_dict') else tc for tc in assistant_message.tool_calls]
                
                _set_message_attributes(llm_span, [output_message], SpanAttributes.LLM_OUTPUT_MESSAGES)
            
            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                print(f"🔧 Using tools: {[tc.function.name for tc in assistant_message.tool_calls]}")
                
                # Add the assistant's message to the conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [tc.to_dict() if hasattr(tc, 'to_dict') else tc for tc in assistant_message.tool_calls]
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
                with self.tracer.start_as_current_span(
                    name="LLM Call - Final",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                        SpanAttributes.LLM_MODEL_NAME: self.model,
                        SpanAttributes.LLM_PROVIDER: "openai",
                        SpanAttributes.LLM_SYSTEM: "openai",
                    }
                ) as final_llm_span:
                    # Set input messages with flattened attributes
                    _set_message_attributes(final_llm_span, messages, SpanAttributes.LLM_INPUT_MESSAGES)
                    
                    final_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7
                    )
                    
                    final_message = final_response.choices[0].message.content
                    
                    # Create output message and set flattened attributes
                    output_message = {
                        "role": "assistant",
                        "content": final_message
                    }
                    _set_message_attributes(final_llm_span, [output_message], SpanAttributes.LLM_OUTPUT_MESSAGES)
            else:
                final_message = assistant_message.content
            
            # Update conversation history (keep last 6 messages to avoid token limits)
            self.conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": final_message}
            ])
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
            
            agent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, final_message)
            return final_message
                
            # except Exception as e:
                # error_msg = f"Sorry, I encountered an error: {str(e)}"
                # agent_span.record_exception(e)
                # agent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, error_msg)
                # return error_msg
    
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
    
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(
        name="Agent Demo Session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: "demo_session",
            "agent.mode": "demo"
        }
    ) as demo_span:
        agent = LLMAgent()
        
        # Example queries to demonstrate different capabilities
        demo_queries = [
            "What's the weather like in San Francisco?",
            "I'm planning a trip to London tomorrow. How's the weather?",
            "Compare the weather between New York and Tokyo",
            "What should I wear today if I'm in London?",
            "What's 2 + 2?",  # Non-weather query to show general reasoning
        ]
        
        for i, query in enumerate(demo_queries):
            print(f"\n💬 User: {query}")
            response = agent.chat(query)
            print(f"🤖 Assistant: {response}")
            print("-" * 50)
        
        demo_span.set_attribute("demo.queries_count", len(demo_queries))
        print("\n✅ Demo completed!")


if __name__ == "__main__":
    demonstrate_agent() 