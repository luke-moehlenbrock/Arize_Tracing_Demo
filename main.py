"""
Main entry point for the LLM Agent demonstration.
"""

import os
import sys
from dotenv import load_dotenv
from llm_agent import LLMAgent, demonstrate_agent

from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues, MessageAttributes


def setup_environment():
    """Load environment variables and check configuration."""
    load_dotenv()
    
    # Check for required OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file based on .env.example and add your OpenAI API key.")
        return False
    
    # Optional Arize keys (for future instrumentation)
    arize_keys = ["ARIZE_API_KEY", "ARIZE_SPACE_ID", "ARIZE_PROJECT_NAME"]
    missing_arize = [key for key in arize_keys if not os.getenv(key)]
    
    if missing_arize:
        print("⚠️  Note: Some Arize configuration is missing (for future observability):")
        for key in missing_arize:
            print(f"   - {key}")
        print("   This is fine for the basic demo, but you'll need these for instrumentation later.")

    tracer_provider = register(
        space_id = os.getenv("ARIZE_SPACE_ID"),
        api_key = os.getenv("ARIZE_API_KEY"),
        project_name = os.getenv("ARIZE_PROJECT_NAME"),
    )

    OpenAIInstrumentor(tracer_provider=tracer_provider).instrument()

    print("✅ Environment setup complete!")
    return tracer_provider


def interactive_mode():
    """Run the agent in interactive mode."""
    print("\n🎯 Interactive Mode - Chat with the AI Agent")
    print("Type 'quit', 'exit', or 'q' to stop")
    print("Type 'demo' to run the automated demonstration")
    print("Type 'reset' to clear conversation history")
    print("-" * 50)

    # Tracer allows us to create spans and traces
    tracer = trace.get_tracer(__name__)
    
    agent = LLMAgent()
    
    with tracer.start_as_current_span(
        name="Interactive Agent Session", 
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: "interactive_session",
            "agent.mode": "interactive"
        }
    ) as span:
        messages = []
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                messages.append({"role": "user", "content": user_input})
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'demo':
                    demonstrate_agent()
                    continue
                elif user_input.lower() == 'reset':
                    agent.reset_conversation()
                    continue
                elif not user_input:
                    continue
                
                
                response = agent.chat(user_input)
                print(f"🤖 Assistant: {response}")
                
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                span.record_exception(e)


def main():
    """Main function to run the demonstration."""
    print("🌟 LLM Agent Demo for Observability Workshop")
    print("=" * 60)
    
    # Setup environment
    tracer_provider = setup_environment()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--demo', '-d']:
            demonstrate_agent()
        elif sys.argv[1] in ['--interactive', '-i']:
            interactive_mode()
        elif sys.argv[1] in ['--help', '-h']:
            print("\nUsage:")
            print("  python main.py              # Interactive mode (default)")
            print("  python main.py --demo       # Run automated demonstration")
            print("  python main.py --interactive # Interactive chat mode")
            print("  python main.py --help       # Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default to interactive mode
        interactive_mode()

    tracer_provider.shutdown()


if __name__ == "__main__":
    main() 