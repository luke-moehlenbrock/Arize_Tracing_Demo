# LLM Agent Demo for Observability Workshop

This is a simple demonstration project for exploring LLM + Agent observability using Arize. The project showcases:

- **Prompt Templates**: Configurable system prompts with template variables
- **Tool Usage**: A simple weather tool that returns hardcoded example data
- **Reasoning LLM**: Uses OpenAI's GPT-4o model for intelligent responses
- **Agent Workflow**: Demonstrates how an LLM can reason about when to use tools

## 🚀 Quick Start

### 1. Create Virtual Environment (Recommended)

Create and activate a Python virtual environment to isolate dependencies:

**Using venv (Python 3.3+):**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp env_example.txt .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The Arize keys are optional for the basic demo but will be needed for observability instrumentation:

```env
ARIZE_API_KEY=your_arize_api_key_here
ARIZE_SPACE_ID=your_arize_space_id_here
ARIZE_PROJECT_NAME=llm-agent-demo
```

### 4. Run the Demo

**Interactive Mode (default):**
```bash
python main.py
```

**Automated Demonstration:**
```bash
python main.py --demo
```

## 📁 Project Structure

```
├── main.py                 # Main entry point with interactive mode
├── llm_agent.py            # Core LLM agent with tool usage
├── weather_tool.py         # Simple weather tool with hardcoded data
├── prompt_templates.py     # Configurable prompt templates
├── requirements.txt        # Python dependencies
├── env_example.txt        # Environment configuration template
└── README.md              # This file
```

## 🔧 Key Components

### LLM Agent (`llm_agent.py`)
- Uses OpenAI's GPT-4o model
- Implements tool calling for weather queries
- Maintains conversation history
- Demonstrates reasoning about when to use tools

### Weather Tool (`weather_tool.py`)
- Simple function that returns hardcoded weather data
- Supports several cities (San Francisco, New York, London, Tokyo)
- Demonstrates tool integration with OpenAI function calling

### Prompt Templates (`prompt_templates.py`)
- Configurable system prompts with template variables
- Different prompt styles for various use cases
- Easy to modify for different agent behaviors

## 💡 Usage Examples

The agent can handle various types of queries:

**Weather Queries:**
- "What's the weather like in San Francisco?"
- "I'm planning a trip to London tomorrow. How's the weather?"
- "Compare the weather between New York and Tokyo"

**General Queries:**
- "What should I wear today if I'm in London?"
- "What's 2 + 2?"
- Any other general question

## 🔍 Observability Workshop Notes

This project is designed as a foundation for exploring LLM observability. The `arize-otel` package is already included in dependencies. During the workshop, we'll add:

1. **OpenTelemetry Instrumentation**: Add spans for prompt construction, LLM inference, and tool usage
2. **Arize Integration**: Send traces and logs to Arize for monitoring
3. **Metrics Collection**: Track latency, token usage, and tool effectiveness
4. **Evaluation Framework**: Set up evals for response quality

## 🎯 Workshop Agenda Items Covered

- ✅ **Prompt Templates**: System prompt with user query variable
- ✅ **Tool Usage**: Weather tool with hardcoded responses
- ✅ **Reasoning LLM**: OpenAI GPT-4o for intelligent responses
- ✅ **Environment Setup**: Ready for Arize instrumentation
- 🔄 **Instrumentation**: To be added during workshop
- 🔄 **Observability**: To be configured during workshop

## 🛠️ Extending the Demo

This is intentionally a simple foundation. You can extend it by:

- Adding more tools (calculator, web search, etc.)
- Implementing more sophisticated prompt templates
- Adding different reasoning strategies
- Creating more complex agent workflows

The simplicity makes it perfect for demonstrating observability concepts without getting lost in complex application logic. 