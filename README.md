# Following below openai agent SDK

[![OpenAI Agents SDK](https://img.shields.io/badge/Agents%20SDK-Quickstart-000?logo=openai&logoColor=white)](https://openai.github.io/openai-agents-python/quickstart/)


⟶ A companion to [OpenAI_S2S]((https://github.com/dudududukim/OpenAI_S2S)), purpose-built to implement a chained architecture.

# Setup

```bash
uv sync
uv run python main.py
```

## Summuries

### 1. Agent definition

Agent = name / instructions / handoff_description

```python
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)
```

- Handoff ⊂ Tool
- Tool: function-calling, hosted tools(retrieval/web/computer use), agents-as-tools





# References
- handoff : https://openai.github.io/openai-agents-python/ref/agent/?utm_source=chatgpt.com
- 