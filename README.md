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


### 2. Voice Agent

[![OpenAI Agents SDK](https://img.shields.io/badge/Agents%20SDK-Voice%20Quickstart-000?logo=openai&logoColor=white)](https://openai.github.io/openai-agents-python/voice/quickstart/)

<img src="assets/imgs/voice_agent.png" alt="alt text" width="500px">

**Voice angent workflow SDK** : [SingleAgentVoiceWorkflow](https://openai.github.io/openai-agents-python/ref/voice/workflow/#agents.voice.workflow.SingleAgentVoiceWorkflow)

**PCM** : Pulse Code Modulation



# References
- handoff : https://openai.github.io/openai-agents-python/ref/agent/?utm_source=chatgpt.com
- voice agent : https://openai.github.io/openai-agents-python/voice/quickstart/