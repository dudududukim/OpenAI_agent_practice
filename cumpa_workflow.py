# not impl. the cumpa ACT scenario

# voice_workflow.py
import random
from typing import AsyncIterator, Callable, List

from agents import Agent, Runner, TResponseInputItem, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper

@function_tool
def get_weather(city: str) -> str:
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."

Korean_agent = Agent(
    name="Korean",
    handoff_description="A Korean speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Korean."
    ),
    model="gpt-4o-mini",
)

# agent = Agent(
#     name="Assistant",
#     instructions=prompt_with_handoff_instructions(
#         "You're speaking to a human, so be polite and concise. If the user speaks in Korean, handoff to the Korean agent."
#     ),
#     model="gpt-4o-mini",
#     handoffs=[Korean_agent],
#     tools=[get_weather],
# )


agent = Agent(
    name="Assistant",
    instructions="You're speaking to a human, so be polite and concise. If the user speaks in Korean, answer in Korean.",
    model="gpt-4o-mini",
)

class MyWorkflow(VoiceWorkflowBase):
    def __init__(self, secret_word: str, on_start: Callable[[str], None] | None = None):
        self._input_history: List[TResponseInputItem] = []
        self._current_agent = agent
        self._secret_word = secret_word.lower()
        self._on_start = on_start or (lambda _: None)

    async def run(self, transcription: str) -> AsyncIterator[str]:
        try:
            self._on_start(transcription)
            self._input_history.append({"role": "user", "content": transcription})

            if self._secret_word in transcription.lower():
                reply = "쿰파쿰파쿰파쿰파"
                self._input_history.append({"role": "assistant", "content": reply})
                yield reply
                return

            result = Runner.run_streamed(self._current_agent, self._input_history)
            async for chunk in VoiceWorkflowHelper.stream_text_from(result):
                yield chunk

            # update history / last agent
            self._input_history = result.to_input_list()
            self._current_agent = result.last_agent
        except Exception as exc:
            print(f"[voice_workflow] error: {exc}")
            yield "Sorry, something went wrong while processing your request."
