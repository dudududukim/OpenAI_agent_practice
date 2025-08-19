from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv

load_dotenv()

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    # final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_homework,
    )

# async def homework_guardrail(ctx, agent, input_data):
#     print(f"Input data: {input_data}")
    
#     # Execute guardrail agent
#     result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
#     print(f"Raw result: {result.final_output}")
#     print(f"Raw result type: {type(result.final_output)}")
    
#     # Convert to structured output
#     final_output = result.final_output_as(HomeworkOutput)
#     print(f"Converted output: {final_output}")
#     print(f"Converted output type: {type(final_output)}")
    
#     # Check individual fields
#     print(f"is_homework: {final_output.is_homework}")
#     print(f"reasoning: {final_output.reasoning}")
    
#     # Calculate tripwire logic
#     tripwire_value = not final_output.is_homework
#     print(f"tripwire_triggered: not {final_output.is_homework} = {tripwire_value}")
    
#     guardrail_output = GuardrailFunctionOutput(
#         output_info=final_output,
#         tripwire_triggered=tripwire_value,
#     )
    
#     print(f"Final GuardrailFunctionOutput: {guardrail_output}")
#     print("-" * 50)
    
#     return guardrail_output


triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    # Example 1: History question
    try:
        result = await Runner.run(triage_agent, "who was the first president of the united states?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

    # Example 2: General/philosophical question
    try:
        result = await Runner.run(triage_agent, "What is life?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

if __name__ == "__main__":
    asyncio.run(main())

"""
 ~/De/I/OpenAI_agent_practice | on main !2 ?5  uv run python main.py                                                          ok | took 9s | base py | at 21:06:32
The first President of the United States was George Washington. He served from 1789 to 1797 after being unanimously elected by the Electoral College. Washington played a crucial role in leading the American colonies to victory in the Revolutionary War and was influential in the drafting of the U.S. Constitution. His leadership established many practices and protocols for the new federal government.
Guardrail blocked this input: Guardrail InputGuardrail triggered tripwire
"""