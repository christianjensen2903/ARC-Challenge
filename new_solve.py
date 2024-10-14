from langgraph.graph import END, START, Graph
import dotenv
from langchain_openai import ChatOpenAI
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from example_converter import ExampleConverter
import logging


dotenv.load_dotenv()


def load_data() -> tuple[dict, dict]:
    with open("data/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)

    with open("data/arc-agi_training_solutions.json") as f:
        solutions = json.load(f)

    return challenges, solutions


challenges, solutions = load_data()


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def load_demonstrations(id: str) -> list[dict]:
    logging.info(f"Loading demonstrations for {id}")
    demonstrations = challenges[id]["train"]
    return demonstrations


def format_demonstrations(demonstrations: list[dict]) -> str:
    logging.info("Formatting demonstrations")
    example_converter = ExampleConverter()
    return example_converter.extract(demonstrations)


def call_model(formatted_demonstrations: str) -> BaseMessage:
    logging.info("Calling model")
    response = model.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant that extracts the pattern from the demonstrations."
            ),
            HumanMessage(content=formatted_demonstrations),
        ]
    )
    return response


workflow = Graph()
workflow.add_node("load", load_demonstrations)
workflow.add_node("format", format_demonstrations)
workflow.add_node("agent", call_model)

workflow.add_edge(START, "load")
workflow.add_edge("load", "format")
workflow.add_edge("format", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()

if __name__ == "__main__":
    final_state = app.invoke("05f2a901")
    print(final_state)
