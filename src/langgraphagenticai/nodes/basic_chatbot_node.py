from src.langgraphagenticai.state.state import State

class BasicChatbotNode:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        print(state["messages"])
        return {"messages": self.llm.invoke(state["messages"])}