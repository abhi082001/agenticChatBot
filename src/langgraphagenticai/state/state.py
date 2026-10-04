from typing_extensions import TypedDict, Literal
from typing import Annotated, List, Sequence
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    Represent's the structure of the state used in the graph.
    """
    messages: Annotated[List, add_messages]