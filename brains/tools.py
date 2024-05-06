from datetime import datetime

from loguru import logger
from openai.types.chat import ChatCompletionToolParam


def move_item(item: str, target_place: str, source_place: str | None = None) -> str:
    logger.info(
        f"Received tool request with args: item: '{item}', target_place: '{target_place}', source_place: '{source_place}'"
    )
    return f"I've sucessfully moved the {item} to the {target_place}" + (
        f"from the {source_place}." if source_place else "."
    )


move_item_tool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "move_item",
        "description": "Moves item A, which is optionally near/on place B, to place C. If you are unsure about an item or a place, ask user before calling this function",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "Name of the item, e.g. cup",
                },
                "source_place": {
                    "type": "string",
                    "description": "Place where to find an item, e.g. chair. Optional.",
                },
                "target_place": {
                    "type": "string",
                    "description": "Place where to put an item, e.g. table. If person asks to bring him something, then this field should be 'person'",
                },
            },
            "required": ["item", "target_place"],
        },
    },
}


def get_time() -> str:
    logger.info(f"Received tool request")
    return f"Now is {datetime.today().strftime('%I:%M %p')}."


get_time_tool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Returns current time",
    },
}

pickup_leaves_tool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "pickup_leaves",
        "description": "Picks up leaves",
    },
}

openai_tools = [move_item_tool, get_time_tool, pickup_leaves_tool]
