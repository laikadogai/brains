import json
from typing import List

from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from brains import args, prompts
from brains.control import collect_items
from brains.tools import get_time, move_item, openai_tools
from brains.utils import play_text

client = OpenAI()

msg_history: List[ChatCompletionMessageParam] = [
    {"role": "system", "content": prompts.system_base},
    {"role": "assistant", "content": prompts.assistant_first_msg},
]


async def submit_request(text: str):

    msg_history.append({"role": "user", "content": text})
    response = ""
    response_object = {
        "role": "assistant",
        "fn_calls": [],
    }

    # Loop to support function calls and subsequent generations by OpenAI
    for _ in range(args.openai_max_function_calls):

        # We store function calls in a slightly different format than the
        # one expected by OpenAI - convert it here. `response_object` is object
        # containing function calls within single user request
        openai_funcs: List[ChatCompletionMessageParam] = []
        for fn in response_object["fn_calls"]:
            openai_funcs.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": fn["name"],
                            "arguments": str(fn["arguments"]),
                        },
                    },
                    {
                        "role": "function",
                        "name": fn["name"],
                        "content": str(fn["results"]),
                    },
                ]
            )

        logger.info(
            "completions request:"
            + "\n"
            + f"\n{'@'*20}\n".join(
                [
                    f"{message['role']}: {message['content'] if 'content' in message else ''}"
                    for message in msg_history + openai_funcs
                ]
            )
        )

        stream = client.chat.completions.create(
            model=args.openai_completions_model,
            messages=msg_history + openai_funcs,
            tools=openai_tools,
            tool_choice="auto",
            stream=True,
            temperature=args.openai_temperature,
            max_tokens=args.openai_max_tokens,
        )

        tool_calls_recovered = {}

        for chunk in stream:
            if answer_piece := chunk.choices[0].delta.content:
                response += answer_piece
            if tool_calls := chunk.choices[0].delta.tool_calls:
                piece = tool_calls[0]
                if piece.index not in tool_calls_recovered:
                    tool_calls_recovered[piece.index] = {
                        "id": None,
                        "function": {
                            "arguments": "",
                            "name": "",
                        },
                        "type": "function",
                    }
                if piece.id:
                    tool_calls_recovered[piece.index]["id"] = piece.id
                if piece.function and piece.function.name:
                    tool_calls_recovered[piece.index]["function"]["name"] = piece.function.name
                if piece.function and piece.function.arguments:
                    tool_calls_recovered[piece.index]["function"]["arguments"] += piece.function.arguments

        # This branching determines whether we've received intent to execute
        # function(s) so we should process and execute them, or we received
        # message stream, which was already display so we should do nothing
        if tool_calls_recovered != {}:

            logger.debug(f"Received tool calls: {tool_calls_recovered}")

            # Now we have to figure out whether it is single function call
            # or multiple functions call, after which we transform function to
            # (function_name, function_args) pairs
            if tool_calls_recovered[0]["function"]["name"] == "multi_tool_use.parallel":
                tools = []
                for tool_raw in json.loads(tool_calls_recovered[0]["function"]["arguments"])["tool_uses"]:
                    tools.append((tool_raw["recipient_name"].replace("functions.", ""), tool_raw["parameters"]))
            else:
                tools = [
                    (tool_raw["function"]["name"], json.loads(tool_raw["function"]["arguments"]))
                    for tool_raw in tool_calls_recovered.values()
                ]

            logger.debug(tools)

            # The cycle is for the case of several tools at once
            for func_name, func_args in tools:

                # status = status_space.status(f"Executing `{func_name}`")

                response_object_item = {
                    "name": func_name,
                    "arguments": func_args,
                    "state": "running",
                    "results": "",
                }

                if func_name == "move_item":

                    item = func_args.get("item", "")
                    target_place = func_args.get("target_place", "")
                    source_place = func_args.get("source_place", "")

                    if not item:
                        response_object_item["state"] = "error"
                        response_object_item["results"] = f"Unable to move item: specify item to move"

                    if not target_place:
                        response_object_item["state"] = "error"
                        response_object_item["results"] = f"Unable to move item: specify target place"

                    intermediate_response = f"I will try to move the {item} to the {target_place}" + (
                        f"from the {source_place}." if source_place else "."
                    )
                    msg_history.append({"role": "assistant", "content": intermediate_response})
                    play_text(intermediate_response)

                    result = move_item(item=item, target_place=target_place, source_place=source_place)

                    response_object_item["state"] = "complete"
                    response_object_item["results"] = result
                elif func_name == "get_time":

                    result = get_time()
                    response_object_item["state"] = "complete"
                    response_object_item["results"] = result
                elif func_name == "pick_up_items":
                    items = func_args.get("items", "")
                    await collect_items(items)

                    response_object_item["state"] = "complete"
                    response_object_item["results"] = "Task is finished!"
                else:
                    logger.error(f"LLM tried to call unknown function: {func_name}")

                response_object["fn_calls"].append(response_object_item)
                # response_object["content"] = response_object["fn_calls"][-1]["results"]
                msg_history.append({"role": "assistant", "content": response_object["fn_calls"][-1]["results"]})
                play_text(response_object["fn_calls"][-1]["results"])

            # If last function call requires simple action execution
            # then print the result of it, save as a response and stop executing
            # to avoid generating next reply by LLM
            if response_object["fn_calls"][-1]["name"] in ("move_item", "get_time", "pickup_leaves"):
                # # response_object["content"] = response_object["fn_calls"][-1]["results"]
                # msg_history.append({"role": "assistant", "content": response_object["fn_calls"][-1]["results"]})
                # play_voice(response_object["fn_calls"][-1]["results"])
                break

        elif response:
            logger.info(response)
            # response_object["content"] = response
            msg_history.append({"role": "assistant", "content": response})
            play_text(response)
            break
        else:
            logger.error("Unexpected response from LLM")
