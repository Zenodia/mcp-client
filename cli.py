"""
This module contains the Cli client for the MCP servers.
"""
import asyncio
import os
import sys
import traceback
from datetime import datetime
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain_core.runnables.base import RunnableBinding
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.graph import CompiledGraph
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from colorama import Fore
from dotenv import load_dotenv
from mcp_client.base import (
    load_server_config,
    create_server_parameters,
    convert_mcp_to_langchain_tools,
    create_agent_executor
)

load_dotenv()
nvapi_key=os.environ["NVIDIA_API_KEY"]
print("valid NVIDIA API Key : ",nvapi_key[-4:] )
llm=ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

## structural output using LMFE 
class ToolFormat(BaseModel):     
    tool_name: str = Field(description="Name of the tool that will be called")
    tool_input : str = Field(description="extract appropriate input to the tool")

llm_with_tool_structure=llm.with_structured_output(ToolFormat)     


## construct the system prompt 
prompt_template = """
### [INST]

You are an expert in extracting relevant information in order to use a tool.
Your task is to extract the name of the tool as well as appropriate tool_input from the user input query.
------
{input_query}
------
The output MUST always follow the below format :

'''
tool_name: the name of the tool that is going to be exeucted
tool_input: extract appropriate information that we can send to the tool for processing
'''
Example:
input_query : what is 2 times 10.2

output:
tool_name: calculate
tool_input: 2*10.2

Begin!

[/INST]
 """
prompt = PromptTemplate(
input_variables=['input_query'],
template=prompt_template,
)

extract_tool_inputs_chain = ( prompt | llm_with_tool_structure)

async def output_to_invoke_tools(out, tool):
    tool_name=out.tool_name
    tool_input=out.tool_input
    print(Fore.GREEN + " Using tool =", tool_name,"with tool_input=" ,tool_input)
    output=await tool.ainvoke({"expression":tool_input})
    tool_output=output[0].text
    return tool_output


async def list_tools() -> None:
    """List available tools from the server."""
    server_config = load_server_config()
    server_params = create_server_parameters(server_config)
    langchain_tools = await convert_mcp_to_langchain_tools(server_params)

    for tool in langchain_tools:
        print(f"{tool.name}")




## construct the content_creator agent

async def simple_qa(user_message: str)-> str:
    server_config = load_server_config()
    server_params = create_server_parameters(server_config)
    langchain_tools = await convert_mcp_to_langchain_tools(server_params)

    for tool in langchain_tools:
        print(f"{tool.name}")
    #llm_with_calculator_tool=llm.bind_tools([langchain_tools[0]],tool_choice=langchain_tools[0].name)
    #output=await llm_with_calculator_tool.ainvoke(user_message)
    output=extract_tool_inputs_chain.invoke({"input_query":user_message})
    #print(Fore.YELLOW + " simple_qa output: ", type(output), output , Fore.RESET)
    tool_output = await output_to_invoke_tools(output, langchain_tools[0])
    print(Fore.YELLOW + "Response from LangChain Agent output: ", type(tool_output), tool_output , Fore.RESET)
    return tool_output
    

async def handle_chat_mode():
    """Handle chat mode for the LangChain agent."""
    print("\nInitializing chat mode...")
    agent_executor_cli = await create_agent_executor("cli")

    print("\nInitialized chat mode...")

    # Maintain a chat history of messages
    chat_history = []

    # Start the chat loop
    while True:
        try:
            user_message = input("\nYou: ").strip()
            if user_message.lower() in ["exit", "quit"]:
                print("Exiting chat mode.")
                break
            if user_message.lower() in ["clear", "cls"]:
                os.system("cls" if sys.platform == "win32" else "clear")
                chat_history = []
                continue
            all_messages = []
            # Append the chat history to all messages
            all_messages.extend(chat_history)
            all_messages = [HumanMessage(content=user_message)]
            input_messages = {
                "messages": all_messages,
                "today_datetime": datetime.now().isoformat(),
            }
            print(Fore.YELLOW + " input msg to the agent_executor :\n", input_messages)
            # Query the assistant and get a fully formed response
            #assistant_response = await query_response(input_messages, agent_executor_cli)
            assistant_response = await simple_qa(user_message)

            # Append the assistant's response to the history
            chat_history.append(AIMessage(content=assistant_response))
        except Exception as e:
            error_trace = traceback.format_exc()
            print(error_trace)
            print(f"\nError processing message: {e}")
            continue



async def query_response(input_messages: TypedDict, agent_executor: CompiledGraph) -> str:
    """Query the assistant and get a fully formed response."""
    
    output=routing_chain.invoke(input_messages)
    print(output)
    
    collected_response = []

    async for chunk in agent_executor.astream(
            input_messages,
            stream_mode=["messages", "values"]
    ):
        # Process the chunk and append the response to the collected response
        print(Fore.MAGENTA + "chunk ", type(chunk), chunk)
        process_chunk(chunk)
        if isinstance(chunk, dict) and "messages" in chunk:
            print(Fore.LIGHTGREEN_EX + " processing chunk of message from the agent executor :", chunk["messages"][-1].content)
            collected_response.append(chunk["messages"][-1].content)

    print("")  # Ensure a newline after the conversation ends
    print(Fore.GREEN + "collected_response :\n", collected_response )
    return "".join(collected_response)


def process_chunk(chunk):
    """Process the chunk and print the response."""
    if isinstance(chunk, tuple) and chunk[0] == "messages":
        process_message_chunk(chunk[1][0])
    elif isinstance(chunk, dict) and "messages" in chunk:
        process_final_value_chunk()
    elif isinstance(chunk, tuple) and chunk[0] == "values":
        process_tool_calls(chunk[1]['messages'][-1])


def process_message_chunk(message_chunk):
    """Process the message chunk and print the content."""
    if isinstance(message_chunk, AIMessageChunk):
        content = message_chunk.content  # Get the content of the message chunk
        if isinstance(content, list):
            extracted_text = ''.join(item['text'] for item in content if 'text' in item)
            print(extracted_text, end="", flush=True)  # Print message content incrementally
        else:
            print(content, end="", flush=True)


def process_final_value_chunk():
    """Process the final value chunk and print the content."""
    print("\n", flush=True)  # Ensure a newline after complete message


def process_tool_calls(message):
    """Process the tool calls and print the results."""
    print(Fore.RED + "processing tool calls :\n", type(message), message, Fore.RESET)
    if isinstance(message, AIMessage) and message.tool_calls:
        message.pretty_print()  # Format and print tool call results


async def interactive_mode():
    """Run the CLI in interactive mode."""
    print("\nWelcome to the Interactive MCP Command-Line Tool")
    print("Type 'help' for available commands or 'chat' to start chat or 'quit' to exit")

    while True:
        try:
            command = input(">>> ").strip()  # Get user input
            if not command:
                continue
            should_continue = await handle_command(command)  # Handle the command
            if not should_continue:
                return
        except KeyboardInterrupt:
            print("\nUse 'quit' or 'exit' to close the program")
        except EOFError:
            break
        except Exception as e:
            print(f"\nError: {e}")


async def handle_command(command: str):
    """ Handle specific commands dynamically."""
    try:
        if command == "list-tools":
            print("\nFetching Tools List...\n")
            # Implement list-tools logic here
            await list_tools()
        elif command == "chat":
            print("\nEntering chat mode...")
            await handle_chat_mode()
            # Implement chat mode logic here
        elif command in ["quit", "exit"]:
            print("\nGoodbye!")
            return False
        elif command == "clear":
            if sys.platform == "win32":
                os.system("cls")
            else:
                os.system("clear")
        elif command == "help":
            print("\nAvailable commands:")
            print("  list-tools    - Display available tools")
            print("  chat          - Enter chat mode")
            print("  clear         - Clear the screen")
            print("  help          - Show this help message")
            print("  quit/exit     - Exit the program")
        else:
            print(f"\nUnknown command: {command}")
            print("Type 'help' for available commands")
    except Exception as e:
        print(f"\nError executing command: {e}")

    return True


def main() -> None:
    """ Entry point for the script."""


asyncio.run(interactive_mode())  # Run the main asynchronous function

if __name__ == "__main__":
    main()  # Execute the main function when script is run directly
