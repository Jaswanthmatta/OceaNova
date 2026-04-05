@@ -0,0 +1,151 @@
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, END, START, StateGraph
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal


async def get_response(user_query):
    """
    Method to return the response using the streaming chain
    :param user_query:
    :param conversation_history:
    :return:
    """

    llm = ChatOllama(model='qwen3:4b')

    client = MultiServerMCPClient({
        "query": {
            "command": "python",
            "args": ["./server.py"],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    system_message = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        If tool_calll resultl is None answer should be None.

        Then you should query the schema of the most relevant tables.
        Table argo_profiles
            PLATFORM_NUMBER (UInt32)
            CYCLE_NUMBER (UInt32)
            MAX_LEVELS (UInt32)
            DIRECTION (FixedString(1))
            DATA_CENTRE (LowCardinality(String))
            JULD (DateTime)
            JULD_LOCATION (DateTime)
            LATITUDE (Float64)
            LONGITUDE (Float64)
            JULD_QC (FixedString(1))
            POSITION_QC (FixedString(1))
            PROFILE_PRES_QC (FixedString(1))
            PROFILE_TEMP_QC (FixedString(1))
            PROFILE_PSAL_QC (FixedString(1))
        Table argo_readings
            PLATFORM_NUMBER (UInt32)
            CYCLE_NUMBER (UInt32)
            N_LEVELS (UInt32)
            PRES (Nullable(Float32))
            TEMP (Nullable(Float32))
            PSAL (Nullable(Float32))
            PRES_ADJUSTED (Nullable(Float32))
            TEMP_ADJUSTED (Nullable(Float32))
            PSAL_ADJUSTED (Nullable(Float32))
            PRES_ADJUSTED_QC (FixedString(1))
            TEMP_ADJUSTED_QC (FixedString(1))
            PSAL_ADJUSTED_QC (FixedString(1))
            PRES_ADJUSTED_ERROR (Nullable(Float32))
            TEMP_ADJUSTED_ERROR (Nullable(Float32))
            PSAL_ADJUSTED_ERROR (Nullable(Float32))
        Table argo_calibrations
            PLATFORM_NUMBER (UInt32)
            CYCLE_NUMBER (UInt32)
            PARAMETER (String)
            SCIENTIFIC_CALIB_EQUATION (String)
            SCIENTIFIC_CALIB_COEFFICIENT (String)
            SCIENTIFIC_CALIB_COMMENT (String)
            SCIENTIFIC_CALIB_DATE (Nullable(DateTime))
        """.format(
        dialect="SQLite",
        top_k=5,
    )

    async def llm_call(state: MessagesState):
        return {
            "messages": [
                await llm_with_tools.ainvoke(
                    [
                        SystemMessage(
                            content=system_message
                        )
                    ]
                    + state["messages"]
                )
            ]
        }

    tool_node = ToolNode(tools)

    async def should_continue(state: MessagesState) -> Literal["Action", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "Action"
        return END

    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("environment", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "Action": "environment",
            END: END
        }
    )
    agent_builder.add_edge("environment", "llm_call")

    agent = agent_builder.compile()


    # note: use .invoke() method for non-streaming
    msg = HumanMessage(content=user_query)
    async for value in agent.astream(
        {
            "messages": msg
        }, stream_mode="messages"
    ):
        yield value[0].content