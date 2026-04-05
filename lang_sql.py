@@ -0,0 +1,217 @@
import io

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, END, START, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from typing_extensions import Literal, TypedDict
from IPython.display import Image, display
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
import asyncio

llm = ChatOllama(
    model = "qwen3:4b",
    reasoning= False,
    temperature=0

)

async def main():

    client = MultiServerMCPClient({
            "query": {
                "command": "python",
                "args": ["./server.py"],
                "transport": "stdio",
            }
    })

    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    # sys_prompt = """Give insights on data by accessing the data from database writing the SQL query
    # JOIN tables on PLATFORM_NUMBER and CYCLE_NUMBER
    # With tables containing columns of types:
    # Table argo_profiles
    #     PLATFORM_NUMBER (UInt32)
    #     CYCLE_NUMBER (UInt32)
    #     MAX_LEVELS (UInt32)
    #     DIRECTION (FixedString(1))
    #     DATA_CENTRE (LowCardinality(String))
    #     JULD (DateTime)
    #     JULD_LOCATION (DateTime)
    #     LATITUDE (Float64)
    #     LONGITUDE (Float64)
    #     JULD_QC (FixedString(1))
    #     POSITION_QC (FixedString(1))
    #     PROFILE_PRES_QC (FixedString(1))
    #     PROFILE_TEMP_QC (FixedString(1))
    #     PROFILE_PSAL_QC (FixedString(1))
    # Table argo_readings
    #     PLATFORM_NUMBER (UInt32)
    #     CYCLE_NUMBER (UInt32)
    #     N_LEVELS (UInt32)
    #     PRES (Nullable(Float32))
    #     TEMP (Nullable(Float32))
    #     PSAL (Nullable(Float32))
    #     PRES_ADJUSTED (Nullable(Float32))
    #     TEMP_ADJUSTED (Nullable(Float32))
    #     PSAL_ADJUSTED (Nullable(Float32))
    #     PRES_ADJUSTED_QC (FixedString(1))
    #     TEMP_ADJUSTED_QC (FixedString(1))
    #     PSAL_ADJUSTED_QC (FixedString(1))
    #     PRES_ADJUSTED_ERROR (Nullable(Float32))
    #     TEMP_ADJUSTED_ERROR (Nullable(Float32))
    #     PSAL_ADJUSTED_ERROR (Nullable(Float32))
    # Table argo_calibrations
    #     PLATFORM_NUMBER (UInt32)
    #     CYCLE_NUMBER (UInt32)
    #     PARAMETER (String)
    #     SCIENTIFIC_CALIB_EQUATION (String)
    #     SCIENTIFIC_CALIB_COEFFICIENT (String)
    #     SCIENTIFIC_CALIB_COMMENT (String)
    #     SCIENTIFIC_CALIB_DATE (Nullable(DateTime))"""

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

    # display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
    # img = agent.get_graph(xray=True).draw_mermaid_png()
    # img = Image.open(io.BytesIO(img))
    # plt.imshow(img)
    # plt.axis("off")   # Hide axes
    # plt.show()

    messages = [HumanMessage(content="What is the avg salinity")]
    # print(messages[0])

    # Stream the agent
    final_state = None
    # async for event in agent.astream({"messages": "Show me salinity profiles near the equator in March 2023"}, stream_mode="values"):
    #     if "messages" in event:
    #         # Each event has messages – take the new one
    #         msg = event["messages"][-1]
    #         # Print the new text incrementally
    #         print(msg.content, end="", flush=True)

    chunks = []
    async for chunk in agent.astream({"messages": "Total avg temp and salinity"}, stream_mode="messages"):
        print(chunk[0].content, end="", flush=True)

    # # At the end, `final_state` will contain the same info as `agent.invoke`
    # print("\nFinal Messages:")
    # for m in final_state["messages"]:
    #     print(m)


if __name__ == "__main__":
    asyncio.run(main())