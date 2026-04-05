@@ -0,0 +1,23 @@
from mcp.server.fastmcp import FastMCP
import clickhouse_connect

mcp = FastMCP()

client = clickhouse_connect.get_client(
    host="localhost",
    username="default",
    password="",
    database="default"
)

@mcp.tool()
async def run_query(query) -> list:
    """
    Sends a query to the server and returns a list of elements
    """
    print("Running query:", query)
    result = client.query(query)
    return list(result.result_rows)

if __name__ == "__main__":
    mcp.run(transport='stdio')