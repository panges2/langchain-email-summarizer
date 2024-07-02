from dotenv import load_dotenv
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.agent_toolkits import GmailToolkit
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from langchain_community.tools.gmail.utils import build_resource_service
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import tool
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# Load environment variables from .env file
load_dotenv()

# Retrieve and print the API key
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
langsmith_api_key = os.getenv('LANGCHAIN_API_KEY')

print("OPENAI API Key:", openai_api_key)
print("SERPAPI API Key:", serpapi_api_key)  
print("langsmith_api_key:", langsmith_api_key)

os.environ["LANGCHAIN_TRACING_V2"] = "true"

SCOPES = ['https://mail.google.com/']

def get_gmail_credentials(token_file, client_secrets_file, scopes):
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes)
            creds = flow.run_local_server(port=8080, host='127.0.0.1')  # Ensure this port and host match the authorized redirect URI
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    return creds

credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=SCOPES,
    client_secrets_file="credentials.json",  # Ensure this matches the location of your file
)

print("Credentials obtained successfully")

# Build the Gmail API service
api_resource = build_resource_service(credentials=credentials)

# Initialize the GmailToolkit with the Gmail API service
toolkit = GmailToolkit(api_resource=api_resource)

# Get tools from the toolkit

tools = toolkit.get_tools()

# Function to clean and condense email content using BeautifulSoup
@tool
def clean_and_condense_email_content(email_content, max_tokens=2048) ->str:
    
    """
    Clean and condense email content using BeautifulSoup.
    
    This function takes an email content string and a maximum token limit.
    It cleans the content by stripping out HTML tags and condenses the text
    if it exceeds the specified token limit.
    
    Args:
        email_content (str): The email content to be cleaned and condensed.
        max_tokens (int): The maximum number of tokens allowed. Default is 2048.
    
    Returns:
        str: The cleaned and condensed email content.
    """

    soup = BeautifulSoup(email_content, 'html.parser')
    cleaned_text = soup.get_text()
    # Tokenize the cleaned text and condense if necessary
    tokens = cleaned_text.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens]) + '... [content truncated]'
    return cleaned_text

tools.append(clean_and_condense_email_content)

print("\n------\n tools: ", tools, "\n------\n")

tool_executor = ToolExecutor(tools)

model = ChatOpenAI(model_name="gpt-4o", temperature=0)

model = model.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
def call_tool(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        tool_invocations.append(action)

    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    tool_messages = [
        ToolMessage(
            content=str(response),
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for tc, response in zip(last_message.tool_calls, responses)
    ]

    # Clean and condense email content
    for message in tool_messages:
        if "email_content" in message.content:
            email_content = message.content.get("email_content", "")
            message.content["email_content"] = clean_and_condense_email_content(email_content)

    return {"messages": tool_messages}

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

# Finally, we compile it!
app = workflow.compile()

inputs = {"messages": [HumanMessage(content="\
                                    search my gmail 50 emails at a time for the last 3 days\
                                    if the email search comes in broken skip the search and move on\
                                    use beautiful soup to clean everything\
                                    figure out which 10 emails are the most important from the last 3 days. \
                                    Specifically find emails that are not promotions or mnewsletters\
                                    summarize those 10 emails for me")]}
for output in app.stream(inputs, stream_mode="values"):
    messages = output["messages"]
    for message in messages:
        message.pretty_print()
    print("\n---\n")