#!pip install ddgs trafilatura
#!pip install lxml_html_clean

import os
import json
from openai import OpenAI
from dotenv import load_dotenv 
from pprint import pprint
from IPython.display import Markdown, display

from ddgs import DDGS
import trafilatura

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API Key is missing")

client = OpenAI()

Model = "gpt-4.1-mini"

def search_web(query:str): #def search_web(query:str, num_results:int):
    """Search the web using the DuckDuckGo browser, return results"""
    ddgs = DDGS()
    results = ddgs.text(query, max_results = 3) #results = ddgs.text(query, max_results = num_results)
    print(f"  \u2705 Got results")
    return json.dumps(results, indent=2)

#search_web("News for Qualcomm AI from the past week", 3)

def fetch_url(url:str):
    """Fetch the content of URL using Trafilatura tool"""
    #url = "https://finance.yahoo.com/news/qualcomm-weighs-ai-export-rules-150751298.html"
    download_text= trafilatura.fetch_url(url)
    #print(download_text)

    if download_text:
        extract_text = trafilatura.extract(download_text)
        if extract_text:
            print(f". \u2705 Got text:{len(extract_text)} chars")
            return extract_text
    print(f". \u274c Failed to fetch or extract text from {url}")
    return (f"Could not extract text from {url}. Try a different source")

#result = fetch_url("https://finance.yahoo.com/news/qualcomm-weighs-ai-export-rules-150751298.html")
#print(result)

tools=[]

search_web_function = {
    "name":"search_web",
    "description":"Search the web using the DuckDuckGo browser, return results",
    "parameters": {
        "type":"object",
        "properties":{
            "query": {
                "type":"string",
                "description":"The search query to find relevant webpages"
            },
            #"num_results": {
             #   "type":"int",
              #  "description":"The number of search results to return"
            #}
        },
         "required":["query"] #"required":["query","num_results"]
    }

}

tools.append({"type":"function", "function":search_web_function})

fetch_url_function = {
    "name":"fetch_url",
    "description":"Fetch and extract the text content from a web page given the url",
    "parameters": {
        "type":"object",
        "properties":{
            "url":{
                "type":"string",
                "description":"The url from the search results to fetch and extract text from"
            }
        },
         "required":["url"]
    }

}

tools.append({"type":"function", "function":fetch_url_function})

def handle_tool_calls(tool_calls):
    tool_results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f". \U0001f527 Calling function {function_name} with arguments: {args}")

        if function_name == "search_web":
            result = search_web(args["query"]) #result = search_web(args["query","num_results"])
            content = f"Search results:{result}"
            #print(f"Sent Notification:{args['message']}")
        elif function_name == "fetch_url":
            result = fetch_url(args["url"])
            content = f"Fetched web content:{result}"
        #elif function_name == "function_name_2":
            #content = function_name_2(args["message"])
        #...
        else:
            content = f"Unknown function:{function_name}"

        tool_call_result = {
            "role":"tool",
            "content":str(content),
            "tool_call_id":tool_call.id
        }

        tool_results.append(tool_call_result)

    return tool_results

RESEARCH_AGENT_SYSTEM_PROMPT = """
You are a research specialist that follows all instructions excels at searching the web for news on a given topic and summarizing your findings into a 
comprehensive research brief.

You MUST gather information from at least 6 distinct sources before delivering your brief. 
If you have fewer than 6 sources, keep searching. You need to pause and reflect on the results after you get the information 
from 3 sources so that you can identify the best next 3 sources to retrieve.

VERY IMPORTANT: ALWAYS look for recent news when possible. DO NOT use older news articles if newer ones from reputable sources exist.
Prioritize sources such as press releases or articles directly from the company and organization that is being searched if relevant. 
Also prioritize sources from reputable news sources. 

You have access to (1) a search_web tool that you can use to search the web for information and 
(2) a fetch_url tool to extract and read the text from the URLs of the webpages in the search results.

Your typical process:
1. Search for the topic to find the best sources to fetch
2. Reflect on the search results — which sources look most relevant and why?
3. Fetch the full content of the best URLs based on the max number of results requested
4. Reflect on what you have gathered. Do you have enough? Are there gaps? Are you unable to access or fetch text from any of the URLs?
5. If there are gaps, search again with a different query that captures the intent of the user
6. When you have enough information from at least 3 different sources, synthesize into a research brief

When you are ready to deliver your final research brief, start your response with "DONE:" followed by the brief itself.
The word "DONE:" is a control signal, not a label. Never use it as a heading, section marker, or inline annotation. 
ONLY use the word "DONE:" as per the instructions above -- it has to come at the start of a reply.

Your research brief MUST include:
- Main themes, headlines, and important text from the sources
- Key facts, data points, and statistics
- Summary
- Source URLs for attribution and dates of the content if from news sources

Until you are ready, just keep working — search, fetch, think, reflect.
Do not rush. Take time to reflect between tool calls before deciding your next step.
Not every response needs a tool call — sometimes just thinking through what you have is the right move. 

"""

def run_research_agent(topic:str, max_iterations:int = 10):
    """ Returns research brief on a topic as a string"""
    print(f"\n\U0001F50D Researching {topic}:")
    print("-"*50)

    messages = [
        {"role":"system", "content":RESEARCH_AGENT_SYSTEM_PROMPT},
        {"role":"user", "content":f"Research the following topic {topic} to produce a comprehensive brief"}
    ]

    iteration = 0
    while iteration < max_iterations:
        iteration+=1
        print(f"\n -- Iteration {iteration} --")

        response = client.chat.completions.create(
            model = Model,
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            tool_results = handle_tool_calls(message.tool_calls)
            messages.extend(tool_results)
        else:
            content = message.content
            if content.startswith("DONE:"):
                research_brief = content[len("DONE:"):].strip()
                print(f"\n\u2705 Research Complete!")
                return research_brief
            else:
                print(f". \U0001F4AD Agent is thinking:")
                pprint(content)
        
        if (iteration == max_iterations-1):
            print(". \u26a0 Stopping research as maximum number of iterations is about to be reached")
            messages.append({"role":"user", "content":"You have reached the maximum number of iterations. Please deliver your research brief now. You must respond with DONE: followed by your brief."})

    return "Research incomplete. Unable to deliver research brief on your topic as maximum iterations were reached"

brief = run_research_agent("NVIDIA GTC 2026")
display(Markdown(brief))
