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

#brief = run_research_agent("NVIDIA GTC 2026")
#display(Markdown(brief))

JUDGE_PROMPT = """ 
You are scoring a research agent **output/brief** to determine if there was an
**insufficient source breadth failure**.
Return **only** `TRUE` or `FALSE`.

## Definitions

**Source:** Any distinct reference used to support claims in the brief. A source
can be a URL, a named publication, a named report, a named dataset, a named
person/interview, or a clearly identifiable document. Two references to the same
publication count as **one** source.

**Insufficient source breadth failure (label TRUE):** The research brief does
**not** reference at least 5 distinct sources. Any of the below applies:

1. **Fewer than 5 sources total:** The brief cites, links, names, or otherwise
   references fewer than 5 distinct sources anywhere in the document (inline,
   footnotes, appendix, or bibliography).
2. **Duplicate sources inflating count:** The same source is referenced multiple
   times under different names/URLs but resolves to the same origin — and once
   deduplicated, the true distinct count falls below 5.
3. **Vague or unattributed references that cannot be counted:** Phrases like
   "according to industry reports," "some studies suggest," or "experts say"
   without a traceable, nameable source do **not** count toward the 5-source
   threshold.
4. **Self-referential sources:** The agent citing its own prior output, its
   system prompt, or the user's original request as a "source" does not count.

**No failure (label FALSE):** The brief references at least 5 distinct,
identifiable sources. Sources do not need to be formally cited — a URL, a
publication name, a report title, or a named dataset each qualify as long as
they are traceable and distinct.

## Output Format

Return exactly one token: `TRUE` or `FALSE`. No explanations.
"""

TOPICS = [
    "Nvidia AI news and announcements in March 2026",
    "X86 vs ARM news and annoucements in March 2026",
    "Enterprise AI Adoption in March 2026",
    #"Geopolitical risks in rare earth mineral supply chains",
    #"AI regulation frameworks across the US, EU, and China",
    #"Carbon capture technology economics and scalability",
    #"The rise of agentic AI in enterprise software",
    #"Neuroplasticity research and mental health therapies",
    #"Autonomous vehicle liability and insurance frameworks",
    #"Central bank digital currencies (CBDCs) adoption trends",
]

# LLM as Judge

import time, random
import contextlib, io

JUDGE_MODEL = "gpt-4.1"

print(f"Running {len(TOPICS)} topics...", flush=True)

def judge(brief, retries=3):
    for i in range(retries):
        try:
            r = client.responses.create(
                model= JUDGE_MODEL,
                input=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": brief[:120000]},
                ],
            )
            return r.output_text.strip()
        except Exception:
            if i == retries - 1:
                return "ERROR"
            time.sleep((2 ** i) + random.random())

results = []
for i, topic in enumerate(TOPICS, 1):
    #with contextlib.redirect_stdout(io.StringIO()): # Suppress print outputs from agent for clearer evaluation logs
    brief = run_research_agent(topic, max_iterations=12)
    results.append({
        "topic": topic,
        "label": judge(brief),
        "chars": len(brief),
        "brief": brief,
    })
    print(f"\n=== {i}. {topic} ===\n{brief[:1000]}{' ...' if len(brief) > 1000 else ''}\n", flush=True)
    time.sleep(1.2 + random.random() * 0.8)

print(f"Collected {len(results)} results", flush=True)
print(f'Passed {sum(r["label"] == "FALSE" for r in results)}/{len(results)} tests', flush=True)

### Alternate LLM as judge with parallel execution, while accounting for Rate limits

import time
import random
import contextlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

JUDGE_MODEL = "gpt-4.1"
MAX_WORKERS = min(2, len(TOPICS))  # tune this based on your API rate limits

print(f"Running {len(TOPICS)} topics with {MAX_WORKERS} workers...", flush=True)


def judge(brief, retries=3):
    for i in range(retries):
        try:
            r = client.responses.create(
                model=JUDGE_MODEL,
                input=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": brief[:120000]},
                ],
            )
            return r.output_text.strip()
        except Exception:
            if i == retries - 1:
                return "ERROR"
            time.sleep((2 ** i) + random.random())


def process_topic(idx, topic):
    #with contextlib.redirect_stdout(io.StringIO()):
    brief = run_research_agent(topic, max_iterations=10)

    label = judge(brief)

    return {
        "idx": idx,
        "topic": topic,
        "label": label,
        "chars": len(brief),
        "brief": brief,
    }


results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_topic = {
        executor.submit(process_topic, i, topic): (i, topic)
        for i, topic in enumerate(TOPICS, 1)
    }

    for future in as_completed(future_to_topic):
        i, topic = future_to_topic[future]
        try:
            result = future.result()
        except Exception as e:
            result = {
                "idx": i,
                "topic": topic,
                "label": "ERROR",
                "chars": 0,
                "brief": f"ERROR: {e}",
            }

        results.append(result)

        brief = result["brief"]
        print(
            f"\n=== {result['idx']}. {result['topic']} ===\n"
            f"{brief[:1000]}{' ...' if len(brief) > 1000 else ''}\n",
            flush=True,
        )

# Optional: restore original topic order
results.sort(key=lambda r: r["idx"])

print(f"Collected {len(results)} results", flush=True)
print(
    f'Passed {sum(r["label"] == "FALSE" for r in results)}/{len(results)} tests',
    flush=True,
)

