import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool 
import requests
from langchain_core.messages import HumanMessage , AIMessage
load_dotenv()

# Page Config
st.set_page_config(
    page_icon="🧳",
    page_title="AI Travel Planner",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# Sidebar
with st.sidebar:
    st.title("🧳 Trip Settings")
    st.markdown("Customize your preferences or track system configurations here.")
    st.divider()

    st.subheader("System Status")
    groq_ok = "GROQ Ok" if os.getenv("GROQ_API_KEY") else "❌ GROQ Missing"
    tavily_ok = "Tavily Ok" if os.getenv("TAVILY_API_KEY") else "⚠ Tavily Missing"
    weather_ok = "Weather Fetched" if os.getenv("WEATHER_API_KEY") else "⚠ Weather Missing"

    st.caption(groq_ok)
    st.caption(tavily_ok)
    st.caption(weather_ok)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Weather Tool
@tool
def get_live_weather(city : str) -> str:
    """ 
    Useful to fetch the live weather or climate information for a specific city.
    Use this when planning clothing or checking trip conditions.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url , timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            humidity = data['main']['humidity']
            return f"The Current Weather in {city} is {temp}°C with {desc} . Humidity is {humidity}"
        elif response.status_code == 404:
            st.error(f"Error : City {city} Not Found. Please Check Spelling.")
            return None
        else:
            st.error(f"Weather API Error Code : {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to weather service : {str(e)}")    
        return None

# System Agent
@st.cache_resource
def init_agent():
    """ Cache the agent creation so it does not reload on every single button press / user input """
    if not os.getenv("GROQ_API_KEY"):
            st.error("The GROQ Api Key is missing from your dotenv file")
    if not os.getenv("TAVILY_API_KEY"):
        st.warning("The Tavily Api Key is missing from your dotenv file") 

    try:
        llm = ChatGroq(temperature=0.2 , model="openai/gpt-oss-120b")
        search_tools = TavilySearchResults(max_results = 3)
        tools = [search_tools , get_live_weather]
        return create_react_agent(llm , tools)
    except Exception as e:
        st.error(f"Error Initializing LLM : {str(e)}")
        return None
    
agent_executer = init_agent()

# Main Interactive Interface
st.title("🗺️ AI Autonomous Travel Planner")
st.markdown(
    "Discover tailored itineraries, travel spots, and live updates based on your "
    "unique **preferences, duration, group size, and budget**."
)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [
    AIMessage(content="Hello! I am your personal travel assistant. ✈️\n\nTo build your perfect itinerary, please share your details like:\n* **Destination/Preferences**\n* **Duration**\n* **Group Size**\n* **Budget**")
    ]

for message in st.session_state.messages:
    if isinstance(message , dict):
        role = "user" if message.get("role") == "user" else "assistant"
        content = message.get("content","")
    else:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content

    with st.chat_message(role):
        st.markdown(content)

if user_query := st.chat_input("Tell me about your dream vacation:"):
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message('assistant'):
        status_placeholder = st.empty()

        with status_placeholder.status("🧠 Consulting travel networks & checking weather..." , expanded=True) as status:
            if agent_executer is None:
                status.update(label="❌ Configuration Error " , state="error")
                st.error("Agent could not execute because the LLM engine failed to initialize.")
            else:
                try:
                    response = agent_executer.invoke({"messages":st.session_state.messages})
                    last_msg = response['messages'][-1]

                    if isinstance(last_msg , dict):
                        final_answer = last_msg.get("content","")
                    else:
                        final_answer = last_msg.content

                    status.update(label="✨ Tailored Itinerary Generated", state="complete", expanded=True)

                    st.markdown(final_answer)
                    st.session_state.messages.append(AIMessage(content=final_answer))
                except Exception as e:
                    status.update(label="❌ Search Interupted ", state="error")
                    st.error(f"Failed to execute agent loop : {str(e)}")