import streamlit as st
from crewai.flow import start, listen, Flow
from crewai import LLM, Agent, Task, Crew, Process
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()

# ---------------------------
# State
# ---------------------------
class FlowState(BaseModel):
    topic: str = ""
    result: str = ""


# ---------------------------
# LLM
# ---------------------------
ollama_llm = LLM(
    model=os.getenv("MODEL"),
    temperature=0
)


# ---------------------------
# Agents
# ---------------------------
search_agent = Agent(
    role="General Search Agent",
    goal="Write simple explanations",
    backstory="Expert in giving explanations",
    llm=ollama_llm,
    verbose=True,
    allow_delegation=True
)

summary_agent = Agent(
    role="Summarizer",
    goal="To Summarize the given data",
    backstory="Expert in Summarizing",
    llm=ollama_llm,
    verbose=True,
    allow_delegation=True
)


# ---------------------------
# Flow
# ---------------------------
class PracticeFlow(Flow[FlowState]):

    def __init__(self, topic_input):
        super().__init__()
        self.input_topic = topic_input

    @start()
    def generate_topic(self):
        self.state.topic = self.input_topic

    @listen(generate_topic)
    def search_topic(self):

        search_task = Task(
            description=f"Give detailed explanation on the topic {self.state.topic}",
            agent=search_agent,
            expected_output="Detailed explanation"
        )

        summary_task = Task(
            description="Summarize the output in less than 50 words",
            agent=summary_agent,
            expected_output="Short Summary"
        )

        crew = Crew(
            agents=[search_agent, summary_agent],
            tasks=[search_task, summary_task],
            verbose=True,
            process=Process.hierarchical,
            manager_llm=ollama_llm
        )

        result = crew.kickoff()
        self.state.result = result.raw

    @listen(search_topic)
    def show_output(self):
        return self.state.result


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🤖 CrewAI Agent App")

# Input from user
topic = st.text_input("Enter your topic:")

# Button click
if st.button("Run Agent"):

    if topic.strip() == "":
        st.warning("Please enter a topic")
    else:
        with st.spinner("Running agents..."):

            flow = PracticeFlow(topic)
            output = flow.kickoff()

            st.success("Done ✅")
            st.subheader("Output:")
            st.write(output)