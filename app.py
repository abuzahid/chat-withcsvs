import  streamlit as st
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# openai.api_key = st.secrets["OPENAI_API_KEY"]
model = st.secrets["OPENAI_MODEL"]
llm=ChatOpenAI(model=model,temperature=0)

st.title("CSV CHAT APP")
st.write("upload your csv & and ask anything")
file=st.file_uploader("select your file",type=["csv"])
if file is not None:
    df=pd.read_csv(file)
    #st.write(df)
    input=st.text_area("ask your question here")
    if input is not None:
        button=st.button("Submit")
        agent=create_pandas_dataframe_agent(
        llm,df,verbose=False,agent_type=AgentType.OPENAI_FUNCTIONS,allow_dangerous_code=True
        )
        if button:
            result=agent.invoke(input)
            st.write(result["output"])