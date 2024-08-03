"""
python agent to genearte qrc odees for favourite music video
"""
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool


load_dotenv()

def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question. 
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    # python repl executes python code in an interpretor(not recomended)
    # uses our virtual env and its packages
    # don't use in production
    # as PythonREPL in an inbuilt tool we dont have to pass toolname, description and other paramteres
    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke(
        input={
            "input": """generate and save 5 QRcodes in a new directory named qr codes in current working directory 
                                that point to https://www.youtube.com/watch?v=TWf3r9NXz7k, you have qrcode package installed already"""
        }
    )


if __name__ == "__main__":
    main()