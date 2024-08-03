from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent


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

    # tools = [PythonREPLTool()]

    # agent = create_react_agent(
    #     prompt=prompt,
    #     llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    #     tools=tools,
    # )
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # agent_executor.invoke(
    #     input={
    #         "input": """generate and save 5 QRcodes in a new directory named qr codes in current working directory 
    #                             that point to https://www.youtube.com/watch?v=TWf3r9NXz7k, you have qrcode package installed already"""
    #     }
    # )


    # csv agent uses pandas under the hood, also based on ReAct algo
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    csv_agent.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )
    # csv_agent.invoke(
    #     input={"input": "in the file episode_info.csv, Give me name of the writer wrote the most episodes? how many episodes did he write?"}
    # )
    # csv_agent.invoke(
    #     input={
    #         "input": "print the seasons by ascending order of the number of episodes they have"
    #     }
    # )


if __name__ == "__main__":
    main()
