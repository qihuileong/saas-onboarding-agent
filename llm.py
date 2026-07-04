from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

if __name__ == "__main__":
    response = llm.invoke("Say hello in one short sentence.")  # send a prompt, wait for reply
    print(response.content)                                   