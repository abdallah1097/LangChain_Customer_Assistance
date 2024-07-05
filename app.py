from src.customer_assistance_Agent import CustomerAssistanceAgent

if __name__ == '__main__':
    # Define CustomerAssistanceAgent instance
    agent = CustomerAssistanceAgent()

    question = "What is LLRT used for?"


    answer = agent.pipeline.run(question)
    response = agent.prompt.format(question=question, answer=answer)

    print(response)
