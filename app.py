from src.customer_assistance_Agent import CustomerAssistanceAgent

if __name__ == '__main__':
    # Define CustomerAssistanceAgent instance
    agent = CustomerAssistanceAgent()

    # Type a question
    question = "What is LLRT used for?"

    # Get answer from pipline
    answer = agent.pipeline.run(question)

    # Format & print output answer
    response = agent.prompt.format(question=question, answer=answer)
    print(response)
