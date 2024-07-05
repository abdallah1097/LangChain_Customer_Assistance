from src.customer_assistance_Agent import CustomerAssistanceAgent

if __name__ == '__main__':
    # Define CustomerAssistanceAgent instance
    agent = CustomerAssistanceAgent()

    while True:
        # Get user input
        question = input("Enter your question: ")

        # Get answer from pipeline
        answer = agent.pipeline.run(question)

        # Format & print output answer
        print(agent.prompt.format(question=question, answer=answer))
