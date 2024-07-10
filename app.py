from src.customer_assistance_Agent import CustomerAssistanceAgent

if __name__ == '__main__':
    # Define CustomerAssistanceAgent instance
    agent = CustomerAssistanceAgent()

    while True:
        # Get user input
        user_input = input("You: ")

        # Check if the user wants to exit the chat
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break  # Exit the loop to end the conversation

        # Get answer from pipeline
        response = agent.query_with_prefix(user_input)

        # # Get answer from pipeline
        # answer = agent.query_with_prefix(question)

        # # Format & print output answer
        # print(agent.prompt.format(question=question, answer=answer))
