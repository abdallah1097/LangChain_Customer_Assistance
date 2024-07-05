from fastapi import Body, FastAPI

from src.customer_assistance_Agent import CustomerAssistanceAgent

app = FastAPI()


@app.post("/answer")
async def answer(question: str = Body(...)):
    """
    Receives a question in the request body and returns the answer from the pipeline.
    """
    pass


# Run the main APP
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
