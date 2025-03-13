import os

from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI

if __name__ == "__main__":
    llm = ChatVertexAI(
        project=os.environ["PROJECT_ID"],
        location="us-central1",
        model="gemini-1.5-flash-002"
    )

    prompt_template = PromptTemplate.from_template("""
    {dish}  {ingredients}
    """)

    prompt = prompt_template.format(dish="dessert", ingredients="strawberries, chocolate, and whipped cream")

    response = llm.invoke(prompt)
    print(response.content)
