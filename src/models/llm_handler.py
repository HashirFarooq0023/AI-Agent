import os
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LLMHandler:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: GROQ_API_KEY not found in environment variables.")
        
        # Initialize Groq Client
        # Using Llama 3.3 70B for high quality and current support.
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            api_key=self.api_key
        )

    def generate(self, prompt_text: str) -> str:
        """
        Simple generation from a string prompt.
        """
        try:
            messages = [("user", prompt_text)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"

    def generate_with_context(self, query: str, context: List[Dict], system_prompt: str) -> str:
        """
        RAG Generation: context + system prompt + user query.
        """
        try:
            # Format context into a string
            context_str = ""
            if context:
                context_str = "HERE IS THE RETRIEVED CONTEXT (Products/Info):\n"
                for idx, item in enumerate(context):
                    context_str += f"--- Item {idx+1} ---\n{item.get('content', '')}\n"
            else:
                context_str = "No specific product information found for this query."

            # Construct the full prompt
            messages = [
                ("system", system_prompt),
                ("system", context_str),
                ("user", query)
            ]

            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"
