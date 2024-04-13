import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

ClaudeHaiku = ChatAnthropic(
  model_name="claude-3-haiku-20240307",
  api_key=anthropic_api_key,
)

ClaudeOpus = ChatAnthropic(
  model_name="claude-3-opus-20240229",
  api_key=anthropic_api_key,
  temperature=0.6
)

ClaudeSonnet = ChatAnthropic(
  model_name="claude-3-sonnet-20240229",
  api_key=anthropic_api_key,
  temperature=0.6
)

GPT4Turbo = ChatOpenAI(
  temperature=0.5, model="gpt-4"
)

GPT3Turbo = ChatOpenAI(
  model="gpt-3.5-turbo"
)
