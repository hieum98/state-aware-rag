import os
from litellm import completion

# check for the environment variable
if "AWS_BEARER_TOKEN_BEDROCK" in os.environ:
    # Print out the value of the environment variable
    print("AWS_BEARER_TOKEN_BEDROCK is set.")
    print("Value:", os.environ["AWS_BEARER_TOKEN_BEDROCK"])

response = completion(
  model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
print(response)