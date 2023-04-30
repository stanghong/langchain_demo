
# # Scenario 1: Financial Advisor ChatBot

#pip install -U pip
#pip install openai==0.27.0
#pip install gradio==2.0.7

import openai
import gradio

# %%
openai.api_key = "your openai apikey"

messages = [{"role": "system", "content": "You are a tutor in python coding"}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply


demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "python coding tutor")

demo.launch(share=True)
