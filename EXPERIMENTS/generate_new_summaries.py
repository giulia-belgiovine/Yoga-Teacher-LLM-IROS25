import os
import langchain
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0.5,
    max_tokens=2048, #was 128
)


prompt_template = PromptTemplate(
    input_variables=["summary", "raw_conversation"],
    template="""
    In the following raw conversation, a robot yoga tutor and a human trainee talk about trainees general interest and sport experience. Then the robot propose to start the yoga training.

    About the talks on user's general interest and practiced sport, please highlight the following information structured in the following way:
    User Information:
    1) user interests and hobbies (which fall not in the sport category), 
    2) user sport practiced currently and how often they practice it (specify frequency and then add a category between regularly, occasionally and never), 
    3) user past sport experience, if any
    4) User experience with yoga: specify whether the user explicitly mention actual or past experience with yoga practice, but not get confused with what the user says about the current yoga training.
    
    About the yoga session done with the robot, please highlight structured in the following way:
    Training Session:
    5) particular difficulties encountered by the user, if any
    6) Feedback given by the user to the robot about training enjoyment and engagement (please be short)
    
    Conclude integrating with "additional Information" that you find in {summary}.
    
    The raw conversation is: {raw_conversation}

    """
    )


def process_user_data(base_folder):
    for user_folder in os.listdir(base_folder):
        user_path = os.path.join(base_folder, user_folder)
        if os.path.isdir(user_path):
            # Find the interaction folder
            interaction_folders = [f for f in os.listdir(user_path) if f.startswith("interaction")]
            if not interaction_folders:
                continue
            interaction_path = os.path.join(user_path, interaction_folders[0])

            # Read Raw_Chat.txt
            raw_chat_path = os.path.join(interaction_path, "Raw_Chat.txt")
            if not os.path.exists(raw_chat_path):
                continue
            with open(raw_chat_path, "r", encoding="utf-8") as f:
                raw_chat = f.read()

            # Read Summary.txt
            summary_path = os.path.join(interaction_path, "Summary.txt")
            summary_info = ""
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_text = f.read()
                    start_marker = "The user's most important information is:"
                    if start_marker in summary_text:
                        summary_info = summary_text.split(start_marker, 1)[1].strip()

            # Generate new summary using LLM
            formatted_prompt = prompt_template.format(summary=summary_info, raw_conversation=raw_chat).strip()
            new_summary = llm.invoke(formatted_prompt).content


            # Save the new summary
            new_summary_path = os.path.join(interaction_path, "Summary_new.txt")
            with open(new_summary_path, "w", encoding="utf-8") as f:
                f.write(new_summary)
            print(f"Processed and saved: {new_summary_path}")


# Specify the base folder containing user data
base_folder = "yoga_tutor_memories"
process_user_data(base_folder)
