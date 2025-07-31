from typing import List
import json
import os
import random
import yarp
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}


class FakeUserModule(yarp.RFModule):
    """ This module is a fake user that interacts with the Yoga Teacher iCub."""
    def __init__(self):
        super(FakeUserModule, self).__init__()
        self.interaction_index = 0
        self.llm_chain = None

        self.input_port = yarp.BufferedPortBottle()
        self.output_port = yarp.Port()
        self.rpc_output_port = yarp.RpcClient()

        self.llm = AzureChatOpenAI(
        temperature=0.5,
        max_tokens=128,
        )

        self.profiles_dict = {}
        self.user_name = ""
        self.user_bio = ""
        self.user_success_probability = None

    def configure(self, rf):

        self.input_port.open("/fakeUser/input:i")
        self.output_port.open("/fakeUser/output:o")
        self.rpc_output_port.open("/fakeUser/rpc")

        # connect ports
        yarp.Network.connect('/llmYogaTeacher/speak:o', '/fakeUser/input:i')
        yarp.Network.connect('/fakeUser/output:o', '/llmYogaTeacher/speech_recognition:i')
        yarp.Network.connect('/fakeUser/rpc', '/llmYogaTeacher')

        #load fake profiles from the fake_user_profiles folder
        self.load_fake_profiles()
        # randomly select a profile and initialize the fake user
        self.initialize_fake_user_profile()

        self.set_success_probability()
        self.send_to_yoga_teacher("hi")
        print("\n[DEBUG]: CONTACTING THE YOGA TEACHER TO START\n")

        return True
    
    def load_fake_profiles(self):
        """
        We load the fake user profiles from the fake_user_profiles folder.
        """

        # we read from the fake_user_profiles folder all the .txt and we fill the profiles dictionary
        os.chdir(os.path.dirname(__file__))
        for filename in os.listdir("fake_user_profiles"):
            with open(f"fake_user_profiles/{filename}", "r") as f:
                # remove the .txt extension from the filename
                username = filename.split(".")[0]
                self.profiles_dict[username] = f.read()
        
        return True

    def initialize_fake_user_profile(self):
        """
        We initialize the fake-user-LLM with the fake user profile, giving it a name and some context about habits and interests.
        Each time we call this function, we select a random profile from the list of profiles without the possibility to select the same profile twice.
        """
        
        # we select a random profile from the list of profiles
        self.user_name = "Elena Gomez" #random.choice(list(self.profiles_dict.keys()))

        self.user_bio = self.profiles_dict[self.user_name]

        #find the user success probability from the user bio text, it is expressed as a percentage e.g: **Probability of yoga success**: 85%
        #TODO: do we want to use a json format ?
        self.user_success_probability = int(self.user_bio.split("**Probability of yoga success**: ")[1].split("%")[0])


        # we drop the current name from the list of fake names, so we don't use it again
        self.profiles_dict.pop(self.user_name)
        print(f"LOADED PROFILE: {self.user_bio}")

        prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(
                            #chat: feel free to say yes if you like / ALWAYS SAY NO
                            content= "You are interacting with iCub, a robot assistant for Yoga sessions composed of maximum 3 poses. When the robot ask you to chat, feel free to say yes if you like, briefly explain your interests. Never ask the robot for poses he does not knows. Before starting the exercise tell icub someting about you. When training ends give you honest feedback making up detsils about your performance based on the trainer's feedback. Then say goodbye without asking to wrap up. Always answer the robot in a concise way.\
                            Interact with the Yoga Teacher iCub kwnowing that you are impersonating a yoga student with the following bio: " + self.user_bio
                        ),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}"),
                    ])
        
        def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryHistory()
            return store[session_id]
        
        self.llm_chain = RunnableWithMessageHistory(
                    prompt | self.llm, #chain without history
                    get_by_session_id,
                    input_messages_key="question",
                    history_messages_key="history",
                )
        
        return True

    def set_success_probability(self):
        """ We tell the yoga teacher the probability of success of the fake user. We use its rpc port."""
        bottle = yarp.Bottle()
        bottle.clear()
        response = yarp.Bottle()
        response.clear()
        bottle.addString("set")
        bottle.addString("prob")
        bottle.addInt64(self.user_success_probability)
        self.rpc_output_port.write(bottle, response)
        if response.size() > 0 and response.get(0).asString() == "ok":
            print("\n[DEBUG]: YOGA TEACHER HAS CORRECTLY SET THE SUCCESS PROBABILITY}\n")
        else:
            print("\n[ERROR]: YOGA TEACHER HAS NOT SET THE SUCCESS PROBABILITY}\n")

        return True

    def interruptModule(self):
        self.input_port.interrupt()
        self.output_port.interrupt()
        self.rpc_output_port.interrupt()
        return True

    def close(self):
        self.input_port.close()
        self.output_port.close()
        self.rpc_output_port.close()
        return True

    def respond(self, command, reply):
        if command.toString() == "READY2INTERACT":
            print("\n\033[92mINTERACTION STARTED 🚀\033[0m")

        return True

    def getPeriod(self):
        return 1.0

    def updateModule(self):

        input_bottle = self.input_port.read(True)

        if input_bottle is not None:
            iCub_msg = input_bottle.toString()
            print(f"\033[91mICUB: {iCub_msg}\033[0m")

            # avoid replying to certain messages from the iCub 
            if "mmh..." in iCub_msg or "Now it is your turn" in iCub_msg or "Okay, I will show you the position and then it is your turn to replicate it." in iCub_msg:
                store[self.user_name].add_messages([HumanMessage(content=iCub_msg)])
                store[self.user_name].add_messages([AIMessage(content="")])
                input_bottle.clear()
                
                return True
            
            if "👋 Goodbye" in iCub_msg:
                print("\n\033[92m👋 INTERACTION ENDED SUCCESSFULLY 👋\033[0m")
                self.log_chat_history(store[self.user_name])
                #ask user if he wants to continue loading a new profile, if not, close the module
                # print("\n[DEBUG]: DO YOU WANT TO LOAD A NEW PROFILE? (y/n)")
                # user_input = input()
                # if user_input.lower() == "y":
                #     self.initialize_fake_user_profile()
                #     self.send_to_yoga_teacher("hi")
                #     print("\n[DEBUG]: CONTACTING THE YOGA TEACHER TO START\n")
                # else:
                #     print("\n[DEBUG]: CLOSING MODULE\n")
                #     self.close()

                #wait 5 seconds and then send hi
                yarp.delay(5)
                self.initialize_fake_user_profile()
                self.send_to_yoga_teacher("hi")

                input_bottle.clear()
                return True

            reply = self.llm_chain.invoke(
                {
                    "question": iCub_msg
                },
                {'configurable': {'session_id': self.user_name}}
            )

            print(f"\033[96mHUMAN: {reply.content}\033[0m")
            self.send_to_yoga_teacher(reply.content)
            #reset message
            input_bottle.clear()


        return True
    
    def send_to_yoga_teacher(self, msg):
        """
        Send text to acapela speak module to vocalize the text
        """
        speech_bottle = yarp.Bottle()
        speech_bottle.clear()
        speech_bottle.addString(msg)
        self.output_port.write(speech_bottle)

        return True
    
    def log_chat_history(self, history):
        """
        Log the chat history to a file
        """
        #create the chat_history folder if it does not exist
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "chat_history")):
            os.makedirs(os.path.join(os.path.dirname(__file__), "chat_history"))
        # Percorso del file JSONL, che è ./chat_history/{user_name}.jsonl
        jsonl_path = os.path.join(os.path.dirname(__file__), "chat_history", f"{self.user_name}.jsonl")

        # Salva i messaggi in JSONL in this format:
        # Example row: {"inputs": {"input": "value"}, "outputs": {"output": "value"}}
        # The input is the Human message and the output is the AI message
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(0, len(history.messages), 2):
                human_msg = history.messages[i].content
                ai_msg = history.messages[i+1].content
                f.write(json.dumps({"inputs": {"input": human_msg}, "outputs": {"output": ai_msg}}) + "\n")

        print(f"Messaggi salvati in {jsonl_path}")
        
        return True

if __name__ == "__main__":
    yarp.Network.init()

    module = FakeUserModule()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(["--from", "config.ini"])
    module.runModule(rf)
    yarp.Network.fini()