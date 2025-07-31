"""
This module contains the class that represents the agent that interacts with the user.
"""

import os
import json
import configparser
import threading

from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain.schema import messages_to_dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from memory_handler import MemoryHandler

from dotenv import load_dotenv

load_dotenv()

def print_log(msg):
    """print log messages in green"""
    print(f"\n\033[92m[LLM AGENT] {msg}\033[00m")

class LLMAgent:
    """
    This class represents the agent that interacts with the user.
    Like a state machine, the robot goes autonomously through different stages of the interaction
    """

    def __init__(
        self,
        temperature: float,
        max_tokens: int,
        prompts_path: str,
        conf_path: str,
        language: str,
        memory_path: str,
    ) -> None:
        # Params from arguments
        self.config_path = conf_path
        self.memory = MemoryHandler(memory_path)

        # Initialize training variables and flags
        self.user_name = None
        self.pose_in_progress = None
        self.language = language
        self.training_level = "unknown"
        self.end_training = False
        self.waiting_for_1st_user_msg = True # this flag is used to wait for the first user message before starting the interaction, reset after the first message      

        # Initialize the agent
        self.setup_llm(temperature, max_tokens)
        self.setup_poses()
        self.initialize_tools()
        self.chain = None
        self.prompts_dict = {}
        self.contexts_dict = {}
        self.initialize_working_memory()
        self.load_prompts_and_contexts(prompts_path)

        # Robot Knowledge and retrieved memories
        self.chat_history = ChatMessageHistory()
        self.window_history = 4 #number of messages to keep in the chat history (gabry used 10)
        self.retrieved_memories = {"memories": "No memories"}

        # Training Instructions
        self.training_instructions = {
            "italian": self.read_from_config_ini("Conversation", "training_instructions_ita"),
            "english": self.read_from_config_ini("Conversation", "training_instructions_eng")
        }

    #### Initialization Functions ####
    def setup_llm(self, temperature, max_tokens):
        """
        setup the llm agent parameters
        """
        self.llm = AzureChatOpenAI(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def setup_poses(self):
        """
        Load poses from the poses folder and populate the poses dictionary.
        We get a dictionary with the level as key and the list of poses as value
        """
        # Loaded dynamically dict of the poses by level, base on the content of "yoga_poses" folder
        poses_path = self.read_from_config_ini("Parameters", "pose_folder")
        
        level_mapping = {
            "level_1": "beginner",
            "level_2": "intermediate",
            "level_3": "advanced"
        }

        self.poses_dict = {}

        try:
            for folder in os.listdir(poses_path):
                folder_path = os.path.join(poses_path, folder)
                if os.path.isdir(folder_path) and folder in level_mapping:
                    level_name = level_mapping[folder]
                    poses = [os.path.splitext(pose)[0].replace('_pose', '')
                             for pose in os.listdir(folder_path)
                             if pose.endswith('_pose.JSON')
                             ]
                    self.poses_dict[level_name] = poses
        except FileNotFoundError:
            print_log(f"Error: Poses directory `{poses_path}` not found.")
        except Exception as e:
            print_log(f"Unexpected error in populate_pose_dict: {e}")

    def initialize_working_memory(self):
        """
        Initialize Working memory structure and its relevant variables for short-term memory system.
        It is used to store the information that the robot needs to keep track of during the interaction and attched to the prompt.
        """
        
        self.working_memory = {
            "User Name": "unknown",
            "User Language": self.language,
            "Training Level": "unknown",
            "Poses Shown": [],
            "Remaining Poses to show": [],
            "Current Goal": {},
            "Status": "Welcoming user"
        }

    def generate_prompt_template(self, system_prompt: str, memories: bool):
        """
        Define prompt template, with WM only or WM + long-term memories
        """

        if memories:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("system", "Your Working Memory: {working_memory}"),
                ("system", "Your Memories: {memories}"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}")
            ]).partial(relevant_info=self.working_memory, memories=self.retrieved_memories)
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("system", "Your Working Memory: {working_memory}"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}")
            ])

        return prompt_template

    #### Auxiliary Functions ####
    def clear_chat_history(self):
        """
        Clear chat history and the tool call history
        """
        self.chat_history.clear()

    def check_user_name(self):
        """
        Sets the proper introduction prompt and tool list based on the user's name.
        """
        is_user_known = self.user_name and self.user_name != "unknown"
        self.waiting_for_1st_user_msg = False  # End the interaction initialization state

        if is_user_known:
            self.working_memory["User Name"] = self.user_name
            # retrieve memories of the past interaction with the user from the graph
            self.retrieved_memories["memories"] = self.memory.retrieve_memories(self.user_name)
            # set chain to the known user prompt
            self.set_chain(self.prompts_dict["intro_known_prompt"].partial(language=self.language, memories=self.retrieved_memories))
        else:
            # set chain to the unknown user prompt
            self.set_chain(self.prompts_dict["take_name_prompt"].partial(language=self.language))
    
    def set_chain(self, prompt):
        """
        We call this function every time we need to change the chain of the conversation
        """

        self.chain = prompt | self.llm_with_tools

    def read_from_config_ini(self, type: str, name: str) -> str:
        """
        Read the configuration parameters and variables from the ini file
        """
        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
            return config.get(type, name)
        except configparser.NoOptionError as e:
            print_log(f"INI Error: Missing key '{name}' in section '{type}'.")
            return ""
        except FileNotFoundError:
            print_log(f"Error: Configuration file '{self.config_path}' not found.")
            return ""

    def load_prompts_and_contexts(self, file_path):
        """ create a dictionary with the prompts from the ini file """
        config = configparser.ConfigParser()
        config.read(file_path)
        if 'Prompt' in config:
            for key in config['Prompt']:
                self.prompts_dict[key] = self.generate_prompt_template(system_prompt = config['Prompt'][key], memories=False)
        if 'Context' in config:
            for key in config['Context']:
                self.contexts_dict[key] = config['Context'][key]

        print_log("Error: No 'Prompt' section in the prompts file.")
        return
    #### Tools Activation Functions ####

    def handle_tool_activation(self, user_input, response):
        """
        The function is responsible for activating the tool and returning the response to the llm.
        Each tool represent a different interaction state of the agent.
        """
        args = json.loads(response.additional_kwargs["tool_calls"][0]["function"]["arguments"])
        tool_name = response.additional_kwargs["tool_calls"][0]["function"]["name"]
        
        print_log(f"TOOL ACTIVATED: {tool_name}")

        if tool_name == "save_user_name":
            self.activate_save_user_name(args)
            return "Thank you, I will rembember your name"
        elif tool_name == "user_profiling":
            self.activate_user_profiling()
            return "Now let me ask you a few questions to evaluate your training level."
        elif tool_name == "start_training":
            return self.activate_start_training(args) #IMPORTANTE: qui forziamo risposta con le istruzioni di allenamento, quindi bypassiamo la risposta dell'LLM
        elif tool_name == "show_yoga_pose":
            self.activate_show_yoga_pose(args)
            return "Okay, I will show you the position and then it is your turn to replicate it." #Here we show a pose, the robot doesn't need to reply
        elif tool_name == "chatting":
            self.activate_chatting()
            return "Let's chat a bit! Ask me anything you want to know."
        elif tool_name == "ask_final_feedback":
            self.ask_final_feedback()
            return "I hope you enjoyed the training! Please, let me know your feedback."
        elif tool_name == "goodbye":
            return self.end_yoga_session() #WE FORCE THE REPLY TO END THE INTERACTION
        else:
            print("Error: Tool not found")
            answer = "Error: Tool not found"

        # we let the robot reply to the user after the tool activation
        answer = self.chain.invoke({"input": user_input, "chat_history": self._get_windowed_chat_history(), "working_memory": self.working_memory})
        
        if "tool_calls" in answer.additional_kwargs:
            print("ERROR: double tool activation")
            print(f"Function call after tool activation: {answer.additional_kwargs['tool_calls'][0]['function']['name']}")

        return answer.content

    def activate_save_user_name(self, args):
        """
        Se save the user name in the working memory and create the user folder in the memory
        """
        # Log the tool activation and the retrieved user name
        self.memory.collect_logs("TOOL ACTIVATED: NAME RETRIEVER")

        # Retrieve and store the user's name,
        self.user_name = args["name"].lower()
        self.working_memory["User Name"] = self.user_name

        print_log(f"User name set: {self.user_name}")

        # Ensure the user's folder exists, creating it if necessary
        self.memory.create_user_folder(self.user_name)
        # start logging the conversation
        self.memory.collect_logs(f"User name: {self.user_name}")

        # Set the main prompt for the next stage
        self.set_chain(self.prompts_dict["intro_new_user_prompt"])

    def activate_user_profiling(self):
        """
        Activation of the tool for user profiling
        """
        # Log the activation of the user profiling tool
        self.memory.collect_logs("TOOL ACTIVATED: READY FOR TRAINING")

        # Update the working memory
        self.working_memory["Status"] = "User profiling"
        self.working_memory["Current Goal"] = "Evaluate the user's training level"

        # Update the main prompt to reflect the user profiling stage
        self.set_chain(self.prompts_dict["user_profiling_prompt"])

    def activate_start_training(self, args):
        """
        Activation of the tool that plans the yoga training session, based on the user's training level. 
        This function updates the relevant information and the training status.
        """
        # Read tool activation arguments
        self.training_level = args["training_level"]

        # Log tool activation
        self.memory.collect_logs("TOOL ACTIVATED: PLAN YOGA SESSION (training level: " + self.training_level + ")")

        # Update the working memory
        self.working_memory.update({
            "Remaining Poses to show": self.poses_dict[self.training_level],
            "Training Level": self.training_level,
            "Status": "Training started",
            "Current Goal": "Show the yoga poses"
        })

        # Update the main prompt to continue with the next training stage
        self.set_chain(self.prompts_dict["training_prompt"])

        yoga_instructions = self.training_instructions[self.language]

        return yoga_instructions

    def activate_show_yoga_pose(self, args):
        """
        Activation of the tool for showing a yoga pose
        """
        print(f"🧘 showing {args['pose_name'].upper()} pose 🧘")
        # Log tool activation
        self.memory.collect_logs(f"TOOL ACTIVATED: SHOW_YOGA_POSE ({args["pose_name"]})")

        # setting the pose in progress will trigger the robot to show the pose
        self.pose_in_progress = args["pose_name"]

        # Update the working memory
        self.working_memory["Remaining Poses to show"].remove(self.pose_in_progress)
        self.working_memory["Poses Shown"].append(self.pose_in_progress)
        self.working_memory["Status"] = "Training started"

        # if we have shown all the poses, we update the status of the interaction to drive the conversation to the end
        if len(self.working_memory["Remaining Poses to show"]) == 0:
            self.working_memory.update({
            "Status": "Training completed",
            "Current Goal": "End the training"})

    def activate_chatting(self):
        """
        Activate the tool for chatting
        """
        # Log tool activation
        self.memory.collect_logs("TOOL ACTIVATED: CHATTING")

        # Update the working memory
        self.working_memory["Status"] = "Chatting"
        self.working_memory["Current Goal"] = "Chat with the user"

        #FIXME: check how this thing works, we need to change the prompt to the chatting one

        # The chat is activated only when the user asks for it, we can chat anytime but depending on the training stage we can have different contexts
        if self.working_memory["Status"] == "User profiling":
            if self.training_level == "unknown":
                context = self.contexts_dict["context_profiling_without_level"]
            else:
                context = self.contexts_dict["context_profiling_with_level"]
        else:
            context = self.contexts_dict["context_intro"]

        # changing the main prompt in order to pass in the chatting stage
        self.set_chain(self.prompts_dict["chatting_prompt"].partial(context=context, name=self.user_name))

    def ask_final_feedback(self):
        '''
        Activate the tool for taking the user feedback after the training
        '''
        # Log tool activation
        self.memory.collect_logs("TOOL ACTIVATED: END TRAINING")
        self.end_training = True #FIXME: i dont like this way of telling the main.py that we want to save the rawperfomanace @Luca

        # Update the working memory
        self.working_memory["Status"] = "Asking final feedback"
        self.working_memory["Current Goal"] = "Take the user feedback"

        # changing the main prompt in order to pass in the feedback stage
        self.set_chain(self.prompts_dict["feedback_prompt"])

    def end_yoga_session(self):
        """
        This function is called at the end of the interaction, in order to save the conversation in the memory
        and build a graph with the relevant information.
        Finally, this function resets the interaction, waiting for a new interaction.
        """
        # Log tool activation
        self.memory.collect_logs("TOOL ACTIVATED: GOODBYE")

        self.working_memory["Status"] = "Goodbye"
        self.working_memory["Current Goal"] = "Say goodbye"

        ### MEMORY PIPELINE ###
        # 1) Save the raw chat in the memory
        self.memory.save_raw_chat(messages_to_dict(self.chat_history.messages))

        # 2) Create Summary based on the raw chat and then build the Knowledge graph, these two processes are done sequencially but in a separate thread
        thread = threading.Thread(target=lambda: (
            self.memory.save_the_raw_table(),
            self.memory.create_meta_performance_table(),
            self.memory.create_summary(self.llm, self.working_memory),
            #self.memory.build_knowledge_graph()
        ))
        thread.start()

        reply = "👋 Goodbye " + self.user_name + "!"

        # The interaction is ended but we wait for the memory to process the chat before resetting the interaction
        print_log("Waiting for the memory to process the chat...")
        thread.join()

        # finally, reset the interaction in order to wait for a new interaction
        self.reset_interaction()
        
        return reply

    def reset_interaction(self):
        """ this function initializes the interaction before starting a new one """
        self.training_level = "unknown"
        self.pose_in_progress = None
        self.user_name = None
        self.end_training = False
        self.waiting_for_1st_user_msg = True #FIXME: this way of enabling the interaction is a total shit
        self.clear_chat_history()
        self.initialize_working_memory()
        self.setup_poses()
        self.memory.clear_memory()

    def _get_windowed_chat_history(self):
        """
        Return the last n messages of chat history or the full history if shorter.
        """
        if len(self.chat_history.messages) >= self.window_history:
            return self.chat_history.messages[-self.window_history:]
        return self.chat_history.messages

    def chat(self, user_input):
        """
        Invoke the LLM with the appropriate prompt, chat history and working memory.
        Handle tool calls if necessary
        """

        # we fake the first user input to be "ciao"
        if self.waiting_for_1st_user_msg:
            self.check_user_name()
            user_input  = "Ciao"
            self.waiting_for_1st_user_msg = False

        self.chat_history.add_user_message(user_input)

        response = self.chain.invoke({"input": user_input, "chat_history": self._get_windowed_chat_history(), "working_memory": self.working_memory})
        reply = response.content

        # Check if the LLM called a tool
        if "tool_calls" in response.additional_kwargs:
            reply = self.handle_tool_activation(user_input, response)

        # finally we add the robot message to the chat history
        self.chat_history.add_ai_message(reply)

        return reply

    def initialize_tools(self):
        """
        This function is used to initialize the tools that the llm can use
        """

        @tool("chatting", return_direct=False)
        def general_chat() -> bool:
            """
            Use it only when the user asks you questions about general topics, not necessarily related to the yoga.
            For example, "user: 'I want to chat a bit'" or "user: 'I have a question'".
            Don't call this tool if in your Working Memory:{"Status": "Chatting"}.
            """
            return True

        @tool("user_profiling", return_direct=False)
        def user_profiling() -> bool:
            """
            Use it after the greetings and before starting the yoga training, in order to evaluate the training level of the user.
            For example, when you have this conversation:
            "assistant: 'Are you ready for the yoga training?'
             user: 'Yes' or user: 'I am ready' or user: 'we can start the training'
             or user: 'let's begin/start' "
             NEVER CALL THIS TOOL TWICE, E.G. IF IN YOUR WORKING MEMORY THE STATUS IS "User profiling" DO NOT CALL THIS FUNCTION AGAIN.
             .
            """
            return True

        @tool("start_training", return_direct=False)
        def start_training(training_level: str) -> bool:
            """
            Call this function to start the exercise once you know the training level of the user, always call this function before showing yoga poses.
            For example, when you have this conversation:
            "assistant: 'I would consider you as an advanced level. Do you agree?'
            user: 'yes'."
            Never call this function if in your Working Memory "Training Level" is known or if 'Status': 'Training started'.
            The input of this tool is the training level of the person, that can be: 'beginner', 'intermediate' or 'advanced'.
            """
            return True

        @tool("show_yoga_pose", return_direct=False)
        def show_the_pose(pose_name: str) -> bool:
            """
            Once you and the user agreed on the pose to show, call this function to show the yoga pose. Wait for the user to be ready.
            Choose a pose among the ones in the list of the poses to show in your working memory.
            Example: "Assistant: Are you ready for the yoga pose 'albero'?" "user: 'yes'"
            """
            return True

        @tool("ask_final_feedback", return_direct=False)
        def ask_final_feedback() -> bool:
            """
            Call this function when the user wants to end the training, the Training is completed and/or when your goal is to ask a final feedback about the training.
            Be careful not to call this function when you still poses to show or when the user asks to do the final pose.
            Never call this function if in your Working Memory the "Status" is "Asking final feedback".
            For example, call this function when you have this conversation:
            "user: 'I want to end the training'" or "user: 'I have to go now, let's continue next time'"
            or "user: 'I have to leave'" or "user: 'I am sorry, I can not continue the training'"
            or "user: 'I want to finish the training' " or "user: 'that's enough for today'" or when the user has completed all the yoga poses.
            """
            return True

        @tool("save_user_name", return_direct=False)
        def save_user_name(name: str) -> bool:
            """
            Use this tool to save the user name and surname if in your Working Memory the "User Name" is "unknown".
            Never call this function if the user name is already known.
            args:
                - name: name and surname of the user (e.g. "John Doe")
            """
            return True


        @tool("goodbye", return_direct=False)
        def log() -> bool:
            """
            Use only after you have obtained, the user feedback on the yoga training, and when the user says goodbye to you.
            for example: "user: 'goodbye"
            """
            return True

        self.llm_with_tools = self.llm.bind_tools([save_user_name, user_profiling, start_training, show_the_pose, general_chat, ask_final_feedback, log])
