import os
from datetime import datetime
import pandas as pd
from prettytable import PrettyTable
from knowledge_Graph.knowledge_graph import KnowledgeGraph
from langchain_core.prompts import ChatPromptTemplate

def log_message(prefix, msg):
    """Generic logging function used for different types of messages."""
    colors = {
        "[MEMORY HANDLER]":"\033[95m",
        "[YARP INFO]":"\033[92m",
        "[DEBUG]":"\033[96m",
        "[WARNING]":"\033[93m",
        "[ERROR]":"\033[91m"
    }
    reset_color = "\033[00m"
    color = colors.get(prefix.upper(), "")
    print(f"{color}{prefix.upper()}: {msg}{reset_color}")

class MemoryHandler:
    """
    Class that handles the memory of the Yoga Tutor. 
    It is responsible for saving:
    - conversation history
    - summary of the conversation
    - raw performance table
    - meta performance table. 

    The memory folder tree is as follows:

    memory_path
    ├── user_name
    │   ├── interaction_of_date_time
    │   │   ├── Raw_Chat.txt
    │   │   ├── Summary.txt
    │   │   ├── Raw_Performance.csv
    │   │   ├── Meta_Performance.csv
    │   │   └── experiment_log.txt

    """

    BASE_FOLDER = "yoga_tutor_memories"
    RAW_CHAT_FILE = "Raw_Chat.txt"
    SUMMARY_FILE = "Summary.txt"
    RAW_PERFORMANCE_FILE = "Raw_Performance.csv"
    META_PERFORMANCE_FILE = "Meta_Performance.csv"
    LOG_FILE = "experiment_log.txt"

    def __init__(self, memory_path: str):
        #these two objects have been developed by Gabriele
        self.knowledge_graph = KnowledgeGraph()

        # list of dictionaries which contains the training data
        self.raw_table_data = []

        # names of the table columns corresponding to the joints error
        self.name_of_joint_error = ["Neck error", "Left shoulder error", "Right shoulder error",
                                    "Left elbow error", "Right elbow error", "Left knee error", 
                                    "Right knee error", "Left wrist error", "Right wrist error"]

        # names of the joint angles index
        self.joint_angles_index = ["neck", "left_shoulder", "right_shoulder",
                                   "left_elbow", "right_elbow", "left_knee",
                                   "right_knee", "left_wrist", "right_wrist"]
       
        log_message("[MEMORY HANDLER]",f"Memory Handler initialized with memory path: {memory_path}")

        # path of the folders where the memories will be stored, (1) the general folder and (2) the user folder (3) the current interaction folder
        self.memory_path = memory_path #this is the user's subfolder named with the time of the interaction e.g. LucaGarello/interaction_of_2022-01-01_12-00-00
        self.user_folder = None
        self.interaction_folder = None

        # create the memory folder, if it does not exist yet
        self.create_memory_folder()

    ### DEFINE DYNAMIC FOLDERS, FILES AND PATHS ###
    # By chainging the parent "interaction folder", we all the other paths will be updated
    @property
    def summary_path(self):
        return os.path.join(self.interaction_folder, self.SUMMARY_FILE)
    @property
    def raw_table_path(self):
        return os.path.join(self.interaction_folder, self.RAW_PERFORMANCE_FILE)
    @property
    def log_file_path(self):
        return os.path.join(self.interaction_folder, self.LOG_FILE)
    @property
    def raw_chat_path(self):
        return os.path.join(self.interaction_folder, self.RAW_CHAT_FILE)
    @property
    def meta_table_path(self):
        return os.path.join(self.interaction_folder, self.META_PERFORMANCE_FILE)
    
    def create_memory_folder(self):
        """Create the general folder where all the memories will be stored"""
        # checks if the folder already exsist, otherwise it creates it
        if not os.path.exists(self.memory_path):
            os.makedirs(self.memory_path)
            log_message("YARP INFO", f"yoga_tutor_memories folder created in {self.memory_path}")
        else:
            log_message("YARP INFO",f"yoga_tutor_memories folder already exists in {self.memory_path}")

    def create_user_folder(self, user_name):
        """Create the folder for the user inside the general memory folder"""
        # gets the user folder path
        self.user_folder = os.path.join(self.memory_path, user_name)
        #checks if the folder already exists
        if not os.path.exists(self.user_folder):
            # creates the folder
            os.makedirs(self.user_folder)
            log_message("[MEMORY HANDLER]",f"NEW USER '{user_name}' has been added to {self.memory_path}.")
        else:
            log_message("[MEMORY HANDLER]",f"'{user_name}' has been found in the memory.")

        # once the user_folder exists, create the interaction folder, this contains each yoga session
        self.create_interaction_folder()

    def create_interaction_folder(self):
        """ Create the folder for the current interaction inside the user folder """
        folder_name = f"interaction_of_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        #definition of the interaction folder
        self.interaction_folder = os.path.join(self.user_folder, folder_name)
        # checks if the interaction folder already exists
        if not os.path.exists(self.interaction_folder):
            # creates the interaction folder
            os.makedirs(self.interaction_folder)
            log_message("[MEMORY HANDLER]",f"Folder '{folder_name}' created successfully.")
        else:
            log_message("[MEMORY HANDLER]",f"Folder '{folder_name}' already exists.")

    def save_raw_chat(self, conversation):
        """Save the Raw_Chat.txt file containing the conversation"""
        try:
            with open(self.raw_chat_path, 'a', encoding='utf-8') as file:
                for item in conversation:
                    # Extract 'type' and 'content' from each element
                    item_type = item['type']
                    item_content = item['data']['content']
                    # Write 'type' and 'content' to the file
                    file.write(f"{item_type}: {item_content}\n")
            log_message("[MEMORY HANDLER]","Raw Chat successfully saved ✅") #in {self.interaction_folder}")
        except Exception as e:
            log_message("ERROR", f"An error occurred while saving the conversation: {e}")

    def create_summary(self, llm, working_memory):
        """Summarize the conversation using the LLM then attach to it the most important information of the user using the working memory"""

        # create the prompt for the summarization
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You have to summarize the input and you must give more importance to the following aspects of the conversation: {main_points}."),
            ("human", "{input}"),
        ]).partial(main_points = ["user name", "interests and hobbies", "sport", "yoga poses", "feedbacks on the training"])

        # create a chain to summarize the conversation
        chain = summary_prompt | llm

        try:
            with open(self.raw_chat_path, 'r', encoding='utf-8') as input_file:
                # gets the conversation content
                conversation_content = input_file.read()
                # summarize the conversation using the LLM
                summary = chain.invoke({"input": conversation_content})
            return self.save_augmented_summary(summary.content, working_memory)
        except Exception as e:
            log_message("ERROR", f"An error occurred while saving the conversation: {e}")
            return None

    def save_augmented_summary(self, summary: str, working_memory: dict):
        """Save the summary of the conversation in a text file + the most important information of the user"""
        
        try:
            with open(self.summary_path, 'w', encoding='utf-8') as summary_file:
                # 1) writes the summary on the file
                summary_file.write(f"{summary}\n")
                # 2) writes the most important information of the user from the working memory
                summary_file.write("\nThe user's most important information is:\n")
                mem_to_extract = ["User Name", "User Language", "Training Level", "Poses Shown"]
                tmp = {k: working_memory[k] for k in mem_to_extract}
                #replace the key "Poses Shown" with "Poses Practiced"
                tmp["Poses Practiced"] = tmp.pop("Poses Shown")
                for key, value in tmp.items():
                    summary_file.write(f"- {key}: {value}\n")

                # 3) finally we incude the meta performance table (part of, only the successful poses, not the numbers)
                summary_file.write("- Poses Done Correctly: ")
                #open the meta performance table as pandas dataframe
                df = pd.read_csv(self.meta_table_path)
                #iterate over the rows of the dataframe
                list_of_successful_poses = []
                for index, row in df.iterrows():
                    if row['Success']:
                        list_of_successful_poses.append(row['Pose'])
                #write the list of successful poses
                summary_file.write(f"{list_of_successful_poses}\n")

            log_message("[MEMORY HANDLER]","Summary saved successfully. ✅")
        except Exception as e:
            log_message("ERROR", f"An error occurred while saving the summary: {e}")

    def collect_raw_data(self, timestamp: float, pose_name: str, joint_error: list, pose_success: bool, joint_to_fix: str):
        """ Log the raw data of the user's performance. this function is called for each frame of the yoga session """
        raw_data = {}
        raw_data['Timestamp'] = timestamp
        raw_data['Pose'] = pose_name

        # this is necessary because for some yoga positions, there is not the neck joint
        if len(joint_error) < len(self.joint_angles_index):
            # set the neck joint error to zero as default
            raw_data[self.name_of_joint_error[0]] = "None"
            for i in range(len(joint_error)):
                raw_data[self.name_of_joint_error[i+1]] = joint_error[self.joint_angles_index[i+1]]
        else:
            for i in range(len(joint_error)):
                raw_data[self.name_of_joint_error[i]] = joint_error[self.joint_angles_index[i]]

        raw_data["Done"] = pose_success
        raw_data["Max error joint"] = joint_to_fix
        self.raw_table_data.append(raw_data)

    def save_the_raw_table(self):
        # Convert all collected data to a DataFrame
        df = pd.DataFrame(self.raw_table_data)
        # Write DataFrame to a CSV file
        df.to_csv(self.raw_table_path, index=False)
        log_message("[MEMORY HANDLER]","Raw Performance file successfully saved ✅") #at {self.raw_table_path}")

    def create_meta_performance_table(self):
        """ Important function that creates the meta performance table based on the raw performance table """
        df = pd.read_csv(self.raw_table_path)

        attempted_poses = df['Pose'].unique()

        # Initialize an empty list to store the data for the new DataFrame
        data = []

        # iterate over Pose Name column to get all the unique pose names
        for pose_name in attempted_poses:

            # Extract from the raw dataframe only the rows with the required pose name
            df_ = df[df['Pose'] == pose_name]
            # Calculate the mean error values for each error column
            mean_errors = df_[['Neck error', 'Left shoulder error', 'Right shoulder error', 'Left elbow error', 'Right elbow error', 'Left knee error', 'Right knee error', 'Left wrist error', 'Right wrist error']].mean().round(2)
            # Calculate the joint that most frequently had the largest error in absolute value
            max_error_joint = df_['Max error joint'].value_counts().idxmax()
            # Find if the pose was successful, a True value on the column is enough
            success = df_['Done'].any()
            # Extract the max execution time based on Frame number
            max_time = df_['Timestamp'].max()
            # Extract the time elapsed before success
            time_elapsed_before_success = df_[df_['Done'] == True]['Timestamp'].min() if success else None
            # Append the data to the list
            data.append([
                pose_name,
                mean_errors['Neck error'],
                mean_errors['Left shoulder error'],
                mean_errors['Right shoulder error'],
                mean_errors['Left elbow error'],
                mean_errors['Right elbow error'],
                mean_errors['Left knee error'],
                mean_errors['Right knee error'],
                mean_errors['Left wrist error'],
                mean_errors['Right wrist error'],
                time_elapsed_before_success,
                max_time,
                success
            ])

            ### DO NOT CANCEL THIS PRINT, IT IS USEFUL FOR DEBUGGING ###
            # table = PrettyTable()
            # table.field_names = ["Joint", "Mean Error"]
            # for joint, error in mean_errors.items():
            #     table.add_row([joint, error])
            # print("\n")
            # print(f"            {pose_name.upper()}")
            # print(table)
            # print(f"Worst joint during pose: \033[91m{max_error_joint}\033[00m")
            # print(f"Maximal time available: \033[91m{int(max_time)} sec.\033[00m")
            # if success:
            #     print(f"Success: \033[92m{success}\033[00m\n")
            #     print(f"Execution time: \033[91m{int(time_elapsed_before_success)} sec. \033[00m")
            # else:
            #     print(f"Success: \033[91m{success}\033[00m\n")

            ### SHORTER VERSION OF THE PRINT, BETTER FOR EXPERIMENTS ###
            print(f"{pose_name.upper()}")
            if success:
                print(f"Success: \033[92m{success}\033[00m\n")
            else:
                print(f"Success: \033[91m{success}\033[00m\n")

        # Create a new DataFrame from the collected data
        columns = [
            'Pose', 'Neck mean error', 'Left shoulder mean error', 'Right shoulder mean error',
            'Left elbow mean error', 'Right elbow mean error', 'Left knee mean error', 'Right knee mean error',
            'Left wrist mean error', 'Right wrist mean error', 'Time elapsed (s)',
            'Max time available (s)', 'Success'
        ]
        summary_df = pd.DataFrame(data, columns=columns)
        
        # Save Meta Performance Table to a CSV file
        summary_df.to_csv(self.meta_table_path, index=False)
        log_message("[MEMORY HANDLER]",f"Meta Performance Table file successfully saved at {self.meta_table_path} ✅")

    def retrieve_memories(self, name):
        """Retrieve the knowledge stored in the knowledge graph"""
        return self.knowledge_graph.retrieve_knowledge(name)

    def retrieve_whole_knowledge(self):
        """Retrieve all the knowledge stored in the knowledge graph"""
        return self.knowledge_graph.retrieve_all_the_knowledge()

    def collect_logs(self,sentence):
        """ This saves a log for what the experiment is seeing on the terminal """
        # takes timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as file:
                # Write the sentence to the file
                file.write(f"{timestamp} - {sentence}\n")
        except Exception as e:
            log_message("ERROR", f"An error occurred while adding the sentence: {e}")
    
    def build_knowledge_graph(self):
        """Build the knowledge graph based on current interaction folder"""
        self.knowledge_graph.build_knowledge_graph(self.summary_path, None)

    def clear_memory(self):
        """Reset all variables and clear the memory of the current interaction"""
        self.raw_table_data = []
        #we dont delete files but we clear the raw table buffer
        log_message("[MEMORY HANDLER]","Memory cleared successfully. ✅")