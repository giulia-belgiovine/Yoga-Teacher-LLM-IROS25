import configparser
import random
import sys
import os
import time
import yarp

from pose import Pose
from trainer import YogaTrainer
from ultralytics import YOLO
from llm_agent import LLMAgent
from memory_handler import MemoryHandler
import numpy as np
import cv2


def log_message(prefix, msg):
    """Generic logging function used for different types of messages."""
    colors = {
        "ICUB": "\033[91m",
        "[YARP INFO]": "\033[92m",
        "[DEBUG]": "\033[96m",
        "[WARNING]": "\033[93m",
        "[ERROR]": "\033[91m"
    }
    reset_color = "\033[00m"
    color = colors.get(prefix.upper(), "")
    print(f"{color}{prefix.upper()}: {msg}{reset_color}")



class YogaModule(yarp.RFModule):
    """
    Description:
        Class to recognize yoga pose from iCub cameras
    """

    def __init__(self):
        yarp.RFModule.__init__(self)
        # Initialize configuration and basic defaults
        self.DEBUG_MODE = True
        self.FAKE_USER_MODE = True

        # warn the user that FAKE_USER_MODE automatically sets DEBUG_MODE to True
        if self.FAKE_USER_MODE:
            log_message("[DEBUG]", "FAKE_USER_MODE is active, DEBUG_MODE is automatically set to True.")
            self.DEBUG_MODE = True
            
        self.frequency_feedback = 10  # frames to give feedback
        self.max_exercise_time = 2 if self.DEBUG_MODE else 60
        self.exercise_success_time = 10
        self.lower_threshold = 30
        self.higher_threshold = self.lower_threshold + 30
        self.threshold = self.lower_threshold

        # Initialize YARP input and output ports.
        self.image_in_port = yarp.BufferedPortImageRgb()
        self.action_port = yarp.RpcClient()
        self.action_port.setRpcMode(True)
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)
        self.speech_recognition_port = yarp.BufferedPortBottle()
        self.faceID_input_port = yarp.BufferedPortBottle()

        # Output port
        self.output_img_port = yarp.BufferedPortImageRgb()
        self.display_buf_image = yarp.ImageRgb()
        self.speech_port = yarp.Port()
        self.faceID_output_port = yarp.RpcClient()
        self.faceID_output_port.setRpcMode(True)
        self.stop_speech_recognition_port = yarp.RpcClient()
        self.stop_speech_recognition_port.setRpcMode(True)

        # Initialize variables related to the yoga system.
        self.current_pose = None
        self.target_pose_name = "rest"
        self.elapsed_time = 0
        self.frame_count = 0
        self.joint_to_fix = None
        self.face_obtained = False   # used as a flag to stop sending the username to FaceID.
        self.name = None

        # Scripted sentence used during yoga practice
        self.show_print = {}
        self.correct_position_print = {}

        # Flags for pose management
        self.start_pose_time = None
        self.pose_success_start = None  # the exact time when the user starts performing the yoga pose correctly.
        self.start_training = False
        self.pose_success = False
        self.pose_showed = False

        # image_variables
        self.width_img = 640
        self.height_img = 480
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8).tobytes()


        # Initialize models: YOLO, memory handler, trainer, and LLM.
        self.yolo_model = None
        self.yoga_trainer = None
        self.llm_agent = None


        if self.DEBUG_MODE and not self.FAKE_USER_MODE:
            self.cap = cv2.VideoCapture(0)

    def configure(self, rf):

        # Configure module parameters #
        module_name = rf.check("name",
                                    yarp.Value("llmYogaTeacher"),
                                    "module name (string)").asString()

        # select size of the YOLO-pose model
        model_size = rf.check("model_size",
                                yarp.Value("n")).asString()
        
        language = rf.check("language",
                            yarp.Value("english"),
                            "language of the prompts (english/italian)").asString()
        
        # This looks for the prompts.ini in the context of the module 
        prompts_path = rf.check('prompts_path',
                                yarp.Value(rf.findFileByName("prompts.ini")),
                                'path containing the ini file with the prompts.').asString()
        log_message("[YARP INFO]", f"Reading prompts from context: {prompts_path}")

        # This looks for the configuration file
        conf_path = rf.check('conf_path',
                            yarp.Value(rf.findFileByName("llmYogaTeacher.ini")),
                            'path containing the ini file with the configurations.').asString()
        log_message("[YARP INFO]", f"Reading conf from context: {conf_path}")


        memory_path = rf.check('memory_path',
                                yarp.Value("/usr/local/src/robot/cognitiveInteraction/yoga-teacher-llm/yoga_tutor_memories"),
                                'path containing the Yoga Tutor Memories').asString()


        # Scripted sentence used during yoga practice
        self.correct_position_print = self.load_messages_from_config(conf_path, "correct_position_print")


        ########## LLM MODEL ##########
        self.llm_agent = LLMAgent(
            temperature = 0.7,
            max_tokens = 128,
            prompts_path = prompts_path,
            conf_path = conf_path,
            language = language,
            memory_path = memory_path
        )

        ########## YOLO MODEL ##########
        # Set the folder where the models are downloaded and stored: icubyogateacher/weights
        yolo_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../weights')
        if not os.path.exists(yolo_folder):
            os.makedirs(yolo_folder)
        self.yolo_model = YOLO(yolo_folder + '/yolo11' + model_size + '-pose.pt')
        self.pose = Pose()

        ########## YOGA TRAINER ##########
        self.yoga_trainer = YogaTrainer(conf_path, os.path.dirname(os.path.abspath(__file__)), self.llm_agent.language)

        ########## OPEN PORTS ##########
        self.handle_port.open('/' + module_name)
        # port to receive the user input
        self.speech_recognition_port.open('/' + module_name + '/speech_recognition:i')
        self.image_in_port.open('/' + module_name + '/image:i')
        self.faceID_input_port.open("/" + module_name + "/faceID:i")
        self.faceID_output_port.open("/" + module_name + "/name:o")
        self.stop_speech_recognition_port.open("/" + module_name + "/thr:o")
        self.speech_port.open('/' + module_name + '/speak:o')
        self.action_port.open('/' + module_name + '/action:o')
        self.output_img_port.open('/' + module_name + '/image:o')

        log_message("[YARP INFO]", "Initialization complete. Yeah!\n")

        return True

    def respond(self, command, reply):
        """ handle rpc commands to the module """
        reply.clear()
        if command.get(0).asString() == "set":
            if command.get(1).asString() == "pose":
                self.target_pose_name = command.get(2).asString()
                if self.yoga_trainer.set_pose(self.target_pose_name):
                    msg = f"Facciamo la posa {self.target_pose_name.split('_')[0]}"
                    self.send_to_acapela(msg)
                    self.execute_action(self.target_pose_name)
                    #time.sleep(6) BUG: if we sleep here the model will not respond anymore, hence the GUI will not detect the change of excercise
                    self.execute_action("go_home_human")
                    self.start_training = True
                    reply.addString(f"Set pose: {self.target_pose_name}")
                else:
                    reply.addString("No pose in progress")
            elif command.get(1).asString() == "threshold":
                self.threshold = command.get(2).asInt16()
                reply.addString("ok")
            elif command.get(1).asString() == "prob":
                self.user_success_probability = command.get(2).asInt16()/100
                #log_message("[DEBUG]", f"Fake user success probability set to {int(self.user_success_probability*100)}%")
                reply.addString("ok")

        elif command.get(0).asString() == "get":
            if command.get(1).asString() == "pose":
                pose_name, _ = self.yoga_trainer.get_pose()
                reply.addString(pose_name)

            elif command.get(1).asString() == "threshold":
                reply.addInt16(self.threshold)
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "start":
            self.start_training = True
            reply.addString("ok")

        elif command.get(0).asString() == "stop":
            self.execute_action("go_home_human")
            self.yoga_trainer.set_pose("rest")
            self.start_training = False
            self.pose_showed = True
            reply.addString("ok")

        elif command.get(0).asString() == "quit":
            reply.addString("quitting")
            self.execute_action("go_home_human")
            self.close()
            return False

        elif command.get(0).asString() == "help":
            help_msg = "Yoga teacher module command are: \n"
            help_msg += "set pose <name_pose> \n"
            help_msg += "set threshold <int> \n"
            help_msg += "get pose \n"
            help_msg += "get threshold: get the current error threshold \n"
            print(help_msg)
            reply.addString(help_msg)

        return True

    def getPeriod(self):
        """
           Module refresh rate.
           Returns : The period of the module in seconds.
        """
        return 0.05

    def updateModule(self):
        # Chose whether to read frame and speech in debug or standard mode
        if self.DEBUG_MODE:
            #DEBUG MODE: evaluate poses from webcam and read input from terminal ###
            
            if self.FAKE_USER_MODE:
                frame = 0
                user_input = self.read_speech_recognition()
            else:
                frame = self.read_from_webcam()
                user_input = self.read_input_from_keyboard()
        else:
            #STANDARD MODE: read frame from and user inputs from yarp ports ###
            frame = self.read_yarp_image()
            user_input = self.read_speech_recognition()

        # Handle face Identification
        self.process_face_interaction()

        ### send the user input to the LLM agent
        if user_input is not None and len(user_input) != "":
            # takes the answer from the LLM
            answer = self.llm_agent.chat(user_input)
            if answer is not None:
                self.llm_agent.memory.collect_logs(f"ICUB: {answer}")
                self.send_to_acapela(answer)

        ###### IF WE ARE IN <SHOW POSE> STATE #######
        if self.llm_agent.pose_in_progress is not None:
            self.handle_show_pose()

        ###### IF WE ARE IN <END TRAINING> STATE ####### #FIXME: this is called at the end of the three poses but it should be called only at the end of the training
        elif self.llm_agent.end_training:
            self.handle_end_training()

        # Check if frame is valid for training phase and output visualization
        if frame is not None:
            ###### IF ICUB IS SHOWING A POSE #######
            if self.yoga_trainer.pose_in_progress:
                self.frame_count += 1 #increase frame count for giving feedback at regular intervals
                self.elapsed_time = time.time() - self.start_pose_time
                
                if not self.FAKE_USER_MODE:
                    frame, joint_angle_errors = self.evaluate_current_pose(frame)
                else:
                    joint_angle_errors = self.yoga_trainer.get_random_joint_errors()

                if self.is_exercise_done(joint_angle_errors):
                    # if the pose is done, we reset the variables
                    self.reset_pose_variables()
                else:
                    print(f"{self.max_exercise_time - self.elapsed_time:.2f} seconds left to complete the pose")
                    self.llm_agent.memory.collect_raw_data(self.elapsed_time, self.llm_agent.pose_in_progress, joint_angle_errors, "False", self.joint_to_fix)


            ### VISUALIZATION ###
            if self.DEBUG_MODE and not self.FAKE_USER_MODE:
                # output final frame as python window
                self.output_as_python_window(frame)
            else:
                # write annotated image with skeleton and joint names on yarp ports
                if self.output_img_port.getOutputCount():
                    self.write_yarp_image(frame)

        return True

    ######### FUNCTIONS ###########

    def read_yarp_image(self):
        # Read input image from yarp port
        input_yarp_image = self.image_in_port.read(False)
        frame = self.get_image_from_bottle(input_yarp_image)

        return frame

    def read_from_webcam(self):
        frame = self.cap.read()[1]
        self.width_img = frame.shape[1]
        self.height_img = frame.shape[0]

        return frame

    def read_input_from_keyboard(self):
        user_input = input("\033[96m" + "HUMAN: " + "\033[00m") if not self.llm_agent.pose_in_progress else None
        if user_input:
            self.llm_agent.memory.collect_logs(f"HUMAN: {user_input}")

        return user_input

    def read_speech_recognition(self):
        # check if speech recognition is reading something
        if self.speech_recognition_port.getInputCount():
            user_input = self.speech_recognition_port.read(shouldWait=False)
            if user_input:
                user_input = user_input.toString()
                print("\033[96m" + "HUMAN: ", user_input)
                self.llm_agent.memory.collect_logs(f"HUMAN: {user_input}")

            return user_input

    def process_face_interaction(self):
        # Check if the user name is valid and send it to the faceID module if the face has not been obtained.
        if self.llm_agent.user_name and self.llm_agent.user_name != "unknown" and not self.face_obtained:
            self.send_name_to_faceID(self.llm_agent.user_name)
            self.face_obtained = True

        # Handle the face interaction, read face data, set user name, and process the response from the LLM.
        if self.llm_agent.waiting_for_1st_user_msg and not self.FAKE_USER_MODE:
            face_bottle = self.faceID_input_port.read(shouldWait=False)
            log_message("[DEBUG]", "Waiting for a face reading from faceID")

            if face_bottle:
                for i in range(face_bottle.size()):
                    face_data = face_bottle.get(i).asList()
                    face_label = face_data.get(0).asList().get(1).asString()

                    # Determine the name based on the face label
                    self.name = "unknown" if "Recognizing" in face_label or face_label == "Unknown face" else face_label
                    self.llm_agent.user_name = self.name

                    # Get and process the LLM's response
                    answer = self.llm_agent.chat("")
                    if answer:
                        self.send_to_acapela(answer)
                        self.start = False
                        self.llm_agent.memory.collect_logs(f"ICUB: {answer}")

    def handle_show_pose(self):
        """ this function is called when the LLM agent activates the tool show_pose """

        # On the beginning of the pose, we show the pose to the user
        if not self.pose_showed:
            # move the robot to the pose
            self.yoga_trainer.set_pose(self.llm_agent.pose_in_progress+"_pose") 
            time.sleep(1)
            self.execute_action(self.llm_agent.pose_in_progress+"_pose")
            self.execute_action("go_home_human")
            self.send_to_acapela("Now it is your turn.")

            # Record that the robot showed the pose, then start the timer
            self.pose_showed = True
            self.start_pose_time = time.time()

    def handle_end_training(self):
        # execute actions
        self.execute_action("go_home_human")
        self.yoga_trainer.set_pose("rest")
        # reset the flag
        self.llm_agent.end_training = False

    def send_message_and_log(self, message_key):
        message = self.show_print[message_key][self.llm_agent.language]
        self.send_to_acapela(message)
        self.llm_agent.chat_history.add_ai_message(message)
        self.llm_agent.memory.collect_logs(f"ICUB: {message}")

    def reset_pose_variables(self):
        # Record the initial start time of the yoga exercise.
        self.elapsed_time = 0
        # set to None the exact time when the user starts performing the yoga pose correctly.
        self.pose_success_start = None
        self.joint_to_fix = None
        self.elapsed_time = 0
        self.pose_showed = False #Flag to track if the robot has phisically shown the pose

    def evaluate_current_pose(self, frame):
        """
        Evaluate current pose and provide feedback to the user.
        """
        result = self.yolo_model(frame, verbose=False)[0]
        if len(result.names) > 0:
            frame = result.plot(boxes=False)
            #convert the joint values to the pose object
            joint_values = result.keypoints.xy.cpu().numpy()[0]
            # convert the joint values to the pose object
            self.current_pose = self.pose(joint_values)

            vocal_feedback, correction_movement, self.joint_to_fix, joint_angle_errors = self.yoga_trainer.get_pose_feedback(
                self.current_pose, self.threshold)

            if self.joint_to_fix:
                # set to False the flag used to note when the yoga pose is performed successfully
                self.pose_success = False
                self.pose_success_start = None
                if self.frame_count % self.frequency_feedback == 0:
                    log_message("ICUB", vocal_feedback)
                    self.send_to_acapela(vocal_feedback)
                    self.execute_action(correction_movement)

            return frame, joint_angle_errors
        else:
            return

    def is_exercise_done(self, joint_angle_errors):
        """ handle the end of the pose when (1) the pose is done OR (2) the time is expired OR (3) we are in fake user mode"""

        #if fake user mode is on, we randomly decide if the pose is done correctly or not
        if self.FAKE_USER_MODE:
            # 1) randomly create a performance report
            self.pose_done_correctly = np.random.choice([True, False], p=[self.user_success_probability, 1-self.user_success_probability])
            self.joint_to_fix = random.choice(list(joint_angle_errors.keys()))

            # 2) log in the performance table
            self.llm_agent.memory.collect_raw_data(timestamp=self.elapsed_time, pose_name=self.llm_agent.pose_in_progress, joint_error=joint_angle_errors, pose_success=self.pose_done_correctly, joint_to_fix=self.joint_to_fix)
            
            # 3) We tell the fake user how the pose has been done so that her/his fake feedback can be consistent with the performance
            trainer_feedback = f"mmh... {self.generate_pose_feedback(self.llm_agent.pose_in_progress, self.pose_done_correctly, self.joint_to_fix)}"
            self.llm_agent.chat_history.add_ai_message(trainer_feedback)
            self.send_to_acapela(trainer_feedback)
            
        # if time is expired, we set the pose as not done correctly
        elif self.elapsed_time > self.max_exercise_time:
            print(f"⏱️​ Timer Expired: You had {self.max_exercise_time} seconds to complete the pose ⏱️​")
            self.llm_agent.memory.collect_logs(f"elapsed time greater than {self.max_exercise_time} seconds")
            self.pose_done_correctly = False
        # otherwise we set the pose as done correctly
        elif self.joint_to_fix is None:
            self.pose_done_correctly = True
        else:
            return False

        # reset the robot pose #FIXME:  user a smarter way to reset the robot pose
        self.execute_action("go_home_human")
        self.yoga_trainer.set_pose("rest")

        # finally we write the exercise info
        self.llm_agent.memory.collect_raw_data(self.elapsed_time, self.llm_agent.pose_in_progress, joint_angle_errors, self.pose_done_correctly, "None")

        # we fake a message from the user to tell the llm that the pose is done
        self.llm_agent.pose_in_progress = None
        answer = self.llm_agent.chat("Done")
        if answer:
            self.send_to_acapela(answer)
            #self.llm_agent.memory.collect_logs(f"ICUB: {answer}")
        
        return True

    def generate_pose_feedback(self, pose_name, success, joint_to_fix):
        """
        Generate natural-sounding feedback for pose performance.
        
        Args:
            pose_name (str): Name of the pose being attempted
            success (bool): Whether the pose was done correctly
            joint_to_fix (str): The joint that needs the most adjustment
        
        Returns:
            str: Natural feedback message
        """
        if success:
            feedback_options = [
                f"Excellent work on the {pose_name}! Your form looks perfect!",
                f"Beautiful {pose_name}! You've really mastered this pose.",
                f"Great job with the {pose_name}! Your alignment is spot-on.",
                f"Wonderful execution of the {pose_name}! Keep up the great work!"
            ]
        else:
            feedback_options = [
                f"Nice attempt at the {pose_name}! Try adjusting your {joint_to_fix} for better alignment.",
                f"You're getting there with the {pose_name}! Let's focus on your {joint_to_fix} position.",
                f"Almost there! For the {pose_name}, pay attention to your {joint_to_fix} placement.",
                f"Good effort on the {pose_name}! A small adjustment to your {joint_to_fix} will make it perfect."
            ]
        
        # Use random choice for variety in feedback
        import random
        return random.choice(feedback_options)

    def get_image_from_bottle(self, yarp_image):
        """ Extract the image from the yarp bottle """
        if yarp_image is not None:
            if yarp_image.width() != self.width_img or yarp_image.height() != self.height_img:
                log_message("[DEBUG]", f"Input image has different size from default 640x480, we'll resize it to {self.width_img}x{self.height_img}")
                self.width_img = yarp_image.width()
                self.height_img = yarp_image.height()
                self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)

            # Convert yarp image to numpy array
            yarp_image.setExternal(self.input_img_array, self.width_img, self.height_img)
            frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
                (self.height_img, self.width_img, 3))  #.copy()?
            return frame
        else:
            return None

    def write_yarp_image(self, image):
        """
        Handle function to stream the recognize pose
        """
        # write the target pose in the top left
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, "Pose: " + self.yoga_trainer.get_pose()[0], (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        display_buf_image = self.output_img_port.prepare()
        display_buf_image.resize(640, 480)
        display_buf_image.setExternal(image.tobytes(), 640, 480)
        
        self.output_img_port.write()

    def send_to_acapela(self, msg):
        """
        Send text to acapela speak module to vocalize the text
        """
        log_message("ICUB", msg)
        speak_bottle = yarp.Bottle()
        speak_bottle.clear()
        speak_bottle.addString(msg)
        self.speech_port.write(speak_bottle)

        return True

    def execute_action(self, action):
        """
        Send action to interactionInterface
        :param action:
        :return: None
        """
        if self.action_port.getOutputCount():

            action_bottle = yarp.Bottle()
            response = yarp.Bottle()
            action_bottle.clear()
            response.clear()
            action_bottle.addString("exe")
            action_bottle.addString(action)
            self.action_port.write(action_bottle, response)
            log_message("[DEBUG]", f"Sending to action port cmd: {action_bottle.toString()}")

        return True

    def output_as_python_window(self, frame):
        # print output as python window
        cv2.putText(frame, "Pose: " + self.yoga_trainer.get_pose()[0], (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("YOGA Teacher", frame)
        cv2.waitKey(1)
   
    def send_name_to_faceID(self, user_name):
        """
        Send the user name to the faceID module
        """
        if self.faceID_output_port.getOutputCount():
            name_bottle = yarp.Bottle()
            response = yarp.Bottle()
            name_bottle.clear()
            response.clear()
            name_bottle.addString("name")
            name_bottle.addString(user_name)
            name_bottle.addString("id")
            name_bottle.addString("11")
            self.faceID_output_port.write(name_bottle, response)
            log_message("[DEBUG]", f"Sending to faceID output port cmd: {name_bottle.toString()}")
        return True

    def set_speech_recognition_thr(self, thr):
        """
        Set the energy threshold of the speech2text module
        :param: threshold
        :return: True
        """
        if self.stop_speech_recognition_port.getOutputCount():

            thr_bottle = yarp.Bottle()
            response = yarp.Bottle()
            thr_bottle.clear()
            response.clear()
            thr_bottle.addString("set")
            thr_bottle.addString("thr")
            thr_bottle.addString(thr)
            self.stop_speech_recognition_port.write(thr_bottle, response)
            log_message("[DEBUG]", "Sending to faceID output port cmd: {}".format(thr_bottle.toString()))

        return True

    def load_messages_from_config(self, conf_path, config_section: str) -> dict:
        """
        Loads print messages from the configuration file for a given section.

        :param config_section: Section in the configuration file (e.g., "show_print" or "correct_position_print")
        :return: A dictionary with all relevant multilingual data.
        """
        messages = {}
        try:
            # Iterate through the structure keys (e.g., part_one, part_two, etc.)
            for key_prefix in ["part_one", "part_two", "end"]:
                for language in ["italian", "english"]:
                    # Build the full key (e.g., part_one_italian, part_two_english)
                    key = f"{key_prefix}_{language}"
                    message = self.read_from_config_ini(conf_path, config_section, key)
                    # Populate nested dictionary
                    if key_prefix not in messages:
                        messages[key_prefix] = {}
                    if message:
                        messages[key_prefix][language] = message
        except Exception as e:
            log_message("[ERROR]", f"Failed to load {config_section} from configuration: {e}")
        return messages

    def read_from_config_ini(self, conf_path: str, type: str, name: str) -> str:
        """
        Read the configuration parameters and variables from the ini file
        """
        config = configparser.ConfigParser()
        try:
            config.read(conf_path)
            return config.get(type, name)
        except configparser.NoOptionError as e:
            log_message("[ERROR]", f"INI Error: Missing key '{name}' in section '{type}'.")
            return ""
        except FileNotFoundError:
            log_message("[ERROR]", "Configuration file '{self.prompts_path}' not found.")
            return ""

    def interruptModule(self):
        log_message("[YARP INFO]", "stopping the module")
        self.execute_action("go_home_human")
        self.handle_port.interrupt()
        self.image_in_port.interrupt()
        self.speech_port.interrupt()
        self.action_port.interrupt()
        self.output_img_port.interrupt()
        self.speech_recognition_port.interrupt()
        self.faceID_input_port.interrupt()
        self.faceID_output_port.interrupt()
        self.close()
        return True

    def close(self):
        log_message("[YARP INFO]", "Closing ports")
        self.handle_port.close()
        self.image_in_port.close()
        self.speech_port.close()
        self.action_port.close()
        self.output_img_port.close()
        self.speech_recognition_port.close()
        self.faceID_input_port.close()
        self.faceID_output_port.interrupt()
        return True


if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        print("Unable to find a yarp server, exiting...")
        sys.exit(1)

    yarp.Network.init()
    yogaTeacherModule = YogaModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('llmYogaTeacher')
    rf.setDefaultConfigFile('llmYogaTeacher.ini')

    if rf.configure(sys.argv):
        yogaTeacherModule.runModule(rf)

    sys.exit()
