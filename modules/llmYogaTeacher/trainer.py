import os
import glob
import random
import json
from prettytable import PrettyTable

class YogaTrainer:
    """
    Class to manage the yoga poses and the training session
    """

    def __init__(self, conf_path, yoga_data_path, language):

        self.root_dir = yoga_data_path
        self.conf_path = conf_path
        self.language = language

        self.yoga_poses = {}
        self.load_yoga_poses()

        self.current_level = 1
        self.pose_index = 0
        self.total_pose_index = 0
        self.n_poses_session = 8
        # name of the pose in progress, "Rest" if no pose is in progress
        self.pose_in_progress = None
        self.ctr_pose = 0
        self.ctr_pose_success = 0

        self.good_job = {"italian": "Esercizio completato, Ottimo lavoro!", "english": "Exercise completed, great job!"}
        self.not_see_well = {"italian": "Non riesco a vederti interamente", "english": "I can't see you fully"}

    def load_yoga_poses(self):
        """ class to store the yoga poses"""
        class yoga_pose:
            def __init__(self, name, joints, difficulty, index, msg_correction, show_correction):
                self.name = name
                self.target_joint_angles = joints
                self.difficulty = difficulty
                self.index = index
                self.msg_correction = msg_correction
                self.show_correction = show_correction

        list_dir = os.listdir(os.path.join(self.root_dir, "yoga_poses"))

        # initiate a list of yoga poses
        self.yoga_poses = []
        for current_dir in list_dir:
            # extract the level from the folder name
            level = int(current_dir.split("_")[1])
            yoga_files = glob.glob(os.path.join(self.root_dir, "yoga_poses", current_dir, "*.JSON"))
            index = 0
            for f in yoga_files:
                # extract the pose name from the file name
                pose_name = f.split('/')[-1].split("_")[0] + "_pose"
                yoga_dict = json.load(open(f))
                # create a yoga_pose object
                pose = yoga_pose(pose_name, yoga_dict["value"], level, index, yoga_dict["ins"][self.language], yoga_dict["show_correction"])
                # add the pose to the list
                self.yoga_poses.append(pose)
                index += 1

        # sort list based on difficulty
        self.yoga_poses.sort(key=lambda x: x.difficulty)

        # transform the list into a dictionary where the key is the pose name
        self.yoga_poses = {pose.name: pose for pose in self.yoga_poses}

        # print loaded poses in a fancy table
        table = PrettyTable()
        table.field_names = ["Loaded Poses", "Difficulty", "ID"]
        for pose in self.yoga_poses.values():
            table.add_row([pose.name, pose.difficulty, pose.index])
        print("\n")
        print(table)
        print("\n")

    def evaluate_yoga_pose(self, current_pose, threshold):
        """
        Evaluate the current pose with respect to the target pose
        We return the joint with the biggest error and the error itself
        :param current_pose: current dict of joints angles
        :return: ang_pose_error, joint_to_fix
        """
        ang_pose_errors = {}
        joint_to_fix_name = None
        max_diff_joint = 1e-1

        # for each joint we evaluate the difference between the target and current pose
        if self.pose_in_progress:
            for name_joint, angle in self.pose_in_progress.target_joint_angles.items():
                diff = abs(angle) - abs(current_pose[name_joint])
    
                # we save the error for each joint
                ang_pose_errors[name_joint] = diff
                if threshold < abs(diff) > max_diff_joint:
                    max_diff_joint = abs(diff)
                    joint_to_fix_name = name_joint

        return ang_pose_errors, joint_to_fix_name

    def update_exercise(self, target_pose_name):
        """
        Update the current exercise, moving to the next pose
        :return:
        """
        # if we reached the number of poses per session, we stop
        if self.total_pose_index >= self.n_poses_session:
            print("Poses finished")
        # if we reached the last pose of the current level, we move to the next level
        elif self.pose_index == 2:
            self.pose_index = 0
            self.total_pose_index += 1
            self.current_level += 1
        # otherwise we move to the next pose on the same level
        else:
            self.pose_index += 1
            self.total_pose_index += 1

        # use updated index to get the pose name
        print(f"Total pose index: {self.total_pose_index}")
        # based on the index we get the pose from the dictionary
        self.pose_in_progress = list(self.yoga_poses.values())[self.total_pose_index]

    def set_pose(self, pose_name):
        """
        Set the pose exercise, use None to reset the pose
        :param pose_name: name of the pose to set, "rest" is used when the module is stopped
        :return: True if the pose is found, False otherwise
        """
        pose_name = pose_name.lower()

        # when the module is stopped we set the pose to "rest"
        if pose_name in self.yoga_poses.keys():
            #print(f"🧘🤖 [TRAINER] Pose set to: {pose_name} 🧘🤖")
            self.pose_in_progress = self.yoga_poses[pose_name]
            return True

        elif pose_name == "rest":
            self.pose_in_progress = None
            #print("[Trainer] Setting pose to rest")
            return False
        # if the pose is found, we set it with the corresponding level and index

        else:
            print("Desired pose not found, setting pose to idle state 'rest' ")
            self.pose_in_progress = None
            return False

    def get_pose(self):
        """
        Get the current POSE NAME and corresponding TARGET JOINT ANGLES
        """
        if self.pose_in_progress is None:
            joint_angles = None
            pose_name = "rest"
        else:
            pose_name = self.pose_in_progress.name
            joint_angles = self.pose_in_progress.target_joint_angles

        return pose_name, joint_angles

    def get_pose_feedback(self, current_pose, threshold):
        """
        Give textual feedback on the current pose (e.g. "Raise your left arm")
        Args: current_pose: dict of joint angles
        Returns: vocal_feedback, joint_to_fix
        """
        correction_movement = ""
        # read taget pose angles
        target_pose_name = self.get_pose()[0]

        # compute the difference between target and current pose
        angle_errors, joint_to_fix = self.evaluate_yoga_pose(current_pose, threshold)

        # if pose is correct return positive feedback
        if joint_to_fix is None:
            self.ctr_pose_success += 1
            msg = self.good_job[self.language]
        elif 0 in list(current_pose.values()):
            msg = self.not_see_well[self.language]
        else:
            action_msg = self.yoga_poses[target_pose_name].msg_correction[joint_to_fix][0]
            msg = action_msg
            correction_movement = self.yoga_poses[target_pose_name].show_correction[joint_to_fix][0]# added by Mati

        return f"{msg}", correction_movement, joint_to_fix, angle_errors
    
    def get_random_joint_errors(self):
        """
        Get a random joint dictionary with a random error. Used for testing purposes.
        """
        joints = {}
        for joint in self.pose_in_progress.target_joint_angles.keys():
            joints[joint] = random.uniform(-0.1, 0.1)
        return joints