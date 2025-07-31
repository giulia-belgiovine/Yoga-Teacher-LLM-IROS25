""" This class describes the angles of the body segments of a person in a yoga pose. """
import numpy as np

class Pose:
    """ This class computes the angles of the body segments of a person in a yoga pose """
    def __init__(self):
        self.angle_dict = {"neck": ["neck", "nose"], 
                           "left_shoulder": ["neck", "left_shoulder"],
                           "right_shoulder": ["neck", "right_shoulder"],
                           "left_elbow": ["left_elbow", "left_shoulder"],
                           "right_elbow": ["right_elbow", "right_shoulder"],
                           "left_knee": ["left_knee", "left_hip"], 
                           "right_knee": ["right_knee", "right_hip"],
                           "right_wrist": ["right_wrist", "right_elbow"],
                           "left_wrist": ["left_wrist", "left_elbow"]}

        # yolo follows the MPII body parts order
        self.joint_dictionary = {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
                                "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7,
                                "right_elbow": 8, "left_wrist": 9, "right_wrist": 10, "left_hip": 11,
                                "right_hip": 12, "left_knee": 13, "right_knee": 14, "left_ankle": 15,
                                "right_ankle": 16}
        self.yoga_angles_dict = {}

    def __call__(self, keypoints):
        """ For a given set of keypoints we compute the angles of the body segments 
        Args:
            keypoints: numpy array of shape 16,2 with the x,y coordinates of the body parts
        Returns:
            yoga_angles_dict: dictionary with the 17 angles of the body segments.
        """
        # compute the neck joint and add it to the keypoints array
        keypoints = self.add_neck_coordinate(keypoints)

        # for each body segment (link) we compute its orientation angle
        for name_angle, parts_angle in self.angle_dict.items():
            root_joint = self.joint_dictionary[parts_angle[0]]
            target_joint = self.joint_dictionary[parts_angle[1]]
            self.yoga_angles_dict[name_angle] = self.get_angle(keypoints[root_joint][0],
                                                              keypoints[root_joint][1],
                                                              keypoints[target_joint][0],
                                                              keypoints[target_joint][1])
        return self.yoga_angles_dict

    def get_angle(self, x1, y1, x2, y2):
        """ For a given link defined by two points (x1, y1) and (x2, y2)
        we compute the angle of the link with respect to the x-axis """
        # get link angular orientation
        angle_rad = np.arctan2(y1 - y2, x1 - x2)
        # convert to degrees
        angle_deg = angle_rad * 180 / np.pi

        return angle_deg

    def add_neck_coordinate(self, keypoints):
        """ Since yolo does not provide the neck joint, we add it to the dictionary """
        self.joint_dictionary["neck"] = 17
        # then we compute the neck joint position as the mean of the left and right shoulder
        neck = [(keypoints[5][0] + keypoints[6][0]) / 2, (keypoints[5][1] + keypoints[6][1]) / 2]
        # append the neck joint to the keypoints numpy array with shape 16,2
        keypoints = np.vstack([keypoints, neck])
        return keypoints