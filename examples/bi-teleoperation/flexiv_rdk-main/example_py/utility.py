#!/usr/bin/env python

"""utility.py

Utility methods.
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import math
# pip install scipy
from scipy.spatial.transform import Rotation as R


def quat2eulerZYX(quat, degree=False):
    """
    Convert quaternion to Euler angles with ZYX axis rotations.

    Parameters
    ----------
    quat : float list
        Quaternion input in [w,x,y,z] order.
    degree : bool
        Return values in degrees, otherwise in radians.

    Returns
    ----------
    float list
        Euler angles in [x,y,z] order, radian by default unless specified otherwise.
    """

    # Convert target quaternion to Euler ZYX using scipy package's 'xyz' extrinsic rotation
    # NOTE: scipy uses [x,y,z,w] order to represent quaternion
    eulerZYX = R.from_quat([quat[1], quat[2],
                            quat[3], quat[0]]).as_euler('xyz', degrees=degree).tolist()

    return eulerZYX


def list2str(ls):
    """
    Convert a list to a string.

    Parameters
    ----------
    ls : list
        Source list of any size.

    Returns
    ----------
    str
        A string with format "ls[0] ls[1] ... ls[n] ", i.e. each value 
        followed by a space, including the last one.
    """

    ret_str = ""
    for i in ls:
        ret_str += str(i) + " "
    return ret_str


def parse_pt_states(pt_states, parse_target):
    """
    Parse the value of a specified primitive state from the pt_states string list.

    Parameters
    ----------
    pt_states : str list
        Primitive states string list returned from Robot::getPrimitiveStates().
    parse_target : str
        Name of the primitive state to parse for.

    Returns
    ----------
    str
        Value of the specified primitive state in string format. Empty string is 
        returned if parse_target does not exist.
    """
    for state in pt_states:
        # Split the state sentence into words
        words = state.split()

        if words[0] == parse_target:
            return words[-1]

    return ""
