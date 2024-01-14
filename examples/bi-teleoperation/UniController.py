
import socket
import json
import time
from threading import Thread
from flexiv_controller import FlexivController
from bi_flexiv_controller import get_custom_world_model
import numpy as np  
import transforms3d as t3d

def unity2zup_right_frame(pos_quat):
        pos_quat*=np.array([1,-1,1,1,-1,1,-1])
        rot_mat = t3d.quaternions.quat2mat(pos_quat[3:])
        pos_vec = pos_quat[:3]
        T=np.eye(4)
        T[:3,:3]= rot_mat
        T[:3,3]=pos_vec
        fit_mat = t3d.euler.axangle2mat([0,1,0],np.pi/2)
        fit_mat = fit_mat@t3d.euler.axangle2mat([0,0,1],-np.pi/2)
        target_rot_mat=fit_mat@rot_mat
        target_pos_vec=fit_mat@pos_vec
        target = np.array(target_pos_vec.tolist()+t3d.quaternions.mat2quat(target_rot_mat).tolist())
        return target

class UniController(Thread):
    def __init__(self, controller:FlexivController, port=8082):
        self.address = ("192.168.2.223", port)
        self.socket_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #self.socket_obj.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket_obj.setblocking(0)
        self.controller = controller
        self.isOn=True

    def run(self):
        self.socket_obj.bind(self.address)
        print("start receiving...")
        while self.isOn:
            self.receive()
            time.sleep(0.01)

    def receive(self):
        try:
            data, _ = self.socket_obj.recvfrom(1024)
            s=json.loads(data)
            if self.controller.homing_state:
                return True
            self.controller.gripper.move(s["rightHand"]["squeeze"], 10, 20)

            if s["rightHand"]["cmd"]==3:
                self.controller.robot_go_home()
                return True

            r_pos_from_unity = unity2zup_right_frame(np.array(s["rightHand"]["pos"]+s["rightHand"]["quat"]))


            if self.controller.homing_state:
                self.controller.tracking_state=False
            else:
                if s["rightHand"]["cmd"]==2:
                    if self.controller.tracking_state:
                        print("robot stop tracking")
                        self.controller.tracking_state=False
                    else:
                        print("robot start tracking")
                        self.controller.set_start_tcp(r_pos_from_unity)



            if not self.controller.homing_state:
                right_target = self.controller.get_relative_target(r_pos_from_unity)
                if np.linalg.norm(right_target[:3]-self.controller.get_current_tcp()[:3])>0.5:
                    if self.controller.tracking_state:
                        print("robot lost sync")
                    self.controller.tracking_state=False
                if not self.controller.tracking_state:
                    right_target =self.controller.get_current_tcp()

                #self.controller.mpc_excute(left_target,right_target)
            return True
        except:
            #print("error in udp")
            return False
        
if __name__ == "__main__":

    FC = FlexivController(world_model=get_custom_world_model(),
                          robot_ip="192.168.2.101",
                          origin_offset=[0.0,0.0,0.0])
    FC.init_motion_gen()
    FC.robot_go_home()
    r = UniController(controller=FC)
    r.run()