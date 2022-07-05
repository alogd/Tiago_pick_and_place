import RobotDART as rd
import numpy as np
import py_trees
import dartpy
from utils import damped_pseudoinverse, desired_position

#Dof types 
upper_body=['torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
upper_body_rot_z=['rootJoint_rot_z','torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
rot_z_pos_xy=['rootJoint_rot_z','rootJoint_pos_x', 'rootJoint_pos_y' ]
all_dofs=[ 'rootJoint_rot_z', 'rootJoint_pos_x', 'rootJoint_pos_y', 'torso_lift_joint', 'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']


class PITask:
    def __init__(self, target, dt, Kp = 10., Ki = 0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0.
        self.time_past = 0.
    
    def set_target(self, target):
        self._target = target
        self._sum_error = 0.
        self.time_past = 0.
    
    # function to compute error
    def error(self, tf):

        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation() 
        return np.r_[rot_error, lin_error]
    
    def update(self, current):

        error_in_world_frame = self.error(current)
        self._sum_error = self._sum_error + error_in_world_frame * self._dt
    
        if self.time_past<1:
            self.time_past+=self._dt
            return self._Kp * error_in_world_frame * self.time_past + self._Ki * self._sum_error

        return self._Kp * error_in_world_frame + self._Ki * self._sum_error


class JointPositionResetPI:
    def __init__(self,robot, dt,PI=[5.,0.1]):
        self.robot=robot
        self._dt=dt
        self.Kp=PI[0]
        self.Ki=PI[1]
        self._sum_error=np.array([0]*8)
        self.init_pos=np.array([0.,np.pi/2., np.pi/4., 0., np.pi/2., 0. , 0., np.pi/2.])
        self.dofs=upper_body

    def error(self, cur_pos):

        error=self.init_pos-cur_pos
        return error

    def PError(self):
        cur_pso=self.robot.positions(self.dofs)
        error = self.error(cur_pso)
        self._sum_error = self._sum_error + error * self._dt

        return self.Kp * error + self.Ki * self._sum_error

    def update(self):
        
        cmd=self.PError()

        if all(i < 0.075 for i in cmd): 
            self._sum_error=np.array([0]*8)

        return cmd    



class ReachTarget(py_trees.behaviour.Behaviour):
    def __init__(self, robot, item, dt,link_name,cancel_type=0, type_of_error="XYZ", abs_error=1e-3, PI=[10., 0.1],name="ReachTarget"):
        super(ReachTarget, self).__init__(name)
        # robot
        self.robot = robot
        # end-effector name
        self.eef_link_name = link_name
        # set target tf
        self.item=item
        self.name=name
        self.tf_desired = desired_position(item, name)

        # dt
        self.dt = dt
        #additional variables
        self.cancel_type = cancel_type
        self.type_of_error=type_of_error
        self.desired_error=abs_error
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.Kp=PI[0]
        self.Ki=PI[1]

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" % (self.__class__.__name__))

    def initialise(self):
        self.logger.debug("%s.initialise()->init controller" % (self.__class__.__name__))
        self.controller = PITask(self.tf_desired, self.dt, self.Kp, self.Ki)


        if self.name== "Move to zero":
            self.controller2=JointPositionResetPI(self.robot, self.dt)
        else:
            self.controller2=None

    def update_target(self, item):
        self.item=item
        self.tf_desired = desired_position(item, self.name)
        self.controller.set_target(self.tf_desired)


    def update(self):

        new_status = py_trees.common.Status.RUNNING
        # control the robot
        tf = self.robot.body_pose(self.eef_link_name)
        vel = self.controller.update(tf)
        
        dofs = all_dofs

        if self.cancel_type==1:      #move in xy rotz 
            dofs = rot_z_pos_xy
        elif self.cancel_type==2:    #move only upper body
            dofs = upper_body
        elif self.cancel_type==3:    #Move upper body and rotz
            dofs = upper_body_rot_z

        jac = self.robot.jacobian(self.eef_link_name, dofs) # this is in world frame
        jac_pinv = damped_pseudoinverse(jac) # np.linalg.pinv(jac) # get pseudo-inverse
        cmd = jac_pinv @ vel

        self.robot.set_commands(cmd,dofs)
        
        #Reseting joint position while moving to 0,0
        if self.name=="Move to zero": 
            cmd = self.controller2.update()
            self.robot.set_commands(cmd,upper_body)

        # if error too small, report success
        if self.type_of_error=="XYZ":    
            err = np.linalg.norm(self.controller.error(tf)[3:6])
        elif self.type_of_error=="XY":
            err = np.linalg.norm(self.controller.error(tf)[3:5])
        elif self.type_of_error=="ALL":
            err = np.linalg.norm(self.controller.error(tf))

        if self.name!= "Move to zero" and err < self.desired_error :
            new_status = py_trees.common.Status.SUCCESS
        elif err < self.desired_error and all(i < 0.075 for i in cmd):
            new_status = py_trees.common.Status.SUCCESS

        if new_status == py_trees.common.Status.SUCCESS:
            self.robot.set_commands([0]*11, all_dofs)
            self.feedback_message = "Reached target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class MoveFinger(py_trees.behaviour.Behaviour):
    def __init__(self,robot, dt,direction,name="ReachTarget"):
        super(MoveFinger, self).__init__(name)
        self.robot = robot
        self.dt = dt
        self.direction=direction
        self.total_time=0.
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" % (self.__class__.__name__))

    def initialise(self):
        self.logger.debug("%s.initialise()->init controller" % (self.__class__.__name__))
        self.robot.set_commands([self.direction*0.1], ['gripper_finger_joint'])


    def update(self):
        new_status = py_trees.common.Status.RUNNING
        self.total_time+=self.dt

        if self.total_time>1.:
            self.total_time=0
            self.robot.set_commands([self.direction*0.06], ['gripper_finger_joint'])
            new_status = py_trees.common.Status.SUCCESS

        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Finger moved"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "Still moving"
            self.logger.debug("%s.update()[%s][%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
