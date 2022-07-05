import numpy as np
import RobotDART as rd
import control_bt_utils as c
import dartpy
import py_trees


from utils import create_grid, box_into_basket, note_successful_pair

dt = 0.001

# Create robot
packages = [("tiago_description", "tiago/tiago_description")]
robot = rd.Tiago(int(1. / dt), "tiago/tiago_steel.urdf", packages)

arm_dofs = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint", "gripper_finger_joint", "gripper_right_finger_joint"]
robot.set_positions(np.array([np.pi/2., np.pi/4., 0., np.pi/2., 0. , 0., np.pi/2., 0.03, 0.03]), arm_dofs)
robot.set_cast_shadows(False)
# Control base - we make the base fully controllable
robot.set_actuator_type("servo", "rootJoint", False, True, False)

# Create position grid for the box/basket
basket_positions, box_positions = create_grid()


# Create box
box_size = [0.04, 0.04, 0.04]

# Random cube position
box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0], box_positions[box_pt][1], box_size[2] / 2.0]
box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [0.9, 0.1, 0.1, 1.0], "box_" + str(0))
box.set_draw_axis(box.body_name(0), 0.25)

# Create basket
basket_packages = [("basket", "tiago_pick_place/models/basket")]
basket = rd.Robot("tiago_pick_place/models/basket/basket.urdf", basket_packages, "basket")
# Random basket position
basket_pt = np.random.choice(len(basket_positions)) 
basket_z_angle = 0.
basket_pose = [0., 0., basket_z_angle, basket_positions[basket_pt][0], basket_positions[basket_pt][1], 0.0008]
basket.set_positions(basket_pose)
basket.fix_to_world()



#create [0,0] point
zero_point = rd.Robot.create_box(np.ones((3,1))*0.00001, np.array([0.,0.,np.pi/2.,0.,0.,0.]), "fixed", color=[0., 0., 0., 1.], box_name="world_frame")
zero_point.set_ghost() # do not simulate dynamics/collisions for this: visual only
zero_point.set_cast_shadows(False) # do not cast shadows

# Create Graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 480
gconfig.height = 360
graphics = rd.gui.Graphics(gconfig)
# Create simulator object
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("bullet")
simu.set_control_freq(100)
simu.set_graphics(graphics)
graphics.look_at((0., 4.5, 2.5), (0., -1., -1.))
simu.add_checkerboard_floor()
simu.add_robot(robot)
simu.add_robot(box)
simu.add_robot(basket)
simu.add_robot(zero_point)
finish_counter = 0

box_position=box.base_pose().translation()
basket_position=basket.base_pose().translation()


################################
#####Behavior tree##############
################################

# Behavior Tree
py_trees.logging.level = py_trees.logging.Level.ERROR

# Create tree root
root = py_trees.composites.Parallel(name="Root", policy=py_trees.common.ParallelPolicy.SuccessOnOne())
# Create sequence node (for sequential targets)
sequence = py_trees.composites.Sequence(name="Sequence")

#Move to box
PI=[3.,0.1]
move_to_box=c.ReachTarget(robot,box,dt,"base_link",1,"XY",1e-2,PI, "Move to box")
sequence.add_child(move_to_box)

#Hover over box 
PI=[16.,0.1]
hover_over_box=c.ReachTarget(robot,box,dt,"gripper_grasping_frame",3,"XYZ",1e-2,PI, "Hover over box")
sequence.add_child(hover_over_box)

#Drop to box 
PI=[40.,0.1]
drop_to_box=c.ReachTarget(robot,box,dt,"gripper_grasping_frame",2,"XYZ",1.3e-2,PI, "Drop to box")
sequence.add_child(drop_to_box)

#Close fingers
close_finger=c.MoveFinger(robot,dt,-1,"Closing Finger")
sequence.add_child(close_finger)


#Lift box safely 
PI=[8.,0.1]
lift_box=c.ReachTarget(robot,box,dt,"gripper_grasping_frame",0,"ALL",1e-1,PI, "Lift box")
sequence.add_child(lift_box)

#Move to basket
PI=[3.,0.1]
move_to_basket=c.ReachTarget(robot,basket,dt,"base_link",1,"XY",1e-2,PI, "Move to basket")
sequence.add_child(move_to_basket)

#Hover over basket
PI=[8.,0.1]
hover_over_basket=c.ReachTarget(robot,basket,dt,"gripper_grasping_frame",3,"XY",3e-2,PI, "Hover over basket")
sequence.add_child(hover_over_basket)

#Open fingers
open_fingers=c.MoveFinger(robot,dt,1,"Opening Finger")
sequence.add_child(open_fingers)


#Move back to [0,0]
PI=[2.,0.1]
move_to_zero=c.ReachTarget(robot,zero_point,dt,"base_link",1,"XY",2e-2,PI, "Move to zero")
sequence.add_child(move_to_zero)

root.add_child(sequence)

root.tick_once()

while True:
    if simu.step_world():
        break

    # update our Behavior Tree to control the robot
    root.tick_once()

    box_translation = box.base_pose().translation()
    basket_translation = basket.base_pose().translation()
    if box_into_basket(box_translation, basket_translation, basket_z_angle):
        finish_counter += 1

    if (finish_counter == 10):
        note_successful_pair(box_position, basket_position)

    seq=root.tip()

    if seq.name=="Move to zero" and seq.status==py_trees.common.Status.SUCCESS:
        #Remove old objects
        simu.remove_robot(box)
        simu.remove_robot(basket)
        simu.remove_robot(robot)

        #New box position
        box_pt = np.random.choice(len(box_positions))
        box_pose = [0., 0., 0., box_positions[box_pt][0], box_positions[box_pt][1], box_size[2] / 2.0]        
        box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [0.9, 0.1, 0.1, 1.0], "box_" + str(0))
        box.set_draw_axis(box.body_name(0), 0.25)

        #New basket position
        basket_pt = np.random.choice(len(basket_positions))
        basket_z_angle = 0.
        basket_pose = [0., 0., basket_z_angle, basket_positions[basket_pt][0], basket_positions[basket_pt][1], 0.0008]
        basket = rd.Robot("tiago_pick_place/models/basket/basket.urdf", basket_packages, "basket")
        basket.set_positions(basket_pose)
        basket.fix_to_world()
        
        #Add robots to simulation
        simu.add_robot(box)
        simu.add_robot(basket)
        simu.add_robot(robot)

        #Reset counter
        finish_counter=0

        #Save new positions
        box_position=box.base_pose().translation()
        basket_position=basket.base_pose().translation()

        #Update targets for controllers
        move_to_box.update_target(box)
        hover_over_box.update_target(box)
        drop_to_box.update_target(box)
        lift_box.update_target(box)
        move_to_basket.update_target(basket)
        hover_over_basket.update_target(basket)
