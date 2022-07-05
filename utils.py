import numpy as np
import RobotDART as rd
import dartpy

def box_into_basket(box_translation, basket_translation, basket_angle):
    basket_xy_corners = np.array([basket_translation[0] + 0.14, basket_translation[0] + 0.14, basket_translation[0] - 0.14, basket_translation[0] - 0.14,
                                  basket_translation[1] - 0.08, basket_translation[1] + 0.08, basket_translation[1] + 0.08, basket_translation[1] - 0.08], dtype=np.float64).reshape(2, 4)

    rotation_matrix = np.array([np.cos(basket_angle), np.sin(basket_angle), -np.sin(basket_angle), np.cos(basket_angle)], dtype=np.float64).reshape(2, 2)

    basket_center = np.array([basket_translation[0], basket_translation[1]], dtype=np.float64).reshape(2, 1)
    rotated_basket_xy_corners = np.matmul(rotation_matrix, (basket_xy_corners - basket_center)) + basket_center

    d1 = (rotated_basket_xy_corners[0][1] - rotated_basket_xy_corners[0][0]) * (box_translation[1] - rotated_basket_xy_corners[1][0]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][0]) * (rotated_basket_xy_corners[1][1] - rotated_basket_xy_corners[1][0])
    d2 = (rotated_basket_xy_corners[0][2] - rotated_basket_xy_corners[0][1]) * (box_translation[1] - rotated_basket_xy_corners[1][1]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][1]) * (rotated_basket_xy_corners[1][2] - rotated_basket_xy_corners[1][1])
    d3 = (rotated_basket_xy_corners[0][3] - rotated_basket_xy_corners[0][2]) * (box_translation[1] - rotated_basket_xy_corners[1][2]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][2]) * (rotated_basket_xy_corners[1][3] - rotated_basket_xy_corners[1][2])
    d4 = (rotated_basket_xy_corners[0][0] - rotated_basket_xy_corners[0][3]) * (box_translation[1] - rotated_basket_xy_corners[1][3]) - \
        (box_translation[0] - rotated_basket_xy_corners[0][3]) * (rotated_basket_xy_corners[1][0] - rotated_basket_xy_corners[1][3])

    if ((d1 > 0.0) and (d2 > 0.0) and (d3 > 0.0) and (d4 > 0.0) and (box_translation[2] <= 0.04)):
        return True
    else:
        return False

def create_grid(box_step_x=0.5, box_step_y=0.5, basket_step_x=1., basket_step_y=1.):
    basket_positions = []
    basket_x_min = -2.
    basket_x_max = 2.
    basket_y_min = -2.
    basket_y_max = 2.

    basket_nx_steps = int(np.floor((basket_x_max-basket_x_min) / basket_step_x))
    basket_ny_steps = int(np.floor((basket_y_max-basket_y_min) / basket_step_y))

    for x in range(basket_nx_steps+1):
        for y in range(basket_ny_steps+1):
            basket_x = basket_x_min + x * basket_step_x
            basket_y = basket_y_min + y * basket_step_y
            if (np.linalg.norm([basket_x, basket_y]) < 2.):
                continue
            basket_positions.append((basket_x, basket_y))

    box_positions = []
    box_x_min = -1.
    box_x_max = 1.
    box_y_min = -1.
    box_y_max = 1.

    box_nx_steps = int(np.floor((box_x_max-box_x_min) / box_step_x))
    box_ny_steps = int(np.floor((box_y_max-box_y_min) / box_step_y))

    for x in range(box_nx_steps+1):
        for y in range(box_ny_steps+1):
            box_x = box_x_min + x * box_step_x
            box_y = box_y_min + y * box_step_y
            if (np.linalg.norm([box_x, box_y]) < 1.):
                continue
            box_positions.append((box_x, box_y))

    return (basket_positions, box_positions)

def damped_pseudoinverse(jac, l = 0.01):
    m, n = jac.shape
    if n >= m:
        return jac.T @ np.linalg.inv(jac @ jac.T + l*l*np.eye(m))
    return np.linalg.inv(jac.T @ jac + l*l*np.eye(n)) @ jac.T

#Calculates the angle that the robot will approach the box/basket
def calculate_approach_angle(des_pos):
    #For calculation with subtraction in one only axis
    if np.abs(des_pos[1])>np.abs(des_pos[0]):
        euler_z=np.pi/2 if des_pos[1]>0 else -np.pi/2
    else:
        euler_z=0 if des_pos[0]>0 else -np.pi
    return euler_z

#Returns the rotation matrix at the desired position
def rot_matrix_Z_axis(des_pos):
    z_angle=calculate_approach_angle( des_pos)
    rot=dartpy.math.eulerXYZToMatrix([0.,0., z_angle]) 

    return rot

#Calculates the translation matrix of the robot at the desired position
def trans_matrix_offset( des_pos, offset=0.5, height_offset=0.):
    '''
    An offset can be set for the x or y axis 
    An offset can be set for height
    '''
    #For calculation with subtraction in one only axis
    trans=np.zeros(3, float)
    trans[2]=height_offset

    if np.abs(des_pos[1])>np.abs(des_pos[0]): #if |y|>|x|
        trans[0]=des_pos[0]
        trans[1]=des_pos[1]-offset if des_pos[1]>0 else des_pos[1]+offset
    else:                                     #if |y|<=|x|
        trans[0]=des_pos[0]-offset if des_pos[0]>0 else des_pos[0]+offset
        trans[1]=des_pos[1]

    return trans

#Returns the desired position depending on the stage that the behavior tree is in 
def desired_position(item, stage):
    desired_position=item.base_pose()

    if stage=="Move to box": #Must move close to box, offset=0.5 
        box_pos_trans = item.base_pose().translation()
        trans=trans_matrix_offset(box_pos_trans)
        rot=rot_matrix_Z_axis(box_pos_trans)
        desired_position=dartpy.math.Isometry3(rot, trans)

    elif stage=="Hover over box": #Must move arm over the box in the right rotation and translation
        box_pos_trans = item.base_pose().translation()
        robot_rot = dartpy.math.eulerXYZToMatrix([0., np.pi/2., 0.])
        box_trans_hover = box_pos_trans+[0., 0., 0.11]   
        desired_position = dartpy.math.Isometry3(robot_rot, box_trans_hover)

    elif stage=="Drop to box": #Keep previous rotation, just drop the arm down
        box_pos_trans = item.base_pose().translation()
        robot_rot = dartpy.math.eulerXYZToMatrix([0., np.pi/2., 0.])
        box_trans_grab = box_pos_trans+[0., 0., 0.01]   
        desired_position = dartpy.math.Isometry3(robot_rot, box_trans_grab)

    elif stage=="Lift box": #Lifts box right above the place that it was, 
        #turns hand so that it is hard to fall from grasping tool
        box_pos_trans = item.base_pose().translation()
        box_trans_lift=trans_matrix_offset(box_pos_trans, offset=0., height_offset= 0.8) 
        robot_rot_lift=dartpy.math.eulerXYZToMatrix([0.,np.pi/2.,np.pi/2.])
        if box_pos_trans[1]==-1: 
            robot_rot_lift=dartpy.math.eulerXYZToMatrix([0.,np.pi/2.,-np.pi/2.])
        desired_position = dartpy.math.Isometry3(robot_rot_lift, box_trans_lift)

    elif stage=="Move to basket": #Moves base close to basket 
        basket_trans=item.base_pose().translation()
        rot2=rot_matrix_Z_axis(basket_trans)
        trans2=trans_matrix_offset(basket_trans,0.5)
        desired_position=dartpy.math.Isometry3(rot2, trans2)
    
    elif stage=="Hover over basket": #Move arm above basket in a position safe to drop the box
        basket_trans_hover=item.base_pose().translation()
        trans=[basket_trans_hover[0],basket_trans_hover[1], 0.6]
        robot_rot_hover_basket=dartpy.math.eulerXYZToMatrix([0.,np.pi/2.,0.])
        desired_position=dartpy.math.Isometry3(robot_rot_hover_basket, trans)
    
    return desired_position

#Notes the new successful position if it does not already exists in the list 
def note_successful_pair(box_position, basket_position):
    f=open("successful_positions.txt", 'r')
    new_position=str(box_position[0:2])+"\t"+str(basket_position[0:2])+"\n"

    exists=False
    for line in f:
        if line==new_position: 
            exists=True
            print("This pair of positions is already in the list of successful positions")
    
    f.close()
    if not exists:
        f=open("successful_positions.txt", 'a')        
        f.write(new_position)
        f.close()
