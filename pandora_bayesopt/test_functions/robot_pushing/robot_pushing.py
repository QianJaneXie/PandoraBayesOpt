#!/usr/bin/env python
# Reference: GitHub repo by Raul Astudillo https://github.com/RaulAstudillo06/BudgetedBO/blob/740cbf8397ac68bcdb1a196f2b56381adb1be1b4/experiments/robot_pushing_src/robot_pushing_3d.py
# Copyright (c) 2017 Zi Wang
from .push_world import *


def robot_pushing_3d(
    rx: float,
    ry: float,
    duration: float,
) -> float:
    simu_steps = int(10 * duration)
    # set it to False if no gui needed
    world = b2WorldInterface(False)
    oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (0.3,1) 
    thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))

    init_angle = np.arctan(ry/rx)
    robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
    final_location = simu_push(world, thing, robot, base, simu_steps)
    return final_location


def robot_pushing_4d(
    rx: float,
    ry: float,
    duration: float,
    init_angle: float,
) -> float:
    simu_steps = int(10 * duration)
    # set it to False if no gui needed
    world = b2WorldInterface(False)
    oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (0.3,1) 
    thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))
    
    xvel = -rx;
    yvel = -ry;
    regu = np.linalg.norm([xvel,yvel])
    xvel = xvel / regu * 10;
    yvel = yvel / regu * 10;
    robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
    final_location = simu_push2(world, thing, robot, base, xvel, yvel, simu_steps)
    return final_location


def robot_pushing_14d(
    rx: float,
    ry: float,
    xvel: float,
    yvel: float,
    duration: float,
    init_angle: float,
    rx2: float,
    ry2: float,
    xvel2: float,
    yvel2: float,
    duration2: float,
    init_angle2: float,
    rtor: float,
    rtor2: float
) -> float:
    simu_steps = int(10 * duration)
    simu_steps2 = int(10 * duration2)
    # set it to False if no gui needed
    world = b2WorldInterface(False)
    oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (1,0.3) #'circle', 0.3#
    #thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (0,0))
    base = make_base(500, 500, world)
    thing = make_1thing(base, world, 'rectangle', (0.5,0.5), ofriction, odensity, (0, 2))
    thing2 = make_1thing(base, world, 'circle', 1, ofriction, odensity, (0,-2))
    #xvel = np.cos(init_angle)*5;
    #yvel = np.sin(init_angle)*5;
    
    robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
    robot2 = end_effector(world, (rx2,ry2), base, init_angle2, hand_shape, hand_size)
    final_location = simu_push_2robot2thing(world, thing, thing2, robot, robot2, base, xvel, yvel, xvel2, yvel2, rtor, rtor2, simu_steps, simu_steps2)
    return final_location