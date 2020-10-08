from ur5_env.env.mujoco_controller import MJ_Controller
import time

# create controller instance
controller = MJ_Controller()

#time.sleep(10)

# Display robot information
controller.show_model_info()

#time.sleep(10)

# Move ee to position above the object, plot the trajectory to an image file, show a marker at the target location
controller.move_ee([0.0, -0.6 , 0.95], plot=True, marker=True)

#time.sleep(5)

# Move down to object
controller.move_ee([0.0, -0.6 , 0.895])

#time.sleep(5)

# Wait a second
controller.stay(1000)

#time.sleep(5)

# Attempt grasp
controller.grasp()

#time.sleep(5)

# Move up again
controller.move_ee([0.0, -0.6 , 1.0])

#time.sleep(5)

# Throw the object away
controller.toss_it_from_the_ellbow()

#time.sleep(5)

# Move ee to position above the object, plot the trajectory to an image file, show a marker at the target location
controller.move_ee([0.0, -0.6 , 0.95], plot=True, marker=True)

#time.sleep(5)

# Move down to object
controller.move_ee([0.0, -0.6 , 0.895])

#time.sleep(5)

# Wait a second
controller.stay(1000)

#time.sleep(5)

# Attempt grasp
controller.grasp()

#time.sleep(5)

# Move up again
controller.move_ee([0.0, -0.6 , 1.0])

#time.sleep(5)

# Throw the object away
controller.toss_it_from_the_ellbow()

# Wait before finishing
controller.stay(2000)
