from ur5_env.env.mujoco_controller import MJ_Controller
import random


def get_action():
    low = np.array([-3.14159, -3.14159, -3.14159, -3.14159, -3.14159], dtype=np.float32)
    high = np.array([+3.14159,0,+3.14159,+3.14159,+3.14159], dtype=np.float32)
    action = np.zeros(5, dtype=np.float32)

    for i in range(len(low)):
        action[i] = random.uniform(low[i], high[i])

    return action


STEPS = 100
# create controller instance
controller = MJ_Controller()

# Display robot information
controller.show_model_info()

for s in range(STEPS):
    actiuon = get_action()
    # Move ee to position above the object, plot the trajectory to an image file, show a marker at the target location
    controller.move_group_to_joint_target(group='Arm', target=action, max_steps=1000, quiet=True, render=True, marker=False, tolerance=0.05, plot=False)


# Wait before finishing
controller.stay(2000)
