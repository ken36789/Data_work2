"""The template of the main script of the machine learning process
python MLGame.py -r arkanoid 1 -i ml_play_template_20191007.py
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop():
    """The main loop of the machine learning process
    This loop is run in a separate process, and communicates with the game process.
    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_position_history = [ ]
    # 2. Inform the game process that ml process is ready before start the loop.
    import pickle
    import numpy as np
    filename = 'games\\arkanoid\\ml\\knn_example.sav'
    model = pickle.load(open(filename,'rb'))
    comm.ml_ready()
    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        platform_center_x = scene_info.platform[0]
        ball_position_history.append(scene_info.ball)
        if (len(ball_position_history) > 1):
            vy = ball_position_history[-1][1] - ball_position_history[-2][1]
            vx = ball_position_history[-1][0] - ball_position_history[-2][0]
            inp_temp = np.array([scene_info.ball[0], scene_info.ball[1], scene_info.platform[0],vx,vy])
            input = inp_temp[np.newaxis, :]
            print(input)
        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_OVER:
            print("game over")
            comm.ml_ready()
            continue
        elif scene_info.status == GameStatus.GAME_PASS:
            #scene_info = comm.get_scene_info()
            # Do some stuff if needed
            
            # 3.2.1. Inform the game process that ml process is ready
            print("game pass")
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information
        if (len(ball_position_history) > 1):
            
            move = model.predict(input)
            print(move)
        else:
            move = 0
        # 3.4. Send the instruction for this frame to the game process
        if move < 0:
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
        elif move > 0:
            comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
        else:
            comm.send_instruction(scene_info.frame, PlatformAction.NONE)