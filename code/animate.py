import numpy as np
import vispy.scene
import vispy
from vispy.scene import visuals
import sys
from vispy import app


iterator = 0
def animate(path, view_cell_type='all', view_polarity_type='ABP', view_camera='fly'):
    """
    Animates the cell data in 3D

    Parameters
    ----------
    path : str
        The path to the data file. Often just "folder/data.npy"
    view_cell_type : str or int
        The type of cell to view.
        Accepted values are ['all', 0, 1]
    view_polarity_type : str
        The type of polarity to view.
        Accepted values are ['ABP', 'PCP', 'none']
    view_camera : str
        The type of camera view to use.
        Accepted values are ['fly', 'turntable']

    """
    # Checking if all parameters are valid
    assert view_camera in ['fly', 'turntable'], "Invalid view_camera, must be one of ['fly', 'turntable']"
    assert view_polarity_type in ['ABP', 'PCP', 'none'], "Invalid view_polarity_type, must be one of ['ABP', 'PCP', 'none']"
    assert view_cell_type in ['all', 0, 1], "Invalid view_cell_type, must be one of ['all', 0, 1]"

    # Getting the data
    data = np.load(path, allow_pickle=True)
    mask_lst, x_lst, p_lst, q_lst = data

    #check if mask_lst contains only None values
    if all(mask is None for mask in mask_lst):
        multiple_types = False
        print("Only one cell type detected")

    else:
        multiple_types = True
        print("Multiple cell types detected")

    #are we visualizing all the cells?
    #if mask_lst only contains Nones we ignore this
    if view_cell_type != 'all' and multiple_types:
        x_lst = [x_lst[i][mask_lst[i] == view_cell_type] for i in range(len(x_lst))]
        p_lst = [p_lst[i][mask_lst[i] == view_cell_type] for i in range(len(p_lst))]
        q_lst = [q_lst[i][mask_lst[i] == view_cell_type] for i in range(len(q_lst))]
        mask_lst = [mask_lst[i][mask_lst[i] == view_cell_type] for i in range(len(mask_lst))]
        print('Visualizing cell type:', view_cell_type)
        multiple_types = False
    elif multiple_types:
        multiple_types = True
        print('Visualizing all cell types')
    else:
        multiple_types = False
        print('Visualizing only cell type')

    # Making sure the polarities are normalized
    p_lst = [p_lst[i]/ np.sqrt(np.sum(p_lst[i] ** 2, axis=1))[:, None] for i in range(len(p_lst))]
    q_lst = [q_lst[i]/ np.sqrt(np.sum(q_lst[i] ** 2, axis=1))[:, None] for i in range(len(q_lst))]

    #if multiple types we partition the polarity lists in 2
    if view_polarity_type == 'ABP':
        if multiple_types:
            p_pos_lst0 = [x_lst[i][mask_lst[i] == 0] + 0.2 * p_lst[i][mask_lst[i] == 0] for i in range(len(x_lst))]
            p_pos_lst1 = [x_lst[i][mask_lst[i] == 1] + 0.2 * p_lst[i][mask_lst[i] == 1] for i in range(len(x_lst))]
        else:
            p_pos_lst0 = [x_lst[i] + 0.2 * p_lst[i] for i in range(len(x_lst))]
            p_pos_lst1 = []
        print('Apicobasal polarity is shown')
    elif view_polarity_type == 'PCP':
        if multiple_types:
            p_pos_lst0 = [x_lst[i][mask_lst[i] == 0] + 0.2 * q_lst[i][mask_lst[i] == 0] for i in range(len(x_lst))]
            p_pos_lst1 = [x_lst[i][mask_lst[i] == 1] + 0.2 * q_lst[i][mask_lst[i] == 1] for i in range(len(x_lst))]
        else:
            p_pos_lst0 = [x_lst[i] + 0.2 * q_lst[i] for i in range(len(x_lst))]
            p_pos_lst1 = []
        print('Planar cell polarity is shown')
    else:
        p_pos_lst0 = []
        p_pos_lst1 = []
        print('No polarity is shown')

    # Make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()                                   

    # Create scatter object and fill in the data
    type0_scatter = visuals.Markers(scaling=True, alpha=10, spherical=True)             #Type0 - green
    type0_polarity_scatter = visuals.Markers(scaling=True, alpha=10, spherical=True)    #Type0 polarity - red
    type1_scatter = visuals.Markers(scaling=True, alpha=10, spherical=True)             #Type1 - blue
    type1_polarity_scatter = visuals.Markers(scaling=True, alpha=10, spherical=True)    #Type1 polarity - yellow


    #Setting the data depending on the number of cell types and polarities
    if view_cell_type == 'all' and multiple_types:
        type0_scatter.set_data(x_lst[0][mask_lst[0] == 0], edge_width=0, face_color='green', size=2.5)
        view.add(type0_scatter)
        if p_pos_lst0:         #if not p_pos_lst0 is empty we add polarity scatter
            type0_polarity_scatter.set_data(p_pos_lst0[0], edge_width=0, face_color='red', size=2.5)
            view.add(type0_polarity_scatter)
        type1_scatter.set_data(x_lst[0][mask_lst[0] == 1], edge_width=0, face_color='blue', size=2.5)
        view.add(type1_scatter)
        if p_pos_lst1:
            type1_polarity_scatter.set_data(p_pos_lst1[0], edge_width=0, face_color='yellow', size=2.5)
            view.add(type1_polarity_scatter)
    else:
        type0_scatter.set_data(x_lst[0], edge_width=0, face_color=(0.1, 1, 1, .5), size=2.5)
        view.add(type0_scatter)
        if p_pos_lst0:         #if not p_pos_lst0 is empty we add polarity scatter
            type0_polarity_scatter.set_data(p_pos_lst0[0], edge_width=0, face_color=(1, 1, 1, .5), size=2.5)
            view.add(type0_polarity_scatter)


    def update(ev):
        """
        Updates the visualization a timestep.

        Parameters:
            ev : vispy.event.Event
                The event that triggered the update.
        Returns:
            None
        """

        global x, iterator
        iterator += 1
        x = x_lst[int(iterator) % len(x_lst)]
        mask = mask_lst[int(iterator) % len(x_lst)]
        if view_cell_type == 'all' and multiple_types:
            type0_scatter.set_data(x[mask == 0], edge_width=0, face_color='green', size=2.5)
            if p_pos_lst0:         #if not p_pos_lst0 is empty we add polarity scatter
                p_pos0 = p_pos_lst0[int(iterator) % len(x_lst)]
                type0_polarity_scatter.set_data(p_pos0, edge_width=0, face_color='red', size=2.5)
            type1_scatter.set_data(x[mask == 1], edge_width=0, face_color='blue', size=2.5)
            if p_pos_lst1:
                p_pos1 = p_pos_lst1[int(iterator) % len(x_lst)]
                type1_polarity_scatter.set_data(p_pos1, edge_width=0, face_color='yellow', size=2.5)
        else:
            type0_scatter.set_data(x, edge_width=0, face_color='green', size=2.5)
            if p_pos_lst0:
                p_pos0 = p_pos_lst0[int(iterator) % len(x_lst)]
                type0_polarity_scatter.set_data(p_pos0, edge_width=0, face_color='red', size=2.5)

    timer = app.Timer(interval=1/60)        # Create a timer that updates the visualization at 60 FPS
    timer.connect(update)                   # Connect the timer to the update function
    timer.start()                           # Start the timer

    @canvas.connect
    def on_key_press(event):
        """
        Function to handle key press events.

        Functionalities:
            space: Pause/Resume the animation
            r: Rewind the animation by 50 frames
            t: Fast forward the animation by 50 frames
            ,: Step back the animation by 1 frames
            .: Step forward the animation by 1 frame
        Parameters:
            event : vispy.event.Event
                The event that triggered the key press.
            
        """
        #Numbers are off by 1 as we also advance the simulation 1 timestep
        global iterator
        if event.text == ' ':
            if timer.running:
                timer.stop()
            else:
                timer.start()
        elif event.text == 'r':
            iterator -= 51
            update(1)
        elif event.text == 't':
            iterator += 49
            update(1)
        elif event.text == ',':
            iterator -= 2
            update(1)
        elif event.text == '.':
            update(1)

    # We want to fly around
    view.camera = view_camera

    # We launch the app
    if sys.flags.interactive != 1:
        vispy.app.run()