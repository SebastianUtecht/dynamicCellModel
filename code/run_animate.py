from animate import *

animate('bounds/data.npy',          #Path to data
        view_cell_type='all',       #Type of cell to view. This is only relevant if multiple cell types are present. Arguments are: ['all', 0, 1]
        view_polarity_type='ABP',   #Type of polarity to view. Arguments are: ['ABP', 'PCP', 'none']
        view_camera='turntable')    #Type of camera view to use. Arguments are: ['fly', 'turntable']