import numpy as np

filters = [
    # 1. local vertical texture
    {"filter": {"size": (17,17), "sigma": 5, "theta": np.pi/2,
                "lambd": 12, "psi": 0, "gamma": 1.0},
     "stride": (6, 12), "padding": "default", "start_position": (8, 8)},

    # 2. horizontal/angular texture
    {"filter": {"size": (17,33), "sigma": 5, "theta": 0,
                "lambd": 12, "psi": 0, "gamma": 0.5},
     "stride": (6, 12), "padding": "default", "start_position": (8, 8)},

    # 3. medium diagonal -
    {"filter": {"size": (33,17), "sigma": 5, "theta": -np.pi/4,
                "lambd": 16, "psi": 0, "gamma": 0.5},
     "stride": (12, 6), "padding": "default", "start_position": (8, 8)},

    # 4. medium diagonal +
    {"filter": {"size": (33,17), "sigma": 5, "theta": np.pi/4,
                "lambd": 16, "psi": 0, "gamma": 0.5},
     "stride": (12, 6), "padding": "default", "start_position": (8, 8)},

    # replace high-frequency vertical filter with a safer coarse diagonal
    {"filter": {"size": (65,17), "sigma": 6, "theta": -np.pi/4,
                "lambd": 24, "psi": 0, "gamma": 0.25},
     "stride": (24, 6), "padding": "default", "start_position": (8, 8)}
]
