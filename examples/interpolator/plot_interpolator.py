#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT
import numpy as np

import S4

interpolator = S4.NewInterpolator(
    "cubic hermite spline",
    (
        (3.0, (14.2, 32.0)),  # x, and list of y values
        (5.4, (4.6, 10.0)),
        (5.7, (42.7, 20.0)),
        (8.0, (35.2, 40.0)),
    ),
)

xs = np.arange(3, 8, 0.1)

for x in xs:
    y1, y2 = interpolator.Get(x)
    print(x, y1, y2)
