#!/usr/bin/env python

"""utils.py: module for helper functions"""

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"

import base64

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+ encoded




