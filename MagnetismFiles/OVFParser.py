import numpy as np
import re
import matplotlib.pyplot as plt
import io
from scipy.interpolate import RegularGridInterpolator


class OVF:
    
    '''Key-value pairs relating the name of a property to both the regular expression
    preceeding its value in the OVF file and the property's desired data type.
    '''
    field_vars = {"title" : ("Title", str),
                  "meshunit" : ("meshunit", str),
                  "xmin" : ("xmin", float),
                  "ymin" : ("ymin", float),
                  "zmin" : ("zmin", float),
                  "xmax" : ("xmax", float),
                  "ymax" : ("ymax", float),
                  "zmax" : ("zmax", float),
                  "value_dim" : ("valuedim", int),
                  "value_labels" : ("valuelabels", str),
                  "value_units" : ("valueunits", str),
                  "xbase" : ("xbase", float),
                  "ybase" : ("ybase", float),
                  "zbase" : ("zbase", float),
                  "xnodes" : ("xnodes", int),
                  "ynodes" : ("ynodes", int),
                  "znodes" : ("znodes", int),
                  "xstepsize" : ("xstepsize", float),
                  "ystepsize" : ("ystepsize", float),
                  "zstepsize" : ("zstepsize", float)}
    
    '''
    Flexibility in creating an instance.
    Can pass file path or string containing data
    from file.
    '''
    def __init__(self, *dat, **kwargs):
        if ('file' in kwargs.keys()):
            with open(kwargs['file'], 'r') as f:
                self.txt = f.read()
        elif ('data' in kwargs.keys()):
            self.txt = kwargs['data']
        else:
            self.txt = dat[0]
        self.get_contents()
        

    def get_contents(self):
        regex = r"# _-_:\s*(.*?)\s*\n"
        start_data = re.search(r"Begin: Data Text\n", self.txt).span()[1]
        for k in self.field_vars:
            expr, typ = self.field_vars[k]
            query = regex.replace("_-_", expr)
            result = re.search(query, self.txt[:start_data])
            if result:
                setattr(self, k, typ(result.group(1)))
        field_list = np.loadtxt(io.StringIO(self.txt[start_data:]))
        try:
            self.field = field_list
            x_field, y_field, z_field = field_list.T
            new_shape = (self.znodes, self.ynodes, self.xnodes)
            self.xfield = np.reshape(x_field, new_shape)
            self.yfield = np.reshape(y_field, new_shape)
            self.zfield = np.reshape(z_field, new_shape)
        except:
            print("Error unpacking OVF vector field.")
    
    def get(self, **kwargs):
        x_space = np.arange(self.xmin, self.xmax, self.xstepsize) + self.xbase
        y_space = np.arange(self.ymin, self.ymax, self.ystepsize) + self.ybase
        z_space = np.arange(self.zmin, self.zmax, self.zstepsize) + self.zbase
        x_val = x_space
        y_val = y_space
        z_val = self.zmax - 0.5 * self.zstepsize
        if (('r' in kwargs.keys() and 'theta' in kwargs.keys()) and\
            ('x' in kwargs.keys() or 'y' in kwargs.keys()) and\
            ('theta' in kwargs.keys() or 'r' in kwargs.keys())):
            print("Error: Either \n(i) 'theta' (or 'r') can not be constant if\
                    x (and/or y) is also constant. \n OR \
                    \n(ii) 'theta' and 'r' can not be constant.")
            return
        if ('z' in kwargs.keys()):
            z_val = kwargs['z']
        if type(z_val) not in [list, np.ndarray]:
            if ('nz' in kwargs.keys()):
                nz = kwargs[nz]
            else:
                nz = 1
            z_val = np.zeros(nz) + z_val
        if ('x' in kwargs.keys()):
            x_val = kwargs['x']
        if type(x_val) not in [list, np.ndarray]:
            if ('nx' in kwargs.keys()):
                nx = kwargs[nx]
            else:
                nx = self.xnodes
            x_val = np.zeros(nx) + x_val
        if ('y' in kwargs.keys()):
            y_val = kwargs['y']
        if type(y_val) not in [list, np.ndarray]:
            if ('ny' in kwargs.keys()):
                ny = kwargs[ny]
            else:
                ny = self.ynodes
            y_val = np.zeros(ny) + y_val
        if ('r' in kwargs.keys()):
            if 'ntheta' in kwargs.keys():
                ntheta = kwargs['ntheta']
            else:
                ntheta = np.ceil(np.sqrt(self.xnodes * self.ynodes))
            theta = np.linspace(0, 2 * np.pi, ntheta)
            x_val, y_val = kwargs['r'] * (np.cos(theta), np.sin(theta))
        X, Y, Z = np.meshgrid(x_val, y_val, z_val)
        if ("comp" in kwargs.keys()):
            components = kwargs["comp"]
        else:
            components = ["x", "y", "z"]
        if type(components) != list:
            interp = RegularGridInterpolator((z_space, y_space, x_space),\
                                        getattr(self, components+"field"))
            return interp(np.array([Z, Y, X]).T)
        else:
            to_return = []
            for c in components:
                interp = RegularGridInterpolator((z_space, y_space, x_space),\
                                        getattr(self, c+"field"))
                to_return.append(interp(np.array([Z, Y, X]).T))
        return to_return
