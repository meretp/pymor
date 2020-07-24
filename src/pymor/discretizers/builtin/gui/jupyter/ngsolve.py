import math
import numpy as np
from matplotlib.cm import get_cmap
import os
from ipywidgets import DOMWidget, register
from traitlets import Unicode

import ngsolve as ngs
import ngsolve.webgui

# the build script fills the contents of the variables below
render_js_code = ngsolve.webgui.render_js_code
widgets_version = ngsolve.webgui.widgets_version

try:
    __IPYTHON__
    _IN_IPYTHON = True
except NameError:
    _IN_IPYTHON = False

try:
    import google.colab

    _IN_GOOGLE_COLAB = True
except ImportError:
    _IN_GOOGLE_COLAB = False

#           <script src="https://cdn.jsdelivr.net/npm/three@0.115.0/build/three.min.js"></script>
#           <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.js"></script>
#           <script src="https://cdnjs.cloudflare.com/ajax/libs/stats.js/r16/Stats.min.js"></script>
#
html_template = """
<!DOCTYPE html>
<html>
    <head>
        <title>NGSolve WebGUI</title>
        <meta name='viewport' content='width=device-width, user-scalable=no'/>
        <style>
            body{
                margin:0;
                overflow:hidden;
            }
            canvas{
                cursor:grab;
                cursor:-webkit-grab;
                cursor:-moz-grab;
            }
            canvas:active{
                cursor:grabbing;
                cursor:-webkit-grabbing;
                cursor:-moz-grabbing;
            }
        </style>
    </head>
    <body>
          <script src="https://requirejs.org/docs/release/2.3.6/minified/require.js"></script>
          <script>
            {render}

            require(["ngsolve_jupyter_widgets"], ngs=>
            {
                let scene = new ngs.Scene();
                scene.init(document.body, render_data);
            });
          </script>
    </body>
</html>
"""


class WebGLScene:
    def __init__(self, cf, mesh, order, min_, max_, draw_vol, draw_surf, autoscale, deformation, interpolate_multidim,
                 animate):
        from IPython.display import display, Javascript
        import threading
        self.cf = cf
        self.mesh = mesh
        self.order = order
        self.min = min_
        self.max = max_
        self.draw_vol = draw_vol
        self.draw_surf = draw_surf
        self.autoscale = autoscale
        self.interpolate_multidim = interpolate_multidim
        self.animate = animate

        self.deformation = deformation

    def GetData(self, set_minmax=True):
        import json
        d = BuildRenderData(self.mesh, self.cf, self.order)

        if set_minmax:
            if self.min is not None:
                d['funcmin'] = self.min
            if self.max is not None:
                d['funcmax'] = self.max
            d['autoscale'] = self.autoscale

        return d

    def GenerateHTML(self, filename=None):
        import json
        d = self.GetData()

        data = json.dumps(d)

        html = html_template.replace('{data}', data)
        jscode = "var render_data = {}\n".format(data) + render_js_code
        html = html.replace('{render}', jscode)

        if filename is not None:
            open(filename, 'w').write(html)
        return html

    def Draw(self):
        self.widget = NGSWebGuiWidget()
        d = self.GetData()
        self.widget.value = d
        display(self.widget)

    def Redraw(self):
        d = self.GetData(set_minmax=False)
        self.widget.value = d

    def __repr__(self):
        return ""


bezier_trig_trafos = {}  # cache trafos for different orders

timer = ngs.Timer("BuildRenderData")
timer2 = ngs.Timer("edges")
timermult = ngs.Timer("timer2 - mult")
timer3 = ngs.Timer("els")
timer3Bvals = ngs.Timer("timer3, bezier")
timer3minmax = ngs.Timer("els minmax")
timer2list = ngs.Timer("timer2 - make list")
timer3list = ngs.Timer("timer3 - make list")
timer4 = ngs.Timer("func")


def Draw(grid, u, name='function', order=2, min=None, max=None, draw_vol=True, draw_surf=True, autoscale=True,
         deformation=False, interpolate_multidim=False, animate=False):
    scene = WebGLScene(u, grid, order, min_=min, max_=max, draw_vol=draw_vol, draw_surf=draw_surf, autoscale=autoscale,
                       deformation=deformation, interpolate_multidim=interpolate_multidim, animate=animate)
    # render scene using widgets.DOMWidget
    scene.Draw()
    return scene


@register
class NGSWebGuiWidget(DOMWidget):
    from traitlets import Dict, Unicode
    _view_name = Unicode('NGSolveView').tag(sync=True)
    _view_module = Unicode('ngsolve_jupyter_widgets').tag(sync=True)
    _view_module_version = Unicode(widgets_version).tag(sync=True)
    value = Dict({"ngsolve_version": '0.0.0'}).tag(sync=True)


tencode = ngs.Timer("encode")


def encodeData(array):
    from base64 import b64encode
    tencode.Start()
    values = np.array(array.flatten(), dtype=np.float32)
    res = b64encode(values).decode("ascii")
    tencode.Stop()
    return res


_jupyter_lab_extension_path = os.path.join(os.path.dirname(ngs.__file__), "labextension")


def howtoInstallJupyterLabextension():
    import ngsolve, os
    d = os.path.dirname(ngsolve.__file__)
    labdir = os.path.join(d, "labextension")
    print("""# To install jupyter lab extension:
jupyter labextension install --clean {labdir}
""".format(labdir=_jupyter_lab_extension_path))


from pymor.core import config
from pymor.discretizers.builtin.grids.referenceelements import triangle, square
from pymor.discretizers.builtin.grids.constructions import flatten_grid
from pymor.vectorarrays.interface import VectorArray


def BuildRenderData(grid, u, order=1):
    d = {}
    d['ngsolve_version'] = ngs.__version__
    d['mesh_dim'] = 2

    d['order2d'] = order

    d['draw_vol'] = False
    d['draw_surf'] = True

    d['show_wireframe'] = False
    d['show_mesh'] = False
    d['funcdim'] = 1
    if order > 0:
        og = order
        d['show_wireframe'] = True
        d['show_mesh'] = True
        timer2.Start()

        timer3Bvals.Start()

        # transform point-values to Bernsteinbasis
        def Binomial(n, i):
            return math.factorial(n) / math.factorial(i) / math.factorial(n - i)

        def Bernstein(x, i, n):
            return Binomial(n, i) * x ** i * (1 - x) ** (n - i)

        Bvals = ngs.Matrix(og + 1, og + 1)
        for i in range(og + 1):
            for j in range(og + 1):
                Bvals[i, j] = Bernstein(i / og, j, og)
        iBvals = Bvals.I
        timer3Bvals.Stop()

        Bezier_points = []

        # TODO: Quads
        #         ipts = [(i/og,0) for i in range(og+1)] + [(0, i/og) for i in range(og+1)] + [(i/og,1.0-i/og) for i in range(og+1)]
        #         ir_trig = ngs.IntegrationRule(ipts, [0,]*len(ipts))
        #         ipts = [(i/og,0) for i in range(og+1)] + [(0, i/og) for i in range(og+1)] + [(i/og,1.0) for i in range(og+1)] + [(1.0, i/og) for i in range(og+1)]
        #         ir_quad = ngs.IntegrationRule(ipts, [0,]*len(ipts))

        #         vb = [ngs.VOL, ngs.BND][mesh.dim-2]
        #         cf = func1 if draw_surf else func0
        #         pts = mesh.MapToAllElements({ngs.ET.TRIG: ir_trig, ngs.ET.QUAD: ir_quad}, vb)
        #         pmat = cf(pts)

        subentities, coordinates, entity_map = flatten_grid(grid)

        codim = 2
        if grid.reference_element == triangle:
            if codim == 2:
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
                indices = subentities
            else:
                vertices = np.zeros((len(subentities) * 3, 3))
                VERTEX_POS = coordinates[subentities]
                vertices[:, 0:2] = VERTEX_POS.reshape((-1, 2))
                indices = np.arange(len(subentities) * 3, dtype=np.uint32)
        else:
            if codim == 2:
                vertices = np.zeros((len(coordinates), 3))
                vertices[:, :-1] = coordinates
                indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
            else:
                num_entities = len(subentities)
                vertices = np.zeros((num_entities * 6, 3))
                VERTEX_POS = coordinates[subentities]
                vertices[0:num_entities * 3, 0:2] = VERTEX_POS[:, 0:3, :].reshape((-1, 2))
                vertices[num_entities * 3:, 0:2] = VERTEX_POS[:, [0, 2, 3], :].reshape((-1, 2))
                indices = np.arange(len(subentities) * 6, dtype=np.uint32)

        #         print(vertices.shape)
        #         print(indices.shape)
        # todo: quads
        ne = len(indices)

        pmat = np.zeros((3 * ne, 2, 4))
        for i in range(ne):
            for j in range(3):
                pmat[3 * i + j][0][:3] = vertices[indices[i][j]]
                pmat[3 * i + j][1][:3] = vertices[indices[i][(j + 1) % 3]]

        timermult.Start()
        # pmat = pmat.reshape(-1, og+1, 4)
        #         print('pmat', pmat)
        BezierPnts = np.tensordot(iBvals.NumPy(), pmat, axes=(1, 1))
        #         print(BezierPnts)
        timermult.Stop()

        timer2list.Start()
        for i in range(og + 1):
            Bezier_points.append(encodeData(BezierPnts[i]))
        timer2list.Stop()

        d['Bezier_points'] = Bezier_points
        d['edges'] = Bezier_points

        ndtrig = int((og + 1) * (og + 2) / 2)

        if og in bezier_trig_trafos.keys():
            iBvals_trig = bezier_trig_trafos[og]
        else:
            def BernsteinTrig(x, y, i, j, n):
                return math.factorial(n) / math.factorial(i) / math.factorial(j) / math.factorial(n - i - j) \
                       * x ** i * y ** j * (1 - x - y) ** (n - i - j)

            Bvals = ngs.Matrix(ndtrig, ndtrig)
            ii = 0
            for ix in range(og + 1):
                for iy in range(og + 1 - ix):
                    jj = 0
                    for jx in range(og + 1):
                        for jy in range(og + 1 - jx):
                            Bvals[ii, jj] = BernsteinTrig(ix / og, iy / og, jx, jy, og)
                            jj += 1
                    ii += 1
            iBvals_trig = Bvals.I
            bezier_trig_trafos[og] = iBvals_trig

        # Bezier_points = [ [] for i in range(ndtrig) ]
        Bezier_points = []
        values = u.to_numpy()[0][entity_map]
        #         values = u[0][entity_map]
        pmat = np.zeros((ne, 3, 4))
        for i in range(ne):
            for j in range(3):
                pmat[i][j][:3] = vertices[indices[i][j]]
                pmat[i][j][3] = values[indices[i][j]]
        funcmin = np.min(pmat[:, :, 3])
        funcmax = np.max(pmat[:, :, 3])

        #         pmat = pmat.reshape(-1, len(ir_trig), 4)
        BezierPnts = np.tensordot(iBvals_trig.NumPy(), pmat, axes=(1, 1))

        timer3list.Start()
        for i in range(ndtrig):
            Bezier_points.append(encodeData(BezierPnts[i]))
        timer3list.Stop()

        d['Bezier_trig_points'] = Bezier_points
        d['mesh_center'] = (0, 0, 0)
        d['mesh_radius'] = 1.0
        timer3.Stop()

    d['funcmin'] = funcmin
    d['funcmax'] = funcmax
    #     print(d)
    return d
# BuildRenderData(data['grid'], U, order=1)


def visualize_ngsolve(grid, U, bounding_box=([0, 0], [1, 1]), codim=2, title=None, legend=None,
                    separate_colorbars=False, rescale_colorbars=False, columns=2,
         color_map=get_cmap('viridis')):
    if isinstance(U, tuple):
        raise NotImplementedError('tuples of VectorArrays cannot yet be visualized with the ngsolve backend')
    return Draw(grid, U, order=1)