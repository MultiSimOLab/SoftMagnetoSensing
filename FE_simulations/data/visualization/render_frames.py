"""Render the frames of the deformation animation of one configuration with ParaView.

    pvbatch render_frames.py <vtk folder with Results.pvd> <output folder> [every]

Every frame shows the deformed device (mirrored about the symmetry plane x = 0) with the magnetic field
lines of the spatial field h in the air, the air coloured by |h| (log scale) and the grippers coloured by
material: soft part (light grey), hard-magnetic material of the reference concentration (black) and of the
study concentration (dark grey). `every` renders one out of every `every` load steps (default 1).
"""
import os, sys, glob, re
from paraview.simple import *
import numpy as np
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy

vtkdir, outdir = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
os.makedirs(outdir, exist_ok=True)

# Load factor of every step from the collection file
pvd = open(os.path.join(vtkdir, "Results.pvd")).read()
steps = [(float(t), f) for t, f in re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', pvd)]
steps.sort()
steps = steps[::every] + ([steps[-1]] if (len(steps) - 1) % every else [])
T0 = 0.1

W, H = 1040, 1180
view = CreateView("RenderView")
view.ViewSize = [W, H]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.CameraParallelProjection = 1
view.CameraPosition = [0, 0, 100]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewUp = [0, 1, 0]
view.CameraParallelScale = 33.0

reader = XMLUnstructuredGridReader(FileName=[os.path.join(vtkdir, steps[0][1])])
# P1 fields are written per cell: merge the coincident nodes so that the field lines cross the cells
merged = CleantoGrid(Input=reader)
half = WarpByVector(Input=merged, Vectors=["POINTS", "u"])     # deformed configuration (half model, x >= 0)

air = Threshold(Input=half, Scalars=["CELLS", "region"], LowerThreshold=0, UpperThreshold=0)
solids = [Threshold(Input=half, Scalars=["CELLS", "region"], LowerThreshold=r, UpperThreshold=r) for r in (1, 2, 3)]
# seeds of the field lines: a uniform grid over the visible part of the air
seeds = Plane(Origin=[0.05, -34.0, 0.0], Point1=[30.0, -34.0, 0.0], Point2=[0.05, 34.0, 0.0],
              XResolution=11, YResolution=26)
lines = StreamTracerWithCustomSource(Input=half, SeedSource=seeds, Vectors=["POINTS", "h"],
                                     IntegrationDirection="BOTH", MaximumStreamlineLength=200.0)

def placed(src, z, mirror):
    """`src` lifted to the layer z (draw order) and, with `mirror`, reflected about x = 0 (full device)."""
    t = Transform(Input=src)
    t.Transform.Translate = [0, 0, z]
    t.Transform.Scale = [-1 if mirror else 1, 1, 1]
    return t

# |h| in the air, log colour scale with a range fixed on the fully magnetised state
hlut = GetColorTransferFunction("|h|")
hlut.ApplyPreset("Jet", True)
hlut.UseLogScale = 1
colors = [[0.78, 0.77, 0.73], [0.04, 0.04, 0.04], [0.37, 0.37, 0.35]]
for mirror in (False, True):
    d = Show(placed(air, 0.0, mirror), view); ColorBy(d, ("POINTS", "|h|")); d.LookupTable = hlut
    d.Ambient, d.Diffuse = 1.0, 0.0
    d = Show(placed(lines, 0.01, mirror), view); d.ColorArrayName = ["POINTS", ""]
    d.AmbientColor = d.DiffuseColor = [1, 1, 1]; d.LineWidth = 1.0; d.Ambient, d.Diffuse = 1.0, 0.0
    for so, c in zip(solids, colors):
        d = Show(placed(so, 0.02, mirror), view); d.ColorArrayName = ["POINTS", ""]
        d.AmbientColor = d.DiffuseColor = c; d.Ambient, d.Diffuse = 1.0, 0.0

text = Text(); tdisp = Show(text, view)
# camera: the device and the air around it (set after the first Show, which resets it)
view.CameraPosition = [0, 0, 100]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewUp = [0, 1, 0]
view.CameraParallelScale = 33.0
tdisp.FontSize = 24; tdisp.WindowLocation = "Upper Left Corner"; tdisp.Color = [1, 1, 1]; tdisp.Bold = 1; tdisp.Shadow = 1

def magnitude_range():
    a = servermanager.Fetch(air)
    v = vtk_to_numpy(a.GetPointData().GetArray("|h|"))
    v = v[v > 0]
    return float(np.percentile(v, 50)), float(np.percentile(v, 99.8))

# colour range: first step at or beyond the end of the magnetisation ramp
iref = next((i for i, (t, _) in enumerate(steps) if t >= T0 - 1e-9), len(steps) - 1)
reader.FileName = [os.path.join(vtkdir, steps[iref][1])]
lo, hi = magnitude_range()
hlut.RescaleTransferFunction(lo, hi)
GetOpacityTransferFunction("|h|").RescaleTransferFunction(lo, hi)
hlut.AutomaticRescaleRangeMode = "Never"

for k, (t, f) in enumerate(steps):
    reader.FileName = [os.path.join(vtkdir, f)]
    phase = "magnetisation" if t <= T0 + 1e-12 else "compression"
    text.Text = f"load factor {t:.3f}   {phase}"
    Render(view)
    view.CameraPosition, view.CameraFocalPoint, view.CameraViewUp = [0, 0, 100], [0, 0, 0], [0, 1, 0]
    view.CameraParallelScale = 33.0
    SaveScreenshot(os.path.join(outdir, f"frame_{k:04d}.png"), view, ImageResolution=[W, H])
print("frames:", len(steps))
