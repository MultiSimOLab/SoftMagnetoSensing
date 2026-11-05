# Gmsh geometry and mesh of the binary sensor (half device, symmetry plane at x = 0): two hard-magnetic
# grippers facing each other across a gap, embedded in an air box. Physical groups name the subdomains
# (Air, MagnetoOmega1-3 for the magnetic problem, MechanoOmega1-3 for the mechanical one), the
# solid–air interface, the Dirichlet boundaries and the loaded faces used in the post-processing.
using Gmsh: Gmsh, gmsh

"File name of the mesh of the configuration (w, e, αgrip) in `dir`."
mesh_file(w, e, αgrip; dir) = joinpath(dir, "MagnetoGrippers_SmallDevice_w_$(w)_e_$(e)_α_$(αgrip).msh")

"""
    gripper_mesh(w, e, αgrip; dir, visualize=false) -> path

Generate the mesh of the sensor with gripper width `w`, gap `e` and loaded-height parameter `αgrip`
and write it to `mesh_file(w, e, αgrip; dir)`. `visualize=true` opens the Gmsh window (blocking).
"""
function gripper_mesh(w, e, αgrip; dir, visualize=false)
   Gmsh.initialize()
   # Geometry (half device, symmetry at x = 0): L1 height of a gripper, L3 gap between the two
   # grippers, L4 half width of the central conductor, e gap of the gripper, w its width, Lcut height of
   # the loaded part; L2, L5, L6, L7 and Lv* are derived
   L1   =  17
   L3   =  15
   L4   =  1.0
   L5   =  4.5-(L4+e)
   L2   =  L1-w
   L6   =  e
   L7   =  w
   Lcut =  (1+αgrip)*w        # height from the gap to the loaded part
   Lv1  =  L1*2
   Lv2  =  (L4+L5)*2


   lc     =  L1/100            # element size in the grippers
   lv     =  L1/5              # element size at the outer boundary of the air
   geo = gmsh.model.geo
   #-----------------------------------
   #  Points
   #-----------------------------------
   #  corners of the air box
   geo.addPoint(0.0, -L3/2-L1-Lv1, 0,  lv, 1)
   geo.addPoint(Lv1, -L3/2-L1-Lv1, 0, lv, 2)
   geo.addPoint(Lv1, L3/2+L1+Lv1, 0, lv, 3)
   geo.addPoint(0.0,  L3/2+L1+Lv1, 0, lv, 4)
   #  bottom gripper
   geo.addPoint(0.0, -L3/2-L1, 0,  lc, 5)
   geo.addPoint(L4, -L3/2-L1, 0,  lc, 6)
   geo.addPoint(L4+L5, -L3/2-L1, 0,  lc, 7)
   geo.addPoint(L4+L5+L6, -L3/2-L1, 0,  lc, 8)
   geo.addPoint(0.0, -L3/2-(L1-L2), 0,  lc, 9)
   geo.addPoint(L4, -L3/2-(L1-L2), 0,  lc, 10)
   geo.addPoint(L4+L5, -L3/2-(L1-L2), 0,  lc, 11)
   geo.addPoint(L4+L5+L6, -L3/2-(L1-L2), 0,  lc, 12)
   geo.addPoint(0.0, -L3/2, 0,  lc, 13)
   geo.addPoint(L4, -L3/2, 0,  lc, 14)
   geo.addPoint(L4+L5, -L3/2, 0,  lc, 15)
   geo.addPoint(L4+L5+L6, -L3/2, 0,  lc, 16)

   #  top gripper
   geo.addPoint(0.0, L3/2, 0,  lc, 17)
   geo.addPoint(L4, L3/2, 0,  lc, 18)
   geo.addPoint(L4+L5, L3/2, 0,  lc, 19)
   geo.addPoint(L4+L5+L6, L3/2, 0,  lc, 20)
   geo.addPoint(0.0, L3/2+(L1-L2), 0,  lc, 21)
   geo.addPoint(L4, L3/2+(L1-L2), 0,  lc, 22)
   geo.addPoint(L4+L5, L3/2+(L1-L2), 0,  lc, 23)
   geo.addPoint(L4+L5+L6, L3/2+(L1-L2), 0,  lc, 24)
   geo.addPoint(0.0, L3/2+L1, 0,  lc, 25)
   geo.addPoint(L4, L3/2+L1, 0,  lc, 26)
   geo.addPoint(L4+L5, L3/2+L1, 0,  lc, 27)
   geo.addPoint(L4+L5+L6, L3/2+L1, 0,  lc, 28)
   #  Points at the horizontal cut y = L3/2+Lcut
   geo.addPoint(0.0, L3/2+Lcut, 0,  lc, 29)
   geo.addPoint(L4, L3/2+Lcut, 0,  lc, 30)
   geo.addPoint(L4+L5, L3/2+Lcut, 0,  lc, 31)
   geo.addPoint(L4+L5+L6, L3/2+Lcut, 0,  lc, 32)
   #  Points at the horizontal cut y = -L3/2-Lcut
   geo.addPoint(0.0, -L3/2-Lcut, 0,  lc, 33)
   geo.addPoint(L4, -L3/2-Lcut, 0,  lc, 34)
   geo.addPoint(L4+L5, -L3/2-Lcut, 0,  lc, 35)
   geo.addPoint(L4+L5+L6, -L3/2-Lcut, 0,  lc, 36)

   #-----------------------------------
   #  Lines
   #-----------------------------------
   geo.addLine(5, 6, 1)
   geo.addLine(6, 7, 2)
   geo.addLine(7, 8, 3)
   geo.addLine(9, 10, 4)
   geo.addLine(10, 11, 5)
   geo.addLine(11, 12, 6)
   geo.addLine(13, 14, 7)
   geo.addLine(14, 15, 8)
   geo.addLine(15, 16, 9)

   #top solid horizontal lines
   geo.addLine(17, 18, 10)
   geo.addLine(18, 19, 11)
   geo.addLine(19, 20, 12)
   geo.addLine(21, 22, 13)
   geo.addLine(22, 23, 14)
   geo.addLine(23, 24, 15)
   geo.addLine(25, 26, 16)
   geo.addLine(26, 27, 17)
   geo.addLine(27, 28, 18)

   #bottom solid vertical lines — inner layer (y=-L3/2-(L1-L2) to y=-L3/2)
   geo.addLine(9, 13, 23)
   geo.addLine(10, 14, 24)
   geo.addLine(11, 15, 25)
   geo.addLine(12, 16, 26)
   #bottom solid vertical lines — lower half of outer layer (y=-L3/2-L1 to y=-L3/2-Lcut)
   geo.addLine(5, 33, 55)
   geo.addLine(6, 34, 56)
   geo.addLine(7, 35, 57)
   geo.addLine(8, 36, 58)
   #bottom solid vertical lines — upper half of outer layer (y=-L3/2-Lcut to y=-L3/2-(L1-L2))
   geo.addLine(33, 9, 59)
   geo.addLine(34, 10, 60)
   geo.addLine(35, 11, 61)
   geo.addLine(36, 12, 62)

   #top solid vertical lines (first layer: y=L3/2 to y=L3/2+(L1-L2))
   geo.addLine(17, 21, 27)
   geo.addLine(18, 22, 28)
   geo.addLine(19, 23, 29)
   geo.addLine(20, 24, 30)

   #top solid vertical lines — lower half of second layer (y=L3/2+(L1-L2) to y=L3/2+Lcut)
   geo.addLine(21, 29, 44)
   geo.addLine(22, 30, 45)
   geo.addLine(23, 31, 46)
   geo.addLine(24, 32, 47)

   #top solid vertical lines — upper half of second layer (y=L3/2+Lcut to y=L3/2+L1)
   geo.addLine(29, 25, 48)
   geo.addLine(30, 26, 49)
   geo.addLine(31, 27, 50)
   geo.addLine(32, 28, 51)

   #horizontal lines at top cut level
   geo.addLine(29, 30, 41)
   geo.addLine(30, 31, 42)
   geo.addLine(31, 32, 43)
   #horizontal lines at bottom cut level
   geo.addLine(33, 34, 52)
   geo.addLine(34, 35, 53)
   geo.addLine(35, 36, 54)

   #horizontal lines in air
   geo.addLine(1, 2, 35)
   geo.addLine(4, 3, 36)
   #vertical lines in air
   geo.addLine(1, 5, 37)
   geo.addLine(13, 17, 38)
   geo.addLine(25, 4, 39)
   geo.addLine(2, 3, 40)
   #-----------------------------------
   #  Curve loops
   #-----------------------------------
   #air (outer layer split: -22 → -62,-58 for 12→36→8)
   geo.addCurveLoop([35,40,-36,-39,16,17,18,-51,-47,-30,-12,-11,-10,-38,7,8,9,-26,-62,-58,-3,-2,-1,-37], 1)
   #bottom solid outer layer — lower halves (usolid_nonzero)
   geo.addCurveLoop([1, 56, -52, -55], 17)
   geo.addCurveLoop([2, 57, -53, -56], 18)
   geo.addCurveLoop([3, 58, -54, -57], 19)
   #bottom solid outer layer — upper halves (no usolid_nonzero tag)
   geo.addCurveLoop([52, 60, -4, -59], 20)
   geo.addCurveLoop([53, 61, -5, -60], 21)
   geo.addCurveLoop([54, 62, -6, -61], 22)
   geo.addCurveLoop([4,24,-7,-23], 5)
   geo.addCurveLoop([5,25,-8,-24], 6)
   geo.addCurveLoop([6,26,-9,-25], 7)
   #top solid first layer
   geo.addCurveLoop([10,28,-13,-27], 8)
   geo.addCurveLoop([11,29,-14,-28], 9)
   geo.addCurveLoop([12,30,-15,-29], 10)
   #top solid second layer — upper halves (usolid_nonzero)
   geo.addCurveLoop([41,49,-16,-48], 11)
   geo.addCurveLoop([42,50,-17,-49], 12)
   geo.addCurveLoop([43,51,-18,-50], 13)
   #top solid second layer — lower halves (no usolid_nonzero tag)
   geo.addCurveLoop([13,45,-41,-44], 14)
   geo.addCurveLoop([14,46,-42,-45], 15)
   geo.addCurveLoop([15,47,-43,-46], 16)
   #-----------------------------------
   # Plane surfaces
   #-----------------------------------
   geo.addPlaneSurface([1], 1)
   geo.addPlaneSurface([17], 17)   # lower half — usolid_nonzero
   geo.addPlaneSurface([18], 18)   # lower half — usolid_nonzero
   geo.addPlaneSurface([19], 19)   # lower half — usolid_nonzero
   geo.addPlaneSurface([20], 20)   # upper half — no usolid_nonzero
   geo.addPlaneSurface([21], 21)   # upper half — no usolid_nonzero
   geo.addPlaneSurface([22], 22)   # upper half — no usolid_nonzero
   geo.addPlaneSurface([5], 5)
   geo.addPlaneSurface([6], 6)
   geo.addPlaneSurface([7], 7)
   geo.addPlaneSurface([8], 8)
   geo.addPlaneSurface([9], 9)
   geo.addPlaneSurface([10], 10)
   geo.addPlaneSurface([11], 11)   # upper half — usolid_nonzero
   geo.addPlaneSurface([12], 12)   # upper half — usolid_nonzero
   geo.addPlaneSurface([13], 13)   # upper half — usolid_nonzero
   geo.addPlaneSurface([14], 14)   # lower half — no usolid_nonzero
   geo.addPlaneSurface([15], 15)   # lower half — no usolid_nonzero
   geo.addPlaneSurface([16], 16)   # lower half — no usolid_nonzero
   gmsh.model.geo.synchronize()
   #-----------------------------------
   # Meshing and physical tags
   #-----------------------------------
   gmsh.model.addPhysicalGroup(0, [16,15,14,13,20,19,18,17], 1, "Interface")
   gmsh.model.addPhysicalGroup(1, [7,8,9,26,10,11,12,30], 1, "Interface")

   gmsh.model.addPhysicalGroup(0, [1,2,3,4,5,6,7,8,36], 2, "uair_fixed")
   gmsh.model.addPhysicalGroup(1, [35,40,36,1,2,3,58], 2, "uair_fixed")

   gmsh.model.addPhysicalGroup(0, [32,25,26,27,28], 3, "uair_nonzero")
   gmsh.model.addPhysicalGroup(1, [16,17,18,51], 3, "uair_nonzero")

   gmsh.model.addPhysicalGroup(1, [39,38,37], 4, "uair_symmetry")

   gmsh.model.addPhysicalGroup(0, [13,17], 5, "usolid_symmetry")
   gmsh.model.addPhysicalGroup(1, [23,27,44,59], 5, "usolid_symmetry")

   gmsh.model.addPhysicalGroup(0, [5,6,7,8,33,34,35,36], 6, "usolid_fixed")
   gmsh.model.addPhysicalGroup(1, [1,2,3,52,53,54,55,56,57,58], 6, "usolid_fixed")
   gmsh.model.addPhysicalGroup(2, [17,18,19], 6, "usolid_fixed")

   # usolid_nonzero: top upper halves (11,12,13) + bottom lower halves (17,18,19)
   gmsh.model.addPhysicalGroup(0, [29,30,31,32,25,26,27,28], 7, "usolid_nonzero")
   gmsh.model.addPhysicalGroup(1, [41,42,43,16,17,18,48,49,50,51], 7, "usolid_nonzero")
   gmsh.model.addPhysicalGroup(2, [11,12,13], 7, "usolid_nonzero")

   gmsh.model.addPhysicalGroup(2, [1], 8, "Air")
   # MagnetoOmega1: cols 2&3 of both solids (bottom outer split into 18,19,21,22)
   gmsh.model.addPhysicalGroup(2, [6,9,12,13,15,16,18,19,21,22], 9, "MagnetoOmega1")
   # MagnetoOmega2: col 1 of both solids (bottom outer split into 17,20)
   gmsh.model.addPhysicalGroup(2, [5,8,11,14,17,20], 10, "MagnetoOmega2")
   gmsh.model.addPhysicalGroup(2, [7,10], 11, "MagnetoOmega3")

   gmsh.model.addPhysicalGroup(2, [9,15,16,6,21,22], 12, "MechanoOmega1")
   gmsh.model.addPhysicalGroup(2, [5,8,14,20], 13, "MechanoOmega2")
   gmsh.model.addPhysicalGroup(2, [7,10], 14, "MechanoOmega3")

   gmsh.model.addPhysicalGroup(0, [1], 15, "phi_Dirichlet")

   gmsh.model.addPhysicalGroup(0, [29,30,31,32], 16, "top_traction_boundary")
   gmsh.model.addPhysicalGroup(1, [41,42,43], 16, "top_traction_boundary")

   gmsh.model.addPhysicalGroup(0, [33,34,35,36], 17, "bottom_traction_boundary")
   gmsh.model.addPhysicalGroup(1, [52,53,54], 17, "bottom_traction_boundary")

   gmsh.model.addPhysicalGroup(0, [17,18,19,20], 18, "top_contact_boundary")
   gmsh.model.addPhysicalGroup(1, [10,11,12], 18, "top_contact_boundary")

   gmsh.model.addPhysicalGroup(0, [13,14,15,16], 19, "bottom_contact_boundary")
   gmsh.model.addPhysicalGroup(1, [7,8,9], 19, "bottom_contact_boundary")

   gmsh.model.mesh.generate()

   #-----------------------------------
   # Saving
   #-----------------------------------
   gmsh.model.geo.synchronize()
   output_file = mesh_file(w, e, αgrip; dir)
   gmsh.write(output_file)
   visualize && gmsh.fltk.run()
   Gmsh.finalize()
   return output_file
end
