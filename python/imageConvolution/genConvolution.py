from read_xyz import read_xyz
from STEM_mod import STEM
import matplotlib.pyplot as plt

probe_step=.088052 #angstroms

resolution = 1/probe_step
HWHM = 0.5#.9/probe_step/2
print(resolution)
dimensions = [58.4749,59.0436]
parameters = {'HWHM':HWHM,'dimensions':dimensions,'resolution':resolution}

generator = STEM(parameters=parameters)

#get atoms object using ASE and pre defined
atoms = read_xyz('FCC_Cu_111_350nm.xyz',format='xyz')

image = generator.get_image(atoms)

plt.imshow(image)
plt.show()