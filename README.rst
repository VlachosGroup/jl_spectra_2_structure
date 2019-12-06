jl_spectra_2_structure
======================
Methods for converting spectra to structure and solving the materials gap

This is documentation for https://github.com/JLans/jl_spectra_2_structure.
jl_spectra_2_structure trains neural network models to learn quantitative descriptions of surface coordination
for either extended surfaces or nanoparticles. The model is trained on complex synthetic IR data (secondary data).
A class is provided for generating complex (secondary) data from low coverage dft frequency and intensity data (primary data).
Another class is provided for generating the primary data from forces on atoms and integrated charges computed by vasp and chargemol respectively.

Documentation
-------------

See our `documentation page`_ for examples, equations used, and docstrings.

Developer
---------

-  Joshua Lansford (lansford@udel.edu)

Dependencies
------------

-  Python3
-  `Atomic Simulation Environment`_: Used for I/O operations and for visualiztion
-  `Numpy`_: Used for vector and matrix operations
-  `Pandas`_: Used to import data from concatenated chargemol files and summing intensities from similar modes
-  `SciPy`_: Used for fitting coverage scaling relations
-  `Matplotlib`_: Used for plotting data
-  `scikit-learn`_: Used for batching data during neural network training and k-means clustering
-  `JSON_tricks`_: Used for reading and writing neural network and cross validation parameters
-  `Imbalanced-learn`_: Used for balancing the primary data before generating secondary data
-  `StatsModels`_: Used to get descriptive statistics of the coverage scaling parameters
-  `Pillow`_: Used for writing jpeg files
-  `uuid`_: Used forgetting unique identifiers when writing cross validation results during mpi runs

License
-------

This project is licensed under the MIT License - see the `LICENSE.md`_
file for details.

.. _`documentation page`: https://jlans.github.io/jl_spectra_2_structure/
.. _Atomic Simulation Environment: https://wiki.fysik.dtu.dk/ase/
.. _Numpy: http://www.numpy.org/
.. _Pandas: https://pandas.pydata.org/
.. _SciPy: https://www.scipy.org/
.. _Matplotlib: https://matplotlib.org/
.. _Imbalanced-learn: https://imbalanced-learn.readthedocs.io/en/stable/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _StatsModels: https://www.statsmodels.org/stable/index.html
.. _Pillow: https://pillow.readthedocs.io/en/stable/
.. _JSON_tricks: https://json-tricks.readthedocs.io/en/latest/
.. _uuid: https://docs.python.org/3/library/uuid.html
