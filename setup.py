#!/usr/bin/env python


from setuptools import setup
from pkg_resources import DistributionNotFound, get_distribution, parse_version


def get_dist(pkgname, version):
    try:
        out = get_distribution(pkgname)
        v_ok = parse_version(out.version) >= parse_version(version)
        if v_ok:
            print('Requirement already satisfied: tensorflow>=1.14 in ', out.location, '(from differentiable-filters==1.0.1) (', out.version, ')', file=sys.stderr)
        return v_ok
    except DistributionNotFound:
        return False


install_deps = ['matplotlib', 'tensorflow-probability>=0.7.0', 'pyaml', 'opencv-python']

if not get_dist('tensorflow', '1.14') and not get_dist('tensorflow_gpu', '1.14'):
    install_deps.append('tensorflow')


with open("README.md", "r") as fh:
    long_description = fh.read()

print(setuptools.find_packages())

setuptools.setup(
    name="differentiable_filters", 
    version="1.0.1",
    author='Alina Kloss, MPI-IS Tuebingen, Autonomous Motion',
    author_email='alina.kloss@yahoo.de',
    description="TensorFlow code for differentiable filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/akloss/differentiable_filters',
    packages=setuptools.find_packages(),
    entry_points = {
        'console_scripts': ['run_df_filter_experiment=differentiable_filters.paper_training_code.run_experiment:main',
                            'run_df_example=differentiable_filters.example_training_code.run_example:main',
			    'create_disc_tracking_dataset=differentiable_filters.data.create_disc_tracking_dataset:main',
			    'create_kitti_dataset=differentiable_filters.data.create_kitti_dataset:main'],
    },
    install_requires=install_deps,
    python_requires='>=3.6',
)

