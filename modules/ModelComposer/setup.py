from setuptools import find_packages, setup

setup(
    name='ModelComposerLib',
    packages=find_packages(include=['model_composer', 'model_composer.*']),
    version='0.1.0',
    description='ModelComposerLib',
    author='Philipp Raithel',
    license='MIT',
    package_data={'core': ['serialized_fitting_description.capnp',
                           'serialized_simulation_description.capnp']},
    include_package_data=True,
    install_requires=['datetime', 'pyzmq', 'pycapnp',
                      'pandas', 'matplotlib', 'numpy', 'fabio']
)
