from setuptools import setup, find_namespace_packages

requirements = [
"numpy",
"pandas",
"moviepy",
"tqdm",
"python-opencv",
"matplotlib",
"seaborn",
"sklearn",
"scipy",
"psychopy",
"pypylon",
]


setup(
    name="fiberphotometry",
    version="0.0.0.1",
    author_email="federicoclaudi@protonmail.com",
    description="bunch of utility functions to analyse fiberphotometry data",
    packages=find_namespace_packages(exclude=()),
    include_package_data=True,
    install_requires=requirements,
    url="https://github.com/BrancoLab/Fiberphotometry",
    author="Federico Claudi, Yu Lin Tan",
    zip_safe=False,
)
