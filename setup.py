import setuptools

setuptools.setup(
    name='gym_jass',
    version='0.1.5',
    install_requires=['gym>=0.10.8', 'schieber>=0.1.6'],
    author="Joel Niklaus",
    author_email="me@joelniklaus.ch",
    description="A gym environment for the schieber variant of the Swiss card game Jassen",
    license=open('LICENSE', "r").read(),
    long_description=open('README.md', "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoelNiklaus/gym-jass",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
