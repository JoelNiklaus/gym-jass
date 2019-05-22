import setuptools

setuptools.setup(
    name='gym_jass',
    version='0.1.19',
    install_requires=['gym>=0.10.8', 'schieber>=0.1.8'],
    author="Joel Niklaus",
    author_email="me@joelniklaus.ch",
    description="A gym environment for the Schieber variant of the Swiss card game Jass",
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
