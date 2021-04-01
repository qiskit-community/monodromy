# I'd have preferred a setup.cfg, but `pip -e` rejects it.

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="monodromy", # Replace with your own username
    version="0.0.1",
    author="Eric Peterson",
    author_email="Eric.Peterson@ibm.com",
    description="Computations in the monodromy polytope for quantum gate sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/IBM-Q-Software/monodromy",
    project_urls={
        "Bug Tracker": "https://github.ibm.com/IBM-Q-Software/monodromy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
