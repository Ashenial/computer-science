from setuptools import setup, find_packages

setup(
    name='fje',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'fje = fje.main:main'
        ]
    }
)