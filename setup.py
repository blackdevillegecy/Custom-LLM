from setuptools import find_packages, setup

setup(
    name='custom-llm',
    version='0.1.0',
    packages=find_packages(),
    description='A Python library where you can build your own customized LLM.',
    author='Ayush Gautam',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests'
)