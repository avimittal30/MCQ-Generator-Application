from setuptools import find_packages,setup

setup(
name="mcqgenerator",
version='0.0.0.1',
author='aviral mittal',
author_email='avimittal30@gmail.com',
install_requires=['openai','streamlit', 'langchain', 'python-dotenv', 'PyPDF2'],
packages=find_packages()
)