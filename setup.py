from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as req:
        content = req.read()
        requirements = content.split('\n')
    return requirements

setup(
    name='CELLama',
    version='0.1.0',
    description = "CELLama",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url = "https://github.com/portrai-io/CELLama.git",
    author = "Portraier",
    install_requires=read_requirements(),
    #packages=find_packages(),
    packages=find_packages(include=['cellama_st', 'cellama_training']),
    py_modules=['cellama','_nn_model','_examples_to_json'],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
