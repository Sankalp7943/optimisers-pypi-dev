
from setuptools import setup, find_packages

setup(   
    name = 'mliiitl',
    version = '2.0.0',

    description = 'A helping package for ML written by Sankalp',
    long_description = 'A helping package for quick model performance comparison based on different optimisers',

    long_description_content_type='text/x-rst',
    url='https://github.com/Sankalp7943/optimisers-pypi-dev',

    author='Sankalp Sharma',
    author_email='sharma.sankalp97@gmail.com',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='optimisers, machine learning, development',
    project_urls={
        'Developer': 'https://sharmasankalp.com'
    },
    python_requires='>=3.6',
    packages=['.mliiitl'],
    install_requires=[  'numpy',
                        'pandas',
                        'matplotlib',
                        'keras',
                        'tensorflow-gpu',
                        ],
)

