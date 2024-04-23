from setuptools import setup, find_packages

setup(
    name='pbpm_ssl_suite',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'your_script_name = your_package_name.module_name:main_function',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='Description of your package',
    url='https://github.com/your_username/your_package',
)
