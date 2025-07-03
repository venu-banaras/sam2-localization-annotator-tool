from setuptools import setup, find_packages

setup(
    name='sam2_localizer',
    version='0.1.0',
    install_requires=['tqdm', 'numpy'],
    extras_require=dict(tests=['pytest']),
    packages=find_packages(), 
    url='https://git-codecommit.us-east-1.amazonaws.com/v1/repos/sam2_localizer',
    license='',
    author='AI Model Team',
    author_email='airi@constems-ai.com',
    description='SAM2 based localization',
)
