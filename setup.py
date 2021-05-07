#!/usr/bin/env python
# coding: utf-8

# In[1]:


import setuptools

with open('C:/Users/Hindy/Desktop/Container_RecurrsiveFeatureSelector/README.md', 'r') as fh:
    long_descripion = fh.read()
    
setuptools.setup(
    name='RecurrsiveFeatureSelector', 
    version='1.1.3',
    author='Hindy Yuen', 
    author_email='hindy888@hotmail.com',
    license='MIT',
    description='Recurrsively selecting features for machine learning task.', 
    long_description=long_descripion, 
    long_description_content_type='text/markdown', 
    url='https://github.com/HindyDS/RecurrsiveFeatureSelector', 
    classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
],
    keywords='Recurrsive features selection machine learning data science computer science',
    package_dir={"":"RecurrsiveFeatureSelector"},
    packages=['RecurrsiveFeatureSelector'],
    python_requires='>=3.6'
)

