#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: samples.py
# Author: anon
# Email: anon@cs.anon.edu
# Created on: 2020-04-16
# 
# This file is part of Clevr-Parser
# Distributed under terms of the MIT License

import random
random.seed(42)

from rsmlkit.logging import get_logger
logger = get_logger(__file__)

__all__ = ['TEMPLATES', 'get_s_sample']

TEMPLATES=("and_mat_spa",
              "compare_mat",
              "compare_mat_spa",
              "embed_mat_spa",
              "embed_spa_mat",
              "or_mat",
              "or_mat_spa")

def get_s_sample(template:str, dist='train') -> str:
    """
    :param template: Question family
    :param dist: 'train' or 'test'
    :return: A sample question from template and distribution
    """
    if template not in TEMPLATES:
        raise ValueError("Unknown template type")

    suffix = "baseline" if dist == 'train' else "val"
    template = f"{template}_{suffix}"
    """
    [and_mat_spa_baseline]
    Final program module = query_color
    Question type: query, answer: cyan, question for CLEVR_val_011582.png
    """
    s_ams_bline = "There is a thing that is on the right side of the tiny cyan rubber thing " \
                  "and to the left of the large green matte cylinder; what is its color?"

    """
    [and_mat_spa_val]
    Final program module = query_size
    Question type: query, answer: small, question for CLEVR_val_000019.png, 
    """
    s_ams_val = "What is the size of the thing that is in front of the big yellow object " \
                "and is the same shape as the big green thing?"

    """
    Final program module = count
    Question type: count, answer: 2, question for CLEVR_val_008452.png, 
    """
    s_oms_bline = "How many things are either small green objects in front of the small purple cylinder " \
                  "or large metallic things that are behind the red matte thing ?"

    """
    Final program module = count
    Question type: count, answer: 2, question for CLEVR_val_000439.png, 
    """
    s_oms_val = "How many things are cylinders that are behind the large purple metal thing " \
                "or purple cylinders that are the same size as the cyan thing ?"

    if template == f"and_mat_spa_{suffix}":
        return s_ams_bline
    elif template == f"and_mat_spa_val":
        return s_ams_val
    elif template == f"or_mat_spa_{suffix}":
        return s_oms_bline
    elif template == f"or_mat_spa_val":
        return s_oms_val
    else:
        raise ValueError("template must be one of [and|or]_mat_spa_[baseline|val]")



