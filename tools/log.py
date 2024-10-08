# -*- coding:utf-8 -*-
# @author: 牧锦程
# @微信公众号: AI算法与电子竞赛
# @Email: m21z50c71@163.com
# @VX：fylaicai


import sys
from rich.console import Console

ROOT_PATH = sys.path[0]  # 项目根目录


# 日志管理
def log_management(logContent, logName="face_label.log", logSaveMode="a"):
    logFile = open(f"{ROOT_PATH}/{logName}", logSaveMode)  # 日志文件
    logFile.write(logContent)  # 日志写入


# 日志管理（rich版）
def rich_log(logContent, logName="face_label.log", logSaveMode="a"):
    report_file = open(f"{ROOT_PATH}/{logName}", logSaveMode)
    console = Console(file=report_file)
    console.log(logContent)
