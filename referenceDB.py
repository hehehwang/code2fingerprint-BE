import sqlite3
from hashlib import sha1

import numpy as np
from tqdm import tqdm

from fingerprinter import ScFp

"""
Table: reference
id : integer, primary key, autoincrement
sha: text, unique, not null
fileName: text, not null
methodName: text, not null
fingerprint: blob, not null
sourceCode: text, not null
description: text
"""
DB = "referenceDB.db"


def checkAndCreateDB():
    conn = sqlite3.connect(DB)
    curr = conn.cursor()
    curr.execute("SELECT name FROM sqlite_master WHERE type='table'")
    lst = curr.fetchall()
    if len(lst) < 1:
        curr.execute(
            "CREATE TABLE reference(id INTEGER PRIMARY KEY AUTOINCREMENT, \
        sha TEXT UNIQUE NOT NULL, fileName TEXT, methodName TEXT NOT NULL, sourceCode TEXT NOT NULL, fingerprint BLOB NOT NULL, description TEXT NOT NULL)"
        )
    conn.close()


def insert(sha, fileName, methodName, sourceCode, fingerprint, description):
    conn = sqlite3.connect(DB)
    curr = conn.cursor()
    curr.execute(
        "INSERT OR IGNORE INTO reference(sha, fileName, methodName, sourceCode, fingerprint, description) VALUES(?,?,?,?,?,?)",
        (sha, fileName, methodName, sourceCode, fingerprint, description),
    )
    conn.commit()
    conn.close()


def retrieve():
    conn = sqlite3.connect(DB)
    curr = conn.cursor()
    curr.execute("SELECT * FROM reference")
    lst = curr.fetchall()
    rtn = []
    for item in lst:
        nid, sha, fileName, methodName, sourceCode, fingerprint, description = item
        rtn.append(
            (
                sha,
                fileName,
                methodName,
                sourceCode,
                np.frombuffer(fingerprint),
                description,
            )
        )

    return rtn


def processRefSourceCode(sourceCode, fileName, description):
    sc = ScFp(sourceCode)
    for method in sc.methods:
        sha = sha1(method.sourceCode.encode("utf-8")).hexdigest()
        insert(
            sha,
            fileName,
            method.methodName,
            method.sourceCode,
            method.fingerprint,
            description,
        )


def pushSomeRefData():
    import os.path

    checkAndCreateDB()
    targetPath = "otherData\딥 러닝을 이용한 자연어 처리 입문"
    for fileName in tqdm(os.listdir(targetPath)):
        if fileName.endswith(".py"):
            with open(os.path.join(targetPath, fileName), "r", encoding="utf-8") as f:
                processRefSourceCode(
                    f.read(), fileName, "https://wikidocs.net/book/2155"
                )


def check():
    from pprint import pprint

    pprint(retrieve()[:10])


if __name__ == "__main__":
    check()
