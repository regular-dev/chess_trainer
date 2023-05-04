import sqlite3
import random
import sys

if len(sys.argv) != 2:
    print("Invalid number of arguments!")
    exit(0)

__db_file = sys.argv[1]

conn = sqlite3.connect(__db_file)
cursor = conn.cursor()

cursor.execute("SELECT * FROM positions")
rows = cursor.fetchall()
random.shuffle(rows)


cursor.execute("DELETE FROM positions")
for row in rows:
    cursor.execute("INSERT INTO positions VALUES (?,?)", row)

conn.commit()
conn.close()